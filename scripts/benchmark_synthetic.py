"""Benchmark GCN inference on one synthetic graph / format / device / repetition.

The synthetic counterpart of benchmark_gcn.py: same canonical adjacency,
model, timing and memory code, but the graph is a uniform random G(n, m)
graph and the model has fixed random weights (no training, no labels).
Equivalence is checked on the logits against COO on CPU.

Run it once per (graph, format, repetition) in separate processes; see
scripts/run_synthetic_sweep.py. The graph is regenerated from its seed in
every process, and `graph_fingerprint` in the result lets the summary check
that all formats saw the same edges.

Example:
    python scripts/benchmark_synthetic.py --nodes 1000 --avg-degree 8 --graph-seed 42 \
        --format csr --device cuda:0 --output results/synthetic/pilot/n1000_d8_g42_csr_rep0.json

Seeds are separate and recorded: --graph-seed varies across official runs;
--feature-seed and --model-seed stay fixed for a whole sweep, and use their
own random streams, so changing the graph never changes the features or weights.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.adjacency import (STORAGE_FORMATS, adjacency_storage_breakdown, build_normalized_coo,
                               convert_adjacency, estimate_dense_bytes)
from src.gcn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, set_seed, torch_dtype
from src.gcn.evaluation import compare_logits, is_equivalent, run_inference
from src.gcn.measurement import (check_format_supported, configure_precision, cuda_memory_stats, device_info,
                                 reset_peak_memory, resolve_device, summarize_latency, synchronize,
                                 time_forward)
from src.gcn.model import GCN
from src.synthetic.edge_generator import (edges_for_avg_degree, edges_for_density, num_possible_edges,
                                          sample_undirected_edges)

SCHEMA_VERSION = 1

# Fixed for every sweep (input and output widths; hidden width comes from the config).
DEFAULT_FEATURE_DIM = 64
DEFAULT_OUTPUT_DIM = 7
# Distinct from each other and from the graph seeds, so no two streams coincide.
DEFAULT_FEATURE_SEED = 0
DEFAULT_MODEL_SEED = 1


def graph_tag(num_nodes, avg_degree=None, density=None, graph_seed=42):
    """Short name used in result file names, e.g. n1000_d8_g42 or n5000_p0.01_g43."""
    spec = f"d{avg_degree:g}" if avg_degree is not None else f"p{density:g}"
    return f"n{num_nodes}_{spec}_g{graph_seed}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, required=True)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--avg-degree", type=float, help="average neighbors per node, before self-loops")
    size.add_argument("--density", type=float, help="fraction of the N(N-1)/2 possible edges, e.g. 0.01")
    parser.add_argument("--graph-seed", type=int, default=42)
    parser.add_argument("--feature-seed", type=int, default=DEFAULT_FEATURE_SEED)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    parser.add_argument("--format", required=True, choices=STORAGE_FORMATS, dest="storage_format")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--repetition", type=int, default=0, help="repetition index recorded in the result")
    parser.add_argument("--output", default=None,
                        help="result JSON (default: results/synthetic/raw/<graph>_<format>_<device>_rep<N>.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    dtype = torch_dtype(config["dtype"])

    device = resolve_device(args.device)
    check_format_supported(device, args.storage_format)
    precision = configure_precision(device, config["tf32"])

    # 1. Graph: exactly m undirected edges, each stored once with i < j.
    n = args.nodes
    if args.avg_degree is not None:
        num_edges = edges_for_avg_degree(n, args.avg_degree)
    else:
        num_edges = edges_for_density(n, args.density)
    start = time.perf_counter()
    edge_index = sample_undirected_edges(n, num_edges, args.graph_seed)
    generation_ms = (time.perf_counter() - start) * 1e3
    fingerprint = hashlib.sha256(edge_index.numpy().tobytes()).hexdigest()[:16]

    # 2. Fixed random features and model weights, each from its own seed.
    features = torch.randn(n, args.feature_dim, generator=torch.Generator().manual_seed(args.feature_seed),
                           dtype=dtype)
    set_seed(args.model_seed)
    model = GCN(args.feature_dim, config["hidden_dim"], args.output_dim, config["dropout"]).to(dtype)
    model.eval()
    reference_model = copy.deepcopy(model)  # stays on CPU for the COO reference

    # 3. Canonical normalized COO (num_nodes passed so isolated nodes are kept).
    start = time.perf_counter()
    coo = build_normalized_coo(edge_index, n, dtype)
    normalization_ms = (time.perf_counter() - start) * 1e3
    if coo._nnz() != 2 * num_edges + n:
        raise SystemExit(f"expected {2 * num_edges + n} nonzeros, got {coo._nnz()}")

    # 4. Only the requested representation.
    start = time.perf_counter()
    adj = convert_adjacency(coo, args.storage_format)
    conversion_ms = (time.perf_counter() - start) * 1e3
    adjacency_bytes = adjacency_storage_breakdown(adj)

    # 5. Move only that representation, the features, and the model.
    start = time.perf_counter()
    adj_device = adj.to(device)
    x_device = features.to(device)
    model = model.to(device)
    synchronize(device)
    transfer_ms = (time.perf_counter() - start) * 1e3

    # 6. Warm-ups and synchronized timed passes.
    reset_peak_memory(device)
    resident = cuda_memory_stats(device)["allocated_bytes"]
    logits, times_ms = time_forward(model, x_device, adj_device, device,
                                    config["warmup_runs"], config["measured_runs"])
    memory = cuda_memory_stats(device)

    # 7. Logit equivalence against the CPU COO reference (outside the timer).
    reference_logits = run_inference(reference_model, features, coo)
    prediction = compare_logits(reference_logits, logits)
    prediction["equivalent"] = is_equivalent(prediction)
    prediction["reference"] = "coo on cpu"

    official = device.type == "cuda"
    result = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "synthetic",
        "format": args.storage_format,
        "repetition": args.repetition,
        **device_info(device),
        "latency_is_debug_only": not official,
        "note": ("CUDA run." if official else
                 "CPU latency is for debugging only; it is not a GPU result and must not be reported as one."),
        "dtype": config["dtype"],
        **precision,
        "graph": {
            "generator": "uniform G(n, m), undirected, no self-loops",
            "requested_avg_degree": args.avg_degree,
            "requested_density": args.density,
            "num_nodes": n,
            "num_edges": num_edges,
            "avg_degree": 2 * num_edges / n,
            "density": num_edges / num_possible_edges(n),
            "graph_seed": args.graph_seed,
            "graph_fingerprint": fingerprint,
            "generation_ms": generation_ms,
        },
        "model": {
            "input_dim": args.feature_dim,
            "hidden_dim": config["hidden_dim"],
            "output_dim": args.output_dim,
            "feature_seed": args.feature_seed,
            "model_seed": args.model_seed,
            "weights": "random (Glorot), untrained",
        },
        "num_nodes": n,
        "nnz": coo._nnz(),
        # Exact adjacency bytes (numel * element_size of each component).
        "adjacency_bytes": sum(adjacency_bytes.values()),
        "adjacency_bytes_breakdown": adjacency_bytes,
        "dense_adjacency_bytes_estimate": estimate_dense_bytes(n, dtype),
        "normalization_ms": normalization_ms,
        "conversion_ms": conversion_ms,
        "transfer_ms": transfer_ms,
        "warmup_runs": config["warmup_runs"],
        "measured_runs": config["measured_runs"],
        "latency_ms": summarize_latency(times_ms),
        "latency_samples_ms": times_ms,
        # CUDA memory (None off CUDA).
        "cuda_allocated_before_inference_bytes": resident,
        "cuda_peak_allocated_bytes": memory["peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": memory["peak_reserved_bytes"],
        "prediction": prediction,
    }

    tag = graph_tag(n, args.avg_degree, args.density, args.graph_seed)
    device_tag = str(device).replace(":", "")
    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / "synthetic" / "raw" / f"{tag}_{args.storage_format}_{device_tag}_rep{args.repetition}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    label = "DEBUG ONLY (not a GPU result)" if not official else "CUDA"
    print(f"{tag} ({num_edges:,} edges, nnz {coo._nnz():,}) {args.storage_format} on {device} rep {args.repetition}: "
          f"median {result['latency_ms']['median']:.3f} ms [{label}], "
          f"adjacency {result['adjacency_bytes']:,} bytes, "
          f"max |dz| {prediction['max_abs_logit_diff']:.1e}, equivalent: {prediction['equivalent']}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
