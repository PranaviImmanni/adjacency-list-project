"""Benchmark GCN inference for exactly one dataset / format / device / repetition.

Writes one JSON result and exits. Run it once per repetition (in separate
processes) so allocator and cache state do not leak between runs.

Example:
    python scripts/benchmark_gcn.py --dataset cora --format csr --device cpu \
        --checkpoint checkpoints/cora_seed42.pt --output results/raw/cora_csr_cpu_rep0.json

Only the forward passes are timed. Dataset loading, normalization, format
conversion and the CPU-to-device transfer happen before the timer starts
(conversion and transfer are timed separately). Latency measured on CPU is
for debugging only and is not a result for the GPU study.
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.adjacency import (STORAGE_FORMATS, adjacency_storage_breakdown, build_normalized_coo,
                               convert_adjacency, estimate_dense_bytes)
from src.gcn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, set_seed, torch_dtype
from src.gcn.datasets import load_dataset
from src.gcn.evaluation import compare_to_reference, is_equivalent, run_inference
from src.gcn.measurement import (check_format_supported, configure_precision, cuda_memory_stats, device_info,
                                 reset_peak_memory, resolve_device, summarize_latency, synchronize,
                                 time_forward)
from src.gcn.model import load_gcn_checkpoint

SCHEMA_VERSION = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--format", required=True, choices=STORAGE_FORMATS, dest="storage_format")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "cora_seed42.pt"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--repetition", type=int, default=0, help="repetition index recorded in the result")
    parser.add_argument("--output", default=None,
                        help="result JSON (default: results/raw/<dataset>_<format>_<device>_rep<N>.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    dtype = torch_dtype(config["dtype"])

    # 1. Seed (inference is deterministic in eval mode; this pins any incidental randomness).
    set_seed(config["seed"])

    device = resolve_device(args.device)
    check_format_supported(device, args.storage_format)
    precision = configure_precision(device, config["tf32"])

    # 2. Dataset and checkpoint on CPU.
    graph = load_dataset(args.dataset)
    model, checkpoint = load_gcn_checkpoint(args.checkpoint, "cpu")
    if checkpoint.get("dataset", args.dataset) != args.dataset or checkpoint["input_dim"] != graph.num_features:
        raise SystemExit(f"checkpoint {args.checkpoint} was not trained on {args.dataset}")
    reference_model = copy.deepcopy(model)  # stays on CPU for the COO reference

    # 3. Canonical normalized COO.
    start = time.perf_counter()
    coo = build_normalized_coo(graph.edge_index, graph.num_nodes, dtype)
    normalization_ms = (time.perf_counter() - start) * 1e3

    # 4. Only the requested representation.
    start = time.perf_counter()
    adj = convert_adjacency(coo, args.storage_format)
    conversion_ms = (time.perf_counter() - start) * 1e3
    adjacency_bytes = adjacency_storage_breakdown(adj)

    # 5. Move only that representation, the features, and the model.
    start = time.perf_counter()
    adj_device = adj.to(device)
    x_device = graph.x.to(device=device, dtype=dtype)
    model = model.to(device)
    synchronize(device)
    transfer_ms = (time.perf_counter() - start) * 1e3

    # 6-9. Eval + inference mode, warm-ups, synchronized timed passes.
    reset_peak_memory(device)
    resident = cuda_memory_stats(device)["allocated_bytes"]
    logits, times_ms = time_forward(model, x_device, adj_device, device,
                                    config["warmup_runs"], config["measured_runs"])
    memory = cuda_memory_stats(device)

    # 10. Prediction metrics against the CPU COO reference (outside the timer).
    reference_logits = run_inference(reference_model, graph.x.to(dtype), coo)
    prediction = compare_to_reference(reference_logits, logits, graph.y, graph.test_mask)
    prediction["equivalent"] = is_equivalent(prediction)
    prediction["reference"] = "coo on cpu"

    official = device.type == "cuda"
    result = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "format": args.storage_format,
        "repetition": args.repetition,
        **device_info(device),
        "latency_is_debug_only": not official,
        "note": ("CUDA run." if official else
                 "CPU latency is for debugging only; it is not a GPU result and must not be reported as one."),
        "dtype": config["dtype"],
        **precision,
        "checkpoint": args.checkpoint,
        "checkpoint_best_epoch": checkpoint["best_epoch"],
        "num_nodes": graph.num_nodes,
        "num_features": graph.num_features,
        "num_classes": graph.num_classes,
        "nnz": coo._nnz(),
        # 11. Exact adjacency bytes (numel * element_size of each component).
        "adjacency_bytes": sum(adjacency_bytes.values()),
        "adjacency_bytes_breakdown": adjacency_bytes,
        "dense_adjacency_bytes_estimate": estimate_dense_bytes(graph.num_nodes, dtype),
        "normalization_ms": normalization_ms,
        "conversion_ms": conversion_ms,
        "transfer_ms": transfer_ms,
        "warmup_runs": config["warmup_runs"],
        "measured_runs": config["measured_runs"],
        "latency_ms": summarize_latency(times_ms),
        "latency_samples_ms": times_ms,
        # 12. CUDA memory (None off CUDA).
        "cuda_allocated_before_inference_bytes": resident,
        "cuda_peak_allocated_bytes": memory["peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": memory["peak_reserved_bytes"],
        "prediction": prediction,
    }

    # 13. Save one JSON result.
    device_tag = str(device).replace(":", "")
    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / "raw" / f"{args.dataset}_{args.storage_format}_{device_tag}_rep{args.repetition}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    label = "DEBUG ONLY (not a GPU result)" if not official else "CUDA"
    print(f"{args.dataset} {args.storage_format} on {device} rep {args.repetition}: "
          f"median {result['latency_ms']['median']:.3f} ms [{label}], "
          f"adjacency {result['adjacency_bytes']:,} bytes, equivalent to COO reference: {prediction['equivalent']}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
