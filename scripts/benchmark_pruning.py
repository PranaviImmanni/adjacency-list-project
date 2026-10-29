"""Benchmark one pruned graph: dataset / method / pruning seed / removal fraction / format / repetition.

The pruning counterpart of benchmark_gcn.py: same frozen checkpoint, same
canonical normalization, same timing and memory code. Run it once per
condition and repetition in separate processes; see
scripts/run_pruning_experiments.py.

Example:
    python scripts/benchmark_pruning.py --dataset cora --checkpoint checkpoints/cora_seed42_gpu.pt \
        --method random --prune-seed 101 --fraction 0.5 --device cuda:1 \
        --output results/pruning/cora_cuda1/cora_random_s101_f50_csr_cuda1_rep0.json

Timed separately, never inside the inference latency: edge scoring/ordering,
selection, normalization, format conversion and the transfer to the device.

Only the target adjacency, the features and the model are on the device while
memory is measured. The peak counters are reset after the warm-up passes, so
the peak covers the measured passes and includes every resident tensor.
After the measurement the run computes, outside the timer:
  - pruning effect: these logits vs the original graph in CSR on the same
    device and checkpoint (src/gcn/error_analysis.py),
  - format check: these logits vs COO on CPU for the same pruned graph.
"""

import argparse
import copy
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.adjacency import (STORAGE_FORMATS, adjacency_dtypes, adjacency_storage_breakdown,
                               build_normalized_coo, convert_adjacency, estimate_dense_bytes)
from src.gcn.config import DEFAULT_CONFIG_PATH, load_config, set_seed, torch_dtype
from src.gcn.datasets import load_dataset
from src.gcn.error_analysis import exact_accuracy, format_error, pruning_effect
from src.gcn.evaluation import compare_to_reference, is_equivalent, run_inference
from src.gcn.measurement import (check_format_supported, configure_precision, cuda_memory_stats, device_info,
                                 resolve_device, summarize_latency, synchronize, time_forward)
from src.gcn.model import checkpoint_identity, load_gcn_checkpoint
from src.gcn.pruning import METHODS, prune_graph, save_or_check_ordering, tensor_sha256

SCHEMA_VERSION = 1


def run(args):
    config = load_config(args.config)
    dtype = torch_dtype(config["dtype"])
    set_seed(config["seed"])  # inference is deterministic; this pins any incidental randomness

    device = resolve_device(args.device)
    check_format_supported(device, args.storage_format)
    precision = configure_precision(device, config["tf32"])

    # 1. Dataset and the frozen checkpoint, on CPU.
    graph = load_dataset(args.dataset)
    model, checkpoint = load_gcn_checkpoint(args.checkpoint, "cpu")
    if checkpoint.get("dataset", args.dataset) != args.dataset or checkpoint["input_dim"] != graph.num_features:
        raise SystemExit(f"checkpoint {args.checkpoint} was not trained on {args.dataset}")
    reference_model = copy.deepcopy(model)  # stays on CPU for the COO format check
    x = graph.x.to(dtype)

    # 2. Pruning: ordering (scoring) and selection are timed inside prune_graph.
    pruned = prune_graph(graph.edge_index, graph.num_nodes, args.method, args.fraction,
                         x=x, seed=args.prune_seed)
    ordering_hash = None
    if pruned["order"] is not None and args.orderings_dir:
        name = f"{args.dataset}_{args.method}" + (f"_s{args.prune_seed}" if args.method == "random" else "")
        ordering_file = Path(args.orderings_dir) / f"{name}_order.pt"
        ordering_hash = save_or_check_ordering(ordering_file, pruned)

    # 3. Canonical normalized COO of the pruned graph, then only the requested format.
    start = time.perf_counter()
    coo = build_normalized_coo(pruned["retained_edges"], graph.num_nodes, dtype)
    normalization_ms = (time.perf_counter() - start) * 1e3
    if coo._nnz() != pruned["pruned_graph"]["normalized_nnz"]:
        raise SystemExit(f"expected {pruned['pruned_graph']['normalized_nnz']} nonzeros, got {coo._nnz()}")
    start = time.perf_counter()
    adj = convert_adjacency(coo, args.storage_format)
    conversion_ms = (time.perf_counter() - start) * 1e3
    adjacency_bytes = adjacency_storage_breakdown(adj)

    # 4. Only this representation, the features and the model go to the device.
    start = time.perf_counter()
    adj_device = adj.to(device)
    x_device = x.to(device)
    model = model.to(device)
    synchronize(device)
    transfer_ms = (time.perf_counter() - start) * 1e3

    # 5. Warm-ups, peak reset, synchronized timed passes.
    resident = cuda_memory_stats(device)["allocated_bytes"]
    logits, times_ms = time_forward(model, x_device, adj_device, device, config["warmup_runs"],
                                    config["measured_runs"], reset_peak_after_warmup=True)
    memory = cuda_memory_stats(device)
    logits = logits.cpu()
    del adj_device

    # 6. Outside the measurement: original graph in CSR on the same device.
    original_coo = build_normalized_coo(graph.edge_index, graph.num_nodes, dtype)
    original_logits = run_inference(model, x_device, convert_adjacency(original_coo, "csr").to(device)).cpu()
    effect = pruning_effect(original_logits, logits, graph.y, graph.val_mask, graph.test_mask)
    effect["reference"] = f"original graph, csr on {device}"

    # 7. Format check: the same pruned graph in COO on CPU.
    cpu_coo_logits = run_inference(reference_model, x, coo)
    format_check = compare_to_reference(cpu_coo_logits, logits, graph.y, graph.test_mask)
    format_check.update(format_error(cpu_coo_logits, logits, graph.test_mask, format_check))
    format_check["equivalent"] = is_equivalent(format_check)
    format_check["reference"] = "same pruned graph, coo on cpu"

    if args.logits_out:
        path = Path(args.logits_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": logits, "original_logits": original_logits}, path)
        labels = path.parent / f"{args.dataset}_labels.pt"
        if not labels.exists():
            torch.save({"y": graph.y, "train_mask": graph.train_mask, "val_mask": graph.val_mask,
                        "test_mask": graph.test_mask}, labels)

    official = device.type == "cuda"
    pruning = {key: pruned[key] for key in ("method", "seed", "requested_removal_fraction",
                                            "actual_removal_fraction", "original_unique_edges",
                                            "retained_unique_edges", "removed_unique_edges")}
    pruning["ordering_sha256"] = ordering_hash
    pruning["similarity"] = ("cosine of row-normalized features over existing edges, float64; "
                             "zero vector -> 0.0; ties by (u, v)") if args.method == "similarity" else None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "experiment": "pruning",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_info": graph.metadata(),
        "checkpoint": checkpoint_identity(args.checkpoint, checkpoint),
        "format": args.storage_format,
        "repetition": args.repetition,
        **device_info(device),
        "latency_is_debug_only": not official,
        "note": ("CUDA run." if official else
                 "CPU latency is for debugging only; it is not a GPU result and must not be reported as one."),
        "dtype": config["dtype"],
        **precision,
        "pruning": pruning,
        "graph": {"original": pruned["original_graph"], "pruned": pruned["pruned_graph"]},
        "model": {"input_dim": checkpoint["input_dim"], "hidden_dim": checkpoint["hidden_dim"],
                  "output_dim": checkpoint["output_dim"]},
        "num_nodes": graph.num_nodes,
        "nnz": coo._nnz(),
        "adjacency_bytes": sum(adjacency_bytes.values()),
        "adjacency_bytes_breakdown": adjacency_bytes,
        **adjacency_dtypes(adj),
        "dense_adjacency_bytes_estimate": estimate_dense_bytes(graph.num_nodes, dtype),
        "timing_ms": {
            "order_ms": pruned["order_ms"],
            "selection_ms": pruned["selection_ms"],
            "normalization_ms": normalization_ms,
            "conversion_ms": conversion_ms,
            "transfer_ms": transfer_ms,
        },
        "warmup_runs": config["warmup_runs"],
        "measured_runs": config["measured_runs"],
        "latency_ms": summarize_latency(times_ms),
        "latency_samples_ms": times_ms,
        "peak_reset": "after warm-up, before the measured passes",
        "cuda_allocated_before_inference_bytes": resident,
        "cuda_peak_allocated_bytes": memory["peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": memory["peak_reserved_bytes"],
        "accuracy": {"val": exact_accuracy(logits, graph.y, graph.val_mask),
                     "test": exact_accuracy(logits, graph.y, graph.test_mask)},
        "pruning_effect": effect,
        "format_check": format_check,
        "logits_sha256": tensor_sha256(logits),
        "original_logits_sha256": tensor_sha256(original_logits),
        # Relative to this result's folder, so the analysis works wherever the folder is copied.
        "logits_file": f"logits/{Path(args.logits_out).name}" if args.logits_out else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True, help="the dataset's one frozen checkpoint")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--fraction", type=float, required=True, help="removal fraction, e.g. 0.5")
    parser.add_argument("--prune-seed", type=int, default=None, help="random pruning only")
    parser.add_argument("--format", default="csr", choices=STORAGE_FORMATS, dest="storage_format")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--orderings-dir", default=None, help="save (or check) the edge ordering here")
    parser.add_argument("--logits-out", default=None, help="save the logits (.pt) for offline analysis")
    parser.add_argument("--output", required=True, help="result JSON")
    args = parser.parse_args()
    if args.method == "random" and args.prune_seed is None:
        parser.error("--method random needs --prune-seed")
    if args.method != "random":
        args.prune_seed = None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as error:
        with open(output, "w") as f:
            json.dump({"schema_version": SCHEMA_VERSION, "status": "failed", "experiment": "pruning",
                       "timestamp": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                       "error": repr(error), "traceback": traceback.format_exc()}, f, indent=2)
        print(f"FAILED: {error!r}. Wrote {output}")
        raise SystemExit(1)

    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    effect = result["pruning_effect"]
    label = "CUDA" if not result["latency_is_debug_only"] else "DEBUG ONLY (not a GPU result)"
    print(f"{args.dataset} {args.method}{'' if args.prune_seed is None else f' s{args.prune_seed}'} "
          f"remove {result['pruning']['actual_removal_fraction']:.1%} {args.storage_format} rep {args.repetition}: "
          f"test acc {result['accuracy']['test']:.3f} (loss {effect['test_accuracy_loss_pp']:+.1f} pp), "
          f"median {result['latency_ms']['median']:.3f} ms [{label}], format check ok: "
          f"{result['format_check']['equivalent']}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
