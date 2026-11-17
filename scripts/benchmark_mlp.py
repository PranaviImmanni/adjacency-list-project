"""Benchmark one MLP variant: model seed / pruning condition / format / batch size / repetition.

Run once per process (scripts/run_mlp_experiments.py does this), so no other
variant's tensors or allocator state are present.

Example:
    python scripts/benchmark_mlp.py --checkpoint checkpoints/fashion_mlp_seed42_cuda1.pt \
        --method magnitude --sparsity 0.9 --format csr --batch-size 32 --device cuda:1 \
        --output results/mlp/cuda1/benchmark/m42_magnitude_s90_csr_b32_cuda1_rep0.json

Outside the latency: loading, pruning, building the format and the transfer
(each timed separately). Only the requested format's tensors, the biases and
the input batch are on the device during the measurement; the run verifies
that a sparse variant holds no dense weight or mask there. The peak counters
are reset after the warm-up, so the peak covers the measured passes and
includes every resident tensor. After the measurement, outside the timer and
after this variant is released, the same batch is run through the pruned
dense model on the same device as a format check.

Energy is not measured here (null) until the harness has a reliable method.
"""

import argparse
import json
import sys
import warnings
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# PyTorch prints this notice every time a CSR tensor is created; it is not about our results.
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from src.gcn.config import set_seed
from src.gcn.measurement import (check_format_supported, configure_precision, cuda_memory_stats, device_info,
                                 resolve_device, summarize_latency)
from src.gcn.pruning import tensor_sha256
from src.mlp.config import DEFAULT_CONFIG_PATH, load_config
from src.mlp.data import load_fashion_mnist
from src.mlp.experiment import format_comparison
from src.mlp.formats import FORMATS
from src.mlp.harness import check_resident, load_variant, make_batch, time_inference
from src.mlp.weight_pruning import METHODS

SCHEMA_VERSION = 1


def run(args):
    config = load_config(args.config)
    set_seed(0)  # inference is deterministic; this pins any incidental randomness
    device = resolve_device(args.device)
    check_format_supported(device, args.storage_format)
    precision = configure_precision(device, config["tf32"])

    data = load_fashion_mnist(config["val_size"], config["split_seed"])
    variant = load_variant(args.checkpoint, args.method, args.sparsity, args.prune_seed, args.storage_format,
                           device, config["pruned_layers"])
    if not check_resident(variant):
        raise RuntimeError("a sparse variant holds a dense weight or mask on the device")
    x = make_batch(data.x_test, args.batch_size, device)
    y = data.y_test[:args.batch_size]
    resident_tensor_bytes = variant.model.resident_bytes() + x.numel() * x.element_size()

    resident = cuda_memory_stats(device)["allocated_bytes"]
    logits, times_ms = time_inference(variant.model, x, device, config["warmup_runs"], config["measured_runs"])
    memory = cuda_memory_stats(device)
    logits = logits.cpu()
    meta, load_timing = variant.metadata, variant.timing_ms
    del variant

    format_check = None
    if args.method != "none" and args.storage_format != "dense":
        reference = load_variant(args.checkpoint, args.method, args.sparsity, args.prune_seed, "dense", device,
                                 config["pruned_layers"])
        with torch.inference_mode():
            reference_logits = reference.model(x).cpu()
        format_check = format_comparison(reference_logits, logits, y, torch.ones(len(y), dtype=torch.bool))
        format_check["reference"] = f"pruned dense on {device}, same batch"

    official = device.type == "cuda"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "experiment": "mlp_benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "fashion_mnist",
        "checkpoint": meta["checkpoint"],
        "model_seed": meta["model_seed"],
        "method": meta["method"],
        "pruning_seed": meta["pruning_seed"],
        "requested_sparsity": meta["requested_sparsity"],
        "actual_sparsity": meta["actual_sparsity"],
        "pruned_layers": meta["pruned_layers"],
        "pruning": meta["pruning"],
        "condition": meta["condition"],
        "format": meta["format"],
        "batch_size": args.batch_size,
        "repetition": args.repetition,
        **device_info(device),
        "latency_is_debug_only": not official,
        "note": ("CUDA run." if official else
                 "CPU latency is for debugging only; it is not a GPU result and must not be reported as one."),
        "dtype": config["dtype"],
        **precision,
        "storage": meta["storage"],
        "weight_storage_bytes": meta["storage"]["weight_bytes"],
        "bias_bytes": meta["storage"]["bias_bytes"],
        "resident_tensor_bytes": resident_tensor_bytes,
        "timing_ms": load_timing,
        "warmup_runs": config["warmup_runs"],
        "measured_runs": config["measured_runs"],
        "latency_ms": summarize_latency(times_ms),
        "latency_samples_ms": times_ms,
        "peak_reset": "after warm-up, before the measured passes",
        "cuda_allocated_before_inference_bytes": resident,
        "cuda_peak_allocated_bytes": memory["peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": memory["peak_reserved_bytes"],
        "energy_joules": None,
        "format_check": format_check,
        "logits_sha256": tensor_sha256(logits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True, choices=("none",) + METHODS)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--prune-seed", type=int, default=None, help="random pruning only")
    parser.add_argument("--format", required=True, choices=FORMATS, dest="storage_format")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--output", required=True)
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
            json.dump({"schema_version": SCHEMA_VERSION, "status": "failed", "experiment": "mlp_benchmark",
                       "timestamp": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                       "error": repr(error), "traceback": traceback.format_exc()}, f, indent=2)
        print(f"FAILED: {error!r}. Wrote {output}")
        raise SystemExit(1)

    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    check = result["format_check"]
    label = "CUDA" if not result["latency_is_debug_only"] else "DEBUG ONLY (not a GPU result)"
    print(f"m{result['model_seed']} {args.method}"
          f"{'' if args.prune_seed is None else f' p{args.prune_seed}'} s{args.sparsity:g} {result['format']} "
          f"b{args.batch_size} rep {args.repetition}: median {result['latency_ms']['median']:.4f} ms [{label}], "
          f"weights {result['weight_storage_bytes']:,} B"
          + ("" if check is None else f", matches pruned dense: {check['equivalent']}"))


if __name__ == "__main__":
    main()
