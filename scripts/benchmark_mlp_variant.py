"""Benchmark one follow-up variant: one format, one batch size, one repetition, one process.

Same protocol as scripts/benchmark_mlp.py: the variant is loaded and built on
CPU, only the requested format's tensors (plus biases and any adapters) are
moved to the device, the run checks that nothing else is resident (no dense
pruned weights or masks for CSR; int8 weight matrices only for INT8), the
peak counters are reset after the warm-up, and after the timing (outside the
timer, with the variant released) the same batch is run through the
reference implementation of the same modified model as a format check.

INT8 W8A8 runs torch._int_mm on the device; the run fails rather than fall
back if the device does not support it. Energy is not measured (null).

Example:
    python scripts/benchmark_mlp_variant.py --variant checkpoints/variants/cuda1/m42_statistical_k1.pt \
        --format csr --batch-size 32 --device cuda:1 \
        --output results/mlp_followup/cuda1/benchmark/m42_statistical_k1_csr_b32_cuda1_rep0.json
"""

import argparse
import json
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from src.gcn.config import set_seed
from src.gcn.measurement import (configure_precision, cuda_memory_stats, device_info, resolve_device,
                                 summarize_latency, synchronize)
from src.gcn.pruning import tensor_sha256
from src.mlp.config import FOLLOWUP_CONFIG_PATH, load_followup_config
from src.mlp.data import load_for_checkpoint
from src.mlp.experiment import format_comparison
from src.mlp.harness import make_batch, time_inference
from src.mlp.variants import REFERENCE_FORMAT, build_variant, check_resident, load_variant_file, variant_summary

SCHEMA_VERSION = 1


def run(args):
    config = load_followup_config(args.config)
    set_seed(0)
    device = resolve_device(args.device)
    precision = configure_precision(device, config["tf32"])
    variant = load_variant_file(args.variant)
    if args.storage_format not in variant["deploy_formats"]:
        raise ValueError(f"{variant['tag']} is deployed as {variant['deploy_formats']}, not {args.storage_format}")
    data = load_for_checkpoint(variant)

    start = time.perf_counter()
    cpu_model = build_variant(variant, args.storage_format)
    build_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    model = cpu_model.to(device)
    synchronize(device)
    transfer_ms = (time.perf_counter() - start) * 1e3
    del cpu_model
    if not check_resident(model):
        raise RuntimeError("unexpected tensors on the device for this format")
    storage, support = model.storage(), getattr(model, "support", None)

    x = make_batch(data.x_test, args.batch_size, device)
    y = data.y_test[:args.batch_size]
    resident_tensor_bytes = model.resident_bytes() + x.numel() * x.element_size()
    resident = cuda_memory_stats(device)["allocated_bytes"]
    logits, times_ms = time_inference(model, x, device, config["warmup_runs"], config["measured_runs"])
    memory = cuda_memory_stats(device)
    logits = logits.cpu()
    del model

    reference_format = REFERENCE_FORMAT[args.storage_format]
    reference = build_variant(variant, reference_format).to(device)
    with torch.inference_mode():
        reference_logits = reference(x).cpu()
    check = format_comparison(reference_logits, logits, y, torch.ones(len(y), dtype=torch.bool))
    check["reference"] = f"{reference_format} on {device}, same batch"

    summary = variant_summary(variant)
    official = device.type == "cuda"
    return {
        "schema_version": SCHEMA_VERSION, "status": "ok", "experiment": "mlp_followup_benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(), "dataset": "fashion_mnist",
        "tag": summary["tag"], "followup_experiment": summary["experiment"], "kind": summary["kind"],
        "params": summary["params"], "model_seed": summary["model_seed"],
        "model_sparsity": summary["model_sparsity"], "model_retained_weights": summary["model_retained_weights"],
        "format": args.storage_format, "batch_size": args.batch_size, "repetition": args.repetition,
        **device_info(device), "latency_is_debug_only": not official,
        "note": "CUDA run." if official else "CPU latency is for debugging only, not a GPU result.",
        "dtype": config["dtype"], **precision,
        "storage": storage, "weight_storage_bytes": storage["weight_bytes"],
        "adapter_bytes": storage["adapter_bytes"], "stored_bytes": storage["stored_bytes"],
        "bias_bytes": storage["bias_bytes"], "resident_tensor_bytes": resident_tensor_bytes,
        "timing_ms": {"format_build_ms": build_ms, "transfer_ms": transfer_ms},
        "warmup_runs": config["warmup_runs"], "measured_runs": config["measured_runs"],
        "latency_ms": summarize_latency(times_ms), "latency_samples_ms": times_ms,
        "peak_reset": "after warm-up, before the measured passes",
        "cuda_allocated_before_inference_bytes": resident,
        "cuda_peak_allocated_bytes": memory["peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": memory["peak_reserved_bytes"],
        "int_mm_support": support, "energy_joules": None,
        "format_check": check, "logits_sha256": tensor_sha256(logits), "variant": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--format", required=True, dest="storage_format",
                        choices=("csr", "int8_weight_only", "int8_w8a8"))
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=str(FOLLOWUP_CONFIG_PATH))
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as error:
        output.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "status": "failed",
                                      "experiment": "mlp_followup_benchmark",
                                      "timestamp": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                                      "error": repr(error), "traceback": traceback.format_exc()}, indent=2))
        print(f"FAILED: {error!r}. Wrote {output}")
        raise SystemExit(1)

    output.write_text(json.dumps(result, indent=2))
    label = "CUDA" if not result["latency_is_debug_only"] else "DEBUG ONLY"
    print(f"{result['tag']} {result['format']} b{args.batch_size} rep {args.repetition}: median "
          f"{result['latency_ms']['median']:.4f} ms [{label}], stored {result['stored_bytes']:,} B, "
          f"matches reference: {result['format_check']['equivalent']}")


if __name__ == "__main__":
    main()
