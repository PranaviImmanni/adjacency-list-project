"""Benchmark one model of the GPU format comparison: seed / model / format / batch size / repetition.

Run once per process (scripts/run_mlp_gpu_comparison.py does this), so no other model's
tensors or allocator state are present. Same procedure as scripts/benchmark_mlp.py:

  1. the model is built on CPU (pruning or SVD, format build) and only the requested
     format's tensors, the biases and the input batch (the first B test images) are
     moved to the device; a sparse or list format must hold no dense weight there;
  2. latency: warm-up passes, peak-memory reset, then synchronized timed passes;
     resident and peak CUDA memory;
  3. energy (CUDA + NVML): idle power, then synchronized inference of the same batch for
     at least energy_seconds (src/mlp/energy.py); outside the latency timing;
  4. outside all timing, after this model is released, the same batch through the
     reference format (pruned dense or dense factors) on the same device.

Example (lab GPU):
    python scripts/benchmark_mlp_gpu_comparison.py --seed 42 --tag m42_magnitude_s90 \
        --format adjacency_list --batch-size 32 --device cuda:1 \
        --output results/mlp_gpu/cuda1/benchmark/m42_magnitude_s90_adjacency_list_b32_cuda1_rep0.json
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

from src.gcn.config import PROJECT_ROOT, set_seed
from src.gcn.measurement import (check_format_supported, configure_precision, cuda_memory_stats, device_info,
                                 resolve_device, summarize_latency, synchronize)
from src.gcn.pruning import tensor_sha256
from src.mlp.config import COMPARISON_CONFIG_PATH, load_comparison_config
from src.mlp.data import load_for_checkpoint
from src.mlp.energy import close_meter, open_meter, unavailable
from src.mlp.gpu_comparison import build, equivalence, exact_forward, find_spec, prepare, resident_ok, spec_fields
from src.mlp.harness import make_batch, time_inference
from src.mlp.model import checkpoint_identity, load_mlp_checkpoint

SCHEMA_VERSION = 1


def measure_energy(model, x, device, config, enabled):
    if not enabled:
        return unavailable("disabled with --no-energy")
    meter, reason = open_meter(device, config["energy_sample_interval_ms"])
    if meter is None:
        return unavailable(reason)

    def step():
        model(x)
        synchronize(device)

    try:
        with torch.inference_mode():
            return meter.measure(step, len(x), config["energy_seconds"], config["energy_idle_seconds"],
                                 config["energy_preroll_seconds"])
    except Exception as error:  # energy failing must not lose the latency / memory result
        return unavailable(f"measurement failed: {error!r}")
    finally:
        close_meter(meter)


def run(args):
    config = load_comparison_config(args.config)
    set_seed(0)
    device = resolve_device(args.device)
    check_format_supported(device, args.storage_format)
    precision = configure_precision(device, config["tf32"])
    run_tag = args.device.replace(":", "")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{args.seed}_{args.checkpoint_run or run_tag}.pt"

    model, checkpoint = load_mlp_checkpoint(checkpoint_path)
    data = load_for_checkpoint(checkpoint)
    spec = find_spec(config, args.seed, args.tag)
    if args.storage_format not in spec["formats"]:
        raise ValueError(f"{args.tag} is not benchmarked in {args.storage_format}")

    start = time.perf_counter()
    prepared = prepare(spec, model, config)
    prepare_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    cpu_model = build(spec, prepared, model, args.storage_format)
    build_ms = (time.perf_counter() - start) * 1e3
    storage = cpu_model.storage()
    start = time.perf_counter()
    device_model = cpu_model.to(device)
    synchronize(device)
    transfer_ms = (time.perf_counter() - start) * 1e3
    del cpu_model
    if not resident_ok(device_model, args.storage_format):
        raise RuntimeError("a sparse or list format holds a dense weight matrix or mask on the device")

    x = make_batch(data.x_test, args.batch_size, device)
    y = data.y_test[:args.batch_size]
    resident_tensor_bytes = device_model.resident_bytes() + x.numel() * x.element_size()
    resident = cuda_memory_stats(device)["allocated_bytes"]
    logits, times_ms = time_inference(device_model, x, device, config["warmup_runs"], config["measured_runs"])
    memory = cuda_memory_stats(device)
    logits = logits.cpu()
    energy = measure_energy(device_model, x, device, config, not args.no_energy)
    del device_model

    format_check = None
    reference_format = spec["reference_format"]
    if reference_format is not None and args.storage_format != reference_format:
        reference = build(spec, prepared, model, reference_format).to(device)
        with torch.inference_mode():
            reference_logits = reference(x).cpu()
        exact = exact_forward(spec, prepared, model)(x)
        format_check = equivalence(reference_logits, logits, exact, y, torch.ones(len(y), dtype=torch.bool))
        format_check["reference"] = f"{reference_format} on {device}, same batch"

    summary = prepared["summary"]
    official = device.type == "cuda"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "experiment": "mlp_gpu_comparison_benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "fashion_mnist",
        "checkpoint": checkpoint_identity(checkpoint_path, checkpoint),
        **spec_fields(spec),
        "actual_sparsity": summary.get("actual_sparsity"),
        "edges": (summary["factor_edges"] if spec["kind"] == "low_rank"
                  else sum(int(m.sum()) for m in prepared.get("masks", {}).values()) or None),
        "pruning": summary if spec["kind"] != "low_rank" else None,
        "low_rank": summary if spec["kind"] == "low_rank" else None,
        "format": args.storage_format,
        "reference_format": reference_format,
        "batch_size": args.batch_size,
        "repetition": args.repetition,
        **device_info(device),
        "latency_is_debug_only": not official,
        "note": ("CUDA run." if official else
                 "CPU latency is for debugging only; it is not a GPU result and must not be reported as one."),
        "dtype": config["dtype"],
        **precision,
        "storage": storage,
        "weight_storage_bytes": storage["weight_bytes"],
        "bias_bytes": storage["bias_bytes"],
        "resident_tensor_bytes": resident_tensor_bytes,
        "timing_ms": {"prepare_ms": prepare_ms, "format_build_ms": build_ms, "transfer_ms": transfer_ms},
        "warmup_runs": config["warmup_runs"],
        "measured_runs": config["measured_runs"],
        "latency_ms": summarize_latency(times_ms),
        "latency_samples_ms": times_ms,
        "peak_reset": "after warm-up, before the measured passes",
        "cuda_allocated_before_inference_bytes": resident,
        "cuda_peak_allocated_bytes": memory["peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": memory["peak_reserved_bytes"],
        "energy": energy,
        "energy_joules": energy.get("energy_per_batch_j"),  # per batch, gross (the original sweep's empty column)
        "format_check": format_check,
        "logits_sha256": tensor_sha256(logits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True, help="model tag from the plan, e.g. m42_magnitude_s90")
    parser.add_argument("--format", required=True, dest="storage_format")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint-run", default=None, help="trained checkpoints (default: the device tag)")
    parser.add_argument("--config", default=str(COMPARISON_CONFIG_PATH))
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--no-energy", action="store_true", help="skip the NVML energy measurement")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as error:
        output.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "status": "failed",
                                      "experiment": "mlp_gpu_comparison_benchmark",
                                      "timestamp": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                                      "error": repr(error), "traceback": traceback.format_exc()}, indent=2))
        print(f"FAILED: {error!r}. Wrote {output}")
        raise SystemExit(1)

    output.write_text(json.dumps(result, indent=2))
    energy, check = result["energy"], result["format_check"]
    label = "CUDA" if not result["latency_is_debug_only"] else "DEBUG ONLY (not a GPU result)"
    energy_text = (f", {1e6 * energy['energy_per_example_j']:.1f} uJ/example at {energy['average_power_w']:.1f} W "
                   f"(idle {energy['idle_power_w']:.1f} W, {energy['method']}"
                   + (", OTHER PROCESSES ON GPU" if energy["other_processes_present"] else "") + ")"
                   if energy["available"] else f", energy: {energy['reason']}")
    print(f"{args.tag} {args.storage_format} b{args.batch_size} rep {args.repetition}: median "
          f"{result['latency_ms']['median']:.4f} ms [{label}], weights {result['weight_storage_bytes']:,} B"
          + energy_text + ("" if check is None else
                           f", vs {result['reference_format']}: same predictions {check['same_predictions']}, "
                           f"within tolerance {check['equivalent']}"))


if __name__ == "__main__":
    main()
