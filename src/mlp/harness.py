"""Model loading and inference entry points for the benchmark harness (Dai).

    variant = load_variant(checkpoint, method, sparsity, pruning_seed, storage_format, device)
    x = make_batch(test_images, batch_size, device)
    with torch.inference_mode():
        logits = variant.model(x)

`variant.metadata` holds what every result must record: model seed, pruning
seed, method, requested and actual sparsity, format, storage bytes and the
checkpoint identity. The harness adds batch size, repetition and its own
timing, memory and energy numbers.

GPU residency: the checkpoint is loaded and pruned on CPU, the requested
format is built on CPU, and only that format's tensors (plus biases) are
moved to the device. The dense nn.Module never leaves the CPU, so a sparse
run holds no dense weights or masks on the GPU. `variant.model.tensors()`
lists exactly what is resident, and `check_resident` verifies it.

`time_inference` is the reference timing loop used by
scripts/benchmark_mlp.py, matching the GCN benchmarks (synchronized samples,
peak counters reset after the warm-up).
"""

import time
from dataclasses import dataclass

import torch

from src.gcn.measurement import reset_peak_memory, synchronize

from .formats import FORMATS, SPARSE_FORMATS, build_format_model
from .model import checkpoint_identity, load_mlp_checkpoint
from .weight_pruning import prune_layers


@dataclass
class Variant:
    model: object        # FormatMLP on the target device
    metadata: dict
    timing_ms: dict      # pruning, format build and transfer: never part of inference latency


def load_variant(checkpoint_path, method, sparsity, pruning_seed, storage_format, device, pruned_layers=None):
    """One model variant, built on CPU, with only its own format moved to `device`."""
    if method == "none" and (sparsity != 0 or storage_format != "dense"):
        raise ValueError("the original model ('none') is dense at sparsity 0")
    if storage_format not in FORMATS:
        raise ValueError(f"unknown format {storage_format!r}")
    device = torch.device(device)
    model, checkpoint = load_mlp_checkpoint(checkpoint_path)
    pruned_layers = list(pruned_layers or model.layer_names)

    start = time.perf_counter()
    if method == "none":
        masks, summary = {}, {"method": "none", "pruning_seed": None, "requested_sparsity": 0.0,
                              "actual_sparsity": 0.0, "layers": {}}
    else:
        weights = {name: getattr(model, name).weight.detach() for name in pruned_layers}
        pruned = prune_layers(weights, method, sparsity, pruning_seed)
        masks, summary = pruned["masks"], pruned["summary"]
    pruning_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    cpu_model = build_format_model(model, masks, storage_format)
    build_ms = (time.perf_counter() - start) * 1e3
    del model, masks  # nothing dense is kept beyond what the format itself stores

    start = time.perf_counter()
    device_model = cpu_model.to(device)
    synchronize(device)
    transfer_ms = (time.perf_counter() - start) * 1e3

    metadata = {
        "checkpoint": checkpoint_identity(checkpoint_path, checkpoint),
        "model_seed": checkpoint["seed"],
        "method": method,
        "pruning_seed": summary["pruning_seed"],
        "requested_sparsity": summary["requested_sparsity"],
        "actual_sparsity": summary["actual_sparsity"],
        "pruned_layers": pruned_layers if method != "none" else [],
        "pruning": summary,
        "format": storage_format if method != "none" else "dense",
        "condition": "original" if method == "none" else "pruned",
        "storage": cpu_model.storage(),
        "data_normalization": checkpoint["data"]["normalization"],
    }
    return Variant(device_model, metadata, {"pruning_ms": pruning_ms, "format_build_ms": build_ms,
                                             "transfer_ms": transfer_ms})


def make_batch(x_test, batch_size, device):
    """The first `batch_size` test images (already standardized), on `device`."""
    if batch_size > len(x_test):
        raise ValueError(f"batch size {batch_size} exceeds {len(x_test)} test images")
    return x_test[:batch_size].contiguous().to(device)


def check_resident(variant):
    """Confirm a sparse variant holds no dense weight matrix or mask on the device."""
    if variant.metadata["format"] not in SPARSE_FORMATS:
        return True
    for layer in variant.model.layers:
        if layer.pruned and (layer.weight.layout == torch.strided or layer.mask is not None):
            return False
    return True


def time_inference(model, x, device, warmup_runs, measured_runs):
    """Warm-ups, peak reset, then synchronized per-pass latencies in ms.

    Returns (logits of the last pass, list of latencies).
    """
    times_ms = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(x)
        synchronize(device)
        reset_peak_memory(device)
        for _ in range(measured_runs):
            start = time.perf_counter()
            logits = model(x)
            synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1e3)
    return logits, times_ms
