"""Device handling, timing, and memory helpers for inference benchmarks.

Every CUDA-specific call is guarded by `device.type == "cuda"`, so this
module runs on a CPU-only (or Apple M1) machine. GPU metrics that do not
exist on the current device are reported as None, never 0.
"""

import platform
import statistics
import time

import torch

SPARSE_FORMATS = ("coo", "csr")


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {name} requested but CUDA is not available in this PyTorch build/machine")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"--device {name} requested but MPS is not available")
    if device.type not in ("cpu", "cuda", "mps"):
        raise RuntimeError(f"unsupported device type {device.type!r} (use cpu or cuda:N)")
    return device


def check_format_supported(device, storage_format):
    """Sparse-format comparisons are supported on CPU and CUDA only."""
    if device.type == "mps" and storage_format in SPARSE_FORMATS:
        raise ValueError(
            f"{storage_format.upper()} runs on MPS are not supported in this project: MPS sparse "
            "support is incomplete, so results would not be comparable. "
            "Use --device cpu for development or --device cuda:0 on the lab GPU."
        )


def configure_precision(device, tf32=False):
    """On CUDA, force IEEE FP32 matmul/conv (tf32=False) or allow TF32. No-op elsewhere.

    Returns a dict describing what was applied, for the results file.
    """
    if device.type != "cuda":
        return {"tf32_requested": tf32, "fp32_precision_applied": None}

    precision = "tf32" if tf32 else "ieee"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):  # PyTorch >= 2.9
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.conv.fp32_precision = precision
    else:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    return {"tf32_requested": tf32, "fp32_precision_applied": precision}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(device):
    """Current and peak CUDA memory in bytes; all None off CUDA."""
    if device.type == "cuda":
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return {"allocated_bytes": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None}


def time_forward(model, x, adj, device, warmup_runs, measured_runs):
    """Run warm-up passes, then time each measured forward pass in milliseconds.

    The device is synchronized before the clock starts and before it stops,
    so asynchronous CUDA work is included in each sample.

    Returns (logits from the last pass, list of per-run latencies in ms).
    """
    model.eval()
    times_ms = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(x, adj)
        synchronize(device)
        for _ in range(measured_runs):
            start = time.perf_counter()
            logits = model(x, adj)
            synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1e3)
    return logits, times_ms


def summarize_latency(times_ms):
    return {
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
    }


def device_info(device):
    info = {
        "device": str(device),
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(device)
    else:
        info["device_name"] = platform.processor() or platform.machine()
    return info
