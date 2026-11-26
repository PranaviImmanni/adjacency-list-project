"""Low-rank factor networks with adjacency-list factors (Pranavi's experiment, on the GPU harness).

Each weight matrix W (out x in) is replaced by a truncated SVD

    W ~= B A,   B = U_r diag(S_r)  (out x r),   A = V_r^T  (r x in),

computed exactly as in scripts/benchmark_low_rank_adjacency.py: NumPy SVD in
float64 of the trained weights, rank capped at min(out, in) (fc3 is 10 x 100,
so its rank is at most 10). Factor entries are then selected per factor F:

    all      keep every entry
    abs_std  keep f when |f| > mean(|F|) + k std(|F|)   (population std, float64)

and the kept values are stored in FP32. Nothing is trained: this replaces W
with its approximation (it is not LoRA, which keeps W and adds a trained
update).

Formats:
  factor_adjacency  both factors as adjacency lists (src/mlp/formats.AdjacencyList);
                    a layer computes B-lists(A-lists(h)) + b, edge by edge
  factor_dense      both factors as dense FP32 matrices with the unselected
                    entries set to 0; F.linear twice. This is the dense
                    implementation of the same modified model, used as the
                    reference for the equivalence check (and timed too)

Storage of a factor adjacency list: 4 B value + 8 B neighbor index per kept
entry, plus 8 B per (rows + 1) offsets, which is the accounting of
reports/low_rank_adjacency.csv.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .formats import AdjacencyList, _nbytes

FACTOR_FORMATS = ("factor_adjacency", "factor_dense")
FACTOR_RULES = ("all", "abs_std")
FORMAT_LABELS = {"factor_adjacency": "factor adjacency list", "factor_dense": "dense factors"}


def rule_label(rule, k=None):
    """all, abs_std_k0, abs_std_k1 (used in tags and tables)."""
    if rule == "all":
        return "all"
    if rule == "abs_std":
        return f"abs_std_k{k:g}".replace(".", "p")
    raise ValueError(f"factor rule must be one of {FACTOR_RULES}")


def factorize(weight, rank):
    """(B, A) in float64 NumPy, as Pranavi's factorize(): B = U_r * S_r, A = Vt_r."""
    w = weight.detach().cpu().to(torch.float64).numpy()
    u, singular, vt = np.linalg.svd(w, full_matrices=False)
    rank = min(int(rank), len(singular))
    return u[:, :rank] * singular[:rank], vt[:rank, :]


def factor_mask(factor, rule, k=None):
    """(keep mask, threshold) for one float64 factor, as Pranavi's factor_mask()."""
    if rule == "all":
        return np.ones(factor.shape, dtype=bool), None
    if rule != "abs_std":
        raise ValueError(f"factor rule must be one of {FACTOR_RULES}")
    magnitudes = np.abs(factor)
    threshold = float(magnitudes.mean() + k * magnitudes.std())
    return magnitudes > threshold, threshold


def low_rank_factors(model, rank, rule, k=None):
    """Factors, masks and a summary for every layer of `model` (an src/mlp/model.MLP).

    Returns {"layers": {name: {"B", "A" (FP32 tensors, unselected entries 0), "B_mask", "A_mask",
    "bias"}}, "summary": {...}}. The summary records the rank used per layer, kept entries,
    thresholds and the weight reconstruction error (unselected factors, as in Pranavi's CSV, and
    selected factors).
    """
    if rule == "abs_std" and k is None:
        raise ValueError("abs_std needs k")
    layers, per_layer, thresholds = {}, {}, []
    mse_full, mse_kept = [], []
    total_entries = total_edges = 0
    for name, linear in zip(model.layer_names, model.layers()):
        weight = linear.weight.detach().cpu()
        b, a = factorize(weight, rank)
        b_mask, b_threshold = factor_mask(b, rule, k)
        a_mask, a_threshold = factor_mask(a, rule, k)
        w64 = weight.to(torch.float64).numpy()
        b32, a32 = b.astype(np.float32), a.astype(np.float32)  # stored values are FP32, as in Pranavi's lists
        mse_full.append(float(np.mean((w64 - b.dot(a)) ** 2)))
        kept = np.where(b_mask, b32, 0).astype(np.float64).dot(np.where(a_mask, a32, 0).astype(np.float64))
        mse_kept.append(float(np.mean((w64 - kept) ** 2)))
        layers[name] = {
            "B": torch.from_numpy(np.where(b_mask, b32, np.float32(0))),
            "A": torch.from_numpy(np.where(a_mask, a32, np.float32(0))),
            "B_mask": torch.from_numpy(b_mask), "A_mask": torch.from_numpy(a_mask),
            "bias": linear.bias.detach().cpu().clone(),
        }
        edges = int(b_mask.sum() + a_mask.sum())
        entries = b.size + a.size
        total_edges += edges
        total_entries += entries
        if rule == "abs_std":
            thresholds += [b_threshold, a_threshold]
        per_layer[name] = {
            "shape": list(weight.shape), "rank": int(a.shape[0]),
            "B_shape": list(b.shape), "A_shape": list(a.shape),
            "B_edges": int(b_mask.sum()), "A_edges": int(a_mask.sum()), "factor_entries": int(entries),
            "B_threshold": b_threshold, "A_threshold": a_threshold,
            "weight_reconstruction_mse": mse_full[-1], "kept_factor_reconstruction_mse": mse_kept[-1],
        }
    summary = {
        "rank": int(rank), "factor_rule": rule, "k": k if rule == "abs_std" else None,
        "rule_label": rule_label(rule, k),
        "factor_edges": total_edges, "factor_entries": total_entries,
        "factor_sparsity": 1.0 - total_edges / total_entries,
        "mean_weight_reconstruction_mse": float(np.mean(mse_full)),  # Pranavi's column (factors before selection)
        "mean_kept_factor_reconstruction_mse": float(np.mean(mse_kept)),
        "threshold_min": min(thresholds) if thresholds else None,
        "threshold_max": max(thresholds) if thresholds else None,
        "selection": ("every factor entry" if rule == "all" else
                      f"|f| > mean(|F|) + {k:g} * std(|F|) per factor (population std, float64)"),
        "layers": per_layer,
    }
    return {"layers": layers, "summary": summary}


class LowRankMLP:
    """Inference-only MLP whose layers are factor pairs. Call it under torch.inference_mode()."""

    def __init__(self, layers, storage_format):
        """layers: [(name, A store, B store, bias)]; stores are AdjacencyList or dense FP32 tensors."""
        if storage_format not in FACTOR_FORMATS:
            raise ValueError(f"unknown factor format {storage_format!r} (available: {FACTOR_FORMATS})")
        self.layers, self.storage_format = layers, storage_format

    def to(self, device):
        return LowRankMLP([(name, a.to(device), b.to(device), bias.to(device)) for name, a, b, bias in self.layers],
                          self.storage_format)

    def tensors(self):
        out = []
        for _, a, b, bias in self.layers:
            for store in (a, b):
                out += store.tensors() if isinstance(store, AdjacencyList) else [store]
            out.append(bias)
        return out

    def resident_bytes(self):
        return sum(_nbytes(t) for t in self.tensors())

    def holds_only_lists(self):
        """factor_adjacency keeps no dense factor matrix on the device."""
        return all(isinstance(a, AdjacencyList) and isinstance(b, AdjacencyList) for _, a, b, _ in self.layers)

    def storage(self):
        layers = {}
        for name, a, b, bias in self.layers:
            entry = {"bias_bytes": _nbytes(bias)}
            for label, store in (("A", a), ("B", b)):
                if isinstance(store, AdjacencyList):
                    parts = store.storage_parts()
                    entry[label] = {"shape": list(store.shape), "stored_values": store.num_edges,
                                    "bytes": sum(parts.values()), "bytes_breakdown": parts}
                else:
                    entry[label] = {"shape": list(store.shape), "stored_values": store.numel(),
                                    "bytes": _nbytes(store), "bytes_breakdown": {"values": _nbytes(store)}}
            entry["weight_bytes"] = entry["A"]["bytes"] + entry["B"]["bytes"]
            entry["rank"] = entry["A"]["shape"][0]
            layers[name] = entry
        weight_bytes = sum(layer["weight_bytes"] for layer in layers.values())
        return {"format": self.storage_format, "weight_bytes": weight_bytes, "adapter_bytes": 0,
                "stored_bytes": weight_bytes, "bias_bytes": sum(layer["bias_bytes"] for layer in layers.values()),
                "layers": layers}

    def __call__(self, x):
        last = len(self.layers) - 1
        if self.storage_format == "factor_adjacency":
            h = x.t().contiguous()
            for i, (_, a, b, bias) in enumerate(self.layers):
                out = b.aggregate(a.aggregate(h)) + bias[:, None]
                h = torch.relu(out) if i < last else out
            return h.t()
        h = x
        for i, (_, a, b, bias) in enumerate(self.layers):
            out = F.linear(F.linear(h, a), b, bias)
            h = torch.relu(out) if i < last else out
        return h


def build_low_rank_model(factors, storage_format):
    """LowRankMLP on CPU from low_rank_factors(...)["layers"]."""
    layers = []
    for name, f in factors.items():
        if storage_format == "factor_adjacency":
            a, b = AdjacencyList.from_mask(f["A"], f["A_mask"]), AdjacencyList.from_mask(f["B"], f["B_mask"])
        else:
            a, b = f["A"].clone(), f["B"].clone()
        layers.append((name, a, b, f["bias"].clone()))
    return LowRankMLP(layers, storage_format)
