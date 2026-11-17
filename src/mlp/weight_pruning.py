"""Signed-weight pruning: the interface shared with Pranavi, plus a reference implementation.

Interface
---------
    prune_layers(weights, method, sparsity, pruning_seed=None) -> {"masks": {layer: keep-mask}, "summary": {...}}
    prune_mask(weight, method, sparsity, seed=None)            -> keep-mask for a single matrix

`weights` maps layer names to Linear weights [out_features, in_features], in
model order. A mask is a CPU bool tensor, True = keep. Pruning only chooses
positions; it never changes a value. The storage formats apply the mask
(src/mlp/formats.py), so retained weights keep their exact values and signs,
and biases are never pruned.

Methods (as in the project plan)
--------------------------------
- magnitude (global): one threshold over all pruned layers together. The
  floor(s * N) entries with the smallest |w| among all N pruned weights are
  removed, so each layer ends up with its own sparsity. Ties are removed in
  layer order, then row-major order. At a fixed target sparsity the plan's
  "global percentile" cutoff selects exactly this set, so it is not a
  separate condition.
- topk_per_neuron: every row (output neuron) of every layer removes its
  floor(s * in) smallest |w|, so every row keeps the same K = in - floor(s * in).
  Ties are removed in column order.
- random (global): removes the first floor(s * N) positions of
  torch.randperm(N) over all pruned weights (concatenated in layer order,
  row-major), from a generator seeded with the pruning seed.

Contract (checked by tests/test_weight_pruning.py for any implementation)
---------------------------------------------------------------------------
- Inputs are not modified; masks are CPU bool tensors of the weights' shapes.
- Counts use floor with exact decimals: Fraction(str(sparsity)).
- Nested: for one method and seed, whatever is removed at a lower sparsity
  is also removed at every higher sparsity.
- Magnitude uses |w|, so a large negative weight is kept over a small
  positive one. Signs play no other role.
"""

import hashlib
from fractions import Fraction
from math import floor

import torch

METHODS = ("magnitude", "topk_per_neuron", "random")


def removal_count(n, sparsity):
    """floor(sparsity * n), exact for decimal sparsities such as 0.95."""
    if not 0 <= sparsity < 1:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    return floor(Fraction(str(sparsity)) * n)


def _split(flat_keep, weights):
    masks, start = {}, 0
    for name, w in weights.items():
        masks[name] = flat_keep[start:start + w.numel()].reshape(w.shape).clone()
        start += w.numel()
    return masks


def _keep_masks(weights, method, sparsity, pruning_seed):
    weights = {name: w.detach().to("cpu") for name, w in weights.items()}
    for name, w in weights.items():
        if w.dim() != 2:
            raise ValueError(f"{name}: expected a 2-D weight, got shape {tuple(w.shape)}")

    if method == "topk_per_neuron":
        masks = {}
        for name, w in weights.items():
            keep = torch.ones(w.shape, dtype=torch.bool)
            order = torch.sort(w.abs(), dim=1, stable=True).indices
            keep.scatter_(1, order[:, :removal_count(w.shape[1], sparsity)], False)
            masks[name] = keep
        return masks

    total = sum(w.numel() for w in weights.values())
    k = removal_count(total, sparsity)
    if method == "magnitude":
        flat = torch.cat([w.abs().flatten() for w in weights.values()])
        order = torch.sort(flat, stable=True).indices
    elif method == "random":
        if pruning_seed is None:
            raise ValueError("random pruning needs a seed")
        order = torch.randperm(total, generator=torch.Generator().manual_seed(pruning_seed))
    else:
        raise ValueError(f"unknown method {method!r} (available: {METHODS})")
    keep = torch.ones(total, dtype=torch.bool)
    keep[order[:k]] = False
    return _split(keep, weights)


def mask_sha256(mask):
    return hashlib.sha256(mask.to(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def prune_layers(weights, method, sparsity, pruning_seed=None):
    """Masks for the pruned layers plus requested and actual sparsity, per layer and overall."""
    masks = _keep_masks(weights, method, sparsity, pruning_seed)
    layers = {}
    for name, mask in masks.items():
        retained = int(mask.sum())
        layers[name] = {"shape": list(mask.shape), "numel": mask.numel(), "retained": retained,
                        "removed": mask.numel() - retained,
                        "actual_sparsity": (mask.numel() - retained) / mask.numel(),
                        "mask_sha256": mask_sha256(mask)}
    total = sum(layer["numel"] for layer in layers.values())
    removed = sum(layer["removed"] for layer in layers.values())
    summary = {"method": method, "scope": "per output neuron" if method == "topk_per_neuron" else "global",
               "pruning_seed": pruning_seed if method == "random" else None,
               "requested_sparsity": sparsity, "actual_sparsity": removed / total,
               "pruned_weights": total, "removed_weights": removed, "layers": layers}
    return {"masks": masks, "summary": summary}


def prune_mask(weight, method, sparsity, seed=None):
    """Keep-mask for a single matrix (global methods then act on this matrix alone)."""
    return _keep_masks({"weight": weight}, method, sparsity, seed)["weight"]


def apply_mask(weight, mask):
    """Pruned weight: retained entries unchanged (same value and sign), the rest +0.0."""
    return torch.where(mask, weight, torch.zeros((), dtype=weight.dtype))
