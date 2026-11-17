"""Storage formats for the MLP weights and one inference model that runs any of them.

Formats of a pruned model (the original model is condition "none" in "dense"):
  dense       pruned weights stored densely, zeros already applied
  dense_mask  original dense weights + a bool mask, multiplied at every
              forward pass (what torch.nn.utils.prune does)
  coo, csr    only the retained weights, built from the mask positions (not
              from "value != 0"), so a retained weight that is exactly 0.0 is
              still stored and every format holds the same retained set

Nothing from the GCN pipeline is applied to weights: no symmetrization, no
self-loops, no normalization. A weight matrix is used exactly as trained.

Layouts: dense formats compute y = x W^T + b (F.linear). Sparse formats keep
activations feature-major and compute h' = W h + b with torch.sparse.mm,
because torch.sparse.mm needs the sparse operand first. Both compute the same
function; only the floating-point summation order differs.

Biases are always dense and never pruned. They are counted separately from
the weight storage.
"""

import torch
import torch.nn.functional as F

from .weight_pruning import apply_mask

FORMATS = ("dense", "dense_mask", "coo", "csr")
SPARSE_FORMATS = ("coo", "csr")
FORMAT_LABELS = {"dense": "pruned dense", "dense_mask": "dense+mask", "coo": "COO", "csr": "CSR"}


def _nbytes(t):
    return t.numel() * t.element_size()


def sparse_from_mask(weight, mask):
    """Coalesced COO of the retained entries, in row-major order."""
    indices = mask.nonzero().t()
    values = weight[mask]
    coo = torch.sparse_coo_tensor(indices, values, weight.shape, check_invariants=True)
    return coo.coalesce()


class StoredLayer:
    """One Linear layer in one storage format."""

    def __init__(self, name, weight, bias, mask=None, storage_format="dense", pruned=False):
        self.name, self.weight, self.bias, self.mask = name, weight, bias, mask
        self.storage_format, self.pruned = storage_format, pruned

    def to(self, device):
        mask = None if self.mask is None else self.mask.to(device)
        return StoredLayer(self.name, self.weight.to(device), self.bias.to(device), mask,
                           self.storage_format, self.pruned)

    def tensors(self):
        """Every tensor buffer this layer keeps (sparse tensors as their components)."""
        w = self.weight
        if w.layout == torch.sparse_coo:
            out = [w._values(), w._indices()]
        elif w.layout == torch.sparse_csr:
            out = [w.values(), w.col_indices(), w.crow_indices()]
        else:
            out = [w]
        if self.mask is not None:
            out.append(self.mask)
        return out + [self.bias]

    def storage(self):
        w = self.weight
        if w.layout == torch.sparse_coo:
            values = w._values()
            parts = {"values": _nbytes(values), "indices": _nbytes(w._indices())}
            index_dtype = str(w._indices().dtype)
        elif w.layout == torch.sparse_csr:
            values = w.values()
            parts = {"values": _nbytes(values), "col_indices": _nbytes(w.col_indices()),
                     "crow_indices": _nbytes(w.crow_indices())}
            index_dtype = str(w.col_indices().dtype)
        else:
            values = w
            parts = {"values": _nbytes(w)}
            index_dtype = None
        if self.mask is not None:
            parts["mask"] = _nbytes(self.mask)
        return {"format": self.storage_format if self.pruned else "dense", "pruned": self.pruned,
                "shape": list(w.shape), "stored_values": values.numel(),
                "weight_bytes": sum(parts.values()), "weight_bytes_breakdown": parts,
                "value_dtype": str(w.dtype), "index_dtype": index_dtype,
                "mask_dtype": None if self.mask is None else str(self.mask.dtype),
                "bias_bytes": _nbytes(self.bias)}


class FormatMLP:
    """Inference-only MLP over StoredLayers. Call it under torch.inference_mode()."""

    def __init__(self, layers, storage_format):
        self.layers, self.storage_format = layers, storage_format
        self.sparse = storage_format in SPARSE_FORMATS

    def to(self, device):
        return FormatMLP([layer.to(device) for layer in self.layers], self.storage_format)

    def tensors(self):
        return [t for layer in self.layers for t in layer.tensors()]

    def resident_bytes(self):
        return sum(_nbytes(t) for t in self.tensors())

    def storage(self):
        layers = {layer.name: layer.storage() for layer in self.layers}
        return {"format": self.storage_format,
                "weight_bytes": sum(layer["weight_bytes"] for layer in layers.values()),
                "bias_bytes": sum(layer["bias_bytes"] for layer in layers.values()),
                "layers": layers}

    def __call__(self, x):
        last = len(self.layers) - 1
        if not self.sparse:
            h = x
            for i, layer in enumerate(self.layers):
                weight = layer.weight if layer.mask is None else layer.weight * layer.mask
                h = F.linear(h, weight, layer.bias)
                if i < last:
                    h = torch.relu(h)
            return h
        h = x.t()
        for i, layer in enumerate(self.layers):
            if layer.weight.layout == torch.strided:  # a layer that is not pruned stays dense
                h = torch.addmm(layer.bias[:, None], layer.weight, h)
            else:
                h = torch.sparse.mm(layer.weight, h) + layer.bias[:, None]
            if i < last:
                h = torch.relu(h)
        return h.t()


def build_format_model(model, masks, storage_format):
    """FormatMLP on CPU for `model` (an src/mlp/model.MLP) pruned by `masks`.

    masks: {layer name: keep-mask} for the pruned layers ({} = original model,
    dense only). Layers without a mask are stored dense in every format.
    Only the tensors of the requested format are created.
    """
    if storage_format not in FORMATS:
        raise ValueError(f"unknown format {storage_format!r} (available: {FORMATS})")
    if not masks and storage_format != "dense":
        raise ValueError("the unpruned model is only stored dense")
    unknown = set(masks) - set(model.layer_names)
    if unknown:
        raise ValueError(f"masks for unknown layers {sorted(unknown)}")

    layers = []
    for name, linear in zip(model.layer_names, model.layers()):
        weight, bias = linear.weight.detach().cpu(), linear.bias.detach().cpu().clone()
        mask = masks.get(name)
        if mask is None:
            layers.append(StoredLayer(name, weight.clone(), bias))
            continue
        if mask.shape != weight.shape or mask.dtype != torch.bool:
            raise ValueError(f"{name}: mask must be bool with shape {tuple(weight.shape)}")
        if storage_format == "dense":
            stored = StoredLayer(name, apply_mask(weight, mask), bias, storage_format="dense", pruned=True)
        elif storage_format == "dense_mask":
            stored = StoredLayer(name, weight.clone(), bias, mask.clone(), "dense_mask", pruned=True)
        else:
            sparse = sparse_from_mask(weight, mask)
            stored = StoredLayer(name, sparse if storage_format == "coo" else sparse.to_sparse_csr(), bias,
                                 storage_format=storage_format, pruned=True)
        layers.append(stored)
    return FormatMLP(layers, storage_format)
