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

Adjacency list (adjacency_list, not part of the original sweep's FORMATS):
each output neuron keeps the list of its incoming edges, (input neuron,
weight), built from the mask positions like COO / CSR. The lists are packed
the standard GPU way: one offsets array (where each neuron's list starts),
one neighbor-index array and one weight array, so the stored bytes are
4 B per edge value + 8 B per neighbor index + 8 B per offset (the same
accounting as reports/low_rank_adjacency.csv). A forward pass is computed
edge by edge with generic PyTorch kernels: gather each edge's source
activation, multiply by the edge weight, and sum each neuron's list
(torch.segment_reduce over the offsets). It holds the same numbers as CSR;
what differs is the computation (graph-style gather and per-list sums vs
cuSPARSE's sparse-matrix multiply).

Low-rank adapters (LoRA) are optional per layer and always stay separate from
the stored weight: a layer computes W x + B (A x) + b, with A (rank x in) and
B (out x rank) dense FP32 (B already includes the alpha / rank scale). They
are never merged into W, so a sparse W stays sparse. Their bytes are reported
as adapter_bytes, next to weight_bytes.
"""

import torch
import torch.nn.functional as F

from .weight_pruning import apply_mask

FORMATS = ("dense", "dense_mask", "coo", "csr")
SPARSE_FORMATS = ("coo", "csr")
ADJACENCY_FORMAT = "adjacency_list"
ALL_FORMATS = FORMATS + (ADJACENCY_FORMAT,)  # FORMATS stays the original sweep's grid
FORMAT_LABELS = {"dense": "pruned dense", "dense_mask": "dense+mask", "coo": "COO", "csr": "CSR",
                 "adjacency_list": "adjacency list"}


def _nbytes(t):
    return t.numel() * t.element_size()


def sparse_from_mask(weight, mask):
    """Coalesced COO of the retained entries, in row-major order."""
    indices = mask.nonzero().t()
    values = weight[mask]
    coo = torch.sparse_coo_tensor(indices, values, weight.shape, check_invariants=True)
    return coo.coalesce()


class AdjacencyList:
    """Weighted adjacency lists of one weight matrix (rows = output neurons).

    Row r's incoming edges are neighbors[offsets[r]:offsets[r + 1]] (input
    neuron indices, ascending) with weights[offsets[r]:offsets[r + 1]].
    """

    layout = "adjacency_list"  # never torch.strided: a resident check treats it like a sparse tensor

    def __init__(self, offsets, neighbors, weights, shape):
        self.offsets, self.neighbors, self.weights = offsets, neighbors, weights
        self.shape = torch.Size(shape)

    @classmethod
    def from_mask(cls, weight, mask):
        """Lists of the retained positions of `mask` (row-major, like COO / CSR), values from `weight`."""
        if mask.shape != weight.shape or mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool with shape {tuple(weight.shape)}")
        weight, mask = weight.detach().cpu(), mask.cpu()
        positions = mask.nonzero()  # row-major, so each row's neighbors are contiguous and ascending
        offsets = torch.zeros(weight.shape[0] + 1, dtype=torch.int64)
        offsets[1:] = mask.sum(dim=1).cumsum(0)
        lists = cls(offsets, positions[:, 1].contiguous(), weight[mask].clone(), weight.shape)
        lists.validate()
        return lists

    def validate(self):
        """Checked once here, so the forward pass can skip segment_reduce's own (synchronizing) checks."""
        offsets = self.offsets.cpu()
        if (offsets.dim() != 1 or len(offsets) != self.shape[0] + 1 or offsets[0] != 0
                or offsets[-1] != len(self.neighbors) or bool((offsets.diff() < 0).any())
                or len(self.weights) != len(self.neighbors)):
            raise ValueError("malformed adjacency list offsets")
        if len(self.neighbors) and not (0 <= int(self.neighbors.min()) and int(self.neighbors.max()) < self.shape[1]):
            raise ValueError("adjacency list neighbor index out of range")

    @property
    def num_edges(self):
        return len(self.neighbors)

    def to(self, device):
        return AdjacencyList(self.offsets.to(device), self.neighbors.to(device), self.weights.to(device), self.shape)

    def tensors(self):
        return [self.weights, self.neighbors, self.offsets]

    def storage_parts(self):
        return {"values": _nbytes(self.weights), "neighbors": _nbytes(self.neighbors),
                "offsets": _nbytes(self.offsets)}

    def to_dense(self):
        dense = torch.zeros(self.shape, dtype=self.weights.dtype)
        rows = torch.repeat_interleave(torch.arange(self.shape[0]), self.offsets.cpu().diff())
        dense[rows, self.neighbors.cpu()] = self.weights.cpu()
        return dense

    def aggregate(self, h):
        """W h for feature-major activations h (in x batch, contiguous): (out x batch).

        One message per edge (its source activation times its weight), then
        each output neuron sums its own list; a neuron with no edges gets 0.
        """
        messages = h.index_select(0, self.neighbors) * self.weights[:, None]
        return torch.segment_reduce(messages, "sum", offsets=self.offsets, axis=0, unsafe=True)


class StoredLayer:
    """One Linear layer in one storage format."""

    def __init__(self, name, weight, bias, mask=None, storage_format="dense", pruned=False, adapter=None):
        self.name, self.weight, self.bias, self.mask = name, weight, bias, mask
        self.storage_format, self.pruned, self.adapter = storage_format, pruned, adapter

    def to(self, device):
        mask = None if self.mask is None else self.mask.to(device)
        adapter = None if self.adapter is None else tuple(t.to(device) for t in self.adapter)
        return StoredLayer(self.name, self.weight.to(device), self.bias.to(device), mask,
                           self.storage_format, self.pruned, adapter)

    def tensors(self):
        """Every tensor buffer this layer keeps (sparse tensors as their components)."""
        w = self.weight
        if isinstance(w, AdjacencyList):
            out = w.tensors()
        elif w.layout == torch.sparse_coo:
            out = [w._values(), w._indices()]
        elif w.layout == torch.sparse_csr:
            out = [w.values(), w.col_indices(), w.crow_indices()]
        else:
            out = [w]
        if self.mask is not None:
            out.append(self.mask)
        if self.adapter is not None:
            out += list(self.adapter)
        return out + [self.bias]

    def storage(self):
        w = self.weight
        if isinstance(w, AdjacencyList):
            values = w.weights
            parts = w.storage_parts()
            index_dtype = str(w.neighbors.dtype)
        elif w.layout == torch.sparse_coo:
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
                "value_dtype": str(values.dtype), "index_dtype": index_dtype,
                "mask_dtype": None if self.mask is None else str(self.mask.dtype),
                "adapter_rank": None if self.adapter is None else self.adapter[0].shape[0],
                "adapter_bytes": 0 if self.adapter is None else sum(_nbytes(t) for t in self.adapter),
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
        weight_bytes = sum(layer["weight_bytes"] for layer in layers.values())
        adapter_bytes = sum(layer["adapter_bytes"] for layer in layers.values())
        return {"format": self.storage_format, "weight_bytes": weight_bytes, "adapter_bytes": adapter_bytes,
                "stored_bytes": weight_bytes + adapter_bytes,
                "bias_bytes": sum(layer["bias_bytes"] for layer in layers.values()),
                "layers": layers}

    def __call__(self, x):
        last = len(self.layers) - 1
        if self.storage_format == ADJACENCY_FORMAT:
            h = x.t().contiguous()  # feature-major: each edge gathers one row of h
            for i, layer in enumerate(self.layers):
                if isinstance(layer.weight, AdjacencyList):
                    out = layer.weight.aggregate(h) + layer.bias[:, None]
                else:  # a layer that is not pruned stays dense
                    out = torch.addmm(layer.bias[:, None], layer.weight, h)
                h = torch.relu(out) if i < last else out
            return h.t()
        if not self.sparse:
            h = x
            for i, layer in enumerate(self.layers):
                weight = layer.weight if layer.mask is None else layer.weight * layer.mask
                out = F.linear(h, weight, layer.bias)
                if layer.adapter is not None:
                    a, b = layer.adapter
                    out = out + F.linear(F.linear(h, a), b)
                h = torch.relu(out) if i < last else out
            return h
        h = x.t()
        for i, layer in enumerate(self.layers):
            if layer.weight.layout == torch.strided:  # a layer that is not pruned stays dense
                out = torch.addmm(layer.bias[:, None], layer.weight, h)
            else:
                out = torch.sparse.mm(layer.weight, h) + layer.bias[:, None]
            if layer.adapter is not None:
                a, b = layer.adapter
                out = out + b @ (a @ h)
            h = torch.relu(out) if i < last else out
        return h.t()


def build_format_model(model, masks, storage_format, adapters=None):
    """FormatMLP on CPU for `model` (an src/mlp/model.MLP) pruned by `masks`.

    masks: {layer name: keep-mask} for the pruned layers ({} = original model,
    dense only). Layers without a mask are stored dense in every format.
    adapters: optional {layer name: (A, B)} low-rank adapters, kept separate.
    Only the tensors of the requested format are created.
    """
    adapters = adapters or {}
    if storage_format not in ALL_FORMATS:
        raise ValueError(f"unknown format {storage_format!r} (available: {ALL_FORMATS})")
    if storage_format == ADJACENCY_FORMAT and adapters:
        raise ValueError("adapters are not supported with adjacency lists")
    if not masks and storage_format != "dense":
        raise ValueError("the unpruned model is only stored dense")
    unknown = (set(masks) | set(adapters)) - set(model.layer_names)
    if unknown:
        raise ValueError(f"masks or adapters for unknown layers {sorted(unknown)}")

    layers = []
    for name, linear in zip(model.layer_names, model.layers()):
        weight, bias = linear.weight.detach().cpu(), linear.bias.detach().cpu().clone()
        adapter = adapters.get(name)
        if adapter is not None:
            a, b = (t.detach().cpu().clone() for t in adapter)
            if a.shape[1] != weight.shape[1] or b.shape[0] != weight.shape[0] or a.shape[0] != b.shape[1]:
                raise ValueError(f"{name}: adapter shapes {tuple(a.shape)}, {tuple(b.shape)} do not fit "
                                 f"weight {tuple(weight.shape)}")
            adapter = (a, b)
        mask = masks.get(name)
        if mask is None:
            layers.append(StoredLayer(name, weight.clone(), bias, adapter=adapter))
            continue
        if mask.shape != weight.shape or mask.dtype != torch.bool:
            raise ValueError(f"{name}: mask must be bool with shape {tuple(weight.shape)}")
        if storage_format == "dense":
            stored = StoredLayer(name, apply_mask(weight, mask), bias, storage_format="dense", pruned=True,
                                 adapter=adapter)
        elif storage_format == "dense_mask":
            stored = StoredLayer(name, weight.clone(), bias, mask.clone(), "dense_mask", pruned=True,
                                 adapter=adapter)
        elif storage_format == ADJACENCY_FORMAT:
            stored = StoredLayer(name, AdjacencyList.from_mask(weight, mask), bias, storage_format=storage_format,
                                 pruned=True)
        else:
            sparse = sparse_from_mask(weight, mask)
            stored = StoredLayer(name, sparse if storage_format == "coo" else sparse.to_sparse_csr(), bias,
                                 storage_format=storage_format, pruned=True, adapter=adapter)
        layers.append(stored)
    return FormatMLP(layers, storage_format)
