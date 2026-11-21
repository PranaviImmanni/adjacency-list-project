"""INT8 storage of the original (unpruned) MLP, executed on the device.

Two real INT8 paths; nothing is simulated with FP32 "fake quantization":

  int8_weight_only  every weight stored as packed int8 (1 byte) with one FP32
                    scale per output row (symmetric: scale = max|w_row| / 127,
                    q = round(w / scale) in [-127, 127]). Each pass dequantizes
                    on the device (q * scale) and runs an FP32 matmul, so it
                    saves weight memory while the compute stays FP32.
  int8_w8a8         the same int8 weights; each layer's input is quantized per
                    row (per sample, symmetric absmax / 127) at run time, the
                    product is an int8 x int8 -> int32 matmul (torch._int_mm,
                    cuBLASLt integer GEMM on CUDA), rescaled to FP32, and the
                    FP32 bias is added. _int_mm needs inner and output
                    dimensions that are multiples of 8, so the weights are
                    stored zero-padded (300 -> 304, 100 -> 104, 10 -> 16), and
                    a batch with fewer rows than the device accepts is
                    zero-padded (both padding and its bytes are recorded).

Reference implementations of the same quantized model, used only for the
equivalence check (never timed):

  int8_weight_only_reference  FP32 dense model whose weights are the
                              dequantized q * scale (same values the int8
                              path computes on the fly)
  int8_w8a8_reference         same activation quantization, with the integer
                              matmul done exactly in float64 (a sum of at most
                              784 products of int8 values is exact)

Storage counts the int8 values, the FP32 scales (quantization metadata) and
any padding. Activation scales are computed per pass and are not stored.
"""

import torch
import torch.nn.functional as F

INT8_FORMATS = ("int8_weight_only", "int8_w8a8")
REFERENCE_FORMATS = {"int8_weight_only": "int8_weight_only_reference", "int8_w8a8": "int8_w8a8_reference"}
ALL_INT8_FORMATS = INT8_FORMATS + tuple(REFERENCE_FORMATS.values())
QMAX = 127
PAD_MULTIPLE = 8


def _nbytes(t):
    return t.numel() * t.element_size()


def _round_up(n, multiple):
    return (n + multiple - 1) // multiple * multiple


def quantize_weight(weight):
    """Per-output-row symmetric int8: (q int8 [out, in], scale float32 [out])."""
    amax = weight.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.round(weight / scale[:, None]).clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_rows(x):
    """Per-row (per-sample) symmetric int8 activations: (q int8, scale float32 [rows, 1])."""
    amax = x.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    return torch.round(x / scale).clamp(-QMAX, QMAX).to(torch.int8), scale


def int_mm_support(device, k=784, n=304):
    """Check torch._int_mm on `device`: the smallest batch it accepts and exactness against int64.

    Returns a dict for the results file. 'supported' is True only when the
    kernel ran on this device and matched an exact integer product.
    """
    device = torch.device(device)
    result = {"device": str(device), "function": "torch._int_mm", "available": hasattr(torch, "_int_mm")}
    if not result["available"]:
        result["supported"] = False
        return result
    gen = torch.Generator().manual_seed(0)
    weight = torch.randint(-QMAX, QMAX + 1, (n, k), generator=gen, dtype=torch.int8)
    errors = {}
    for rows in (1, 8, 16, 17, 24, 32):
        x = torch.randint(-QMAX, QMAX + 1, (rows, k), generator=gen, dtype=torch.int8)
        try:
            out = torch._int_mm(x.to(device), weight.to(device).t())
        except RuntimeError as error:
            errors[str(rows)] = str(error).splitlines()[0][:200]
            continue
        exact = torch.equal(out.cpu().to(torch.int64), x.to(torch.int64) @ weight.to(torch.int64).t())
        result.update(min_rows=rows, exact=exact, output_dtype=str(out.dtype), output_device=str(out.device))
        break
    if errors:
        result["errors_by_rows"] = errors
    result["supported"] = bool(result.get("exact")) and result.get("output_device", "").startswith(device.type)
    return result


class Int8Layer:
    """One Linear layer quantized to int8 (or its FP32 reference)."""

    def __init__(self, name, weight, scale, bias, out_features, in_features, mode):
        self.name, self.weight, self.scale, self.bias = name, weight, scale, bias
        self.out_features, self.in_features, self.mode = out_features, in_features, mode

    def to(self, device):
        scale = None if self.scale is None else self.scale.to(device)
        return Int8Layer(self.name, self.weight.to(device), scale, self.bias.to(device),
                         self.out_features, self.in_features, self.mode)

    def tensors(self):
        return [t for t in (self.weight, self.scale, self.bias) if t is not None]

    def storage(self):
        parts = {"values": _nbytes(self.weight)}
        if self.scale is not None:
            parts["scales"] = _nbytes(self.scale)
        stored_shape = list(self.weight.shape)
        return {"format": self.mode, "pruned": False, "shape": [self.out_features, self.in_features],
                "stored_shape": stored_shape,
                "padding_values": self.weight.numel() - self.out_features * self.in_features,
                "stored_values": self.weight.numel(), "weight_bytes": sum(parts.values()),
                "weight_bytes_breakdown": parts, "value_dtype": str(self.weight.dtype),
                "scale_dtype": None if self.scale is None else str(self.scale.dtype),
                "quantization": None if self.scale is None else
                "symmetric per-output-row int8, scale = max|w_row| / 127",
                "index_dtype": None, "mask_dtype": None, "adapter_rank": None, "adapter_bytes": 0,
                "bias_bytes": _nbytes(self.bias)}


class Int8MLP:
    """Inference-only MLP over Int8Layers. Call it under torch.inference_mode()."""

    def __init__(self, layers, storage_format, min_rows=1, support=None):
        self.layers, self.storage_format = layers, storage_format
        self.min_rows, self.support = min_rows, support

    def to(self, device):
        device = torch.device(device)
        layers = [layer.to(device) for layer in self.layers]
        min_rows, support = 1, None
        if self.storage_format == "int8_w8a8":
            support = int_mm_support(device)
            if not support["supported"]:
                raise RuntimeError(f"torch._int_mm is not supported on {device}: {support}")
            min_rows = support["min_rows"]
        return Int8MLP(layers, self.storage_format, min_rows, support)

    def tensors(self):
        return [t for layer in self.layers for t in layer.tensors()]

    def resident_bytes(self):
        return sum(_nbytes(t) for t in self.tensors())

    def check_resident(self):
        """The deployed formats hold int8 weight matrices only (scales and biases are vectors)."""
        if self.storage_format not in INT8_FORMATS:
            return True
        return all(layer.weight.dtype == torch.int8 for layer in self.layers)

    def storage(self):
        layers = {layer.name: layer.storage() for layer in self.layers}
        weight_bytes = sum(layer["weight_bytes"] for layer in layers.values())
        return {"format": self.storage_format, "weight_bytes": weight_bytes, "adapter_bytes": 0,
                "stored_bytes": weight_bytes, "bias_bytes": sum(layer["bias_bytes"] for layer in layers.values()),
                "padding_values": sum(layer["padding_values"] for layer in layers.values()),
                "min_rows": self.min_rows, "layers": layers}

    def __call__(self, x):
        last = len(self.layers) - 1
        if self.storage_format == "int8_weight_only":
            h = x
            for i, layer in enumerate(self.layers):
                out = F.linear(h, layer.weight.to(torch.float32) * layer.scale[:, None], layer.bias)
                h = torch.relu(out) if i < last else out
            return h
        if self.storage_format == "int8_weight_only_reference":
            h = x
            for i, layer in enumerate(self.layers):
                out = F.linear(h, layer.weight, layer.bias)
                h = torch.relu(out) if i < last else out
            return h

        # W8A8 and its reference: padded widths throughout, sliced at the end.
        rows = x.shape[0]
        h = x
        first = self.layers[0]
        if first.weight.shape[1] > h.shape[1]:
            h = F.pad(h, (0, first.weight.shape[1] - h.shape[1]))
        if self.storage_format == "int8_w8a8" and rows < self.min_rows:
            h = F.pad(h, (0, 0, 0, self.min_rows - rows))
        for i, layer in enumerate(self.layers):
            q, scale = quantize_rows(h)
            if self.storage_format == "int8_w8a8":
                acc = torch._int_mm(q, layer.weight.t()).to(torch.float32)
            else:
                acc = (q.to(torch.float64) @ layer.weight.to(torch.float64).t()).to(torch.float32)
            out = acc * scale * layer.scale + layer.bias
            h = torch.relu(out) if i < last else out
        return h[:rows, :self.layers[-1].out_features]


def build_int8_model(model, storage_format):
    """Int8MLP on CPU for the original `model` (an src/mlp/model.MLP)."""
    if storage_format not in ALL_INT8_FORMATS:
        raise ValueError(f"unknown INT8 format {storage_format!r} (available: {ALL_INT8_FORMATS})")
    padded = storage_format in ("int8_w8a8", "int8_w8a8_reference")
    layers, previous_out = [], None
    for name, linear in zip(model.layer_names, model.layers()):
        weight, bias = linear.weight.detach().cpu(), linear.bias.detach().cpu()
        out_features, in_features = weight.shape
        q, scale = quantize_weight(weight)
        if storage_format == "int8_weight_only_reference":
            layers.append(Int8Layer(name, q.to(torch.float32) * scale[:, None], None, bias.clone(),
                                    out_features, in_features, storage_format))
            continue
        if padded:
            out_p = _round_up(out_features, PAD_MULTIPLE)
            in_p = previous_out if previous_out is not None else _round_up(in_features, PAD_MULTIPLE)
            q_p = torch.zeros(out_p, in_p, dtype=torch.int8)
            q_p[:out_features, :in_features] = q
            scale_p = torch.ones(out_p, dtype=torch.float32)
            scale_p[:out_features] = scale
            bias_p = torch.zeros(out_p, dtype=bias.dtype)
            bias_p[:out_features] = bias
            q, scale, bias = q_p, scale_p, bias_p
            previous_out = out_p
        layers.append(Int8Layer(name, q.contiguous(), scale, bias.clone(), out_features, in_features,
                                storage_format))
    return Int8MLP(layers, storage_format)
