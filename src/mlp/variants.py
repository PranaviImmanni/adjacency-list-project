"""Model variants for the follow-up experiments: what each is, how it is made, how it is stored.

A variant is one modified model of one model seed: the trained weights
(fine-tuned for some), optional keep-masks for the pruned layers, and optional
LoRA adapters. It is made once (scripts/make_mlp_variants.py), saved as
checkpoints/variants/<run>/<tag>.pt, then evaluated and benchmarked in its
deployment format(s). Every deployment format is checked against a dense
implementation of the same modified model.

| Experiment  | Variant (seed 42)          | What                                                 | Deployed as      |
|-------------|----------------------------|------------------------------------------------------|------------------|
| statistical | m42_statistical_k1p5       | per matrix, keep |w - mean| > k * std                 | CSR              |
|             | m42_magmatch_k1p5          | global magnitude with the same total retained count  | CSR              |
| finetune    | m42_finetune_s90           | magnitude 90% + masked fine-tuning                   | CSR              |
| protect_fc3 | m42_fc3dense_s90           | fc3 dense; fc1+fc2 global magnitude, retained count  | CSR + dense fc3  |
|             |                            | chosen so the stored bytes match finetune_s90's      |                  |
|             | m42_fc3dense_s90_finetune  | the same + masked fine-tuning                        | CSR + dense fc3  |
| lora        | m42_lora_s90               | magnitude 90% frozen + rank-4 adapters (separate)    | CSR + adapters   |
| int8        | m42_original               | the unpruned model                                   | int8_weight_only,|
|             |                            |                                                      | int8_w8a8        |

Reference (dense implementation of the same modified model) per deployment
format: csr -> dense (pruned dense, adapters separate); int8_weight_only ->
int8_weight_only_reference; int8_w8a8 -> int8_w8a8_reference.

The pruning-only magnitude 80% / 90% models and the original dense model are
the original sweep's results (results/mlp/<run>), reused as baselines.
"""

import hashlib
from datetime import datetime, timezone

import torch

from .formats import build_format_model
from .model import MLP
from .quantization import ALL_INT8_FORMATS, INT8_FORMATS, REFERENCE_FORMATS, build_int8_model
from .recovery import lora_recover, masked_finetune
from .weight_pruning import prune_layers, prune_statistical, prune_to_count

FORMAT_VERSION = 1
REFERENCE_FORMAT = {"csr": "dense", **REFERENCE_FORMATS}
DEPLOY_FORMATS = {
    "statistical": ["csr"], "magnitude_matched": ["csr"], "finetune": ["csr"], "protected": ["csr"],
    "lora": ["csr"], "original": list(INT8_FORMATS),
}


def k_label(k):
    return "k" + f"{k:g}".replace(".", "p")


def s_label(sparsity):
    return f"s{round(100 * sparsity)}"


def variant_plan(experiment, config, seed):
    """The variants of one experiment for one model seed, in run order."""
    m = f"m{seed}"
    plan = []
    if experiment == "statistical":
        for k in config["statistical_k"]:
            plan.append({"tag": f"{m}_statistical_{k_label(k)}", "kind": "statistical", "params": {"k": k}})
            plan.append({"tag": f"{m}_magmatch_{k_label(k)}", "kind": "magnitude_matched", "params": {"k": k}})
    elif experiment == "finetune":
        plan = [{"tag": f"{m}_finetune_{s_label(s)}", "kind": "finetune", "params": {"sparsity": s}}
                for s in config["recovery_sparsities"]]
    elif experiment == "protect_fc3":
        for s in config["recovery_sparsities"]:
            for tuned in (False, True):
                plan.append({"tag": f"{m}_fc3dense_{s_label(s)}" + ("_finetune" if tuned else ""),
                             "kind": "protected", "params": {"matched_to_sparsity": s, "finetune": tuned}})
    elif experiment == "lora":
        plan = [{"tag": f"{m}_lora_{s_label(s)}", "kind": "lora", "params": {"sparsity": s}}
                for s in config["recovery_sparsities"]]
    elif experiment == "int8":
        plan = [{"tag": f"{m}_original", "kind": "original", "params": {}}]
    else:
        raise ValueError(f"unknown experiment {experiment!r}")
    for spec in plan:
        spec.update(experiment=experiment, deploy_formats=DEPLOY_FORMATS[spec["kind"]])
    return plan


def _weights(model, names):
    return {name: getattr(model, name).weight.detach() for name in names}


def csr_weight_bytes(model, masks):
    return build_format_model(model, masks, "csr").storage()["weight_bytes"]


def retained_for_csr_bytes(model, pruned_names, target_bytes):
    """Retained-weight count in `pruned_names` (CSR) that makes the model's stored weight bytes ~ target.

    CSR bytes per pruned layer: (value + column index) per retained weight +
    a row pointer per (rows + 1). Layers not pruned are stored dense.
    """
    value_bytes = torch.empty(0, dtype=torch.float32).element_size()
    index_bytes = torch.empty(0, dtype=torch.int64).element_size()
    fixed = 0
    for name, linear in zip(model.layer_names, model.layers()):
        rows = linear.weight.shape[0]
        fixed += index_bytes * (rows + 1) if name in pruned_names else linear.weight.numel() * value_bytes
    return round((target_bytes - fixed) / (value_bytes + index_bytes))


def make_variant(spec, model, checkpoint, identity, data, config, device="cpu"):
    """Build one variant on CPU (recovery training runs on `device`). Returns the variant dict."""
    kind, params, seed = spec["kind"], spec["params"], checkpoint["seed"]
    pruned_layers = list(config["pruned_layers"])
    masks, adapters, pruning, matching, recovery = {}, {}, None, None, None
    tuned = model

    if kind == "statistical":
        result = prune_statistical(_weights(model, pruned_layers), params["k"])
    elif kind == "magnitude_matched":
        statistical = prune_statistical(_weights(model, pruned_layers), params["k"])["summary"]
        result = prune_to_count(_weights(model, pruned_layers), "magnitude", statistical["retained_weights"])
        matching = {"matched_to": f"m{seed}_statistical_{k_label(params['k'])}",
                    "target_retained": statistical["retained_weights"],
                    "achieved_retained": result["summary"]["retained_weights"]}
    elif kind in ("finetune", "lora"):
        result = prune_layers(_weights(model, pruned_layers), "magnitude", params["sparsity"])
    elif kind == "protected":
        protected = list(config["protected_layers"])
        names = [name for name in pruned_layers if name not in protected]
        reference = prune_layers(_weights(model, pruned_layers), "magnitude", params["matched_to_sparsity"])
        target = csr_weight_bytes(model, reference["masks"])
        retained = retained_for_csr_bytes(model, names, target)
        result = prune_to_count(_weights(model, names), "magnitude", retained)
        achieved = csr_weight_bytes(model, result["masks"])
        matching = {"matched_to": f"m{seed}_finetune_{s_label(params['matched_to_sparsity'])}",
                    "protected_layers": protected, "target_weight_bytes": target, "achieved_weight_bytes": achieved,
                    "mismatch_bytes": achieved - target, "mismatch_fraction": (achieved - target) / target,
                    "retained_in_pruned_layers": retained}
    elif kind == "original":
        result = None
    else:
        raise ValueError(f"unknown variant kind {kind!r}")

    if result is not None:
        masks, pruning = result["masks"], result["summary"]
    if kind == "finetune" or (kind == "protected" and params["finetune"]):
        tuned, recovery = masked_finetune(model, masks, data, config, seed, device)
    elif kind == "lora":
        adapter_tensors, recovery = lora_recover(model, masks, data, config, seed, device)
        adapters = {name: {"a": a, "b": b} for name, (a, b) in adapter_tensors.items()}

    return {
        "format_version": FORMAT_VERSION, "tag": spec["tag"], "experiment": spec["experiment"], "kind": kind,
        "params": params, "model_seed": seed, "deploy_formats": spec["deploy_formats"],
        "architecture": {"input_dim": checkpoint["input_dim"], "hidden_dims": list(checkpoint["hidden_dims"]),
                         "output_dim": checkpoint["output_dim"]},
        "state_dict": {k: v.detach().cpu().clone() for k, v in tuned.state_dict().items()},
        "masks": masks, "adapters": adapters, "pruning": pruning, "matching": matching, "recovery": recovery,
        "base_checkpoint": identity, "data": checkpoint["data"],
        "created": datetime.now(timezone.utc).isoformat(), "train_device": str(device),
        "torch_version": str(torch.__version__),
    }


def save_variant(variant, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(variant, path)


def load_variant_file(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def variant_mlp(variant):
    arch = variant["architecture"]
    model = MLP(arch["input_dim"], arch["hidden_dims"], arch["output_dim"])
    model.load_state_dict(variant["state_dict"])
    return model.eval()


def build_variant(variant, storage_format):
    """The variant in `storage_format`, on CPU (FormatMLP or Int8MLP)."""
    model = variant_mlp(variant)
    if storage_format in ALL_INT8_FORMATS:
        if variant["masks"] or variant["adapters"]:
            raise ValueError("INT8 formats apply to the unpruned model without adapters")
        return build_int8_model(model, storage_format)
    adapters = {name: (ab["a"], ab["b"]) for name, ab in variant["adapters"].items()}
    return build_format_model(model, variant["masks"], storage_format, adapters)


def check_resident(format_model):
    """Deployed sparse layers hold no dense weight or mask; INT8 layers hold int8 weights only."""
    if hasattr(format_model, "check_resident"):
        return format_model.check_resident()
    if format_model.storage_format != "csr":
        return True
    return all(not (layer.pruned and (layer.weight.layout == torch.strided or layer.mask is not None))
               for layer in format_model.layers)


def _sha(tensors):
    digest = hashlib.sha256()
    for t in tensors:
        digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def variant_summary(variant):
    """Everything about a variant except its tensors (for the results JSON)."""
    adapters = variant["adapters"]
    weights = sum(t.numel() for k, t in variant["state_dict"].items() if k.endswith(".weight"))
    removed = variant["pruning"]["removed_weights"] if variant["pruning"] else 0
    return {
        **{k: variant[k] for k in ("format_version", "tag", "experiment", "kind", "params", "model_seed",
                                   "deploy_formats", "architecture", "pruning", "matching", "recovery",
                                   "base_checkpoint", "created", "train_device", "torch_version")},
        "model_weights": weights, "model_retained_weights": weights - removed,
        "model_sparsity": removed / weights,
        "weights_sha256": _sha(variant["state_dict"][k] for k in sorted(variant["state_dict"])),
        "pruned_layers": sorted(variant["masks"]),
        "adapter_layers": sorted(adapters),
        "adapter_parameters": sum(ab["a"].numel() + ab["b"].numel() for ab in adapters.values()),
    }
