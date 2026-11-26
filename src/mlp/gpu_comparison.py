"""The GPU format comparison: what is run, how each model is built, and the checks against Pranavi's CPU results.

Two parts, each on the three trained checkpoints (configs/mlp_gpu_comparison.json):
  pruned    the original sweep's conditions (magnitude, top-K per neuron and
            random at 50-99%; the original model) in dense, dense+mask, COO,
            CSR and adjacency list. Reference: pruned dense.
  low_rank  Pranavi's low-rank factor networks (src/mlp/low_rank.py) at every
            rank and factor rule, as factor adjacency lists and dense factors.
            Reference: dense factors.

Accuracy and equivalence use every random pruning seed; timing and energy use
`benchmark_random_seeds`, as in the original sweep. Tags reuse the original
sweep's (m42_magnitude_s90), so results can be matched with results/mlp.
"""

import csv

import torch

from src.gcn.config import PROJECT_ROOT

from .experiment import condition_tag, conditions, format_comparison
from .formats import ADJACENCY_FORMAT, SPARSE_FORMATS, build_format_model
from .low_rank import build_low_rank_model, low_rank_factors, rule_label
from .weight_pruning import prune_layers

PARTS = ("pruned", "low_rank")
REFERENCE_FORMAT = {"original": None, "pruned": "dense", "low_rank": "factor_dense"}
PRANAVI_ADJACENCY_CSV = "reports/adjacency_list_mlp_benchmark.csv"
PRANAVI_LOW_RANK_CSVS = ("reports/low_rank_adjacency.csv", "reports/low_rank_adjacency_k1.csv")


def low_rank_tag(model_seed, rank, rule, k=None):
    return f"m{model_seed}_lowrank_r{rank}_{rule_label(rule, k)}"


def plan(config, model_seed, purpose="benchmark", parts=PARTS):
    """Every model of one seed: [{tag, kind, formats, reference_format, ...}], original model first."""
    specs = []
    if "pruned" in parts:
        for method, sparsity, pruning_seed in conditions(config, purpose):
            kind = "original" if method == "none" else "pruned"
            specs.append({"tag": condition_tag(model_seed, method, sparsity, pruning_seed), "kind": kind,
                          "model_seed": model_seed, "method": method, "sparsity": sparsity,
                          "pruning_seed": pruning_seed,
                          "formats": ["dense"] if kind == "original" else list(config["pruned_formats"]),
                          "reference_format": REFERENCE_FORMAT[kind]})
    if "low_rank" in parts:
        for rank in config["low_rank_ranks"]:
            for rule in config["low_rank_rules"]:
                k = rule.get("k")
                specs.append({"tag": low_rank_tag(model_seed, rank, rule["rule"], k), "kind": "low_rank",
                              "model_seed": model_seed, "rank": rank, "rule": rule["rule"],
                              "k": None if k is None else float(k), "rule_label": rule_label(rule["rule"], k),
                              "formats": list(config["low_rank_formats"]),
                              "reference_format": REFERENCE_FORMAT["low_rank"]})
    return specs


def pilot_plan(config, model_seed):
    """Original model, magnitude 90% (every pruned format) and rank 32 with the first two factor rules."""
    specs = plan(config, model_seed, "benchmark")
    rank = 32 if 32 in config["low_rank_ranks"] else config["low_rank_ranks"][0]
    keep_low_rank = {low_rank_tag(model_seed, rank, r["rule"], r.get("k")) for r in config["low_rank_rules"][:2]}
    keep = {condition_tag(model_seed, "none", 0.0, None), condition_tag(model_seed, "magnitude", 0.9, None)}
    return [s for s in specs if s["tag"] in keep | keep_low_rank]


def find_spec(config, model_seed, tag):
    for purpose in ("benchmark", "evaluate"):
        for spec in plan(config, model_seed, purpose):
            if spec["tag"] == tag:
                return spec
    raise ValueError(f"unknown tag {tag!r} for seed {model_seed}")


def prepare(spec, model, config):
    """Pruning masks or low-rank factors of `spec` (computed once, shared by every format)."""
    if spec["kind"] == "original":
        return {"masks": {}, "summary": {"method": "none", "pruning_seed": None, "requested_sparsity": 0.0,
                                         "actual_sparsity": 0.0, "layers": {}}}
    if spec["kind"] == "pruned":
        weights = {name: getattr(model, name).weight.detach() for name in config["pruned_layers"]}
        pruned = prune_layers(weights, spec["method"], spec["sparsity"], spec["pruning_seed"])
        return {"masks": pruned["masks"], "summary": pruned["summary"]}
    factors = low_rank_factors(model, spec["rank"], spec["rule"], spec["k"])
    return {"factors": factors["layers"], "summary": factors["summary"]}


def build(spec, prepared, model, storage_format):
    """The CPU inference model of `spec` in `storage_format`."""
    if storage_format not in spec["formats"] and storage_format != spec["reference_format"]:
        raise ValueError(f"{spec['tag']}: format {storage_format!r} is not part of this model's plan")
    if spec["kind"] == "low_rank":
        return build_low_rank_model(prepared["factors"], storage_format)
    return build_format_model(model, prepared["masks"], storage_format)


def resident_ok(format_model, storage_format):
    """A sparse or list format holds no dense weight matrix or mask of a pruned layer on the device."""
    if storage_format == "factor_adjacency":
        return format_model.holds_only_lists()
    if storage_format not in SPARSE_FORMATS + (ADJACENCY_FORMAT,):
        return True
    return all(not (layer.pruned and (layer.weight.layout == torch.strided or layer.mask is not None))
               for layer in format_model.layers)


def spec_fields(spec):
    """The identifying fields every result records."""
    fields = {"tag": spec["tag"], "kind": spec["kind"], "model_seed": spec["model_seed"]}
    if spec["kind"] == "low_rank":
        fields.update(rank=spec["rank"], factor_rule=spec["rule"], k=spec["k"], rule_label=spec["rule_label"],
                      method="low_rank", requested_sparsity=None, pruning_seed=None)
    else:
        fields.update(method=spec["method"], requested_sparsity=spec["sparsity"], pruning_seed=spec["pruning_seed"],
                      rank=None, factor_rule=None, k=None, rule_label=None)
    return fields


# ---------------------------------------------------------------------------------------------
# Pranavi's CPU results (NumPy, float64 compute on FP32 values, test set only)
# ---------------------------------------------------------------------------------------------

def _read_csv(relative, root):
    path = root / relative
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pranavi_reference(spec, root=PROJECT_ROOT):
    """Her matching CSV row (accuracy and structure), or None when she did not run this model."""
    seed = str(spec["model_seed"])
    if spec["kind"] == "pruned":
        for row in _read_csv(PRANAVI_ADJACENCY_CSV, root):
            if (row["model_seed"] == seed and row["method"] == spec["method"] and row["format"] == "adjacency_list_cpu"
                    and float(row["sparsity"]) == spec["sparsity"]
                    and row["pruning_seed"] == ("" if spec["pruning_seed"] is None else str(spec["pruning_seed"]))):
                return {"source": PRANAVI_ADJACENCY_CSV, "test_accuracy": float(row["accuracy"]),
                        "edges": int(row["nnz"]),
                        "prediction_match_to_pruned_dense": float(row["prediction_match_rate"])}
        return None
    if spec["kind"] != "low_rank":
        return None
    for relative in PRANAVI_LOW_RANK_CSVS:
        for row in _read_csv(relative, root):
            if row["model_seed"] != seed or int(row["rank"]) != spec["rank"] or row["factor_edge_rule"] != spec["rule"]:
                continue
            if spec["rule"] == "abs_std" and float(row["threshold_r"]) != spec["k"]:
                continue
            return {"source": relative, "test_accuracy": float(row["accuracy"]), "edges": int(row["factor_edges"]),
                    "stored_bytes": int(row["factor_adjacency_bytes"]),
                    "mean_weight_reconstruction_mse": float(row["mean_weight_reconstruction_mse"]),
                    "prediction_match_to_original": float(row["prediction_match_to_dense"])}
    return None


def cross_check(reference, test_accuracy, edges, test_images, stored_bytes=None, reconstruction_mse=None):
    """Our GPU result vs her CPU row: accuracy difference in points and in images, structure equality."""
    if reference is None:
        return {"found": False}
    difference = test_accuracy - reference["test_accuracy"]
    out = {"found": True, "source": reference["source"], "pranavi_test_accuracy": reference["test_accuracy"],
           "test_accuracy": test_accuracy, "difference_pp": 100 * difference,
           "difference_images": round(difference * test_images),
           "edges_equal": edges == reference["edges"]}
    if "stored_bytes" in reference and stored_bytes is not None:
        out["stored_bytes_equal"] = stored_bytes == reference["stored_bytes"]
    if "mean_weight_reconstruction_mse" in reference and reconstruction_mse is not None:
        ref = reference["mean_weight_reconstruction_mse"]
        out["reconstruction_mse_relative_difference"] = abs(reconstruction_mse - ref) / ref if ref else None
    return out


# ---------------------------------------------------------------------------------------------
# Equivalence: the project's tolerance, and a check against an exact float64 computation
# ---------------------------------------------------------------------------------------------

ROUNDING_SLACK = 2.0  # a format may be up to 2x as far from exact float64 as its FP32 reference


def exact_forward(spec, prepared, model):
    """Float64 forward on CPU of exactly the same modified model (the same stored FP32 values).

    Its own rounding error is about 1e-15 relative, so it measures how far each FP32
    implementation (the reference included) is from the exact result.
    """
    if spec["kind"] == "low_rank":
        layers = [((f["A"].double(), f["B"].double()), f["bias"].double()) for f in prepared["factors"].values()]
    else:
        masks = prepared["masks"]
        layers = []
        for name, linear in zip(model.layer_names, model.layers()):
            weight = linear.weight.detach().cpu()
            if name in masks:
                weight = torch.where(masks[name], weight, torch.zeros((), dtype=weight.dtype))
            layers.append(((weight.double(),), linear.bias.detach().cpu().double()))

    def forward(x):
        h = x.detach().cpu().double()
        for i, (matrices, bias) in enumerate(layers):
            for matrix in matrices:  # (W,) or (A, B): h A^T B^T
                h = h @ matrix.t()
            h = h + bias
            if i < len(layers) - 1:
                h = torch.relu(h)
        return h

    return forward


def equivalence(reference_logits, logits, exact_logits, y, mask):
    """The project's format check, plus the same predictions with FP32-rounding-level error.

    strict_equivalent: identical predictions and accuracy, and allclose(atol=1e-5,
    rtol=1e-4) against the FP32 reference (src/gcn/evaluation.is_equivalent).
    within_fp32_rounding: identical predictions and accuracy, and the format is at most
    ROUNDING_SLACK times as far from the exact float64 result as the FP32 reference is.
    Large logits (the low-rank models reach about 100) make FP32 rounding alone exceed
    the absolute tolerance, for the reference as much as for the format.
    equivalent: either of the two; equivalence_basis says which.
    """
    check = format_comparison(reference_logits, logits, y, mask)
    error = (logits.detach().cpu().double() - exact_logits).abs().max().item()
    reference_error = (reference_logits.detach().cpu().double() - exact_logits).abs().max().item()
    same_predictions = check["prediction_agreement_all"] == 1.0 and check["abs_accuracy_difference_pp"] == 0.0
    rounding = same_predictions and error <= ROUNDING_SLACK * reference_error + 1e-6
    strict = check["equivalent"]
    check.update(strict_equivalent=strict, within_fp32_rounding=rounding, equivalent=strict or rounding,
                 equivalence_basis="tolerance" if strict else ("fp32_rounding" if rounding else None),
                 max_abs_error_vs_float64=error, reference_max_abs_error_vs_float64=reference_error,
                 max_abs_exact_logit=exact_logits.abs().max().item())
    return check
