"""Format error, pruning effects and prediction-margin stability (float64).

Two kinds of comparison, kept apart (docs/pruning_and_error_analysis.md):
  format error     same graph, different storage format. Should be ~1e-7 and
                   never change a prediction (evaluation.compare_to_reference).
  pruning effect   original graph vs pruned graph, same format (CSR) and
                   checkpoint. Changed predictions are an experimental
                   outcome, not a correctness failure.

Units: accuracies are fractions in [0, 1]; every field ending in `_pp` is a
difference in percentage points (100 x a difference of fractions). Logit
differences are raw logit units and are never mixed with accuracy.

Margin condition: if node i's reference margin (largest minus second-largest
logit) exceeds 2 * epsilon_i, where epsilon_i is the largest absolute change
of any of its logits, its predicted class cannot change. The condition is
sufficient, not necessary, so a node can fail it and still keep its prediction.
"""

import torch


def _float64_cpu(t):
    return torch.as_tensor(t).detach().to(device="cpu", dtype=torch.float64)


def exact_accuracy(logits, y, mask):
    """Correct / total as an exact Python fraction (no float32 rounding in pp differences)."""
    mask = torch.as_tensor(mask).cpu().bool()
    correct = torch.as_tensor(logits).detach().cpu().argmax(dim=1)[mask] == torch.as_tensor(y).cpu()[mask]
    return int(correct.sum()) / int(mask.sum())


def prediction_margins(logits):
    """Per node: largest logit minus second-largest (0 when the top two tie)."""
    top2 = _float64_cpu(logits).topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def node_epsilon(ref_logits, logits):
    """Per node: max over classes of |logit - reference logit|."""
    return (_float64_cpu(logits) - _float64_cpu(ref_logits)).abs().max(dim=1).values


def margin_analysis(ref_logits, logits, masks):
    """Margin-condition coverage and disagreements for each node subset.

    masks: {name: bool mask, or None for all nodes}. For each subset:
      nodes, margin_condition_fraction (margin > 2 * epsilon),
      disagreement_rate, disagreements, and
      disagreements_satisfying_condition (always 0 if the theory holds).
    """
    margin = prediction_margins(ref_logits)
    epsilon = node_epsilon(ref_logits, logits)
    stable = margin > 2 * epsilon
    disagree = _float64_cpu(ref_logits).argmax(dim=1) != _float64_cpu(logits).argmax(dim=1)

    out = {}
    for name, mask in masks.items():
        mask = torch.ones_like(stable) if mask is None else torch.as_tensor(mask).cpu().bool()
        n = int(mask.sum())
        out[name] = {
            "nodes": n,
            "margin_condition_fraction": int((stable & mask).sum()) / n,
            "disagreement_rate": int((disagree & mask).sum()) / n,
            "disagreements": int((disagree & mask).sum()),
            "disagreements_satisfying_condition": int((disagree & stable & mask).sum()),
            "median_margin": margin[mask].median().item(),
            "max_node_epsilon": epsilon[mask].max().item(),
        }
    return out


def logit_change(ref_logits, logits):
    diff = (_float64_cpu(logits) - _float64_cpu(ref_logits)).abs()
    return {"mean_abs_logit_diff": diff.mean().item(), "epsilon_logit": diff.max().item()}


def format_error(ref_logits, logits, test_mask, comparison):
    """Extra format-error fields for one evaluation.compare_to_reference result.

    Adds epsilon_logit, the absolute test-accuracy difference, and the margin
    analysis on all nodes and test nodes.
    """
    return {
        "epsilon_logit": comparison["max_abs_logit_diff"],
        "abs_accuracy_difference_pp": abs(comparison["accuracy_difference_pp"]),
        "margin": margin_analysis(ref_logits, logits, {"all": None, "test": test_mask}),
    }


def pruning_effect(original_logits, pruned_logits, y, val_mask, test_mask):
    """Original graph vs pruned graph, same checkpoint and format.

    accuracy_loss_pp = original - pruned, in percentage points, signed: it is
    negative when pruning improves accuracy.
    """
    original, pruned = _float64_cpu(original_logits), _float64_cpu(pruned_logits)
    y, val_mask, test_mask = (torch.as_tensor(t).cpu() for t in (y, val_mask, test_mask))
    out = {}
    for split, mask in (("val", val_mask), ("test", test_mask)):
        before, after = exact_accuracy(original, y, mask), exact_accuracy(pruned, y, mask)
        out[f"original_{split}_accuracy"] = before
        out[f"{split}_accuracy"] = after
        out[f"{split}_accuracy_loss_pp"] = 100.0 * (before - after)
    margin = margin_analysis(original, pruned, {"all": None, "test": test_mask})
    out["prediction_disagreement_all"] = margin["all"]["disagreement_rate"]
    out["prediction_disagreement_test"] = margin["test"]["disagreement_rate"]
    out.update(logit_change(original, pruned))
    out["margin"] = margin
    return out
