"""The experiment grid: pruning conditions, naming, and evaluation helpers.

A condition is (method, sparsity, pruning seed). "none" at sparsity 0 is the
original dense model. Random pruning uses every seed in
`random_pruning_seeds` for accuracy and epsilon, and only
`benchmark_random_seeds` for timing (latency and memory depend on the number
of retained weights, not on which ones).

Two comparisons, kept apart as in the GCN pruning study:
  pruning effect  pruned model vs the original dense model (same device):
                  accuracy loss, disagreement, logit change, margins
  format error    each format vs the pruned dense model (same weights,
                  same device): epsilon, allclose, agreement, margins
"""

import torch

from src.gcn.error_analysis import exact_accuracy, format_error, pruning_effect
from src.gcn.evaluation import compare_to_reference, is_equivalent

ORIGINAL = ("none", 0.0, None)


def conditions(config, purpose="evaluate", seeds_for_random=None):
    """All pruning conditions, original model first."""
    if seeds_for_random is None:
        key = "random_pruning_seeds" if purpose == "evaluate" else "benchmark_random_seeds"
        seeds_for_random = config[key]
    out = [ORIGINAL]
    for method in config["methods"]:
        for sparsity in config["sparsities"]:
            if method == "random":
                out += [(method, sparsity, seed) for seed in seeds_for_random]
            else:
                out.append((method, sparsity, None))
    return out


def condition_tag(model_seed, method, sparsity, pruning_seed):
    """e.g. m42_none_s0, m42_magnitude_s90, m43_random_p101_s95, m44_topk_per_neuron_s99."""
    return (f"m{model_seed}_{method}" + (f"_p{pruning_seed}" if pruning_seed is not None else "")
            + f"_s{round(100 * sparsity)}")


def logits_in_batches(format_model, x, batch_size, device):
    """Logits on CPU, computed on `device` in fixed-size chunks under inference mode."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            out.append(format_model(x[i:i + batch_size].to(device)).cpu())
    return torch.cat(out)


def accuracies(logits, y, val_mask, test_mask):
    return {"val": exact_accuracy(logits, y, val_mask), "test": exact_accuracy(logits, y, test_mask)}


def pruning_comparison(original_logits, pruned_logits, y, val_mask, test_mask):
    """Pruned vs original. accuracy_loss_pp = original - pruned (signed)."""
    return pruning_effect(original_logits, pruned_logits, y, val_mask, test_mask)


def format_comparison(reference_logits, logits, y, test_mask):
    """A storage format vs the pruned dense model: the GCN format checks, unchanged.

    bitwise_equal records whether the logits are identical to the last bit
    (dense+mask is, so agreement against dense+mask equals agreement against
    pruned dense).
    """
    result = compare_to_reference(reference_logits, logits, y, test_mask)
    result.update(format_error(reference_logits, logits, test_mask, result))
    result["equivalent"] = is_equivalent(result)
    result["bitwise_equal"] = bool(torch.equal(reference_logits.cpu(), logits.cpu()))
    return result
