"""Inference and prediction-equivalence metrics across adjacency formats.

COO is the numerical reference: it is the canonical representation the other
formats are derived from, and it is the format the checkpoint was trained with.
"""

import torch

from src.measurement.memory_utils import output_closeness, prediction_match_rate

# Tolerance for "same logits" in FP32.
ATOL = 1e-5
RTOL = 1e-4


def accuracy(logits, y, mask):
    return (logits[mask].argmax(dim=1) == y[mask]).float().mean().item()


def run_inference(model, x, adj):
    model.eval()
    with torch.inference_mode():
        return model(x, adj)


def compare_to_reference(ref_logits, logits, y, test_mask, atol=ATOL, rtol=RTOL):
    """Equivalence of `logits` against the reference (COO) logits.

    All inputs are moved to CPU first. Agreement is reported on all nodes
    and on the test nodes. Cosine similarity is deliberately not used:
    max/mean absolute difference and allclose are stricter.
    """
    ref_logits, logits, y, test_mask = (t.detach().cpu() for t in (ref_logits, logits, y, test_mask))
    ref_pred, pred = ref_logits.argmax(dim=1), logits.argmax(dim=1)
    abs_diff = (logits.double() - ref_logits.double()).abs()

    ref_acc = accuracy(ref_logits, y, test_mask)
    test_acc = accuracy(logits, y, test_mask)
    return {
        "test_accuracy": test_acc,
        "reference_test_accuracy": ref_acc,
        "accuracy_difference_pp": 100.0 * (test_acc - ref_acc),
        "prediction_agreement_all": prediction_match_rate(pred.numpy(), ref_pred.numpy()),
        "prediction_agreement_test": prediction_match_rate(pred[test_mask].numpy(), ref_pred[test_mask].numpy()),
        "mean_abs_logit_diff": abs_diff.mean().item(),
        "max_abs_logit_diff": output_closeness(logits.numpy(), ref_logits.numpy(), metric="max_abs_diff"),
        "allclose": bool(torch.allclose(logits, ref_logits, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def is_equivalent(metrics):
    """The correctness target: identical labels, identical accuracy, logits within tolerance."""
    return (metrics["prediction_agreement_all"] == 1.0
            and metrics["accuracy_difference_pp"] == 0.0
            and metrics["allclose"])
