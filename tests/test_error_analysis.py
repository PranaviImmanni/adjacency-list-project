import pytest
import torch

from src.gcn.error_analysis import margin_analysis, node_epsilon, prediction_margins, pruning_effect
from src.gcn.evaluation import compare_to_reference


def test_prediction_margins_and_ties():
    logits = torch.tensor([[3.0, 1.0, 0.5], [2.0, 2.0, 0.0], [-1.0, 0.0, -4.0]])
    assert prediction_margins(logits).tolist() == [2.0, 0.0, 1.0]
    assert prediction_margins(logits).dtype == torch.float64


def test_node_epsilon_is_per_node_max_abs_change():
    ref = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    other = torch.tensor([[1.5, 1.0], [0.0, -0.25]])
    assert node_epsilon(ref, other).tolist() == [1.0, 0.25]


def test_margin_condition_counts():
    ref = torch.tensor([[3.0, 0.0],    # margin 3, eps 1   -> stable, same class
                        [1.0, 0.0],    # margin 1, eps 0.5 -> not stable (1 > 1 is false), same class
                        [1.0, 0.0],    # margin 1, eps 1   -> not stable, flips
                        [0.5, 0.0]])   # margin 0.5, eps 0 -> stable
    other = torch.tensor([[2.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 0.0]])
    test_mask = torch.tensor([False, True, True, False])
    out = margin_analysis(ref, other, {"all": None, "test": test_mask})
    assert out["all"]["nodes"] == 4
    assert out["all"]["margin_condition_fraction"] == 0.5
    assert out["all"]["disagreements"] == 1
    assert out["all"]["disagreement_rate"] == 0.25
    assert out["all"]["disagreements_satisfying_condition"] == 0
    assert out["test"]["nodes"] == 2
    assert out["test"]["margin_condition_fraction"] == 0.0
    assert out["test"]["disagreement_rate"] == 0.5


def test_margin_condition_is_sufficient_on_random_perturbations():
    gen = torch.Generator().manual_seed(0)
    ref = torch.randn(5000, 5, generator=gen)
    other = ref + 0.3 * torch.randn(5000, 5, generator=gen)
    out = margin_analysis(ref, other, {"all": None})["all"]
    assert out["disagreements"] > 0  # the perturbation does flip some predictions
    assert 0 < out["margin_condition_fraction"] < 1
    assert out["disagreements_satisfying_condition"] == 0


def test_accuracy_differences_are_percentage_points_with_sign():
    y = torch.tensor([0, 1, 0, 1])
    test_mask = torch.ones(4, dtype=torch.bool)
    val_mask = torch.tensor([True, True, False, False])
    original = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])  # 3/4 correct
    worse = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])     # 1/4 correct
    better = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])    # 4/4 correct

    loss = pruning_effect(original, worse, y, val_mask, test_mask)
    assert loss["original_test_accuracy"] == 0.75 and loss["test_accuracy"] == 0.25
    assert loss["test_accuracy_loss_pp"] == pytest.approx(50.0)  # pp, not a fraction
    assert loss["prediction_disagreement_test"] == 0.5
    gain = pruning_effect(original, better, y, val_mask, test_mask)
    assert gain["test_accuracy_loss_pp"] == pytest.approx(-25.0)  # negative loss is kept
    assert gain["val_accuracy_loss_pp"] == pytest.approx(0.0)

    # Format comparison: signed (format minus reference), also in pp.
    m = compare_to_reference(original, worse, y, test_mask)
    assert m["accuracy_difference_pp"] == pytest.approx(-50.0)


def test_accuracy_difference_is_bounded_by_disagreement():
    gen = torch.Generator().manual_seed(1)
    y = torch.randint(0, 4, (2000,), generator=gen)
    mask = torch.rand(2000, generator=gen) < 0.5
    ref = torch.randn(2000, 4, generator=gen)
    other = ref + 0.5 * torch.randn(2000, 4, generator=gen)
    out = pruning_effect(ref, other, y, mask, mask)
    assert abs(out["test_accuracy_loss_pp"]) <= 100 * out["prediction_disagreement_test"] + 1e-9
    assert out["epsilon_logit"] >= out["mean_abs_logit_diff"]
