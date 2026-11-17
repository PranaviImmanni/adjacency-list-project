"""Contract tests for the signed-weight pruning interface (src/mlp/weight_pruning.py).

Any implementation of prune_mask (for example Pranavi's) must pass these.
"""

import pytest
import torch

from src.mlp.weight_pruning import METHODS, apply_mask, prune_layers, prune_mask, removal_count

SPARSITIES = (0.5, 0.8, 0.9, 0.95, 0.99)


@pytest.fixture
def weight():
    return torch.randn(30, 40, generator=torch.Generator().manual_seed(0))


def test_removal_count_is_exact_floor():
    assert removal_count(100, 0.95) == 95  # 0.95 * 100 is 94.999... in binary floating point
    assert removal_count(784, 0.99) == 776
    assert removal_count(10, 0.0) == 0
    with pytest.raises(ValueError):
        removal_count(10, 1.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sparsity", SPARSITIES)
def test_mask_shape_dtype_count_and_input_untouched(weight, method, sparsity):
    before = weight.clone()
    mask = prune_mask(weight, method, sparsity, seed=7)
    assert mask.dtype == torch.bool and mask.shape == weight.shape and mask.device.type == "cpu"
    assert torch.equal(weight, before)
    if method == "topk_per_neuron":
        k = 40 - removal_count(40, sparsity)
        assert torch.equal(mask.sum(dim=1), torch.full((30,), k))
    else:
        assert int((~mask).sum()) == removal_count(weight.numel(), sparsity)


def test_magnitude_keeps_largest_absolute_values_including_negative():
    w = torch.tensor([[0.1, -5.0, 0.2], [3.0, -0.05, 0.3]])
    mask = prune_mask(w, "magnitude", 0.5)  # remove 3 of 6
    assert mask.tolist() == [[False, True, False], [True, False, True]]


def test_magnitude_ties_removed_in_row_major_order():
    w = torch.tensor([[1.0, -1.0], [1.0, 2.0]])
    mask = prune_mask(w, "magnitude", 0.5)  # remove 2 of the three |w| = 1 entries: (0,0), (0,1)
    assert mask.tolist() == [[False, False], [True, True]]


def test_topk_per_neuron_keeps_largest_per_row():
    w = torch.tensor([[0.1, -0.9, 0.5, 0.0], [4.0, 3.0, -2.0, 1.0]])
    mask = prune_mask(w, "topk_per_neuron", 0.5)  # each row keeps 2
    assert mask.tolist() == [[False, True, True, False], [True, True, False, False]]


def test_random_depends_only_on_seed(weight):
    a = prune_mask(weight, "random", 0.9, seed=101)
    assert torch.equal(a, prune_mask(weight * 3, "random", 0.9, seed=101))  # values do not matter
    assert not torch.equal(a, prune_mask(weight, "random", 0.9, seed=102))
    with pytest.raises(ValueError):
        prune_mask(weight, "random", 0.9)


@pytest.mark.parametrize("method", METHODS)
def test_removed_sets_are_nested(weight, method):
    masks = [prune_mask(weight, method, s, seed=5) for s in SPARSITIES]
    for lower, higher in zip(masks, masks[1:]):
        assert not (higher & ~lower).any()  # anything kept at the higher sparsity was kept at the lower


def test_apply_mask_preserves_values_and_signs_exactly(weight):
    mask = prune_mask(weight, "magnitude", 0.8)
    pruned = apply_mask(weight, mask)
    assert torch.equal(pruned[mask], weight[mask])
    assert torch.equal(torch.sign(pruned[mask]), torch.sign(weight[mask]))
    assert (pruned[~mask] == 0).all() and not torch.signbit(pruned[~mask]).any()  # +0.0, never -0.0


@pytest.fixture
def two_layers():
    gen = torch.Generator().manual_seed(1)
    return {"fc1": torch.randn(20, 50, generator=gen) * 0.1, "fc2": torch.randn(5, 20, generator=gen)}


def test_global_magnitude_uses_one_threshold_across_layers(two_layers):
    out = prune_layers(two_layers, "magnitude", 0.9)
    summary = out["summary"]
    assert summary["scope"] == "global"
    assert summary["removed_weights"] == removal_count(1100, 0.9)
    kept = torch.cat([w[out["masks"][n]].abs() for n, w in two_layers.items()])
    removed = torch.cat([w[~out["masks"][n]].abs() for n, w in two_layers.items()])
    assert kept.min() >= removed.max()  # every kept |w| is at least every removed |w|
    # fc1's weights are 10x smaller, so fc1 is pruned harder than fc2.
    assert summary["layers"]["fc1"]["actual_sparsity"] > summary["layers"]["fc2"]["actual_sparsity"]


def test_global_magnitude_ties_follow_layer_then_row_major_order():
    weights = {"a": torch.tensor([[1.0, 2.0]]), "b": torch.tensor([[1.0, -1.0]])}
    masks = prune_layers(weights, "magnitude", 0.5)["masks"]  # remove 2 of the three |w| = 1 entries
    assert masks["a"].tolist() == [[False, True]] and masks["b"].tolist() == [[False, True]]


def test_global_random_count_and_seed(two_layers):
    out = prune_layers(two_layers, "random", 0.9, pruning_seed=101)
    assert out["summary"]["removed_weights"] == removal_count(1100, 0.9)
    same = prune_layers({n: w * 7 for n, w in two_layers.items()}, "random", 0.9, pruning_seed=101)
    assert all(torch.equal(out["masks"][n], same["masks"][n]) for n in two_layers)  # values do not matter
    other = prune_layers(two_layers, "random", 0.9, pruning_seed=102)
    assert not torch.equal(out["masks"]["fc1"], other["masks"]["fc1"])
    with pytest.raises(ValueError):
        prune_layers(two_layers, "random", 0.9)


def test_topk_is_per_row_in_every_layer(two_layers):
    out = prune_layers(two_layers, "topk_per_neuron", 0.9)
    assert out["summary"]["scope"] == "per output neuron"
    assert torch.equal(out["masks"]["fc1"].sum(dim=1), torch.full((20,), 5))
    assert torch.equal(out["masks"]["fc2"].sum(dim=1), torch.full((5,), 2))


@pytest.mark.parametrize("method", METHODS)
def test_global_removed_sets_are_nested(two_layers, method):
    outs = [prune_layers(two_layers, method, s, pruning_seed=5)["masks"] for s in SPARSITIES]
    for lower, higher in zip(outs, outs[1:]):
        assert not any((higher[n] & ~lower[n]).any() for n in two_layers)
