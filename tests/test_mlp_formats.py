import pytest
import torch

from src.mlp.experiment import format_comparison, logits_in_batches
from src.mlp.formats import FORMATS, build_format_model, sparse_from_mask
from src.mlp.model import MLP
from src.mlp.weight_pruning import apply_mask, prune_layers


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = MLP(20, [16, 8], 4).eval()
    x = torch.randn(37, 20, generator=torch.Generator().manual_seed(1))
    weights = {name: getattr(model, name).weight.detach() for name in model.layer_names}
    masks = prune_layers(weights, "magnitude", 0.8)["masks"]
    return model, x, masks


def test_original_dense_matches_the_module(setup):
    model, x, _ = setup
    with torch.inference_mode():
        assert torch.equal(build_format_model(model, {}, "dense")(x), model(x))


@pytest.mark.parametrize("storage_format", FORMATS)
def test_every_format_matches_pruned_dense(setup, storage_format):
    model, x, masks = setup
    reference = logits_in_batches(build_format_model(model, masks, "dense"), x, 16, "cpu")
    logits = logits_in_batches(build_format_model(model, masks, storage_format), x, 16, "cpu")
    assert logits.shape == (37, 4)
    check = format_comparison(reference, logits, torch.zeros(37, dtype=torch.long), torch.ones(37, dtype=torch.bool))
    assert check["equivalent"] and check["epsilon_logit"] < 1e-5
    assert check["margin"]["all"]["disagreements_satisfying_condition"] == 0


def test_pruned_dense_equals_manual_masked_forward(setup):
    model, x, masks = setup
    h = x
    for i, name in enumerate(model.layer_names):
        layer = getattr(model, name)
        h = h @ apply_mask(layer.weight, masks[name]).T + layer.bias
        h = torch.relu(h) if i < 2 else h
    with torch.inference_mode():
        assert torch.allclose(build_format_model(model, masks, "dense")(x), h.detach(), atol=1e-6)


def test_sparse_formats_store_exactly_the_retained_weights(setup):
    model, _, masks = setup
    for storage_format in ("coo", "csr"):
        fm = build_format_model(model, masks, storage_format)
        for layer in fm.layers:
            weight = getattr(model, layer.name).weight.detach()
            mask = masks[layer.name]
            dense = layer.weight.to_dense()
            assert torch.equal(dense[mask], weight[mask])  # same values and signs
            assert int(fm.storage()["layers"][layer.name]["stored_values"]) == int(mask.sum())
            assert layer.mask is None and layer.weight.layout != torch.strided  # no dense copy, no mask


def test_retained_exact_zero_is_still_stored():
    w = torch.tensor([[0.0, 2.0], [3.0, -4.0]])
    mask = torch.tensor([[True, False], [True, True]])
    coo = sparse_from_mask(w, mask)
    assert coo._nnz() == 3 and coo.is_coalesced()
    assert torch.equal(coo.to_dense(), torch.tensor([[0.0, 0.0], [3.0, -4.0]]))


def test_exact_storage_bytes():
    model = MLP(10, [6], 3)
    weights = {name: getattr(model, name).weight.detach() for name in model.layer_names}
    masks = prune_layers(weights, "topk_per_neuron", 0.5)["masks"]  # fc1 keeps 5 of 10 per row, fc2 3 of 6
    nnz = {"fc1": 6 * 5, "fc2": 3 * 3}
    expected = {
        "dense": 4 * (60 + 18),
        "dense_mask": 4 * (60 + 18) + 1 * (60 + 18),
        "coo": sum(4 * n + 2 * 8 * n for n in nnz.values()),
        "csr": sum(4 * n + 8 * n for n in nnz.values()) + 8 * ((6 + 1) + (3 + 1)),
    }
    for storage_format, weight_bytes in expected.items():
        storage = build_format_model(model, masks, storage_format).storage()
        assert storage["weight_bytes"] == weight_bytes, storage_format
        assert storage["bias_bytes"] == 4 * (6 + 3)


def test_dense_mask_stores_original_weights_and_bool_mask(setup):
    model, _, masks = setup
    fm = build_format_model(model, masks, "dense_mask")
    layer = fm.layers[0]
    assert torch.equal(layer.weight, model.fc1.weight.detach()) and layer.mask.dtype == torch.bool


def test_build_rejects_bad_requests(setup):
    model, _, masks = setup
    with pytest.raises(ValueError):
        build_format_model(model, {}, "csr")  # the original model is dense only
    with pytest.raises(ValueError):
        build_format_model(model, masks, "bsr")
    with pytest.raises(ValueError):
        build_format_model(model, {"fc9": masks["fc1"]}, "coo")


def test_layer_left_unpruned_stays_dense_in_sparse_format(setup):
    model, x, masks = setup
    partial = {"fc1": masks["fc1"]}
    fm = build_format_model(model, partial, "csr")
    assert fm.layers[1].weight.layout == torch.strided and not fm.layers[1].pruned
    reference = build_format_model(model, partial, "dense")
    with torch.inference_mode():
        assert torch.allclose(fm(x), reference(x), atol=1e-5)
