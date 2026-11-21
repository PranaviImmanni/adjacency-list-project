"""Follow-up experiments: statistical pruning, matched baselines, adapters, INT8, recovery, variants."""

import numpy as np
import pytest
import torch

from src.mlp.config import load_followup_config
from src.mlp.data import split_and_normalize
from src.mlp.formats import build_format_model
from src.mlp.model import MLP
from src.mlp.quantization import build_int8_model, int_mm_support, quantize_weight
from src.mlp.recovery import LoRAMLP, lora_recover, masked_finetune
from src.mlp.training import train_mlp
from src.mlp.variants import (REFERENCE_FORMAT, build_variant, check_resident, csr_weight_bytes, load_variant_file,
                              make_variant, retained_for_csr_bytes, save_variant, variant_plan, variant_summary)
from src.mlp.weight_pruning import apply_mask, prune_layers, prune_statistical, prune_to_count


def _weights(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return {"fc1": torch.randn(30, 20, generator=gen) * 0.1 + 0.02, "fc2": torch.randn(10, 30, generator=gen) * 0.3}


# --- pruning rules -----------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 0.5, 1, 2])
def test_statistical_rule_per_matrix(k):
    weights = _weights()
    result = prune_statistical(weights, k)
    for name, w in weights.items():
        a = w.double().numpy()
        expected = np.abs(a - a.mean()) > k * a.std()  # numpy std is population (ddof=0)
        assert np.array_equal(result["masks"][name].numpy(), expected)
        layer = result["summary"]["layers"][name]
        assert layer["mean"] == pytest.approx(a.mean()) and layer["std"] == pytest.approx(a.std())
        assert layer["retained"] == int(expected.sum())
    summary = result["summary"]
    assert summary["k"] == k and summary["retained_weights"] == sum(int(m.sum()) for m in result["masks"].values())
    assert weights["fc1"].equal(_weights()["fc1"])  # inputs untouched


def test_statistical_keeps_values_and_signs():
    weights = _weights()
    mask = prune_statistical(weights, 1)["masks"]["fc1"]
    pruned = apply_mask(weights["fc1"], mask)
    assert torch.equal(pruned[mask], weights["fc1"][mask]) and torch.all(pruned[~mask] == 0)


def test_prune_to_count_matches_sparsity_version():
    weights = _weights()
    by_sparsity = prune_layers(weights, "magnitude", 0.9)
    retained = by_sparsity["summary"]["retained_weights"]
    by_count = prune_to_count(weights, "magnitude", retained)
    assert by_count["summary"]["retained_weights"] == retained
    for name in weights:
        assert torch.equal(by_count["masks"][name], by_sparsity["masks"][name])
    with pytest.raises(ValueError):
        prune_to_count(weights, "topk_per_neuron", retained)


# --- adapters and INT8 ---------------------------------------------------------------------------

@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return MLP(784, [300, 100], 10).eval()


def test_adapters_dense_and_csr_agree(mlp):
    masks = prune_layers({n: getattr(mlp, n).weight.detach() for n in mlp.layer_names}, "magnitude", 0.9)["masks"]
    gen = torch.Generator().manual_seed(1)
    adapters = {n: (torch.randn(4, l.weight.shape[1], generator=gen) * 0.05,
                    torch.randn(l.weight.shape[0], 4, generator=gen) * 0.05)
                for n, l in zip(mlp.layer_names, mlp.layers())}
    x = torch.randn(32, 784, generator=gen)
    dense = build_format_model(mlp, masks, "dense", adapters)
    csr = build_format_model(mlp, masks, "csr", adapters)
    with torch.inference_mode():
        assert torch.allclose(dense(x), csr(x), atol=1e-5, rtol=1e-4)
        zero = {n: (a, torch.zeros_like(b)) for n, (a, b) in adapters.items()}
        assert torch.equal(build_format_model(mlp, masks, "dense", zero)(x), build_format_model(mlp, masks, "dense")(x))
    storage = csr.storage()
    assert storage["adapter_bytes"] == 4 * 4 * (784 + 300 + 300 + 100 + 100 + 10)
    assert storage["stored_bytes"] == storage["weight_bytes"] + storage["adapter_bytes"]
    assert check_resident(csr.to("cpu"))


def test_int8_weight_quantization(mlp):
    w = mlp.fc1.weight.detach()
    q, scale = quantize_weight(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127 and scale.dtype == torch.float32
    assert torch.all((q.float() * scale[:, None] - w).abs() <= scale[:, None] / 2 + 1e-7)


def test_int8_paths_match_their_references(mlp):
    x = torch.randn(32, 784, generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        for fmt, ref in (("int8_weight_only", "int8_weight_only_reference"), ("int8_w8a8", "int8_w8a8_reference")):
            model, reference = build_int8_model(mlp, fmt).to("cpu"), build_int8_model(mlp, ref).to("cpu")
            out, ref_out = model(x), reference(x)
            assert out.shape == (32, 10)
            assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-4)
            assert model.check_resident()
            for rows in (1, 3):  # small batches are padded internally and sliced back
                assert torch.allclose(model(x[:rows]), ref_out[:rows], atol=1e-5, rtol=1e-4)
        assert torch.equal(build_int8_model(mlp, "int8_weight_only")(x),
                           build_int8_model(mlp, "int8_weight_only_reference")(x))
        assert torch.allclose(build_int8_model(mlp, "int8_weight_only")(x), mlp(x), atol=0.05)


def test_int8_storage_includes_scales_and_padding(mlp):
    wo = build_int8_model(mlp, "int8_weight_only").storage()
    assert wo["weight_bytes"] == 266_200 + 4 * 410 and wo["padding_values"] == 0
    w8 = build_int8_model(mlp, "int8_w8a8").storage()
    stored = 304 * 784 + 104 * 304 + 16 * 104
    assert w8["padding_values"] == stored - 266_200
    assert w8["weight_bytes"] == stored + 4 * (304 + 104 + 16)


def test_int_mm_probe_on_cpu():
    support = int_mm_support("cpu")
    assert support["available"] and support["supported"] and support["exact"]


# --- recovery training -----------------------------------------------------------------------------

@pytest.fixture
def toy():
    gen = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 3, (160,), generator=gen)
    images = torch.randint(0, 256, (160, 2, 3), generator=gen, dtype=torch.uint8)
    images[:, 0, 0] = (labels * 100).to(torch.uint8)
    data = split_and_normalize(images, labels, images[:40], labels[:40], val_size=30, split_seed=0)
    config = load_followup_config()
    config.update(hidden_dims=[8, 4], epochs=2, train_batch_size=16, eval_batch_size=16,
                  recovery_epochs=2, recovery_learning_rates=[0.01, 0.001], recovery_batch_size=16)
    model, checkpoint = train_mlp(data, config, seed=42, log_every=0)
    return data, config, model, checkpoint


def test_masked_finetune_keeps_removed_weights_zero(toy):
    data, config, model, _ = toy
    masks = prune_layers({n: getattr(model, n).weight.detach() for n in model.layer_names}, "magnitude", 0.5)["masks"]
    before = model.fc1.weight.detach().clone()
    tuned, record = masked_finetune(model, masks, data, config, seed=42)
    for name, mask in masks.items():
        w = getattr(tuned, name).weight.detach()
        assert torch.all(w[~mask] == 0)
    assert record["selected"]["epoch"] >= 1 and len(record["runs"]) == 2
    assert csr_weight_bytes(tuned, masks) == csr_weight_bytes(model, masks)
    assert not torch.equal(tuned.fc1.weight, model.fc1.weight)  # it did train
    assert torch.equal(model.fc1.weight.detach(), before)  # the input model is not modified


def test_lora_starts_from_pruned_model_and_trains_adapters_only(toy):
    data, config, model, _ = toy
    masks = prune_layers({n: getattr(model, n).weight.detach() for n in model.layer_names}, "magnitude", 0.5)["masks"]
    module = LoRAMLP(model, masks, config["lora_rank"], config["lora_alpha"], config["lora_layers"])
    x = data.x_test
    with torch.inference_mode():
        assert torch.equal(module(x), build_format_model(model, masks, "dense")(x))
    adapters, record = lora_recover(model, masks, data, config, seed=42)
    assert set(adapters) == set(config["lora_layers"])
    a, b = adapters["fc1"]
    assert a.shape == (4, 6) and b.shape == (8, 4)
    assert record["adapter_parameters"] == sum(a.numel() + b.numel() for a, b in adapters.values())
    with torch.inference_mode():
        dense = build_format_model(model, masks, "dense", adapters)(x)
        csr = build_format_model(model, masks, "csr", adapters)(x)
    assert torch.allclose(dense, csr, atol=1e-5, rtol=1e-4)


# --- protected output layer: byte matching -----------------------------------------------------------

@pytest.mark.parametrize("sparsity", [0.8, 0.9])
def test_protected_fc3_matches_bytes(mlp, sparsity):
    all_masks = prune_layers({n: getattr(mlp, n).weight.detach() for n in mlp.layer_names}, "magnitude", sparsity)
    target = csr_weight_bytes(mlp, all_masks["masks"])
    retained = retained_for_csr_bytes(mlp, ["fc1", "fc2"], target)
    masks = prune_to_count({n: getattr(mlp, n).weight.detach() for n in ("fc1", "fc2")}, "magnitude", retained)["masks"]
    achieved = csr_weight_bytes(mlp, masks)
    assert abs(achieved - target) < 12  # within one stored value + index
    model = build_format_model(mlp, masks, "csr")
    assert model.layers[2].weight.layout == torch.strided and not model.layers[2].pruned


# --- variants: plan, make, save, build ----------------------------------------------------------------

def test_variant_plan_tags():
    config = load_followup_config()
    tags = [s["tag"] for s in variant_plan("statistical", config, 42)]
    assert tags[:2] == ["m42_statistical_k0p5", "m42_magmatch_k0p5"] and len(tags) == 8
    assert [s["tag"] for s in variant_plan("protect_fc3", config, 43)] == [
        "m43_fc3dense_s80", "m43_fc3dense_s80_finetune", "m43_fc3dense_s90", "m43_fc3dense_s90_finetune"]
    assert variant_plan("int8", config, 44)[0]["deploy_formats"] == ["int8_weight_only", "int8_w8a8"]


@pytest.mark.parametrize("experiment", ["statistical", "finetune", "protect_fc3", "lora", "int8"])
def test_make_save_build_every_experiment(toy, tmp_path, experiment):
    data, config, model, checkpoint = toy
    identity = {"sha256": "test", "model_seed": 42}
    x = data.x_test
    for spec in variant_plan(experiment, config, 42)[:2]:
        variant = make_variant(spec, model, checkpoint, identity, data, config)
        path = tmp_path / f"{spec['tag']}.pt"
        save_variant(variant, path)
        loaded = load_variant_file(path)
        summary = variant_summary(loaded)
        assert summary["tag"] == spec["tag"] and "state_dict" not in summary
        if spec["kind"] == "magnitude_matched":
            assert loaded["matching"]["target_retained"] == loaded["matching"]["achieved_retained"]
        for fmt in loaded["deploy_formats"]:
            deployed, reference = build_variant(loaded, fmt).to("cpu"), build_variant(loaded, REFERENCE_FORMAT[fmt])
            assert check_resident(deployed)
            with torch.inference_mode():
                assert torch.allclose(deployed(x), reference(x), atol=1e-5, rtol=1e-4)
