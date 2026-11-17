"""Data parsing, training, checkpoint and harness tests on small synthetic data (no download)."""

import gzip
import struct

import pytest
import torch

from src.mlp.config import load_config
from src.mlp.data import parse_idx, read_idx_gz, split_and_normalize
from src.mlp.experiment import ORIGINAL, condition_tag, conditions
from src.mlp.harness import check_resident, load_variant, make_batch, time_inference
from src.mlp.model import load_mlp_checkpoint
from src.mlp.training import save_checkpoint, train_mlp


def idx_bytes(array):
    header = struct.pack(">HBB", 0, 0x08, array.dim()) + struct.pack(f">{array.dim()}I", *array.shape)
    return header + bytes(array.flatten().tolist())


def test_parse_idx_round_trip(tmp_path):
    images = torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8)
    assert torch.equal(parse_idx(idx_bytes(images)), images)
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(idx_bytes(images)))
    assert torch.equal(read_idx_gz(path), images)
    with pytest.raises(ValueError):
        parse_idx(idx_bytes(images)[:-1])  # truncated body
    with pytest.raises(ValueError):
        parse_idx(b"\x00\x00\x0d\x01" + bytes(8))  # float32 IDX is not supported


@pytest.fixture
def toy_data():
    gen = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 3, (120,), generator=gen)
    images = torch.randint(0, 256, (120, 2, 3), generator=gen, dtype=torch.uint8)
    images[:, 0, 0] = (labels * 100).to(torch.uint8)  # make the task learnable
    return split_and_normalize(images, labels, images[:30], labels[:30], val_size=20, split_seed=0)


def test_split_is_fixed_and_normalized_on_train_only(toy_data):
    again = split_and_normalize(torch.zeros(120, 2, 3, dtype=torch.uint8), torch.zeros(120, dtype=torch.long),
                                torch.zeros(1, 2, 3, dtype=torch.uint8), torch.zeros(1, dtype=torch.long),
                                val_size=20, split_seed=0)
    assert again.val_indices_sha256 == toy_data.val_indices_sha256  # depends only on the split seed
    assert len(toy_data.y_train) == 100 and len(toy_data.y_val) == 20 and toy_data.num_features == 6
    assert toy_data.x_train.dtype == torch.float32
    assert abs(toy_data.x_train.double().mean().item()) < 1e-6
    x, y, val_mask, test_mask = toy_data.eval_set()
    assert len(x) == 50 and int(val_mask.sum()) == 20 and int(test_mask.sum()) == 30


@pytest.fixture
def trained(tmp_path, toy_data):
    config = load_config()
    config.update(hidden_dims=[8, 4], epochs=3, train_batch_size=16, eval_batch_size=16)
    model, checkpoint = train_mlp(toy_data, config, seed=42, log_every=0)
    path = tmp_path / "toy_mlp.pt"
    save_checkpoint(checkpoint, path)
    return model, checkpoint, path, toy_data


def test_training_checkpoint_round_trip(trained):
    model, checkpoint, path, data = trained
    assert 1 <= checkpoint["best_epoch"] <= 3 and len(checkpoint["history"]) == 3
    loaded, saved = load_mlp_checkpoint(path)  # weights_only=True must accept every field
    assert saved["seed"] == 42 and saved["data"]["split_seed"] == 0
    with torch.inference_mode():
        assert torch.equal(loaded(data.x_test), model(data.x_test))


def test_training_is_reproducible(toy_data):
    config = load_config()
    config.update(hidden_dims=[8, 4], epochs=2, train_batch_size=16, eval_batch_size=16)
    a = train_mlp(toy_data, config, seed=7, log_every=0)[1]["model_state_dict"]
    b = train_mlp(toy_data, config, seed=7, log_every=0)[1]["model_state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("storage_format", ["dense", "dense_mask", "coo", "csr"])
def test_harness_variant_metadata_and_residency(trained, storage_format):
    _, _, path, data = trained
    variant = load_variant(path, "random", 0.5, 101, storage_format, "cpu")
    meta = variant.metadata
    assert meta["model_seed"] == 42 and meta["pruning_seed"] == 101 and meta["method"] == "random"
    assert meta["format"] == storage_format and meta["actual_sparsity"] == pytest.approx(0.5, abs=0.02)
    assert check_resident(variant)
    x = make_batch(data.x_test, 8, "cpu")
    logits, times = time_inference(variant.model, x, torch.device("cpu"), 1, 3)
    assert logits.shape == (8, data.num_classes) and len(times) == 3


def test_harness_original_is_dense_only(trained):
    _, _, path, data = trained
    variant = load_variant(path, "none", 0.0, None, "dense", "cpu")
    assert variant.metadata["condition"] == "original" and variant.metadata["actual_sparsity"] == 0.0
    with pytest.raises(ValueError):
        load_variant(path, "none", 0.0, None, "csr", "cpu")
    with pytest.raises(ValueError):
        make_batch(data.x_test, 10_000, "cpu")


def test_condition_grid():
    config = load_config()
    evaluate = conditions(config)
    assert evaluate[0] == ORIGINAL
    assert len(evaluate) == 1 + len(config["sparsities"]) * (2 + len(config["random_pruning_seeds"]))
    bench = conditions(config, purpose="benchmark")
    assert len(bench) == 1 + len(config["sparsities"]) * (2 + len(config["benchmark_random_seeds"]))
    assert condition_tag(43, "random", 0.95, 101) == "m43_random_p101_s95"
    assert condition_tag(42, "none", 0.0, None) == "m42_none_s0"
