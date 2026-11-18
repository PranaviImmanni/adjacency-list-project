"""The numpy bundle: contents, numpy-only reload, and matching against result JSONs."""

import gzip
import json
import struct

import numpy as np
import pytest
import torch

from src.mlp.config import load_config
from src.mlp.data import FILES, split_and_normalize
from src.mlp.experiment import conditions
from src.mlp.export import (build_bundle, mask_sha256, match_results, numpy_forward, verify_bundle,
                            with_training_normalization)
from src.mlp.training import save_checkpoint, train_mlp
from src.mlp.weight_pruning import mask_sha256 as torch_mask_sha256


@pytest.fixture
def bundle_path(tmp_path):
    gen = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 3, (120,), generator=gen)
    images = torch.randint(0, 256, (120, 2, 3), generator=gen, dtype=torch.uint8)
    images[:, 0, 0] = (labels * 100).to(torch.uint8)
    data = split_and_normalize(images, labels, images[:30], labels[:30], val_size=20, split_seed=0)
    config = load_config()
    config.update(hidden_dims=[8, 4], epochs=2, train_batch_size=16, eval_batch_size=16)
    models = []
    for seed in (42, 43):
        model, checkpoint = train_mlp(data, config, seed=seed, log_every=0)
        path = tmp_path / f"toy_seed{seed}.pt"
        save_checkpoint(checkpoint, path)
        models.append((path, model, checkpoint))
    arrays = build_bundle(models, data, config)
    out = tmp_path / "bundle.npz"
    np.savez_compressed(out, **arrays)
    return out, config


def test_bundle_contents(bundle_path):
    path, config = bundle_path
    with np.load(path, allow_pickle=False) as b:
        meta = json.loads(str(b["metadata"]))
        assert b["x_test"].shape == (30, 6) and b["x_test"].dtype == np.float32 and b["y_test"].dtype == np.int64
        classes = meta["architecture"][-1]
        assert meta["architecture"][:3] == [6, 8, 4] and meta["layers"] == ["fc1", "fc2", "fc3"]
        assert b["m42_fc1_weight"].shape == (8, 6) and b["m42_fc3_bias"].shape == (classes,)
        n_pruned = len(conditions(config)) - 1
        assert len(meta["conditions"]) == 2 * n_pruned
        tag = "m42_magnitude_s90"
        assert b[f"{tag}_fc1_mask"].dtype == np.bool_
        assert b[f"{tag}_logits_batch"].shape == (30, classes)  # reference batch clipped to the test set
        recorded = next(c for c in meta["conditions"] if c["tag"] == tag)["layers"]["fc1"]["mask_sha256"]
        assert mask_sha256(b[f"{tag}_fc1_mask"]) == recorded
        assert torch_mask_sha256(torch.from_numpy(b[f"{tag}_fc1_mask"])) == recorded
        # pruned weights keep their exact values and signs
        pruned = np.where(b[f"{tag}_fc1_mask"], b["m42_fc1_weight"], 0)
        kept = b[f"{tag}_fc1_mask"]
        assert np.array_equal(pruned[kept], b["m42_fc1_weight"][kept])
        assert np.allclose(numpy_forward(b, 42, b["x_test"]), b["m42_logits_test"], atol=1e-5, rtol=1e-4)
        assert [c["benchmarked"] for c in meta["conditions"] if c["method"] == "random"].count(True) == 2 * 5


def test_verify_bundle_passes_and_catches_a_changed_mask(bundle_path, tmp_path):
    path, _ = bundle_path
    assert verify_bundle(path) == []
    with np.load(path, allow_pickle=False) as b:
        arrays = {k: b[k] for k in b.files}
    arrays["m42_magnitude_s90_fc1_mask"] = ~arrays["m42_magnitude_s90_fc1_mask"]
    broken = tmp_path / "broken.npz"
    np.savez_compressed(broken, **arrays)
    assert any("m42_magnitude_s90 fc1" in p for p in verify_bundle(broken))


def test_training_normalization_reproduces_inputs(tmp_path):
    gen = torch.Generator().manual_seed(1)
    labels = torch.randint(0, 3, (120,), generator=gen)
    images = torch.randint(0, 256, (120, 2, 3), generator=gen, dtype=torch.uint8)
    test_images = images[:30]
    data = split_and_normalize(images, labels, test_images, labels[:30], val_size=20, split_seed=0)
    header = struct.pack(">HBB", 0, 0x08, 3) + struct.pack(">3I", *test_images.shape)
    (tmp_path / FILES["test_images"][0]).write_bytes(gzip.compress(header + bytes(test_images.flatten().tolist())))
    saved = {"data": {"split_seed": 0, "val_indices_sha256": data.val_indices_sha256,
                      "normalization": {"mean": data.mean, "std": data.std}}}
    same = with_training_normalization(data, saved, raw_dir=tmp_path)
    assert torch.equal(same.x_test, data.x_test)  # identical constants -> bit-identical inputs
    saved["data"]["normalization"]["std"] = data.std + 2e-11  # another machine's float64 reduction
    other = with_training_normalization(data, saved, raw_dir=tmp_path)
    assert other.std == data.std + 2e-11 and torch.allclose(other.x_test, data.x_test, rtol=1e-9, atol=0)
    saved["data"]["normalization"]["std"] = data.std + 1e-3
    with pytest.raises(ValueError):
        with_training_normalization(data, saved, raw_dir=tmp_path)


def test_match_results(bundle_path, tmp_path):
    path, _ = bundle_path
    with np.load(path, allow_pickle=False) as b:
        meta = json.loads(str(b["metadata"]))
    evaluate = tmp_path / "results" / "evaluate"
    evaluate.mkdir(parents=True)
    for seed, info in meta["models"].items():
        (evaluate / f"m{seed}_none_s0.json").write_text(json.dumps(
            {"status": "ok", "checkpoint": {"sha256": info["checkpoint_sha256"]}}))
    for c in meta["conditions"]:
        layers = {name: {"mask_sha256": layer["mask_sha256"]} for name, layer in c["layers"].items()}
        (evaluate / f"{c['tag']}.json").write_text(json.dumps(
            {"status": "ok", "pruning": {"layers": layers}, "accuracy": {"test": c["test_accuracy_cpu"]}}))
    checked, problems, diffs = match_results(meta, tmp_path / "results")
    assert checked == len(meta["conditions"]) and problems == [] and diffs == []

    first = meta["conditions"][0]
    record = json.loads((evaluate / f"{first['tag']}.json").read_text())
    record["pruning"]["layers"]["fc1"]["mask_sha256"] = "0" * 16
    (evaluate / f"{first['tag']}.json").write_text(json.dumps(record))
    assert len(match_results(meta, tmp_path / "results")[1]) == 1
