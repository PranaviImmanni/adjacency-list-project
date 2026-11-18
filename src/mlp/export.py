"""Export trained MLPs and their pruning masks as one numpy bundle (.npz).

For work outside this pipeline (e.g. the Python adjacency-list inference):
reading the bundle needs only numpy, `np.load(path)` (no pickle, no torch).

Arrays (float32 unless noted):
  metadata                 JSON string: conventions, models, conditions, settings
  x_test                   (10000, 784) standardized test images, exactly as every run used them
  y_test                   (10000,) int64 labels
  m{S}_{layer}_weight      (out, in) trained weights of model seed S (layer = fc1, fc2, fc3)
  m{S}_{layer}_bias        (out,) biases (never pruned)
  m{S}_logits_test         (10000, 10) original dense model's logits for x_test (CPU)
  {tag}_{layer}_mask       (out, in) bool keep-mask of one pruning condition, True = keep
  {tag}_logits_batch       (256, 10) pruned dense logits for x_test[:256] (CPU)

`tag` is the condition name used in results/, e.g. m42_magnitude_s90,
m43_topk_per_neuron_s95, m44_random_p101_s80.

Forward pass (layers in order, ReLU after every layer except the last):
    h = relu(h @ W.T + b),  with W replaced by where(mask, W, 0) for a pruned condition.
The benchmark batches are x_test[:B] for B = 1, 32, 256.

Each mask's `mask_sha256` (sha256 of its uint8 bytes, first 16 hex digits)
is the value recorded in the evaluate and benchmark JSONs, so a bundle can be
matched to the committed results.
"""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.mlp.data import FILES, RAW_DIR, read_idx_gz
from src.mlp.experiment import condition_tag, conditions, logits_in_batches
from src.mlp.formats import build_format_model
from src.mlp.model import checkpoint_identity
from src.mlp.weight_pruning import prune_layers

FORMAT_VERSION = 1
REFERENCE_BATCH = 256


def mask_sha256(mask):
    """Same digest as src.mlp.weight_pruning.mask_sha256, from a numpy bool array."""
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()[:16]


def _accuracy(logits, y):
    return int((logits.argmax(1) == y).sum()) / len(y)


def check_data(data, checkpoint):
    """The bundle's test inputs must be the ones the checkpoint was evaluated with."""
    saved = checkpoint["data"]
    if (saved["split_seed"] != data.split_seed or saved["val_indices_sha256"] != data.val_indices_sha256
            or saved["normalization"]["mean"] != data.mean or saved["normalization"]["std"] != data.std):
        raise ValueError(f"seed {checkpoint['seed']}: data split or normalization differs from training")


def with_training_normalization(data, checkpoint, raw_dir=RAW_DIR):
    """Re-standardize the test images with the constants saved in the checkpoint.

    The training split's mean and std are float64 reductions over 43 million
    pixels, and they differ in the 11th digit between machines (Linux x86 vs
    macOS ARM). The standardization itself is elementwise and exact, so using
    the saved constants reproduces the test inputs of the machine that trained
    and evaluated the model bit for bit.
    """
    saved = checkpoint["data"]
    if saved["split_seed"] != data.split_seed or saved["val_indices_sha256"] != data.val_indices_sha256:
        raise ValueError("data split differs from the one the checkpoint was trained with")
    mean, std = saved["normalization"]["mean"], saved["normalization"]["std"]
    if abs(mean - data.mean) > 1e-9 or abs(std - data.std) > 1e-9:
        raise ValueError("normalization differs by more than float64 rounding: not the same training data")
    images = read_idx_gz(Path(raw_dir) / FILES["test_images"][0])
    x_test = ((images.reshape(len(images), -1).to(torch.float64) / 255.0 - mean) / std).to(torch.float32)
    return replace(data, x_test=x_test, mean=mean, std=std)


def build_bundle(models, data, config, source=None):
    """{name: array} for np.savez_compressed. `models` is a list of (path, model, checkpoint) on CPU."""
    x, y = data.x_test, data.y_test
    batch = min(REFERENCE_BATCH, len(y))
    pruned_layers = list(config["pruned_layers"])
    layer_names = list(models[0][1].layer_names)
    arrays = {"x_test": x.numpy().copy(), "y_test": y.numpy().copy()}
    meta_models, meta_conditions = {}, []

    for path, model, checkpoint in models:
        check_data(data, checkpoint)
        seed = checkpoint["seed"]
        for name in layer_names:
            layer = getattr(model, name)
            arrays[f"m{seed}_{name}_weight"] = layer.weight.detach().numpy().copy()
            arrays[f"m{seed}_{name}_bias"] = layer.bias.detach().numpy().copy()
        logits = logits_in_batches(build_format_model(model, {}, "dense"), x, config["eval_batch_size"], "cpu")
        arrays[f"m{seed}_logits_test"] = logits.numpy()
        identity = checkpoint_identity(path, checkpoint)
        meta_models[str(seed)] = {
            "checkpoint_file": Path(path).name, "checkpoint_sha256": identity["sha256"],
            "train_device": identity["train_device"], "best_epoch": checkpoint["best_epoch"],
            "saved_test_accuracy": checkpoint["test_accuracy"], "test_accuracy_cpu": _accuracy(logits, y),
        }

        weights = {name: getattr(model, name).weight.detach() for name in pruned_layers}
        for method, sparsity, pruning_seed in conditions(config)[1:]:
            tag = condition_tag(seed, method, sparsity, pruning_seed)
            pruned = prune_layers(weights, method, sparsity, pruning_seed)
            for name, mask in pruned["masks"].items():
                arrays[f"{tag}_{name}_mask"] = mask.numpy().copy()
            reference = build_format_model(model, pruned["masks"], "dense")
            logits = logits_in_batches(reference, x, config["eval_batch_size"], "cpu")
            arrays[f"{tag}_logits_batch"] = logits[:batch].numpy()
            summary = pruned["summary"]
            meta_conditions.append({
                "tag": tag, "model_seed": seed, "method": method, "requested_sparsity": sparsity,
                "pruning_seed": pruning_seed, "actual_sparsity": summary["actual_sparsity"],
                "benchmarked": method != "random" or pruning_seed in config["benchmark_random_seeds"],
                "test_accuracy_cpu": _accuracy(logits, y),
                "layers": {name: {"retained": info["retained"], "actual_sparsity": info["actual_sparsity"],
                                  "mask_sha256": info["mask_sha256"]}
                           for name, info in summary["layers"].items()},
            })

    first = models[0][2]
    metadata = {
        "format_version": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "source": source or {},
        "dataset": "fashion_mnist",
        "architecture": [first["input_dim"], *first["hidden_dims"], first["output_dim"]],
        "layers": layer_names,
        "pruned_layers": pruned_layers,
        "conventions": {
            "weight": "(out, in); layer output = h @ W.T + b",
            "activation": "ReLU after every layer except the last",
            "mask": "bool, True = keep; pruned weight = where(mask, W, 0); retained values and signs unchanged",
            "bias": "never pruned",
            "mask_sha256": "sha256 of mask.astype(uint8).tobytes(), first 16 hex digits; matches results/ JSONs",
        },
        "x_test": {"shape": list(arrays["x_test"].shape), "dtype": "float32",
                   "normalization": {"mean": data.mean, "std": data.std, "computed_on": "training split",
                                     "formula": "(pixel / 255 - mean) / std"},
                   "split_seed": data.split_seed, "benchmark_batches": "x_test[:B] for B in batch_sizes"},
        "reference_logits": {
            "device": "cpu", "torch_version": torch.__version__, "batch_rows": batch,
            "note": "GPU logits differ from these by float32 rounding only (max 1.5e-5 in the GPU runs)",
        },
        "settings": {key: config[key] for key in ("sparsities", "methods", "random_pruning_seeds",
                                                  "benchmark_random_seeds", "batch_sizes", "warmup_runs",
                                                  "measured_runs", "repetitions", "dtype", "tf32")},
        "models": meta_models,
        "conditions": meta_conditions,
    }
    arrays["metadata"] = np.array(json.dumps(metadata, indent=1))
    return arrays


def numpy_forward(bundle, seed, x, tag=None, layers=None):
    """Logits from the bundle alone, as an outside implementation would compute them."""
    layers = layers or json.loads(str(bundle["metadata"]))["layers"]
    h = x
    for i, name in enumerate(layers):
        weight = bundle[f"m{seed}_{name}_weight"]
        mask_key = f"{tag}_{name}_mask"
        if tag is not None and mask_key in bundle:
            weight = np.where(bundle[mask_key], weight, np.float32(0))
        h = h @ weight.T + bundle[f"m{seed}_{name}_bias"]
        if i < len(layers) - 1:
            h = np.maximum(h, np.float32(0))
    return h


def verify_bundle(path, atol=1e-5, rtol=1e-4):
    """Reload with numpy only and recheck every model and condition. Returns a list of problems."""
    problems = []
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle["metadata"]))
        x, y = bundle["x_test"], bundle["y_test"]
        batch = meta["reference_logits"]["batch_rows"]
        for seed, info in meta["models"].items():
            logits = numpy_forward(bundle, seed, x, layers=meta["layers"])
            if not np.allclose(logits, bundle[f"m{seed}_logits_test"], atol=atol, rtol=rtol):
                problems.append(f"m{seed}: numpy logits differ from the stored original logits")
            if _accuracy(logits, y) != info["test_accuracy_cpu"]:
                problems.append(f"m{seed}: numpy test accuracy differs from the recorded one")
        for condition in meta["conditions"]:
            tag, seed = condition["tag"], condition["model_seed"]
            for name, layer in condition["layers"].items():
                mask = bundle[f"{tag}_{name}_mask"]
                if mask.dtype != np.bool_ or mask_sha256(mask) != layer["mask_sha256"] \
                        or int(mask.sum()) != layer["retained"]:
                    problems.append(f"{tag} {name}: mask does not match its recorded hash or count")
            logits = numpy_forward(bundle, seed, x[:batch], tag, meta["layers"])
            if not np.allclose(logits, bundle[f"{tag}_logits_batch"], atol=atol, rtol=rtol):
                problems.append(f"{tag}: numpy logits differ from the stored pruned logits")
    return problems


def match_results(metadata, results_dir):
    """Compare the bundle with committed result JSONs: checkpoint hashes and mask hashes must agree.

    Returns (checked, problems, accuracy_differences).
    """
    results_dir = Path(results_dir)
    evaluations = {}
    for file in (results_dir / "evaluate").glob("*.json"):
        record = json.loads(file.read_text())
        if record.get("status") == "ok":
            evaluations[file.stem] = record
    checked, problems, accuracy_diffs = 0, [], []
    for seed, info in metadata["models"].items():
        original = evaluations.get(f"m{seed}_none_s0")
        if original is None:
            problems.append(f"m{seed}: no evaluate result in {results_dir}")
        elif original["checkpoint"]["sha256"] != info["checkpoint_sha256"]:
            problems.append(f"m{seed}: checkpoint differs from the one behind the results")
    for condition in metadata["conditions"]:
        record = evaluations.get(condition["tag"])
        if record is None:
            problems.append(f"{condition['tag']}: no evaluate result")
            continue
        checked += 1
        for name, layer in condition["layers"].items():
            if record["pruning"]["layers"][name]["mask_sha256"] != layer["mask_sha256"]:
                problems.append(f"{condition['tag']} {name}: mask differs from the one behind the results")
        if record["accuracy"]["test"] != condition["test_accuracy_cpu"]:
            accuracy_diffs.append((condition["tag"], record["accuracy"]["test"], condition["test_accuracy_cpu"]))
    return checked, problems, accuracy_diffs
