"""The follow-up report on synthetic GPU-style results (peak memory, baseline timing, every section)."""

import csv
import json

import pytest

from src.mlp.config import load_followup_config
from src.mlp.followup_reporting import aggregate, collect, summarize_followup


def _check(eps=1e-6):
    return {"epsilon_logit": eps, "equivalent": True, "bitwise_equal": eps == 0, "prediction_agreement_all": 1.0}


def _variant(seed, experiment, kind, row, params, sparsity):
    layers = {n: {"actual_sparsity": sparsity} for n in ("fc1", "fc2", "fc3")}
    recovery = {"selected": {"learning_rate": 0.001, "epoch": 3}} if kind in ("finetune", "lora", "protected") else None
    matching = {"target_weight_bytes": 642_184, "mismatch_bytes": 0} if kind == "protected" else None
    return {"tag": f"m{seed}_{row}", "experiment": experiment, "kind": kind, "params": params, "model_seed": seed,
            "model_sparsity": sparsity, "pruning": {"layers": layers} if sparsity else None, "recovery": recovery,
            "matching": matching}


def _evaluation(seed, experiment, kind, row, params, sparsity, accs):
    formats = {}
    for fmt, acc in accs.items():
        formats[fmt] = {"accuracy": {"val": acc, "test": acc},
                        "storage": {"stored_bytes": 1000, "weight_bytes": 900, "adapter_bytes": 100},
                        "effect_vs_original": {"test_accuracy_loss_pp": 100 * (0.89 - acc)},
                        "check": _check(0.0 if fmt == "int8_weight_only" else 1e-6),
                        "int_mm_support": {"device": "cuda:1", "supported": True, "min_rows": 17}
                        if fmt == "int8_w8a8" else None}
    return {"status": "ok", "variant": _variant(seed, experiment, kind, row, params, sparsity), "formats": formats}


def _bench(seed, experiment, row, fmt, batch, rep, latency):
    return {"status": "ok", "tag": f"m{seed}_{row}", "format": fmt, "followup_experiment": experiment,
            "model_seed": seed, "batch_size": batch, "repetition": rep, "latency_ms": {"median": latency},
            "latency_is_debug_only": False, "cuda_allocated_before_inference_bytes": 300_000 + batch,
            "cuda_peak_allocated_bytes": 400_000 + batch, "format_check": {"equivalent": True, "epsilon_logit": 1e-6},
            "int_mm_support": {"device": "cuda:1", "supported": True, "min_rows": 17} if fmt == "int8_w8a8" else None}


@pytest.fixture
def dirs(tmp_path):
    follow, base = tmp_path / "followup", tmp_path / "base"
    for d in (follow / "evaluate", follow / "benchmark", base / "evaluate", base / "benchmark"):
        d.mkdir(parents=True)
    records, benches, base_evals, base_benches = [], [], [], []
    for seed in (42, 43):
        records += [
            _evaluation(seed, "statistical", "statistical", "statistical_k1", {"k": 1}, 0.73, {"csr": 0.75}),
            _evaluation(seed, "statistical", "magnitude_matched", "magmatch_k1", {"k": 1}, 0.73, {"csr": 0.80}),
            _evaluation(seed, "finetune", "finetune", "finetune_s80", {"sparsity": 0.8}, 0.8, {"csr": 0.895}),
            _evaluation(seed, "protect_fc3", "protected", "fc3dense_s80", {"matched_to_sparsity": 0.8,
                                                                           "finetune": False}, 0.797, {"csr": 0.76}),
            _evaluation(seed, "lora", "lora", "lora_s80", {"sparsity": 0.8}, 0.8, {"csr": 0.893}),
            _evaluation(seed, "int8", "original", "original", {}, 0.0,
                        {"int8_weight_only": 0.889, "int8_w8a8": 0.888}),
        ]
        for experiment, row, fmt in (("statistical", "statistical_k1", "csr"), ("statistical", "magmatch_k1", "csr"),
                                     ("finetune", "finetune_s80", "csr"), ("protect_fc3", "fc3dense_s80", "csr"),
                                     ("lora", "lora_s80", "csr"), ("int8", "original", "int8_weight_only"),
                                     ("int8", "original", "int8_w8a8")):
            for batch in (1, 32, 256):
                for rep in range(2):
                    benches.append(_bench(seed, experiment, row, fmt, batch, rep, 0.2 + 0.01 * rep))
        base_evals.append({"status": "ok", "method": "none", "requested_sparsity": 0.0, "model_seed": seed,
                           "actual_sparsity": 0.0, "accuracy": {"val": 0.89, "test": 0.89},
                           "storage": {"dense": {"weight_bytes": 1_064_800}}})
        for s, acc in ((0.5, 0.88), (0.8, 0.76), (0.9, 0.57)):
            base_evals.append({"status": "ok", "method": "magnitude", "requested_sparsity": s, "model_seed": seed,
                               "actual_sparsity": s, "accuracy": {"val": acc, "test": acc},
                               "storage": {"csr": {"weight_bytes": int(1e6 * (1 - s))}},
                               "format_checks": {"csr": _check()},
                               "pruning_effect": {"test_accuracy_loss_pp": 100 * (0.89 - acc)}})
            for batch in (1, 32, 256):
                base_benches.append({"status": "ok", "method": "magnitude", "requested_sparsity": s, "format": "csr",
                                     "model_seed": seed, "batch_size": batch, "repetition": 0,
                                     "latency_ms": {"median": 0.27}, "latency_is_debug_only": False,
                                     "cuda_allocated_before_inference_bytes": 330_000,
                                     "cuda_peak_allocated_bytes": 340_000, "format_check": _check()})
        for batch in (1, 32, 256):
            base_benches.append({"status": "ok", "method": "none", "requested_sparsity": 0.0, "format": "dense",
                                 "model_seed": seed, "batch_size": batch, "repetition": 0,
                                 "latency_ms": {"median": 0.12}, "latency_is_debug_only": False,
                                 "cuda_allocated_before_inference_bytes": 1_070_000,
                                 "cuda_peak_allocated_bytes": 9_600_000, "format_check": None})
    for directory, items in ((follow / "evaluate", records), (follow / "benchmark", benches),
                             (base / "evaluate", base_evals), (base / "benchmark", base_benches)):
        for i, item in enumerate(items):
            (directory / f"{i}.json").write_text(json.dumps(item))
    return follow, base


def test_collect_and_aggregate(dirs):
    follow, base = dirs
    rows = collect(follow, base, include_baseline_timing=True)
    agg = aggregate(rows[("statistical_k1", "csr")], [1, 32, 256])
    assert agg["test"]["mean"] == pytest.approx(0.75) and agg["seeds"] == [42, 43]
    assert agg["batches"][32]["latency"]["mean"] == pytest.approx(0.205) and agg["batches"][32]["reps"] == 2
    assert agg["batches"][256]["peak"]["mean"] == 400_256
    original = aggregate(rows[("original", "dense")], [1, 32, 256])
    assert original["batches"][1]["latency"]["mean"] == pytest.approx(0.12)
    assert aggregate(rows[("original", "dense")], [1])["equivalent"] is None
    without = collect(follow, base, include_baseline_timing=False)
    assert aggregate(without[("original", "dense")], [1])["batches"] == {}


def test_summary_files(dirs, tmp_path):
    follow, base = dirs
    written = summarize_followup(follow, base, tmp_path / "summary", load_followup_config(), True)
    names = {p.name for p in written}
    assert {"summary.md", "combined.csv", "statistical.csv", "finetune.csv", "protect_fc3.csv", "lora.csv",
            "int8.csv", "gpu_memory.csv", "equivalence.csv"} <= names
    text = (tmp_path / "summary" / "summary.md").read_text()
    for heading in ("## Combined results", "## 1. Statistical pruning", "## 2. Pruning + fine-tuning",
                    "## 3. Protect the output layer", "## 4. Pruning + LoRA", "## 5. INT8 without pruning",
                    "## GPU memory by batch size", "## Energy"):
        assert heading in text
    assert "CPU debug" not in text and "0 of 1 thresholds" in text and "ran and matched" in text
    with open(tmp_path / "summary" / "combined.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 + 2 + 7  # original + magnitude 80/90 + 7 follow-up model/format rows
    assert rows[0]["row_id"] == "original" and rows[0]["latency_ms_b1"] != ""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert {"accuracy_vs_memory.png", "latency_by_model.png", "statistical_vs_magnitude.png"} <= names
