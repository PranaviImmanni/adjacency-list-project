"""The GPU comparison report on synthetic GPU-style results with energy (tables, CSVs, consistency checks)."""

import csv
import json

import pytest

from src.mlp.config import load_comparison_config
from src.mlp.gpu_comparison_reporting import aggregate_benchmarks, summarize_comparison


def _check(strict=True):
    """A format check as src/mlp/gpu_comparison.equivalence writes it."""
    return {"equivalent": strict, "same_predictions": True, "epsilon_logit": 1e-6 if strict else 1.2e-4,
            "bitwise_equal": False, "prediction_agreement_all": 1.0, "max_abs_error_vs_float64": 1.2e-4,
            "reference_max_abs_error_vs_float64": 1.2e-4, "max_abs_exact_logit": 116.0}


def _energy(per_example, batch):
    per_batch = per_example * batch
    return {"available": True, "method": "total_energy_counter", "device": {"name": "RTX 3090", "matched_by": "uuid"},
            "duration_s": 2.01, "batches": 1000, "energy_j": per_batch * 1000, "average_power_w": 105.0,
            "energy_per_batch_j": per_batch, "energy_per_example_j": per_example, "idle_power_w": 30.0,
            "net_average_power_w": 75.0, "net_energy_per_batch_j": per_batch * 0.7,
            "net_energy_per_example_j": per_example * 0.7, "idle": {"counter_update_interval_ms": 100.0},
            "idle_power_reading_w": 31.0, "active_power_reading_w": 106.0, "active": {"batches": 1000},
            "state_before": {"gpu_utilization_pct": 0}, "other_processes": [3700], "other_processes_present": True}


def _spec_fields(kind, method=None, sparsity=None, rank=None, rule=None):
    if kind == "low_rank":
        return {"kind": kind, "method": "low_rank", "requested_sparsity": None, "pruning_seed": None, "rank": rank,
                "rule_label": rule, "factor_rule": "all" if rule == "all" else "abs_std"}
    return {"kind": kind, "method": method, "requested_sparsity": sparsity, "pruning_seed": None, "rank": None,
            "rule_label": None, "factor_rule": None}


@pytest.fixture
def dirs(tmp_path):
    root, base = tmp_path / "gpu", tmp_path / "base"
    for d in (root / "evaluate", root / "benchmark", base / "benchmark"):
        d.mkdir(parents=True)
    n = 0
    for seed in (42, 43):
        models = [("original", "none", 0.0, None, None, ["dense"]),
                  ("pruned", "magnitude", 0.9, None, None, ["dense", "dense_mask", "coo", "csr", "adjacency_list"]),
                  ("low_rank", None, None, 32, "all", ["factor_adjacency", "factor_dense"])]
        for kind, method, sparsity, rank, rule, formats in models:
            fields = _spec_fields(kind, method, sparsity, rank, rule)
            tag = f"m{seed}_" + ("none_s0" if kind == "original" else "magnitude_s90" if kind == "pruned"
                                 else "lowrank_r32_all")
            reference = {"original": "dense", "pruned": "dense", "low_rank": "factor_dense"}[kind]
            acc = {"original": 0.889, "pruned": 0.55, "low_rank": 0.856}[kind]
            record = {"status": "ok", "tag": tag, "model_seed": seed, **fields, "formats": formats,
                      "device": "cuda:1", "device_name": "RTX 3090", "accuracy": {"val": acc, "test": acc},
                      "format_accuracy": {f: {"val": acc, "test": acc} for f in formats},
                      "storage": {f: {"weight_bytes": 322_744 if f in ("csr", "adjacency_list") else 1_064_800}
                                  for f in formats},
                      "format_checks": {f: _check(kind != "low_rank") for f in formats if f != reference},
                      "edges": 26_620, "actual_sparsity": sparsity,
                      "original_sweep": {"found": kind != "low_rank", "path": "results/mlp/cuda1/evaluate/x.json",
                                         "test_accuracy_equal": True, "logits_sha256_equal": True}}
            if kind != "original":
                record["effect_vs_original"] = {"test_accuracy_loss_pp": 100 * (0.889 - acc),
                                                "prediction_disagreement_test": 0.1}
            if kind == "low_rank":
                record["low_rank"] = {"factor_sparsity": 0.0, "mean_weight_reconstruction_mse": 0.0025}
                record["pranavi_cpu"] = {"found": True, "source": "reports/low_rank_adjacency.csv",
                                         "pranavi_test_accuracy": acc, "test_accuracy": acc, "difference_pp": 0.0,
                                         "difference_images": 0, "edges_equal": True, "stored_bytes_equal": True}
            (root / "evaluate" / f"{tag}.json").write_text(json.dumps(record))
            for storage_format in formats:
                for batch in (1, 32, 256):
                    for rep in range(2):
                        latency = 0.25 + (0.1 if storage_format == "adjacency_list" else 0) + 0.01 * rep
                        bench = {"status": "ok", "tag": tag, "model_seed": seed, **fields, "format": storage_format,
                                 "batch_size": batch, "repetition": rep, "latency_ms": {"median": latency},
                                 "cuda_allocated_before_inference_bytes": 330_000,
                                 "cuda_peak_allocated_bytes": 340_000 + batch, "energy": _energy(2e-5 / batch, batch),
                                 "weight_storage_bytes": record["storage"][storage_format]["weight_bytes"],
                                 "format_check": None if storage_format == reference else _check(kind != "low_rank"),
                                 "latency_is_debug_only": False, "device": "cuda:1", "device_name": "RTX 3090",
                                 "edges": 26_620, "checkpoint": {"sha256": "abc"}}
                        (root / "benchmark" / f"{n}.json").write_text(json.dumps(bench))
                        n += 1
                        if kind != "low_rank" and storage_format != "adjacency_list" and rep == 0:
                            old = {"status": "ok", "method": method, "requested_sparsity": sparsity,
                                   "pruning_seed": None, "format": storage_format, "batch_size": batch,
                                   "model_seed": seed, "repetition": 0, "latency_ms": {"median": 0.25},
                                   "device": "cuda:1"}
                            (base / "benchmark" / f"{n}.json").write_text(json.dumps(old))
    return root, base


def test_aggregate_uses_median_over_repetitions(dirs):
    root, _ = dirs
    benches = [json.loads(p.read_text()) for p in (root / "benchmark").glob("*.json")]
    agg = aggregate_benchmarks(benches)
    point = agg[(("pruned", "magnitude", 0.9), "adjacency_list", 32)]
    assert point["latency"]["mean"] == pytest.approx(0.355) and point["seeds"] == [42, 43] and point["runs"] == 4
    assert point["energy_per_example_j"]["mean"] == pytest.approx(2e-5 / 32)


def test_summary_sections_and_files(dirs, tmp_path):
    root, base = dirs
    written = summarize_comparison(root, base, tmp_path / "summary", load_comparison_config())
    names = {p.name for p in written}
    assert {"summary.md", "pruned.csv", "low_rank.csv", "equivalence.csv", "pranavi_crosscheck.csv", "runs.csv",
            "stability.csv"} <= names
    text = (tmp_path / "summary" / "summary.md").read_text()
    for heading in ("## Energy measurement", "## Pruned weights: all five formats", "## Low-rank factor networks",
                    "## Equivalence with the reference", "## Agreement with Pranavi's CPU results",
                    "## Consistency with the original sweep"):
        assert heading in text
    assert "total_energy_counter" in text and "matched by uuid" in text and "CPU debug" not in text
    assert "PIDs 3700" in text and "right before the measurement: 0 runs" in text
    with open(tmp_path / "summary" / "equivalence.csv") as f:
        equivalence = {r["Format"]: r for r in csv.DictReader(f)}
    assert equivalence["CSR"]["Within tolerance"] == "2 / 2" and equivalence["CSR"]["Same predictions"] == "yes"
    lists = equivalence["factor adjacency list"]
    assert lists["Within tolerance"] == "0 / 2" and lists["Same predictions"] == "yes"
    assert lists["Batches within tolerance"] == "0 / 12" and lists["Max |logit|"] == "116"
    assert equivalence["CSR"]["Batches within tolerance"] == "12 / 12"
    with open(tmp_path / "summary" / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * (1 + 5 + 2) * 3 * 2 and rows[0]["energy_per_example_j"] != ""
    with open(tmp_path / "summary" / "stability.csv") as f:
        stability = list(csv.DictReader(f))
    assert stability and all(float(r["median_ratio"]) == pytest.approx(1.02) for r in stability)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert {"latency_vs_sparsity.png", "energy_vs_sparsity.png", "low_rank_accuracy_vs_storage.png"} <= names
