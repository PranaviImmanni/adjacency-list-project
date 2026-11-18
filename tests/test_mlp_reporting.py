import json

import pytest

from src.mlp.reporting import accuracy_points, benchmark_points, load_mlp_results, summarize_mlp


def _check(epsilon):
    return {"epsilon_logit": epsilon, "mean_abs_logit_diff": epsilon / 10, "allclose": True,
            "prediction_agreement_all": 1.0, "accuracy_difference_pp": 0.0, "equivalent": True,
            "margin": {"test": {"margin_condition_fraction": 1.0}, "all": {"disagreements_satisfying_condition": 0}}}


def _evaluation(seed, method, sparsity, pruning_seed, test_acc):
    record = {"experiment": "mlp_evaluate", "status": "ok", "model_seed": seed, "method": method,
              "requested_sparsity": sparsity, "pruning_seed": pruning_seed, "actual_sparsity": sparsity,
              "device": "cuda:1", "accuracy": {"val": test_acc, "test": test_acc},
              "storage": {"dense": {"weight_bytes": 1000, "bias_bytes": 40}}}
    if method != "none":
        nnz = round(250 * (1 - sparsity))
        record["storage"].update(dense_mask={"weight_bytes": 1250}, coo={"weight_bytes": 20 * nnz},
                                 csr={"weight_bytes": 12 * nnz + 80})
        record["pruning_effect"] = {"test_accuracy_loss_pp": 100 * (0.9 - test_acc),
                                    "prediction_disagreement_test": 0.1}
        record["format_checks"] = {f: _check(1e-6) for f in ("dense_mask", "coo", "csr")}
    return record


def _benchmark(seed, method, sparsity, pruning_seed, storage_format, batch, rep, median, timestamp="t1"):
    return {"experiment": "mlp_benchmark", "status": "ok", "model_seed": seed, "method": method,
            "requested_sparsity": sparsity, "pruning_seed": pruning_seed, "format": storage_format,
            "batch_size": batch, "repetition": rep, "timestamp": timestamp, "device": "cuda:1",
            "latency_ms": {"median": median}, "latency_is_debug_only": False, "measured_runs": 50,
            "cuda_peak_allocated_bytes": 5000 + batch, "cuda_allocated_before_inference_bytes": 1000 + batch,
            "weight_storage_bytes": 1000,
            "format_check": None if storage_format == "dense" else {"equivalent": True, "epsilon_logit": 1e-6}}


@pytest.fixture
def results_dir(tmp_path):
    root = tmp_path / "cuda1"
    for sub in ("baseline", "evaluate", "benchmark"):
        (root / sub).mkdir(parents=True)
    records = {"baseline": [], "evaluate": [], "benchmark": []}
    for seed, base in ((42, 0.89), (43, 0.88)):
        records["baseline"].append({"experiment": "mlp_baseline", "status": "ok", "model_seed": seed,
                                    "best_epoch": 10, "validation_accuracy": base, "test_accuracy": base,
                                    "device": "cuda:1"})
        records["evaluate"].append(_evaluation(seed, "none", 0.0, None, base))
        for sparsity in (0.5, 0.9):
            records["evaluate"].append(_evaluation(seed, "magnitude", sparsity, None, base - sparsity / 2))
            records["evaluate"].append(_evaluation(seed, "random", sparsity, 101, base - sparsity))
            for batch in (1, 256):
                records["benchmark"].append(_benchmark(seed, "none", 0.0, None, "dense", batch, 0, 0.05))
                for storage_format in ("dense", "dense_mask", "coo", "csr"):
                    for rep, median in ((0, 0.10), (1, 0.30), (2, 0.20)):
                        records["benchmark"].append(_benchmark(seed, "magnitude", sparsity, None, storage_format,
                                                               batch, rep, median + (seed - 42)))
    # A rerun of one repetition replaces the older file.
    records["benchmark"].append(_benchmark(42, "magnitude", 0.5, None, "csr", 1, 0, 0.20, timestamp="t2"))
    for sub, items in records.items():
        for i, record in enumerate(items):
            (root / sub / f"{i}.json").write_text(json.dumps(record))
    (root / "evaluate" / "broken.json").write_text("{not json")
    return root


def test_aggregation(results_dir):
    baselines, evaluations, benchmarks = load_mlp_results(results_dir)
    assert len(baselines) == 2
    acc = accuracy_points(evaluations)
    assert acc[("magnitude", 0.5)]["test"]["n"] == 2
    assert acc[("magnitude", 0.5)]["test"]["mean"] == pytest.approx((0.64 + 0.63) / 2)
    bench = benchmark_points(benchmarks)
    # seed 42: reps 0.20 (rerun), 0.30, 0.20 -> median 0.20; seed 43: 1.10, 1.30, 1.20 -> 1.20
    point = bench[("magnitude", 0.5, "csr", 1)]
    assert point["latency"]["mean"] == pytest.approx(0.70) and point["latency"]["min"] == pytest.approx(0.20)
    assert point["units"] == 2 and point["min_reps"] == 3
    assert point["resident"]["mean"] == 1001 and point["peak_minus_resident"]["mean"] == 4000


def test_summary_files(results_dir, tmp_path):
    text, written = summarize_mlp(results_dir, tmp_path / "summary")
    names = {p.name for p in written}
    assert {"summary.md", "accuracy.csv", "weight_storage.csv", "latency.csv", "peak_memory.csv",
            "epsilon.csv", "baseline.csv", "runs.csv"} <= names
    summary = (tmp_path / "summary" / "summary.md").read_text()
    assert "## Accuracy vs sparsity" in summary and "## Energy" in summary
    assert "CPU debug" not in summary
    runs = (tmp_path / "summary" / "runs.csv").read_text().splitlines()
    assert runs[0].startswith("experiment,dataset,device") and len(runs) == 1 + 2 * (2 + 48)  # per seed: original x 2 batches + 2 levels x 2 batches x 4 formats x 3 reps
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert {"accuracy_vs_sparsity.png", "weight_storage_vs_sparsity.png", "latency_vs_sparsity_magnitude.png",
            "peak_memory_vs_sparsity_magnitude.png"} <= names


def test_summary_without_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_mlp(tmp_path, tmp_path / "out")
