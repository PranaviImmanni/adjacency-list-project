import json

import pytest

from src.gcn.reporting import human_bytes, load_results, summarize


def _metrics(nbytes):
    return {"test_accuracy": 0.8, "reference_test_accuracy": 0.8, "accuracy_difference_pp": 0.0,
            "prediction_agreement_all": 1.0, "prediction_agreement_test": 1.0, "mean_abs_logit_diff": 0.0,
            "max_abs_logit_diff": 0.0, "allclose": True, "atol": 1e-5, "rtol": 1e-4,
            "adjacency_bytes": nbytes, "equivalent": True}


def _benchmark(storage_format, repetition, median, nbytes, timestamp="2026-09-30T00:00:00+00:00"):
    return {"schema_version": 1, "timestamp": timestamp, "dataset": "toy", "format": storage_format,
            "repetition": repetition, "device": "cpu", "measured_runs": 3,
            "latency_ms": {"median": median}, "latency_samples_ms": [median - 0.1, median, median + 0.1],
            "adjacency_bytes": nbytes, "adjacency_bytes_breakdown": {"values": nbytes},
            "dense_adjacency_bytes_estimate": 64, "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None, "prediction": {"equivalent": True}}


@pytest.fixture
def results_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    comparison = {"dataset": "toy", "device": "cpu", "checkpoint": str(tmp_path / "missing.pt"),
                  "reference_format": "coo", "all_equivalent": True,
                  "formats": {"dense": _metrics(64), "coo": _metrics(160), "csr": _metrics(136)}}
    (raw / "toy_comparison_cpu.json").write_text(json.dumps(comparison))
    (raw / "toy_csr_cpu_rep0.json").write_text(json.dumps(_benchmark("csr", 0, 1.0, 136)))
    (raw / "toy_csr_cpu_rep1.json").write_text(json.dumps(_benchmark("csr", 1, 3.0, 136)))
    # A rerun of rep 0 replaces the older file's numbers.
    (raw / "toy_csr_cpu_rep0_rerun.json").write_text(
        json.dumps(_benchmark("csr", 0, 2.0, 136, timestamp="2026-10-01T00:00:00+00:00")))
    (raw / "notes.json").write_text("{not json")
    return raw


def test_load_results_keeps_latest_run_per_repetition(results_dir):
    comparisons, benchmarks = load_results(results_dir)
    assert list(comparisons) == [("toy", "cpu")]
    reps = benchmarks[("toy", "cpu")]["csr"]
    assert sorted(r["latency_ms"]["median"] for r in reps) == [2.0, 3.0]


def test_summarize_writes_tables_and_charts(results_dir, tmp_path):
    text, written = summarize(results_dir, tmp_path / "summary")
    summary = (tmp_path / "summary" / "summary.md").read_text()

    assert "DEBUG ONLY" in text and "DEBUG ONLY" in summary
    assert "| dense | 80.00% | +0.00 | 100.00% |" in summary
    assert "| Mean \\|Δlogit\\| |" in summary  # pipes in headers must not split Markdown columns
    assert "| csr | 136 | 136 B | 212.50% |" in summary  # toy dense is 64 bytes
    assert "| csr | 2 | 2.500 |" in summary  # median of rep medians 2.0 and 3.0
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert [p.name for p in written] == ["summary.md"]
        return
    assert {p.name for p in written} == {"summary.md", "toy_adjacency_memory.png", "toy_latency_cpu.png"}
    assert all(p.stat().st_size > 0 for p in written)


def test_summarize_without_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path, tmp_path / "out")


def test_human_bytes():
    assert human_bytes(29_333_056) == "29.3 MB"
    assert human_bytes(265_280) == "265.3 KB"
    assert human_bytes(64) == "64 B"
    assert human_bytes(None) == "n/a"
