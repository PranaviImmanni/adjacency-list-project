import json

import pytest

from src.gcn.synthetic_reporting import (aggregate, compare_to_dense, find_crossovers, load_sweeps,
                                         summarize_synthetic, sweep_variable)


def _run(nodes, storage_format, seed, rep, latency, nbytes, peak, density=None, fingerprint=None):
    edges = nodes * 4
    return {"schema_version": 1, "dataset": "synthetic", "format": storage_format, "repetition": rep,
            "device": "cuda:0", "latency_is_debug_only": False, "num_nodes": nodes, "nnz": 2 * edges + nodes,
            "graph": {"requested_avg_degree": None if density else 8, "requested_density": density,
                      "num_nodes": nodes, "num_edges": edges, "avg_degree": 8.0, "density": 8 / (nodes - 1),
                      "graph_seed": seed, "graph_fingerprint": fingerprint or f"g{nodes}-{seed}"},
            "latency_ms": {"median": latency}, "adjacency_bytes": nbytes, "cuda_peak_allocated_bytes": peak,
            "prediction": {"max_abs_logit_diff": 1e-7, "equivalent": True}}


@pytest.fixture
def size_runs():
    """Two sizes; dense is faster at 100 nodes and slower at 1000 nodes."""
    runs = []
    for seed in (42, 43, 44):
        for rep in (0, 1, 2):
            runs += [_run(100, "dense", seed, rep, 0.2, 40_000, 50_000),
                     _run(100, "csr", seed, rep, 0.3 + 0.01 * rep, 9_000, 20_000),
                     _run(1000, "dense", seed, rep, 2.0, 4_000_000, 4_100_000),
                     _run(1000, "csr", seed, rep, 0.3 + 0.1 * (seed - 42), 90_000, 120_000)]
    return runs


def test_sweep_variable(size_runs):
    assert sweep_variable(size_runs) == "nodes"
    density = [_run(500, "csr", 42, 0, 1.0, 1, 1, density=p) for p in (0.01, 0.1)]
    assert sweep_variable(density) == "density"
    assert sweep_variable(size_runs[:1]) is None


def test_aggregate_median_of_seed_medians(size_runs):
    graphs, points = aggregate(size_runs, "nodes")
    assert list(points) == [100, 1000]
    # Per seed the repetition medians are 0.30/0.31/0.32 -> 0.31 for every seed.
    assert points[100]["csr"]["latency"] == pytest.approx((0.31, 0.31, 0.31))
    # Seeds give 0.3, 0.4, 0.5 at 1000 nodes: median 0.4, range 0.3-0.5.
    assert points[1000]["csr"]["latency"] == pytest.approx((0.4, 0.3, 0.5))
    assert points[1000]["dense"]["adjacency_bytes"][0] == 4_000_000
    assert points[1000]["dense"]["runs"] == 9
    assert graphs[100]["same_graph"] and graphs[100]["seeds"] == [42, 43, 44]


def test_aggregate_flags_different_graphs_across_formats():
    runs = [_run(100, "dense", 42, 0, 1, 1, 1, fingerprint="a"), _run(100, "csr", 42, 0, 1, 1, 1, fingerprint="b")]
    graphs, _ = aggregate(runs, None)
    assert not graphs[100]["same_graph"]


def test_find_crossovers_interpolates():
    # sparse - dense goes from -1 at x=0 to +3 at x=4: zero at x=1.
    (x0, x1, estimate, ties, better_below), = find_crossovers([0, 4], [1, 7], [2, 4])
    assert (x0, x1, ties, better_below) == (0, 4, [], True)
    assert estimate == pytest.approx(1.0)


def test_find_crossovers_in_log_x():
    (_, _, estimate, _, better_below), = find_crossovers([100, 10_000], [1, 3], [2, 2], log_x=True)
    assert estimate == pytest.approx(1000)
    assert better_below


def test_ties_are_not_crossovers():
    # Within 2% of dense at the first two points: ties, so no crossover is reported.
    assert compare_to_dense([0.448, 0.453, 1.0], [0.449, 0.449, 0.449]) == [0, 0, 1]
    assert find_crossovers([1, 2, 3], [0.448, 0.453, 1.0], [0.449, 0.449, 0.449]) == []
    # A tie between a better and a worse point is the crossover itself.
    (x0, x1, estimate, ties, better_below), = find_crossovers([1, 2, 3], [0.2, 0.449, 1.0], [0.449] * 3)
    assert (x0, x1, estimate, ties, better_below) == (1, 3, None, [2], True)


def test_summarize_writes_tables_and_charts(size_runs, tmp_path):
    sweep = tmp_path / "results" / "size_cuda0"
    sweep.mkdir(parents=True)
    for i, run in enumerate(size_runs):
        (sweep / f"run{i}.json").write_text(json.dumps(run))
    (sweep / "broken.json").write_text("{not json")
    (tmp_path / "results" / "empty").mkdir()

    assert list(load_sweeps(tmp_path / "results")) == ["size_cuda0"]
    text, written = summarize_synthetic(tmp_path / "results", tmp_path / "out")
    summary = (tmp_path / "out" / "summary.md").read_text()

    assert "## size_cuda0" in summary
    assert "| 1,000 | 2.000 | 0.400 (0.300–0.500) | csr |" in summary
    assert "| Latency | csr | between 100 and 1,000 |" in summary
    assert "All runs equivalent to the CPU COO reference: yes" in text
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert [p.name for p in written] == ["summary.md"]
        return
    assert {p.name for p in written} == {"summary.md", "size_cuda0_latency.png", "size_cuda0_adjacency_memory.png",
                                         "size_cuda0_peak_memory.png"}


def test_summarize_without_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_synthetic(tmp_path, tmp_path / "out")
