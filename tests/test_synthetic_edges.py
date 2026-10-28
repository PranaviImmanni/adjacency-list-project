import math

import pytest
import torch

from src.gcn.adjacency import build_normalized_coo
from src.synthetic.edge_generator import (edges_for_avg_degree, edges_for_density, num_possible_edges,
                                          pair_from_index, sample_undirected_edges)


def test_pair_from_index_enumerates_every_pair_once():
    n = 37
    i, j = pair_from_index(torch.arange(num_possible_edges(n)))
    assert (i < j).all() and (i >= 0).all() and (j < n).all()
    pairs = set(zip(i.tolist(), j.tolist()))
    assert pairs == {(a, b) for b in range(n) for a in range(b)}


def test_pair_from_index_exact_at_large_n():
    # Last and first pairs of a 10M-node graph: float sqrt alone would be off by one here.
    n = 10_000_000
    total = num_possible_edges(n)
    i, j = pair_from_index(torch.tensor([0, 1, 2, total - 2, total - 1]))
    assert i.tolist() == [0, 0, 1, n - 3, n - 2]
    assert j.tolist() == [1, 2, 2, n - 1, n - 1]


@pytest.mark.parametrize("num_nodes,num_edges", [(1000, 4000), (200, 7960), (50, 1225), (10, 0)])
def test_exact_count_unique_ordered_in_range(num_nodes, num_edges):
    edges = sample_undirected_edges(num_nodes, num_edges, seed=42)
    assert edges.shape == (2, num_edges)
    assert edges.dtype == torch.long
    source, target = edges
    assert (source < target).all()
    assert (source >= 0).all() and (target < num_nodes).all()
    assert (target * num_nodes + source).unique().numel() == num_edges


def test_complete_graph():
    edges = sample_undirected_edges(6, 15, seed=0)
    assert edges.size(1) == num_possible_edges(6)


def test_rejects_too_many_edges():
    with pytest.raises(ValueError):
        sample_undirected_edges(5, 11, seed=0)


def test_same_seed_same_graph_different_seed_different_graph():
    a = sample_undirected_edges(500, 2000, seed=42)
    assert torch.equal(a, sample_undirected_edges(500, 2000, seed=42))
    assert not torch.equal(a, sample_undirected_edges(500, 2000, seed=43))


def test_does_not_touch_global_rng():
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    sample_undirected_edges(1000, 5000, seed=42)
    assert torch.equal(torch.rand(3), expected)


def test_huge_graph_without_quadratic_memory():
    # 1M nodes has ~5e11 possible pairs; materializing them would need terabytes.
    edges = sample_undirected_edges(1_000_000, 1000, seed=0)
    assert edges.shape == (2, 1000)
    assert edges.max() < 1_000_000


def test_pairs_are_uniform():
    # 10 nodes, 45 possible pairs, 9 chosen per graph: each pair is picked with probability 0.2.
    n, m, trials = 10, 9, 4000
    counts = torch.zeros(num_possible_edges(n))
    for seed in range(trials):
        source, target = sample_undirected_edges(n, m, seed)
        counts[target * (target - 1) // 2 + source] += 1
    expected = trials * m / num_possible_edges(n)
    std = math.sqrt(trials * 0.2 * 0.8)
    assert (counts - expected).abs().max() < 5 * std


def test_normalized_adjacency_has_2m_plus_n_nonzeros():
    # Few edges, so many nodes are isolated; they must still get a self-loop.
    n, m = 300, 100
    edges = sample_undirected_edges(n, m, seed=42)
    coo = build_normalized_coo(edges, n)
    assert coo.shape == (n, n)
    assert coo._nnz() == 2 * m + n


def test_edge_count_helpers():
    assert num_possible_edges(5000) == 12_497_500
    assert edges_for_avg_degree(1000, 8) == 4000
    assert edges_for_density(5000, 0.01) == 124_975
    m = edges_for_density(5000, 0.4)
    assert abs(m / num_possible_edges(5000) - 0.4) < 1e-6
