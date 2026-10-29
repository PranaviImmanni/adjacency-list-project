import pytest
import torch

from src.gcn.adjacency import build_normalized_coo
from src.gcn.pruning import (cosine_edge_similarity, num_to_remove, prune_graph, save_or_check_ordering,
                             similarity_edge_order, unique_undirected_edges)


def _pairs(edges):
    return set(zip(*edges.tolist()))


def test_unique_undirected_edges(toy_graph):
    # 0->1 twice and 1->0 collapse to one pair; 2->2 is a self-loop and is dropped.
    edges = unique_undirected_edges(toy_graph["edge_index"], toy_graph["num_nodes"])
    assert edges.tolist() == [[0, 1], [1, 2]]


@pytest.mark.parametrize("method,seed", [("none", None), ("random", 101), ("similarity", None)])
def test_zero_pruning_preserves_the_original_graph(random_graph, method, seed):
    g = random_graph
    pruned = prune_graph(g["edge_index"], g["num_nodes"], method, 0.0, x=g["x"], seed=seed)
    assert pruned["removed_unique_edges"] == 0
    original = build_normalized_coo(g["edge_index"], g["num_nodes"])
    rebuilt = build_normalized_coo(pruned["retained_edges"], g["num_nodes"])
    assert torch.equal(original.indices(), rebuilt.indices())
    assert torch.equal(original.values(), rebuilt.values())


def test_none_rejects_a_removal_fraction(random_graph):
    with pytest.raises(ValueError):
        prune_graph(random_graph["edge_index"], random_graph["num_nodes"], "none", 0.1)


def test_random_pruning_needs_a_seed(random_graph):
    with pytest.raises(ValueError):
        prune_graph(random_graph["edge_index"], random_graph["num_nodes"], "random", 0.1)


def test_num_to_remove_is_an_exact_floor():
    assert num_to_remove(0.1, 5278) == 527
    assert num_to_remove(0.25, 44324) == 11081
    assert num_to_remove(0.1, 10) == 1  # 0.1 * 10 is 1.0000000000000002 in float, still 1
    assert num_to_remove(0.3, 10) == 3  # float gives 2.9999999999999996; floor of that would be 2
    with pytest.raises(ValueError):
        num_to_remove(1.5, 10)


@pytest.mark.parametrize("method,seed", [("random", 101), ("similarity", None)])
def test_removal_counts_and_nested_removal_sets(random_graph, method, seed):
    g = random_graph
    previous_removed = set()
    for fraction in (0.1, 0.25, 0.5, 0.75):
        pruned = prune_graph(g["edge_index"], g["num_nodes"], method, fraction, x=g["x"], seed=seed)
        total = pruned["original_unique_edges"]
        assert pruned["removed_unique_edges"] == num_to_remove(fraction, total)
        assert pruned["retained_unique_edges"] == total - pruned["removed_unique_edges"]
        assert pruned["actual_removal_fraction"] == pruned["removed_unique_edges"] / total
        removed = _pairs(pruned["edges"]) - _pairs(pruned["retained_edges"])
        assert len(removed) == pruned["removed_unique_edges"]
        assert previous_removed <= removed  # nested across rates
        previous_removed = removed


def test_random_selection_is_reproducible(random_graph):
    g = random_graph
    a = prune_graph(g["edge_index"], g["num_nodes"], "random", 0.5, seed=101)
    b = prune_graph(g["edge_index"], g["num_nodes"], "random", 0.5, seed=101)
    c = prune_graph(g["edge_index"], g["num_nodes"], "random", 0.5, seed=102)
    assert torch.equal(a["retained_edges"], b["retained_edges"])
    assert not torch.equal(a["retained_edges"], c["retained_edges"])


def test_random_pruning_does_not_touch_global_rng(random_graph):
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    prune_graph(random_graph["edge_index"], random_graph["num_nodes"], "random", 0.5, seed=101)
    assert torch.equal(torch.rand(3), expected)


def test_pruned_adjacency_is_symmetric_with_one_self_loop_and_renormalized(random_graph):
    g = random_graph
    n = g["num_nodes"]
    pruned = prune_graph(g["edge_index"], n, "random", 0.75, seed=103)
    coo = build_normalized_coo(pruned["retained_edges"], n)
    assert coo._nnz() == pruned["pruned_graph"]["normalized_nnz"] == 2 * pruned["retained_unique_edges"] + n

    dense = coo.to_dense()
    assert torch.equal(dense, dense.T)
    a_plus_i = (dense != 0).float()
    assert torch.equal(torch.diagonal(a_plus_i), torch.ones(n))  # exactly one self-loop per node
    # Degrees recomputed from the pruned graph, not reused from the original.
    deg = a_plus_i.sum(dim=1)
    expected = torch.diag(deg.pow(-0.5)) @ a_plus_i @ torch.diag(deg.pow(-0.5))
    assert torch.allclose(dense, expected, atol=1e-7)
    # Isolated nodes before self-loops are counted, and keep their node.
    degree = torch.bincount(pruned["retained_edges"].flatten(), minlength=n)
    assert pruned["pruned_graph"]["isolated_nodes_before_self_loops"] == int((degree == 0).sum())


def test_similarity_removes_least_similar_first():
    # Node 0 matches node 1 exactly, is orthogonal to node 2, and halfway to node 3.
    x = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
    pruned = prune_graph(edge_index, 4, "similarity", 0.34, x=x)  # floor(0.34 * 3) = 1
    assert pruned["retained_edges"].tolist() == [[0, 0], [1, 3]]
    assert torch.allclose(pruned["scores"], torch.tensor([1.0, 0.0, 2 ** -0.5], dtype=torch.float64))


def test_zero_feature_vector_scores_zero_and_ties_break_by_pair():
    x = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    # Pairs (0,1) and (1,2) touch the zero vector; (0,2) is orthogonal: all score 0.
    edges = unique_undirected_edges(torch.tensor([[2, 1, 0, 2], [3, 0, 2, 1]]), 4)
    assert edges.tolist() == [[0, 0, 1, 2], [1, 2, 2, 3]]
    order, scores = similarity_edge_order(x, edges)
    assert scores.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert not torch.isnan(scores).any()
    assert order.tolist() == [0, 1, 2, 3]  # equal scores keep (u, v) order


def test_similarity_chunking_matches_one_pass(random_graph):
    g = random_graph
    edges = unique_undirected_edges(g["edge_index"], g["num_nodes"])
    one_pass = cosine_edge_similarity(g["x"], edges, chunk_size=10_000)
    chunked = cosine_edge_similarity(g["x"], edges, chunk_size=7)
    assert torch.equal(one_pass, chunked)


def test_saved_ordering_is_checked(tmp_path, random_graph):
    g = random_graph
    path = tmp_path / "order.pt"
    a = prune_graph(g["edge_index"], g["num_nodes"], "random", 0.5, seed=101)
    first = save_or_check_ordering(path, a)
    assert save_or_check_ordering(path, a) == first
    b = prune_graph(g["edge_index"], g["num_nodes"], "random", 0.5, seed=102)
    with pytest.raises(RuntimeError):
        save_or_check_ordering(path, b)
