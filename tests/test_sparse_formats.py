import pytest
import torch

from src.gcn.adjacency import (adjacency_storage_breakdown, adjacency_storage_bytes, build_normalized_coo,
                               convert_adjacency, estimate_dense_bytes, layout_name)


def test_duplicate_edges_collapse(toy_graph):
    coo = build_normalized_coo(toy_graph["edge_index"], toy_graph["num_nodes"])
    # 0->1 appears twice and 1->0 once: still exactly one (0,1) and one (1,0) entry.
    assert coo._nnz() == toy_graph["nnz"]
    keys = coo.indices()[0] * toy_graph["num_nodes"] + coo.indices()[1]
    assert keys.unique().numel() == keys.numel()
    # Unweighted: the duplicate contributes weight 1, not 2.
    assert torch.allclose(coo.to_dense()[0, 1], torch.tensor(1 / (2 * 3) ** 0.5))


def test_exactly_one_self_loop_per_node(toy_graph):
    coo = build_normalized_coo(toy_graph["edge_index"], toy_graph["num_nodes"])
    row, col = coo.indices()
    diagonal_rows = row[row == col]
    assert torch.equal(diagonal_rows, torch.arange(toy_graph["num_nodes"]))
    # Diagonal of D^-1/2 (A+I) D^-1/2 is 1/deg, so node 2's existing loop was not double counted.
    assert torch.allclose(coo.to_dense().diagonal(), 1 / toy_graph["degrees"])


def test_coo_is_coalesced_fp32_and_row_major(toy_graph):
    coo = build_normalized_coo(toy_graph["edge_index"], toy_graph["num_nodes"])
    assert coo.layout == torch.sparse_coo
    assert coo.is_coalesced()
    assert coo.dtype == torch.float32
    keys = coo.indices()[0] * toy_graph["num_nodes"] + coo.indices()[1]
    assert torch.equal(keys, keys.sort().values)


def test_normalized_values_and_shape(toy_graph):
    coo = build_normalized_coo(toy_graph["edge_index"], toy_graph["num_nodes"])
    assert coo.shape == (4, 4)
    assert torch.allclose(coo.to_dense(), toy_graph["expected"])
    assert torch.allclose(coo.to_dense(), coo.to_dense().T)  # undirected


def test_one_directional_edge_is_symmetrized():
    coo = build_normalized_coo(torch.tensor([[0], [1]]), 2)
    dense = coo.to_dense()
    assert dense[0, 1] > 0 and dense[1, 0] > 0


def test_graph_without_edges_is_identity():
    coo = build_normalized_coo(torch.empty(2, 0, dtype=torch.long), 3)
    assert torch.equal(coo.to_dense(), torch.eye(3))


def test_rejects_out_of_range_node_ids():
    with pytest.raises(ValueError):
        build_normalized_coo(torch.tensor([[0], [5]]), 3)


def test_dense_coo_csr_equivalent(toy_graph):
    coo = build_normalized_coo(toy_graph["edge_index"], toy_graph["num_nodes"])
    dense = convert_adjacency(coo, "dense")
    csr = convert_adjacency(coo, "csr")

    assert dense.layout == torch.strided
    assert csr.layout == torch.sparse_csr
    assert torch.allclose(coo.to_dense(), dense)
    assert torch.allclose(coo.to_dense(), csr.to_dense())
    # Derived from the same values, so equal bit for bit.
    assert torch.equal(dense, csr.to_dense())
    assert [layout_name(a) for a in (dense, coo, csr)] == ["dense", "coo", "csr"]


def test_convert_rejects_unknown_format_and_non_canonical_input(toy_graph):
    coo = build_normalized_coo(toy_graph["edge_index"], toy_graph["num_nodes"])
    with pytest.raises(ValueError):
        convert_adjacency(coo, "csc")
    with pytest.raises(ValueError):
        convert_adjacency(coo.to_dense(), "csr")


def test_exact_storage_bytes(toy_graph):
    n, nnz = toy_graph["num_nodes"], toy_graph["nnz"]
    coo = build_normalized_coo(toy_graph["edge_index"], n)
    dense, csr = convert_adjacency(coo, "dense"), convert_adjacency(coo, "csr")

    # FP32 values (4 bytes) and int64 indices (8 bytes).
    assert adjacency_storage_breakdown(dense) == {"values": n * n * 4}
    assert adjacency_storage_breakdown(coo) == {"values": nnz * 4, "indices": 2 * nnz * 8}
    assert adjacency_storage_breakdown(csr) == {"values": nnz * 4, "col_indices": nnz * 8,
                                                "crow_indices": (n + 1) * 8}
    assert adjacency_storage_bytes(dense) == 64
    assert adjacency_storage_bytes(coo) == 160
    assert adjacency_storage_bytes(csr) == 136


def test_estimate_dense_bytes():
    assert estimate_dense_bytes(4, torch.float32) == 64
    assert estimate_dense_bytes(4, "float32") == 64
    assert estimate_dense_bytes(2708, torch.float32) == 2708 * 2708 * 4
