import pytest
import torch


@pytest.fixture
def toy_graph():
    """4-node graph exercising the preprocessing rules.

    Edges (source -> target): 0->1 twice (duplicate), 1->0 (reverse of the
    duplicate), 1->2 (one direction only), 2->2 (existing self-loop).
    Node 3 is isolated.

    Undirected A + I:          degrees:
        [1 1 0 0]              2
        [1 1 1 0]              3
        [0 1 1 0]              2
        [0 0 0 1]              1
    """
    edge_index = torch.tensor([[0, 0, 1, 1, 2],
                               [1, 1, 0, 2, 2]])
    a_plus_i = torch.tensor([[1, 1, 0, 0],
                             [1, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 1]], dtype=torch.float32)
    deg = torch.tensor([2.0, 3.0, 2.0, 1.0])
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    expected = d_inv_sqrt @ a_plus_i @ d_inv_sqrt
    return {"edge_index": edge_index, "num_nodes": 4, "expected": expected, "nnz": 8, "degrees": deg}


@pytest.fixture
def random_graph():
    """Larger random graph (with duplicates and self-loops) plus features, for model tests."""
    gen = torch.Generator().manual_seed(0)
    num_nodes, num_features, num_classes = 60, 12, 4
    edge_index = torch.randint(0, num_nodes, (2, 240), generator=gen)
    x = torch.rand(num_nodes, num_features, generator=gen)
    return {"edge_index": edge_index, "x": x, "num_nodes": num_nodes,
            "num_features": num_features, "num_classes": num_classes}

