import pytest
import torch

from src.gcn.adjacency import STORAGE_FORMATS, adjacency_multiply, build_normalized_coo, convert_adjacency
from src.gcn.model import GCN


@pytest.fixture
def setup(random_graph):
    torch.manual_seed(0)
    g = random_graph
    model = GCN(g["num_features"], 16, g["num_classes"], dropout=0.5)
    coo = build_normalized_coo(g["edge_index"], g["num_nodes"])
    return model, g, coo


@pytest.mark.parametrize("storage_format", STORAGE_FORMATS)
def test_forward_runs_for_each_format(setup, storage_format):
    model, g, coo = setup
    model.eval()
    with torch.inference_mode():
        out = model(g["x"], convert_adjacency(coo, storage_format))
    assert out.shape == (g["num_nodes"], g["num_classes"])
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_output_shapes_match_across_formats(setup):
    model, g, coo = setup
    model.eval()
    with torch.inference_mode():
        shapes = {f: model(g["x"], convert_adjacency(coo, f)).shape for f in STORAGE_FORMATS}
    assert len(set(shapes.values())) == 1


def test_training_step_with_coo(setup):
    model, g, coo = setup
    model.train()
    out = model(g["x"], coo)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_adjacency_multiply_matches_dense_matmul(setup):
    _, g, coo = setup
    expected = coo.to_dense() @ g["x"]
    for storage_format in STORAGE_FORMATS:
        assert torch.allclose(adjacency_multiply(convert_adjacency(coo, storage_format), g["x"]), expected,
                              atol=1e-6)


def test_adjacency_multiply_rejects_other_layouts(setup):
    _, g, coo = setup
    with pytest.raises(ValueError):
        adjacency_multiply(coo.to_sparse_csc(), g["x"])
