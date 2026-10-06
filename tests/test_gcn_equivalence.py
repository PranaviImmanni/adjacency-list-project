import pytest
import torch

from src.gcn.adjacency import STORAGE_FORMATS, build_normalized_coo, convert_adjacency
from src.gcn.evaluation import ATOL, RTOL, compare_to_reference, is_equivalent, run_inference
from src.gcn.model import GCN, load_gcn_checkpoint


def _logits_per_format(model, x, coo):
    return {f: run_inference(model, x, convert_adjacency(coo, f)) for f in STORAGE_FORMATS}


def test_one_model_state_gives_close_logits_on_toy_graph(toy_graph):
    torch.manual_seed(0)
    x = torch.rand(toy_graph["num_nodes"], 5)
    model = GCN(5, 8, 3)
    coo = build_normalized_coo(toy_graph["edge_index"], toy_graph["num_nodes"])

    logits = _logits_per_format(model, x, coo)
    for storage_format in ("dense", "csr"):
        assert torch.allclose(logits[storage_format], logits["coo"], atol=ATOL, rtol=RTOL)
        assert torch.equal(logits[storage_format].argmax(1), logits["coo"].argmax(1))


def test_compare_to_reference_reports_equivalence(random_graph):
    torch.manual_seed(0)
    g = random_graph
    model = GCN(g["num_features"], 16, g["num_classes"])
    coo = build_normalized_coo(g["edge_index"], g["num_nodes"])
    y = torch.randint(0, g["num_classes"], (g["num_nodes"],))
    test_mask = torch.arange(g["num_nodes"]) % 2 == 0

    logits = _logits_per_format(model, g["x"], coo)
    for storage_format in STORAGE_FORMATS:
        m = compare_to_reference(logits["coo"], logits[storage_format], y, test_mask)
        assert m["prediction_agreement_all"] == 1.0
        assert m["prediction_agreement_test"] == 1.0
        assert m["accuracy_difference_pp"] == 0.0
        assert m["max_abs_logit_diff"] <= ATOL
        assert m["allclose"]
        assert is_equivalent(m)


def test_compare_to_reference_detects_a_difference():
    ref = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    other = torch.tensor([[2.0, 0.0], [3.0, 2.0]])  # second node flips class
    y = torch.tensor([0, 1])
    m = compare_to_reference(ref, other, y, torch.tensor([True, True]))
    assert m["prediction_agreement_all"] == 0.5
    assert m["accuracy_difference_pp"] == pytest.approx(-50.0)
    assert m["max_abs_logit_diff"] == 3.0
    assert not m["allclose"]
    assert not is_equivalent(m)


def test_checkpoint_round_trip_preserves_logits(tmp_path, random_graph):
    torch.manual_seed(0)
    g = random_graph
    model = GCN(g["num_features"], 16, g["num_classes"], dropout=0.5)
    coo = build_normalized_coo(g["edge_index"], g["num_nodes"])
    path = tmp_path / "toy.pt"
    torch.save({"model_state_dict": model.state_dict(), "input_dim": g["num_features"], "hidden_dim": 16,
                "output_dim": g["num_classes"], "dropout": 0.5}, path)

    loaded, _ = load_gcn_checkpoint(path, "cpu")
    assert not loaded.training
    assert torch.equal(run_inference(loaded, g["x"], coo), run_inference(model, g["x"], coo))


def test_trained_checkpoint_saves_and_reloads(tmp_path, random_graph):
    from src.gcn.config import load_config
    from src.gcn.datasets import GraphData
    from src.gcn.training import save_checkpoint, train_gcn

    g = random_graph
    n = g["num_nodes"]
    idx = torch.arange(n)
    graph = GraphData(name="toy", x=g["x"], edge_index=g["edge_index"],
                      y=torch.randint(0, g["num_classes"], (n,), generator=torch.Generator().manual_seed(1)),
                      train_mask=idx % 3 == 0, val_mask=idx % 3 == 1, test_mask=idx % 3 == 2,
                      num_nodes=n, num_features=g["num_features"], num_classes=g["num_classes"])
    config = load_config()
    config["max_epochs"] = 5
    coo = build_normalized_coo(graph.edge_index, n)
    model, checkpoint = train_gcn(graph, coo, config, log_every=0)

    assert 1 <= checkpoint["best_epoch"] <= 5
    assert checkpoint["preprocessing"] == {"undirected": True, "self_loops": True, "normalization": "symmetric"}
    path = tmp_path / "trained.pt"
    save_checkpoint(checkpoint, path)
    loaded, reloaded = load_gcn_checkpoint(path, "cpu")  # weights_only=True must accept every field
    assert reloaded["test_accuracy"] == checkpoint["test_accuracy"]
    assert torch.equal(run_inference(loaded, graph.x, coo), run_inference(model, graph.x, coo))


@pytest.mark.integration
def test_cora_end_to_end_short_training():
    """Downloads Cora on first run. Run with `pytest -m integration`."""
    from src.gcn.config import load_config
    from src.gcn.datasets import load_dataset
    from src.gcn.training import train_gcn

    graph = load_dataset("cora")
    assert graph.summary() == {
        "dataset": "cora", "num_nodes": 2708, "num_raw_edges": 10556, "num_features": 1433,
        "num_classes": 7, "num_train": 140, "num_val": 500, "num_test": 1000,
    }
    coo = build_normalized_coo(graph.edge_index, graph.num_nodes)
    assert coo._nnz() == 10556 + 2708  # Cora is already undirected with no duplicates or self-loops

    config = load_config()
    config["max_epochs"] = 20
    model, checkpoint = train_gcn(graph, coo, config, log_every=0)

    logits = _logits_per_format(model, graph.x, coo)
    for storage_format in STORAGE_FORMATS:
        assert is_equivalent(compare_to_reference(logits["coo"], logits[storage_format], graph.y, graph.test_mask))
