"""Train one GCN checkpoint (with COO adjacency) and save it.

Training time and memory are not part of the inference experiment, so
nothing here is timed or profiled.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from .adjacency import layout_name
from .config import set_seed, torch_dtype
from .evaluation import accuracy
from .model import GCN

PREPROCESSING = {"undirected": True, "self_loops": True, "normalization": "symmetric"}


def train_gcn(graph, adj, config, device="cpu", log_every=20):
    """Train with Adam and keep the weights with the best validation accuracy.

    Ties on validation accuracy are broken by lower validation loss; a later
    epoch must be strictly better to replace the current best.

    Returns (model with best weights, checkpoint dict ready for save_checkpoint).
    """
    device = torch.device(device)
    set_seed(config["seed"])

    dtype = torch_dtype(config["dtype"])
    x = graph.x.to(device=device, dtype=dtype)
    y = graph.y.to(device)
    train_mask, val_mask, test_mask = (m.to(device) for m in (graph.train_mask, graph.val_mask, graph.test_mask))
    adj = adj.to(device)

    model = GCN(graph.num_features, config["hidden_dim"], graph.num_classes, config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])

    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": -1, "state": None}
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            val_loss = F.cross_entropy(logits[val_mask], y[val_mask]).item()
            val_acc = accuracy(logits, y, val_mask)

        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best.update(val_acc=val_acc, val_loss=val_loss, epoch=epoch,
                        state=copy.deepcopy(model.state_dict()))

        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"epoch {epoch:3d}  train_loss {loss.item():.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        test_acc = accuracy(model(x, adj), y, test_mask)

    checkpoint = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": graph.num_features,
        "hidden_dim": config["hidden_dim"],
        "output_dim": graph.num_classes,
        "dropout": config["dropout"],
        "seed": config["seed"],
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["val_acc"],
        "test_accuracy": test_acc,
        "preprocessing": dict(PREPROCESSING),
        # Extra provenance, not used to rebuild the model.
        "dataset": graph.name,
        "train_adjacency_format": layout_name(adj),
        "dtype": config["dtype"],
        "torch_version": str(torch.__version__),  # plain str so weights_only loading accepts it
    }
    return model, checkpoint


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
