"""Train the dense baseline MLP for one model seed and keep the best-validation weights.

Adam, cross-entropy, mini-batches drawn from a per-epoch permutation of a
generator seeded with the model seed. The saved weights come from the epoch
with the best validation accuracy; ties go to the lower validation loss (a
later epoch must be strictly better). The test set is evaluated once, with
the selected weights, and never used for selection.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from src.gcn.config import set_seed, torch_dtype
from src.gcn.error_analysis import exact_accuracy

from .model import MLP


def evaluate(model, x, y, batch_size):
    """(logits, mean cross-entropy) in eval mode, in fixed-size chunks."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    return logits, F.cross_entropy(logits, y).item()


def train_mlp(data, config, seed, device="cpu", log_every=5):
    """Returns (model with the best weights, on CPU; checkpoint dict)."""
    device = torch.device(device)
    set_seed(seed)
    dtype = torch_dtype(config["dtype"])

    x_train, y_train = data.x_train.to(device, dtype), data.y_train.to(device)
    x_val, y_val = data.x_val.to(device, dtype), data.y_val.to(device)
    model = MLP(data.num_features, config["hidden_dims"], data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    shuffle = torch.Generator().manual_seed(seed)
    batch_size, eval_batch = config["train_batch_size"], config["eval_batch_size"]

    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(y_train), generator=shuffle).to(device)
        total, count = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            count += len(idx)

        val_logits, val_loss = evaluate(model, x_val, y_val, eval_batch)
        val_acc = exact_accuracy(val_logits, y_val, torch.ones_like(y_val, dtype=torch.bool))
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss, "val_accuracy": val_acc})
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best.update(val_acc=val_acc, val_loss=val_loss, epoch=epoch, state=copy.deepcopy(model.state_dict()))
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"seed {seed} epoch {epoch:3d}  train_loss {total / count:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}", flush=True)

    model.load_state_dict(best["state"])
    model = model.cpu().eval()
    test_logits, test_loss = evaluate(model, data.x_test, data.y_test, eval_batch)
    test_acc = exact_accuracy(test_logits, data.y_test, torch.ones_like(data.y_test, dtype=torch.bool))

    checkpoint = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "dataset": "fashion_mnist",
        "input_dim": data.num_features,
        "hidden_dims": list(config["hidden_dims"]),
        "output_dim": data.num_classes,
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["val_acc"],
        "best_validation_loss": best["val_loss"],
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "selection": "best validation accuracy; ties -> lower validation loss",
        "data": data.summary(),
        "training": {key: config[key] for key in ("epochs", "train_batch_size", "learning_rate",
                                                  "weight_decay", "dtype", "tf32")},
        "history": history,
        "train_device": str(device),
        "torch_version": str(torch.__version__),
    }
    return model, checkpoint


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
