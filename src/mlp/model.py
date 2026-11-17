"""The one fixed MLP architecture and its checkpoints.

784 -> 300 -> 100 -> 10 with ReLU (LeNet-300-100, the standard fully
connected pruning benchmark). Layers are named fc1, fc2, fc3 and every one
has a bias. There is no dropout, so training and inference compute the same
function apart from the optimizer.
"""

import hashlib

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        dims = [input_dim, *self.hidden_dims, output_dim]
        self.layer_names = [f"fc{i + 1}" for i in range(len(dims) - 1)]
        for name, (d_in, d_out) in zip(self.layer_names, zip(dims[:-1], dims[1:])):
            self.add_module(name, nn.Linear(d_in, d_out))

    def layers(self):
        return [getattr(self, name) for name in self.layer_names]

    def forward(self, x):
        layers = self.layers()
        for layer in layers[:-1]:
            x = F.relu(layer(x))
        return layers[-1](x)


def build_model(checkpoint):
    model = MLP(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["output_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def load_mlp_checkpoint(path):
    """Load on CPU only. Returns (model in eval mode on CPU, checkpoint dict).

    Callers move the representation they need to the device themselves, so
    no dense copy of the weights reaches the GPU by accident.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return build_model(checkpoint), checkpoint


def checkpoint_identity(path, checkpoint):
    """What a result file needs to identify the frozen checkpoint it used."""
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    return {
        "path": str(path),
        "sha256": digest,
        "dataset": checkpoint.get("dataset"),
        "model_seed": checkpoint["seed"],
        "architecture": [checkpoint["input_dim"], *checkpoint["hidden_dims"], checkpoint["output_dim"]],
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_accuracy": checkpoint["best_validation_accuracy"],
        "saved_test_accuracy": checkpoint["test_accuracy"],
        "train_device": checkpoint.get("train_device"),
    }
