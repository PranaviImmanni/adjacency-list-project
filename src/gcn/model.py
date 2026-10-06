"""Two-layer GCN that runs unchanged on dense, COO, or CSR adjacency.

    H = ReLU(A_hat (X W1) + b1)
    H = Dropout(H)            (training only)
    Z = A_hat (H W2) + b2     (class logits)

The feature transform is applied before aggregation for every format, and
aggregation always goes through adjacency_multiply, so the only thing that
changes between runs is the adjacency layout. PyG's GCNConv is deliberately
not used, to keep that comparison controlled.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .adjacency import adjacency_multiply


class GCNLayer(nn.Module):
    """A_hat (X W) + b. The bias is added after aggregation, as in Kipf & Welling and PyG's GCNConv."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        return adjacency_multiply(adj, self.linear(x)) + self.bias


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.conv1 = GCNLayer(input_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        h = F.relu(self.conv1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, adj)


def load_gcn_checkpoint(path, device="cpu"):
    """Rebuild the model from a checkpoint, loading on CPU and then moving to `device`.

    Returns (model, checkpoint_dict). The model is in eval mode.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GCN(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(torch.device(device)), checkpoint
