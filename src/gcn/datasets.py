"""Dataset loaders that return a format-agnostic GraphData.

The GCN code only depends on GraphData, never on PyG/Cora-specific objects,
so new sources (the partner's synthetic generator, OGB datasets) only need a
new loader registered in LOADERS.
"""

from dataclasses import dataclass

import torch

from .config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "data"


@dataclass
class GraphData:
    name: str
    x: torch.Tensor           # [num_nodes, num_features] float
    edge_index: torch.Tensor  # [2, num_raw_edges] long, PyG order: [source, target]
    y: torch.Tensor           # [num_nodes] long
    train_mask: torch.Tensor  # [num_nodes] bool
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_nodes: int
    num_features: int
    num_classes: int

    def summary(self):
        return {
            "dataset": self.name,
            "num_nodes": self.num_nodes,
            "num_raw_edges": int(self.edge_index.size(1)),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "num_train": int(self.train_mask.sum()),
            "num_val": int(self.val_mask.sum()),
            "num_test": int(self.test_mask.sum()),
        }


def load_cora(root=DATA_ROOT):
    """Cora (Planetoid public split) with row-normalized node features.

    Downloads to data/Planetoid/Cora on first use.
    """
    # Imported lazily so the offline unit tests don't need PyG at import time.
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(root=str(root / "Planetoid"), name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    return GraphData(
        name="cora",
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_nodes=data.num_nodes,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    )


# Extension point: add "synthetic" (adapter over src/synthetic/generator.py)
# and "ogbn-arxiv" loaders here.
LOADERS = {
    "cora": load_cora,
}


def load_dataset(name):
    try:
        loader = LOADERS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r} (available: {sorted(LOADERS)})") from None
    return loader()


def print_summary(graph):
    s = graph.summary()
    print(f"Dataset {s['dataset']}: {s['num_nodes']} nodes, {s['num_raw_edges']} raw edges "
          f"(directed edge_index entries), {s['num_features']} features, {s['num_classes']} classes")
    print(f"  split: {s['num_train']} train / {s['num_val']} val / {s['num_test']} test")
