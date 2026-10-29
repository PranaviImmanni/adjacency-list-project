"""Dataset loaders that return a format-agnostic GraphData.

The GCN code only depends on GraphData, never on PyG/Cora-specific objects,
so new sources (e.g. OGB datasets) only need a
new loader registered in LOADERS.

Cora and PubMed use the public Planetoid split. DATASET_METADATA records what
each dataset actually provides; `available_fields` is read from the loaded
data at run time. Anything not listed there (publication years, edge
weights or confidence, raw text) does not exist in these datasets.
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
    available_fields: tuple = ()  # attributes present in the source data object

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

    def metadata(self):
        """summary() plus the fields present in the source data and what they contain."""
        return {**self.summary(), "available_fields": list(self.available_fields),
                **DATASET_METADATA.get(self.name, {})}


DATASET_METADATA = {
    "cora": {
        "source": "Planetoid (Yang et al., 2016), public split",
        "node_features": "binary bag-of-words over a 1,433-word vocabulary, row-normalized",
        "classes": "7 machine-learning paper topics",
        "edges": "citations, unweighted; stored directed in both directions",
        "not_available": "publication years, edge weights or confidence, raw text",
    },
    "pubmed": {
        "source": "Planetoid (Yang et al., 2016), public split",
        "node_features": "TF-IDF weights over a 500-word vocabulary, row-normalized",
        "classes": "3 diabetes categories of PubMed papers",
        "edges": "citations, unweighted; stored directed in both directions",
        "not_available": "publication years, edge weights or confidence, raw text",
    },
}


def load_planetoid(name, pyg_name, root=DATA_ROOT):
    """A Planetoid dataset (public split) with row-normalized node features.

    Downloads to data/Planetoid/<pyg_name> on first use. Feature and class
    counts come from the dataset, not from constants.
    """
    # Imported lazily so the offline unit tests don't need PyG at import time.
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(root=str(root / "Planetoid"), name=pyg_name, split="public", transform=NormalizeFeatures())
    data = dataset[0]
    return GraphData(
        name=name,
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_nodes=data.num_nodes,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        available_fields=tuple(sorted(data.keys())),
    )


def load_cora(root=DATA_ROOT):
    return load_planetoid("cora", "Cora", root)


def load_pubmed(root=DATA_ROOT):
    return load_planetoid("pubmed", "PubMed", root)


# Extension point: add "ogbn-arxiv" here. Synthetic graphs are unlabeled and
# built directly by scripts/benchmark_synthetic.py (src/synthetic/edge_generator.py).
LOADERS = {
    "cora": load_cora,
    "pubmed": load_pubmed,
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
