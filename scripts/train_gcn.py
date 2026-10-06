"""Train one two-layer GCN checkpoint with the canonical COO adjacency.

Example:
    python scripts/train_gcn.py --dataset cora --device cpu --output checkpoints/cora_seed42.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.adjacency import build_normalized_coo
from src.gcn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, torch_dtype
from src.gcn.datasets import load_dataset, print_summary
from src.gcn.measurement import check_format_supported, configure_precision, resolve_device
from src.gcn.training import save_checkpoint, train_gcn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--output", default=None,
                        help="checkpoint path (default: checkpoints/<dataset>_seed<seed>.pt)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    device = resolve_device(args.device)
    check_format_supported(device, "coo")
    configure_precision(device, config["tf32"])

    graph = load_dataset(args.dataset)
    print_summary(graph)
    adj = build_normalized_coo(graph.edge_index, graph.num_nodes, torch_dtype(config["dtype"]))
    print(f"Normalized adjacency (COO): {adj._nnz()} nonzeros, including {graph.num_nodes} self-loops")

    _, checkpoint = train_gcn(graph, adj, config, device)
    checkpoint["config"] = config

    output = Path(args.output) if args.output else PROJECT_ROOT / "checkpoints" / f"{args.dataset}_seed{config['seed']}.pt"
    save_checkpoint(checkpoint, output)
    print(f"Best epoch {checkpoint['best_epoch']}: val acc {checkpoint['best_validation_accuracy']:.4f}, "
          f"test acc {checkpoint['test_accuracy']:.4f}")
    print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
