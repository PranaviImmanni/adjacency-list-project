"""Train the dense Fashion-MNIST MLP baseline for one model seed.

Saves the checkpoint (best validation accuracy) and a baseline result JSON
with the configuration, data split, training history and test accuracy.

Example:
    python scripts/train_mlp.py --seed 42 --device cuda:1
    # -> checkpoints/fashion_mlp_seed42_cuda1.pt, results/mlp/cuda1/baseline/fashion_mlp_seed42.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.gcn.measurement import configure_precision, device_info, resolve_device
from src.mlp.config import DEFAULT_CONFIG_PATH, load_config
from src.mlp.data import load_fashion_mnist
from src.mlp.model import checkpoint_identity
from src.mlp.training import save_checkpoint, train_mlp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="model seed")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=None, help="checkpoint (default: checkpoints/fashion_mlp_seed<S>_<device>.pt)")
    parser.add_argument("--results-dir", default=None, help="default: results/mlp/<device>/baseline")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    precision = configure_precision(device, config["tf32"])
    tag = str(device).replace(":", "")

    data = load_fashion_mnist(config["val_size"], config["split_seed"])
    s = data.summary()
    print(f"Fashion-MNIST: {s['num_train']} train / {s['num_val']} val / {s['num_test']} test, "
          f"{s['num_features']} features, {s['num_classes']} classes (split seed {s['split_seed']})")

    _, checkpoint = train_mlp(data, config, args.seed, device)
    checkpoint["precision"] = precision
    output = Path(args.output) if args.output else PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{args.seed}_{tag}.pt"
    save_checkpoint(checkpoint, output)

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results" / "mlp" / tag / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"fashion_mlp_seed{args.seed}.json"
    with open(result_path, "w") as f:
        json.dump({"experiment": "mlp_baseline", "status": "ok",
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "checkpoint": checkpoint_identity(output, checkpoint),
                   "model_seed": args.seed, "best_epoch": checkpoint["best_epoch"],
                   "validation_accuracy": checkpoint["best_validation_accuracy"],
                   "test_accuracy": checkpoint["test_accuracy"], "data": checkpoint["data"],
                   "training": checkpoint["training"], "history": checkpoint["history"],
                   "config": config, **device_info(device), **precision}, f, indent=2)

    print(f"Seed {args.seed}: best epoch {checkpoint['best_epoch']}, val acc "
          f"{checkpoint['best_validation_accuracy']:.4f}, test acc {checkpoint['test_accuracy']:.4f}")
    print(f"Saved {output} and {result_path}")


if __name__ == "__main__":
    main()
