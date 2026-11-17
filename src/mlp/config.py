"""Configuration for the MLP weight-pruning experiment (configs/mlp_fashion_mnist.json).

Keys are validated like the GCN config, so a typo fails loudly. Seeding,
dtype and device helpers are shared with the GCN code.
"""

import json

from src.gcn.config import PROJECT_ROOT, SUPPORTED_DTYPES

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mlp_fashion_mnist.json"

REQUIRED_KEYS = {
    "dataset", "hidden_dims", "val_size", "split_seed", "model_seeds", "epochs", "train_batch_size",
    "learning_rate", "weight_decay", "pruned_layers", "methods", "sparsities", "random_pruning_seeds",
    "benchmark_random_seeds", "batch_sizes", "eval_batch_size", "dtype", "tf32", "warmup_runs",
    "measured_runs", "repetitions",
}


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path) as f:
        config = json.load(f)

    missing = REQUIRED_KEYS - config.keys()
    unknown = config.keys() - REQUIRED_KEYS
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    if config["dtype"] not in SUPPORTED_DTYPES:
        raise ValueError(f"{path}: dtype {config['dtype']!r} not supported (supported: {sorted(SUPPORTED_DTYPES)})")
    if not set(config["benchmark_random_seeds"]) <= set(config["random_pruning_seeds"]):
        raise ValueError(f"{path}: benchmark_random_seeds must be a subset of random_pruning_seeds")
    if not all(0 < s < 1 for s in config["sparsities"]):
        raise ValueError(f"{path}: sparsities must be in (0, 1)")
    return config
