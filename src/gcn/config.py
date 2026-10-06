"""Experiment configuration and seeding.

The config is a flat JSON file (see configs/gcn_baseline.json). Keys are
validated so a typo fails loudly instead of silently using a default.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "gcn_baseline.json"

REQUIRED_KEYS = {
    "hidden_dim", "dropout", "learning_rate", "weight_decay", "max_epochs",
    "seed", "dtype", "tf32", "warmup_runs", "measured_runs", "repetitions",
}

# Only FP32 is in scope for now. Low-precision dtypes are a later experiment
# and get added here once the model/benchmark paths are checked for them.
SUPPORTED_DTYPES = {"float32": torch.float32}


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path) as f:
        config = json.load(f)

    missing = REQUIRED_KEYS - config.keys()
    unknown = config.keys() - REQUIRED_KEYS
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    if config["dtype"] not in SUPPORTED_DTYPES:
        raise ValueError(f"{path}: dtype {config['dtype']!r} not supported yet "
                         f"(supported: {sorted(SUPPORTED_DTYPES)})")
    return config


def torch_dtype(name):
    return SUPPORTED_DTYPES[name]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
