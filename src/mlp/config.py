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


FOLLOWUP_CONFIG_PATH = PROJECT_ROOT / "configs" / "mlp_followup.json"
FOLLOWUP_KEYS = {
    "base_config", "experiments", "statistical_k", "recovery_sparsities", "recovery_epochs",
    "recovery_learning_rates", "recovery_batch_size", "protected_layers", "lora_rank", "lora_alpha", "lora_layers",
}
FOLLOWUP_EXPERIMENTS = ("statistical", "finetune", "protect_fc3", "lora", "int8")


def load_followup_config(path=FOLLOWUP_CONFIG_PATH):
    """The follow-up experiments' settings on top of the original experiment's (base) config.

    Every setting of the original experiment (data split, batch sizes, timing,
    precision, pruned layers) is inherited unchanged; a follow-up key may not
    override a base key.
    """
    with open(path) as f:
        followup = json.load(f)
    missing = FOLLOWUP_KEYS - followup.keys()
    unknown = followup.keys() - FOLLOWUP_KEYS
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    if set(followup["experiments"]) - set(FOLLOWUP_EXPERIMENTS):
        raise ValueError(f"{path}: experiments must be among {FOLLOWUP_EXPERIMENTS}")
    if not all(k >= 0 for k in followup["statistical_k"]):
        raise ValueError(f"{path}: statistical_k must be >= 0")
    base = load_config(PROJECT_ROOT / followup["base_config"])
    if followup.keys() & base.keys():
        raise ValueError(f"{path}: may not override base settings {sorted(followup.keys() & base.keys())}")
    if set(followup["recovery_sparsities"]) - set(base["sparsities"]):
        raise ValueError(f"{path}: recovery_sparsities must be levels of the original sweep (reused as baselines)")
    return {**base, **followup}


COMPARISON_CONFIG_PATH = PROJECT_ROOT / "configs" / "mlp_gpu_comparison.json"
COMPARISON_KEYS = {
    "base_config", "pruned_formats", "low_rank_ranks", "low_rank_rules", "low_rank_formats", "energy_seconds",
    "energy_idle_seconds", "energy_preroll_seconds", "energy_sample_interval_ms",
}


def load_comparison_config(path=COMPARISON_CONFIG_PATH):
    """The GPU format comparison (adjacency list, low-rank factors, energy) on top of the base config.

    Every setting of the original experiment (checkpoints' seeds, pruning grid,
    batch sizes, warm-ups, timed passes, repetitions, precision) is inherited
    unchanged; a comparison key may not override a base key.
    """
    from .formats import ALL_FORMATS
    from .low_rank import FACTOR_FORMATS, FACTOR_RULES

    with open(path) as f:
        comparison = json.load(f)
    missing = COMPARISON_KEYS - comparison.keys()
    unknown = comparison.keys() - COMPARISON_KEYS
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    if set(comparison["pruned_formats"]) - set(ALL_FORMATS) or "dense" not in comparison["pruned_formats"]:
        raise ValueError(f"{path}: pruned_formats must be among {ALL_FORMATS} and include dense (the reference)")
    if (set(comparison["low_rank_formats"]) - set(FACTOR_FORMATS)
            or "factor_dense" not in comparison["low_rank_formats"]):
        raise ValueError(f"{path}: low_rank_formats must be among {FACTOR_FORMATS} and include factor_dense")
    for rule in comparison["low_rank_rules"]:
        if rule.get("rule") not in FACTOR_RULES or (rule["rule"] == "abs_std") != ("k" in rule):
            raise ValueError(f"{path}: low_rank_rules entries are {{'rule': 'all'}} or {{'rule': 'abs_std', 'k': K}}")
    if not all(isinstance(r, int) and r > 0 for r in comparison["low_rank_ranks"]):
        raise ValueError(f"{path}: low_rank_ranks must be positive integers")
    if comparison["energy_seconds"] <= 0 or comparison["energy_idle_seconds"] <= 0:
        raise ValueError(f"{path}: energy durations must be positive")
    base = load_config(PROJECT_ROOT / comparison["base_config"])
    if comparison.keys() & base.keys():
        raise ValueError(f"{path}: may not override base settings {sorted(comparison.keys() & base.keys())}")
    return {**base, **comparison}
