"""Run the MLP weight-pruning experiment in stages.

Stages (settings fixed in configs/mlp_fashion_mnist.json before any result is seen):
  pilot      model seed 42 only: train it, evaluate the original model and
             magnitude 90%, benchmark the original (dense) and magnitude 90% in
             every format at every batch size, 1 repetition. Checks the whole
             pipeline before the grid.
  train      train every model seed (one process each).
  evaluate   every model seed x every condition: accuracy, pruning effect and
             format epsilon (one process per model seed).
  benchmark  every repetition x model seed x condition x format x batch size,
             one scripts/benchmark_mlp.py process each. Random pruning uses only
             `benchmark_random_seeds`. The repetition is the outer loop.

Finished outputs are skipped, so rerunning a command resumes it.

Example (lab GPU):
    python scripts/run_mlp_experiments.py --stage pilot --device cuda:1
    python scripts/run_mlp_experiments.py --stage train --device cuda:1
    python scripts/run_mlp_experiments.py --stage evaluate --device cuda:1
    python scripts/run_mlp_experiments.py --stage benchmark --device cuda:1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.mlp.config import DEFAULT_CONFIG_PATH, load_config
from src.mlp.experiment import ORIGINAL, condition_tag, conditions
from src.mlp.formats import FORMATS

SCRIPTS = Path(__file__).resolve().parent
PILOT_SEED = 42
PILOT_CONDITION = ("magnitude", 0.9, None)


def finished(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status", "ok") == "ok"
    except json.JSONDecodeError:
        return False


def checkpoint_path(seed, device_tag):
    return PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{seed}_{device_tag}.pt"


def run(command):
    print("  $", " ".join(str(c) for c in command[1:]), flush=True)
    return subprocess.run([str(c) for c in command]).returncode


def train(seeds, args, device_tag, results_root):
    failed = []
    for seed in seeds:
        path = checkpoint_path(seed, device_tag)
        if path.exists():
            print(f"checkpoint exists, skipping training: {path.name}")
            continue
        if run([sys.executable, SCRIPTS / "train_mlp.py", "--seed", seed, "--device", args.device,
                "--config", args.config, "--output", path, "--results-dir", results_root / "baseline"]) != 0:
            failed.append(f"train seed {seed}")
    return failed


def evaluate(seeds, args, device_tag, output_dir, pilot):
    failed = []
    for seed in seeds:
        command = [sys.executable, SCRIPTS / "evaluate_mlp_pruning.py", "--checkpoint",
                   checkpoint_path(seed, device_tag), "--device", args.device, "--config", args.config,
                   "--output-dir", output_dir]
        if pilot:
            command += ["--conditions", "pilot"]
        if run(command) != 0:
            failed.append(f"evaluate seed {seed}")
    return failed


def benchmark(runs, args, device_tag, output_dir):
    failed = []
    for i, (rep, seed, (method, sparsity, prune_seed), storage_format, batch) in enumerate(runs, 1):
        tag = condition_tag(seed, method, sparsity, prune_seed)
        output = output_dir / f"{tag}_{storage_format}_b{batch}_{device_tag}_rep{rep}.json"
        if finished(output):
            continue
        print(f"[{i}/{len(runs)}] ", end="", flush=True)
        command = [sys.executable, SCRIPTS / "benchmark_mlp.py", "--checkpoint", checkpoint_path(seed, device_tag),
                   "--method", method, "--sparsity", sparsity, "--format", storage_format, "--batch-size", batch,
                   "--device", args.device, "--config", args.config, "--repetition", rep, "--output", output]
        if prune_seed is not None:
            command += ["--prune-seed", prune_seed]
        if subprocess.run([str(c) for c in command]).returncode != 0:
            failed.append(output.name)
    return failed


def benchmark_grid(config, seeds, condition_list, repetitions):
    runs = []
    for rep in range(repetitions):
        for seed in seeds:
            for condition in condition_list:
                formats = ["dense"] if condition == ORIGINAL else list(FORMATS)
                for storage_format in formats:
                    for batch in config["batch_sizes"]:
                        runs.append((rep, seed, condition, storage_format, batch))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["pilot", "train", "evaluate", "benchmark"])
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: model_seeds in the config")
    parser.add_argument("--repetitions", type=int, default=None, help="default: 'repetitions' in the config")
    parser.add_argument("--results-dir", default=None,
                        help="default: results/mlp/<device> (results/mlp/pilot_<device> for the pilot)")
    args = parser.parse_args()

    config = load_config(args.config)
    device_tag = args.device.replace(":", "")
    pilot = args.stage == "pilot"
    seeds = [PILOT_SEED] if pilot else (args.seeds or config["model_seeds"])
    results_root = Path(args.results_dir) if args.results_dir else (
        PROJECT_ROOT / "results" / "mlp" / (f"pilot_{device_tag}" if pilot else device_tag))
    failed = []

    if args.stage in ("pilot", "train"):
        failed += train(seeds, args, device_tag, PROJECT_ROOT / "results" / "mlp" / device_tag)
    missing = [seed for seed in seeds if not checkpoint_path(seed, device_tag).exists()]
    if args.stage != "train" and missing:
        raise SystemExit(f"missing checkpoints for seeds {missing}: run --stage train first")
    if args.stage in ("pilot", "evaluate"):
        failed += evaluate(seeds, args, device_tag, results_root / "evaluate", pilot)
    if args.stage in ("pilot", "benchmark"):
        condition_list = [ORIGINAL, PILOT_CONDITION] if pilot else conditions(config, purpose="benchmark")
        repetitions = 1 if pilot else (args.repetitions or config["repetitions"])
        runs = benchmark_grid(config, seeds, condition_list, repetitions)
        print(f"benchmark on {args.device}: {len(runs)} runs -> {results_root / 'benchmark'}")
        failed += benchmark(runs, args, device_tag, results_root / "benchmark")

    print(f"\nDone: {len(failed)} failed.")
    for name in failed:
        print(f"  FAILED: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
