"""Export the trained MLPs, test set and pruning masks as one numpy bundle.

The bundle is for benchmarking other implementations (e.g. the Python
adjacency list) against the committed dense / dense+mask / COO / CSR results.
Reading it needs only numpy; see src/mlp/export.py for the contents.

Runs on CPU. After writing, the bundle is reloaded with numpy alone and every
model and condition is recomputed from it. With results present, every
checkpoint and mask hash must match the evaluate JSONs behind the results.

Example:
    python scripts/export_mlp_bundle.py --run cuda1
      reads   checkpoints/fashion_mlp_seed{42,43,44}_cuda1.pt
      checks  results/mlp/cuda1/evaluate/
      writes  exports/fashion_mlp_cuda1.npz
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.mlp.config import DEFAULT_CONFIG_PATH, load_config
from src.mlp.data import load_fashion_mnist
from src.mlp.export import build_bundle, match_results, verify_bundle, with_training_normalization
from src.mlp.model import load_mlp_checkpoint


def git_source():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        return {"git_commit": commit, "git_branch": branch}
    except (OSError, subprocess.CalledProcessError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cuda1", help="device tag of the checkpoints and results (default cuda1)")
    parser.add_argument("--seeds", type=int, nargs="+", help="model seeds (default: the config's)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--results-dir", help="default results/mlp/<run>")
    parser.add_argument("--output", help="default exports/fashion_mlp_<run>.npz")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = args.seeds or config["model_seeds"]
    paths = [PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{seed}_{args.run}.pt" for seed in seeds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit("missing checkpoint(s):\n  " + "\n  ".join(missing))
    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results" / "mlp" / args.run
    output = Path(args.output) if args.output else PROJECT_ROOT / "exports" / f"fashion_mlp_{args.run}.npz"

    models = [(path, *load_mlp_checkpoint(path)) for path in paths]
    # Test inputs exactly as the training machine standardized them (see with_training_normalization).
    data = with_training_normalization(load_fashion_mnist(config["val_size"], config["split_seed"]), models[0][2])
    local = load_fashion_mnist(config["val_size"], config["split_seed"]).x_test
    changed = int((local != data.x_test).sum())
    print(f"Test inputs rebuilt with the checkpoints' normalization constants "
          f"({changed} of {local.numel():,} float32 values differ from this machine's own standardization)")
    print(f"Exporting seeds {seeds} from {args.run} checkpoints ...", flush=True)
    arrays = build_bundle(models, data, config, source={**git_source(), "run": args.run})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(arrays)} arrays)")

    problems = verify_bundle(output)
    print(f"Reloaded with numpy only and recomputed every model and condition: "
          f"{'OK' if not problems else f'{len(problems)} problem(s)'}")

    metadata = json.loads(str(arrays["metadata"]))
    if (results_dir / "evaluate").is_dir():
        checked, mismatches, accuracy_diffs = match_results(metadata, results_dir)
        problems += mismatches
        print(f"Matched against {results_dir}: {checked} conditions, "
              f"{'all checkpoint and mask hashes agree' if not mismatches else f'{len(mismatches)} mismatch(es)'}")
        for tag, recorded, here in accuracy_diffs:
            print(f"  note: {tag} test accuracy {recorded:.4f} in the results, {here:.4f} on CPU here")
    else:
        print(f"No results at {results_dir}; skipped matching against them.")

    for problem in problems:
        print(f"  PROBLEM: {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
