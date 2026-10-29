"""Run the pruning experiments for one dataset with its frozen checkpoint.

Stages (settings fixed in advance, see docs/pruning_and_error_analysis.md):
  pilot   original graph, random (seed 101) and similarity at 50% removal; CSR, 1 repetition
  full    original graph once, random at 10/25/50/75% x seeds 101/102/103,
          similarity at 10/25/50/75%; CSR, 3 repetitions -> 17 conditions x 3 = 51 runs
  verify  dense/COO/CSR equivalence (COO on the same device as reference) on the
          original graph, similarity 50% and random seed 101 at 50%

Each benchmark run is a separate scripts/benchmark_pruning.py process. The
repetition is the outer loop, so every repetition is one pass over all
conditions and slow drift on the machine does not favor one condition.
Finished runs are skipped, so rerunning the same command resumes a stage.

Example:
    python scripts/run_pruning_experiments.py --dataset cora --stage pilot --device cuda:1
    python scripts/run_pruning_experiments.py --dataset cora --stage full --device cuda:1
    python scripts/run_pruning_experiments.py --dataset cora --stage verify --device cuda:1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from src.gcn.pruning import RANDOM_SEEDS, REMOVAL_FRACTIONS

SCRIPTS = Path(__file__).resolve().parent
PRUNED_FRACTIONS = [f for f in REMOVAL_FRACTIONS if f > 0]

# Each condition is (method, pruning seed or None, removal fraction).
BASELINE = ("none", None, 0.0)
STAGES = {
    "pilot": [BASELINE, ("random", 101, 0.5), ("similarity", None, 0.5)],
    "full": [BASELINE,
             *[("random", seed, f) for seed in RANDOM_SEEDS for f in PRUNED_FRACTIONS],
             *[("similarity", None, f) for f in PRUNED_FRACTIONS]],
    "verify": [BASELINE, ("similarity", None, 0.5), ("random", 101, 0.5)],
}


def condition_tag(dataset, method, seed, fraction):
    """e.g. cora_none_f0, cora_random_s101_f50, pubmed_similarity_f75."""
    return f"{dataset}_{method}" + (f"_s{seed}" if seed is not None else "") + f"_f{round(100 * fraction)}"


def finished(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status", "ok") == "ok"
    except json.JSONDecodeError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["cora", "pubmed"])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint", default=None, help="default: checkpoints/<dataset>_seed42_gpu.pt")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--repetitions", type=int, default=None,
                        help="default: 1 for the pilot, 'repetitions' in the config for the full stage")
    parser.add_argument("--output-dir", default=None,
                        help="default: results/pruning/<dataset>_<device> (pilot_/verify_ prefix for those stages)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or f"checkpoints/{args.dataset}_seed42_gpu.pt"
    if not (Path(checkpoint).exists() or (PROJECT_ROOT / checkpoint).exists()):
        raise SystemExit(f"checkpoint {checkpoint} not found; train it first (scripts/train_gcn.py)")
    device_tag = args.device.replace(":", "")
    prefix = "" if args.stage == "full" else f"{args.stage}_"
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "pruning" / f"{prefix}{args.dataset}_{device_tag}")
    conditions = STAGES[args.stage]
    failed, not_equivalent = [], []

    if args.stage == "verify":
        print(f"verify on {args.device}: {len(conditions)} graphs x dense/coo/csr -> {output_dir}")
        for i, (method, seed, fraction) in enumerate(conditions, 1):
            tag = condition_tag(args.dataset, method, seed, fraction)
            output = output_dir / f"{tag}_formats_{device_tag}.json"
            if output.exists():
                print(f"[{i}/{len(conditions)}] exists, skipping {output.name}")
                continue
            print(f"[{i}/{len(conditions)}] {tag}", flush=True)
            command = [sys.executable, str(SCRIPTS / "compare_gcn_formats.py"), "--dataset", args.dataset,
                       "--device", args.device, "--checkpoint", checkpoint, "--config", args.config,
                       "--prune-method", method, "--prune-fraction", str(fraction),
                       "--output", str(output), "--logits-file", str(output_dir / "logits" / f"{tag}_formats.pt")]
            if seed is not None:
                command += ["--prune-seed", str(seed)]
            code = subprocess.run(command).returncode
            if not output.exists():
                failed.append(output.name)
            elif code != 0:
                not_equivalent.append(output.name)
    else:
        repetitions = args.repetitions or (1 if args.stage == "pilot" else load_config(args.config)["repetitions"])
        runs = [(rep, condition) for rep in range(repetitions) for condition in conditions]
        print(f"{args.dataset} {args.stage} on {args.device}: {len(runs)} runs (CSR) -> {output_dir}")
        for i, (rep, (method, seed, fraction)) in enumerate(runs, 1):
            tag = condition_tag(args.dataset, method, seed, fraction)
            output = output_dir / f"{tag}_csr_{device_tag}_rep{rep}.json"
            if finished(output):
                print(f"[{i}/{len(runs)}] exists, skipping {output.name}")
                continue
            print(f"[{i}/{len(runs)}] ", end="", flush=True)
            command = [sys.executable, str(SCRIPTS / "benchmark_pruning.py"), "--dataset", args.dataset,
                       "--checkpoint", checkpoint, "--method", method, "--fraction", str(fraction),
                       "--format", "csr", "--device", args.device, "--config", args.config,
                       "--repetition", str(rep), "--orderings-dir", str(output_dir / "orderings"),
                       "--output", str(output)]
            if seed is not None:
                command += ["--prune-seed", str(seed)]
            if rep == 0:  # predictions are identical across repetitions; logits differ by ~1e-7 at most on GPU
                command += ["--logits-out", str(output_dir / "logits" / f"{tag}_csr.pt")]
            if subprocess.run(command).returncode != 0:
                failed.append(output.name)
                continue
            if not json.loads(output.read_text())["format_check"]["equivalent"]:
                not_equivalent.append(output.name)

    print(f"\nDone: {len(failed)} failed, {len(not_equivalent)} not equivalent across formats.")
    for name in failed:
        print(f"  FAILED: {name}")
    for name in not_equivalent:
        print(f"  NOT EQUIVALENT: {name}")
    sys.exit(1 if failed or not_equivalent else 0)


if __name__ == "__main__":
    main()
