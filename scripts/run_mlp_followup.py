"""Run the follow-up MLP experiments in stages.

Experiments (configs/mlp_followup.json): statistical, finetune, protect_fc3, lora, int8.
Stages:
  variants   make the variants (pruning; recovery training on --device), one process per seed x experiment
  evaluate   accuracy and equivalence of every variant, one process per seed
  benchmark  repetition x seed x variant x deployment format x batch size, one process each
  all        the three in order

The original dense model and the pruning-only magnitude 80% / 90% models are
reused from the original sweep (results/mlp/<run>), not rerun.

Finished outputs are skipped, so rerunning a command resumes it. --seeds is
required: run seed 42 first, check it, then 43 and 44.

Example (lab GPU):
    python scripts/run_mlp_followup.py --experiments statistical --stage all --seeds 42 --device cuda:1
    python scripts/run_mlp_followup.py --experiments statistical --stage all --seeds 43 44 --device cuda:1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.mlp.config import FOLLOWUP_CONFIG_PATH, FOLLOWUP_EXPERIMENTS, load_followup_config
from src.mlp.variants import variant_plan

SCRIPTS = Path(__file__).resolve().parent


def finished(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status", "ok") == "ok"
    except json.JSONDecodeError:
        return False


def run(command):
    print("  $", " ".join(str(c) for c in command[1:]), flush=True)
    return subprocess.run([str(c) for c in command]).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", choices=FOLLOWUP_EXPERIMENTS, default=None,
                        help="default: all, in config order")
    parser.add_argument("--stage", required=True, choices=["variants", "evaluate", "benchmark", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint-run", default=None,
                        help="trained checkpoints to start from (default: the device tag, e.g. cuda1)")
    parser.add_argument("--repetitions", type=int, default=None, help="default: 'repetitions' in the base config")
    parser.add_argument("--config", default=str(FOLLOWUP_CONFIG_PATH))
    args = parser.parse_args()

    config = load_followup_config(args.config)
    experiments = args.experiments or config["experiments"]
    run_tag = args.device.replace(":", "")
    checkpoint_run = args.checkpoint_run or run_tag
    missing = [s for s in args.seeds
               if not (PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{s}_{checkpoint_run}.pt").exists()]
    if missing:
        sys.exit(f"missing trained checkpoints for seeds {missing} (checkpoints/fashion_mlp_seed<S>_{checkpoint_run}.pt)")
    common = ["--device", args.device, "--checkpoint-run", checkpoint_run, "--config", args.config]
    variants_dir = PROJECT_ROOT / "checkpoints" / "variants" / run_tag
    benchmark_dir = PROJECT_ROOT / "results" / "mlp_followup" / run_tag / "benchmark"
    failed = []

    if args.stage in ("variants", "all"):
        for seed in args.seeds:
            for experiment in experiments:
                if run([sys.executable, SCRIPTS / "make_mlp_variants.py", "--experiment", experiment,
                        "--seed", seed, *common]) != 0:
                    failed.append(f"variants {experiment} seed {seed}")

    if args.stage in ("evaluate", "all"):
        for seed in args.seeds:
            if run([sys.executable, SCRIPTS / "evaluate_mlp_variants.py", "--seed", seed,
                    "--experiments", *experiments, *common]) != 0:
                failed.append(f"evaluate seed {seed}")

    if args.stage in ("benchmark", "all"):
        repetitions = args.repetitions or config["repetitions"]
        runs = [(rep, spec, storage_format, batch)
                for rep in range(repetitions) for seed in args.seeds for experiment in experiments
                for spec in variant_plan(experiment, config, seed)
                for storage_format in spec["deploy_formats"] for batch in config["batch_sizes"]]
        print(f"benchmark on {args.device}: {len(runs)} runs -> {benchmark_dir}")
        for i, (rep, spec, storage_format, batch) in enumerate(runs, 1):
            output = benchmark_dir / f"{spec['tag']}_{storage_format}_b{batch}_{run_tag}_rep{rep}.json"
            if finished(output):
                continue
            variant = variants_dir / f"{spec['tag']}.pt"
            if not variant.exists():
                failed.append(f"{output.name} (variant missing: run the variants stage)")
                continue
            print(f"[{i}/{len(runs)}] ", end="", flush=True)
            command = [sys.executable, SCRIPTS / "benchmark_mlp_variant.py", "--variant", variant,
                       "--format", storage_format, "--batch-size", batch, "--device", args.device,
                       "--config", args.config, "--repetition", rep, "--output", output]
            if subprocess.run([str(c) for c in command]).returncode != 0:
                failed.append(output.name)

    print(f"\nDone: {len(failed)} failed.")
    for name in failed:
        print(f"  FAILED: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
