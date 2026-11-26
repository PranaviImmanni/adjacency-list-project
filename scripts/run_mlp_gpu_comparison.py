"""Run the GPU format comparison (dense, dense+mask, COO, CSR, adjacency list, low-rank factors) with energy.

Stages:
  pilot      seed(s) given: evaluate and benchmark (repetition 0) the original model, magnitude
             90% in every pruned format, and rank 32 low-rank factors. Check it before the rest.
  evaluate   accuracy and equivalence of every model, one process per seed
  benchmark  repetition x seed x model x format x batch size, one process each (latency,
             memory, energy)
  all        evaluate, then benchmark

Same checkpoints, seeds, batch sizes (1/32/256), warm-ups (20), timed passes (50) and
repetitions (3) as the original CUDA sweep (configs/mlp_fashion_mnist.json). Results go to
results/mlp_gpu/<run>/. Finished outputs are skipped, so rerunning a command resumes it;
the pilot's runs are reused by the full run.

Example (lab GPU, inside tmux):
    python scripts/run_mlp_gpu_comparison.py --stage pilot --seeds 42 --device cuda:1
    python scripts/run_mlp_gpu_comparison.py --stage all --seeds 42 43 44 --device cuda:1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.mlp.config import COMPARISON_CONFIG_PATH, load_comparison_config
from src.mlp.gpu_comparison import PARTS, pilot_plan, plan

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
    parser.add_argument("--stage", required=True, choices=["pilot", "evaluate", "benchmark", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS),
                        help="pruned (original sweep conditions, 5 formats) and/or low_rank (factor networks)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--checkpoint-run", default=None, help="trained checkpoints (default: the device tag)")
    parser.add_argument("--repetitions", type=int, default=None, help="default: 'repetitions' in the base config")
    parser.add_argument("--no-energy", action="store_true", help="skip the NVML energy measurement")
    parser.add_argument("--config", default=str(COMPARISON_CONFIG_PATH))
    args = parser.parse_args()

    config = load_comparison_config(args.config)
    run_tag = args.device.replace(":", "")
    checkpoint_run = args.checkpoint_run or run_tag
    missing = [s for s in args.seeds
               if not (PROJECT_ROOT / "checkpoints" / f"fashion_mlp_seed{s}_{checkpoint_run}.pt").exists()]
    if missing:
        sys.exit(f"missing trained checkpoints for seeds {missing} "
                 f"(checkpoints/fashion_mlp_seed<S>_{checkpoint_run}.pt)")
    common = ["--device", args.device, "--checkpoint-run", checkpoint_run, "--config", args.config]
    benchmark_dir = PROJECT_ROOT / "results" / "mlp_gpu" / run_tag / "benchmark"
    failed = []

    if args.stage in ("pilot", "evaluate", "all"):
        for seed in args.seeds:
            extra = ["--pilot"] if args.stage == "pilot" else ["--parts", *args.parts]
            if run([sys.executable, SCRIPTS / "evaluate_mlp_gpu_comparison.py", "--seed", seed, *common, *extra]) != 0:
                failed.append(f"evaluate seed {seed}")

    if args.stage in ("pilot", "benchmark", "all"):
        repetitions = 1 if args.stage == "pilot" else (args.repetitions or config["repetitions"])
        runs = [(rep, seed, spec, storage_format, batch)
                for rep in range(repetitions) for seed in args.seeds
                for spec in (pilot_plan(config, seed) if args.stage == "pilot"
                             else plan(config, seed, "benchmark", args.parts))
                for storage_format in spec["formats"] for batch in config["batch_sizes"]]
        print(f"benchmark on {args.device}: {len(runs)} runs -> {benchmark_dir.relative_to(PROJECT_ROOT)}", flush=True)
        for i, (rep, seed, spec, storage_format, batch) in enumerate(runs, 1):
            output = benchmark_dir / f"{spec['tag']}_{storage_format}_b{batch}_{run_tag}_rep{rep}.json"
            if finished(output):
                continue
            print(f"[{i}/{len(runs)}] ", end="", flush=True)
            command = [sys.executable, SCRIPTS / "benchmark_mlp_gpu_comparison.py", "--seed", seed,
                       "--tag", spec["tag"], "--format", storage_format, "--batch-size", batch,
                       "--repetition", rep, "--output", output, *common]
            if args.no_energy:
                command.append("--no-energy")
            if subprocess.run([str(c) for c in command]).returncode != 0:
                failed.append(output.name)

    print(f"\nDone: {len(failed)} failed.")
    for name in failed:
        print(f"  FAILED: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
