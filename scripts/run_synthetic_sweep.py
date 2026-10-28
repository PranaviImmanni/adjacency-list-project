"""Run one synthetic sweep: every graph x graph seed x repetition x format,
each as a separate scripts/benchmark_synthetic.py process.

Sweeps (one variable changes at a time):
  pilot    1,000 nodes, average degree 8, graph seed 42
  size     1k, 5k, 10k, 20k, 40k nodes at average degree 8
  density  5,000 nodes at 0.1%, 1%, 5%, 10%, 20%, 30%, 35%, 40% of possible edges

Formats run back to back for each graph and repetition, so slow drift on the
machine does not favor one format. Existing result files are skipped, so an
interrupted sweep can be resumed by running the same command again.

Example:
    python scripts/run_synthetic_sweep.py --sweep pilot --device cuda:0
    python scripts/run_synthetic_sweep.py --sweep size --device cuda:0 --output-dir results/synthetic/size_cuda0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.adjacency import STORAGE_FORMATS
from src.gcn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config

BENCHMARK = Path(__file__).resolve().parent / "benchmark_synthetic.py"
OFFICIAL_SEEDS = [42, 43, 44]

# Each graph is (nodes, "--avg-degree" or "--density", value).
SWEEPS = {
    "pilot": {"graphs": [(1000, "--avg-degree", 8)], "seeds": [42]},
    "size": {"graphs": [(n, "--avg-degree", 8) for n in (1000, 5000, 10000, 20000, 40000)],
             "seeds": OFFICIAL_SEEDS},
    "density": {"graphs": [(5000, "--density", p) for p in (0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4)],
                "seeds": OFFICIAL_SEEDS},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, choices=SWEEPS)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override the sweep's graph seeds")
    parser.add_argument("--repetitions", type=int, default=None, help="default: 'repetitions' in the config")
    parser.add_argument("--output-dir", default=None, help="default: results/synthetic/<sweep>_<device>")
    args = parser.parse_args()

    sweep = SWEEPS[args.sweep]
    seeds = args.seeds or sweep["seeds"]
    repetitions = args.repetitions or load_config(args.config)["repetitions"]
    device_tag = args.device.replace(":", "")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "synthetic" / f"{args.sweep}_{device_tag}")

    runs = [(graph, seed, rep, storage_format)
            for graph in sweep["graphs"] for seed in seeds
            for rep in range(repetitions) for storage_format in STORAGE_FORMATS]
    print(f"{args.sweep} sweep on {args.device}: {len(runs)} runs -> {output_dir}")

    failed, not_equivalent = [], []
    for i, ((nodes, size_flag, size_value), seed, rep, storage_format) in enumerate(runs, 1):
        spec = f"{'d' if size_flag == '--avg-degree' else 'p'}{size_value:g}"
        output = output_dir / f"n{nodes}_{spec}_g{seed}_{storage_format}_{device_tag}_rep{rep}.json"
        if output.exists():
            print(f"[{i}/{len(runs)}] exists, skipping {output.name}")
            continue
        print(f"[{i}/{len(runs)}] ", end="", flush=True)
        command = [sys.executable, str(BENCHMARK), "--nodes", str(nodes), size_flag, str(size_value),
                   "--graph-seed", str(seed), "--format", storage_format, "--device", args.device,
                   "--config", args.config, "--repetition", str(rep), "--output", str(output)]
        if subprocess.run(command).returncode != 0:
            failed.append(output.name)
            continue
        with open(output) as f:
            if not json.load(f)["prediction"]["equivalent"]:
                not_equivalent.append(output.name)

    print(f"\nDone: {len(runs)} runs, {len(failed)} crashed, {len(not_equivalent)} not equivalent to the COO reference.")
    for name in failed:
        print(f"  crashed: {name}")
    for name in not_equivalent:
        print(f"  NOT EQUIVALENT: {name}")
    sys.exit(1 if failed or not_equivalent else 0)


if __name__ == "__main__":
    main()
