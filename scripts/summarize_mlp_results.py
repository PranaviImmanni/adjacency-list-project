"""Tables, CSVs and charts for the MLP weight-pruning experiment.

Example:
    python scripts/summarize_mlp_results.py --results-dir results/mlp/cuda1
    # -> results/mlp/summary_cuda1/summary.md (+ CSVs and charts)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.mlp.reporting import summarize_mlp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "mlp" / "cuda1"))
    parser.add_argument("--out-dir", default=None, help="default: results/mlp/summary_<results-dir name>")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir.parent / f"summary_{results_dir.name}"
    try:
        text, written = summarize_mlp(results_dir, out_dir)
    except FileNotFoundError as error:
        raise SystemExit(f"{error}\nRun scripts/run_mlp_experiments.py first.")
    print(text)
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
