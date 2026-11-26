"""Summarize the GPU format comparison: tables, CSVs and charts in results/mlp_gpu/summary_<run>/.

    python scripts/summarize_mlp_gpu_comparison.py --run cuda1

--baseline-run: the original sweep (results/mlp/<run>) used for the accuracy and latency
consistency checks (default: the same run tag).
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from src.gcn.config import PROJECT_ROOT
from src.mlp.config import COMPARISON_CONFIG_PATH, load_comparison_config
from src.mlp.gpu_comparison_reporting import summarize_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cuda1", help="results/mlp_gpu/<run>")
    parser.add_argument("--baseline-run", default=None, help="results/mlp/<run> (default: --run)")
    parser.add_argument("--config", default=str(COMPARISON_CONFIG_PATH))
    args = parser.parse_args()
    root = PROJECT_ROOT / "results"
    written = summarize_comparison(root / "mlp_gpu" / args.run, root / "mlp" / (args.baseline_run or args.run),
                                   root / "mlp_gpu" / f"summary_{args.run}", load_comparison_config(args.config))
    print("Wrote:")
    for path in written:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
