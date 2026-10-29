"""Summarize the pruning experiments and the epsilon/margin analysis as tables and charts.

Reads results/pruning/<dataset>_<device>/ (pruning runs), results/pruning/verify_<dataset>_<device>/
(format checks on pruned graphs) and the unpruned format comparison in results/gpu/. Writes
summary.md and PNG charts to results/pruning/summary/. Needs no GPU: everything is
recomputed from the saved JSON and logits.

Example:
    python scripts/summarize_pruning_results.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.pruning_reporting import default_dirs, summarize_pruning


def main():
    pruning_dir, formats_dir, out_dir = default_dirs()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pruning-dir", default=str(pruning_dir))
    parser.add_argument("--formats-dir", default=str(formats_dir),
                        help="folder with the unpruned compare_gcn_formats / benchmark_gcn JSON")
    parser.add_argument("--out-dir", default=str(out_dir))
    args = parser.parse_args()

    try:
        text, written = summarize_pruning(args.pruning_dir, args.formats_dir, args.out_dir)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nRun scripts/run_pruning_experiments.py first.")

    print(text)
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
