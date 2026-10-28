"""Summarize the synthetic sweep results as tables and charts.

Reads every sweep folder under results/synthetic/ (pilot_cuda1, size_cuda1,
density_cuda1, ...), prints the tables, and writes summary.md (Markdown tables
+ charts) and PNG charts to the output folder.

Example:
    python scripts/summarize_synthetic_results.py
    # then open results/synthetic/summary/summary.md (VS Code: Cmd+Shift+V for the preview)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.gcn.synthetic_reporting import summarize_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "synthetic"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "synthetic" / "summary"))
    args = parser.parse_args()

    try:
        text, written = summarize_synthetic(args.results_dir, args.out_dir)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nRun scripts/run_synthetic_sweep.py first.")

    print(text)
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
