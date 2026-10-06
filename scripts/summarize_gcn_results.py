"""Summarize the GCN result JSON files as tables and charts.

Prints the tables, and writes summary.md (Markdown tables + charts) and PNG
charts to the output folder.

Example:
    python scripts/summarize_gcn_results.py
    # then open results/summary/summary.md (VS Code: Cmd+Shift+V for the preview)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.gcn.reporting import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "raw"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "summary"))
    args = parser.parse_args()

    try:
        text, written = summarize(args.results_dir, args.out_dir)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nRun scripts/compare_gcn_formats.py --output ... and/or scripts/benchmark_gcn.py first.")

    print(text)
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
