"""Combined tables, CSVs and charts for the follow-up MLP experiments.

Example:
    python scripts/summarize_mlp_followup.py --run cuda1
      reads   results/mlp_followup/cuda1/ and, as baselines, results/mlp/cuda1/
      writes  results/mlp_followup/summary_cuda1/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn.config import PROJECT_ROOT
from src.mlp.config import FOLLOWUP_CONFIG_PATH, load_followup_config
from src.mlp.followup_reporting import summarize_followup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cuda1", help="follow-up results to summarize (device tag)")
    parser.add_argument("--baseline-run", default="cuda1",
                        help="original sweep whose results are the baselines (default cuda1)")
    parser.add_argument("--config", default=str(FOLLOWUP_CONFIG_PATH))
    args = parser.parse_args()

    config = load_followup_config(args.config)
    followup_dir = PROJECT_ROOT / "results" / "mlp_followup" / args.run
    baseline_dir = PROJECT_ROOT / "results" / "mlp" / args.baseline_run
    out_dir = PROJECT_ROOT / "results" / "mlp_followup" / f"summary_{args.run}"
    written = summarize_followup(followup_dir, baseline_dir, out_dir, config,
                                 include_baseline_timing=args.run == args.baseline_run)
    print("Wrote:")
    for path in written:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
