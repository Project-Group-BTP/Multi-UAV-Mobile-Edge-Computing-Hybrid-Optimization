#!/usr/bin/env python3
"""
Convenience script to compare multiple algorithm training runs and generate comparative plots.

Usage:
    python utils/compare_algorithms.py --logs log_dir1 log_dir2 log_dir3 --names MADDPG MATD3 MAPPO --output comparison_plots

Example:
    python utils/compare_algorithms.py \
        --logs train_logs/maddpg train_logs/matd3 train_logs/mappo \
        --names MADDPG MATD3 MAPPO \
        --output comparative_plots \
        --smoothing 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.comparative_plots import compare_algorithms


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple RL algorithm runs and generate comparative plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="List of log directories containing log_data_*.json files",
    )

    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="List of algorithm names (must match --logs length)",
    )

    parser.add_argument(
        "--output",
        default="comparative_plots",
        help="Output directory for plots (default: comparative_plots)",
    )

    parser.add_argument(
        "--smoothing",
        type=int,
        default=5,
        help="Smoothing window size for moving average (default: 5)",
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.logs) != len(args.names):
        print("❌ Error: Number of --logs must match number of --names")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Comparative Analysis Tool")
    print("=" * 60)
    print(f"Algorithms: {', '.join(args.names)}")
    print(f"Output directory: {args.output}")
    print(f"Smoothing window: {args.smoothing}")
    print("=" * 60 + "\n")

    compare_algorithms(args.logs, args.names, args.output, smoothing_window=args.smoothing)


if __name__ == "__main__":
    main()
