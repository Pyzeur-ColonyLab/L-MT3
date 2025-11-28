#!/usr/bin/env python3
"""
Display metrics summary from batch processing

Usage:
    python show_metrics_summary.py [--metrics-dir DIRECTORY]
"""

import argparse
from pathlib import Path
from metrics_recorder import MetricsRecorder


def main():
    parser = argparse.ArgumentParser(
        description="Display batch processing metrics summary"
    )
    parser.add_argument(
        '--metrics-dir',
        type=str,
        default='metrics',
        help='Directory containing metrics files (default: metrics)'
    )

    args = parser.parse_args()

    # Load metrics recorder
    recorder = MetricsRecorder(output_dir=args.metrics_dir)

    # Check if metrics exist
    if not recorder.json_path.exists():
        print(f"❌ No metrics found in: {args.metrics_dir}")
        print(f"   Run pipeline with --metrics-dir to record metrics")
        return 1

    # Print summary
    recorder.print_summary()

    return 0


if __name__ == "__main__":
    exit(main())
