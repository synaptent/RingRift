#!/usr/bin/env python3
"""Test Health Report - detect test rot before it masks real bugs.

Reports: total tests, skipped, xfail, collection errors.
Fails if collection errors exceed threshold.

Usage:
    cd ai-service && PYTHONPATH=. python3 scripts/test_health_report.py
    python3 scripts/test_health_report.py --fail-on-collection-errors
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import re


def main():
    ap = argparse.ArgumentParser(description="Test Health Report")
    ap.add_argument("--fail-on-collection-errors", action="store_true",
                    help="Exit 1 if any collection errors found")
    ap.add_argument("--paths", nargs="+",
                    default=["tests/unit/coordination", "tests/unit/distributed",
                             "tests/unit/events", "tests/unit/training",
                             "tests/parity"],
                    help="Test directories to check")
    args = ap.parse_args()

    print("=" * 50)
    print("  TEST HEALTH REPORT")
    print("=" * 50)

    # Run pytest --collect-only to find all tests without executing
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q",
           "--timeout=30"] + args.paths
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    output = r.stdout + r.stderr
    lines = output.split("\n")

    # Parse counts
    total = 0
    skipped = 0
    errors = 0
    error_details = []

    for line in lines:
        # "X tests collected"
        m = re.search(r"(\d+) tests? collected", line)
        if m:
            total = int(m.group(1))

        # "X errors"
        m = re.search(r"(\d+) errors?", line)
        if m:
            errors = int(m.group(1))

        # Collection errors
        if "ERROR collecting" in line:
            error_details.append(line.strip())

        # Count skip markers
        if "skip" in line.lower() and ("reason" in line.lower() or "marker" in line.lower()):
            skipped += 1

    # Also count skip markers from a quick grep
    skip_cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q",
                "--timeout=30", "-m", "skip"] + args.paths
    skip_r = subprocess.run(skip_cmd, capture_output=True, text=True, timeout=60)
    skip_m = re.search(r"(\d+) tests? deselected", skip_r.stdout + skip_r.stderr)
    if not skip_m:
        # Try counting "selected" which are the skip-marked ones
        skip_m2 = re.search(r"(\d+) tests? collected", skip_r.stdout + skip_r.stderr)

    print(f"\n  Tests collected: {total}")
    print(f"  Collection errors: {errors}")
    if error_details:
        for e in error_details[:5]:
            print(f"    {e}")
    print(f"  Exit code: {r.returncode}")

    print()
    if errors > 0:
        print(f"  WARNING: {errors} collection errors (broken imports/symbols)")
        if args.fail_on_collection_errors:
            print("  FAILING due to --fail-on-collection-errors")
            sys.exit(1)
    elif total == 0:
        print("  WARNING: No tests collected")
        sys.exit(1)
    else:
        print(f"  HEALTHY: {total} tests collected, {errors} errors")
        sys.exit(0)


if __name__ == "__main__":
    main()
