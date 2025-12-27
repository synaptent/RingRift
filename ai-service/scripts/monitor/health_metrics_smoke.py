#!/usr/bin/env python3
"""Health metrics smoke check.

Fetches the health server /metrics endpoint and verifies key metrics exist.
Exits non-zero if required metrics are missing.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


DEFAULT_URL = "http://localhost:8790/metrics"
DEFAULT_METRICS = [
    "daemon_count",
    "daemon_health_score",
    "event_router_events_routed_total",
]


def fetch_metrics(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health metrics smoke check")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Metrics endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Timeout in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric name to require (can repeat)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = args.metrics or DEFAULT_METRICS

    try:
        text = fetch_metrics(args.url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"Failed to fetch metrics: {exc}")
        return 2

    missing = [metric for metric in required if metric not in text]
    if missing:
        print("Missing metrics:")
        for metric in missing:
            print(f"  - {metric}")
        return 1

    print("OK: required metrics present")
    for metric in required:
        print(f"  - {metric}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
