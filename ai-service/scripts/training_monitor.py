#!/usr/bin/env python3
"""Training Progress Monitor.

Monitors training and self-play progress across the cluster.

Usage:
    python scripts/training_monitor.py
    python scripts/training_monitor.py --continuous --interval 60
    python scripts/training_monitor.py --check-models
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run_ssh(host: str, cmd: str, timeout: int = 10) -> str:
    """Run SSH command and return output."""
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                "-i", str(Path.home() / ".ssh" / "id_cluster"),
                f"ubuntu@{host}", cmd
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def get_cluster_status():
    """Get status from all cluster nodes."""
    hosts = {
        "gh200-a": "100.123.183.70",
        "gh200-c": "100.88.35.19",
        "gh200-d": "100.75.84.47",
        "gh200-e": "100.88.176.74",
        "gh200-f": "100.104.165.116",
        "gh200-g": "100.104.126.58",
        "gh200-h": "100.65.88.62",
        "gh200-i": "100.99.27.56",
        "gh200-k": "100.96.142.42",
        "gh200-l": "100.76.145.60",
    }

    status = {}
    for name, ip in hosts.items():
        gpu = run_ssh(ip, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1")
        selfplay = run_ssh(ip, "pgrep -f 'run_gpu_selfplay' | wc -l")
        training = run_ssh(ip, "pgrep -f 'train_nnue' | wc -l")
        games = run_ssh(ip, "find ~/ringrift/ai-service/data/games -name '*.jsonl' -mmin -60 2>/dev/null | wc -l")

        status[name] = {
            "gpu": gpu,
            "selfplay": selfplay,
            "training": training,
            "recent_games": games,
        }

    return status


def get_model_counts():
    """Count models by configuration."""
    models_dir = Path("models")
    counts = {}
    for pattern, label in [
        ("*sq8_2p*.pth", "sq8_2p"),
        ("*square8*2p*.pth", "sq8_2p"),
        ("*sq19_2p*.pth", "sq19_2p"),
        ("*square19*2p*.pth", "sq19_2p"),
        ("*hex*2p*.pth", "hex_2p"),
        ("*hexagonal*2p*.pth", "hex_2p"),
    ]:
        for p in models_dir.rglob(pattern):
            counts[label] = counts.get(label, 0) + 1
    return counts


def get_composite_elo_summary():
    """Get composite ELO summary."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.training.elo_service import get_elo_service

        elo_service = get_elo_service()
        summary = {}
        for board, players in [("square8", 2), ("square19", 2), ("hexagonal", 2)]:
            lb = elo_service.get_composite_leaderboard(
                board_type=board, num_players=players, min_games=0, limit=1000
            )
            if lb:
                total_games = sum(p.get("games_played", 0) for p in lb)
                top_elo = lb[0].get("rating", 0) if lb else 0
                summary[f"{board[:3]}_{players}p"] = {
                    "participants": len(lb),
                    "total_games": total_games,
                    "top_elo": int(top_elo),
                }
        return summary
    except Exception as e:
        return {"error": str(e)}


def print_status_report():
    """Print a comprehensive status report."""
    print("\n" + "=" * 70)
    print(" TRAINING MONITOR")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n--- Cluster Nodes ---")
    print(f"{'Node':<12} {'GPU%':<8} {'Selfplay':<10} {'Training':<10} {'Games/hr':<10}")
    print("-" * 55)

    status = get_cluster_status()
    for name, data in sorted(status.items()):
        gpu = data.get("gpu", "-")
        selfplay = data.get("selfplay", "-")
        training = data.get("training", "-")
        games = data.get("recent_games", "-")
        print(f"{name:<12} {gpu:<8} {selfplay:<10} {training:<10} {games:<10}")

    print("\n--- Local Model Counts ---")
    counts = get_model_counts()
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    print("\n--- Composite ELO Summary ---")
    elo_summary = get_composite_elo_summary()
    if "error" not in elo_summary:
        print(f"{'Config':<12} {'Participants':<14} {'Total Games':<14} {'Top Elo':<10}")
        print("-" * 50)
        for config, data in sorted(elo_summary.items()):
            print(f"{config:<12} {data['participants']:<14} {data['total_games']:<14} {data['top_elo']:<10}")
    else:
        print(f"  Error: {elo_summary['error']}")

    print("\n" + "=" * 70)


def continuous_mode(interval: int):
    """Run continuous monitoring."""
    print(f"Starting continuous monitoring (interval: {interval}s)")
    print("Press Ctrl+C to stop")
    while True:
        print_status_report()
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Training Progress Monitor")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--check-models", action="store_true")

    args = parser.parse_args()

    if args.check_models:
        counts = get_model_counts()
        for label, count in sorted(counts.items()):
            print(f"{label}: {count}")
        return

    if args.continuous:
        continuous_mode(args.interval)
    else:
        print_status_report()


if __name__ == "__main__":
    main()
