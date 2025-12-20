#!/usr/bin/env python3
"""Aggregate Elo/gauntlet results from multiple cluster nodes.

Collects baseline_gauntlet_results.json from all nodes and merges them
into a unified ranking.

Usage:
    python scripts/aggregate_elo_results.py --collect
    python scripts/aggregate_elo_results.py --report
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AI_SERVICE_ROOT = Path(__file__).parent.parent
RESULTS_DIR = AI_SERVICE_ROOT / "data" / "cluster_elo_results"
AGGREGATED_FILE = AI_SERVICE_ROOT / "data" / "aggregated_gauntlet_results.json"

# Cluster nodes loaded from config/distributed_hosts.yaml
# No hardcoded fallback - must have config file


def _load_hosts_from_config():
    """Load hosts from config/distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print("[Script] Warning: No config found at", config_path)
        return []
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Extract hosts with tailscale_ip or ssh_host
        hosts = []
        for _name, host_config in config.get('hosts', {}).items():
            if host_config.get('status') == 'terminated':
                continue

            # Prefer Tailscale IP, fallback to ssh_host
            ip = host_config.get('tailscale_ip') or host_config.get('ssh_host')
            user = host_config.get('ssh_user', 'ubuntu')

            if ip:
                hosts.append((ip, user))

        return hosts
    except Exception as e:
        print(f"[Script] Error loading config: {e}")
        return []


# Load hosts from config
ALL_NODES = _load_hosts_from_config()


@dataclass
class ModelResult:
    """Aggregated result for a model."""
    name: str
    board_type: str
    num_players: int
    scores: list[float]
    vs_random: list[float]
    vs_heuristic: list[float]
    vs_mcts: list[float]
    sources: list[str]

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0

    @property
    def avg_vs_random(self) -> float:
        return sum(self.vs_random) / len(self.vs_random) if self.vs_random else 0

    @property
    def avg_vs_heuristic(self) -> float:
        return sum(self.vs_heuristic) / len(self.vs_heuristic) if self.vs_heuristic else 0

    @property
    def avg_vs_mcts(self) -> float:
        return sum(self.vs_mcts) / len(self.vs_mcts) if self.vs_mcts else 0


def collect_from_node(host: str, user: str) -> dict[str, Any]:
    """Collect gauntlet results from a single node."""
    try:
        # Try multiple possible result file locations
        # Use list-form command to avoid shell injection
        remote_cmd = (
            'cat ~/ringrift/ai-service/data/baseline_gauntlet_results.json 2>/dev/null || '
            'cat ~/ringrift/ai-service/data/gauntlet_results.json 2>/dev/null || '
            'echo "{}"'
        )
        result = subprocess.run(
            [
                "ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                f"{user}@{host}", remote_cmd
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if data:
                data["_source"] = host
                return data
    except Exception as e:
        print(f"  Error collecting from {host}: {e}", file=sys.stderr)
    return {}


def collect_all_results() -> list[dict[str, Any]]:
    """Collect results from all nodes in parallel."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Collecting from {len(ALL_NODES)} nodes...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(collect_from_node, host, user): (host, user)
            for host, user in ALL_NODES
        }

        for future in concurrent.futures.as_completed(futures):
            host, _user = futures[future]
            try:
                data = future.result()
                if data and data.get("results"):
                    results.append(data)
                    print(f"  ✓ {host}: {len(data.get('results', []))} models")
                    # Save individual result
                    safe_host = host.replace(".", "_").replace("-", "_")
                    with open(RESULTS_DIR / f"{safe_host}.json", "w") as f:
                        json.dump(data, f, indent=2)
                else:
                    print(f"  ✗ {host}: no results")
            except Exception as e:
                print(f"  ✗ {host}: {e}")

    return results


def aggregate_results(all_results: list[dict[str, Any]]) -> dict[str, ModelResult]:
    """Aggregate results across all nodes."""
    models: dict[str, ModelResult] = {}

    for node_data in all_results:
        source = node_data.get("_source", "unknown")
        board_type = node_data.get("board_type", "unknown")
        num_players = node_data.get("num_players", 2)

        for result in node_data.get("results", []):
            name = result.get("model_name", result.get("name", "unknown"))
            key = f"{name}_{board_type}_{num_players}p"

            if key not in models:
                models[key] = ModelResult(
                    name=name,
                    board_type=board_type,
                    num_players=num_players,
                    scores=[],
                    vs_random=[],
                    vs_heuristic=[],
                    vs_mcts=[],
                    sources=[],
                )

            m = models[key]
            m.scores.append(result.get("score", 0))
            m.vs_random.append(result.get("vs_random", 0))
            m.vs_heuristic.append(result.get("vs_heuristic", 0))
            m.vs_mcts.append(result.get("vs_mcts", 0))
            m.sources.append(source)

    return models


def generate_report(models: dict[str, ModelResult]) -> str:
    """Generate a ranking report."""
    lines = ["# Aggregated Gauntlet Results\n"]

    # Group by board type and player count
    grouped = defaultdict(list)
    for model in models.values():
        key = f"{model.board_type}_{model.num_players}p"
        grouped[key].append(model)

    for config, config_models in sorted(grouped.items()):
        lines.append(f"\n## {config}\n")
        lines.append("| Rank | Model | Score | vs Random | vs Heuristic | vs MCTS | Sources |")
        lines.append("|------|-------|-------|-----------|--------------|---------|---------|")

        # Sort by average score descending
        sorted_models = sorted(config_models, key=lambda m: m.avg_score, reverse=True)

        for rank, model in enumerate(sorted_models, 1):
            lines.append(
                f"| {rank} | {model.name} | {model.avg_score:.2f} | "
                f"{model.avg_vs_random:.0%} | {model.avg_vs_heuristic:.0%} | "
                f"{model.avg_vs_mcts:.0%} | {len(model.sources)} |"
            )

    return "\n".join(lines)


def save_aggregated(models: dict[str, ModelResult]):
    """Save aggregated results to JSON."""
    data = {
        "num_models": len(models),
        "results": [
            {
                "name": m.name,
                "board_type": m.board_type,
                "num_players": m.num_players,
                "avg_score": m.avg_score,
                "avg_vs_random": m.avg_vs_random,
                "avg_vs_heuristic": m.avg_vs_heuristic,
                "avg_vs_mcts": m.avg_vs_mcts,
                "num_samples": len(m.scores),
                "sources": m.sources,
            }
            for m in models.values()
        ]
    }

    with open(AGGREGATED_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nAggregated results saved to {AGGREGATED_FILE}")


def get_top_models(models: dict[str, ModelResult], n: int = 5) -> dict[str, list[str]]:
    """Get top N models per config for Elo tournament."""
    grouped = defaultdict(list)
    for model in models.values():
        key = f"{model.board_type}_{model.num_players}p"
        grouped[key].append(model)

    top = {}
    for config, config_models in grouped.items():
        sorted_models = sorted(config_models, key=lambda m: m.avg_score, reverse=True)
        top[config] = [m.name for m in sorted_models[:n]]

    return top


def main():
    parser = argparse.ArgumentParser(description="Aggregate Elo results from cluster")
    parser.add_argument("--collect", action="store_true", help="Collect results from all nodes")
    parser.add_argument("--report", action="store_true", help="Generate report from collected results")
    parser.add_argument("--top", type=int, default=5, help="Number of top models to show per config")
    args = parser.parse_args()

    if args.collect or not AGGREGATED_FILE.exists():
        all_results = collect_all_results()
        if all_results:
            models = aggregate_results(all_results)
            save_aggregated(models)
            print(generate_report(models))

            # Show top models for Elo tournament
            top = get_top_models(models, args.top)
            print(f"\n## Top {args.top} Models per Config (for Elo tournament)\n")
            for config, names in sorted(top.items()):
                print(f"**{config}**: {', '.join(names)}")
        else:
            print("No results collected from any node")
            return 1

    elif args.report:
        if not AGGREGATED_FILE.exists():
            print("No aggregated results found. Run with --collect first.")
            return 1

        with open(AGGREGATED_FILE) as f:
            data = json.load(f)

        # Reconstruct models from saved data
        models = {}
        for r in data.get("results", []):
            key = f"{r['name']}_{r['board_type']}_{r['num_players']}p"
            models[key] = ModelResult(
                name=r["name"],
                board_type=r["board_type"],
                num_players=r["num_players"],
                scores=[r["avg_score"]] * r.get("num_samples", 1),
                vs_random=[r["avg_vs_random"]] * r.get("num_samples", 1),
                vs_heuristic=[r["avg_vs_heuristic"]] * r.get("num_samples", 1),
                vs_mcts=[r["avg_vs_mcts"]] * r.get("num_samples", 1),
                sources=r.get("sources", []),
            )

        print(generate_report(models))

    return 0


if __name__ == "__main__":
    sys.exit(main())
