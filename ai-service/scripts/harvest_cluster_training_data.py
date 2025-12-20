#!/usr/bin/env python3
"""
Cluster-Wide Training Data Harvester

Collects and filters the highest quality training data from across the entire
cluster, optimizing for model improvement.

Quality Criteria:
1. Decisive games (clear winner, not timeout/stalemate)
2. Optimal game length (not too short, not too long)
3. Diverse victory types (elimination, ring formation, territorial)
4. Balanced player representation
5. Recent games from stronger models (higher Elo source)
6. Interesting tactical positions (detected via move patterns)

Usage:
    # Analyze data quality across cluster (dry run)
    python scripts/harvest_cluster_training_data.py --analyze

    # Harvest best data to local directory
    python scripts/harvest_cluster_training_data.py --harvest --output-dir data/harvested

    # Harvest with specific config
    python scripts/harvest_cluster_training_data.py --harvest --board-type square8 --num-players 2

    # Full cluster harvest with quality threshold
    python scripts/harvest_cluster_training_data.py --harvest --min-quality 0.7 --max-games 100000
"""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from scripts.lib.ssh import run_ssh_command

# Cluster nodes with selfplay data
CLUSTER_NODES = [
    "lambda-gh200-e",
    "lambda-gh200-f",
    "lambda-gh200-g",
    "lambda-gh200-h",
    "lambda-gh200-i",
    "lambda-gh200-k",
    "lambda-gh200-l",
    "lambda-2xh100",
]
CLUSTER_SSH_USER = "ubuntu"  # SSH user for Lambda Labs nodes

# Quality scoring weights
QUALITY_WEIGHTS = {
    "decisive_win": 0.25,      # Clear winner (not timeout)
    "game_length": 0.20,       # Optimal length (20-150 moves)
    "victory_type": 0.15,      # Interesting victory type
    "move_diversity": 0.15,    # Varied moves (not repetitive)
    "recency": 0.10,           # Recent games preferred
    "source_elo": 0.10,        # Higher Elo source models
    "tactical_content": 0.05,  # Captures, threats, etc.
}

# Optimal game length ranges by config
OPTIMAL_LENGTH = {
    "square8_2p": (25, 120),
    "square8_3p": (30, 150),
    "square8_4p": (40, 180),
    "hex8_2p": (20, 100),
    "square19_2p": (50, 300),
    "default": (20, 150),
}

# Victory type preferences (higher = more valuable for training)
VICTORY_TYPE_VALUE = {
    "ring": 1.0,              # Ring formation - most tactical
    "ring_formation": 1.0,
    "elimination": 0.9,        # Elimination - clear tactical win
    "ring_elimination": 0.9,
    "territorial": 0.8,        # Territory control
    "territory": 0.8,
    "resignation": 0.7,        # Opponent resigned (usually decisive)
    "timeout": 0.2,            # Timeout - low quality
    "stalemate": 0.3,          # Stalemate - less useful
    "draw": 0.3,
    "unknown": 0.5,
}


@dataclass
class GameQuality:
    """Quality assessment for a single game."""
    game_id: str
    board_type: str
    num_players: int
    quality_score: float
    scores: dict[str, float]
    game_data: dict[str, Any]
    source_node: str
    source_file: str


@dataclass
class ClusterDataStats:
    """Statistics for cluster-wide data."""
    total_games: int = 0
    total_size_gb: float = 0.0
    nodes: dict[str, dict] = field(default_factory=dict)
    configs: dict[str, dict] = field(default_factory=dict)
    victory_types: Counter = field(default_factory=Counter)
    quality_distribution: dict[str, int] = field(default_factory=dict)




def compute_game_quality(game: dict[str, Any], source_node: str, source_file: str) -> GameQuality | None:
    """Compute quality score for a single game."""
    scores = {}

    # Extract basic info
    board_type = game.get("board_type", "square8")
    num_players = game.get("num_players", 2)
    config_key = f"{board_type}_{num_players}p"

    game_id = game.get("game_id", hashlib.md5(json.dumps(game, sort_keys=True).encode()).hexdigest()[:12])

    # 1. Decisive win score
    winner = game.get("winner")
    termination = game.get("termination_reason", "").lower()
    victory_type = game.get("victory_type", "unknown").lower()

    if winner is not None and winner > 0:
        if "timeout" in termination or "timeout" in victory_type:
            scores["decisive_win"] = 0.2
        elif "stalemate" in termination:
            scores["decisive_win"] = 0.4
        else:
            scores["decisive_win"] = 1.0
    else:
        scores["decisive_win"] = 0.1  # No clear winner

    # 2. Game length score
    move_count = game.get("move_count", len(game.get("moves", [])))
    min_len, max_len = OPTIMAL_LENGTH.get(config_key, OPTIMAL_LENGTH["default"])

    if min_len <= move_count <= max_len:
        # Optimal range
        scores["game_length"] = 1.0
    elif move_count < min_len:
        # Too short
        scores["game_length"] = max(0.2, move_count / min_len)
    else:
        # Too long (potential timeout/stalemate)
        scores["game_length"] = max(0.2, 1.0 - (move_count - max_len) / max_len)

    # 3. Victory type score
    scores["victory_type"] = VICTORY_TYPE_VALUE.get(victory_type, 0.5)

    # 4. Move diversity score
    moves = game.get("moves", [])
    if moves:
        unique_moves = len({str(m) for m in moves})
        diversity_ratio = unique_moves / len(moves) if moves else 0
        scores["move_diversity"] = min(1.0, diversity_ratio * 1.5)
    else:
        scores["move_diversity"] = 0.5

    # 5. Recency score
    timestamp = game.get("timestamp") or game.get("created_at")
    if timestamp:
        try:
            if isinstance(timestamp, str):
                game_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                game_time = datetime.fromtimestamp(timestamp)
            age_hours = (datetime.now(game_time.tzinfo) - game_time).total_seconds() / 3600
            # Games from last 24h get full score, decays over 7 days
            scores["recency"] = max(0.3, 1.0 - (age_hours / (7 * 24)))
        except (ValueError, TypeError, OSError):
            scores["recency"] = 0.5
    else:
        scores["recency"] = 0.5

    # 6. Source Elo score (if available)
    source_elo = game.get("model_elo", game.get("source_elo", 1500))
    # Normalize: 1200-2000 Elo range
    scores["source_elo"] = min(1.0, max(0.0, (source_elo - 1200) / 800))

    # 7. Tactical content score
    # Check for captures, interesting patterns
    tactical_indicators = 0
    moves_str = json.dumps(moves)
    if "capture" in moves_str.lower():
        tactical_indicators += 1
    if move_count > 10 and len({str(m) for m in moves[-10:]}) > 7:
        tactical_indicators += 1  # Diverse endgame
    scores["tactical_content"] = min(1.0, tactical_indicators * 0.5 + 0.3)

    # Compute weighted total
    total_score = sum(scores[k] * QUALITY_WEIGHTS[k] for k in QUALITY_WEIGHTS)

    return GameQuality(
        game_id=game_id,
        board_type=board_type,
        num_players=num_players,
        quality_score=total_score,
        scores=scores,
        game_data=game,
        source_node=source_node,
        source_file=source_file,
    )


def analyze_node_data(node: str) -> dict[str, Any]:
    """Analyze selfplay data on a single node."""
    print(f"  Analyzing {node}...")

    # Get data size
    success, output = run_ssh_command(
        node,
        "du -sb ~/ringrift/ai-service/data/selfplay/ 2>/dev/null | cut -f1",
        user=CLUSTER_SSH_USER,
    )
    size_bytes = int(output.strip()) if success and output.strip().isdigit() else 0

    # Get file count
    success, output = run_ssh_command(
        node,
        "find ~/ringrift/ai-service/data/selfplay -name '*.jsonl' 2>/dev/null | wc -l",
        user=CLUSTER_SSH_USER,
    )
    file_count = int(output.strip()) if success and output.strip().isdigit() else 0

    # Get sample of games for quality analysis
    success, output = run_ssh_command(
        node,
        """cd ~/ringrift/ai-service && find data/selfplay -name '*.jsonl' -type f 2>/dev/null | shuf | head -5 | while read f; do head -100 "$f" 2>/dev/null; done | shuf | head -50""",
        user=CLUSTER_SSH_USER,
        timeout=60,
    )

    sample_games = []
    victory_types = Counter()
    configs = Counter()

    if success and output:
        for line in output.strip().split("\n"):
            if line:
                try:
                    game = json.loads(line)
                    sample_games.append(game)
                    victory_types[game.get("victory_type", "unknown")] += 1
                    config = f"{game.get('board_type', 'unknown')}_{game.get('num_players', 2)}p"
                    configs[config] += 1
                except json.JSONDecodeError:
                    pass

    # Calculate average quality from sample
    quality_scores = []
    for game in sample_games:
        quality = compute_game_quality(game, node, "sample")
        if quality:
            quality_scores.append(quality.quality_score)

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    return {
        "node": node,
        "size_gb": size_bytes / (1024**3),
        "file_count": file_count,
        "sample_size": len(sample_games),
        "avg_quality": avg_quality,
        "victory_types": dict(victory_types),
        "configs": dict(configs),
    }


def analyze_cluster() -> ClusterDataStats:
    """Analyze data quality across the entire cluster."""
    print("\nAnalyzing cluster data quality...\n")

    stats = ClusterDataStats()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_node_data, node): node for node in CLUSTER_NODES}

        for future in as_completed(futures):
            node = futures[future]
            try:
                result = future.result()
                stats.nodes[node] = result
                stats.total_games += result.get("sample_size", 0) * result.get("file_count", 0) // 50  # Estimate
                stats.total_size_gb += result.get("size_gb", 0)

                for vt, count in result.get("victory_types", {}).items():
                    stats.victory_types[vt] += count

                for config, count in result.get("configs", {}).items():
                    if config not in stats.configs:
                        stats.configs[config] = {"count": 0, "quality_sum": 0}
                    stats.configs[config]["count"] += count
                    stats.configs[config]["quality_sum"] += count * result.get("avg_quality", 0)

            except Exception as e:
                print(f"  Error analyzing {node}: {e}")

    return stats


def print_analysis_report(stats: ClusterDataStats):
    """Print a formatted analysis report."""
    print("\n" + "=" * 70)
    print("  CLUSTER TRAINING DATA ANALYSIS")
    print("=" * 70)

    print(f"\nTotal Data: {stats.total_size_gb:.1f} GB across {len(stats.nodes)} nodes")
    print(f"Estimated Games: ~{stats.total_games:,}")

    print("\n--- Per-Node Statistics ---")
    print(f"{'Node':<20} {'Size':>8} {'Files':>8} {'Avg Quality':>12}")
    print("-" * 50)

    for node, data in sorted(stats.nodes.items(), key=lambda x: x[1].get("avg_quality", 0), reverse=True):
        print(f"{node:<20} {data.get('size_gb', 0):>7.1f}G {data.get('file_count', 0):>8} {data.get('avg_quality', 0):>11.2f}")

    print("\n--- Victory Type Distribution ---")
    total_vt = sum(stats.victory_types.values())
    for vt, count in stats.victory_types.most_common():
        pct = count / total_vt * 100 if total_vt > 0 else 0
        quality = VICTORY_TYPE_VALUE.get(vt.lower(), 0.5)
        bar = "█" * int(pct / 5)
        print(f"  {vt:<20} {count:>6} ({pct:>5.1f}%) {bar} [q={quality:.1f}]")

    print("\n--- Configuration Distribution ---")
    for config, data in sorted(stats.configs.items()):
        avg_q = data["quality_sum"] / data["count"] if data["count"] > 0 else 0
        print(f"  {config:<15} {data['count']:>6} games  (avg quality: {avg_q:.2f})")

    print("\n--- Recommendations ---")

    # Find best nodes
    best_nodes = sorted(stats.nodes.items(), key=lambda x: x[1].get("avg_quality", 0), reverse=True)[:3]
    print(f"  Top quality nodes: {', '.join(n for n, _ in best_nodes)}")

    # Find problematic victory types
    low_quality_vt = [vt for vt, _ in stats.victory_types.items()
                      if VICTORY_TYPE_VALUE.get(vt.lower(), 0.5) < 0.4]
    if low_quality_vt:
        print(f"  Filter out: {', '.join(low_quality_vt)} (low training value)")

    print("\n" + "=" * 70)


def harvest_games_from_node(
    node: str,
    board_type: str,
    num_players: int,
    min_quality: float,
    max_games: int,
) -> list[GameQuality]:
    """Harvest high-quality games from a single node."""
    print(f"  Harvesting from {node}...")


    # Get list of JSONL files for this config
    success, output = run_ssh_command(
        node,
        f"find ~/ringrift/ai-service/data/selfplay -path '*{board_type}*{num_players}p*' -name '*.jsonl' 2>/dev/null | shuf",
        user=CLUSTER_SSH_USER,
        timeout=30,
    )

    if not success:
        return []

    files = [f.strip() for f in output.strip().split("\n") if f.strip()]

    quality_games = []
    games_checked = 0

    for file_path in files:
        if len(quality_games) >= max_games:
            break

        # Read games from file
        success, output = run_ssh_command(
            node,
            f"cat '{file_path}' 2>/dev/null | shuf | head -500",
            user=CLUSTER_SSH_USER,
            timeout=60,
        )

        if not success:
            continue

        for line in output.strip().split("\n"):
            if not line:
                continue

            games_checked += 1

            try:
                game = json.loads(line)

                # Quick pre-filter
                if game.get("board_type") != board_type:
                    continue
                if game.get("num_players") != num_players:
                    continue

                quality = compute_game_quality(game, node, file_path)

                if quality and quality.quality_score >= min_quality:
                    quality_games.append(quality)

                    if len(quality_games) >= max_games:
                        break

            except json.JSONDecodeError:
                continue

    print(f"    {node}: checked {games_checked}, found {len(quality_games)} high-quality games")
    return quality_games


def harvest_cluster_data(
    board_type: str = "square8",
    num_players: int = 2,
    min_quality: float = 0.6,
    max_games: int = 50000,
    output_dir: Path | None = None,
) -> list[GameQuality]:
    """Harvest high-quality training data from the entire cluster."""
    print(f"\nHarvesting {board_type}_{num_players}p games (min quality: {min_quality})...\n")

    all_games = []
    seen_ids = set()

    games_per_node = max_games // len(CLUSTER_NODES) + 1000  # Get extra for dedup

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                harvest_games_from_node,
                node, board_type, num_players, min_quality, games_per_node
            ): node
            for node in CLUSTER_NODES
        }

        for future in as_completed(futures):
            node = futures[future]
            try:
                games = future.result()

                # Deduplicate
                for game in games:
                    if game.game_id not in seen_ids:
                        seen_ids.add(game.game_id)
                        all_games.append(game)

            except Exception as e:
                print(f"  Error harvesting from {node}: {e}")

    # Sort by quality and take top N
    all_games.sort(key=lambda g: g.quality_score, reverse=True)
    final_games = all_games[:max_games]

    print(f"\nHarvested {len(final_games)} unique high-quality games")

    # Quality distribution
    quality_buckets = Counter()
    for g in final_games:
        bucket = f"{int(g.quality_score * 10) / 10:.1f}"
        quality_buckets[bucket] += 1

    print("\nQuality distribution:")
    for bucket in sorted(quality_buckets.keys(), reverse=True):
        count = quality_buckets[bucket]
        bar = "█" * (count // 100)
        print(f"  {bucket}: {count:>6} {bar}")

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"harvested_{board_type}_{num_players}p_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(output_file, "w") as f:
            for game in final_games:
                # Add quality metadata
                game.game_data["_quality_score"] = game.quality_score
                game.game_data["_quality_scores"] = game.scores
                game.game_data["_source_node"] = game.source_node
                f.write(json.dumps(game.game_data) + "\n")

        print(f"\nSaved to: {output_file}")

        # Also save quality report
        report_file = output_dir / f"quality_report_{board_type}_{num_players}p_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = {
            "config": f"{board_type}_{num_players}p",
            "total_games": len(final_games),
            "min_quality": min_quality,
            "avg_quality": sum(g.quality_score for g in final_games) / len(final_games) if final_games else 0,
            "quality_distribution": dict(quality_buckets),
            "source_nodes": Counter(g.source_node for g in final_games),
            "timestamp": datetime.now().isoformat(),
        }
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {report_file}")

    return final_games


def main():
    parser = argparse.ArgumentParser(description="Harvest high-quality training data from cluster")
    parser.add_argument("--analyze", action="store_true", help="Analyze data quality across cluster")
    parser.add_argument("--harvest", action="store_true", help="Harvest high-quality games")
    parser.add_argument("--board-type", default="square8", help="Board type to harvest")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--min-quality", type=float, default=0.6, help="Minimum quality score (0-1)")
    parser.add_argument("--max-games", type=int, default=50000, help="Maximum games to harvest")
    parser.add_argument("--output-dir", type=str, help="Output directory for harvested data")

    args = parser.parse_args()

    if args.analyze:
        stats = analyze_cluster()
        print_analysis_report(stats)

    if args.harvest:
        output_dir = Path(args.output_dir) if args.output_dir else None
        harvest_cluster_data(
            board_type=args.board_type,
            num_players=args.num_players,
            min_quality=args.min_quality,
            max_games=args.max_games,
            output_dir=output_dir,
        )

    if not args.analyze and not args.harvest:
        parser.print_help()


if __name__ == "__main__":
    main()
