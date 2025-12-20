#!/usr/bin/env python3
"""Canonical Diverse Selfplay - Maximum quality training data generation.

This script generates CANONICAL, PARITY-VERIFIED training games with:
1. ALL AI types including experimental (CAGE, EBMO, GMO)
2. Balanced board types (prioritizing underrepresented: hex, square19)
3. Balanced player counts (2p, 3p, 4p)
4. GPU-accelerated batch inference
5. Parity verification against TypeScript rules engine

Usage:
    # Run balanced canonical selfplay
    python scripts/run_canonical_diverse_selfplay.py

    # Focus on underrepresented configs
    python scripts/run_canonical_diverse_selfplay.py --underrepresented-only
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# ALL AI TYPES - including experimental
ALL_AI_TYPES = [
    # Core AI types
    "nnue-guided",      # NNUE evaluation with search
    "nn-minimax",       # Neural net + minimax
    "mcts",             # Monte Carlo Tree Search
    "gumbel-mcts",      # Gumbel AlphaZero MCTS
    "nn-descent",       # Gradient descent search
    "policy-only",      # Direct policy network

    # Experimental AI types
    "cage",             # CAGE AI (Constraint-Aware Graph Energy-based)
    "ebmo",             # EBMO AI (Energy-Based Move Optimization)
    "gmo",              # GMO AI (Gradient Move Optimization)
    "ig-gmo",           # IG-GMO AI (Information-Gain GMO)

    # Baseline opponents
    "heuristic-only",   # Pure heuristic
    "random",           # Random moves
]

# Board configurations with priority weights (higher = more games needed)
BOARD_CONFIGS = {
    # Severely underrepresented - highest priority
    "hexagonal_2p": {"board": "hexagonal", "players": 2, "priority": 100},
    "hexagonal_3p": {"board": "hexagonal", "players": 3, "priority": 100},
    "hexagonal_4p": {"board": "hexagonal", "players": 4, "priority": 100},
    "square19_2p": {"board": "square19", "players": 2, "priority": 100},
    "square19_3p": {"board": "square19", "players": 3, "priority": 100},
    "square19_4p": {"board": "square19", "players": 4, "priority": 100},

    # Underrepresented player counts
    "square8_3p": {"board": "square8", "players": 3, "priority": 50},
    "square8_4p": {"board": "square8", "players": 4, "priority": 50},

    # Well-represented (lower priority)
    "square8_2p": {"board": "square8", "players": 2, "priority": 10},
}


@dataclass
class CanonicalSelfplayConfig:
    """Configuration for canonical diverse selfplay."""
    board_type: str
    num_players: int
    games_target: int = 100
    gpu_id: int = 0
    batch_size: int = 64
    verify_parity: bool = True
    include_experimental_ai: bool = True
    output_dir: Path = field(default_factory=lambda: Path("data/games"))


def get_current_game_counts() -> Dict[str, int]:
    """Get current game counts from databases."""
    counts = {}
    games_dir = AI_SERVICE_ROOT / "data" / "games"

    for config_key, config in BOARD_CONFIGS.items():
        board = config["board"]
        players = config["players"]

        # Check canonical database
        db_path = games_dir / f"canonical_{board}.db"
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM games WHERE num_players = ?",
                    (players,)
                )
                count = cursor.fetchone()[0]
                conn.close()
                counts[config_key] = count
            except Exception:
                counts[config_key] = 0
        else:
            counts[config_key] = 0

    return counts


def get_priority_configs(underrepresented_only: bool = False) -> List[Tuple[str, Dict]]:
    """Get configs sorted by priority (most underrepresented first)."""
    counts = get_current_game_counts()

    # Calculate effective priority: base_priority / (1 + log(count+1))
    import math
    priorities = []
    for config_key, config in BOARD_CONFIGS.items():
        count = counts.get(config_key, 0)
        base_priority = config["priority"]
        # Higher effective priority for lower counts
        effective_priority = base_priority / (1 + math.log10(count + 1))

        if underrepresented_only and count > 1000:
            continue  # Skip well-represented configs

        priorities.append((config_key, config, effective_priority, count))

    # Sort by effective priority (descending)
    priorities.sort(key=lambda x: x[2], reverse=True)

    return [(k, c) for k, c, _, _ in priorities]


def build_selfplay_command(
    config: CanonicalSelfplayConfig,
    ai_matchup: Tuple[str, str],
    num_games: int,
) -> List[str]:
    """Build command to run selfplay with specific AI matchup."""
    player1_ai, player2_ai = ai_matchup

    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "run_selfplay.py"),
        "--board", config.board_type,
        "--num-players", str(config.num_players),
        "--num-games", str(num_games),
        "--output-dir", str(config.output_dir),
        "--gpu", str(config.gpu_id),
        "--batch-size", str(config.batch_size),
        "--ai-type", player1_ai,
    ]

    # Add opponent AI if different
    if player2_ai != player1_ai:
        cmd.extend(["--opponent-ai", player2_ai])

    # Enable parity verification
    if config.verify_parity:
        cmd.append("--verify-parity")

    # Use canonical database
    cmd.append("--canonical")

    return cmd


def run_selfplay_batch(
    config: CanonicalSelfplayConfig,
    matchups: List[Tuple[str, str]],
    games_per_matchup: int = 10,
) -> Dict[str, int]:
    """Run selfplay for multiple AI matchups."""
    results = {}

    for matchup in matchups:
        ai1, ai2 = matchup
        matchup_key = f"{ai1}_vs_{ai2}"

        cmd = build_selfplay_command(config, matchup, games_per_matchup)

        try:
            print(f"  Running {matchup_key}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per matchup
                cwd=str(AI_SERVICE_ROOT),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(config.gpu_id)},
            )

            if result.returncode == 0:
                results[matchup_key] = games_per_matchup
                print(f"    {matchup_key}: {games_per_matchup} games completed")
            else:
                results[matchup_key] = 0
                print(f"    {matchup_key}: FAILED - {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            results[matchup_key] = 0
            print(f"    {matchup_key}: TIMEOUT")
        except Exception as e:
            results[matchup_key] = 0
            print(f"    {matchup_key}: ERROR - {e}")

    return results


def get_diverse_matchups(include_experimental: bool = True) -> List[Tuple[str, str]]:
    """Generate diverse AI matchups for training."""
    matchups = []

    # Core strong AI types
    strong_ais = ["nnue-guided", "nn-minimax", "mcts", "gumbel-mcts", "policy-only"]
    weak_ais = ["heuristic-only", "random"]
    experimental_ais = ["cage", "ebmo", "gmo"] if include_experimental else []

    # Strong vs Strong (high quality games)
    for i, ai1 in enumerate(strong_ais):
        for ai2 in strong_ais[i:]:
            matchups.append((ai1, ai2))

    # Strong vs Weak (asymmetric for value learning)
    for strong in strong_ais:
        for weak in weak_ais:
            matchups.append((strong, weak))

    # Experimental AI matchups
    for exp in experimental_ais:
        # Experimental vs strong
        for strong in strong_ais[:2]:  # Top 2 strong AIs
            matchups.append((exp, strong))
        # Experimental self-play
        matchups.append((exp, exp))

    return matchups


def main():
    parser = argparse.ArgumentParser(description="Canonical Diverse Selfplay Generator")
    parser.add_argument("--underrepresented-only", action="store_true",
                        help="Only run on underrepresented configs")
    parser.add_argument("--games-per-config", type=int, default=100,
                        help="Target games per config")
    parser.add_argument("--games-per-matchup", type=int, default=10,
                        help="Games per AI matchup")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--no-experimental", action="store_true",
                        help="Skip experimental AI types")
    parser.add_argument("--no-parity", action="store_true",
                        help="Skip parity verification")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without executing")

    args = parser.parse_args()

    print("=" * 60)
    print("  CANONICAL DIVERSE SELFPLAY GENERATOR")
    print("=" * 60)

    # Get current state
    counts = get_current_game_counts()
    print("\nCurrent game counts:")
    for config_key, count in sorted(counts.items(), key=lambda x: x[1]):
        print(f"  {config_key}: {count:,} games")

    # Get priority configs
    priority_configs = get_priority_configs(args.underrepresented_only)

    print(f"\nWill run selfplay on {len(priority_configs)} configs:")
    for config_key, config in priority_configs[:10]:
        print(f"  {config_key} (priority: {config['priority']})")

    if args.dry_run:
        print("\n[DRY RUN] Would execute selfplay on above configs")
        return

    # Get AI matchups
    matchups = get_diverse_matchups(not args.no_experimental)
    print(f"\nUsing {len(matchups)} AI matchups")

    # Run selfplay for each config
    total_games = 0
    for config_key, config_dict in priority_configs:
        print(f"\n{'='*40}")
        print(f"Config: {config_key}")
        print(f"{'='*40}")

        config = CanonicalSelfplayConfig(
            board_type=config_dict["board"],
            num_players=config_dict["players"],
            games_target=args.games_per_config,
            gpu_id=args.gpu,
            verify_parity=not args.no_parity,
            include_experimental_ai=not args.no_experimental,
        )

        results = run_selfplay_batch(
            config,
            matchups,
            args.games_per_matchup,
        )

        config_total = sum(results.values())
        total_games += config_total
        print(f"\nConfig total: {config_total} games")

    print(f"\n{'='*60}")
    print(f"TOTAL GAMES GENERATED: {total_games}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
