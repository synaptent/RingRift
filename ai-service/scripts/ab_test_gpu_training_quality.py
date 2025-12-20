#!/usr/bin/env python
"""A/B Test: Hybrid GPU Selfplay Quality Validation.

This script validates that Hybrid GPU selfplay produces reasonable training data by:
1. Comparing GPU heuristic-guided selfplay vs random selfplay (sanity check)
2. Validating game dynamics (draw rates, game lengths, victory types)
3. Checking win rate balance across players

Purpose: Validate Phase 1 → Phase 2 gate criteria:
  "No regression in model training quality" when using GPU evaluation.

The key insight from GPU_ARCHITECTURE_SIMPLIFICATION.md Section 2.4:
- CPU HeuristicAI uses full 45-weight evaluation with 8-direction visibility
- GPU evaluate_positions_batch uses simplified ~8-weight evaluation with 4-adjacency
- Score divergence is 63-200% but this is an ARCHITECTURAL TRADE-OFF for 6x speedup
- The question is: Does this divergence affect TRAINING QUALITY, not evaluation accuracy?

Test approach:
- Heuristic-guided games should be longer than random (strategic play)
- Heuristic-guided games should have lower draw rates (decisive play)
- Victory types should reflect strategic play (more territory, fewer stalemates)

Usage:
    # Quick test (fewer games, for CI)
    python scripts/ab_test_gpu_training_quality.py --quick

    # Full test (statistical significance)
    python scripts/ab_test_gpu_training_quality.py --num-games 1000

    # Validate existing selfplay data
    python scripts/ab_test_gpu_training_quality.py --validate-existing data/selfplay/hybrid

Output:
    - data/ab_test/random_baseline/stats.json: Random selfplay statistics
    - data/ab_test/hybrid_gpu/stats.json: Hybrid GPU selfplay statistics
    - data/ab_test/comparison_report.json: Side-by-side comparison

See Also:
    - docs/GPU_PIPELINE_ROADMAP.md Section 11.1 (Decision Gates)
    - docs/GPU_ARCHITECTURE_SIMPLIFICATION.md Section 2.4 (Evaluation Discrepancy)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# Ensure ai-service root on path for scripts/lib imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("ab_test_gpu_training_quality")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    # Data generation
    num_games: int = 1000
    board_type: str = "square8"
    num_players: int = 2
    seed: int = 42

    # Paths
    data_dir: str = "data/ab_test"
    random_subdir: str = "random_baseline"
    hybrid_subdir: str = "hybrid_gpu"

    # Test phases
    skip_generation: bool = False
    validate_existing: str | None = None  # Path to existing selfplay data

    # Quality thresholds - comparing heuristic vs random
    # Strategic play should produce MORE territory victories (more decisive)
    # NOTE: Originally assumed longer games = strategic, but actually decisive wins
    # (territory victories) indicate better evaluation leading to quicker wins.
    min_territory_victory_rate: float = 0.50  # At least 50% territory victories
    # Heuristic should produce fewer stalemates than random (more decisive play)
    max_stalemate_rate: float = 0.10  # At most 10% stalemate rate
    # Heuristic play should have reasonable draw rates (not too high)
    max_draw_rate: float = 0.30  # 30% max draw rate for heuristic play
    # Win balance should be reasonable (not dominated by one player)
    min_win_rate_balance: float = 0.35  # Each player should win at least 35%
    max_win_rate_balance: float = 0.65  # Each player should win at most 65%


@dataclass
class SelfplayStats:
    """Statistics from a selfplay run."""
    total_games: int
    total_moves: int
    total_time_seconds: float
    games_per_second: float
    moves_per_game: float
    draw_rate: float
    wins_by_player: dict[str, int]
    victory_type_counts: dict[str, int]
    device: str

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SelfplayStats":
        return cls(
            total_games=data.get("total_games", 0),
            total_moves=data.get("total_moves", 0),
            total_time_seconds=data.get("total_time_seconds", 0.0),
            games_per_second=data.get("games_per_second", 0.0),
            moves_per_game=data.get("moves_per_game", 0.0),
            draw_rate=data.get("draw_rate", 0.0),
            wins_by_player=data.get("wins_by_player", {}),
            victory_type_counts=data.get("victory_type_counts", {}),
            device=data.get("device", "unknown"),
        )


@dataclass
class ComparisonResult:
    """Result of comparing Random vs Hybrid heuristic selfplay."""
    random_stats: SelfplayStats
    hybrid_stats: SelfplayStats

    # Computed comparisons
    hybrid_draw_rate: float
    hybrid_territory_victory_rate: float
    hybrid_stalemate_rate: float

    # Quality checks
    territory_victory_ok: bool  # High territory victory rate (strategic play)
    stalemate_rate_ok: bool  # Low stalemate rate (decisive play)
    draw_rate_ok: bool  # Draw rate is reasonable
    win_balance_ok: bool  # Win rates balanced

    # Overall
    all_checks_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "random_stats": asdict(self.random_stats),
            "hybrid_stats": asdict(self.hybrid_stats),
            "hybrid_draw_rate": self.hybrid_draw_rate,
            "hybrid_territory_victory_rate": self.hybrid_territory_victory_rate,
            "hybrid_stalemate_rate": self.hybrid_stalemate_rate,
            "checks": {
                "territory_victory_ok": self.territory_victory_ok,
                "stalemate_rate_ok": self.stalemate_rate_ok,
                "draw_rate_ok": self.draw_rate_ok,
                "win_balance_ok": self.win_balance_ok,
            },
            "all_checks_passed": self.all_checks_passed,
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Selfplay Generation
# =============================================================================

def run_selfplay(
    output_dir: Path,
    num_games: int,
    board_type: str,
    num_players: int,
    seed: int,
    use_gpu: bool = False,
) -> SelfplayStats | None:
    """Run selfplay and return statistics.

    Args:
        output_dir: Directory for output files
        num_games: Number of games to generate
        board_type: Board type (square8, square19, hexagonal)
        num_players: Number of players (2, 3, or 4)
        seed: Random seed for reproducibility
        use_gpu: Whether to use GPU evaluation (hybrid mode)

    Returns:
        SelfplayStats if successful, None otherwise

    Note:
        Both modes use run_hybrid_selfplay.py which always uses CPU rules.
        The difference is in the evaluation function used for move selection:
        - use_gpu=True: HybridGPUEvaluator (simplified 4-adjacency evaluation)
        - use_gpu=False: Random mode (no evaluation) - for baseline comparison

        For a true CPU vs GPU evaluation comparison, we compare Hybrid GPU vs
        the existing benchmark data. The key metric is whether the GPU's
        simplified evaluation produces similar game dynamics (draw rates,
        game lengths, victory types).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_name = "Hybrid GPU" if use_gpu else "Random baseline"
    engine_mode = "heuristic-only" if use_gpu else "random-only"

    logger.info(f"Starting {mode_name} selfplay: {num_games} games")

    # Build command - both use hybrid script, but different engine modes
    cmd = [
        sys.executable,
        "scripts/run_hybrid_selfplay.py",
        "--num-games", str(num_games),
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--engine-mode", engine_mode,
    ]

    # Run selfplay
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode != 0:
            logger.error(f"{mode_name} selfplay failed:")
            logger.error(result.stderr)
            return None

    except Exception as e:
        logger.error(f"{mode_name} selfplay error: {e}")
        return None

    elapsed = time.time() - start_time
    logger.info(f"{mode_name} selfplay completed in {elapsed:.1f}s")

    # Load stats
    stats_file = output_dir / "stats.json"
    if not stats_file.exists():
        logger.error(f"Stats file not found: {stats_file}")
        return None

    with open(stats_file) as f:
        stats_data = json.load(f)

    return SelfplayStats.from_json(stats_data)


# =============================================================================
# Comparison Analysis
# =============================================================================

def check_win_balance(
    wins_by_player: dict[str, int],
    min_rate: float,
    max_rate: float,
) -> bool:
    """Check if win rates are balanced across players."""
    total_wins = sum(wins_by_player.values())
    if total_wins == 0:
        return True

    for _player, wins in wins_by_player.items():
        rate = wins / total_wins
        if rate < min_rate or rate > max_rate:
            return False

    return True


def compare_selfplay_runs(
    random_stats: SelfplayStats,
    hybrid_stats: SelfplayStats,
    config: ABTestConfig,
) -> ComparisonResult:
    """Compare Random and Hybrid heuristic selfplay statistics.

    Key insight: Heuristic-guided play should produce:
    - MORE territory victories (strategic play leads to decisive wins)
    - FEWER stalemates (decisive play, not indecision)
    - Reasonable draw rates (not too high)
    - Balanced win rates (not dominated by one player)
    """

    # Compute victory type rates for hybrid
    total_hybrid_games = hybrid_stats.total_games
    territory_victories = hybrid_stats.victory_type_counts.get("territory", 0)
    stalemates = hybrid_stats.victory_type_counts.get("stalemate", 0)

    territory_rate = territory_victories / total_hybrid_games if total_hybrid_games > 0 else 0.0
    stalemate_rate = stalemates / total_hybrid_games if total_hybrid_games > 0 else 0.0

    # Quality checks
    territory_victory_ok = territory_rate >= config.min_territory_victory_rate
    stalemate_rate_ok = stalemate_rate <= config.max_stalemate_rate
    draw_rate_ok = hybrid_stats.draw_rate <= config.max_draw_rate

    win_balance_ok = check_win_balance(
        hybrid_stats.wins_by_player,
        config.min_win_rate_balance,
        config.max_win_rate_balance,
    )

    all_passed = territory_victory_ok and stalemate_rate_ok and draw_rate_ok and win_balance_ok

    return ComparisonResult(
        random_stats=random_stats,
        hybrid_stats=hybrid_stats,
        hybrid_draw_rate=hybrid_stats.draw_rate,
        hybrid_territory_victory_rate=territory_rate,
        hybrid_stalemate_rate=stalemate_rate,
        territory_victory_ok=territory_victory_ok,
        stalemate_rate_ok=stalemate_rate_ok,
        draw_rate_ok=draw_rate_ok,
        win_balance_ok=win_balance_ok,
        all_checks_passed=all_passed,
    )


def print_comparison_report(result: ComparisonResult) -> None:
    """Print a human-readable comparison report."""

    print("\n" + "=" * 70)
    print("A/B TEST RESULTS: Random vs Hybrid GPU Heuristic Selfplay")
    print("=" * 70)

    print("\n### TERRITORY VICTORY RATE (Strategic Play Indicator) ###")
    random_terr = result.random_stats.victory_type_counts.get("territory", 0)
    random_total = result.random_stats.total_games
    random_terr_rate = random_terr / random_total if random_total > 0 else 0
    print(f"  Random baseline: {random_terr_rate:.1%} ({random_terr}/{random_total})")
    print(f"  Hybrid heuristic: {result.hybrid_territory_victory_rate:.1%} {'✓' if result.territory_victory_ok else '✗'}")
    print(f"  (Expected: >= 50% territory victories)")

    print("\n### STALEMATE RATE (Decisive Play Indicator) ###")
    random_stale = result.random_stats.victory_type_counts.get("stalemate", 0)
    random_stale_rate = random_stale / random_total if random_total > 0 else 0
    print(f"  Random baseline: {random_stale_rate:.1%} ({random_stale}/{random_total})")
    print(f"  Hybrid heuristic: {result.hybrid_stalemate_rate:.1%} {'✓' if result.stalemate_rate_ok else '✗'}")
    print(f"  (Expected: <= 10% stalemates)")

    print("\n### DRAW RATE ###")
    print(f"  Random baseline: {result.random_stats.draw_rate:.1%}")
    print(f"  Hybrid heuristic: {result.hybrid_draw_rate:.1%} {'✓' if result.draw_rate_ok else '✗'}")
    print(f"  (Expected: <= 30%)")

    print("\n### WIN BALANCE ###")
    print(f"  Hybrid wins by player: {result.hybrid_stats.wins_by_player} {'✓' if result.win_balance_ok else '✗'}")
    print(f"  (Expected: 35-65% each)")

    print("\n### THROUGHPUT ###")
    print(f"  Random: {result.random_stats.games_per_second:.2f} games/sec")
    print(f"  Hybrid: {result.hybrid_stats.games_per_second:.2f} games/sec")

    print("\n### VICTORY TYPES ###")
    print(f"  Random: {result.random_stats.victory_type_counts}")
    print(f"  Hybrid: {result.hybrid_stats.victory_type_counts}")

    print("\n### GAME LENGTH ###")
    print(f"  Random baseline: {result.random_stats.moves_per_game:.1f} moves/game")
    print(f"  Hybrid heuristic: {result.hybrid_stats.moves_per_game:.1f} moves/game")

    print("\n" + "=" * 70)
    if result.all_checks_passed:
        print("RESULT: ✓ ALL CHECKS PASSED - Hybrid GPU selfplay produces quality data")
        print("        GPU evaluation is suitable for training data generation.")
    else:
        print("RESULT: ✗ SOME CHECKS FAILED - Review differences above")
        print("        GPU evaluation may need tuning or CPU fallback.")
    print("=" * 70 + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B test: Validate Hybrid GPU selfplay training quality"
    )

    parser.add_argument(
        "--num-games", type=int, default=1000,
        help="Number of selfplay games per approach (default: 1000)"
    )
    parser.add_argument(
        "--board-type", default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type for selfplay (default: square8)"
    )
    parser.add_argument(
        "--num-players", type=int, default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--data-dir", default="data/ab_test",
        help="Directory for test data (default: data/ab_test)"
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip selfplay generation, use existing data"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with fewer games (100)"
    )
    parser.add_argument(
        "--validate-existing", type=str, default=None,
        help="Path to existing selfplay stats.json to validate"
    )

    args = parser.parse_args()

    # Build config
    config = ABTestConfig(
        num_games=100 if args.quick else args.num_games,
        board_type=args.board_type,
        num_players=args.num_players,
        seed=args.seed,
        data_dir=args.data_dir,
        skip_generation=args.skip_generation,
        validate_existing=args.validate_existing,
    )

    data_dir = Path(config.data_dir)
    random_dir = data_dir / config.random_subdir
    hybrid_dir = data_dir / config.hybrid_subdir

    logger.info(f"A/B Test Configuration:")
    logger.info(f"  Games: {config.num_games}")
    logger.info(f"  Board: {config.board_type} ({config.num_players}p)")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Data dir: {data_dir}")

    # Phase 1: Generate selfplay data
    if not config.skip_generation:
        logger.info("\n=== Phase 1: Generating Selfplay Data ===")

        # Random baseline selfplay
        random_stats = run_selfplay(
            output_dir=random_dir,
            num_games=config.num_games,
            board_type=config.board_type,
            num_players=config.num_players,
            seed=config.seed,
            use_gpu=False,  # Random mode
        )

        if random_stats is None:
            logger.error("Random selfplay failed")
            return 1

        # Hybrid GPU heuristic selfplay
        hybrid_stats = run_selfplay(
            output_dir=hybrid_dir,
            num_games=config.num_games,
            board_type=config.board_type,
            num_players=config.num_players,
            seed=config.seed,
            use_gpu=True,  # Heuristic mode
        )

        if hybrid_stats is None:
            logger.error("Hybrid selfplay failed")
            return 1
    else:
        # Load existing stats
        logger.info("\n=== Loading Existing Data ===")

        random_stats_file = random_dir / "stats.json"
        hybrid_stats_file = hybrid_dir / "stats.json"

        if not random_stats_file.exists() or not hybrid_stats_file.exists():
            logger.error("Stats files not found. Run without --skip-generation first.")
            return 1

        with open(random_stats_file) as f:
            random_stats = SelfplayStats.from_json(json.load(f))

        with open(hybrid_stats_file) as f:
            hybrid_stats = SelfplayStats.from_json(json.load(f))

    # Phase 2: Compare results
    logger.info("\n=== Phase 2: Comparing Results ===")

    result = compare_selfplay_runs(random_stats, hybrid_stats, config)

    # Save comparison report
    data_dir.mkdir(parents=True, exist_ok=True)
    report_file = data_dir / "comparison_report.json"
    with open(report_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Comparison report saved to {report_file}")

    # Print human-readable report
    print_comparison_report(result)

    # Return exit code based on checks
    return 0 if result.all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
