#!/usr/bin/env python3
"""Elo System Calibration and Dashboard.

This script:
1. Registers baseline participants (Random @ 400, Heuristic @ ~1200)
2. Re-registers all canonical models with proper AI method distinction
3. Runs calibration games against baselines
4. Displays a unified Elo dashboard

Elo Calibration Philosophy:
- Random AI is ANCHORED at 400 Elo (fixed, never changes)
- All other ratings are relative to this anchor
- Heuristic starts at ~1200 but can drift based on matches
- Each (NN, Algorithm) combination is a distinct participant

Composite Participant Format:
    {nn_id}:{ai_type}:{config_hash}

Examples:
    none:random:d1                    # Random baseline @ 400
    none:heuristic:d5                 # Heuristic @ ~1200
    canonical_hex8_2p:policy_only:t0.3   # Pure NN, no search
    canonical_hex8_2p:gumbel_mcts:b200   # NN + Gumbel MCTS
    canonical_hex8_2p:mcts:s800          # NN + Standard MCTS

Usage:
    # Show current Elo dashboard
    python scripts/calibrate_elo_system.py --dashboard

    # Register all participants with proper IDs
    python scripts/calibrate_elo_system.py --register

    # Run calibration games (50 games per baseline)
    python scripts/calibrate_elo_system.py --calibrate --games 50

    # Full calibration (register + calibrate + dashboard)
    python scripts/calibrate_elo_system.py --full --games 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.thresholds import (
    BASELINE_ELO_HEURISTIC,
    BASELINE_ELO_HEURISTIC_STRONG,
    BASELINE_ELO_RANDOM,
    INITIAL_ELO_RATING,
)
from app.training.composite_participant import (
    make_composite_participant_id,
    parse_composite_participant_id,
    is_composite_id,
    ParticipantCategory,
)
from app.training.elo_service import get_elo_service, EloService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

ALL_CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# Baseline participants (non-NN)
BASELINES = {
    "none:random:d1": {
        "ai_type": "random",
        "config": {"difficulty": 1},
        "target_elo": BASELINE_ELO_RANDOM,  # 400 - ANCHORED
        "description": "Random baseline (anchor point)",
    },
    "none:heuristic:d5": {
        "ai_type": "heuristic",
        "config": {"difficulty": 5},
        "target_elo": BASELINE_ELO_HEURISTIC,  # ~1200
        "description": "Standard heuristic",
    },
    "none:heuristic:d8": {
        "ai_type": "heuristic",
        "config": {"difficulty": 8},
        "target_elo": BASELINE_ELO_HEURISTIC_STRONG,  # ~1400
        "description": "Strong heuristic",
    },
}

# AI methods to track for each NN
# Note: Config hash uses 'p' for decimal point (e.g., t0p3 = temperature 0.3)
AI_METHODS = {
    "policy_only": {
        "config": {"temperature": 0.3},
        "description": "Pure NN policy (no search)",
        "hash": "t0p3",  # 'p' replaces '.' in config hash encoding
    },
    "gumbel_mcts": {
        "config": {"budget": 200},
        "description": "NN + Gumbel MCTS (budget=200)",
        "hash": "b200",
    },
    "mcts": {
        "config": {"simulations": 800},
        "description": "NN + Standard MCTS (800 sims)",
        "hash": "s800",
    },
    "descent": {
        "config": {"difficulty": 6},
        "description": "NN + Gradient descent (d=6)",
        "hash": "d6",
    },
}


# ============================================================================
# Participant Registration
# ============================================================================

def register_baselines(elo: EloService) -> dict[str, str]:
    """Register baseline participants (Random, Heuristic) for all configs."""
    registered = {}

    for (board_type, num_players) in ALL_CONFIGS:
        config_key = f"{board_type}_{num_players}p"

        for pid, baseline_info in BASELINES.items():
            try:
                # Register composite participant
                full_pid = elo.register_composite_participant(
                    nn_id=None,  # No NN for baselines
                    ai_type=baseline_info["ai_type"],
                    config=baseline_info["config"],
                    board_type=board_type,
                    num_players=num_players,
                )
                registered[f"{config_key}:{pid}"] = full_pid
                logger.debug(f"Registered baseline {pid} for {config_key}")
            except Exception as e:
                logger.warning(f"Failed to register {pid} for {config_key}: {e}")

    logger.info(f"Registered {len(registered)} baseline participants")
    return registered


def register_models_with_methods(elo: EloService) -> dict[str, list[str]]:
    """Register all canonical models with each AI method."""
    registered = {}

    for (board_type, num_players) in ALL_CONFIGS:
        config_key = f"{board_type}_{num_players}p"
        model_id = f"canonical_{config_key}"
        model_path = Path(f"models/canonical_{config_key}.pth")

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        registered[model_id] = []

        for ai_type, method_info in AI_METHODS.items():
            try:
                full_pid = elo.register_composite_participant(
                    nn_id=model_id,
                    ai_type=ai_type,
                    config=method_info["config"],
                    board_type=board_type,
                    num_players=num_players,
                    nn_model_path=str(model_path),
                )
                registered[model_id].append(full_pid)
                logger.debug(f"Registered {full_pid}")
            except Exception as e:
                logger.warning(f"Failed to register {model_id}:{ai_type}: {e}")

        logger.info(f"Registered {model_id} with {len(registered[model_id])} AI methods")

    return registered


# ============================================================================
# Calibration Games
# ============================================================================

def run_calibration_games(
    elo: EloService,
    model_id: str,
    board_type: str,
    num_players: int,
    games_per_baseline: int = 30,
) -> dict[str, dict]:
    """Run calibration games for a model against baselines."""
    from app.training.game_gauntlet import (
        run_baseline_gauntlet,
        BaselineOpponent,
    )

    model_path = Path(f"models/{model_id}.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Run gauntlet evaluation
    results = run_baseline_gauntlet(
        model_path=str(model_path),
        board_type=board_type,
        num_players=num_players,
        opponents=[
            BaselineOpponent.RANDOM,
            BaselineOpponent.HEURISTIC,
        ],
        games_per_opponent=games_per_baseline,
    )

    # Record results to Elo system
    calibration_results = {}

    for opponent, result in results.items():
        wins = result.get("wins", 0)
        losses = result.get("losses", 0)
        draws = result.get("draws", 0)

        # Map opponent to composite ID
        if opponent == BaselineOpponent.RANDOM:
            baseline_pid = "none:random:d1"
        elif opponent == BaselineOpponent.HEURISTIC:
            baseline_pid = "none:heuristic:d5"
        else:
            continue

        # Record each game
        for _ in range(wins):
            # Model wins, baseline loses
            elo.record_match(
                participant_ids=[f"{model_id}:gumbel_mcts:b200", baseline_pid],
                winner_id=f"{model_id}:gumbel_mcts:b200",
                board_type=board_type,
                num_players=num_players,
            )

        for _ in range(losses):
            # Baseline wins, model loses
            elo.record_match(
                participant_ids=[f"{model_id}:gumbel_mcts:b200", baseline_pid],
                winner_id=baseline_pid,
                board_type=board_type,
                num_players=num_players,
            )

        calibration_results[opponent.name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / (wins + losses + draws) if (wins + losses + draws) > 0 else 0,
        }

    return calibration_results


# ============================================================================
# Dashboard
# ============================================================================

@dataclass
class ModelEloSummary:
    """Summary of Elo ratings for a model across all AI methods."""
    model_id: str
    board_type: str
    num_players: int
    methods: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)
    best_method: str = ""
    best_elo: float = 0.0
    unified_elo: float = 0.0  # Weighted average or best

    def calculate_unified_elo(self):
        """Calculate unified Elo as the best method's Elo.

        Priority:
        1. Best composite method with games played > 0
        2. Legacy rating if no composite methods have games
        3. Default 1500 if nothing else available
        """
        if not self.methods:
            return

        # Filter to methods with games played
        methods_with_games = {
            m: elo for m, elo in self.methods.items()
            if self.games_played.get(m, 0) > 0
        }

        if methods_with_games:
            # Use best method that has actual games
            best_method = max(methods_with_games.items(), key=lambda x: x[1])
            self.best_method = best_method[0]
            self.best_elo = best_method[1]
            self.unified_elo = self.best_elo
        elif "legacy" in self.methods and self.games_played.get("legacy", 0) > 0:
            # Fall back to legacy rating
            self.best_method = "legacy"
            self.best_elo = self.methods["legacy"]
            self.unified_elo = self.best_elo
        else:
            # No games played yet, use default
            self.best_method = ""
            self.best_elo = INITIAL_ELO_RATING
            self.unified_elo = INITIAL_ELO_RATING


def get_elo_dashboard(elo: EloService) -> dict[str, ModelEloSummary]:
    """Generate Elo dashboard data for all models."""
    dashboard = {}

    for (board_type, num_players) in ALL_CONFIGS:
        config_key = f"{board_type}_{num_players}p"
        model_id = f"canonical_{config_key}"

        summary = ModelEloSummary(
            model_id=model_id,
            board_type=board_type,
            num_players=num_players,
        )

        # Get Elo for each AI method
        for ai_type, method_info in AI_METHODS.items():
            composite_id = f"{model_id}:{ai_type}:{method_info['hash']}"

            try:
                rating = elo.get_rating(composite_id, board_type, num_players)
                if rating:
                    summary.methods[ai_type] = rating.rating
                    summary.games_played[ai_type] = rating.games_played
            except (KeyError, ValueError, AttributeError):
                pass

        # Also check legacy format (non-composite)
        try:
            legacy_rating = elo.get_rating(model_id, board_type, num_players)
            if legacy_rating and legacy_rating.games_played > 0:
                summary.methods["legacy"] = legacy_rating.rating
                summary.games_played["legacy"] = legacy_rating.games_played
        except (KeyError, ValueError, AttributeError):
            pass

        summary.calculate_unified_elo()
        dashboard[config_key] = summary

    return dashboard


def print_dashboard(dashboard: dict[str, ModelEloSummary], show_methods: bool = True):
    """Print the Elo dashboard."""
    print("\n" + "=" * 80)
    print("                    RINGRIFT ELO DASHBOARD")
    print("=" * 80)
    print(f"  Anchor: Random AI @ {BASELINE_ELO_RANDOM} Elo (fixed)")
    print(f"  Reference: Heuristic @ ~{BASELINE_ELO_HEURISTIC} Elo")
    print("=" * 80)

    # Group by board type
    by_board = {}
    for config_key, summary in dashboard.items():
        board = summary.board_type
        if board not in by_board:
            by_board[board] = []
        by_board[board].append(summary)

    for board_type, summaries in sorted(by_board.items()):
        print(f"\n  {board_type.upper()}")
        print(f"  {'-' * 70}")

        for summary in sorted(summaries, key=lambda x: x.num_players):
            config_key = f"{summary.board_type}_{summary.num_players}p"

            if summary.unified_elo > 0:
                elo_vs_random = summary.unified_elo - BASELINE_ELO_RANDOM
                elo_vs_heuristic = summary.unified_elo - BASELINE_ELO_HEURISTIC

                # Determine strength tier
                if summary.unified_elo >= 2000:
                    tier = "Master"
                elif summary.unified_elo >= 1700:
                    tier = "Strong"
                elif summary.unified_elo >= 1400:
                    tier = "Intermediate"
                elif summary.unified_elo >= 1100:
                    tier = "Basic"
                else:
                    tier = "Weak"

                print(f"    {config_key:15s} | {summary.unified_elo:6.0f} Elo | "
                      f"+{elo_vs_random:4.0f} vs Random | +{elo_vs_heuristic:4.0f} vs Heuristic | "
                      f"{tier}")

                if show_methods and summary.methods:
                    for method, elo_val in sorted(summary.methods.items(), key=lambda x: -x[1]):
                        games = summary.games_played.get(method, 0)
                        best_marker = " (best)" if method == summary.best_method else ""
                        print(f"      └─ {method:15s}: {elo_val:6.0f} Elo ({games} games){best_marker}")
            else:
                print(f"    {config_key:15s} | No Elo data")

    print("\n" + "=" * 80)

    # Summary stats
    total_with_elo = sum(1 for s in dashboard.values() if s.unified_elo > 0)
    avg_elo = sum(s.unified_elo for s in dashboard.values() if s.unified_elo > 0)
    if total_with_elo > 0:
        avg_elo /= total_with_elo
        avg_vs_random = avg_elo - BASELINE_ELO_RANDOM
        print(f"  Summary: {total_with_elo}/12 configs with Elo data")
        print(f"  Average Elo: {avg_elo:.0f} (+{avg_vs_random:.0f} vs Random)")
    else:
        print("  Summary: No Elo data available")

    print("=" * 80 + "\n")


def print_json_dashboard(dashboard: dict[str, ModelEloSummary]):
    """Print dashboard as JSON."""
    output = {}
    for config_key, summary in dashboard.items():
        output[config_key] = {
            "model_id": summary.model_id,
            "unified_elo": summary.unified_elo,
            "best_method": summary.best_method,
            "best_elo": summary.best_elo,
            "methods": summary.methods,
            "games_played": summary.games_played,
            "vs_random": summary.unified_elo - BASELINE_ELO_RANDOM if summary.unified_elo > 0 else None,
            "vs_heuristic": summary.unified_elo - BASELINE_ELO_HEURISTIC if summary.unified_elo > 0 else None,
        }
    print(json.dumps(output, indent=2))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Elo System Calibration and Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Show Elo dashboard",
    )
    parser.add_argument(
        "--register", "-r",
        action="store_true",
        help="Register all participants (baselines + models with AI methods)",
    )
    parser.add_argument(
        "--calibrate", "-c",
        action="store_true",
        help="Run calibration games against baselines",
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Full calibration (register + calibrate + dashboard)",
    )
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=30,
        help="Games per baseline for calibration (default: 30)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config to calibrate (e.g., hex8_2p)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output dashboard as JSON",
    )
    parser.add_argument(
        "--no-methods",
        action="store_true",
        help="Don't show per-method breakdown in dashboard",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get Elo service
    elo = get_elo_service()

    # Default to dashboard if no action specified
    if not any([args.dashboard, args.register, args.calibrate, args.full]):
        args.dashboard = True

    # Full mode sets all actions
    if args.full:
        args.register = True
        args.calibrate = True
        args.dashboard = True

    # Register participants
    if args.register:
        logger.info("Registering baseline participants...")
        register_baselines(elo)

        logger.info("Registering models with AI methods...")
        register_models_with_methods(elo)

    # Run calibration
    if args.calibrate:
        configs_to_calibrate = ALL_CONFIGS

        if args.config:
            # Parse specific config
            parts = args.config.replace("_", " ").replace("p", "").split()
            if len(parts) == 2:
                board_type = parts[0]
                num_players = int(parts[1])
                configs_to_calibrate = [(board_type, num_players)]
            else:
                logger.error(f"Invalid config format: {args.config}")
                return 1

        for board_type, num_players in configs_to_calibrate:
            config_key = f"{board_type}_{num_players}p"
            model_id = f"canonical_{config_key}"

            model_path = Path(f"models/{model_id}.pth")
            if not model_path.exists():
                logger.warning(f"Skipping {config_key}: model not found")
                continue

            logger.info(f"Calibrating {config_key} ({args.games} games per baseline)...")
            try:
                results = run_calibration_games(
                    elo=elo,
                    model_id=model_id,
                    board_type=board_type,
                    num_players=num_players,
                    games_per_baseline=args.games,
                )
                for baseline, result in results.items():
                    logger.info(f"  vs {baseline}: {result['win_rate']:.1%} win rate")
            except Exception as e:
                logger.error(f"Failed to calibrate {config_key}: {e}")

    # Show dashboard
    if args.dashboard:
        dashboard = get_elo_dashboard(elo)

        if args.json:
            print_json_dashboard(dashboard)
        else:
            print_dashboard(dashboard, show_methods=not args.no_methods)

    return 0


if __name__ == "__main__":
    sys.exit(main())
