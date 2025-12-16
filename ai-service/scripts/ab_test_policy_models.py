#!/usr/bin/env python
"""A/B Test: Compare NNUE Policy Models.

This script runs head-to-head matches between different policy models to validate
improvements. Useful for:
1. Comparing new policy model against baseline
2. Validating distillation quality
3. Comparing different model sizes (small/medium/large)
4. Curriculum stage advancement decisions

Usage:
    # Compare two policy models
    python scripts/ab_test_policy_models.py \
        --model-a models/nnue/nnue_policy_square8_2p.pt \
        --model-b models/nnue/nnue_policy_square8_2p_new.pt \
        --num-games 100

    # Compare against no policy (baseline heuristic)
    python scripts/ab_test_policy_models.py \
        --model-a models/nnue/nnue_policy_square8_2p.pt \
        --model-b none \
        --num-games 100

    # Quick test
    python scripts/ab_test_policy_models.py --quick

Output:
    - JSON report with win rates, confidence intervals, and statistical analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.mcts_ai import MCTSAI
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)
from app.game_engine import GameEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single match."""
    game_id: str
    winner: Optional[int]  # 1 or 2, None for draw
    model_a_player: int  # Which player number model A was
    model_b_player: int
    num_moves: int
    game_duration_sec: float
    model_a_think_time_sec: float
    model_b_think_time_sec: float


@dataclass
class ABTestResults:
    """Results of policy A/B test."""
    model_a_path: str
    model_b_path: str
    board_type: str
    num_players: int
    num_games: int

    # Win/loss/draw counts
    model_a_wins: int = 0
    model_b_wins: int = 0
    draws: int = 0

    # Win rates
    model_a_win_rate: float = 0.0
    model_b_win_rate: float = 0.0
    draw_rate: float = 0.0

    # Statistical analysis
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0
    significant_at_95: bool = False

    # Game statistics
    avg_game_length: float = 0.0
    avg_model_a_think_time: float = 0.0
    avg_model_b_think_time: float = 0.0

    # Individual match results
    matches: List[Dict[str, Any]] = field(default_factory=list)

    def compute_statistics(self):
        """Compute win rates and statistical significance."""
        total = self.model_a_wins + self.model_b_wins + self.draws
        if total == 0:
            return

        self.model_a_win_rate = self.model_a_wins / total
        self.model_b_win_rate = self.model_b_wins / total
        self.draw_rate = self.draws / total

        # Compute 95% confidence interval for model A win rate
        # Using Wilson score interval for better small-sample behavior
        n = total
        p = self.model_a_win_rate
        z = 1.96  # 95% confidence

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        spread = z * math.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator

        self.confidence_interval_95 = (
            max(0, center - spread),
            min(1, center + spread)
        )

        # Compute p-value using binomial test (one-sided)
        # H0: model A win rate = 0.5
        # H1: model A win rate > 0.5
        try:
            from scipy import stats
            # Use number of wins vs losses (excluding draws)
            decisive = self.model_a_wins + self.model_b_wins
            if decisive > 0:
                result = stats.binomtest(
                    self.model_a_wins, decisive, p=0.5, alternative='greater'
                )
                self.p_value = result.pvalue
                self.significant_at_95 = self.p_value < 0.05
        except ImportError:
            # Fallback: approximate z-test
            decisive = self.model_a_wins + self.model_b_wins
            if decisive > 0:
                observed_rate = self.model_a_wins / decisive
                z_score = (observed_rate - 0.5) / math.sqrt(0.25 / decisive)
                # One-sided p-value approximation
                self.p_value = 0.5 * (1 - math.erf(z_score / math.sqrt(2)))
                self.significant_at_95 = self.p_value < 0.05


def create_game_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> GameState:
    """Create a fresh game state."""
    size = 8 if board_type == BoardType.SQUARE8 else 19
    if board_type == BoardType.HEXAGONAL:
        size = 5

    board = BoardState(
        type=board_type,
        size=size,
        stacks={},
        markers={},
        collapsed_spaces={},
        eliminated_rings={},
    )

    rings_per_player = 20 if num_players == 2 else (14 if num_players == 3 else 10)
    players = []
    for i in range(num_players):
        players.append(
            Player(
                id=f"p{i+1}",
                username=f"AI{i+1}",
                type="ai",
                player_number=i + 1,
                is_ready=True,
                time_remaining=600000,
                ai_difficulty=4,
                rings_in_hand=rings_per_player,
                eliminated_rings=0,
                territory_spaces=0,
            )
        )

    return GameState(
        id=str(uuid.uuid4()),
        board_type=board_type,
        board=board,
        players=players,
        current_phase=GamePhase.RING_PLACEMENT,
        current_player=1,
        move_history=[],
        time_control=TimeControl(initial_time=600, increment=5, type="standard"),
        game_status=GameStatus.ACTIVE,
        created_at=datetime.now(),
        last_move_at=datetime.now(),
        is_rated=False,
        max_players=num_players,
        total_rings_in_play=0,
        total_rings_eliminated=0,
        victory_threshold=3,
        territory_victory_threshold=10,
        chain_capture_state=None,
    )


def create_mcts_ai(
    player_number: int,
    policy_model_path: Optional[str],
    think_time_ms: int = 500,
) -> MCTSAI:
    """Create MCTS AI with optional policy model."""
    config = AIConfig(
        difficulty=4,
        think_time=think_time_ms,
        use_neural_net=False,  # Don't use full neural net
        use_nnue_policy_priors=(policy_model_path is not None),
    )
    ai = MCTSAI(player_number=player_number, config=config)

    # Manually load policy model if specified
    if policy_model_path and os.path.exists(policy_model_path):
        try:
            from app.ai.nnue_policy import RingRiftNNUEWithPolicy
            import torch
            import re

            checkpoint = torch.load(policy_model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint
            hidden_dim = 256
            num_hidden_layers = 2

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                hidden_dim = int(checkpoint.get("hidden_dim") or hidden_dim)
                num_hidden_layers = int(checkpoint.get("num_hidden_layers") or num_hidden_layers)

            # Infer from state dict
            if isinstance(state_dict, dict):
                try:
                    accumulator_weight = state_dict.get("accumulator.weight")
                    if accumulator_weight is not None and hasattr(accumulator_weight, "shape"):
                        hidden_dim = int(accumulator_weight.shape[0])
                except Exception:
                    pass

                try:
                    layer_indices = set()
                    for key in state_dict:
                        match = re.match(r"hidden\.(\d+)\.weight$", key)
                        if match:
                            layer_indices.add(int(match.group(1)))
                    if layer_indices:
                        num_hidden_layers = len(layer_indices)
                except Exception:
                    pass

            ai.nnue_policy_model = RingRiftNNUEWithPolicy(
                board_type=BoardType.SQUARE8,  # Will be updated based on game
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            if not isinstance(state_dict, dict):
                raise TypeError(f"Unexpected checkpoint: {type(state_dict).__name__}")
            ai.nnue_policy_model.load_state_dict(state_dict)
            ai.nnue_policy_model.eval()
            ai._pending_nnue_policy_init = False
            logger.debug(f"Loaded policy model: {policy_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load policy model {policy_model_path}: {e}")

    return ai


def play_match(
    engine: GameEngine,
    ai_a: MCTSAI,
    ai_b: MCTSAI,
    model_a_player: int,
    max_moves: int = 10000,
) -> MatchResult:
    """Play a single match between two AIs."""
    game_state = create_game_state()
    game_id = game_state.id

    model_a_think_time = 0.0
    model_b_think_time = 0.0
    start_time = time.time()
    num_moves = 0

    while game_state.game_status == GameStatus.ACTIVE and num_moves < max_moves:
        current_player = game_state.current_player

        # Select AI based on player
        if current_player == model_a_player:
            ai = ai_a
            ai.player_number = current_player
        else:
            ai = ai_b
            ai.player_number = current_player

        # Get move
        move_start = time.time()
        try:
            move = ai.select_move(game_state)
            move_time = time.time() - move_start

            if current_player == model_a_player:
                model_a_think_time += move_time
            else:
                model_b_think_time += move_time

            if move is None:
                break

            game_state = engine.apply_move(game_state, move)
            num_moves += 1
        except Exception as e:
            logger.warning(f"Move failed: {e}")
            break

    game_duration = time.time() - start_time

    # Use the game state's winner field (set by game engine on completion)
    winner = game_state.winner

    return MatchResult(
        game_id=game_id,
        winner=winner,
        model_a_player=model_a_player,
        model_b_player=3 - model_a_player,  # 2 if a is 1, 1 if a is 2
        num_moves=num_moves,
        game_duration_sec=game_duration,
        model_a_think_time_sec=model_a_think_time,
        model_b_think_time_sec=model_b_think_time,
    )


def run_ab_test(
    model_a_path: str,
    model_b_path: Optional[str],
    num_games: int,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    think_time_ms: int = 500,
    max_moves: int = 2000,
) -> ABTestResults:
    """Run A/B test between two policy models."""
    logger.info(f"Starting A/B test:")
    logger.info(f"  Model A: {model_a_path}")
    logger.info(f"  Model B: {model_b_path or 'none (baseline)'}")
    logger.info(f"  Games: {num_games}")

    engine = GameEngine()
    results = ABTestResults(
        model_a_path=model_a_path,
        model_b_path=model_b_path or "none",
        board_type=board_type.value,
        num_players=num_players,
        num_games=num_games,
    )

    total_game_length = 0
    total_model_a_time = 0.0
    total_model_b_time = 0.0

    for i in range(num_games):
        # Alternate which player model A is to eliminate first-move advantage
        model_a_player = 1 if i % 2 == 0 else 2

        # Create fresh AIs for each game
        ai_a = create_mcts_ai(model_a_player, model_a_path, think_time_ms)
        ai_b = create_mcts_ai(3 - model_a_player, model_b_path, think_time_ms)

        # Play match
        match_result = play_match(
            engine, ai_a, ai_b, model_a_player, max_moves
        )

        # Record result
        if match_result.winner is None:
            results.draws += 1
        elif match_result.winner == model_a_player:
            results.model_a_wins += 1
        else:
            results.model_b_wins += 1

        total_game_length += match_result.num_moves
        total_model_a_time += match_result.model_a_think_time_sec
        total_model_b_time += match_result.model_b_think_time_sec

        results.matches.append(asdict(match_result))

        # Progress update
        if (i + 1) % 10 == 0:
            logger.info(
                f"Progress: {i+1}/{num_games} games | "
                f"A wins: {results.model_a_wins} | "
                f"B wins: {results.model_b_wins} | "
                f"Draws: {results.draws}"
            )

    # Compute final statistics
    results.avg_game_length = total_game_length / num_games
    results.avg_model_a_think_time = total_model_a_time / num_games
    results.avg_model_b_think_time = total_model_b_time / num_games
    results.compute_statistics()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="A/B test NNUE policy models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-a",
        type=str,
        default="models/nnue/nnue_policy_square8_2p.pt",
        help="Path to policy model A (default: models/nnue/nnue_policy_square8_2p.pt)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Path to policy model B (default: none = baseline without policy)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games to play (default: 50)",
    )
    parser.add_argument(
        "--think-time",
        type=int,
        default=500,
        help="Think time per move in ms (default: 500)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        help="Maximum moves per game (default: 300)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer games (10)",
    )

    args = parser.parse_args()

    if args.quick:
        args.num_games = 10
        args.think_time = 500  # 500ms for quick mode (still fast but games complete)

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map[args.board_type]

    # Check model A exists
    if not os.path.exists(args.model_a):
        logger.error(f"Model A not found: {args.model_a}")
        return 1

    # Run test
    results = run_ab_test(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        num_games=args.num_games,
        board_type=board_type,
        think_time_ms=args.think_time,
        max_moves=args.max_moves,
    )

    # Output results - convert numpy types to Python natives for JSON serialization
    def convert_numpy(obj):
        """Recursively convert numpy types to Python natives."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(v) for v in obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    report = convert_numpy(asdict(results))
    report_json = json.dumps(report, indent=2)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report_json)
        logger.info(f"Report saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)
        print(f"Model A: {results.model_a_path}")
        print(f"Model B: {results.model_b_path}")
        print(f"Games: {results.num_games}")
        print()
        print(f"Model A wins: {results.model_a_wins} ({results.model_a_win_rate:.1%})")
        print(f"Model B wins: {results.model_b_wins} ({results.model_b_win_rate:.1%})")
        print(f"Draws: {results.draws} ({results.draw_rate:.1%})")
        print()
        print(f"95% CI for A win rate: [{results.confidence_interval_95[0]:.1%}, {results.confidence_interval_95[1]:.1%}]")
        print(f"P-value (A > B): {results.p_value:.4f}")
        print(f"Significant at 95%: {results.significant_at_95}")
        print()
        print(f"Avg game length: {results.avg_game_length:.1f} moves")
        print(f"Avg Model A think time: {results.avg_model_a_think_time:.2f}s")
        print(f"Avg Model B think time: {results.avg_model_b_think_time:.2f}s")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
