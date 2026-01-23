#!/usr/bin/env python3
"""
Reanalyze existing games to add MCTS policy distributions.

This script reads existing JSONL game files and backfills MCTS policy
distributions for each move using MCTS search. This enables KL-divergence
training on historical data that was generated without MCTS.

Usage:
    python scripts/reanalyze_mcts_policy.py \
        --input data/selfplay/gpu/games.jsonl \
        --output data/selfplay/reanalyzed/games.jsonl \
        --mcts-sims 100 \
        --max-games 1000

Environment:
    PYTHONPATH must include the ai-service directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hashlib

from app.mcts.improved_mcts import GameState as MCTSGameState
from app.models import BoardType, GameState, Move
from app.rules.default_engine import DefaultRulesEngine
from app.training.generate_data import create_initial_state

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("reanalyze_mcts_policy")


class GameStateAdapter(MCTSGameState):
    """Adapts app.models.GameState to MCTS GameState interface.

    MCTS expects integer move indices, but our game uses Move objects.
    This adapter maintains a bidirectional mapping between them.
    """

    def __init__(self, real_state, engine, move_list=None):
        """
        Args:
            real_state: The actual GameState from app.models
            engine: The rules engine for move generation/application
            move_list: Optional pre-computed list of legal moves
        """
        self.real_state = real_state
        self.engine = engine
        self._current_player = real_state.current_player or 1
        if move_list is not None:
            self._legal_moves = move_list
        else:
            self._legal_moves = engine.get_valid_moves(real_state, self._current_player)

    def get_legal_moves(self) -> list[int]:
        """Return indices [0, 1, 2, ...] for legal moves."""
        return list(range(len(self._legal_moves)))

    def apply_move(self, move_idx: int) -> 'GameStateAdapter':
        """Apply move by index and return new adapted state."""
        if move_idx < 0 or move_idx >= len(self._legal_moves):
            raise ValueError(f"Invalid move index {move_idx}, have {len(self._legal_moves)} moves")
        move = self._legal_moves[move_idx]
        new_real_state = self.engine.apply_move(
            self.real_state.copy(deep=True), move  # Pydantic v1
        )
        return GameStateAdapter(new_real_state, self.engine)

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.real_state.game_status != "active"

    def get_outcome(self, player: int) -> float:
        """Get outcome for specified player."""
        if self.real_state.winner == player:
            return 1.0
        elif self.real_state.winner is not None:
            return -1.0
        return 0.0

    def current_player(self) -> int:
        """Get current player (0-indexed for MCTS)."""
        return (self._current_player - 1) % 2

    def hash(self) -> str:
        """Generate unique hash for transposition table."""
        board_str = str(self.real_state.board)
        player_str = str(self._current_player)
        return hashlib.md5(f"{board_str}:{player_str}".encode()).hexdigest()

    def get_move_by_index(self, idx: int):
        """Get the actual Move object for an index."""
        if 0 <= idx < len(self._legal_moves):
            return self._legal_moves[idx]
        return None


class HeuristicNetworkWrapper:
    """Wraps heuristic evaluator as a neural network interface for MCTS."""

    def __init__(self, evaluator, engine, board_size: int):
        self.evaluator = evaluator
        self.engine = engine
        self.board_size = board_size

    def evaluate(self, state: GameStateAdapter) -> tuple:
        """Return uniform policy over legal moves and heuristic value.

        Policy indices match the legal move indices from GameStateAdapter.
        """
        # Get real state and legal moves from adapter
        real_state = state.real_state
        current_player = state._current_player
        legal_moves = state._legal_moves
        num_legal = len(legal_moves)

        # Create policy - indexed by move position in legal_moves
        max_policy_size = max(num_legal, self.board_size * self.board_size * 4)
        policy = [0.0] * max_policy_size

        if num_legal > 0:
            prob = 1.0 / num_legal
            for i in range(num_legal):
                policy[i] = prob

        # Use heuristic evaluation for value
        try:
            move_scores = self.evaluator.evaluate_moves(
                real_state, legal_moves, current_player, self.engine
            )
            if move_scores:
                scores = [s for _, s in move_scores]
                value = sum(scores) / len(scores) / 1000.0
                value = max(-1.0, min(1.0, value))
            else:
                value = 0.0
        except (AttributeError, TypeError, ValueError):
            value = 0.0

        return policy, value


def parse_move_dict(move_dict: dict[str, Any], move_number: int) -> Move | None:
    """Parse a move dict into a Move object with defaults for missing fields.

    Returns None for unknown move types.
    """
    from datetime import datetime

    from app.models import MoveType, Position

    move_type_str = str(move_dict.get("type") or "").strip()
    if not move_type_str:
        return None

    # Skip unknown move types
    if move_type_str.startswith("unknown_"):
        return None

    try:
        move_type = MoveType(move_type_str)
    except ValueError:
        return None

    # Parse positions
    def parse_pos(pos_dict):
        if not pos_dict or not isinstance(pos_dict, dict):
            return None
        return Position(
            x=pos_dict.get("x", 0),
            y=pos_dict.get("y", 0),
            z=pos_dict.get("z"),
        )

    from_pos = parse_pos(move_dict.get("from") or move_dict.get("from_pos"))
    to_pos = parse_pos(move_dict.get("to"))
    capture_target = parse_pos(move_dict.get("capture_target") or move_dict.get("captureTarget"))

    return Move(
        id=move_dict.get("id", f"reanalyzed-{move_number}"),
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        timestamp=move_dict.get("timestamp", datetime.now()),
        think_time=move_dict.get("think_time", move_dict.get("thinkTime", 0)),
        move_number=move_dict.get("move_number", move_dict.get("moveNumber", move_number + 1)),
    )


def parse_board_type(board_str: str) -> BoardType:
    """Parse board type string to BoardType enum."""
    board_str = board_str.lower()
    if "square8" in board_str or "sq8" in board_str:
        return BoardType.SQUARE8
    elif "square19" in board_str or "sq19" in board_str:
        return BoardType.SQUARE19
    elif "hex" in board_str:
        return BoardType.HEXAGONAL
    else:
        return BoardType.SQUARE8


def get_board_size(board_type: BoardType) -> int:
    """Get board size for board type."""
    if board_type == BoardType.SQUARE8:
        return 8
    elif board_type == BoardType.SQUARE19:
        return 19
    elif board_type == BoardType.HEXAGONAL:
        return 11  # Standard hex board
    return 8


def reanalyze_game(
    game: dict[str, Any],
    mcts,
    engine: DefaultRulesEngine,
    sample_every: int = 1,
) -> dict[str, Any]:
    """Reanalyze a single game to add MCTS policy distributions.

    Args:
        game: Game dict with 'initial_state', 'moves', etc.
        mcts: MCTS instance
        engine: Rules engine
        sample_every: Only analyze every Nth move to save time

    Returns:
        Game dict with 'mcts_policy' added to sampled moves
    """
    # Parse initial state
    board_type_str = game.get("board_type", "square8")
    num_players = game.get("num_players", 2)
    board_type = parse_board_type(board_type_str)

    # Create initial state
    initial_state_dict = game.get("initial_state")
    if initial_state_dict:
        try:
            state = GameState(**initial_state_dict)
        except (TypeError, ValueError, KeyError):
            state = create_initial_state(board_type, num_players)
    else:
        state = create_initial_state(board_type, num_players)

    moves = game.get("moves", [])
    reanalyzed_moves = []

    for i, move_dict in enumerate(moves):
        # Create new move dict (copy existing fields)
        new_move = dict(move_dict)

        # Only analyze every Nth move
        if i % sample_every == 0 and state.game_status == "active":
            try:
                # Get valid moves for this position
                current_player = state.current_player or 1
                valid_moves = engine.get_valid_moves(state, current_player)

                if valid_moves:
                    # Wrap state in adapter for MCTS
                    adapted_state = GameStateAdapter(state, engine, valid_moves)

                    # Run MCTS search to get policy
                    mcts.search(adapted_state, add_noise=False)
                    policy = mcts.get_policy(temperature=1.0)

                    # Convert to sparse dict (only non-zero probs)
                    # Indices are positions in valid_moves list
                    mcts_policy = {
                        idx: prob for idx, prob in enumerate(policy)
                        if prob > 1e-6
                    }

                    if mcts_policy:
                        new_move["mcts_policy"] = mcts_policy

            except Exception as e:
                logger.debug(f"Failed to analyze move {i}: {e}")

        reanalyzed_moves.append(new_move)

        # Apply move to advance state
        try:
            # Parse move with defaults for missing required fields
            move = parse_move_dict(move_dict, i)
            if move is not None:
                state = engine.apply_move(state, move)
            else:
                logger.debug(f"Skipping unknown move type at {i}")
        except Exception as e:
            logger.debug(f"Failed to apply move {i}: {e}")
            break

    # Create output game dict
    output_game = dict(game)
    output_game["moves"] = reanalyzed_moves
    output_game["reanalyzed"] = True

    return output_game


def main():
    parser = argparse.ArgumentParser(description="Reanalyze games to add MCTS policy")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSONL file with games",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output JSONL file with reanalyzed games",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="Only analyze every Nth move to save time (default: 5)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games to process (default: all)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default=None,
        help="Filter to specific board type (square8, square19, hexagonal)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=None,
        help="Filter to specific number of players",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    logger.info("Initializing MCTS and evaluator...")

    from app.ai.hybrid_gpu import create_hybrid_evaluator
    from app.mcts.improved_mcts import ImprovedMCTS, MCTSConfig

    engine = DefaultRulesEngine()
    evaluator = create_hybrid_evaluator(
        board_type=args.board_type or "square8",
        num_players=args.num_players or 2,
        prefer_gpu=False  # Use CPU for reanalysis to avoid memory issues
    )

    # Track statistics
    games_processed = 0
    games_skipped = 0
    moves_analyzed = 0
    start_time = time.time()

    # Process games
    logger.info(f"Processing {args.input}...")

    mode = "a" if args.append else "w"
    with open(args.input) as fin, open(args.output, mode) as fout:
        for line_num, line in enumerate(fin):
            if not line.strip():
                continue

            try:
                game = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Apply filters
            if args.board_type:
                game_board = game.get("board_type", "").lower()
                if args.board_type.lower() not in game_board:
                    games_skipped += 1
                    continue

            if args.num_players and game.get("num_players") != args.num_players:
                games_skipped += 1
                continue

            # Check max games
            if args.max_games and games_processed >= args.max_games:
                break

            # Get board type for this game
            board_type = parse_board_type(game.get("board_type", "square8"))
            board_size = get_board_size(board_type)

            # Create MCTS for this game's board type
            heuristic_network = HeuristicNetworkWrapper(evaluator, engine, board_size)
            mcts_config = MCTSConfig(
                num_simulations=args.mcts_sims,
                cpuct=1.414,
                root_dirichlet_alpha=0.3,
                root_noise_weight=0.0,  # No noise for reanalysis
            )
            mcts = ImprovedMCTS(network=heuristic_network, config=mcts_config)

            # Reanalyze game
            try:
                reanalyzed = reanalyze_game(
                    game, mcts, engine, sample_every=args.sample_every
                )
                fout.write(json.dumps(reanalyzed) + "\n")
                games_processed += 1
                moves_analyzed += len([m for m in reanalyzed.get("moves", []) if "mcts_policy" in m])

                # Progress logging
                if games_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = games_processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Processed {games_processed} games "
                        f"({moves_analyzed} moves analyzed, {rate:.1f} games/s)"
                    )

            except Exception as e:
                logger.warning(f"Failed to reanalyze game {line_num}: {e}")
                games_skipped += 1

    # Final stats
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("REANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Games processed: {games_processed}")
    logger.info(f"Games skipped: {games_skipped}")
    logger.info(f"Moves analyzed: {moves_analyzed}")
    logger.info(f"Time: {elapsed:.1f}s ({games_processed/elapsed:.1f} games/s)")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
