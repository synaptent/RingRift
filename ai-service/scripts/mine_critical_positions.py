#!/usr/bin/env python
"""Mine critical positions from self-play games.

Critical positions are game states where:
1. A player is close to winning (N rings from victory threshold)
2. The game is about to end (last N moves)
3. A decisive swing occurred (large evaluation change)

These positions are high-signal training examples because the outcome
is largely determined by the moves played from these states.

Usage:
    python scripts/mine_critical_positions.py \
        --num-games 100 \
        --board square8 \
        --output data/critical_positions/square8_critical.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections.abc import Iterator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import (
    AIConfig,
    BoardType,
    GameState,
    GameStatus,
    Move,
)
from app.ai.heuristic_ai import HeuristicAI
from app.rules.default_engine import DefaultRulesEngine
from app.db import GameReplayDB
from app.training.initial_state import create_initial_state
from scripts.lib.cli import BOARD_TYPE_MAP


@dataclass
class CriticalPosition:
    """A critical position extracted from a game."""

    # Position identification
    game_id: str
    move_number: int
    board_type: str

    # The game state (serialized)
    state: dict[str, Any]

    # The move that was played from this position
    move_played: dict[str, Any]
    player_to_move: int

    # Outcome information
    game_winner: int
    did_moving_player_win: bool
    moves_until_game_end: int

    # Why this position is critical
    criticality_reason: str
    criticality_score: float  # Higher = more critical

    # Victory proximity at this position
    player1_eliminated_rings: int
    player2_eliminated_rings: int
    victory_threshold: int
    rings_to_victory_p1: int
    rings_to_victory_p2: int


@dataclass
class GameTrajectory:
    """Full trajectory of a game with outcome."""

    game_id: str
    board_type: str
    winner: int
    termination_reason: str
    total_moves: int
    states: list[dict[str, Any]] = field(default_factory=list)
    moves: list[dict[str, Any]] = field(default_factory=list)


def state_to_dict(state: GameState) -> dict[str, Any]:
    """Convert GameState to serializable dict."""
    return state.model_dump(mode="json")


def move_to_dict(move: Move) -> dict[str, Any]:
    """Convert Move to serializable dict."""
    return move.model_dump(mode="json")


def play_game_with_trajectory(
    board_type: str,
    seed: int,
    max_moves: int = 10000,
) -> GameTrajectory:
    """Play a full game and record the trajectory."""
    rules = DefaultRulesEngine()
    board_type_enum = BOARD_TYPE_MAP.get(board_type, BoardType.SQUARE8)
    state = create_initial_state(board_type_enum, num_players=2)

    config1 = AIConfig(difficulty=5, seed=seed)
    config2 = AIConfig(difficulty=5, seed=seed + 1000)
    ai1 = HeuristicAI(1, config1)
    ai2 = HeuristicAI(2, config2)

    trajectory = GameTrajectory(
        game_id=f"game-{board_type}-{seed}",
        board_type=board_type,
        winner=0,
        termination_reason="unknown",
        total_moves=0,
    )

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        # Record state before move
        trajectory.states.append(state_to_dict(state))

        curr_player = state.current_player
        ai = ai1 if curr_player == 1 else ai2

        move = ai.select_move(state)
        if move is None:
            trajectory.termination_reason = "no_valid_moves"
            break

        # Record move
        trajectory.moves.append(move_to_dict(move))

        state = rules.apply_move(state, move)
        move_count += 1

    trajectory.total_moves = move_count
    trajectory.winner = state.winner if state.winner is not None else 0

    if state.game_status == GameStatus.COMPLETED:
        # Determine termination reason from final state
        p1 = state.players[0]
        p2 = state.players[1]
        if p1.eliminated_rings >= state.victory_threshold:
            trajectory.termination_reason = "ring_elimination_p1"
        elif p2.eliminated_rings >= state.victory_threshold:
            trajectory.termination_reason = "ring_elimination_p2"
        elif p1.territory_spaces >= state.territory_victory_threshold:
            trajectory.termination_reason = "territory_p1"
        elif p2.territory_spaces >= state.territory_victory_threshold:
            trajectory.termination_reason = "territory_p2"
        else:
            trajectory.termination_reason = "other"
    elif move_count >= max_moves:
        trajectory.termination_reason = "max_moves"

    return trajectory


def load_trajectory_from_db(
    db: GameReplayDB,
    game_meta: dict,
    initial_state: GameState,
    moves: list[Move],
) -> GameTrajectory:
    """Reconstruct a GameTrajectory from database records by replaying moves.

    This allows mining critical positions from pre-recorded games instead
    of playing new games.
    """
    rules = DefaultRulesEngine()
    game_id = game_meta["game_id"]
    board_type = game_meta.get("board_type", "square8")

    trajectory = GameTrajectory(
        game_id=game_id,
        board_type=board_type,
        winner=game_meta.get("winner", 0) or 0,
        termination_reason=game_meta.get("termination_reason", "unknown") or "unknown",
        total_moves=len(moves),
    )

    # Replay moves to reconstruct state trajectory
    state = initial_state
    for move in moves:
        # Record state before move
        trajectory.states.append(state_to_dict(state))
        trajectory.moves.append(move_to_dict(move))

        # Apply move to get next state
        state = rules.apply_move(state, move)

    return trajectory


def iterate_db_trajectories(
    db_path: str,
    board_type: str | None = None,
    limit: int = 1000,
) -> Iterator[GameTrajectory]:
    """Iterate over games from a database, yielding trajectories.

    Args:
        db_path: Path to the GameReplayDB database
        board_type: Optional board type filter (e.g., 'square8')
        limit: Maximum number of games to load

    Yields:
        GameTrajectory objects reconstructed from the database
    """
    db = GameReplayDB(db_path)

    # Map string board type to enum if provided
    bt_enum = BOARD_TYPE_MAP.get(board_type) if board_type else None

    # Build filter kwargs - iterate_games doesn't support limit param,
    # so we track count ourselves
    filters: dict[str, Any] = {}
    if bt_enum is not None:
        filters["board_type"] = bt_enum

    count = 0
    for game_meta, initial_state, moves in db.iterate_games(**filters):
        if count >= limit:
            break
        yield load_trajectory_from_db(db, game_meta, initial_state, moves)
        count += 1


def extract_critical_positions(
    trajectory: GameTrajectory,
    rings_from_victory: int = 2,
    last_n_moves: int = 10,
) -> list[CriticalPosition]:
    """Extract critical positions from a game trajectory."""
    critical = []

    if trajectory.winner == 0:
        # Game didn't finish with a winner, skip
        return critical

    total_moves = len(trajectory.moves)

    for i, (state_dict, move_dict) in enumerate(zip(trajectory.states, trajectory.moves)):
        moves_until_end = total_moves - i
        player_to_move = state_dict.get("current_player", 1)

        # Get player stats
        players = state_dict.get("players", [])
        if len(players) < 2:
            continue

        p1_elim = players[0].get("eliminated_rings", 0)
        p2_elim = players[1].get("eliminated_rings", 0)
        victory_threshold = state_dict.get("victory_threshold", 5)

        rings_to_win_p1 = victory_threshold - p1_elim
        rings_to_win_p2 = victory_threshold - p2_elim

        criticality_reasons = []
        criticality_score = 0.0

        # Criterion 1: Close to victory threshold
        if rings_to_win_p1 <= rings_from_victory:
            criticality_reasons.append(f"p1_near_victory({rings_to_win_p1}_away)")
            criticality_score += (rings_from_victory - rings_to_win_p1 + 1) * 2
        if rings_to_win_p2 <= rings_from_victory:
            criticality_reasons.append(f"p2_near_victory({rings_to_win_p2}_away)")
            criticality_score += (rings_from_victory - rings_to_win_p2 + 1) * 2

        # Criterion 2: Near end of game
        if moves_until_end <= last_n_moves:
            criticality_reasons.append(f"near_end({moves_until_end}_moves_left)")
            criticality_score += (last_n_moves - moves_until_end + 1) * 0.5

        # Criterion 3: Moving player is about to win
        if player_to_move == 1 and rings_to_win_p1 == 1:
            criticality_reasons.append("p1_one_from_win")
            criticality_score += 5
        elif player_to_move == 2 and rings_to_win_p2 == 1:
            criticality_reasons.append("p2_one_from_win")
            criticality_score += 5

        # Only include if critical
        if criticality_score > 0:
            critical.append(
                CriticalPosition(
                    game_id=trajectory.game_id,
                    move_number=i,
                    board_type=trajectory.board_type,
                    state=state_dict,
                    move_played=move_dict,
                    player_to_move=player_to_move,
                    game_winner=trajectory.winner,
                    did_moving_player_win=(player_to_move == trajectory.winner),
                    moves_until_game_end=moves_until_end,
                    criticality_reason="|".join(criticality_reasons),
                    criticality_score=criticality_score,
                    player1_eliminated_rings=p1_elim,
                    player2_eliminated_rings=p2_elim,
                    victory_threshold=victory_threshold,
                    rings_to_victory_p1=rings_to_win_p1,
                    rings_to_victory_p2=rings_to_win_p2,
                )
            )

    return critical


def run_mining(
    board_type: str = "square8",
    num_games: int = 100,
    max_moves: int = 200,
    seed: int = 42,
    rings_from_victory: int = 2,
    last_n_moves: int = 10,
    verbose: bool = True,
) -> list[CriticalPosition]:
    """Run critical position mining over multiple self-play games."""
    all_critical = []

    print(f"\nMining critical positions from {num_games} {board_type} games...")
    print(f"Criteria: within {rings_from_victory} rings of victory OR last {last_n_moves} moves\n")

    start_time = time.time()
    games_with_winner = 0

    for i in range(num_games):
        game_seed = seed + i
        game_start = time.time()

        trajectory = play_game_with_trajectory(board_type, game_seed, max_moves)

        if trajectory.winner != 0:
            games_with_winner += 1
            critical = extract_critical_positions(
                trajectory,
                rings_from_victory=rings_from_victory,
                last_n_moves=last_n_moves,
            )
            all_critical.extend(critical)

        game_elapsed = time.time() - game_start

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (num_games - i - 1)
            print(
                f"  [{i+1:3d}/{num_games}] {trajectory.total_moves:3d} moves, "
                f"winner={trajectory.winner}, critical={len(critical) if trajectory.winner else 0}, "
                f"total={len(all_critical)} | ETA: {eta/60:.1f}m"
            )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Games with winner: {games_with_winner}/{num_games}")
    print(f"Critical positions found: {len(all_critical)}")

    return all_critical


def run_mining_from_db(
    db_path: str,
    board_type: str | None = None,
    limit: int = 1000,
    rings_from_victory: int = 2,
    last_n_moves: int = 10,
    verbose: bool = True,
) -> list[CriticalPosition]:
    """Run critical position mining over games loaded from a database.

    Args:
        db_path: Path to the GameReplayDB SQLite database
        board_type: Optional board type filter (e.g., 'square8')
        limit: Maximum number of games to load from DB
        rings_from_victory: Include positions within N rings of victory
        last_n_moves: Include last N moves of each game
        verbose: Print progress updates

    Returns:
        List of CriticalPosition objects extracted from the games
    """
    all_critical = []

    board_filter = board_type if board_type else "all"
    print(f"\nMining critical positions from database: {db_path}")
    print(f"Board filter: {board_filter}, Limit: {limit} games")
    print(f"Criteria: within {rings_from_victory} rings of victory OR last {last_n_moves} moves\n")

    start_time = time.time()
    games_processed = 0
    games_with_winner = 0

    for trajectory in iterate_db_trajectories(db_path, board_type, limit):
        games_processed += 1

        if trajectory.winner != 0:
            games_with_winner += 1
            critical = extract_critical_positions(
                trajectory,
                rings_from_victory=rings_from_victory,
                last_n_moves=last_n_moves,
            )
            all_critical.extend(critical)

            if verbose and games_processed % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  [{games_processed:4d}] {trajectory.game_id}: "
                    f"{trajectory.total_moves:3d} moves, winner={trajectory.winner}, "
                    f"critical={len(critical)}, total={len(all_critical)} | "
                    f"elapsed={elapsed:.1f}s"
                )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Games processed: {games_processed}")
    print(f"Games with winner: {games_with_winner}/{games_processed}")
    print(f"Critical positions found: {len(all_critical)}")

    return all_critical


def main():
    parser = argparse.ArgumentParser(
        description="Mine critical positions from self-play games or a database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mine from new self-play games
  python scripts/mine_critical_positions.py --num-games 100 --board square8

  # Mine from an existing database
  python scripts/mine_critical_positions.py --from-db data/games/selfplay.db --db-limit 500

  # Mine from database with board filter
  python scripts/mine_critical_positions.py --from-db data/games/selfplay.db --board hex
""",
    )
    parser.add_argument(
        "--board",
        default="square8",
        choices=["square8", "square19", "hex"],
        help="Board type (used for both self-play and DB filtering)",
    )
    parser.add_argument(
        "--num-games", type=int, default=100, help="Number of self-play games to run (ignored with --from-db)"
    )
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum moves per game (ignored with --from-db)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for self-play (ignored with --from-db)")
    parser.add_argument(
        "--rings-from-victory", type=int, default=2, help="Include positions within N rings of victory threshold"
    )
    parser.add_argument("--last-n-moves", type=int, default=10, help="Include last N moves of each game")
    parser.add_argument(
        "--output", type=str, default="data/critical_positions/positions.jsonl", help="Output JSONL file path"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--from-db", type=str, default=None, help="Load games from a GameReplayDB SQLite database instead of self-play"
    )
    parser.add_argument(
        "--db-limit", type=int, default=1000, help="Maximum number of games to load from database (with --from-db)"
    )
    args = parser.parse_args()

    if args.from_db:
        # Mine from database
        critical_positions = run_mining_from_db(
            db_path=args.from_db,
            board_type=args.board if args.board else None,
            limit=args.db_limit,
            rings_from_victory=args.rings_from_victory,
            last_n_moves=args.last_n_moves,
            verbose=not args.quiet,
        )
    else:
        # Mine from self-play
        critical_positions = run_mining(
            board_type=args.board,
            num_games=args.num_games,
            max_moves=args.max_moves,
            seed=args.seed,
            rings_from_victory=args.rings_from_victory,
            last_n_moves=args.last_n_moves,
            verbose=not args.quiet,
        )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for pos in critical_positions:
            f.write(json.dumps(asdict(pos)) + "\n")

    print(f"\nSaved {len(critical_positions)} critical positions to {args.output}")

    # Print summary statistics
    if critical_positions:
        win_moves = [p for p in critical_positions if p.did_moving_player_win]
        lose_moves = [p for p in critical_positions if not p.did_moving_player_win]

        print(f"\nSummary:")
        print(
            f"  Positions where moving player won:  {len(win_moves)} ({100*len(win_moves)/len(critical_positions):.1f}%)"
        )
        print(
            f"  Positions where moving player lost: {len(lose_moves)} ({100*len(lose_moves)/len(critical_positions):.1f}%)"
        )

        # Criticality distribution
        by_reason = {}
        for p in critical_positions:
            for reason in p.criticality_reason.split("|"):
                by_reason[reason] = by_reason.get(reason, 0) + 1

        print(f"\n  By criticality reason:")
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1])[:10]:
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
