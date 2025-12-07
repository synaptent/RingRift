#!/usr/bin/env python3
"""Generate parity test vector fixtures from game replay databases.

This script extracts game state snapshots at sampled positions from GameReplayDB
databases, comparing Python and TypeScript engine states to create test vectors
for regression testing.

Unlike the parity checker which only captures divergence points, this tool
generates vectors at arbitrary positions to expand test coverage.

Usage:
    cd ai-service
    python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db
    python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db --strategy uniform --interval 10
    python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db --strategy random --sample-rate 0.1
    python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db --strategy key_positions

Options:
    --db PATH           Path to game replay database (required)
    --output DIR        Output directory for fixtures (default: parity_fixtures/generated/)
    --strategy STR      Sampling strategy: uniform, random, key_positions (default: uniform)
    --interval N        For uniform strategy, sample every N moves (default: 10)
    --sample-rate F     For random strategy, fraction of moves to sample (default: 0.1)
    --min-moves N       Minimum game length to include (default: 10)
    --max-vectors N     Maximum total vectors to generate (default: 50)
    --limit-games N     Maximum games to process (default: all)
    --include-matching  Include vectors where Python and TS match (default: only divergent)
    --dry-run           Show what would be generated without writing files
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add ai-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.game_replay import GameReplayDB, _compute_state_hash
from app.models import GamePhase, MoveType


@dataclass
class StateSummary:
    """Summary of game state at a specific move."""
    move_index: int
    current_player: int
    current_phase: str
    game_status: str
    state_hash: str


@dataclass
class ParityVector:
    """A parity test vector capturing Python and TS states."""
    db_path: str
    game_id: str
    move_number: int
    python_summary: StateSummary
    ts_summary: Optional[StateSummary]
    canonical_move: Optional[dict]
    is_match: bool
    mismatch_kinds: List[str]


def repo_root() -> Path:
    """Return the monorepo root (parent of ai-service/)."""
    return Path(__file__).resolve().parents[2]


def _canonicalize_status(status: str | None) -> str:
    """Normalize status strings for comparison."""
    if status is None:
        return "active"
    s = str(status)
    if s == "finished":
        return "completed"
    return s


def summarize_python_state(db: GameReplayDB, game_id: str, move_index: int) -> StateSummary:
    """Get Python state summary after move_index is applied."""
    state = db.get_state_at_move(game_id, move_index)
    if state is None:
        raise RuntimeError(f"Python get_state_at_move returned None for {game_id} @ {move_index}")
    return StateSummary(
        move_index=move_index,
        current_player=state.current_player,
        current_phase=state.current_phase.value
        if hasattr(state.current_phase, "value")
        else str(state.current_phase),
        game_status=_canonicalize_status(
            state.game_status.value
            if hasattr(state.game_status, "value")
            else str(state.game_status)
        ),
        state_hash=_compute_state_hash(state),
    )


def summarize_python_initial_state(db: GameReplayDB, game_id: str) -> StateSummary:
    """Get Python initial state summary (before any moves)."""
    from app.training.generate_data import create_initial_state
    from app.game_engine import BoardType

    state = db.get_initial_state(game_id)
    if state is None:
        # No initial_state record - create from game metadata
        metadata = db.get_game_metadata(game_id)
        if metadata is None:
            raise RuntimeError(f"No initial state or metadata for {game_id}")
        
        board_type_str = metadata.get("board_type", "square8")
        num_players = metadata.get("num_players", 2)
        
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(board_type_str.lower(), BoardType.SQUARE8)
        state = create_initial_state(board_type=board_type, num_players=num_players)
    
    return StateSummary(
        move_index=0,
        current_player=state.current_player,
        current_phase=state.current_phase.value
        if hasattr(state.current_phase, "value")
        else str(state.current_phase),
        game_status=_canonicalize_status(
            state.game_status.value
            if hasattr(state.game_status, "value")
            else str(state.game_status)
        ),
        state_hash=_compute_state_hash(state),
    )


def run_ts_replay(db_path: Path, game_id: str) -> Tuple[int, Dict[int, StateSummary]]:
    """Invoke the TS replay harness to get state summaries at each move.
    
    Returns:
        (total_moves_ts, mapping from k -> StateSummary)
        where k=0 is initial state, k=1 is after move 0, etc.
    """
    root = repo_root()
    cmd = [
        "npx",
        "ts-node",
        "-T",
        "scripts/selfplay-db-ts-replay.ts",
        "--db",
        str(db_path),
        "--game",
        game_id,
    ]
    
    env = os.environ.copy()
    env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    
    if proc.returncode != 0:
        raise RuntimeError(
            f"TS replay harness failed for {db_path} / {game_id} with code {proc.returncode}:\n"
            f"STDERR:\n{stderr}"
        )
    
    total_ts_moves = 0
    summaries: Dict[int, StateSummary] = {}
    
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        kind = payload.get("kind")
        if kind == "ts-replay-initial":
            total_ts_moves = int(payload.get("totalRecordedMoves", 0))
            summary = payload.get("summary") or {}
            summaries[0] = StateSummary(
                move_index=0,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=_canonicalize_status(summary.get("gameStatus")),
                state_hash=summary.get("stateHash"),
            )
        elif kind == "ts-replay-step":
            k = int(payload.get("k", 0))
            summary = payload.get("summary") or {}
            summaries[k] = StateSummary(
                move_index=k,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=_canonicalize_status(summary.get("gameStatus")),
                state_hash=summary.get("stateHash"),
            )
    
    return total_ts_moves, summaries


def sample_positions_uniform(total_moves: int, interval: int) -> List[int]:
    """Sample positions at regular intervals."""
    positions = []
    for k in range(0, total_moves + 1, interval):
        positions.append(k)
    return positions


def sample_positions_random(total_moves: int, sample_rate: float, seed: Optional[int] = None) -> List[int]:
    """Sample positions randomly at given rate."""
    if seed is not None:
        random.seed(seed)
    
    positions = []
    for k in range(total_moves + 1):
        if random.random() < sample_rate:
            positions.append(k)
    
    # Always include first and last
    if 0 not in positions:
        positions.insert(0, 0)
    if total_moves not in positions and total_moves > 0:
        positions.append(total_moves)
    
    return sorted(set(positions))


def sample_positions_key(
    db: GameReplayDB,
    game_id: str,
    total_moves: int,
) -> List[int]:
    """Sample positions at key game events (phase transitions, captures, etc.)."""
    positions: Set[int] = {0}  # Always include initial state
    
    moves = db.get_moves(game_id)
    
    # Track phase transitions
    prev_phase = None
    for i, move in enumerate(moves):
        move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
        
        # Include positions around captures
        if move_type in ("capture", "skip_capture", "chain_capture"):
            positions.add(i)
            if i > 0:
                positions.add(i - 1)
            if i < len(moves) - 1:
                positions.add(i + 1)
        
        # Include positions around territory actions
        if move_type in ("claim_territory", "place_territory_marker", "territory_claim"):
            positions.add(i)
        
        # Include positions around line formations
        if move_type in ("no_line_action", "select_ring_removal"):
            positions.add(i)
        
        # Include forced elimination
        if move_type == "forced_elimination":
            positions.add(i)
            if i > 0:
                positions.add(i - 1)
    
    # Always include last position
    if total_moves > 0:
        positions.add(total_moves - 1)
    
    return sorted(positions)


def compare_summaries(py: StateSummary, ts: StateSummary) -> Tuple[bool, List[str]]:
    """Compare Python and TS summaries, return (is_match, mismatch_kinds)."""
    mismatches = []
    
    if py.current_player != ts.current_player:
        mismatches.append("current_player")
    if py.current_phase != ts.current_phase:
        mismatches.append("current_phase")
    if py.game_status != ts.game_status:
        mismatches.append("game_status")
    if py.state_hash != ts.state_hash:
        mismatches.append("state_hash")
    
    return len(mismatches) == 0, mismatches


def generate_vectors(
    db_path: str,
    output_dir: str,
    strategy: str = "uniform",
    interval: int = 10,
    sample_rate: float = 0.1,
    min_moves: int = 10,
    max_vectors: int = 50,
    limit_games: Optional[int] = None,
    include_matching: bool = False,
    dry_run: bool = False,
) -> None:
    """Generate parity test vectors from a game database.
    
    Args:
        db_path: Path to game replay database
        output_dir: Directory to write fixture JSONs
        strategy: Sampling strategy (uniform, random, key_positions)
        interval: For uniform strategy, sample every N moves
        sample_rate: For random strategy, fraction of moves to sample
        min_moves: Minimum game length to include
        max_vectors: Maximum vectors to generate
        limit_games: Maximum games to process (None = all)
        include_matching: Include vectors where Python and TS match
        dry_run: Show what would be generated without writing files
    """
    db_path_obj = Path(db_path).resolve()
    if not db_path_obj.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)
    
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating parity vectors from {db_path_obj.name}")
    print(f"  Strategy: {strategy}")
    print(f"  Min moves: {min_moves}")
    print(f"  Max vectors: {max_vectors}")
    print(f"  Output: {output_dir}")
    print()
    
    db = GameReplayDB(str(db_path_obj))
    
    # Query games meeting criteria
    games = db.query_games(min_moves=min_moves, limit=limit_games or 100000)
    print(f"Games found: {len(games)}")
    
    if not games:
        print("No games meet criteria.")
        return
    
    vectors_generated = 0
    games_processed = 0
    
    for game_meta in games:
        if vectors_generated >= max_vectors:
            break
        
        game_id = game_meta["game_id"]
        total_moves = game_meta.get("total_moves", 0)
        
        if total_moves < min_moves:
            continue
        
        games_processed += 1
        
        # Get TS replay data
        try:
            total_ts_moves, ts_summaries = run_ts_replay(db_path_obj, game_id)
        except Exception as e:
            print(f"  Skipping {game_id}: TS replay failed: {e}")
            continue
        
        # Sample positions based on strategy
        if strategy == "uniform":
            positions = sample_positions_uniform(total_moves, interval)
        elif strategy == "random":
            positions = sample_positions_random(total_moves, sample_rate)
        elif strategy == "key_positions":
            positions = sample_positions_key(db, game_id, total_moves)
        else:
            print(f"Error: Unknown strategy '{strategy}'")
            sys.exit(1)
        
        # Get moves for canonical move lookup
        moves = db.get_moves(game_id)
        
        for k in positions:
            if vectors_generated >= max_vectors:
                break
            
            # Get Python summary
            try:
                if k == 0:
                    py_summary = summarize_python_initial_state(db, game_id)
                else:
                    py_move_index = k - 1  # TS k -> Python move index
                    py_summary = summarize_python_state(db, game_id, py_move_index)
            except Exception as e:
                print(f"    Skipping k={k}: Python state failed: {e}")
                continue
            
            # Get TS summary
            ts_summary = ts_summaries.get(k)
            if ts_summary is None:
                print(f"    Skipping k={k}: No TS summary")
                continue
            
            # Compare
            is_match, mismatches = compare_summaries(py_summary, ts_summary)
            
            # Skip matching vectors unless requested
            if is_match and not include_matching:
                continue
            
            # Get canonical move if available
            canonical_move = None
            if k > 0 and k - 1 < len(moves):
                try:
                    move_obj = moves[k - 1]
                    canonical_move = json.loads(move_obj.model_dump_json(by_alias=True))
                except Exception:
                    pass
            
            # Build fixture
            db_name = db_path_obj.stem
            fixture = {
                "db_path": str(db_path_obj),
                "game_id": game_id,
                "move_number": k,
                "python_summary": asdict(py_summary),
                "ts_summary": asdict(ts_summary),
                "canonical_move": canonical_move,
                "is_match": is_match,
                "mismatch_kinds": mismatches,
                "total_moves_python": total_moves,
                "total_moves_ts": total_ts_moves,
            }
            
            # Generate filename
            safe_game_id = game_id.replace("/", "_")
            fixture_name = f"{db_name}__{safe_game_id}__k{k}.json"
            fixture_path = output_path / fixture_name
            
            if dry_run:
                status = "MATCH" if is_match else f"MISMATCH({','.join(mismatches)})"
                print(f"  Would write: {fixture_name} [{status}]")
            else:
                with open(fixture_path, "w", encoding="utf-8") as f:
                    json.dump(fixture, f, indent=2, sort_keys=True)
                
                status = "MATCH" if is_match else f"MISMATCH({','.join(mismatches)})"
                print(f"  Wrote: {fixture_name} [{status}]")
            
            vectors_generated += 1
    
    print()
    print(f"Summary:")
    print(f"  Games processed: {games_processed}")
    print(f"  Vectors generated: {vectors_generated}")
    print(f"  Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate parity test vectors from game replay databases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate vectors with uniform sampling every 10 moves
  python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db

  # Generate vectors with random 10% sampling
  python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db --strategy random

  # Generate vectors at key game positions (captures, territory, etc.)
  python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db --strategy key_positions

  # Include matching vectors (not just divergent ones)
  python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db --include-matching

  # Dry run to see what would be generated
  python scripts/generate_parity_vectors.py --db data/games/canonical_square8.db --dry-run
""",
    )
    
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to game replay database",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="parity_fixtures/generated/",
        help="Output directory for fixtures (default: parity_fixtures/generated/)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["uniform", "random", "key_positions"],
        default="uniform",
        help="Sampling strategy (default: uniform)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="For uniform strategy, sample every N moves (default: 10)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.1,
        help="For random strategy, fraction of moves to sample (default: 0.1)",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=10,
        help="Minimum game length to include (default: 10)",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=50,
        help="Maximum vectors to generate (default: 50)",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Maximum games to process (default: all)",
    )
    parser.add_argument(
        "--include-matching",
        action="store_true",
        help="Include vectors where Python and TS match (default: only divergent)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )
    
    args = parser.parse_args()
    
    generate_vectors(
        db_path=args.db,
        output_dir=args.output,
        strategy=args.strategy,
        interval=args.interval,
        sample_rate=args.sample_rate,
        min_moves=args.min_moves,
        max_vectors=args.max_vectors,
        limit_games=args.limit_games,
        include_matching=args.include_matching,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()