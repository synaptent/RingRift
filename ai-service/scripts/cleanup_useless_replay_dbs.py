"""Cleanup script: delete GameReplayDB files with no structurally 'good' games.

We treat a game as "useful for debugging" when:
  - initial_state has empty move_history and an empty board (no stacks/markers/
    collapsedSpaces), and
  - get_state_at_move(game_id, 0) matches initial_state for currentPhase,
    currentPlayer, and gameStatus.

Any game whose initial_state already contains history or board content, or
whose reconstructed move-0 state disagrees with its initial_state, is treated
as a structural anomaly for parity/replay debugging.

This script scans the same locations as the parity checker:
  - data/games
  - ai-service/data/games
  - ai-service/logs/cmaes

For each DB:
  - If it contains zero games, or
  - If none of its games classify as "good",
    then the DB is considered useless for replay/parity debugging and is
    deleted when run with --delete. By default it only reports.

Usage (from ai-service/):

  PYTHONPATH=. python scripts/cleanup_useless_replay_dbs.py        # dry-run
  PYTHONPATH=. python scripts/cleanup_useless_replay_dbs.py --delete
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

from app.db.game_replay import GameReplayDB


def repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def find_dbs() -> List[Path]:
  root = repo_root()
  search_paths = [
      root / "data" / "games",
      root / "ai-service" / "logs" / "cmaes",
      root / "ai-service" / "data" / "games",
  ]

  results: List[Path] = []
  visited = set()

  def walk(dir_path: Path, depth: int) -> None:
    if depth <= 0:
      return
    real = dir_path.resolve()
    if real in visited or not real.exists():
      return
    visited.add(real)
    try:
      entries = list(real.iterdir())
    except OSError:
      return
    for entry in entries:
      if entry.is_dir():
        walk(entry, depth - 1)
      elif entry.is_file() and (entry.name == "games.db" or entry.name.endswith(".db")):
        results.append(entry)

  for base in search_paths:
    walk(base, 7)

  return results


def classify_game_structure(db: GameReplayDB, game_id: str) -> Tuple[str, str]:
  """Classify a game's recording structure; mirrors parity checker logic."""
  initial = db.get_initial_state(game_id)
  if initial is None:
    return "invalid", "no initial_state record"

  move_hist_len = len(initial.move_history or [])
  board = initial.board
  stacks = getattr(board, "stacks", {}) or {}
  markers = getattr(board, "markers", {}) or {}
  collapsed = getattr(board, "collapsed_spaces", {}) or {}

  stack_count = len(stacks)
  marker_count = len(markers)
  collapsed_count = len(collapsed)

  if move_hist_len > 0 or stack_count > 0 or marker_count > 0 or collapsed_count > 0:
    reason = (
        "initial_state contains history/board: "
        f"move_history={move_hist_len}, stacks={stack_count}, "
        f"markers={marker_count}, collapsed={collapsed_count}"
    )
    return "mid_snapshot", reason

  state0 = db.get_state_at_move(game_id, 0)
  if state0 is None:
    return "invalid", "get_state_at_move(0) returned None"

  if (
      initial.current_player != state0.current_player
      or initial.current_phase != state0.current_phase
      or initial.game_status != state0.game_status
  ):
    reason = (
        "initial_state differs from state_at_move_0: "
        f"initial_phase={initial.current_phase}, state0_phase={state0.current_phase}, "
        f"initial_player={initial.current_player}, state0_player={state0.current_player}, "
        f"initial_status={initial.game_status}, state0_status={state0.game_status}"
    )
    return "internal_inconsistent", reason

  return "good", ""


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Delete GameReplayDBs that contain no structurally 'good' games."
  )
  parser.add_argument(
      "--delete",
      action="store_true",
      help="Actually delete useless DBs. Without this flag, runs in dry-run mode.",
  )
  parser.add_argument(
      "--limit-games-per-db",
      type=int,
      default=0,
      help="Optional cap on games inspected per DB (0 = all).",
  )
  args = parser.parse_args()

  db_paths = find_dbs()
  if not db_paths:
    print("No GameReplayDB databases found.")
    return

  useless_dbs: List[Path] = []

  for db_path in db_paths:
    try:
      db = GameReplayDB(str(db_path))
    except Exception as exc:  # pragma: no cover - defensive
      print(f"[cleanup-useless-replay-dbs] Skipping {db_path}: failed to open ({exc})")
      continue

    games = db.query_games(limit=1000000)
    if not games:
      print(f"[cleanup-useless-replay-dbs] {db_path} has 0 games -> marked useless")
      useless_dbs.append(db_path)
      continue

    limit = args.limit_games_per_db or len(games)
    games = games[:limit]

    has_good = False
    for meta in games:
      game_id = meta["game_id"]
      try:
        structure, _ = classify_game_structure(db, game_id)
      except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[cleanup-useless-replay-dbs] {db_path} game {game_id}: classification error ({exc})"
        )
        continue
      if structure == "good":
        has_good = True
        break

    if not has_good:
      print(
          f"[cleanup-useless-replay-dbs] {db_path} has no structurally good games "
          f"in first {limit} inspected -> marked useless"
      )
      useless_dbs.append(db_path)

  if not useless_dbs:
    print("[cleanup-useless-replay-dbs] No useless DBs found.")
    return

  if not args.delete:
    print("[cleanup-useless-replay-dbs] Dry run; the following DBs would be deleted:")
    for p in useless_dbs:
      print(f"  - {p}")
    return

  for p in useless_dbs:
    try:
      os.remove(p)
      print(f"[cleanup-useless-replay-dbs] Deleted {p}")
    except OSError as exc:
      print(f"[cleanup-useless-replay-dbs] Failed to delete {p}: {exc}")
      continue

    for suffix in ("-wal", "-shm"):
      sidecar = Path(str(p) + suffix)
      if sidecar.exists():
        try:
          os.remove(sidecar)
          print(f"[cleanup-useless-replay-dbs] Deleted sidecar {sidecar}")
        except OSError as exc:
          print(f"[cleanup-useless-replay-dbs] Failed to delete sidecar {sidecar}: {exc}")


if __name__ == "__main__":
  main()

