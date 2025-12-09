#!/usr/bin/env python
from __future__ import annotations

"""
Build NPZ training datasets from canonical_* GameReplayDBs.

This is a thin, policy-aware wrapper around scripts/export_replay_dataset.py
that:
  - Restricts --db inputs to canonical_<board>.db by default.
  - Encourages a clear dataset naming convention under data/training/.

It does not change the underlying NPZ layout, which remains compatible with
app.training.generate_data and existing training loops.

Usage examples (from ai-service/):

  # Square-8 canonical dataset
  PYTHONPATH=. python scripts/build_canonical_dataset.py \
    --board-type square8 \
    --db data/games/canonical_square8.db \
    --output data/training/canonical_square8.npz

  # Square-19 and Hex in one go (using defaults for db/output names)
  PYTHONPATH=. python scripts/build_canonical_dataset.py --board-type square19
  PYTHONPATH=. python scripts/build_canonical_dataset.py --board-type hexagonal
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from export_replay_dataset import main as export_main  # type: ignore[import]


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _default_db_for_board(board_type: str) -> Path:
    return (AI_SERVICE_ROOT / "data" / "games" / f"canonical_{board_type}.db").resolve()


def _default_output_for_board(board_type: str) -> Path:
    return (AI_SERVICE_ROOT / "data" / "training" / f"canonical_{board_type}.npz").resolve()


def run_export(board_type: str, db_path: Path, output_path: Path) -> int:
    """Invoke export_replay_dataset.main(...) with canonical-safe arguments."""
    # Basic guard: insist on canonical_*.db basenames by default.
    if not db_path.name.startswith("canonical_"):
        raise SystemExit(
            f"[build-canonical-dataset] Refusing to export from non-canonical DB: {db_path}\n"
            "Expected a basename starting with 'canonical_'. Update TRAINING_DATA_REGISTRY.md "
            "and this script together if you intend to promote a new canonical DB."
        )

    os.makedirs(str(output_path.parent), exist_ok=True)

    argv = [
        "--db",
        str(db_path),
        "--board-type",
        board_type,
        "--num-players",
        "2",
        "--require-completed",
        "--min-moves",
        "10",
        "--output",
        str(output_path),
    ]

    # Delegate to the existing export_replay_dataset CLI main.
    return export_main(argv)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build NPZ training datasets from canonical_* GameReplayDBs.")
    parser.add_argument(
        "--board-type",
        required=True,
        choices=["square8", "square19", "hexagonal"],
        help="Board type whose canonical DB should be exported.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help=("Path to canonical GameReplayDB. Defaults to " "data/games/canonical_<board>.db."),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=("Output NPZ path. Defaults to " "data/training/canonical_<board>.npz."),
    )
    args = parser.parse_args(argv)

    board_type: str = args.board_type
    db_path = Path(args.db).resolve() if args.db else _default_db_for_board(board_type)
    output_path = Path(args.output).resolve() if args.output else _default_output_for_board(board_type)

    return run_export(board_type, db_path, output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
