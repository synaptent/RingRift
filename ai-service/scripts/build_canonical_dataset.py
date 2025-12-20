#!/usr/bin/env python
from __future__ import annotations

"""
Build NPZ training datasets from canonical_* GameReplayDBs.

This is a thin, policy-aware wrapper around scripts/export_replay_dataset.py
that:
  - Restricts --db inputs to canonical_<board>.db by default.
  - Encourages a clear dataset naming convention under data/training/.
  - Defaults to **board-aware policy encoding** for square boards (required for
    v3 training); pass --legacy-maxn-encoding to force the older MAX_N layout.
  - Archives any existing output NPZ before rebuilding (opt out with --append).
  - Supports all board types (square8, square19, hexagonal) and player counts (2, 3, 4).

It does not change the underlying NPZ layout, which remains compatible with
app.training.generate_data and existing training loops.

Usage examples (from ai-service/):

  # Square-8 2-player canonical dataset (default)
  PYTHONPATH=. python scripts/build_canonical_dataset.py \
    --board-type square8 \
    --db data/games/canonical_square8.db \
    --output data/training/canonical_square8_2p.npz

  # Square-8 4-player dataset
  PYTHONPATH=. python scripts/build_canonical_dataset.py \
    --board-type square8 --num-players 4

  # Square-19 and Hex (using defaults for db/output names)
  PYTHONPATH=. python scripts/build_canonical_dataset.py --board-type square19
  PYTHONPATH=. python scripts/build_canonical_dataset.py --board-type hexagonal --num-players 3
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

from export_replay_dataset import main as export_main  # type: ignore[import]
from validate_canonical_training_sources import (  # type: ignore[import]
    validate_canonical_sources,
)


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_ai_service_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (AI_SERVICE_ROOT / path).resolve()


def _default_db_for_board(board_type: str) -> Path:
    return (AI_SERVICE_ROOT / "data" / "games" / f"canonical_{board_type}.db").resolve()


def _default_output_for_board(board_type: str, num_players: int) -> Path:
    return (AI_SERVICE_ROOT / "data" / "training" / f"canonical_{board_type}_{num_players}p.npz").resolve()


def _validate_canonical_source_db(
    db_path: Path,
    registry_path: Path,
    *,
    allow_pending_gate: bool,
) -> None:
    allowed_statuses = ["canonical", "pending_gate"] if allow_pending_gate else ["canonical"]
    result = validate_canonical_sources(
        registry_path=registry_path,
        db_paths=[db_path],
        allowed_statuses=allowed_statuses,
    )
    if result.get("ok"):
        return

    for issue in result.get("problems", []):
        print(f"[canonical-source-error] {issue}", file=sys.stderr)

    hint = (
        "Fix TRAINING_DATA_REGISTRY.md status and/or regenerate the DB via "
        "scripts/generate_canonical_selfplay.py before exporting."
    )
    if allow_pending_gate:
        hint = (
            hint
            + " (You used --allow-pending-gate, so status may be pending_gate, but gate summary must still be canonical_ok.)"
        )
    raise SystemExit(f"[build-canonical-dataset] Refusing to export from non-canonical source DB.\n{hint}")


def run_export(
    board_type: str,
    num_players: int,
    db_path: Path,
    output_path: Path,
    *,
    registry_path: Path,
    allow_pending_gate: bool,
    legacy_maxn_encoding: bool,
    append: bool,
) -> int:
    """Invoke export_replay_dataset.main(...) with canonical-safe arguments."""
    # Basic guard: insist on canonical_*.db basenames by default.
    if not db_path.name.startswith("canonical_"):
        raise SystemExit(
            f"[build-canonical-dataset] Refusing to export from non-canonical DB: {db_path}\n"
            "Expected a basename starting with 'canonical_'. Update TRAINING_DATA_REGISTRY.md "
            "and this script together if you intend to promote a new canonical DB."
        )

    _validate_canonical_source_db(
        db_path=db_path,
        registry_path=registry_path,
        allow_pending_gate=allow_pending_gate,
    )

    os.makedirs(str(output_path.parent), exist_ok=True)

    if output_path.exists() and not append:
        archived = Path(
            f"{output_path.as_posix()}.archived_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        output_path.rename(archived)
        print(f"[build-canonical-dataset] archived existing output -> {archived}", file=sys.stderr)

    argv = [
        "--db",
        str(db_path),
        "--board-type",
        board_type,
        "--num-players",
        str(num_players),
        "--require-completed",
        "--min-moves",
        "10",
        "--output",
        str(output_path),
    ]
    if append:
        argv.append("--append")
    if board_type in {"square8", "square19", "hexagonal"} and not legacy_maxn_encoding:
        argv.append("--board-aware-encoding")

    # Delegate to the existing export_replay_dataset CLI main.
    return export_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build NPZ training datasets from canonical_* GameReplayDBs.")
    parser.add_argument(
        "--board-type",
        required=True,
        choices=["square8", "square19", "hexagonal"],
        help="Board type whose canonical DB should be exported.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (2, 3, or 4). Default: 2.",
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
        help=("Output NPZ path. Defaults to " "data/training/canonical_<board>_<num_players>p.npz."),
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="TRAINING_DATA_REGISTRY.md",
        help="Path to TRAINING_DATA_REGISTRY.md (default: TRAINING_DATA_REGISTRY.md).",
    )
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help=(
            "Allow registry Status=pending_gate as long as the referenced gate summary JSON "
            "reports canonical_ok=true and a passing parity gate."
        ),
    )
    parser.add_argument(
        "--legacy-maxn-encoding",
        action="store_true",
        help=(
            "Export using the legacy MAX_N policy encoding (larger action space). "
            "Default is board-aware encoding for square boards, which is required "
            "for v3 training."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append to an existing output NPZ if present (legacy export_replay_dataset behavior). "
            "Default is to archive any existing output and rebuild from scratch."
        ),
    )
    args = parser.parse_args(argv)

    board_type: str = args.board_type
    num_players: int = args.num_players
    db_path = _resolve_ai_service_path(args.db) if args.db else _default_db_for_board(board_type)
    output_path = _resolve_ai_service_path(args.output) if args.output else _default_output_for_board(board_type, num_players)
    registry_path = _resolve_ai_service_path(args.registry)

    return run_export(
        board_type,
        num_players,
        db_path,
        output_path,
        registry_path=registry_path,
        allow_pending_gate=bool(args.allow_pending_gate),
        legacy_maxn_encoding=bool(args.legacy_maxn_encoding),
        append=bool(args.append),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
