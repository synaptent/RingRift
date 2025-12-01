from __future__ import annotations

"""Lightweight validation helpers for territory / combined-margin datasets.

These helpers perform **structural** and **metadata** validation for the
JSONL records produced by :mod:`generate_territory_dataset`. They are
intended for:

- small CI-style checks over sample files, and
- offline sanity checks when generating new datasets.

They deliberately avoid loading the entire dataset into memory; callers
can stream over files and stop after a bounded number of errors.
"""

import json
from math import isfinite
from typing import Any, Dict, Iterable, List, Tuple

from app.models import GameState


def validate_territory_example(example: Dict[str, Any]) -> List[str]:
    """Validate a single territory/combined-margin example.

    The expected top-level shape matches the output of
    :func:`generate_territory_dataset`:

    .. code-block:: json

        {
          "game_state": { ... },          // GameState JSON (camelCase)
          "player_number": 1,
          "target": 12.0,
          "time_weight": 0.9,
          "engine_mode": "mixed",
          "num_players": 2,
          "ai_type_p1": "descent",
          "ai_difficulty_p1": 9,
          "ai_type_p2": "heuristic",
          "ai_difficulty_p2": 2
        }
    """

    errors: List[str] = []

    # Required top-level fields
    required_fields = [
        "game_state",
        "player_number",
        "target",
        "time_weight",
        "engine_mode",
        "num_players",
    ]
    for field in required_fields:
        if field not in example:
            errors.append(f"missing field {field!r}")

    # If critical fields are missing, further checks are likely to be noisy.
    if errors:
        return errors

    engine_mode = example.get("engine_mode")
    if engine_mode not in {"descent-only", "mixed"}:
        errors.append(f"engine_mode must be 'descent-only' or 'mixed', got {engine_mode!r}")

    num_players = example.get("num_players")
    if not isinstance(num_players, int):
        errors.append(f"num_players must be an int, got {type(num_players).__name__}")
        # Without a valid num_players we cannot meaningfully validate AI metadata.
        return errors
    if num_players < 2 or num_players > 4:
        errors.append(f"num_players must be in [2, 4], got {num_players}")

    player_number = example.get("player_number")
    if not isinstance(player_number, int):
        errors.append(f"player_number must be an int, got {type(player_number).__name__}")
    elif not (1 <= player_number <= num_players):
        errors.append(
            f"player_number must be between 1 and num_players (got {player_number}, num_players={num_players})"
        )

    target = example.get("target")
    if not isinstance(target, (int, float)) or not isfinite(target):
        errors.append(f"target must be a finite number, got {target!r}")

    time_weight = example.get("time_weight")
    if not isinstance(time_weight, (int, float)) or not isfinite(time_weight):
        errors.append(f"time_weight must be a finite number, got {time_weight!r}")
    elif not (0.0 <= float(time_weight) <= 1.0):
        errors.append(f"time_weight must be in [0.0, 1.0], got {time_weight!r}")

    # AI metadata completeness: engine_mode / num_players are already checked.
    for pnum in range(1, num_players + 1):
        type_key = f"ai_type_p{pnum}"
        diff_key = f"ai_difficulty_p{pnum}"

        if type_key not in example:
            errors.append(f"missing AI type metadata field {type_key!r}")
        elif not isinstance(example[type_key], str):
            errors.append(f"{type_key} must be a string, got {type(example[type_key]).__name__}")

        if diff_key not in example:
            errors.append(f"missing AI difficulty metadata field {diff_key!r}")
        elif not isinstance(example[diff_key], int):
            errors.append(f"{diff_key} must be an int, got {type(example[diff_key]).__name__}")

    # Validate that game_state can be parsed into a GameState and is
    # consistent with num_players at a basic structural level.
    try:
        raw_state = example["game_state"]
        state = GameState.model_validate(raw_state)
    except Exception as exc:  # pragma: no cover - error message shape is library-specific
        errors.append(f"game_state failed validation: {exc}")
        return errors

    if len(state.players) != num_players:
        errors.append(
            f"num_players={num_players} but game_state.players has length {len(state.players)}"
        )

    # Basic sanity check for board type.
    if state.board_type not in {"square8", "square19", "hexagonal"}:
        errors.append(f"unexpected board_type {state.board_type!r} in game_state")

    return errors


def iter_territory_dataset_errors(
    jsonl_lines: Iterable[str], *, max_errors: int = 50
) -> List[Tuple[int, str]]:
    """Validate a stream of JSONL lines and collect up to ``max_errors``.

    Returns a list of ``(line_number, message)`` pairs for any errors
    encountered. This function is intended for unit tests and small
    CI checks; larger offline runs may want to stream and act on the
    first error encountered instead.
    """

    collected: List[Tuple[int, str]] = []

    for idx, line in enumerate(jsonl_lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            example = json.loads(line)
        except json.JSONDecodeError as exc:
            collected.append((idx, f"invalid JSON: {exc}"))
            if len(collected) >= max_errors:
                break
            continue

        errs = validate_territory_example(example)
        for msg in errs:
            collected.append((idx, msg))
            if len(collected) >= max_errors:
                break
        if len(collected) >= max_errors:
            break

    return collected


def validate_territory_dataset_file(path: str, *, max_errors: int = 50) -> List[Tuple[int, str]]:
    """Validate a territory dataset JSONL file on disk."""
    with open(path, "r", encoding="utf-8") as f:
        return iter_territory_dataset_errors(f, max_errors=max_errors)


def main(argv: List[str] | None = None) -> int:
    """CLI entrypoint for validating a territory dataset JSONL file.

    Usage (from ``ai-service`` root):

    .. code-block:: bash

        python -m app.training.territory_dataset_validation path/to/dataset.jsonl
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a territory / combined-margin JSONL dataset."
    )
    parser.add_argument("path", help="Path to the JSONL dataset file.")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum number of errors to report before stopping (default: 50).",
    )

    args = parser.parse_args(argv)
    errors = validate_territory_dataset_file(args.path, max_errors=args.max_errors)

    if not errors:
        print(f"{args.path}: OK")
        return 0

    print(f"{args.path}: found {len(errors)} error(s):")
    for line_no, msg in errors:
        print(f"  line {line_no}: {msg}")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

