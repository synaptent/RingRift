#!/usr/bin/env python3
"""Check canonical ladder tiers against local AI artifacts.

This is a lightweight, offline companion to the FastAPI endpoint:
  GET /internal/ladder/health

It is intended for:
  - Local dev sanity checks after training/promotion.
  - CI / deployment scripts that want a simple exit code without running the server.

By default it does not load models; it only checks configuration and file
presence. Use ``--load-checkpoints`` to also verify that CNN checkpoints are
actually loadable (catches truncated/corrupted .pth files).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES
from app.ai.nnue import get_nnue_model_path
from app.config.ladder_config import get_effective_ladder_config, list_ladder_tiers
from app.models import AIType, BoardType


@dataclass(frozen=True)
class _CheckpointLoadResult:
    ok: bool
    error: str | None = None


def _resolve_latest_checkpoint(models_dir: Path, model_id: str) -> Path | None:
    patterns = [
        f"{model_id}.pth",
        f"{model_id}_mps.pth",
        f"{model_id}_*.pth",
        f"{model_id}_*_mps.pth",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(models_dir.glob(pattern))

    unique = sorted(
        {p.resolve() for p in matches if p.is_file()},
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    for candidate in reversed(unique):
        try:
            if candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def _try_load_checkpoint(path: Path) -> _CheckpointLoadResult:
    """Best-effort checkpoint integrity check.

    Uses ``torch.load(..., weights_only=False)`` because RingRift checkpoints can
    include lightweight metadata alongside tensor weights.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        return _CheckpointLoadResult(ok=False, error=f"torch_import_failed: {exc}")

    try:
        _ = torch.load(str(path), map_location="cpu", weights_only=False)
        return _CheckpointLoadResult(ok=True, error=None)
    except Exception as exc:
        return _CheckpointLoadResult(ok=False, error=f"{type(exc).__name__}: {exc}")


def _parse_board(value: str | None) -> BoardType | None:
    if not value:
        return None
    key = value.strip().lower()
    mapping = {
        "square8": BoardType.SQUARE8,
        "sq8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "sq19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex": BoardType.HEXAGONAL,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported board_type={value!r}; expected one of {sorted(mapping)}")
    return mapping[key]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-type", default=None, help="Filter by board type (square8/square19/hexagonal).")
    parser.add_argument("--num-players", type=int, default=None, help="Filter by num players (2/3/4).")
    parser.add_argument("--difficulty", type=int, default=None, help="Filter by difficulty (1..10).")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit non-zero when any artifacts are missing.")
    parser.add_argument(
        "--load-checkpoints",
        action="store_true",
        help=(
            "Attempt to load CNN checkpoints with torch.load to catch corrupted/truncated files. "
            "Adds `corrupt_neural_checkpoints` to the summary and marks per-tier artifacts."
        ),
    )
    parser.add_argument(
        "--fail-on-corrupt",
        action="store_true",
        help="When used with --load-checkpoints, exit non-zero if any CNN checkpoint fails to load.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary instead of human text.")
    parser.add_argument("--verbose", action="store_true", help="Include per-tier details.")
    args = parser.parse_args(argv)

    board_type = _parse_board(args.board_type)
    num_players = args.num_players
    difficulty = args.difficulty

    if num_players is not None and num_players not in (2, 3, 4):
        raise ValueError("num_players must be 2, 3, or 4")
    if difficulty is not None and (difficulty < 1 or difficulty > 10):
        raise ValueError("difficulty must be between 1 and 10")

    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "ai-service" / "models"

    tiers = list_ladder_tiers(board_type=board_type, num_players=num_players)
    base_combos = sorted({(t.board_type, t.num_players) for t in tiers}, key=lambda x: (x[0].value, x[1]))

    rows: list[dict[str, Any]] = []
    missing_profiles = 0
    missing_nnue = 0
    missing_nn = 0
    corrupt_nn = 0

    def _difficulty_matches(level: int) -> bool:
        return difficulty is None or level == difficulty

    if _difficulty_matches(1):
        for bt, np in base_combos:
            if board_type is not None and bt != board_type:
                continue
            if num_players is not None and np != num_players:
                continue
            rows.append(
                {
                    "difficulty": 1,
                    "board_type": bt.value,
                    "num_players": np,
                    "ai_type": AIType.RANDOM.value,
                    "use_neural_net": False,
                    "model_id": None,
                    "heuristic_profile_id": None,
                    "artifacts": {},
                }
            )

    for base in tiers:
        if not _difficulty_matches(base.difficulty):
            continue

        eff = get_effective_ladder_config(base.difficulty, base.board_type, base.num_players)
        artifacts: dict[str, Any] = {}

        if eff.heuristic_profile_id:
            available = eff.heuristic_profile_id in HEURISTIC_WEIGHT_PROFILES
            if not available:
                missing_profiles += 1
            artifacts["heuristic_profile"] = {"id": eff.heuristic_profile_id, "available": available}

        if eff.ai_type == AIType.MINIMAX and eff.use_neural_net:
            nnue_path = get_nnue_model_path(
                board_type=eff.board_type,
                num_players=eff.num_players,
                model_id=eff.model_id,
            )
            exists = nnue_path.exists()
            if exists:
                try:
                    exists = nnue_path.stat().st_size > 0
                except OSError:
                    exists = False
            if not exists:
                missing_nnue += 1
            artifacts["nnue"] = {"model_id": eff.model_id, "path": str(nnue_path), "exists": exists}

        if eff.ai_type in (AIType.MCTS, AIType.DESCENT) and eff.use_neural_net and eff.model_id:
            checkpoint = _resolve_latest_checkpoint(models_dir, eff.model_id)
            exists = checkpoint is not None
            if not exists:
                missing_nn += 1
            neural_artifact: dict[str, Any] = {
                "model_id": eff.model_id,
                "chosen": str(checkpoint) if checkpoint is not None else None,
                "exists": exists,
            }
            if exists and args.load_checkpoints and checkpoint is not None:
                load_result = _try_load_checkpoint(checkpoint)
                neural_artifact.update(
                    {
                        "loadable": bool(load_result.ok),
                        "load_error": load_result.error,
                    }
                )
                if not load_result.ok:
                    corrupt_nn += 1
            artifacts["neural_net"] = neural_artifact

        row = {
            "difficulty": eff.difficulty,
            "board_type": eff.board_type.value,
            "num_players": eff.num_players,
            "ai_type": eff.ai_type.value,
            "use_neural_net": bool(eff.use_neural_net),
            "model_id": eff.model_id,
            "heuristic_profile_id": eff.heuristic_profile_id,
            "artifacts": artifacts,
        }
        if args.verbose:
            rows.append(row)

    summary = {
        "filters": {
            "board_type": board_type.value if board_type else None,
            "num_players": num_players,
            "difficulty": difficulty,
        },
        "summary": {
            "tiers": len(rows) if args.verbose else len(tiers) + (len(base_combos) if _difficulty_matches(1) else 0),
            "missing_heuristic_profiles": missing_profiles,
            "missing_nnue_checkpoints": missing_nnue,
            "missing_neural_checkpoints": missing_nn,
            "corrupt_neural_checkpoints": corrupt_nn,
        },
    }

    if args.json:
        payload = dict(summary)
        if args.verbose:
            payload["tiers"] = rows
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary["summary"], indent=2, sort_keys=True))

    missing_total = missing_profiles + missing_nnue + missing_nn
    if args.fail_on_missing and missing_total > 0:
        return 2
    if args.load_checkpoints and args.fail_on_corrupt and corrupt_nn > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
