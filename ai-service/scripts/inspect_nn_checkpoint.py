#!/usr/bin/env python
"""
Inspect a RingRift neural-network checkpoint for architecture compatibility.

This is a lightweight diagnostic helper intended for debugging issues like:
  - "value_fc1 in_features: checkpoint=212, expected=148"
  - policy head size mismatches after action-encoding changes
  - stale/empty checkpoint selection when using nn_model_id prefixes

Examples (from ai-service/):

  PYTHONPATH=. python scripts/inspect_nn_checkpoint.py \\
    --nn-model-id ringrift_v3_sq8_2p \\
    --board-type square8

  PYTHONPATH=. python scripts/inspect_nn_checkpoint.py \\
    --checkpoint models/sq8_2p_nn_baseline_20251211_163524.pth \\
    --board-type square8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from app.models import BoardType


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _parse_board_type(raw: str) -> BoardType:
    raw = raw.strip().lower()
    if raw in {"square8", "sq8", "8"}:
        return BoardType.SQUARE8
    if raw in {"square19", "sq19", "19"}:
        return BoardType.SQUARE19
    if raw in {"hex", "hexagonal"}:
        return BoardType.HEXAGONAL
    raise ValueError(f"Unknown board_type: {raw}")


def _board_suffix(board_type: BoardType) -> str:
    if board_type == BoardType.HEXAGONAL:
        return "_hex"
    if board_type == BoardType.SQUARE19:
        return "_19x19"
    return ""


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_checkpoint_from_nn_model_id(
    nn_model_id: str,
    board_type: BoardType,
) -> Path:
    """Resolve the checkpoint file path using the same conventions as NeuralNetAI."""
    models_dir = AI_SERVICE_ROOT / "models"
    suffix = _board_suffix(board_type)

    # Prefer MPS-specific files when available, but accept non-MPS fallbacks.
    candidates = [
        models_dir / f"{nn_model_id}{suffix}_mps.pth",
        models_dir / f"{nn_model_id}{suffix}.pth",
    ]
    if suffix:
        candidates.extend(
            [
                models_dir / f"{nn_model_id}_mps.pth",
                models_dir / f"{nn_model_id}.pth",
            ]
        )

    for c in candidates:
        if _is_nonempty_file(c):
            return c

    # Fall back to prefix scan (timestamped checkpoints).
    prefix = f"{nn_model_id}{suffix}"
    matches = list(models_dir.glob(f"{prefix}_*.pth")) + list(models_dir.glob(f"{prefix}_*_mps.pth"))
    matches = [m for m in matches if _is_nonempty_file(m)]
    if matches:
        matches.sort(key=lambda p: p.stat().st_mtime)
        return matches[-1]

    raise FileNotFoundError(
        f"No checkpoint found for nn_model_id={nn_model_id!r} (board={board_type.value}) "
        f"under {models_dir}"
    )


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt).__name__}")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Best-effort legacy format: direct state_dict.
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt  # type: ignore[assignment]
        else:
            raise KeyError(
                "Checkpoint does not contain 'model_state_dict' or a recognizable legacy state_dict."
            )

    if not isinstance(state_dict, dict):
        raise TypeError("state_dict is not a dict")

    meta = ckpt.get("_versioning_metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    return meta, state_dict  # type: ignore[return-value]


def _shape(t: torch.Tensor | None) -> tuple[int, ...] | None:
    if t is None:
        return None
    if not hasattr(t, "shape"):
        return None
    return tuple(int(x) for x in t.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--nn-model-id", type=str, default=None, help="Model ID/prefix to resolve under models/.")
    parser.add_argument("--board-type", type=str, default="square8", help="square8|square19|hexagonal")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on detected mismatch.")
    args = parser.parse_args()

    board_type = _parse_board_type(args.board_type)

    if bool(args.checkpoint) == bool(args.nn_model_id):
        print("Provide exactly one of --checkpoint or --nn-model-id", file=sys.stderr)
        return 2

    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_absolute():
            path = (AI_SERVICE_ROOT / path).resolve()
    else:
        path = _resolve_checkpoint_from_nn_model_id(args.nn_model_id, board_type)  # type: ignore[arg-type]

    meta, state_dict = _load_checkpoint(path)

    cfg = meta.get("config") if isinstance(meta, dict) else None
    if not isinstance(cfg, dict):
        cfg = {}

    model_class = meta.get("model_class")
    arch_version = meta.get("architecture_version")

    conv1 = state_dict.get("conv1.weight")
    value_fc1 = state_dict.get("value_fc1.weight")
    value_fc2 = state_dict.get("value_fc2.weight")
    rank_dist_fc2 = state_dict.get("rank_dist_fc2.weight")
    policy_fc = state_dict.get("policy_fc.weight")

    inferred_filters = _shape(conv1)[0] if _shape(conv1) else None
    inferred_in_features = _shape(value_fc1)[1] if _shape(value_fc1) else None
    inferred_num_players = _shape(value_fc2)[0] if _shape(value_fc2) else None

    print(f"Checkpoint: {path}")
    print(f"Size: {path.stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"Model class: {model_class}")
    print(f"Architecture version: {arch_version}")
    print(f"Config.num_filters: {cfg.get('num_filters')}")
    print(f"Config.num_res_blocks: {cfg.get('num_res_blocks')}")
    print(f"Config.policy_size: {cfg.get('policy_size')}")
    print(f"Config.num_players: {cfg.get('num_players')}")
    print(f"Config.global_features: {cfg.get('global_features')}")
    print(f"Config.history_length: {cfg.get('history_length')}")
    print(f"conv1.weight.shape: {_shape(conv1)}  (filters inferred={inferred_filters})")
    print(f"value_fc1.weight.shape: {_shape(value_fc1)}  (in_features inferred={inferred_in_features})")
    print(f"value_fc2.weight.shape: {_shape(value_fc2)}  (num_players inferred={inferred_num_players})")
    if rank_dist_fc2 is not None:
        print(f"rank_dist_fc2.weight.shape: {_shape(rank_dist_fc2)}")
    print(f"policy_fc.weight.shape: {_shape(policy_fc)}")

    # Basic compatibility checks.
    ok = True
    if inferred_filters is not None and cfg.get("num_filters") is not None and int(cfg["num_filters"]) != int(inferred_filters):
        ok = False
        print(
            f"Mismatch: metadata num_filters={cfg['num_filters']} vs weights={inferred_filters}",
            file=sys.stderr,
        )

    if inferred_num_players is not None and cfg.get("num_players") is not None and int(cfg["num_players"]) != int(inferred_num_players):
        ok = False
        print(
            f"Mismatch: metadata num_players={cfg['num_players']} vs weights={inferred_num_players}",
            file=sys.stderr,
        )

    if inferred_in_features is not None and inferred_filters is not None:
        globals_expected = cfg.get("global_features")
        if globals_expected is not None:
            expected_in = int(inferred_filters) + int(globals_expected)
            if int(inferred_in_features) != expected_in:
                ok = False
                print(
                    f"Mismatch: value_fc1 in_features={inferred_in_features} "
                    f"!= filters+global_features ({inferred_filters}+{globals_expected}={expected_in})",
                    file=sys.stderr,
                )

    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
