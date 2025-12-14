#!/usr/bin/env python3
"""Bootstrap missing ladder artifacts (NNUE + neural checkpoints).

This script is an optional companion to:
  - `scripts/check_ladder_artifacts.py`
  - `GET /internal/ladder/health`

It is intended to make local dev / sandbox runs robust by ensuring that
all ladder tiers which *expect* neural assets have loadable checkpoints on
disk for every `(board_type, num_players)` combination.

By default this script runs in dry-run mode. Pass `--apply --demo` to
generate minimal, fast placeholder checkpoints using the existing training
entrypoints:
  - NNUE: `scripts/train_nnue.py --demo`
  - NN:   `scripts/run_nn_training_baseline.py --demo`

IMPORTANT:
  - The generated checkpoints are intended for wiring/health checks only.
    They are not high-quality trained models.
  - Checkpoints live under `ai-service/models/**` which is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AI_SERVICE_ROOT.parent

if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.ladder_config import get_effective_ladder_config, list_ladder_tiers  # noqa: E402
from app.models import AIType, BoardType  # noqa: E402


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_boards(values: Optional[list[str]]) -> Optional[set[str]]:
    if not values:
        return None
    allowed = {"square8", "square19", "hexagonal"}
    out: set[str] = set()
    for raw in values:
        key = raw.strip().lower()
        if key not in allowed:
            raise SystemExit(f"Unsupported --boards entry {raw!r}; expected one of {sorted(allowed)}")
        out.add(key)
    return out


def _parse_players(values: Optional[list[int]]) -> Optional[set[int]]:
    if not values:
        return None
    allowed = {2, 3, 4}
    out: set[int] = set()
    for raw in values:
        if raw not in allowed:
            raise SystemExit(f"Unsupported --num-players entry {raw!r}; expected one of {sorted(allowed)}")
        out.add(int(raw))
    return out


def _resolve_latest_checkpoint(models_dir: Path, model_id: str) -> Optional[Path]:
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


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _run(cmd: list[str], *, cwd: Path, timeout_seconds: int) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-4000:]}")


def _write_meta_if_missing(
    *,
    model_id: str,
    board_type: str,
    num_players: int,
    cpu_path: Path,
) -> None:
    meta_path = cpu_path.with_suffix(".meta.json")
    if meta_path.exists():
        return

    payload = {
        "alias_id": model_id,
        "board_type": board_type,
        "num_players": int(num_players),
        "source_model_id": model_id,
        "source_checkpoint": str(cpu_path),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "note": "bootstrap placeholder checkpoint (demo training)",
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_mps_copy(*, cpu_path: Path, mps_path: Path) -> None:
    if _is_nonempty_file(mps_path):
        return
    if not _is_nonempty_file(cpu_path):
        return
    try:
        mps_path.write_bytes(cpu_path.read_bytes())
    except Exception:
        pass


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create missing artifacts (otherwise dry-run).")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo training paths to generate placeholder checkpoints.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "nnue", "neural"],
        default="all",
        help="Restrict which artifacts to create (default: all).",
    )
    parser.add_argument("--boards", nargs="*", default=None, help="Filter: square8 square19 hexagonal")
    parser.add_argument("--num-players", nargs="*", type=int, default=None, help="Filter: 2 3 4")
    parser.add_argument(
        "--run-dir-base",
        default=str(AI_SERVICE_ROOT / "runs" / "ladder_bootstrap"),
        help="Base directory for generated training run logs.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Per-command timeout (default: 1800).")
    parser.add_argument("--force", action="store_true", help="Recreate artifacts even if they exist.")
    args = parser.parse_args(argv)

    boards_filter = _parse_boards(args.boards)
    players_filter = _parse_players(args.num_players)

    tiers = list_ladder_tiers()
    required_nnue: set[tuple[str, int, str]] = set()
    required_nn: set[tuple[str, int, str]] = set()

    for base in tiers:
        eff = get_effective_ladder_config(base.difficulty, base.board_type, base.num_players)
        board_value = eff.board_type.value
        if boards_filter is not None and board_value not in boards_filter:
            continue
        if players_filter is not None and eff.num_players not in players_filter:
            continue

        if eff.ai_type == AIType.MINIMAX and eff.use_neural_net and eff.model_id:
            required_nnue.add((board_value, eff.num_players, eff.model_id))

        if eff.ai_type in (AIType.MCTS, AIType.DESCENT) and eff.use_neural_net and eff.model_id:
            required_nn.add((board_value, eff.num_players, eff.model_id))

    models_dir = AI_SERVICE_ROOT / "models"
    nnue_dir = models_dir / "nnue"
    run_dir_base = Path(args.run_dir_base) / _now_tag()
    os.makedirs(run_dir_base, exist_ok=True)

    created = {"nnue": 0, "neural": 0}
    missing = {"nnue": 0, "neural": 0}

    # NNUE ------------------------------------------------------------------
    if args.only in ("all", "nnue"):
        for board_value, num_players, model_id in sorted(required_nnue):
            target = nnue_dir / f"{model_id}.pt"
            exists = _is_nonempty_file(target)
            if exists and not args.force:
                continue

            missing["nnue"] += 1
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "train_nnue.py"),
                "--board-type",
                board_value,
                "--num-players",
                str(num_players),
                "--model-id",
                model_id,
                "--save-path",
                str(target),
                "--run-dir",
                str(run_dir_base / "nnue" / f"{board_value}_{num_players}p"),
            ]
            if args.demo:
                cmd.append("--demo")

            print(json.dumps({"artifact": "nnue", "model_id": model_id, "path": str(target), "cmd": cmd}))
            if not args.apply:
                continue

            if not args.demo:
                raise SystemExit("Refusing to run non-demo NNUE training in bootstrap. Re-run with --demo.")

            _run(cmd, cwd=PROJECT_ROOT, timeout_seconds=int(args.timeout_seconds))
            if not _is_nonempty_file(target):
                raise RuntimeError(f"NNUE bootstrap reported success but checkpoint missing: {target}")
            created["nnue"] += 1

    # Neural nets -----------------------------------------------------------
    if args.only in ("all", "neural"):
        for board_value, num_players, model_id in sorted(required_nn):
            stable_cpu = models_dir / f"{model_id}.pth"
            stable_mps = models_dir / f"{model_id}_mps.pth"

            # If the stable alias already exists, just ensure its companion
            # artifacts (mps copy + meta) are present.
            if _is_nonempty_file(stable_cpu) and not args.force:
                print(
                    json.dumps(
                        {
                            "artifact": "neural",
                            "model_id": model_id,
                            "action": "ensure_meta_mps",
                            "cpu_path": str(stable_cpu),
                        }
                    )
                )
                if args.apply:
                    _ensure_mps_copy(cpu_path=stable_cpu, mps_path=stable_mps)
                    _write_meta_if_missing(
                        model_id=model_id,
                        board_type=board_value,
                        num_players=num_players,
                        cpu_path=stable_cpu,
                    )
                continue

            chosen = _resolve_latest_checkpoint(models_dir, model_id)
            exists = chosen is not None and _is_nonempty_file(chosen)

            # If we have a usable checkpoint for this model_id prefix but the
            # stable alias is missing, promote it by copying into place.
            if exists and chosen is not None and not args.force:
                print(
                    json.dumps(
                        {
                            "artifact": "neural",
                            "model_id": model_id,
                            "action": "promote_to_stable",
                            "from": str(chosen),
                            "to": str(stable_cpu),
                        }
                    )
                )
                if not args.apply:
                    continue

                try:
                    stable_cpu.write_bytes(chosen.read_bytes())
                except Exception:
                    pass

                if _is_nonempty_file(stable_cpu):
                    _ensure_mps_copy(cpu_path=stable_cpu, mps_path=stable_mps)
                    _write_meta_if_missing(
                        model_id=model_id,
                        board_type=board_value,
                        num_players=num_players,
                        cpu_path=stable_cpu,
                    )
                    continue

            missing["neural"] += 1
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_nn_training_baseline.py"),
                "--board",
                board_value,
                "--num-players",
                str(num_players),
                "--run-dir",
                str(run_dir_base / "nn" / f"{board_value}_{num_players}p"),
                "--model-id",
                model_id,
                "--model-version",
                "v3",
            ]
            if args.demo:
                cmd.append("--demo")

            print(json.dumps({"artifact": "neural", "model_id": model_id, "cmd": cmd}))
            if not args.apply:
                continue

            if not args.demo:
                raise SystemExit("Refusing to run non-demo NN training in bootstrap. Re-run with --demo.")

            _run(cmd, cwd=PROJECT_ROOT, timeout_seconds=int(args.timeout_seconds))

            cpu_path = stable_cpu
            if not _is_nonempty_file(cpu_path):
                # Some training runs may publish timestamped checkpoints. Resolve again.
                cpu_path = _resolve_latest_checkpoint(models_dir, model_id) or cpu_path
                if _is_nonempty_file(cpu_path) and cpu_path != stable_cpu:
                    try:
                        stable_cpu.write_bytes(cpu_path.read_bytes())
                        cpu_path = stable_cpu
                    except Exception:
                        pass

            if not _is_nonempty_file(cpu_path):
                raise RuntimeError(f"Neural bootstrap reported success but checkpoint missing: {model_id}")

            # Best-effort: ensure the stable alias paths exist for MPS and metadata.
            _ensure_mps_copy(cpu_path=cpu_path, mps_path=stable_mps)
            _write_meta_if_missing(
                model_id=model_id,
                board_type=board_value,
                num_players=num_players,
                cpu_path=cpu_path,
            )
            created["neural"] += 1

    # Summary ---------------------------------------------------------------
    print(
        json.dumps(
            {
                "apply": bool(args.apply),
                "demo": bool(args.demo),
                "missing": missing,
                "created": created,
                "run_dir": str(run_dir_base),
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
