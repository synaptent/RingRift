#!/usr/bin/env python
"""
Train Square-board NN models (v2 and v3) from canonical data and compare them.

This is a thin orchestration wrapper around:
  - scripts/build_canonical_dataset.py (optional)
  - app/training/train.py (v2/v3 training)
  - scripts/evaluate_ai_models.py (NN vs NN evaluation)

Canonical SSOT:
  - Rules: RULES_CANONICAL_SPEC.md
  - Engine semantics: src/shared/engine/**

Usage (from ai-service/):

  # Quick sanity run (small epochs + small eval)
  PYTHONPATH=. python scripts/train_v2_v3_and_compare.py --quick

  # Full-ish run on canonical square8 2p dataset
  PYTHONPATH=. python scripts/train_v2_v3_and_compare.py \
    --board-type square8 --num-players 2 \
    --epochs 50 --batch-size 256 --learning-rate 0.001 \
    --eval-games 200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path = AI_SERVICE_ROOT,
    env_overrides: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
    )
    return int(proc.returncode)


def _default_dataset_path(board_type: str, num_players: int) -> Path:
    return (AI_SERVICE_ROOT / "data" / "training" / f"canonical_{board_type}_{num_players}p.npz").resolve()


def _default_models_dir() -> Path:
    return (AI_SERVICE_ROOT / "models").resolve()


def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _train_one(
    *,
    model_version: str,
    board_type: str,
    num_players: int,
    data_path: Path,
    save_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    checkpoint_dir: Path | None,
    checkpoint_interval: int,
) -> int:
    cmd = [
        sys.executable,
        "app/training/train.py",
        "--data-path",
        str(data_path),
        "--save-path",
        str(save_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(seed),
        "--board-type",
        board_type,
        "--num-players",
        str(num_players),
        "--model-version",
        model_version,
    ]

    if checkpoint_dir is not None:
        cmd += [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-interval",
            str(checkpoint_interval),
        ]

    env_overrides = {
        "PYTHONPATH": str(AI_SERVICE_ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    print(f"[train] {model_version}: saving to {save_path}", file=sys.stderr, flush=True)
    return _run_cmd(cmd, env_overrides=env_overrides)


def _evaluate_nn_vs_nn(
    *,
    checkpoint_a: Path,
    checkpoint_b: Path,
    board_type: str,
    games: int,
    seed: int,
    max_moves: int,
    output_path: Path,
) -> int:
    cmd = [
        sys.executable,
        "scripts/evaluate_ai_models.py",
        "--player1",
        "neural_network",
        "--player2",
        "neural_network",
        "--checkpoint",
        str(checkpoint_a),
        "--checkpoint2",
        str(checkpoint_b),
        "--board",
        board_type,
        "--games",
        str(games),
        "--max-moves",
        str(int(max_moves)),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    env_overrides = {
        "PYTHONPATH": str(AI_SERVICE_ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    print(
        f"[eval] nn vs nn: games={games} board={board_type}\n"
        f"  p1={checkpoint_a.name}\n"
        f"  p2={checkpoint_b.name}\n"
        f"  out={output_path}",
        file=sys.stderr,
        flush=True,
    )
    return _run_cmd(cmd, env_overrides=env_overrides)


def _maybe_rebuild_canonical_dataset(
    *,
    board_type: str,
    num_players: int,
    db_path: Path | None,
    output_path: Path,
    allow_pending_gate: bool,
    legacy_maxn_encoding: bool,
) -> int:
    cmd = [
        sys.executable,
        "scripts/build_canonical_dataset.py",
        "--board-type",
        board_type,
        "--num-players",
        str(num_players),
        "--output",
        str(output_path),
    ]
    if db_path is not None:
        cmd += ["--db", str(db_path)]
    if allow_pending_gate:
        cmd.append("--allow-pending-gate")
    if legacy_maxn_encoding:
        cmd.append("--legacy-maxn-encoding")

    env_overrides = {
        "PYTHONPATH": str(AI_SERVICE_ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    print(f"[dataset] rebuilding: {output_path}", file=sys.stderr, flush=True)
    return _run_cmd(cmd, env_overrides=env_overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train v2 and v3 models and run an evaluation match.")
    parser.add_argument(
        "--board-type",
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type (default: square8).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to an NPZ dataset. Defaults to data/training/canonical_<board>_<num_players>p.npz.",
    )
    parser.add_argument(
        "--allow-noncanonical-data",
        action="store_true",
        help="Allow using datasets whose basename does not start with 'canonical_'.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the canonical dataset via scripts/build_canonical_dataset.py before training.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional canonical DB path to export from when --rebuild-dataset is set.",
    )
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help="Allow Status=pending_gate in TRAINING_DATA_REGISTRY.md when rebuilding datasets (still requires canonical_ok=true).",
    )
    parser.add_argument(
        "--legacy-maxn-encoding",
        action="store_true",
        help=(
            "Use legacy MAX_N policy encoding when rebuilding datasets. "
            "Not compatible with v3 training; leave unset for board-aware encoding."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for each model (default: 50).")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size (default: 256).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Training learning rate (default: 1e-3).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42).")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to write model checkpoints (default: ai-service/models).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional string tag to include in output filenames (default: timestamp).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional directory to write intermediate checkpoints (default: disabled).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Checkpoint interval in epochs when --checkpoint-dir is set (default: 10).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=200,
        help="Number of evaluation games for v3 vs v2 (default: 200).",
    )
    parser.add_argument(
        "--eval-max-moves",
        type=int,
        default=10000,
        help="Max moves per evaluation game before timeout tie-breaks (default: 10000).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the evaluation match (train only).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: --epochs 3 --eval-games 40 (intended for smoke tests).",
    )
    args = parser.parse_args(argv)

    board_type: str = args.board_type
    num_players: int = args.num_players

    models_dir = Path(args.models_dir).resolve() if args.models_dir else _default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or _timestamp_tag()

    data_path = Path(args.data_path).resolve() if args.data_path else _default_dataset_path(board_type, num_players)
    if not args.allow_noncanonical_data and not data_path.name.startswith("canonical_"):
        raise SystemExit(
            f"[train-v2-v3] Refusing to train on non-canonical dataset: {data_path}\n"
            "Pass --allow-noncanonical-data to override (not recommended for canonical training)."
        )

    if args.quick:
        args.epochs = min(args.epochs, 3)
        args.eval_games = min(args.eval_games, 40)

    if args.rebuild_dataset:
        db_path = Path(args.db).resolve() if args.db else None
        rc = _maybe_rebuild_canonical_dataset(
            board_type=board_type,
            num_players=num_players,
            db_path=db_path,
            output_path=data_path,
            allow_pending_gate=bool(args.allow_pending_gate),
            legacy_maxn_encoding=bool(args.legacy_maxn_encoding),
        )
        if rc != 0:
            raise SystemExit(f"[train-v2-v3] Dataset rebuild failed (rc={rc}).")

    if not data_path.exists():
        raise SystemExit(f"[train-v2-v3] Dataset not found: {data_path}")

    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    if checkpoint_dir is None:
        checkpoint_dir = models_dir / "checkpoints" / f"exp_v2v3_{board_type}_{num_players}p_{tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir_v2 = checkpoint_dir / "v2"
    checkpoint_dir_v3 = checkpoint_dir / "v3"
    checkpoint_dir_v2.mkdir(parents=True, exist_ok=True)
    checkpoint_dir_v3.mkdir(parents=True, exist_ok=True)

    v2_path = models_dir / f"ringrift_exp_v2_{board_type}_{num_players}p_{tag}.pth"
    v3_path = models_dir / f"ringrift_exp_v3_{board_type}_{num_players}p_{tag}.pth"

    rc = _train_one(
        model_version="v2",
        board_type=board_type,
        num_players=num_players,
        data_path=data_path,
        save_path=v2_path,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        checkpoint_dir=checkpoint_dir_v2,
        checkpoint_interval=int(args.checkpoint_interval),
    )
    if rc != 0:
        raise SystemExit(f"[train-v2-v3] v2 training failed (rc={rc}).")

    rc = _train_one(
        model_version="v3",
        board_type=board_type,
        num_players=num_players,
        data_path=data_path,
        save_path=v3_path,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed) + 1,
        checkpoint_dir=checkpoint_dir_v3,
        checkpoint_interval=int(args.checkpoint_interval),
    )
    if rc != 0:
        raise SystemExit(f"[train-v2-v3] v3 training failed (rc={rc}).")

    print(
        "\n".join(
            [
                "[train-v2-v3] training complete:",
                f"  v2={v2_path}",
                f"  v3={v3_path}",
            ]
        ),
        file=sys.stderr,
        flush=True,
    )

    if args.skip_eval:
        return 0

    results_dir = models_dir / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_out = results_dir / f"nn_v3_vs_v2_{board_type}_{num_players}p_{tag}.json"
    rc = _evaluate_nn_vs_nn(
        checkpoint_a=v3_path,
        checkpoint_b=v2_path,
        board_type=board_type,
        games=int(args.eval_games),
        seed=int(args.seed) + 2,
        max_moves=int(args.eval_max_moves),
        output_path=eval_out,
    )
    if rc != 0:
        raise SystemExit(f"[train-v2-v3] evaluation failed (rc={rc}).")

    print(f"[train-v2-v3] evaluation complete: {eval_out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
