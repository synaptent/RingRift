#!/usr/bin/env python
"""Baseline neural-network training experiment for Square-8 2-player.

This script wires up a *minimal* and *cheap* NN training run for Square-8 2p
using the existing training stack under ``ai-service/app/training``.

Goals
-----

- Provide a reproducible, toy-sized NN experiment for Square-8 2-player.
- Exercise the canonical training configuration and model factories.
- Emit a small JSON report plus an optional checkpoint artefact.
- Keep the ``--demo`` path safe and fast enough for CI smoke tests.

This is **not** a production training pipeline. It is intentionally tiny and
meant as a baseline / wiring harness for future work (A3–A7).

Demo mode usage (recommended for CI)
------------------------------------

.. code-block:: bash

    cd ai-service
    PYTHONPATH=. python scripts/run_nn_training_baseline.py \
      --board square8 \
      --num-players 2 \
      --run-dir /tmp/nn_baseline_demo \
      --demo \
      --seed 123

This will:

- Construct a Square-8 2p :class:`TrainConfig` via
  :func:`get_training_config_for_board`.
- Generate a tiny synthetic NPZ dataset in ``--run-dir`` compatible with the
  existing :class:`RingRiftDataset` / :func:`train_model` pipeline.
- Run :func:`train_model` for a single epoch over 128 synthetic samples.
- Write ``nn_training_report.json`` into ``--run-dir`` with:

  - ``board``, ``num_players``, and ``mode`` (``"demo"``),
  - ``model_id`` and resolved paths,
  - basic training params (epochs, batch size, learning rate),
  - a small ``metrics`` block (including ``training_steps`` and ``final_loss``,
    where ``final_loss`` is currently a stub).

Non-demo mode
-------------

Non-demo mode is wired as a **stub** only:

- It constructs a Square-8 2p :class:`TrainConfig` as above.
- It records configuration into the report but **does not** run heavy training.

This is intentional: as of 2025-12-07 there is no fully canonical Square-8
self-play DB in :mod:`TRAINING_DATA_REGISTRY.md`, and the project policy
forbids training new canonical models on legacy or pending-gate data.

Once a canonical Square-8 2p dataset exists, this script can be extended to:

- Export an NPZ dataset from the canonical DB, and
- Call :func:`train_model` with a heavier configuration (more epochs, larger
  datasets) in non-demo mode.

See also
--------

- :mod:`app.training.config` – :class:`TrainConfig`,
  :func:`get_training_config_for_board`.
- :mod:`app.training.train` – :func:`train_model`.
- :mod:`app.ai.neural_net` – :func:`get_model_config_for_board`,
  :func:`create_model_for_board`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType  # noqa: E402
from app.training.config import (  # noqa: E402
    TrainConfig,
    get_training_config_for_board,
)
from app.training.train import train_model  # noqa: E402
from app.training.seed_utils import seed_all  # noqa: E402
from app.ai.neural_net import get_model_config_for_board  # noqa: E402

REPORT_NAME = "nn_training_report.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the baseline NN experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a baseline Square-8 2-player neural-network training "
            "experiment. Intended for tiny demo runs and wiring "
            "validation, not production training."
        ),
    )
    parser.add_argument(
        "--board",
        default="square8",
        help="Board identifier (currently only 'square8' is supported).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (currently only 2 is supported).",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Output directory for nn_training_report.json and any "
            "checkpoints. Will be created if it does not exist."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible demo runs.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use a tiny synthetic dataset and a very short training run "
            "suitable for CI and local smoke tests."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "Optional explicit model id. When omitted an id is generated "
            "based on board/num_players and timestamp."
        ),
    )
    return parser.parse_args(argv)


def _normalise_board(board: str) -> Tuple[str, BoardType]:
    """Normalise board string and return (canonical_name, BoardType).

    For this baseline experiment we support only Square-8 2-player games.
    """
    b = board.strip().lower()
    if b in {"square8", "sq8", "square-8", "8x8"}:
        return "square8", BoardType.SQUARE8
    raise SystemExit(
        f"Unsupported board {board!r}; run_nn_training_baseline currently "
        "supports only Square-8 2-player."
    )


def _ensure_demo_dataset(
    dataset_path: Path,
    board_type: BoardType,
    config: TrainConfig,
    *,
    num_samples: int = 128,
) -> str:
    """Create a tiny synthetic NPZ dataset if it does not already exist.

    The dataset schema is compatible with :class:`RingRiftDataset` and
    :func:`train_model`:

    - features: (N, C, H, W) float32, where C = 10 * (history_length + 1)
    - globals:  (N, 10) float32
    - values:   (N,) float32 in [-1, 1]
    - policy_indices: (N,) object array of int32 indices
    - policy_values:  (N,) object array of float32 probabilities

    All samples are assigned non-empty policy arrays to avoid NaN losses in
    the policy head. The policy indices live in the canonical policy head for
    the given board type.
    """
    if dataset_path.exists():
        return os.fspath(dataset_path)

    history_length = config.history_length
    base_channels = 10
    channels = base_channels * (history_length + 1)

    if board_type == BoardType.SQUARE8:
        board_size = 8
    elif board_type == BoardType.SQUARE19:
        board_size = 19
    else:
        # Hex and other boards are not currently used by this baseline, but we
        # keep the mapping future-proof.
        from app.ai.neural_net import HEX_BOARD_SIZE  # noqa: E402

        board_size = HEX_BOARD_SIZE

    # Resolve policy size either from config or from board defaults.
    from app.ai.neural_net import get_policy_size_for_board  # noqa: E402

    if config.policy_size is not None and config.policy_size > 0:
        policy_size = int(config.policy_size)
    else:
        policy_size = get_policy_size_for_board(board_type)

    rng_seed = config.seed or 123
    rng = np.random.default_rng(rng_seed)

    features = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_samples, channels, board_size, board_size),
    ).astype(np.float32)
    globals_arr = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_samples, 10),
    ).astype(np.float32)

    # Values in [-1, 1]; keep a simple ternary distribution for now.
    values = rng.choice(
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        size=num_samples,
        replace=True,
    ).astype(np.float32)

    policy_indices = np.empty(num_samples, dtype=object)
    policy_values = np.empty(num_samples, dtype=object)

    for i in range(num_samples):
        # Sample a small set of distinct actions per position.
        k = int(rng.integers(1, 6))
        idxs = rng.choice(policy_size, size=k, replace=False)
        probs = rng.random(size=k, dtype=np.float32)
        probs_sum = float(probs.sum())
        if probs_sum > 0:
            probs /= probs_sum
        policy_indices[i] = idxs.astype(np.int32)
        policy_values[i] = probs.astype(np.float32)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        features=features,
        globals=globals_arr,
        values=values,
        policy_indices=policy_indices,
        policy_values=policy_values,
    )
    return os.fspath(dataset_path)


def _build_env_summary(
    board_name: str,
    board_type: BoardType,
    num_players: int,
    seed: int | None,
) -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of the training environment.

    This mirrors the env snapshot used by the tier training pipeline but is
    scoped specifically to Square-8 2-player NN training experiments.
    """
    # For now we keep the environment canonical for Square-8 2p with
    # terminal-only rewards.
    env_cfg = {
        "board_type": board_type.name,
        "board": board_name,
        "num_players": num_players,
        "reward_mode": "terminal",
        "seed": seed,
        "max_moves": 200 if board_type == BoardType.SQUARE8 else None,
    }
    return env_cfg


def _run_demo_training(
    args: argparse.Namespace,
    run_dir: Path,
    board_name: str,
    board_type: BoardType,
) -> Tuple[TrainConfig, Dict[str, Any], Dict[str, Any]]:
    """Run a tiny demo NN training loop suitable for CI.

    Returns (train_config, training_params, metrics) where training_params
    and metrics are ready to be embedded into nn_training_report.json.
    """
    # Seed all RNGs for reproducibility (Python, NumPy, torch).
    seed = args.seed or 123
    seed_all(seed)

    # Start from the board-specific preset so we stay consistent with other
    # tooling.
    cfg = get_training_config_for_board(board_type, TrainConfig())
    cfg.seed = seed

    # Use a small, explicitly demo-labelled model id.
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    model_id = args.model_id or f"{board_name}_nn_baseline_demo_{ts}"
    cfg.model_id = model_id

    # Make the run explicitly tiny: 1 epoch, 1 iteration, modest batch size.
    cfg.epochs_per_iter = 1
    cfg.iterations = 1
    cfg.batch_size = min(cfg.batch_size, 32)

    # Generate a tiny synthetic dataset in the run directory.
    dataset_path = run_dir / "sq8_nn_demo.npz"
    data_path = _ensure_demo_dataset(
        dataset_path,
        board_type,
        cfg,
        num_samples=128,
    )

    # Save checkpoint and any intermediate artefacts under the run directory.
    save_path = run_dir / f"{cfg.model_id}.pth"
    checkpoint_dir = run_dir / "checkpoints"

    # For the demo path we keep scheduling minimal and disable early stopping.
    train_model(
        config=cfg,
        data_path=data_path,
        save_path=os.fspath(save_path),
        early_stopping_patience=0,
        checkpoint_dir=os.fspath(checkpoint_dir),
        checkpoint_interval=1,
        warmup_epochs=0,
        lr_scheduler="none",
        use_streaming=False,
        multi_player=False,
        num_players=args.num_players,
    )

    # Training parameters summary for the report.
    training_params: Dict[str, Any] = {
        "mode": "demo",
        "board_type": cfg.board_type.name,
        "model_id": cfg.model_id,
        "seed": seed,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs_per_iter,
        "iterations": cfg.iterations,
        "learning_rate": cfg.learning_rate,
        "data_path": os.fspath(dataset_path),
        "save_path": os.fspath(save_path),
    }

    # We do not currently plumb through detailed loss curves from train_model;
    # callers can inspect training logs/checkpoints directly if needed.
    metrics: Dict[str, Any] = {
        "training_steps": cfg.epochs_per_iter,
        "final_loss": None,
        "extra": {
            "note": (
                "demo mode; tiny synthetic dataset and 1 epoch of training; "
                "intended for wiring/tests only"
            ),
        },
    }

    return cfg, training_params, metrics


def _run_full_stub(
    args: argparse.Namespace,
    board_name: str,
    board_type: BoardType,
) -> Tuple[TrainConfig, Dict[str, Any], Dict[str, Any]]:
    """Return a stub configuration for non-demo runs without heavy training.

    This path is intentionally conservative: as of 2025-12-07 there is no
    fully canonical Square-8 2p self-play DB in TRAINING_DATA_REGISTRY.md.
    Project policy forbids treating pending-gate or legacy DBs as canonical
    training sources, so this function records configuration only and leaves
    actual training to a future, explicitly-approved task.
    """
    seed = args.seed or 42
    cfg = get_training_config_for_board(board_type, TrainConfig())
    cfg.seed = seed

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    model_id = args.model_id or f"{board_name}_nn_baseline_full_{ts}"
    cfg.model_id = model_id

    # Leave cfg.epochs_per_iter/iterations as preset values for documentation
    # purposes; this describes how a "real" baseline might be run in future.

    training_params: Dict[str, Any] = {
        "mode": "neural_full_stub",
        "board_type": cfg.board_type.name,
        "model_id": cfg.model_id,
        "seed": seed,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs_per_iter,
        "iterations": cfg.iterations,
        "learning_rate": cfg.learning_rate,
        "note": (
            "Non-demo mode is currently a structural stub. As per "
            "TRAINING_DATA_REGISTRY.md, no canonical Square-8 2p dataset "
            "exists yet, so this script intentionally avoids running heavy "
            "training on legacy or pending-gate data."
        ),
    }

    metrics: Dict[str, Any] = {
        "training_steps": 0,
        "loss": None,
        "extra": {},
    }

    return cfg, training_params, metrics


def main(argv: list[str] | None = None) -> int:
    """Entry point for the baseline NN training experiment."""
    args = _parse_args(argv)

    board_name, board_type = _normalise_board(args.board)
    if args.num_players != 2:
        print(
            f"Warning: num_players={args.num_players} is not the canonical "
            "Square-8 2-player setting; proceeding but this configuration is "
            "not covered by the current baseline spec.",
            file=sys.stderr,
        )

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()

    if args.demo:
        cfg, training_params, metrics = _run_demo_training(
            args,
            run_dir,
            board_name,
            board_type,
        )
        mode = "demo"
    else:
        cfg, training_params, metrics = _run_full_stub(
            args,
            board_name,
            board_type,
        )
        mode = "stub"

    env_summary = _build_env_summary(
        board_name=board_name,
        board_type=board_type,
        num_players=args.num_players,
        seed=cfg.seed,
    )

    # Also capture model-shape metadata for downstream tooling.
    model_meta = get_model_config_for_board(cfg.board_type)

    report: Dict[str, Any] = {
        "board": board_name,
        "num_players": args.num_players,
        "mode": mode,
        "created_at": created_at,
        "model_id": cfg.model_id,
        "env": env_summary,
        "train_config": asdict(cfg),
        "model_config": model_meta,
        "training_params": training_params,
        "metrics": metrics,
    }

    report_path = run_dir / REPORT_NAME
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"Wrote baseline NN training report to {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())