"""Offline training script for HeuristicAI weight profiles.

This script provides a first-pass pipeline for fitting the scalar weight
profiles used by :class:`app.ai.heuristic_ai.HeuristicAI` from labelled
examples.

Overview
========

Data format
-----------

The script expects a JSON Lines (``.jsonl``) dataset where each line is a
single example with the following minimal shape::

    {
      "game_state": { ... GameState JSON ... },
      "player_number": 1,
      "target": 0.42
    }

* ``game_state`` must be a JSON object compatible with
  :class:`app.models.GameState` (snake_case field names as used by the
  Python AI service).
* ``player_number`` is the perspective for which the target value is
  defined (the same convention as used by AIs: positive is good for this
  player, negative is bad).
* ``target`` is a scalar evaluation target, typically obtained from a
  stronger teacher (e.g. DescentAI/NeuralNetAI value head, or an
  outcome-based estimate of ``P(win)``).

These files can be produced by future tooling that runs teacher engines
over logged games or specially curated tactical scenarios. For now, they
can also be authored by hand using small snapshots defined in terms of
existing test fixtures.

Model and optimisation
----------------------

Using :mod:`app.training.heuristic_features`, each example is mapped to a
feature vector ``x \in R^D`` and a bias term ``b`` such that the
heuristic evaluation can be approximated as::

    f_w(state) \approx b(state) + x(state) · w

where ``w`` is the vector of heuristic weights (one entry per key in
``BASE_V1_BALANCED_WEIGHTS``).

Given a dataset ``{(x_i, b_i, y_i)}`` of features, biases, and teacher
targets, we fit ``w`` by solving a *ridge regression* problem with
L2-regularisation around the current balanced profile ``w_0``::

    argmin_w  Σ_i (b_i + x_i·w - y_i)^2 + λ ||w - w_0||^2

This yields a closed-form solution::

    (X^T X + λ I) w = X^T (y - b) + λ w_0

which we solve using :func:`numpy.linalg.solve`.

Output
------

The script writes a JSON file containing an updated mapping from profile
ids to weight dictionaries, mirroring the structure of
``HEURISTIC_WEIGHT_PROFILES``::

    {
      "version": "v1",
      "trained_at": "2025-11-23T22:35:00Z",
      "base_profile_id": "heuristic_v1_balanced",
      "profiles": {
        "heuristic_v1_balanced": { ...updated weights... },
        "heuristic_v1_aggressive": { ...unchanged copy... },
        ...
      }
    }

At runtime, these trained profiles can be loaded via an environment
variable (see ``heuristic_weights.load_trained_profiles_if_available``),
allowing designers to swap in tuned weight sets without changing the
code.

Usage
-----

From the ``ai-service`` root::

    python -m app.training.train_heuristic_weights \
        --dataset logs/heuristic/teacher_samples.jsonl \
        --output logs/heuristic/heuristic_profiles.v1.trained.json \
        --lambda 0.001

This is an *offline* tool; it does not affect production behaviour until
its output file is explicitly referenced via configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_PROFILES,
)
from app.models import GameState
from app.training.territory_dataset_validation import (
    validate_territory_dataset_file,
)
from app.training.heuristic_features import (
    HEURISTIC_WEIGHT_KEYS,
    batch_extract_linear_features,
)


@dataclass
class TrainingExample:
    """Single supervised example for heuristic weight fitting.

    ``time_weight`` is an optional per-example weight used to emphasise
    or de-emphasise certain states during training (e.g. gamma^(T - t)
    along a self-play trajectory).
    """

    game_state: GameState
    player_number: int
    target: float
    time_weight: float = 1.0


def _load_jsonl_dataset(path: str) -> List[TrainingExample]:
    """Load a JSONL dataset from *path*.

    Each line must contain ``game_state``, ``player_number``, and
    ``target`` fields as described in the module docstring.
    """

    examples: List[TrainingExample] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc

            try:
                state_payload = data["game_state"]
                player_number = int(data["player_number"])
                target = float(data["target"])
            except KeyError as exc:
                raise ValueError(
                    f"Missing required field {exc!r} on line {line_no}"
                ) from exc

            time_weight = float(data.get("time_weight", 1.0))

            game_state = GameState.model_validate(state_payload)

            examples.append(
                TrainingExample(
                    game_state=game_state,
                    player_number=player_number,
                    target=target,
                    time_weight=time_weight,
                )
            )

    if not examples:
        raise ValueError(f"No training examples found in {path!r}")

    return examples


def _prepare_design_matrix(
    examples: Iterable[TrainingExample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (X, b, y) from an iterable of examples.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape ``[N, D]``.
    b : np.ndarray
        Bias vector of shape ``[N]``.
    y : np.ndarray
        Target vector of shape ``[N]``.
    """

    game_states: List[GameState] = []
    players: List[int] = []
    targets: List[float] = []
    weights: List[float] = []

    for ex in examples:
        game_states.append(ex.game_state)
        players.append(ex.player_number)
        targets.append(ex.target)
        weights.append(ex.time_weight)

    X, b = batch_extract_linear_features(game_states, players)
    y = np.asarray(targets, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)

    return X, b, y, w


def _solve_ridge_regression(
    X: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    base_weights: Dict[str, float],
    lambda_reg: float,
    sample_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Solve the ridge-regression normal equations for new weights.

    Parameters
    ----------
    X, b, y:
        Design matrix, bias vector, and teacher targets as returned by
        :func:`_prepare_design_matrix`.
    base_weights:
        Mapping from weight key to its current scalar value
        (typically ``BASE_V1_BALANCED_WEIGHTS``). This acts as the prior
        ``w_0`` in the regularisation term.
    lambda_reg:
        Non-negative regularisation strength ``λ``. Larger values keep
        the solution closer to ``base_weights``.
    """

    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape [N, D], got {X.shape!r}")

    n_examples, n_features = X.shape
    if n_features != len(HEURISTIC_WEIGHT_KEYS):
        raise ValueError(
            "Feature dimension mismatch: design matrix has "
            f"{n_features} columns, but HEURISTIC_WEIGHT_KEYS has "
            f"{len(HEURISTIC_WEIGHT_KEYS)} entries."
        )

    if n_examples == 0:
        raise ValueError("Empty design matrix; no training examples provided")

    if b.shape != (n_examples,) or y.shape != (n_examples,):
        raise ValueError(
            "Shapes of b and y must both be (N,); got "
            f"b={b.shape!r}, y={y.shape!r}, N={n_examples}."
        )

    if sample_weights is not None and sample_weights.shape != (n_examples,):
        raise ValueError(
            "sample_weights, when provided, must have shape (N,); got "
            f"{sample_weights.shape!r} while N={n_examples}."
        )

    # Target residual after removing bias.
    y_centered = y - b

    # Apply per-example weights via a weighted least-squares transform.
    if sample_weights is not None:
        w_sqrt = np.sqrt(sample_weights.astype(np.float64))
        X_eff = X * w_sqrt[:, None]
        y_eff = y_centered * w_sqrt
    else:
        X_eff = X
        y_eff = y_centered

    # Prior vector w_0 in canonical key order.
    w0 = np.asarray(
        [float(base_weights[k]) for k in HEURISTIC_WEIGHT_KEYS],
        dtype=np.float64,
    )

    # Normal equations: (X_eff^T X_eff + λ I) w = X_eff^T y_eff + λ w0
    XtX = X_eff.T @ X_eff
    Xt_y = X_eff.T @ y_eff

    if lambda_reg > 0.0:
        XtX = XtX + lambda_reg * np.eye(n_features)
        Xt_y = Xt_y + lambda_reg * w0

    # Solve the linear system. We cast to float64 for numerical stability
    # and then back to float32 when constructing the final mapping.
    w = np.linalg.solve(XtX.astype(np.float64), Xt_y.astype(np.float64))

    return {k: float(v) for k, v in zip(HEURISTIC_WEIGHT_KEYS, w.tolist())}


def _build_output_profiles(
    base_profile_id: str,
    new_weights: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Return an updated profile mapping based on *new_weights*.

    For now we only *replace* the specified ``base_profile_id`` and leave
    all other profiles unchanged. This keeps persona definitions
    conservative while still allowing experiments with a new balanced
    profile.
    """

    profiles: Dict[str, Dict[str, float]] = {}

    for pid, weights in HEURISTIC_WEIGHT_PROFILES.items():
        if pid == base_profile_id:
            profiles[pid] = dict(new_weights)
        else:
            # Shallow copy to decouple from the in-memory registry.
            profiles[pid] = dict(weights)

    return profiles


def train_from_jsonl(
    dataset_path: str,
    output_path: str,
    base_profile_id: str = "heuristic_v1_balanced",
    lambda_reg: float = 1e-3,
    validate_territory_schema: bool = False,
    max_validation_errors: int = 50,
) -> None:
    """Convenience entrypoint for training from a JSONL dataset."""

    if validate_territory_schema:
        errors = validate_territory_dataset_file(
            dataset_path, max_errors=max_validation_errors
        )
        if errors:
            preview = "\n".join(
                f"  line {line_no}: {msg}" for line_no, msg in errors[:10]
            )
            raise ValueError(
                f"Territory dataset validation failed for {dataset_path!r} "
                f"with {len(errors)} error(s); first 10:\n{preview}"
            )
        print(f"{dataset_path}: territory schema validation OK")

    examples = _load_jsonl_dataset(dataset_path)
    X, b, y, sample_weights = _prepare_design_matrix(examples)

    print(f"Loaded {len(examples)} examples from {dataset_path!r}")

    base_weights = BASE_V1_BALANCED_WEIGHTS
    new_weights = _solve_ridge_regression(
        X=X,
        b=b,
        y=y,
        base_weights=base_weights,
        lambda_reg=lambda_reg,
        sample_weights=sample_weights,
    )

    # Simple sanity metrics (unweighted MSE on training set)
    w_vec = np.asarray(
        [new_weights[k] for k in HEURISTIC_WEIGHT_KEYS],
        dtype=np.float32,
    )
    preds = b + X @ w_vec
    mse = float(np.mean((preds - y) ** 2))
    print(f"Training MSE: {mse:.4f}")

    profiles = _build_output_profiles(base_profile_id, new_weights)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    payload = {
        "version": "v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_profile_id": base_profile_id,
        "lambda": lambda_reg,
        "num_examples": len(examples),
        "weight_keys": list(HEURISTIC_WEIGHT_KEYS),
        "profiles": profiles,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote trained heuristic profiles to {output_path!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HeuristicAI weight profiles from a JSONL dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL file containing training examples.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path to output JSON file (e.g. "
            "logs/heuristic/heuristic_profiles.v1.trained.json)."
        ),
    )
    parser.add_argument(
        "--base-profile-id",
        default="heuristic_v1_balanced",
        help=(
            "Profile id to treat as the base prior (default: "
            "heuristic_v1_balanced)."
        ),
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_reg",
        type=float,
        default=1e-3,
        help=(
            "L2 regularisation strength λ around the base profile "
            "(default: 1e-3)."
        ),
    )
    parser.add_argument(
        "--validate-territory-schema",
        action="store_true",
        help=(
            "Validate --dataset using the territory/combined-margin JSONL "
            "schema before training (uses territory_dataset_validation)."
        ),
    )
    parser.add_argument(
        "--max-validation-errors",
        type=int,
        default=50,
        help=(
            "Maximum number of territory-schema validation errors to "
            "collect before failing (default: 50)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_from_jsonl(
        dataset_path=args.dataset,
        output_path=args.output,
        base_profile_id=args.base_profile_id,
        lambda_reg=args.lambda_reg,
        validate_territory_schema=args.validate_territory_schema,
        max_validation_errors=args.max_validation_errors,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
