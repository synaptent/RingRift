"""Feature extraction utilities for HeuristicAI weight training.

This module exposes helpers that treat :class:`HeuristicAI`'s position
evaluation as *linear in the scalar weight parameters* defined in
``app.ai.heuristic_weights``.

Design
======

``HeuristicAI.evaluate_position`` is implemented as a sum of independent
heuristic components, each of which is scaled by a named weight attribute
(e.g. ``WEIGHT_STACK_CONTROL``, ``WEIGHT_TERRITORY``). For training we
would like a decomposition of the form::

    score(game_state; w) \approx bias(game_state)
                            + sum_i features_i(game_state) * w_i

where ``w_i`` are the scalar weights and ``features_i`` are fixed
features of the game state, independent of the particular profile.

Rather than re-implementing all of HeuristicAI's internals, we compute
these features generically by exploiting linearity:

* Evaluate the position once with **all heuristic weights set to 0** to
  obtain the bias term ``b(game_state)``.
* For each individual weight key ``k`` in the base profile, evaluate the
  position with **only ``k`` set to 1.0** and all other weights set to 0.
  The difference from the bias, ``score_k - b``, is the feature value
  associated with weight ``k``.

This procedure is agnostic to the exact form of the heuristics as long
as they remain linear in the scalar weights, and it automatically tracks
future refactors of :mod:`app.ai.heuristic_ai`.

These helpers are intended for offline training / analysis tools under
``app.training`` and are not used on the runtime path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS
from app.models import AIConfig, GameState

# Stable, canonical ordering of trainable heuristic weight keys.
HEURISTIC_WEIGHT_KEYS: tuple[str, ...] = tuple(
    sorted(BASE_V1_BALANCED_WEIGHTS.keys())
)


@dataclass
class HeuristicFeatures:
    """Linear feature representation for a single (state, player) pair.

    Attributes
    ----------
    features:
        1D numpy array of shape ``[D]`` where
        ``D == len(HEURISTIC_WEIGHT_KEYS)``. Entry ``i`` corresponds to the
        contribution of weight ``HEURISTIC_WEIGHT_KEYS[i]``.
    bias:
        Scalar bias term ``b(game_state)`` that captures all contributions
        to ``evaluate_position`` that do **not** depend on the trainable
        heuristic weights (e.g. diversification bonuses, hard-coded
        influence terms, terminal win/loss handling).
    """

    features: np.ndarray
    bias: float


def _build_ai_with_weights(
    player_number: int,
    weight_overrides: dict[str, float],
) -> HeuristicAI:
    """Construct a HeuristicAI instance with explicit weight overrides.

    The instance is initialised with a dummy difficulty and then its
    weight attributes (``WEIGHT_*``) are overridden per-instance to avoid
    mutating class-level defaults or shared profiles.
    """

    # Difficulty value is largely irrelevant here; we bypass profile
    # lookup by explicitly overriding all weights below.
    # Use a deterministic but otherwise arbitrary RNG configuration.
    # This keeps feature extraction stable while avoiding global RNG
    # side-effects.
    ai = HeuristicAI(
        player_number=player_number,
        config=AIConfig(difficulty=2, randomness=0.0, rngSeed=0),
    )

    for name, value in weight_overrides.items():
        setattr(ai, name, value)

    return ai


def extract_linear_features(
    game_state: GameState,
    player_number: int,
) -> HeuristicFeatures:
    """Extract linear heuristic features for a single game state.

    Parameters
    ----------
    game_state:
        The state to evaluate.
    player_number:
        Perspective for evaluation (1 or 2, etc.). This must match the
        AI's `player_number` so that `evaluate_position` returns the score
        for that player.

    Returns
    -------
    HeuristicFeatures
        ``features`` is a 1D array of length ``len(HEURISTIC_WEIGHT_KEYS)``;
        ``bias`` is the zero-weights baseline score for this state.

    Notes
    -----
    * Terminal states (``game_status == FINISHED``) will typically yield
      very large-magnitude bias values (±100000). For most training
      regimes it is advisable to *exclude* terminal states from the
      dataset or to clamp/transform the targets before fitting.
    """

    # 1. Baseline with all heuristic weights set to zero.
    zero_weights = dict.fromkeys(HEURISTIC_WEIGHT_KEYS, 0.0)
    ai_zero = _build_ai_with_weights(player_number, zero_weights)
    bias = float(ai_zero.evaluate_position(game_state))

    # 2. Per-weight features via one-hot weight vectors.
    feats: list[float] = []

    for key in HEURISTIC_WEIGHT_KEYS:
        w = dict.fromkeys(HEURISTIC_WEIGHT_KEYS, 0.0)
        w[key] = 1.0
        ai_k = _build_ai_with_weights(player_number, w)
        score_k = float(ai_k.evaluate_position(game_state))
        feats.append(score_k - bias)

    return HeuristicFeatures(
        features=np.asarray(feats, dtype=np.float32),
        bias=bias,
    )


def batch_extract_linear_features(
    game_states: Sequence[GameState],
    player_numbers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised helper for extracting features over many examples.

    Parameters
    ----------
    game_states:
        Sequence of :class:`GameState` instances.
    player_numbers:
        Matching sequence of player numbers (same length as ``game_states``).

    Returns
    -------
    (features, biases):
        * ``features``: array of shape ``[N, D]`` where ``D`` is the number
          of heuristic weights.
        * ``biases``: array of shape ``[N]`` containing the per-example
          bias term.
    """

    if len(game_states) != len(player_numbers):
        raise ValueError(
            "game_states and player_numbers must have the same length; "
            f"got {len(game_states)} and {len(player_numbers)}."
        )

    if not game_states:
        return (
            np.zeros((0, len(HEURISTIC_WEIGHT_KEYS)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    all_feats: list[np.ndarray] = []
    all_biases: list[float] = []

    for state, player in zip(game_states, player_numbers, strict=False):
        hf = extract_linear_features(state, player)
        all_feats.append(hf.features)
        all_biases.append(hf.bias)

    return (
        np.stack(all_feats, axis=0).astype(np.float32),
        np.asarray(all_biases, dtype=np.float32),
    )
