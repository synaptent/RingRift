"""Heuristic weight profiles for RingRift.

This module centralises all scalar weights used by :class:`HeuristicAI` for
position evaluation, and exposes named, versioned profiles that can be
referenced from both runtime configs (``AIConfig.heuristic_profile_id``) and
offline training/tuning tools.

Design goals:

* Keep a **single source of truth** for heuristic weights instead of
  scattering literals across the evaluator.
* Support **versioned base profiles** (e.g. ``heuristic_v1_balanced``)
  backed by RingRift-specific signals described in ``AI_ARCHITECTURE.md``.
* Allow lightweight **personas** (aggressive / territorial / defensive)
  implemented as small deltas over the balanced profile.
* Remain JSON-serialisable so training pipelines can snapshot and tune
  profiles without impacting runtime defaults.

The keys in each profile intentionally mirror the attribute names on
:class:`HeuristicAI` (``WEIGHT_STACK_CONTROL``, ``WEIGHT_TERRITORY``, etc.)
so that instances can simply ``setattr(self, name, value)`` when applying a
profile.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Mapping


HeuristicWeights = Dict[str, float]


# --- v1 Balanced Base Profile -------------------------------------------------
#
# These weights are derived from the legacy class-level constants in
# ``heuristic_ai.py`` and represent the current "balanced" behaviour of
# HeuristicAI. Future training runs are expected to update these numbers
# (typically via offline regression/ML), but the *shape* of the profile and
# the semantic meaning of each key should remain stable.

BASE_V1_BALANCED_WEIGHTS: HeuristicWeights = {
    "WEIGHT_STACK_CONTROL": 10.0,
    "WEIGHT_STACK_HEIGHT": 5.0,
    "WEIGHT_TERRITORY": 8.0,
    "WEIGHT_RINGS_IN_HAND": 3.0,
    "WEIGHT_CENTER_CONTROL": 4.0,
    "WEIGHT_ADJACENCY": 2.0,
    "WEIGHT_OPPONENT_THREAT": 6.0,
    "WEIGHT_MOBILITY": 4.0,
    "WEIGHT_ELIMINATED_RINGS": 12.0,
    "WEIGHT_LINE_POTENTIAL": 7.0,
    "WEIGHT_VICTORY_PROXIMITY": 20.0,
    "WEIGHT_MARKER_COUNT": 1.5,
    "WEIGHT_VULNERABILITY": 8.0,
    "WEIGHT_OVERTAKE_POTENTIAL": 8.0,
    "WEIGHT_TERRITORY_CLOSURE": 10.0,
    "WEIGHT_LINE_CONNECTIVITY": 6.0,
    "WEIGHT_TERRITORY_SAFETY": 5.0,
    "WEIGHT_STACK_MOBILITY": 4.0,
}


def _with_deltas(
    base: Mapping[str, float],
    *,
    scale: Mapping[str, float] | None = None,
    offset: Mapping[str, float] | None = None,
) -> HeuristicWeights:
    """Create a new profile from *base* by applying per-key scale/offset.

    This is deliberately simple: it is intended for designer-friendly persona
    tweaks (e.g. +20% capture emphasis, -20% territory safety) rather than
    for gradient-based optimisation. All keys in ``base`` are preserved so
    that profiles remain structurally compatible.
    """

    scale = scale or {}
    offset = offset or {}
    out: HeuristicWeights = {}
    for key, value in base.items():
        s = scale.get(key, 1.0)
        o = offset.get(key, 0.0)
        out[key] = value * s + o
    return out


# --- v1 Personas --------------------------------------------------------------
#
# These personas are defined as small, interpretable adjustments around the
# balanced profile. They are intentionally conservative so as not to break
# existing expectations while still giving designers room to experiment.

# Balanced: identical to the base profile.
HEURISTIC_V1_BALANCED: HeuristicWeights = dict(BASE_V1_BALANCED_WEIGHTS)

# Aggressive: emphasise capture/elimination and short-term victory progress,
# dial back territory safety and vulnerability penalties slightly so the AI
# is more willing to take calculated risks.
HEURISTIC_V1_AGGRESSIVE: HeuristicWeights = _with_deltas(
    BASE_V1_BALANCED_WEIGHTS,
    scale={
        "WEIGHT_ELIMINATED_RINGS": 1.25,
        "WEIGHT_OVERTAKE_POTENTIAL": 1.25,
        "WEIGHT_VICTORY_PROXIMITY": 1.15,
        "WEIGHT_LINE_POTENTIAL": 1.1,
        # Slightly downweight safety concerns
        "WEIGHT_VULNERABILITY": 0.85,
        "WEIGHT_TERRITORY_SAFETY": 0.85,
    },
)

# Territorial: emphasise territory ownership/closure and line structure,
# de-emphasise raw elimination so the AI is more likely to play for space and
# long-term region control.
HEURISTIC_V1_TERRITORIAL: HeuristicWeights = _with_deltas(
    BASE_V1_BALANCED_WEIGHTS,
    scale={
        "WEIGHT_TERRITORY": 1.25,
        "WEIGHT_TERRITORY_CLOSURE": 1.25,
        "WEIGHT_TERRITORY_SAFETY": 1.2,
        "WEIGHT_LINE_POTENTIAL": 1.15,
        "WEIGHT_LINE_CONNECTIVITY": 1.15,
        "WEIGHT_MARKER_COUNT": 1.1,
        # Slightly downweight pure elimination focus
        "WEIGHT_ELIMINATED_RINGS": 0.9,
    },
)

# Defensive/control: prioritise stack safety, vulnerability awareness, and
# general mobility; slightly reduce aggression so the AI avoids speculative
# captures that leave it exposed.
HEURISTIC_V1_DEFENSIVE: HeuristicWeights = _with_deltas(
    BASE_V1_BALANCED_WEIGHTS,
    scale={
        "WEIGHT_STACK_CONTROL": 1.15,
        "WEIGHT_STACK_MOBILITY": 1.2,
        "WEIGHT_MOBILITY": 1.1,
        "WEIGHT_VULNERABILITY": 1.25,
        "WEIGHT_TERRITORY_SAFETY": 1.15,
        # Tone down overtake/elimination eagerness a bit
        "WEIGHT_OVERTAKE_POTENTIAL": 0.9,
        "WEIGHT_ELIMINATED_RINGS": 0.9,
    },
)


# --- Registry ------------------------------------------------------------------
#
# Public mapping from profile_id -> HeuristicWeights. The keys here are what
# ``AIConfig.heuristic_profile_id`` and the canonical difficulty ladder refer
# to. We include both high-level persona ids and the ladder-oriented ids used
# today so that existing configs/tests continue to work unchanged.

HEURISTIC_WEIGHT_PROFILES: Dict[str, HeuristicWeights] = {
    # High-level, persona-oriented ids (preferred for new configs).
    "heuristic_v1_balanced": HEURISTIC_V1_BALANCED,
    "heuristic_v1_aggressive": HEURISTIC_V1_AGGRESSIVE,
    "heuristic_v1_territorial": HEURISTIC_V1_TERRITORIAL,
    "heuristic_v1_defensive": HEURISTIC_V1_DEFENSIVE,
    # Canonical ladder-linked ids. These currently all reference the
    # balanced profile but can be re-pointed in future without changing the
    # external difficulty contract.
    "v1-heuristic-2": HEURISTIC_V1_BALANCED,
    "v1-heuristic-3": HEURISTIC_V1_BALANCED,
    "v1-heuristic-4": HEURISTIC_V1_BALANCED,
    "v1-heuristic-5": HEURISTIC_V1_BALANCED,
}


def get_weights(profile_id: str) -> HeuristicWeights:
    """Return the weight profile for ``profile_id``.

    Callers should treat a missing id as "no override" and fall back to
    whatever defaults their caller provides (typically the balanced profile
    baked into :class:`HeuristicAI`). This helper deliberately does *not*
    raise if the profile is unknown to keep runtime behaviour backward
    compatible with earlier versions that did not know about personas.
    """

    return HEURISTIC_WEIGHT_PROFILES.get(profile_id, {})


TRAINED_PROFILES_ENV = "RINGRIFT_TRAINED_HEURISTIC_PROFILES"


def load_trained_profiles_if_available(
    path: str | None = None,
    *,
    mode: str = "override",
    suffix: str = "_trained",
) -> Dict[str, HeuristicWeights]:
    """Load trained heuristic profiles from JSON and merge into the registry.

    Parameters
    ----------
    path:
        Optional explicit path to a JSON file produced by
        :mod:`app.training.train_heuristic_weights`. If omitted, the helper
        looks for the ``RINGRIFT_TRAINED_HEURISTIC_PROFILES`` environment
        variable.
    mode:
        - ``"override"``: replace existing entries in
          :data:`HEURISTIC_WEIGHT_PROFILES` with trained values (good for
          production toggles).
        - ``"suffix"``: register trained copies under new ids of the form
          ``f"{profile_id}{suffix}"`` while leaving the baseline ids
          untouched (good for A/B experiments).
    suffix:
        Suffix used when ``mode == "suffix"``.

    Returns
    -------
    Dict[str, HeuristicWeights]
        Mapping of *newly* registered profile ids to their weights.
    """

    if path is None:
        path = os.getenv(TRAINED_PROFILES_ENV)

    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    profiles = payload.get("profiles", {})
    loaded: Dict[str, HeuristicWeights] = {}

    for pid, weights in profiles.items():
        if mode == "override":
            target_id = pid
        elif mode == "suffix":
            target_id = f"{pid}{suffix}"
        else:
            # Unknown mode: skip silently to preserve behaviour rather than
            # raising at runtime.
            continue

        HEURISTIC_WEIGHT_PROFILES[target_id] = dict(weights)
        loaded[target_id] = HEURISTIC_WEIGHT_PROFILES[target_id]

    return loaded
