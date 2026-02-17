#!/usr/bin/env python3
"""Composite Participant ID System for tracking (NN, Algorithm) combinations.

This module provides utilities for managing composite participant IDs that track
neural network + search algorithm combinations as distinct ELO-rated participants.

Participant ID Schema:
    {nn_id}:{ai_type}:{config_hash}

Components:
    - nn_id: Neural network identifier (e.g., "ringrift_v5_sq8_2p") or "none"
    - ai_type: Search algorithm (e.g., "gumbel_mcts", "mcts", "policy_only")
    - config_hash: Short encoding of algorithm configuration (e.g., "b200", "s800")

Config Hash Encoding:
    | Prefix | Meaning              | Example                |
    |--------|----------------------|------------------------|
    | b{N}   | Budget/simulations   | b200 = 200 budget      |
    | s{N}   | Search simulations   | s800 = 800 MCTS sims   |
    | d{N}   | Difficulty level     | d6 = difficulty 6      |
    | t{N}   | Temperature          | t0.3 = temp 0.3        |
    | k{N}   | K-factor override    | k32 = K=32             |

Examples:
    ringrift_v5_sq8_2p:gumbel_mcts:b200     # Gumbel with budget=200
    ringrift_v5_sq8_2p:mcts:s800            # MCTS with 800 simulations
    ringrift_v5_sq8_2p:descent:d6           # Descent at difficulty 6
    ringrift_v5_sq8_2p:policy_only:t0.3     # Policy-only, temperature=0.3
    none:heuristic:d2                       # Heuristic baseline (no NN)
    none:random:d1                          # Random baseline

Usage:
    from app.training.composite_participant import (
        make_composite_participant_id,
        parse_composite_participant_id,
        get_standard_config,
        is_composite_id,
    )

    # Create a composite ID
    pid = make_composite_participant_id(
        nn_id="ringrift_v5_sq8_2p",
        ai_type="gumbel_mcts",
        config={"budget": 200}
    )
    # Returns: "ringrift_v5_sq8_2p:gumbel_mcts:b200"

    # Parse a composite ID
    nn_id, ai_type, config = parse_composite_participant_id(pid)
    # Returns: ("ringrift_v5_sq8_2p", "gumbel_mcts", {"budget": 200})
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Standard algorithm configurations (default values when not specified)
STANDARD_ALGORITHM_CONFIGS: dict[str, dict[str, Any]] = {
    "random": {
        "difficulty": 1,
    },
    "heuristic": {
        "difficulty": 2,
        "randomness": 0.3,
    },
    "policy_only": {
        "temperature": 0.3,
        "top_k": None,
    },
    "mcts": {
        "simulations": 800,
        "c_puct": 1.5,
        "use_neural_net": True,
    },
    "gumbel_mcts": {
        "budget": 200,
        "m": 16,
        "use_neural_net": True,
    },
    "descent": {
        "difficulty": 6,
        "time_ms": 5000,
        "use_neural_net": True,
    },
    "ebmo": {
        "direct_eval": True,
        "optim_steps": 20,
        "num_restarts": 3,
    },
    "gmo": {
        "optim_steps": 50,
        "learning_rate": 0.1,
    },
    "gmo_gumbel": {
        "budget": 150,
        "m": 16,
    },
    # NNUE-compatible algorithms
    "minimax": {
        "depth": 6,
        "use_nnue": True,
        "alpha_beta": True,
    },
    # Multi-player NNUE algorithms
    "maxn": {
        "depth": 7,
        "use_nnue": True,
    },
    "brs": {
        "depth": 5,
        "use_nnue": True,
    },
}

# Config hash key mapping (config parameter → hash prefix)
CONFIG_HASH_KEYS = {
    "budget": "b",           # Budget/simulations for Gumbel
    "simulations": "s",      # MCTS simulations
    "difficulty": "d",       # Difficulty level
    "temperature": "t",      # Temperature parameter
    "k_factor": "k",         # K-factor override
    "time_ms": "ms",         # Time limit in ms
    "optim_steps": "o",      # Optimization steps
    "c_puct": "c",           # PUCT constant
    "m": "m",                # Gumbel m parameter
    "learning_rate": "lr",   # Learning rate
    "num_restarts": "r",     # Number of restarts
    "direct_eval": "de",     # Direct eval mode (bool)
    "depth": "dp",           # Search depth (minimax, maxn, brs)
}

# Reverse mapping for parsing
HASH_KEY_TO_CONFIG = {v: k for k, v in CONFIG_HASH_KEYS.items()}


class ParticipantCategory(Enum):
    """Categories of participants for different tracking purposes."""
    BASELINE = "baseline"        # Fixed anchor points, no NN (e.g., none:random:d1)
    PURE_NN = "pure_nn"          # NN strength without search (e.g., nn:policy_only:t0.3)
    NN_SEARCH = "nn_search"      # Full NN + search combination
    SEARCH_ONLY = "search_only"  # Heuristic eval with search (no NN)
    NNUE = "nnue"                # NNUE with Minimax (2p zero-sum)
    NNUE_MP = "nnue_mp"          # Multi-player NNUE with MaxN/BRS
    LEGACY = "legacy"            # Old-format participant ID (not composite)


class ModelType(Enum):
    """Model architecture types for unified tracking.

    Used to categorize models by their architecture version for:
    - Performance comparison across architectures
    - Training compute allocation based on architecture efficiency
    - Unified Elo tracking of model+harness combinations

    Naming convention:
        {type}_{version}

    Where type is:
        - nn: Full neural network (outputs policy logits + value)
        - nnue: NNUE (outputs scalar value, 2p zero-sum)
        - nnue_mp: Multi-player NNUE (outputs per-player values)
    """
    # Full Neural Networks (policy + value)
    NN_V2 = "nn_v2"              # Original v2 architecture
    NN_V3 = "nn_v3"              # v3 with SE attention
    NN_V3_1 = "nn_v3.1"          # v3.1 minor improvements
    NN_V4 = "nn_v4"              # v4 architecture
    NN_V5 = "nn_v5"              # v5 standard
    NN_V5_HEAVY = "nn_v5_heavy"  # v5.1 with full heuristics
    NN_V5_HEAVY_LARGE = "nn_v5_heavy_large"  # v5-heavy-large (scaled up)
    NN_V5_HEAVY_XL = "nn_v5_heavy_xl"        # v5-heavy-xl (maximum capacity)
    NN_V6 = "nn_v6"              # Deprecated alias for v5-heavy-large

    # NNUE (scalar value output, 2p zero-sum)
    NNUE_V1 = "nnue_v1"          # NNUE v1

    # Multi-player NNUE (per-player value output)
    NNUE_MP_V1 = "nnue_mp_v1"    # Multi-player NNUE v1

    @classmethod
    def from_model_path(cls, model_path: str) -> "ModelType":
        """Infer model type from model path or filename.

        Args:
            model_path: Path to model file or model identifier

        Returns:
            ModelType enum value

        Examples:
            >>> ModelType.from_model_path("models/canonical_hex8_2p.pth")
            ModelType.NN_V5  # Default assumption
            >>> ModelType.from_model_path("models/nnue_sq8_2p.pth")
            ModelType.NNUE_V1
            >>> ModelType.from_model_path("models/canonical_hex8_2p_v5heavy.pth")
            ModelType.NN_V5_HEAVY
        """
        path_lower = model_path.lower()

        # Check for NNUE variants
        if "nnue_mp" in path_lower:
            return cls.NNUE_MP_V1
        if "nnue" in path_lower:
            return cls.NNUE_V1

        # Check for specific NN versions
        if "v5_heavy_large" in path_lower or "v5-heavy-large" in path_lower:
            return cls.NN_V5_HEAVY_LARGE
        if "v5_heavy_xl" in path_lower or "v5-heavy-xl" in path_lower:
            return cls.NN_V5_HEAVY_XL
        if "v5heavy" in path_lower or "v5_heavy" in path_lower:
            return cls.NN_V5_HEAVY
        if "v6" in path_lower:
            return cls.NN_V6  # Deprecated, maps to v5-heavy-large
        if "v5" in path_lower:
            return cls.NN_V5
        if "v4" in path_lower:
            return cls.NN_V4
        if "v3.1" in path_lower or "v3_1" in path_lower:
            return cls.NN_V3_1
        if "v3" in path_lower:
            return cls.NN_V3
        if "v2" in path_lower:
            return cls.NN_V2

        # Default to v5 for canonical models without version suffix
        return cls.NN_V5

    @property
    def is_full_nn(self) -> bool:
        """Check if this is a full neural network (outputs policy + value)."""
        return self.value.startswith("nn_")

    @property
    def is_nnue(self) -> bool:
        """Check if this is any NNUE variant."""
        return "nnue" in self.value

    @property
    def is_multiplayer_nnue(self) -> bool:
        """Check if this is a multi-player NNUE."""
        return self.value.startswith("nnue_mp")

    @property
    def supported_algorithms(self) -> list[str]:
        """Get list of algorithms this model type supports.

        Returns:
            List of algorithm type strings compatible with this model.
        """
        if self.is_full_nn:
            return ["policy_only", "mcts", "gumbel_mcts", "descent"]
        elif self.is_multiplayer_nnue:
            return ["maxn", "brs", "minimax"]  # minimax in 2p paranoid mode
        elif self.is_nnue:
            return ["minimax"]
        return []


# Algorithm types that support neural network evaluation
NN_COMPATIBLE_ALGORITHMS = frozenset({
    "policy_only",
    "mcts",
    "gumbel_mcts",
    "descent",
})

# Algorithm types that support NNUE evaluation
NNUE_COMPATIBLE_ALGORITHMS = frozenset({
    "minimax",
})

# Algorithm types that support multi-player NNUE evaluation
NNUE_MP_COMPATIBLE_ALGORITHMS = frozenset({
    "maxn",
    "brs",
    "minimax",  # Can use multi-player NNUE in paranoid mode
})


@dataclass
class CompositeParticipant:
    """Parsed composite participant information."""
    participant_id: str
    nn_id: str | None           # None for non-NN participants
    ai_type: str                # Algorithm type
    config: dict[str, Any]      # Algorithm configuration
    config_hash: str            # Short hash encoding
    category: ParticipantCategory
    is_composite: bool = True


@dataclass
class ParticipantMetadata:
    """Extended metadata for composite participants."""
    nn_model_id: str | None = None
    nn_model_path: str | None = None
    ai_algorithm: str | None = None
    algorithm_config: dict[str, Any] = field(default_factory=dict)
    is_composite: bool = False
    parent_participant_id: str | None = None


def encode_config_hash(config: dict[str, Any], ai_type: str) -> str:
    """Encode algorithm configuration into a short hash string.

    Uses a deterministic encoding that captures the most important
    configuration parameters for each algorithm type.

    Args:
        config: Algorithm configuration dictionary
        ai_type: Algorithm type (to determine which params matter)

    Returns:
        Short config hash string (e.g., "b200", "s800_c1.5")
    """
    if not config:
        return "std"  # Standard config

    # Get standard config for comparison
    standard = STANDARD_ALGORITHM_CONFIGS.get(ai_type, {})

    # Build hash parts from non-default values
    parts = []

    # Priority order of config keys per algorithm type
    priority_keys = {
        "gumbel_mcts": ["budget", "m"],
        "mcts": ["simulations", "c_puct"],
        "descent": ["difficulty", "time_ms"],
        "policy_only": ["temperature"],
        "ebmo": ["direct_eval", "optim_steps", "num_restarts"],
        "gmo": ["optim_steps", "learning_rate"],
        "gmo_gumbel": ["budget", "m"],
        "random": ["difficulty"],
        "heuristic": ["difficulty"],
    }

    keys_to_check = priority_keys.get(ai_type, list(config.keys()))

    for key in keys_to_check:
        if key not in config:
            continue

        value = config[key]
        standard_value = standard.get(key)

        # Skip if same as standard (for brevity)
        if value == standard_value:
            continue

        prefix = CONFIG_HASH_KEYS.get(key)
        if prefix:
            # Format the value
            if isinstance(value, bool):
                parts.append(f"{prefix}{1 if value else 0}")
            elif isinstance(value, float):
                # Format floats concisely
                if value == int(value):
                    parts.append(f"{prefix}{int(value)}")
                else:
                    parts.append(f"{prefix}{value:.2g}".replace(".", "p"))
            else:
                parts.append(f"{prefix}{value}")

    if not parts:
        # All values are standard - use short identifier for primary param
        primary_key = keys_to_check[0] if keys_to_check else None
        if primary_key and primary_key in config:
            value = config[primary_key]
            prefix = CONFIG_HASH_KEYS.get(primary_key, "x")
            if isinstance(value, float) and value != int(value):
                return f"{prefix}{value:.2g}".replace(".", "p")
            return f"{prefix}{value}"
        return "std"

    return "_".join(parts)


def decode_config_hash(config_hash: str, ai_type: str) -> dict[str, Any]:
    """Decode a config hash string back into configuration dictionary.

    Args:
        config_hash: Short config hash (e.g., "b200", "s800_c1.5")
        ai_type: Algorithm type (for defaults)

    Returns:
        Configuration dictionary
    """
    # Start with standard config as base
    config = dict(STANDARD_ALGORITHM_CONFIGS.get(ai_type, {}))

    if config_hash == "std":
        return config

    # Parse parts
    parts = config_hash.split("_")

    for part in parts:
        # Match prefix and value
        match = re.match(r"([a-z]+)(.+)", part)
        if not match:
            continue

        prefix, value_str = match.groups()

        # Find the config key
        config_key = HASH_KEY_TO_CONFIG.get(prefix)
        if not config_key:
            continue

        # Parse the value
        value_str = value_str.replace("p", ".")  # Restore decimal points

        try:
            # Try int first
            if value_str.isdigit() or (value_str[0] == '-' and value_str[1:].isdigit()):
                value: int | float | bool = int(value_str)
            else:
                value = float(value_str)

            # Handle boolean (de0/de1)
            if config_key in ("direct_eval",) and value in (0, 1):
                value = bool(value)

            config[config_key] = value
        except (ValueError, IndexError):
            continue

    return config


def make_composite_participant_id(
    nn_id: str | None,
    ai_type: str,
    config: dict[str, Any] | None = None,
) -> str:
    """Create a composite participant ID from components.

    Args:
        nn_id: Neural network identifier, or None/"none" for non-NN participants
        ai_type: Search algorithm type
        config: Algorithm configuration (uses defaults if None)

    Returns:
        Composite participant ID string

    Example:
        >>> make_composite_participant_id("ringrift_v5", "gumbel_mcts", {"budget": 200})
        "ringrift_v5:gumbel_mcts:b200"
    """
    # Normalize nn_id: canonical_* → ringrift_best_* for stable Elo tracking
    normalized = normalize_nn_id(nn_id)
    nn_part = normalized if normalized and normalized.lower() != "none" else "none"

    # Get config, defaulting to standard
    actual_config = config or STANDARD_ALGORITHM_CONFIGS.get(ai_type, {})

    # Encode config hash
    config_hash = encode_config_hash(actual_config, ai_type)

    return f"{nn_part}:{ai_type}:{config_hash}"


def parse_composite_participant_id(
    participant_id: str,
) -> tuple[str | None, str, dict[str, Any]]:
    """Parse a composite participant ID into its components.

    Args:
        participant_id: Composite participant ID string

    Returns:
        Tuple of (nn_id, ai_type, config)
        nn_id is None for non-NN participants ("none")

    Raises:
        ValueError: If participant_id is not a valid composite format

    Example:
        >>> parse_composite_participant_id("ringrift_v5:gumbel_mcts:b200")
        ("ringrift_v5", "gumbel_mcts", {"budget": 200, ...})
    """
    parts = participant_id.split(":")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid composite participant ID: {participant_id}. "
            f"Expected format: {{nn_id}}:{{ai_type}}:{{config_hash}}"
        )

    nn_part, ai_type, config_hash = parts

    # Parse nn_id
    nn_id = None if nn_part.lower() == "none" else nn_part

    # Decode config
    config = decode_config_hash(config_hash, ai_type)

    return nn_id, ai_type, config


def is_composite_id(participant_id: str) -> bool:
    """Check if a participant ID is in composite format.

    Args:
        participant_id: Participant ID to check

    Returns:
        True if the ID follows the composite format (contains exactly 2 colons)
    """
    return participant_id.count(":") == 2


def extract_harness_type(participant_id: str) -> str | None:
    """Extract the harness/algorithm type from a participant ID.

    Works with both composite IDs and legacy IDs.

    Args:
        participant_id: Participant ID (composite or legacy)

    Returns:
        Harness type (e.g., "gumbel_mcts", "brs", "maxn") if composite ID,
        None for legacy IDs or invalid formats.

    Examples:
        >>> extract_harness_type("ringrift_v5:gumbel_mcts:b200")
        "gumbel_mcts"
        >>> extract_harness_type("canonical_hex8_2p")
        None
        >>> extract_harness_type("nn:brs:abc123")
        "brs"

    January 2026: Added to support harness_type propagation to match_history.
    Enables per-harness Elo tracking in the training pipeline.
    """
    if not is_composite_id(participant_id):
        return None

    try:
        _, ai_type, _ = parse_composite_participant_id(participant_id)
        return ai_type
    except ValueError:
        return None


def get_standard_config(ai_type: str) -> dict[str, Any]:
    """Get the standard configuration for an algorithm type.

    Args:
        ai_type: Algorithm type

    Returns:
        Standard configuration dictionary (copy)
    """
    return dict(STANDARD_ALGORITHM_CONFIGS.get(ai_type, {}))


def get_participant_category(participant_id: str) -> ParticipantCategory:
    """Determine the category of a participant ID.

    Args:
        participant_id: Participant ID (composite or legacy)

    Returns:
        ParticipantCategory enum value
    """
    if not is_composite_id(participant_id):
        return ParticipantCategory.LEGACY

    try:
        nn_id, ai_type, _ = parse_composite_participant_id(participant_id)
    except ValueError:
        return ParticipantCategory.LEGACY

    has_nn = nn_id is not None

    # Baseline: no NN and basic algorithm
    if not has_nn and ai_type in ("random", "heuristic"):
        return ParticipantCategory.BASELINE

    # Check for NNUE variants based on model ID prefix
    if has_nn:
        nn_id_lower = nn_id.lower() if nn_id else ""
        if nn_id_lower.startswith("nnue_mp"):
            return ParticipantCategory.NNUE_MP
        if nn_id_lower.startswith("nnue"):
            return ParticipantCategory.NNUE

    # Pure NN: NN with policy_only (no search)
    if has_nn and ai_type == "policy_only":
        return ParticipantCategory.PURE_NN

    # Search-only: search algorithm without NN
    if not has_nn and ai_type not in ("random", "heuristic"):
        return ParticipantCategory.SEARCH_ONLY

    # NN + Search combination
    return ParticipantCategory.NN_SEARCH


def parse_participant(participant_id: str) -> CompositeParticipant:
    """Parse a participant ID into a CompositeParticipant object.

    Handles both composite and legacy formats.

    Args:
        participant_id: Participant ID string

    Returns:
        CompositeParticipant with parsed information
    """
    if not is_composite_id(participant_id):
        # Legacy format - treat as MCTS with default config
        return CompositeParticipant(
            participant_id=participant_id,
            nn_id=participant_id,  # Assume the ID is the NN ID
            ai_type="mcts",
            config=get_standard_config("mcts"),
            config_hash="s800",
            category=ParticipantCategory.LEGACY,
            is_composite=False,
        )

    nn_id, ai_type, config = parse_composite_participant_id(participant_id)
    config_hash = encode_config_hash(config, ai_type)
    category = get_participant_category(participant_id)

    return CompositeParticipant(
        participant_id=participant_id,
        nn_id=nn_id,
        ai_type=ai_type,
        config=config,
        config_hash=config_hash,
        category=category,
        is_composite=True,
    )


def migrate_legacy_participant_id(
    legacy_id: str,
    default_ai_type: str = "mcts",
    default_config: dict[str, Any] | None = None,
) -> str:
    """Convert a legacy participant ID to composite format.

    Args:
        legacy_id: Old-format participant ID
        default_ai_type: Algorithm to assume for migration
        default_config: Config to use (defaults to standard)

    Returns:
        Composite participant ID
    """
    if is_composite_id(legacy_id):
        return legacy_id  # Already composite

    return make_composite_participant_id(
        nn_id=legacy_id,
        ai_type=default_ai_type,
        config=default_config,
    )


def get_nn_variants(
    nn_id: str,
    algorithms: list[str] | None = None,
) -> list[str]:
    """Generate composite participant IDs for an NN across algorithms.

    Args:
        nn_id: Neural network identifier
        algorithms: List of algorithm types (defaults to common algorithms)

    Returns:
        List of composite participant IDs
    """
    if algorithms is None:
        algorithms = ["policy_only", "mcts", "gumbel_mcts", "descent"]

    return [
        make_composite_participant_id(nn_id, algo)
        for algo in algorithms
    ]


def get_algorithm_variants(
    ai_type: str,
    nn_ids: list[str],
) -> list[str]:
    """Generate composite participant IDs for an algorithm across NNs.

    Args:
        ai_type: Algorithm type
        nn_ids: List of neural network identifiers

    Returns:
        List of composite participant IDs
    """
    return [
        make_composite_participant_id(nn_id, ai_type)
        for nn_id in nn_ids
    ]


_VERSION_SUFFIX_PATTERN = re.compile(
    r'(_v\d+(?:_heavy(?:_large|_xl)?)?(?:_averaged)?)$'
)


def strip_version_suffix(nn_id: str) -> str:
    """Strip architecture version suffix from nn_id.

    Examples:
        canonical_hexagonal_2p_v2 -> canonical_hexagonal_2p
        ringrift_best_hex8_2p_v5_heavy -> ringrift_best_hex8_2p
        ringrift_best_square8_2p_v5_heavy_large -> ringrift_best_square8_2p
        ringrift_best_hex8_2p_v5_heavy_averaged -> ringrift_best_hex8_2p
        ringrift_best_hex8_2p -> ringrift_best_hex8_2p  (no change)
    """
    return _VERSION_SUFFIX_PATTERN.sub('', nn_id)


def normalize_nn_id(nn_id: str | None, *, strip_version: bool = False) -> str | None:
    """Normalize model stem to canonical participant ID format.

    Converts 'canonical_square8_2p' to 'ringrift_best_square8_2p' so that
    Elo ratings accumulate under a single stable participant ID across model
    promotions (the ringrift_best_*.pth symlink always points to the current
    canonical model).

    Feb 2026: Without this, evaluation_daemon creates composite IDs like
    'canonical_square8_2p:gumbel_mcts:d2' which fragment Elo tracking -
    each model snapshot gets a different participant_id.

    Args:
        nn_id: Neural network identifier (model stem)
        strip_version: If True, also strip architecture version suffixes
            (e.g., _v2, _v5_heavy) for grouping by base model identity.

    Returns:
        Normalized nn_id with 'canonical_' replaced by 'ringrift_best_'
    """
    if not nn_id:
        return nn_id
    if nn_id.startswith("canonical_"):
        nn_id = "ringrift_best_" + nn_id[len("canonical_"):]
    if strip_version:
        nn_id = strip_version_suffix(nn_id)
    return nn_id


def extract_nn_id(participant_id: str) -> str | None:
    """Extract the neural network ID from any participant ID format.

    Args:
        participant_id: Participant ID (composite or legacy)

    Returns:
        Neural network ID, or None if no NN
    """
    if not is_composite_id(participant_id):
        return participant_id  # Legacy ID is assumed to be NN ID

    try:
        nn_id, _, _ = parse_composite_participant_id(participant_id)
        return nn_id
    except ValueError:
        return participant_id


def extract_ai_type(participant_id: str) -> str:
    """Extract the AI algorithm type from any participant ID format.

    Args:
        participant_id: Participant ID (composite or legacy)

    Returns:
        AI algorithm type (defaults to "mcts" for legacy IDs)
    """
    if not is_composite_id(participant_id):
        return "mcts"  # Legacy default

    try:
        _, ai_type, _ = parse_composite_participant_id(participant_id)
        return ai_type
    except ValueError:
        return "mcts"


# Baseline participant IDs
# NOTE: Only Random is PINNED at 400 Elo (see elo_service.py:815-822).
# Other baselines have dynamic Elo that changes based on game results.
# The values below are EXPECTED approximate Elo, not enforced ratings.
BASELINE_PARTICIPANTS = {
    "none:random:d1": 400.0,         # PINNED at 400 (anchor point)
    "none:heuristic:d2": 1000.0,     # Expected ~1000 (dynamic)
    "none:heuristic:d3": 1200.0,     # Expected ~1200 (dynamic)
    "none:heuristic:d4": 1400.0,     # Expected ~1400 (dynamic)
}


def get_baseline_participant_ids() -> list[str]:
    """Get list of standard baseline participant IDs."""
    return list(BASELINE_PARTICIPANTS.keys())


def get_baseline_rating(participant_id: str) -> float | None:
    """Get the expected rating for a baseline participant.

    Returns None if not a known baseline.

    NOTE: Only Random (none:random:*) is actually pinned at 400 Elo.
    Other baselines have dynamic Elo; these are just expected values.
    """
    return BASELINE_PARTICIPANTS.get(participant_id)


def is_baseline_participant(participant_id: str) -> bool:
    """Check if a participant ID is a known baseline."""
    return participant_id in BASELINE_PARTICIPANTS


def extract_model_type(participant_id: str) -> ModelType | None:
    """Extract the model type from a participant ID.

    Args:
        participant_id: Participant ID (composite or legacy)

    Returns:
        ModelType enum, or None if model type cannot be determined
    """
    nn_id = extract_nn_id(participant_id)
    if nn_id is None:
        return None
    return ModelType.from_model_path(nn_id)


def get_compatible_algorithms(model_type: ModelType) -> list[str]:
    """Get list of algorithms compatible with a model type.

    Args:
        model_type: ModelType enum value

    Returns:
        List of algorithm type strings
    """
    return model_type.supported_algorithms


def make_nnue_participant_id(
    nnue_version: str,
    board_config: str,
    ai_type: str = "minimax",
    config: dict[str, Any] | None = None,
) -> str:
    """Create a composite participant ID for an NNUE model.

    Args:
        nnue_version: NNUE version (e.g., "v1")
        board_config: Board configuration (e.g., "sq8_2p")
        ai_type: Algorithm type (defaults to "minimax")
        config: Algorithm configuration

    Returns:
        Composite participant ID

    Example:
        >>> make_nnue_participant_id("v1", "sq8_2p")
        "nnue_v1_sq8_2p:minimax:dp6"
    """
    nn_id = f"nnue_{nnue_version}_{board_config}"
    return make_composite_participant_id(nn_id, ai_type, config)


def make_nnue_mp_participant_id(
    nnue_version: str,
    board_config: str,
    ai_type: str = "maxn",
    config: dict[str, Any] | None = None,
) -> str:
    """Create a composite participant ID for a multi-player NNUE model.

    Args:
        nnue_version: NNUE version (e.g., "v1")
        board_config: Board configuration (e.g., "hex8_4p")
        ai_type: Algorithm type (defaults to "maxn")
        config: Algorithm configuration

    Returns:
        Composite participant ID

    Example:
        >>> make_nnue_mp_participant_id("v1", "hex8_4p")
        "nnue_mp_v1_hex8_4p:maxn:dp7"
    """
    nn_id = f"nnue_mp_{nnue_version}_{board_config}"
    return make_composite_participant_id(nn_id, ai_type, config)


def get_all_harness_variants(
    model_id: str,
    model_type: ModelType | None = None,
) -> list[str]:
    """Generate composite participant IDs for a model across all compatible harnesses.

    Args:
        model_id: Model identifier (e.g., "ringrift_v5_sq8_2p")
        model_type: Optional ModelType (auto-detected if not provided)

    Returns:
        List of composite participant IDs for all compatible harnesses

    Example:
        >>> get_all_harness_variants("ringrift_v5_sq8_2p")
        [
            "ringrift_v5_sq8_2p:policy_only:t0p3",
            "ringrift_v5_sq8_2p:mcts:s800",
            "ringrift_v5_sq8_2p:gumbel_mcts:b200",
            "ringrift_v5_sq8_2p:descent:d6",
        ]
    """
    if model_type is None:
        model_type = ModelType.from_model_path(model_id)

    algorithms = model_type.supported_algorithms
    return [
        make_composite_participant_id(model_id, algo)
        for algo in algorithms
    ]


def is_nnue_participant(participant_id: str) -> bool:
    """Check if a participant ID represents an NNUE model.

    Args:
        participant_id: Participant ID to check

    Returns:
        True if the participant uses an NNUE model
    """
    nn_id = extract_nn_id(participant_id)
    if nn_id is None:
        return False
    return "nnue" in nn_id.lower()


def is_multiplayer_nnue_participant(participant_id: str) -> bool:
    """Check if a participant ID represents a multi-player NNUE model.

    Args:
        participant_id: Participant ID to check

    Returns:
        True if the participant uses a multi-player NNUE model
    """
    nn_id = extract_nn_id(participant_id)
    if nn_id is None:
        return False
    return nn_id.lower().startswith("nnue_mp")


def validate_algorithm_compatibility(
    model_id: str,
    ai_type: str,
) -> bool:
    """Check if an algorithm is compatible with a model type.

    Args:
        model_id: Model identifier
        ai_type: Algorithm type

    Returns:
        True if the algorithm is compatible with the model

    Raises:
        ValueError: If the algorithm is not compatible
    """
    model_type = ModelType.from_model_path(model_id)
    compatible = model_type.supported_algorithms

    if ai_type not in compatible:
        raise ValueError(
            f"Algorithm '{ai_type}' is not compatible with model type "
            f"'{model_type.value}'. Compatible algorithms: {compatible}"
        )
    return True
