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
}

# Reverse mapping for parsing
HASH_KEY_TO_CONFIG = {v: k for k, v in CONFIG_HASH_KEYS.items()}


class ParticipantCategory(Enum):
    """Categories of participants for different tracking purposes."""
    BASELINE = "baseline"        # Fixed anchor points, no NN (e.g., none:random:d1)
    PURE_NN = "pure_nn"          # NN strength without search (e.g., nn:policy_only:t0.3)
    NN_SEARCH = "nn_search"      # Full NN + search combination
    SEARCH_ONLY = "search_only"  # Heuristic eval with search (no NN)
    LEGACY = "legacy"            # Old-format participant ID (not composite)


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
    # Normalize nn_id
    nn_part = nn_id if nn_id and nn_id.lower() != "none" else "none"

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


# Baseline participant IDs (pinned ratings)
BASELINE_PARTICIPANTS = {
    "none:random:d1": 400.0,         # Random baseline - PINNED
    "none:heuristic:d2": 1000.0,     # Heuristic baseline
    "none:heuristic:d3": 1200.0,     # Heuristic medium
    "none:heuristic:d4": 1400.0,     # Heuristic hard
}


def get_baseline_participant_ids() -> list[str]:
    """Get list of standard baseline participant IDs."""
    return list(BASELINE_PARTICIPANTS.keys())


def get_baseline_rating(participant_id: str) -> float | None:
    """Get the pinned rating for a baseline participant.

    Returns None if not a pinned baseline.
    """
    return BASELINE_PARTICIPANTS.get(participant_id)


def is_baseline_participant(participant_id: str) -> bool:
    """Check if a participant ID is a pinned baseline."""
    return participant_id in BASELINE_PARTICIPANTS
