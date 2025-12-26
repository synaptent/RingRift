"""Canonical event name normalization for RingRift event system.

This module provides canonical event naming rules and normalization utilities
to standardize inconsistent event names across the codebase.

Problem:
    The codebase has inconsistent event naming conventions:
    - SYNC_COMPLETE vs DATA_SYNC_COMPLETED
    - SELFPLAY_COMPLETE vs SELFPLAY_BATCH_COMPLETE
    - MODEL_SYNC_COMPLETE vs P2P_MODEL_SYNCED
    - CLUSTER_SYNC_COMPLETE vs SYNC_COMPLETE

Solution:
    This module defines canonical names and normalizes variants on publish.
    All event types are normalized to their canonical form before routing.

Usage:
    from app.coordination.event_normalization import normalize_event_type

    # Normalize any variant to canonical form
    canonical = normalize_event_type("SYNC_COMPLETE")  # → "DATA_SYNC_COMPLETED"
    canonical = normalize_event_type("sync_complete")  # → "DATA_SYNC_COMPLETED"

Created: December 2025
Purpose: Standardize event naming (Phase 14 event system hardening)
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# =============================================================================
# Canonical Event Naming Convention
# =============================================================================

"""
RingRift Canonical Event Naming Convention:

1. Format: {SUBJECT}_{ACTION}_{MODIFIER}
   - SUBJECT: What the event is about (DATA, MODEL, TRAINING, etc.)
   - ACTION: What happened (SYNC, COMPLETED, STARTED, etc.)
   - MODIFIER: Optional qualifier (BATCH, GPU, etc.)

2. Tense Rules:
   - Completion events: Use past tense "_COMPLETED"
   - Start events: Use past tense "_STARTED"
   - State events: Use past participle "_SYNCED"
   - Failure events: Use past tense "_FAILED"

3. Specificity:
   - Be specific about the subject (DATA_SYNC not just SYNC)
   - Avoid ambiguous names (COMPLETE → TRAINING_COMPLETED)

4. Examples:
   - DATA_SYNC_COMPLETED (not SYNC_COMPLETE)
   - TRAINING_STARTED (not TRAINING_START)
   - MODEL_PROMOTED (not PROMOTION_COMPLETE)
   - SELFPLAY_COMPLETE (specific: selfplay batch finished)

5. Cross-Process Convention:
   - All cross-process events use UPPERCASE_SNAKE_CASE
   - Match canonical forms from data_events.py DataEventType enum
"""

# =============================================================================
# Canonical Event Name Mappings
# =============================================================================

# Map all known event name variants to their canonical form
# This enables automatic normalization of legacy/inconsistent names
CANONICAL_EVENT_NAMES: Final[dict[str, str]] = {
    # =============================================================================
    # Data Sync Events
    # =============================================================================
    # Canonical: DATA_SYNC_COMPLETED
    "sync_complete": "DATA_SYNC_COMPLETED",
    "SYNC_COMPLETE": "DATA_SYNC_COMPLETED",
    "sync_completed": "DATA_SYNC_COMPLETED",
    "SYNC_COMPLETED": "DATA_SYNC_COMPLETED",
    "data_sync_complete": "DATA_SYNC_COMPLETED",
    "DATA_SYNC_COMPLETE": "DATA_SYNC_COMPLETED",
    "data_sync_completed": "DATA_SYNC_COMPLETED",
    "DATA_SYNC_COMPLETED": "DATA_SYNC_COMPLETED",  # Already canonical

    # Cluster sync variants
    "cluster_sync_complete": "DATA_SYNC_COMPLETED",
    "CLUSTER_SYNC_COMPLETE": "DATA_SYNC_COMPLETED",
    "cluster_sync_completed": "DATA_SYNC_COMPLETED",
    "CLUSTER_SYNC_COMPLETED": "DATA_SYNC_COMPLETED",

    # Data sync start events
    "sync_started": "DATA_SYNC_STARTED",
    "SYNC_STARTED": "DATA_SYNC_STARTED",
    "data_sync_started": "DATA_SYNC_STARTED",
    "DATA_SYNC_STARTED": "DATA_SYNC_STARTED",  # Canonical

    # Data sync failure events
    "sync_failed": "DATA_SYNC_FAILED",
    "SYNC_FAILED": "DATA_SYNC_FAILED",
    "data_sync_failed": "DATA_SYNC_FAILED",
    "DATA_SYNC_FAILED": "DATA_SYNC_FAILED",  # Canonical

    # =============================================================================
    # Model Sync Events
    # =============================================================================
    # Canonical: P2P_MODEL_SYNCED
    "model_sync_complete": "P2P_MODEL_SYNCED",
    "MODEL_SYNC_COMPLETE": "P2P_MODEL_SYNCED",
    "model_sync_completed": "P2P_MODEL_SYNCED",
    "MODEL_SYNC_COMPLETED": "P2P_MODEL_SYNCED",
    "p2p_model_synced": "P2P_MODEL_SYNCED",
    "P2P_MODEL_SYNCED": "P2P_MODEL_SYNCED",  # Canonical
    "model_synced": "P2P_MODEL_SYNCED",
    "MODEL_SYNCED": "P2P_MODEL_SYNCED",

    # Model distribution events
    "model_distribution_complete": "MODEL_DISTRIBUTION_COMPLETE",
    "MODEL_DISTRIBUTION_COMPLETE": "MODEL_DISTRIBUTION_COMPLETE",  # Canonical

    # =============================================================================
    # Selfplay Events
    # =============================================================================
    # Canonical: SELFPLAY_COMPLETE (single batch completion)
    "selfplay_complete": "SELFPLAY_COMPLETE",
    "SELFPLAY_COMPLETE": "SELFPLAY_COMPLETE",  # Canonical
    "selfplay_completed": "SELFPLAY_COMPLETE",
    "SELFPLAY_COMPLETED": "SELFPLAY_COMPLETE",
    "selfplay_batch_complete": "SELFPLAY_COMPLETE",
    "SELFPLAY_BATCH_COMPLETE": "SELFPLAY_COMPLETE",

    # Canonical selfplay (high-quality TS engine)
    "canonical_selfplay_complete": "SELFPLAY_COMPLETE",
    "CANONICAL_SELFPLAY_COMPLETE": "SELFPLAY_COMPLETE",

    # GPU selfplay
    "gpu_selfplay_complete": "SELFPLAY_COMPLETE",
    "GPU_SELFPLAY_COMPLETE": "SELFPLAY_COMPLETE",

    # =============================================================================
    # Training Events
    # =============================================================================
    # Canonical: TRAINING_COMPLETED
    "training_complete": "TRAINING_COMPLETED",
    "TRAINING_COMPLETE": "TRAINING_COMPLETED",
    "training_completed": "TRAINING_COMPLETED",
    "TRAINING_COMPLETED": "TRAINING_COMPLETED",  # Canonical

    # Training start
    "training_start": "TRAINING_STARTED",
    "TRAINING_START": "TRAINING_STARTED",
    "training_started": "TRAINING_STARTED",
    "TRAINING_STARTED": "TRAINING_STARTED",  # Canonical

    # Training failure
    "training_fail": "TRAINING_FAILED",
    "TRAINING_FAIL": "TRAINING_FAILED",
    "training_failed": "TRAINING_FAILED",
    "TRAINING_FAILED": "TRAINING_FAILED",  # Canonical

    # =============================================================================
    # Evaluation Events
    # =============================================================================
    # Canonical: EVALUATION_COMPLETED
    "evaluation_complete": "EVALUATION_COMPLETED",
    "EVALUATION_COMPLETE": "EVALUATION_COMPLETED",
    "evaluation_completed": "EVALUATION_COMPLETED",
    "EVALUATION_COMPLETED": "EVALUATION_COMPLETED",  # Canonical

    # Shadow tournament (maps to evaluation)
    "shadow_tournament_complete": "EVALUATION_COMPLETED",
    "SHADOW_TOURNAMENT_COMPLETE": "EVALUATION_COMPLETED",

    # Evaluation start
    "evaluation_start": "EVALUATION_STARTED",
    "EVALUATION_START": "EVALUATION_STARTED",
    "evaluation_started": "EVALUATION_STARTED",
    "EVALUATION_STARTED": "EVALUATION_STARTED",  # Canonical

    # Evaluation failure
    "evaluation_fail": "EVALUATION_FAILED",
    "EVALUATION_FAIL": "EVALUATION_FAILED",
    "evaluation_failed": "EVALUATION_FAILED",
    "EVALUATION_FAILED": "EVALUATION_FAILED",  # Canonical

    # =============================================================================
    # Promotion Events
    # =============================================================================
    # Canonical: MODEL_PROMOTED
    "promotion_complete": "MODEL_PROMOTED",
    "PROMOTION_COMPLETE": "MODEL_PROMOTED",
    "promotion_completed": "MODEL_PROMOTED",
    "PROMOTION_COMPLETED": "MODEL_PROMOTED",
    "model_promoted": "MODEL_PROMOTED",
    "MODEL_PROMOTED": "MODEL_PROMOTED",  # Canonical

    # Tier gating (maps to promotion)
    "tier_gating_complete": "MODEL_PROMOTED",
    "TIER_GATING_COMPLETE": "MODEL_PROMOTED",

    # Promotion start
    "promotion_start": "PROMOTION_STARTED",
    "PROMOTION_START": "PROMOTION_STARTED",
    "promotion_started": "PROMOTION_STARTED",
    "PROMOTION_STARTED": "PROMOTION_STARTED",  # Canonical

    # Promotion failure
    "promotion_fail": "PROMOTION_FAILED",
    "PROMOTION_FAIL": "PROMOTION_FAILED",
    "promotion_failed": "PROMOTION_FAILED",
    "PROMOTION_FAILED": "PROMOTION_FAILED",  # Canonical

    # =============================================================================
    # NPZ Export Events
    # =============================================================================
    # Canonical: NPZ_EXPORT_COMPLETE
    "npz_export_complete": "NPZ_EXPORT_COMPLETE",
    "NPZ_EXPORT_COMPLETE": "NPZ_EXPORT_COMPLETE",  # Canonical
    "export_complete": "NPZ_EXPORT_COMPLETE",
    "EXPORT_COMPLETE": "NPZ_EXPORT_COMPLETE",

    # =============================================================================
    # Parity Validation Events
    # =============================================================================
    # Canonical: PARITY_VALIDATION_COMPLETED
    "parity_validation_complete": "PARITY_VALIDATION_COMPLETED",
    "PARITY_VALIDATION_COMPLETE": "PARITY_VALIDATION_COMPLETED",
    "parity_validation_completed": "PARITY_VALIDATION_COMPLETED",
    "PARITY_VALIDATION_COMPLETED": "PARITY_VALIDATION_COMPLETED",  # Canonical

    # =============================================================================
    # Optimization Events
    # =============================================================================
    # CMA-ES
    "cmaes_complete": "CMAES_COMPLETED",
    "CMAES_COMPLETE": "CMAES_COMPLETED",
    "cmaes_completed": "CMAES_COMPLETED",
    "CMAES_COMPLETED": "CMAES_COMPLETED",  # Canonical

    # PBT
    "pbt_complete": "PBT_GENERATION_COMPLETE",
    "PBT_COMPLETE": "PBT_GENERATION_COMPLETE",
    "pbt_completed": "PBT_GENERATION_COMPLETE",
    "PBT_COMPLETED": "PBT_GENERATION_COMPLETE",
    "pbt_generation_complete": "PBT_GENERATION_COMPLETE",
    "PBT_GENERATION_COMPLETE": "PBT_GENERATION_COMPLETE",  # Canonical

    # NAS
    "nas_complete": "NAS_COMPLETED",
    "NAS_COMPLETE": "NAS_COMPLETED",
    "nas_completed": "NAS_COMPLETED",
    "NAS_COMPLETED": "NAS_COMPLETED",  # Canonical

    # =============================================================================
    # Iteration Events
    # =============================================================================
    # Canonical: ITERATION_COMPLETE
    "iteration_complete": "ITERATION_COMPLETE",
    "ITERATION_COMPLETE": "ITERATION_COMPLETE",  # Canonical
}

# =============================================================================
# Event Naming Guidelines Documentation
# =============================================================================

EVENT_NAMING_GUIDELINES = """
RingRift Event Naming Guidelines (December 2025)

1. CANONICAL FORMAT: {SUBJECT}_{ACTION}_{MODIFIER}

   Examples:
   - DATA_SYNC_COMPLETED     # Subject: DATA, Action: SYNC, State: COMPLETED
   - TRAINING_STARTED        # Subject: TRAINING, State: STARTED
   - MODEL_PROMOTED          # Subject: MODEL, State: PROMOTED
   - SELFPLAY_COMPLETE       # Subject: SELFPLAY, State: COMPLETE

2. TENSE RULES:
   - Completion: Use past tense with "_COMPLETED" (TRAINING_COMPLETED)
   - Start: Use past tense with "_STARTED" (EVALUATION_STARTED)
   - State Change: Use past participle (MODEL_PROMOTED, P2P_MODEL_SYNCED)
   - Failure: Use past tense with "_FAILED" (PROMOTION_FAILED)

3. SPECIFICITY RULES:
   - Always include subject (DATA_SYNC not just SYNC)
   - Be explicit about scope (SELFPLAY_COMPLETE for batch, not just COMPLETE)
   - Distinguish between stages (TRAINING_STARTED vs TRAINING_COMPLETED)

4. CONSISTENCY RULES:
   - All cross-process events: UPPERCASE_SNAKE_CASE
   - All data events match DataEventType enum values
   - Stage events use lowercase snake_case (sync_complete)
   - Router normalizes all variants to canonical UPPERCASE form

5. DEPRECATED FORMS TO AVOID:
   - SYNC_COMPLETE          → Use DATA_SYNC_COMPLETED
   - TRAINING_COMPLETE      → Use TRAINING_COMPLETED
   - PROMOTION_COMPLETE     → Use MODEL_PROMOTED
   - CLUSTER_SYNC_COMPLETE  → Use DATA_SYNC_COMPLETED

6. MIGRATION PATH:
   - Old code can continue using legacy names
   - Router automatically normalizes to canonical form
   - Subscribers receive canonicalized events
   - Update code gradually to use canonical names
"""

# =============================================================================
# Normalization Functions
# =============================================================================


def normalize_event_type(event_type: str | object) -> str:
    """Normalize an event type to its canonical form.

    This function handles:
    1. Enum types (extracts .value)
    2. String variants (maps to canonical name)
    3. Case insensitivity (SYNC_COMPLETE = sync_complete)
    4. Pass-through for already-canonical names

    Args:
        event_type: Event type (string or enum)

    Returns:
        Canonical event type name (UPPERCASE_SNAKE_CASE)

    Examples:
        >>> normalize_event_type("sync_complete")
        'DATA_SYNC_COMPLETED'
        >>> normalize_event_type("CLUSTER_SYNC_COMPLETE")
        'DATA_SYNC_COMPLETED'
        >>> normalize_event_type("DATA_SYNC_COMPLETED")
        'DATA_SYNC_COMPLETED'
    """
    # Extract value from enum types
    if hasattr(event_type, 'value'):
        event_type = event_type.value

    # Convert to string
    event_str = str(event_type)

    # Try exact match first (most common case)
    canonical = CANONICAL_EVENT_NAMES.get(event_str)
    if canonical:
        return canonical

    # Try case-insensitive match
    lower = event_str.lower()
    for variant, canonical in CANONICAL_EVENT_NAMES.items():
        if variant.lower() == lower:
            return canonical

    # No mapping found - return as-is (may be already canonical or unknown)
    # Log unmapped events for debugging
    if event_str.upper() != event_str:
        logger.debug(
            f"[EventNormalization] No mapping for '{event_str}', "
            f"passing through as-is"
        )

    return event_str


def is_canonical(event_type: str) -> bool:
    """Check if an event type is already in canonical form.

    Args:
        event_type: Event type string

    Returns:
        True if canonical, False if it has known variants
    """
    normalized = normalize_event_type(event_type)
    return event_type == normalized


def get_variants(canonical_event: str) -> list[str]:
    """Get all known variants of a canonical event type.

    Args:
        canonical_event: Canonical event type name

    Returns:
        List of variant names that map to this canonical form
    """
    variants = [
        variant
        for variant, canonical in CANONICAL_EVENT_NAMES.items()
        if canonical == canonical_event
    ]
    return variants


def validate_event_names() -> list[str]:
    """Validate event name mappings for consistency.

    Checks:
    1. All canonical names are UPPERCASE_SNAKE_CASE
    2. No circular mappings
    3. All variants map to valid canonicals

    Returns:
        List of validation warnings (empty if all valid)
    """
    warnings: list[str] = []

    # Check canonical names follow convention
    canonical_names = set(CANONICAL_EVENT_NAMES.values())
    for canonical in canonical_names:
        if canonical != canonical.upper():
            warnings.append(
                f"Canonical name '{canonical}' should be UPPERCASE_SNAKE_CASE"
            )

        # Check for proper format {SUBJECT}_{ACTION}
        if '_' not in canonical:
            warnings.append(
                f"Canonical name '{canonical}' should contain underscore "
                f"(format: SUBJECT_ACTION)"
            )

    # Check for circular mappings
    for variant, canonical in CANONICAL_EVENT_NAMES.items():
        if variant in canonical_names and variant != canonical:
            warnings.append(
                f"Circular mapping: '{variant}' maps to '{canonical}' "
                f"but '{variant}' is also a canonical name"
            )

    return warnings


# =============================================================================
# Audit Utilities
# =============================================================================


def audit_event_usage(event_history: list[str]) -> dict[str, any]:
    """Audit event name usage and identify normalization opportunities.

    Args:
        event_history: List of event type strings from history

    Returns:
        Dict with usage statistics and recommendations
    """
    # Count usage of each variant
    usage_counts: dict[str, int] = {}
    for event_type in event_history:
        usage_counts[event_type] = usage_counts.get(event_type, 0) + 1

    # Identify non-canonical usage
    non_canonical: dict[str, str] = {}
    canonical_usage: dict[str, int] = {}

    for event_type, count in usage_counts.items():
        normalized = normalize_event_type(event_type)
        if event_type != normalized:
            non_canonical[event_type] = normalized
        else:
            canonical_usage[normalized] = canonical_usage.get(normalized, 0) + count

    # Calculate normalization impact
    total_events = len(event_history)
    non_canonical_count = sum(
        usage_counts[variant] for variant in non_canonical
    )
    normalization_rate = (
        non_canonical_count / total_events if total_events > 0 else 0.0
    )

    return {
        "total_events": total_events,
        "unique_event_types": len(usage_counts),
        "canonical_types": len(canonical_usage),
        "non_canonical_variants": non_canonical,
        "non_canonical_count": non_canonical_count,
        "normalization_rate": normalization_rate,
        "usage_by_canonical": canonical_usage,
        "recommendations": _generate_recommendations(
            non_canonical, usage_counts, normalization_rate
        ),
    }


def _generate_recommendations(
    non_canonical: dict[str, str],
    usage_counts: dict[str, int],
    normalization_rate: float,
) -> list[str]:
    """Generate recommendations for event naming improvements."""
    recommendations: list[str] = []

    if normalization_rate > 0.5:
        recommendations.append(
            f"High normalization rate ({normalization_rate:.1%}). "
            f"Consider updating code to use canonical names."
        )

    # Identify most-used non-canonical variants
    high_usage_variants = [
        (variant, count)
        for variant, count in usage_counts.items()
        if variant in non_canonical and count > 10
    ]

    if high_usage_variants:
        recommendations.append(
            "High-usage non-canonical variants detected:"
        )
        for variant, count in sorted(high_usage_variants, key=lambda x: -x[1]):
            canonical = non_canonical[variant]
            recommendations.append(
                f"  - Replace '{variant}' ({count}x) with '{canonical}'"
            )

    return recommendations


__all__ = [
    "CANONICAL_EVENT_NAMES",
    "EVENT_NAMING_GUIDELINES",
    "audit_event_usage",
    "get_variants",
    "is_canonical",
    "normalize_event_type",
    "validate_event_names",
]
