"""Helpers for inspecting checkpoint architecture metadata."""

from __future__ import annotations

import logging
from pathlib import Path

from app.training.model_versioning import infer_memory_tier_from_config
from app.utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)


def detect_tier_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[str, str, int, int] | None:
    """Detect memory tier and architecture from a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = safe_load_checkpoint(str(checkpoint_path), map_location=device)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Could not load checkpoint for tier detection: %s", exc)
        return None

    metadata = checkpoint.get("_versioning_metadata", {})
    config = metadata.get("config", {})

    memory_tier = metadata.get("memory_tier") or config.get("memory_tier", "")
    num_filters = config.get("num_filters")
    num_res_blocks = config.get("num_res_blocks")

    if not memory_tier and num_filters is not None:
        memory_tier = infer_memory_tier_from_config(config)

    if not memory_tier:
        return None

    tier_to_version = {
        "v4": "v4",
        "v3-high": "v3",
        "v3-low": "v3",
        "v5": "v5",
        "v5.1": "v5-heavy",
        "v5-heavy-large": "v5-heavy",
        "v5-heavy-xl": "v5-heavy",
        "v6": "v5-heavy",
        "v6-xl": "v5-heavy",
        "v2": "v2",
        "v2-lite": "v2",
        "gnn": "gnn",
        "hybrid": "hybrid",
    }
    model_version = tier_to_version.get(memory_tier, "v2")

    if num_filters is None or num_res_blocks is None:
        tier_defaults = {
            "v4": (128, 13),
            "v3-high": (192, 12),
            "v3-low": (96, 6),
            "v5": (160, 11),
            "v5.1": (160, 11),
            "v5-heavy-large": (256, 18),
            "v5-heavy-xl": (320, 20),
            "v6": (256, 18),
            "v6-xl": (320, 20),
            "v2": (96, 6),
            "v2-lite": (64, 6),
        }
        default_filters, default_blocks = tier_defaults.get(memory_tier, (96, 6))
        num_filters = num_filters or default_filters
        num_res_blocks = num_res_blocks or default_blocks

    logger.info(
        "Detected checkpoint architecture: tier=%s, version=%s, filters=%s, blocks=%s",
        memory_tier,
        model_version,
        num_filters,
        num_res_blocks,
    )
    return (memory_tier, model_version, num_filters, num_res_blocks)


__all__ = ["detect_tier_from_checkpoint"]
