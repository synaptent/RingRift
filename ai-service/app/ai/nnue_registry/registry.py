"""NNUE Model Registry - Canonical paths and metadata for NNUE models.

December 2025: Phase 4 NNUE Integration

This module provides centralized path management for NNUE models,
ensuring consistent naming conventions and discovery across the system.

Canonical Path Pattern:
    models/nnue/nnue_canonical_{board_type}_{num_players}p.pt

Usage:
    from app.models.nnue_registry import (
        get_nnue_canonical_path,
        get_all_nnue_paths,
        NNUEModelInfo,
    )

    # Get canonical path for a config
    path = get_nnue_canonical_path("hex8", 2)
    # -> Path("models/nnue/nnue_canonical_hex8_2p.pt")

    # Discover existing models
    for info in get_all_nnue_paths():
        print(f"{info.config_key}: {info.path}, exists={info.exists}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from app.coordination.types import BoardType
from app.utils.canonical_naming import normalize_board_type as _canonical_normalize

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Base directory for NNUE models (relative to ai-service root)
# Path: app/ai/nnue/registry.py -> ../../.. -> ai-service -> models/nnue
NNUE_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "nnue"

# Canonical naming pattern
CANONICAL_PREFIX = "nnue_canonical"

# All 12 canonical configurations
CANONICAL_CONFIGS: list[tuple[str, int]] = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NNUEModelInfo:
    """Metadata for an NNUE model."""

    config_key: str  # e.g., "hex8_2p"
    board_type: str
    num_players: int
    path: Path
    exists: bool = False
    file_size_bytes: int = 0
    modified_time: datetime | None = None
    is_canonical: bool = True

    def __post_init__(self) -> None:
        """Populate file metadata if path exists."""
        if self.path.exists():
            self.exists = True
            stat = self.path.stat()
            self.file_size_bytes = stat.st_size
            self.modified_time = datetime.fromtimestamp(stat.st_mtime)


@dataclass
class NNUERegistryStats:
    """Statistics about NNUE model availability."""

    total_configs: int = 12
    models_present: int = 0
    models_missing: int = 0
    total_size_bytes: int = 0
    oldest_model: datetime | None = None
    newest_model: datetime | None = None
    missing_configs: list[str] = field(default_factory=list)


# =============================================================================
# Path Resolution
# =============================================================================


def _normalize_board_type(board_type: str | BoardType) -> str:
    """Normalize board type to canonical string.

    Delegates to canonical_naming.normalize_board_type for consistency.
    January 2026: Centralized to avoid duplicated normalization logic.
    """
    return _canonical_normalize(board_type)


def get_nnue_models_dir() -> Path:
    """Get the NNUE models directory, creating it if needed."""
    NNUE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return NNUE_MODELS_DIR


def get_nnue_canonical_path(
    board_type: str | BoardType,
    num_players: int,
) -> Path:
    """Get the canonical NNUE model path for a configuration.

    Args:
        board_type: Board type (hex8, square8, square19, hexagonal)
        num_players: Number of players (2, 3, or 4)

    Returns:
        Path to the canonical NNUE model checkpoint
    """
    board = _normalize_board_type(board_type)
    return get_nnue_models_dir() / f"{CANONICAL_PREFIX}_{board}_{num_players}p.pt"


def get_nnue_config_key(board_type: str | BoardType, num_players: int) -> str:
    """Get the config key for a board/player combination."""
    board = _normalize_board_type(board_type)
    return f"{board}_{num_players}p"


def get_nnue_model_info(
    board_type: str | BoardType,
    num_players: int,
) -> NNUEModelInfo:
    """Get full metadata for an NNUE model.

    Args:
        board_type: Board type
        num_players: Number of players

    Returns:
        NNUEModelInfo with path and file metadata
    """
    board = _normalize_board_type(board_type)
    config_key = get_nnue_config_key(board_type, num_players)
    path = get_nnue_canonical_path(board_type, num_players)

    return NNUEModelInfo(
        config_key=config_key,
        board_type=board,
        num_players=num_players,
        path=path,
        is_canonical=True,
    )


def get_all_nnue_paths() -> Iterator[NNUEModelInfo]:
    """Iterate over all canonical NNUE model paths.

    Yields:
        NNUEModelInfo for each of the 12 canonical configurations
    """
    for board_type, num_players in CANONICAL_CONFIGS:
        yield get_nnue_model_info(board_type, num_players)


def get_existing_nnue_models() -> list[NNUEModelInfo]:
    """Get list of NNUE models that actually exist on disk.

    Returns:
        List of NNUEModelInfo for models that exist
    """
    return [info for info in get_all_nnue_paths() if info.exists]


def get_missing_nnue_models() -> list[NNUEModelInfo]:
    """Get list of NNUE models that are missing.

    Returns:
        List of NNUEModelInfo for missing models
    """
    return [info for info in get_all_nnue_paths() if not info.exists]


def get_nnue_registry_stats() -> NNUERegistryStats:
    """Get statistics about NNUE model availability.

    Returns:
        NNUERegistryStats with counts and metadata
    """
    stats = NNUERegistryStats()
    oldest: datetime | None = None
    newest: datetime | None = None

    for info in get_all_nnue_paths():
        if info.exists:
            stats.models_present += 1
            stats.total_size_bytes += info.file_size_bytes
            if info.modified_time:
                if oldest is None or info.modified_time < oldest:
                    oldest = info.modified_time
                if newest is None or info.modified_time > newest:
                    newest = info.modified_time
        else:
            stats.models_missing += 1
            stats.missing_configs.append(info.config_key)

    stats.oldest_model = oldest
    stats.newest_model = newest

    return stats


# =============================================================================
# Symlink Management (for backward compatibility)
# =============================================================================


def get_nnue_legacy_path(
    board_type: str | BoardType,
    num_players: int,
) -> Path:
    """Get the legacy NNUE model path (non-canonical naming).

    This is for backward compatibility with older training runs that used
    nnue_{board}_{n}p.pt or nnue_{board}.pt naming.
    """
    board = _normalize_board_type(board_type)
    models_dir = get_nnue_models_dir()

    # Check for newer naming first: nnue_{board}_{n}p.pt
    primary = models_dir / f"nnue_{board}_{num_players}p.pt"
    if primary.exists():
        return primary

    # Fallback for 2-player: nnue_{board}.pt
    if num_players == 2:
        legacy = models_dir / f"nnue_{board}.pt"
        if legacy.exists():
            return legacy

    return primary


def ensure_nnue_symlink(
    board_type: str | BoardType,
    num_players: int,
) -> Path | None:
    """Create a legacy symlink pointing to the canonical model.

    This helps tools that expect the legacy naming convention.

    Returns:
        Path to the created symlink, or None if canonical doesn't exist
    """
    canonical = get_nnue_canonical_path(board_type, num_players)
    if not canonical.exists():
        return None

    legacy = get_nnue_legacy_path(board_type, num_players)

    # Don't overwrite if legacy is a real file
    if legacy.exists() and not legacy.is_symlink():
        logger.debug(f"Legacy NNUE path {legacy} is a real file, not creating symlink")
        return None

    # Create or update symlink
    if legacy.is_symlink():
        legacy.unlink()

    legacy.symlink_to(canonical)
    logger.info(f"Created NNUE symlink: {legacy} -> {canonical}")
    return legacy


def ensure_all_nnue_symlinks() -> int:
    """Create legacy symlinks for all existing canonical models.

    Returns:
        Number of symlinks created
    """
    count = 0
    for info in get_all_nnue_paths():
        if info.exists:
            if ensure_nnue_symlink(info.board_type, info.num_players):
                count += 1
    return count


# =============================================================================
# Integration with NNUE Training
# =============================================================================


def get_nnue_output_path(
    board_type: str | BoardType,
    num_players: int,
    *,
    staging: bool = False,
) -> Path:
    """Get the output path for NNUE training.

    Args:
        board_type: Board type
        num_players: Number of players
        staging: If True, output to staging path instead of canonical

    Returns:
        Path where NNUE training should save the model
    """
    board = _normalize_board_type(board_type)
    models_dir = get_nnue_models_dir()

    if staging:
        return models_dir / f"nnue_staging_{board}_{num_players}p.pt"

    return get_nnue_canonical_path(board_type, num_players)


def promote_nnue_model(
    board_type: str | BoardType,
    num_players: int,
) -> bool:
    """Promote a staging NNUE model to canonical.

    Args:
        board_type: Board type
        num_players: Number of players

    Returns:
        True if promotion succeeded, False otherwise
    """
    import shutil

    staging = get_nnue_output_path(board_type, num_players, staging=True)
    canonical = get_nnue_canonical_path(board_type, num_players)

    if not staging.exists():
        logger.warning(f"No staging NNUE model at {staging}")
        return False

    # Backup existing canonical if present
    if canonical.exists():
        backup = canonical.with_suffix(".pt.backup")
        shutil.copy2(canonical, backup)
        logger.info(f"Backed up existing canonical to {backup}")

    # Promote staging to canonical
    shutil.move(str(staging), str(canonical))
    logger.info(f"Promoted NNUE model: {staging} -> {canonical}")

    # Update symlinks
    ensure_nnue_symlink(board_type, num_players)

    return True


# =============================================================================
# CLI Helper
# =============================================================================


def print_nnue_registry_status() -> None:
    """Print current NNUE registry status (for CLI use)."""
    stats = get_nnue_registry_stats()

    print("\n=== NNUE Model Registry Status ===\n")
    print(f"Models present:  {stats.models_present}/{stats.total_configs}")
    print(f"Models missing:  {stats.models_missing}/{stats.total_configs}")

    if stats.total_size_bytes > 0:
        size_mb = stats.total_size_bytes / (1024 * 1024)
        print(f"Total size:      {size_mb:.1f} MB")

    if stats.oldest_model:
        print(f"Oldest model:    {stats.oldest_model}")
    if stats.newest_model:
        print(f"Newest model:    {stats.newest_model}")

    if stats.missing_configs:
        print(f"\nMissing configs: {', '.join(stats.missing_configs)}")

    print("\n=== Detailed Status ===\n")
    for info in get_all_nnue_paths():
        status = "✓ exists" if info.exists else "✗ missing"
        if info.exists:
            size_mb = info.file_size_bytes / (1024 * 1024)
            print(f"  {info.config_key:15} {status:15} ({size_mb:.1f} MB)")
        else:
            print(f"  {info.config_key:15} {status}")


if __name__ == "__main__":
    print_nnue_registry_status()
