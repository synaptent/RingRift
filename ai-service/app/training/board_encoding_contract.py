"""Board Encoding Contract — Single source of truth for board/model/channel mappings.

This module prevents the encoding mismatch bug class that wasted 200+ GPU-hours
across 5 manifestations. Every place that needs to know channel counts, encoder
classes, or architecture compatibility MUST use this contract instead of
hardcoding values or computing them locally.

Usage:
    from app.training.board_encoding_contract import (
        get_encoding_contract,
        validate_npz_channels,
        get_expected_channels,
    )

    # Before training:
    contract = get_encoding_contract(BoardType.SQUARE8, "v2")
    validate_npz_channels(npz_path, BoardType.SQUARE8)  # Fails fast if mismatch
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.models import BoardType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoardEncodingContract:
    """Immutable encoding contract for a (board_type, model_version) pair."""
    board_type: BoardType
    model_version: str
    base_channels: int
    history_frames: int = 4

    @property
    def expected_in_channels(self) -> int:
        return self.base_channels * self.history_frames


# THE CONTRACT TABLE — single source of truth
_CONTRACTS: dict[tuple[BoardType, str], BoardEncodingContract] = {}


def _register(board_type: BoardType, model_version: str, base_channels: int) -> None:
    _CONTRACTS[(board_type, model_version)] = BoardEncodingContract(
        board_type=board_type,
        model_version=model_version,
        base_channels=base_channels,
    )


# Hex boards: 10 base channels (v2), 16 (v3/v4/v5-heavy family)
for bt in (BoardType.HEX8, BoardType.HEXAGONAL):
    _register(bt, "v2", 10)
    for version in (
        "v3",
        "v4",
        "v5",
        "v5-gnn",
        "v5-heavy",
        "v5-heavy-large",
        "v5-heavy-xl",
        "v6",
        "v6-xl",
    ):
        _register(bt, version, 16)

# Square boards: 14 base channels for all current architectures
for bt in (BoardType.SQUARE8, BoardType.SQUARE19):
    for version in (
        "v2",
        "v3",
        "v4",
        "v5",
        "v5-gnn",
        "v5-heavy",
        "v5-heavy-large",
        "v5-heavy-xl",
        "v6",
        "v6-xl",
    ):
        _register(bt, version, 14)


def get_encoding_contract(
    board_type: BoardType,
    model_version: str = "v2",
) -> BoardEncodingContract:
    """Get the canonical encoding contract. Raises ValueError if unsupported."""
    key = (board_type, model_version)
    if key not in _CONTRACTS:
        available = [mv for (bt, mv) in _CONTRACTS if bt == board_type]
        raise ValueError(
            f"No encoding contract for {board_type.name}/{model_version}. "
            f"Available: {available}"
        )
    return _CONTRACTS[key]


def get_expected_channels(board_type: BoardType, model_version: str = "v2") -> int:
    """Get expected input channels for a board/model combination."""
    return get_encoding_contract(board_type, model_version).expected_in_channels


def validate_npz_channels(
    npz_path: str | Path,
    board_type: BoardType,
    model_version: str = "v2",
) -> int:
    """Validate NPZ channels match the encoding contract. Returns channel count."""
    import numpy as np

    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    with np.load(str(path), mmap_mode="r", allow_pickle=True) as d:
        if "features" not in d:
            raise ValueError(f"NPZ missing 'features' key: {npz_path}")
        actual = d["features"].shape[1]

    contract = get_encoding_contract(board_type, model_version)
    if actual != contract.expected_in_channels:
        raise ValueError(
            f"ENCODING MISMATCH (contract validation):\n"
            f"  NPZ {npz_path}: {actual} channels\n"
            f"  Contract {board_type.name}/{model_version}: "
            f"{contract.expected_in_channels} channels "
            f"({contract.base_channels} base x {contract.history_frames} frames)"
        )
    return actual


def is_valid_channel_count(channels: int) -> bool:
    """Check if a channel count matches ANY known encoding contract."""
    return any(c.expected_in_channels == channels for c in _CONTRACTS.values())


def infer_model_version_from_channels(in_channels: int, board_type: BoardType) -> str:
    """Infer model version from channel count and board type."""
    for (bt, mv), contract in _CONTRACTS.items():
        if bt == board_type and contract.expected_in_channels == in_channels:
            return mv
    return "v2"  # Safe default
