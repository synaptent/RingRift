"""Unit tests for NNUE Model Registry.

December 2025: Phase 4 NNUE Integration

NOTE: This test file is for a module that hasn't been implemented yet.
The app/ai/nnue/registry.py module needs to be created before these tests can run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip entire module if the registry module doesn't exist
pytest.importorskip("app.ai.nnue.registry", reason="NNUE registry module not yet implemented")

from app.ai.nnue.registry import (
    CANONICAL_CONFIGS,
    NNUEModelInfo,
    NNUERegistryStats,
    get_nnue_canonical_path,
    get_nnue_config_key,
    get_nnue_model_info,
    get_all_nnue_paths,
    get_existing_nnue_models,
    get_missing_nnue_models,
    get_nnue_registry_stats,
    get_nnue_output_path,
    _normalize_board_type,
)
from app.coordination.types import BoardType


class TestNormalizeBoardType:
    """Tests for board type normalization."""

    def test_string_input(self):
        """String board types are normalized to lowercase."""
        assert _normalize_board_type("HEX8") == "hex8"
        assert _normalize_board_type("Square8") == "square8"
        assert _normalize_board_type("HEXAGONAL") == "hexagonal"

    def test_enum_input(self):
        """BoardType enums are converted to lowercase strings."""
        assert _normalize_board_type(BoardType.HEX8) == "hex8"
        assert _normalize_board_type(BoardType.SQUARE8) == "square8"


class TestGetNNUECanonicalPath:
    """Tests for canonical path resolution."""

    def test_hex8_2p(self):
        """Canonical path for hex8 2-player."""
        path = get_nnue_canonical_path("hex8", 2)
        assert path.name == "nnue_canonical_hex8_2p.pt"
        assert "nnue" in str(path)

    def test_square8_4p(self):
        """Canonical path for square8 4-player."""
        path = get_nnue_canonical_path("square8", 4)
        assert path.name == "nnue_canonical_square8_4p.pt"

    def test_with_enum(self):
        """Works with BoardType enum."""
        path = get_nnue_canonical_path(BoardType.HEXAGONAL, 3)
        assert path.name == "nnue_canonical_hexagonal_3p.pt"

    def test_all_configs_have_unique_paths(self):
        """All 12 canonical configs have unique paths."""
        paths = set()
        for board_type, num_players in CANONICAL_CONFIGS:
            path = get_nnue_canonical_path(board_type, num_players)
            assert path not in paths, f"Duplicate path: {path}"
            paths.add(path)
        assert len(paths) == 12


class TestGetNNUEConfigKey:
    """Tests for config key generation."""

    def test_basic_keys(self):
        """Config keys follow expected pattern."""
        assert get_nnue_config_key("hex8", 2) == "hex8_2p"
        assert get_nnue_config_key("square19", 4) == "square19_4p"

    def test_with_enum(self):
        """Works with BoardType enum."""
        assert get_nnue_config_key(BoardType.SQUARE8, 3) == "square8_3p"


class TestNNUEModelInfo:
    """Tests for NNUEModelInfo dataclass."""

    def test_creation_without_file(self):
        """Info for non-existent file."""
        info = NNUEModelInfo(
            config_key="test_2p",
            board_type="test",
            num_players=2,
            path=Path("/nonexistent/path.pt"),
        )
        assert info.exists is False
        assert info.file_size_bytes == 0

    def test_creation_with_existing_file(self, tmp_path):
        """Info for existing file includes metadata."""
        model_path = tmp_path / "test_model.pt"
        model_path.write_text("dummy model content")

        info = NNUEModelInfo(
            config_key="test_2p",
            board_type="test",
            num_players=2,
            path=model_path,
        )
        assert info.exists is True
        assert info.file_size_bytes > 0
        assert info.modified_time is not None


class TestGetNNUEModelInfo:
    """Tests for get_nnue_model_info function."""

    def test_returns_info_object(self):
        """Returns NNUEModelInfo with correct fields."""
        info = get_nnue_model_info("hex8", 2)
        assert info.config_key == "hex8_2p"
        assert info.board_type == "hex8"
        assert info.num_players == 2
        assert info.is_canonical is True

    def test_all_configs(self):
        """Works for all 12 configs."""
        for board_type, num_players in CANONICAL_CONFIGS:
            info = get_nnue_model_info(board_type, num_players)
            assert info.config_key == f"{board_type}_{num_players}p"


class TestGetAllNNUEPaths:
    """Tests for get_all_nnue_paths iterator."""

    def test_returns_12_items(self):
        """Returns exactly 12 NNUEModelInfo objects."""
        paths = list(get_all_nnue_paths())
        assert len(paths) == 12

    def test_all_items_are_model_info(self):
        """All items are NNUEModelInfo instances."""
        for info in get_all_nnue_paths():
            assert isinstance(info, NNUEModelInfo)

    def test_covers_all_configs(self):
        """Covers all 12 canonical configurations."""
        config_keys = {info.config_key for info in get_all_nnue_paths()}
        expected = {f"{b}_{n}p" for b, n in CANONICAL_CONFIGS}
        assert config_keys == expected


class TestExistingAndMissingModels:
    """Tests for existing/missing model discovery."""

    def test_get_existing_empty(self):
        """No models exist initially in fresh directory."""
        # This test depends on the actual models directory state
        # In a fresh install, likely no NNUE models exist
        existing = get_existing_nnue_models()
        # Just verify it returns a list
        assert isinstance(existing, list)

    def test_get_missing(self):
        """Missing models are identified."""
        missing = get_missing_nnue_models()
        assert isinstance(missing, list)
        # Total should be 12 (existing + missing)
        existing = get_existing_nnue_models()
        assert len(existing) + len(missing) == 12


class TestNNUERegistryStats:
    """Tests for registry statistics."""

    def test_default_stats(self):
        """Default stats have expected structure."""
        stats = NNUERegistryStats()
        assert stats.total_configs == 12
        assert stats.models_present == 0
        assert stats.models_missing == 0

    def test_get_nnue_registry_stats(self):
        """get_nnue_registry_stats returns valid stats."""
        stats = get_nnue_registry_stats()
        assert stats.total_configs == 12
        assert stats.models_present + stats.models_missing == 12


class TestGetNNUEOutputPath:
    """Tests for output path generation."""

    def test_canonical_output(self):
        """Non-staging output goes to canonical path."""
        path = get_nnue_output_path("hex8", 2)
        assert "canonical" in path.name

    def test_staging_output(self):
        """Staging output uses staging prefix."""
        path = get_nnue_output_path("hex8", 2, staging=True)
        assert "staging" in path.name
        assert path.name == "nnue_staging_hex8_2p.pt"


class TestCanonicalConfigs:
    """Tests for CANONICAL_CONFIGS constant."""

    def test_has_12_configs(self):
        """12 canonical configurations defined."""
        assert len(CANONICAL_CONFIGS) == 12

    def test_board_types(self):
        """All 4 board types present."""
        board_types = {b for b, n in CANONICAL_CONFIGS}
        assert board_types == {"hex8", "square8", "square19", "hexagonal"}

    def test_player_counts(self):
        """Each board type has 2, 3, 4 player variants."""
        from collections import defaultdict

        by_board = defaultdict(set)
        for board, players in CANONICAL_CONFIGS:
            by_board[board].add(players)

        for board, player_counts in by_board.items():
            assert player_counts == {2, 3, 4}, f"{board} missing player counts"
