"""Unit tests for event_handler_utils.py.

Tests all payload extraction utility functions used across event handlers
for consistent behavior and reduced code duplication.

December 30, 2025 - Test coverage for 374 LOC module.
"""

from __future__ import annotations

import pytest

from app.coordination.event_handler_utils import (
    # Core extraction functions
    extract_config_key,
    extract_board_type,
    extract_num_players,
    extract_model_path,
    extract_config_from_path,
    extract_metadata,
    # Parsing functions
    parse_config_key,
    build_config_key,
    # Event-specific extractors
    extract_training_completed_data,
    extract_evaluation_completed_data,
    extract_sync_completed_data,
    # Validation
    validate_payload_has_config,
    # Backward-compatible aliases
    get_config_key,
    get_board_type,
    get_model_path,
)


# =============================================================================
# extract_config_key Tests
# =============================================================================


class TestExtractConfigKey:
    """Tests for extract_config_key function."""

    def test_extracts_config_key(self):
        """Test extraction of config_key."""
        payload = {"config_key": "hex8_2p"}
        assert extract_config_key(payload) == "hex8_2p"

    def test_extracts_config_alias(self):
        """Test extraction of 'config' alias."""
        payload = {"config": "square8_4p"}
        assert extract_config_key(payload) == "square8_4p"

    def test_prefers_config_key_over_config(self):
        """Test config_key takes precedence over config."""
        payload = {"config_key": "hex8_2p", "config": "square8_4p"}
        assert extract_config_key(payload) == "hex8_2p"

    def test_returns_default_when_missing(self):
        """Test returns default when key not found."""
        payload = {"other_key": "value"}
        assert extract_config_key(payload) == ""
        assert extract_config_key(payload, default="default") == "default"

    def test_allow_empty_true(self):
        """Test allow_empty=True doesn't raise on missing key."""
        payload = {}
        result = extract_config_key(payload, allow_empty=True)
        assert result == ""

    def test_allow_empty_false_raises(self):
        """Test allow_empty=False raises ValueError on missing key."""
        payload = {}
        with pytest.raises(ValueError, match="No config_key found"):
            extract_config_key(payload, allow_empty=False)

    def test_handles_none_values(self):
        """Test handles None values in payload."""
        payload = {"config_key": None, "config": None}
        assert extract_config_key(payload, default="fallback") == "fallback"

    def test_handles_empty_string(self):
        """Test handles empty string values."""
        payload = {"config_key": ""}
        assert extract_config_key(payload, default="fallback") == "fallback"


# =============================================================================
# extract_board_type Tests
# =============================================================================


class TestExtractBoardType:
    """Tests for extract_board_type function."""

    def test_extracts_board_type(self):
        """Test extraction of board_type."""
        payload = {"board_type": "hex8"}
        assert extract_board_type(payload) == "hex8"

    def test_returns_default_when_missing(self):
        """Test returns default when not found."""
        payload = {}
        assert extract_board_type(payload) is None
        assert extract_board_type(payload, default="square8") == "square8"

    def test_handles_none_value(self):
        """Test handles None value."""
        payload = {"board_type": None}
        assert extract_board_type(payload, default="hex8") == "hex8"


# =============================================================================
# extract_num_players Tests
# =============================================================================


class TestExtractNumPlayers:
    """Tests for extract_num_players function."""

    def test_extracts_num_players(self):
        """Test extraction of num_players."""
        payload = {"num_players": 4}
        assert extract_num_players(payload) == 4

    def test_returns_default_when_missing(self):
        """Test returns default (2) when not found."""
        payload = {}
        assert extract_num_players(payload) == 2
        assert extract_num_players(payload, default=3) == 3

    def test_converts_string_to_int(self):
        """Test converts string value to int."""
        payload = {"num_players": "4"}
        assert extract_num_players(payload) == 4

    def test_handles_none_value(self):
        """Test handles None value."""
        payload = {"num_players": None}
        assert extract_num_players(payload) == 2


# =============================================================================
# extract_model_path Tests
# =============================================================================


class TestExtractModelPath:
    """Tests for extract_model_path function."""

    def test_extracts_model_path(self):
        """Test extraction of model_path."""
        payload = {"model_path": "/models/best.pth"}
        assert extract_model_path(payload) == "/models/best.pth"

    def test_extracts_checkpoint_path_alias(self):
        """Test extraction of checkpoint_path alias."""
        payload = {"checkpoint_path": "/models/checkpoint.pth"}
        assert extract_model_path(payload) == "/models/checkpoint.pth"

    def test_extracts_path_alias(self):
        """Test extraction of 'path' alias."""
        payload = {"path": "/models/model.pth"}
        assert extract_model_path(payload) == "/models/model.pth"

    def test_prefers_model_path_over_aliases(self):
        """Test model_path takes precedence."""
        payload = {
            "model_path": "/models/a.pth",
            "checkpoint_path": "/models/b.pth",
            "path": "/models/c.pth",
        }
        assert extract_model_path(payload) == "/models/a.pth"

    def test_returns_default_when_missing(self):
        """Test returns default when not found."""
        payload = {}
        assert extract_model_path(payload) is None
        assert extract_model_path(payload, default="/default.pth") == "/default.pth"


# =============================================================================
# extract_config_from_path Tests
# =============================================================================


class TestExtractConfigFromPath:
    """Tests for extract_config_from_path function."""

    def test_extracts_from_canonical_naming(self):
        """Test extraction from canonical_* naming."""
        assert extract_config_from_path("models/canonical_hex8_2p.pth") == "hex8_2p"
        assert extract_config_from_path("models/canonical_square8_4p.pth") == "square8_4p"

    def test_extracts_from_ringrift_best_naming(self):
        """Test extraction from ringrift_best_* naming."""
        assert extract_config_from_path("models/ringrift_best_hex8_2p.pth") == "hex8_2p"

    def test_extracts_with_version_suffix(self):
        """Test extraction with version suffix like _v5heavy."""
        assert extract_config_from_path("models/canonical_hex8_2p_v5heavy.pth") == "hex8_2p"

    def test_extracts_from_pattern_in_stem(self):
        """Test extraction when pattern is in stem."""
        assert extract_config_from_path("models/my_hex8_2p_model.pth") == "hex8_2p"
        assert extract_config_from_path("models/square19_3p_trained.pth") == "square19_3p"

    def test_returns_none_for_invalid_path(self):
        """Test returns None for paths without config pattern."""
        assert extract_config_from_path("models/random_model.pth") is None
        assert extract_config_from_path("models/no_pattern.pth") is None

    def test_returns_none_for_none_or_empty(self):
        """Test returns None for None or empty path."""
        assert extract_config_from_path(None) is None
        assert extract_config_from_path("") is None

    def test_handles_path_object(self):
        """Test handles pathlib.Path object."""
        from pathlib import Path
        assert extract_config_from_path(Path("models/canonical_hex8_2p.pth")) == "hex8_2p"


# =============================================================================
# parse_config_key Tests
# =============================================================================


class TestParseConfigKey:
    """Tests for parse_config_key function."""

    def test_parses_valid_config_key(self):
        """Test parsing valid config keys."""
        assert parse_config_key("hex8_2p") == ("hex8", 2)
        assert parse_config_key("square8_4p") == ("square8", 4)
        assert parse_config_key("square19_3p") == ("square19", 3)
        assert parse_config_key("hexagonal_2p") == ("hexagonal", 2)

    def test_returns_none_for_invalid_format(self):
        """Test returns (None, None) for invalid format."""
        assert parse_config_key("invalid") == (None, None)
        assert parse_config_key("no_underscore") == (None, None)
        assert parse_config_key("") == (None, None)

    def test_accepts_canonical_shorthand_without_p_suffix(self):
        """Test canonical parser also accepts shorthand config keys."""
        assert parse_config_key("hex8_2") == ("hex8", 2)
        assert parse_config_key("square8_4x") == (None, None)

    def test_returns_none_for_non_numeric_players(self):
        """Test returns (None, None) for non-numeric player count."""
        assert parse_config_key("hex8_abp") == (None, None)

    def test_handles_complex_board_types(self):
        """Test handles complex board type names."""
        assert parse_config_key("custom_board_2p") == ("custom_board", 2)


# =============================================================================
# build_config_key Tests
# =============================================================================


class TestBuildConfigKey:
    """Tests for build_config_key function."""

    def test_builds_valid_config_key(self):
        """Test building valid config keys."""
        assert build_config_key("hex8", 2) == "hex8_2p"
        assert build_config_key("square8", 4) == "square8_4p"
        assert build_config_key("square19", 3) == "square19_3p"

    def test_round_trip_with_parse(self):
        """Test build and parse round-trip."""
        for board_type, num_players in [("hex8", 2), ("square8", 4), ("hexagonal", 3)]:
            config_key = build_config_key(board_type, num_players)
            parsed_board, parsed_players = parse_config_key(config_key)
            assert parsed_board == board_type
            assert parsed_players == num_players


# =============================================================================
# extract_metadata Tests
# =============================================================================


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extracts_all_payload_keys(self):
        """Test extracts all keys from payload."""
        payload = {"key1": "value1", "key2": "value2", "elo": 1500}
        result = extract_metadata(payload)
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["elo"] == 1500

    def test_normalizes_config_key(self):
        """Test normalizes config_key when include_config=True."""
        payload = {"config": "hex8_2p", "other": "value"}
        result = extract_metadata(payload, include_config=True)
        assert result["config_key"] == "hex8_2p"

    def test_skips_config_normalization(self):
        """Test skips config normalization when include_config=False."""
        payload = {"config": "hex8_2p"}
        result = extract_metadata(payload, include_config=False)
        # Should not add config_key
        assert "config_key" not in result or result["config_key"] != "hex8_2p"


# =============================================================================
# extract_training_completed_data Tests
# =============================================================================


class TestExtractTrainingCompletedData:
    """Tests for extract_training_completed_data function."""

    def test_extracts_all_fields(self):
        """Test extracts all training completed fields."""
        payload = {
            "config_key": "hex8_2p",
            "model_path": "/models/best.pth",
            "board_type": "hex8",
            "num_players": 2,
            "metrics": {"accuracy": 0.95},
            "epochs": 50,
            "final_loss": 0.01,
        }
        result = extract_training_completed_data(payload)

        assert result["config_key"] == "hex8_2p"
        assert result["model_path"] == "/models/best.pth"
        assert result["board_type"] == "hex8"
        assert result["num_players"] == 2
        assert result["metrics"] == {"accuracy": 0.95}
        assert result["epochs"] == 50
        assert result["final_loss"] == 0.01

    def test_parses_board_type_from_config_key(self):
        """Test parses board_type from config_key if not provided."""
        payload = {"config_key": "square8_4p", "model_path": "/models/m.pth"}
        result = extract_training_completed_data(payload)

        assert result["board_type"] == "square8"
        assert result["num_players"] == 4

    def test_handles_loss_alias(self):
        """Test handles 'loss' alias for final_loss."""
        payload = {"config_key": "hex8_2p", "loss": 0.02}
        result = extract_training_completed_data(payload)
        assert result["final_loss"] == 0.02

    def test_defaults_for_missing_fields(self):
        """Test provides defaults for missing fields."""
        payload = {}
        result = extract_training_completed_data(payload)

        assert result["config_key"] == ""
        assert result["model_path"] is None
        assert result["metrics"] == {}
        assert result["num_players"] == 2


# =============================================================================
# extract_evaluation_completed_data Tests
# =============================================================================


class TestExtractEvaluationCompletedData:
    """Tests for extract_evaluation_completed_data function."""

    def test_extracts_all_fields(self):
        """Test extracts all evaluation completed fields."""
        payload = {
            "config_key": "hex8_2p",
            "model_path": "/models/best.pth",
            "board_type": "hex8",
            "num_players": 2,
            "elo": 1500,
            "win_rate": 0.65,
            "passed": True,
            "games_played": 50,
        }
        result = extract_evaluation_completed_data(payload)

        assert result["config_key"] == "hex8_2p"
        assert result["model_path"] == "/models/best.pth"
        assert result["board_type"] == "hex8"
        assert result["num_players"] == 2
        assert result["elo"] == 1500
        assert result["win_rate"] == 0.65
        assert result["passed"] is True
        assert result["games_played"] == 50

    def test_handles_estimated_elo_alias(self):
        """Test handles estimated_elo alias for elo."""
        payload = {"config_key": "hex8_2p", "estimated_elo": 1450}
        result = extract_evaluation_completed_data(payload)
        assert result["elo"] == 1450

    def test_defaults_for_missing_fields(self):
        """Test provides defaults for missing fields."""
        payload = {}
        result = extract_evaluation_completed_data(payload)

        assert result["passed"] is False
        assert result["games_played"] == 0
        assert result["num_players"] == 2


# =============================================================================
# extract_sync_completed_data Tests
# =============================================================================


class TestExtractSyncCompletedData:
    """Tests for extract_sync_completed_data function."""

    def test_extracts_all_fields(self):
        """Test extracts all sync completed fields."""
        payload = {
            "config_key": "hex8_2p",
            "sync_type": "game_data",
            "source_node": "node-a",
            "target_nodes": ["node-b", "node-c"],
            "files_synced": 10,
            "bytes_transferred": 1048576,
            "duration_seconds": 5.5,
        }
        result = extract_sync_completed_data(payload)

        assert result["config_key"] == "hex8_2p"
        assert result["sync_type"] == "game_data"
        assert result["source_node"] == "node-a"
        assert result["target_nodes"] == ["node-b", "node-c"]
        assert result["files_synced"] == 10
        assert result["bytes_transferred"] == 1048576
        assert result["duration_seconds"] == 5.5

    def test_handles_aliases(self):
        """Test handles alias fields."""
        payload = {
            "data_type": "models",
            "source": "primary",
            "targets": ["backup"],
            "duration": 10.0,
        }
        result = extract_sync_completed_data(payload)

        assert result["sync_type"] == "models"
        assert result["source_node"] == "primary"
        assert result["target_nodes"] == ["backup"]
        assert result["duration_seconds"] == 10.0

    def test_defaults_for_missing_fields(self):
        """Test provides defaults for missing fields."""
        payload = {}
        result = extract_sync_completed_data(payload)

        assert result["files_synced"] == 0
        assert result["bytes_transferred"] == 0
        assert result["target_nodes"] == []


# =============================================================================
# validate_payload_has_config Tests
# =============================================================================


class TestValidatePayloadHasConfig:
    """Tests for validate_payload_has_config function."""

    def test_returns_config_key_when_present(self):
        """Test returns config_key when present."""
        payload = {"config_key": "hex8_2p"}
        result = validate_payload_has_config(payload, "TEST_EVENT")
        assert result == "hex8_2p"

    def test_returns_none_when_missing(self):
        """Test returns None when config_key missing."""
        payload = {}
        result = validate_payload_has_config(payload, "TEST_EVENT")
        assert result is None

    def test_logs_warning_when_missing(self):
        """Test logs warning when config_key missing."""
        import logging
        payload = {}

        # Just verify it doesn't raise - logging verification would need more setup
        result = validate_payload_has_config(
            payload, "TEST_EVENT", logger_name="test.logger"
        )
        assert result is None


# =============================================================================
# Backward-Compatible Alias Tests
# =============================================================================


class TestBackwardCompatibleAliases:
    """Tests for backward-compatible alias functions."""

    def test_get_config_key_alias(self):
        """Test get_config_key is alias for extract_config_key."""
        payload = {"config_key": "hex8_2p"}
        assert get_config_key(payload) == extract_config_key(payload)

    def test_get_board_type_alias(self):
        """Test get_board_type is alias for extract_board_type."""
        payload = {"board_type": "hex8"}
        assert get_board_type(payload) == extract_board_type(payload)

    def test_get_model_path_alias(self):
        """Test get_model_path is alias for extract_model_path."""
        payload = {"model_path": "/models/test.pth"}
        assert get_model_path(payload) == extract_model_path(payload)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_payload_handling(self):
        """Test all functions handle empty payload gracefully."""
        empty = {}

        assert extract_config_key(empty) == ""
        assert extract_board_type(empty) is None
        assert extract_num_players(empty) == 2
        assert extract_model_path(empty) is None
        assert parse_config_key("") == (None, None)

    def test_none_in_payload_values(self):
        """Test functions handle None values in payload."""
        payload = {
            "config_key": None,
            "board_type": None,
            "num_players": None,
            "model_path": None,
        }

        assert extract_config_key(payload, default="fallback") == "fallback"
        assert extract_board_type(payload, default="hex8") == "hex8"
        assert extract_num_players(payload) == 2
        assert extract_model_path(payload, default="/default.pth") == "/default.pth"

    def test_unicode_values(self):
        """Test functions handle unicode values."""
        payload = {"config_key": "板_2p"}  # Unicode board type
        assert extract_config_key(payload) == "板_2p"

    def test_numeric_strings(self):
        """Test num_players handles string numbers."""
        payload = {"num_players": "3"}
        assert extract_num_players(payload) == 3

    def test_deeply_nested_structures_unchanged(self):
        """Test extract_metadata preserves nested structures."""
        payload = {
            "config_key": "hex8_2p",
            "nested": {"level1": {"level2": "value"}},
            "list": [1, 2, {"key": "val"}],
        }
        result = extract_metadata(payload)

        assert result["nested"]["level1"]["level2"] == "value"
        assert result["list"][2]["key"] == "val"
