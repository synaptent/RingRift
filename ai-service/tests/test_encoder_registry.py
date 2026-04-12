"""Tests for the centralized encoder registry.

Verifies that encoder configurations are correct and validation works properly.
"""

import pytest

from app.models import BoardType
from app.training.encoder_registry import (
    BASE_CHANNEL_COUNTS,
    ENCODER_CHANNEL_COUNTS,
    EncoderConfig,
    get_encoder_config,
    get_expected_channels,
    list_encoder_configs,
    validate_dataset_model_compatibility,
)


class TestEncoderConfig:
    """Tests for EncoderConfig dataclass."""

    def test_frames_property(self):
        """Verify frames = history_length + 1."""
        config = EncoderConfig(
            encoder_type="test",
            encoder_class_name="TestEncoder",
            base_channels=10,
            history_length=3,
            in_channels=40,
            spatial_size=25,
            policy_size=91876,
            board_types=["TEST"],
        )
        assert config.frames == 4
        assert config.in_channels == config.base_channels * config.frames


class TestGetEncoderConfig:
    """Tests for get_encoder_config function."""

    def test_hex_v2_config(self):
        """Verify hex v2 encoder configuration."""
        config = get_encoder_config(BoardType.HEXAGONAL, "v2")
        assert config.encoder_type == "hex_v2"
        assert config.base_channels == 10
        assert config.in_channels == 40
        assert config.spatial_size == 25

    def test_hex_v3_config(self):
        """Verify hex v3 encoder configuration."""
        config = get_encoder_config(BoardType.HEXAGONAL, "v3")
        assert config.encoder_type == "hex_v3"
        assert config.base_channels == 16
        assert config.in_channels == 64
        assert config.spatial_size == 25

    def test_hex8_v2_config(self):
        """Verify hex8 v2 encoder configuration."""
        config = get_encoder_config(BoardType.HEX8, "v2")
        assert config.encoder_type == "hex_v2"
        assert config.base_channels == 10
        assert config.in_channels == 40
        assert config.spatial_size == 9

    def test_hex8_v3_config(self):
        """Verify hex8 v3 encoder configuration."""
        config = get_encoder_config(BoardType.HEX8, "v3")
        assert config.encoder_type == "hex_v3"
        assert config.base_channels == 16
        assert config.in_channels == 64
        assert config.spatial_size == 9

    def test_square8_config(self):
        """Verify square8 encoder configuration."""
        config = get_encoder_config(BoardType.SQUARE8, "v2")
        assert config.encoder_type == "square"
        assert config.base_channels == 14
        assert config.in_channels == 56
        assert config.spatial_size == 8

    def test_v4_uses_v3_encoding(self):
        """Verify v4 models use same encoding as v3."""
        v3_config = get_encoder_config(BoardType.HEXAGONAL, "v3")
        v4_config = get_encoder_config(BoardType.HEXAGONAL, "v4")
        assert v3_config.in_channels == v4_config.in_channels
        assert v3_config.base_channels == v4_config.base_channels

    def test_string_board_type(self):
        """Verify string board type works."""
        config = get_encoder_config("HEXAGONAL", "v2")
        assert config.encoder_type == "hex_v2"
        assert config.in_channels == 40

    def test_custom_history_length(self):
        """Verify custom history length adjusts in_channels."""
        # Default history_length=3 gives 4 frames
        default_config = get_encoder_config(BoardType.HEXAGONAL, "v2", history_length=3)
        assert default_config.in_channels == 40  # 10 * 4

        # Custom history_length=7 gives 8 frames
        custom_config = get_encoder_config(BoardType.HEXAGONAL, "v2", history_length=7)
        assert custom_config.in_channels == 80  # 10 * 8
        assert custom_config.history_length == 7

    def test_invalid_board_type_raises(self):
        """Verify invalid board type raises ValueError."""
        with pytest.raises(ValueError, match="No encoder configuration"):
            get_encoder_config("INVALID_BOARD", "v2")


class TestGetExpectedChannels:
    """Tests for get_expected_channels convenience function."""

    def test_hex_v2_channels(self):
        """Verify hex v2 returns 40 channels."""
        assert get_expected_channels(BoardType.HEXAGONAL, "v2") == 40

    def test_hex_v3_channels(self):
        """Verify hex v3 returns 64 channels."""
        assert get_expected_channels(BoardType.HEXAGONAL, "v3") == 64

    def test_square_channels(self):
        """Verify square returns 56 channels."""
        assert get_expected_channels(BoardType.SQUARE8, "v2") == 56


class TestValidateDatasetModelCompatibility:
    """Tests for validate_dataset_model_compatibility function."""

    def test_valid_hex_v2_compatibility(self):
        """Verify valid hex v2 dataset passes validation."""
        # Should not raise
        validate_dataset_model_compatibility(
            dataset_in_channels=40,
            board_type=BoardType.HEXAGONAL,
            model_version="v2",
        )

    def test_valid_hex_v3_compatibility(self):
        """Verify valid hex v3 dataset passes validation."""
        # Should not raise
        validate_dataset_model_compatibility(
            dataset_in_channels=64,
            board_type=BoardType.HEXAGONAL,
            model_version="v3",
        )

    def test_channel_mismatch_raises(self):
        """Verify channel mismatch raises ValueError."""
        with pytest.raises(ValueError, match="channel mismatch"):
            validate_dataset_model_compatibility(
                dataset_in_channels=56,  # Square encoding
                board_type=BoardType.HEXAGONAL,  # Hex model
                model_version="v2",  # Expects 40
            )

    def test_error_message_includes_details(self):
        """Verify error message includes helpful details."""
        with pytest.raises(ValueError) as exc_info:
            validate_dataset_model_compatibility(
                dataset_in_channels=40,
                board_type=BoardType.HEXAGONAL,
                model_version="v3",  # Expects 64
            )
        error_msg = str(exc_info.value)
        assert "40" in error_msg  # Dataset channels
        assert "64" in error_msg  # Expected channels
        assert "hex_v3" in error_msg  # Encoder type


class TestListEncoderConfigs:
    """Tests for list_encoder_configs function."""

    def test_returns_all_configs(self):
        """Verify list returns all configurations."""
        configs = list_encoder_configs()
        assert len(configs) >= 6  # At least hex/square × v2/v3

    def test_config_structure(self):
        """Verify list returns proper tuples."""
        configs = list_encoder_configs()
        for board_type, model_version, config in configs:
            assert isinstance(board_type, str)
            assert isinstance(model_version, str)
            assert model_version
            assert isinstance(config, EncoderConfig)


class TestChannelCountConstants:
    """Tests for convenience constants."""

    def test_encoder_channel_counts(self):
        """Verify ENCODER_CHANNEL_COUNTS matches registry."""
        assert ENCODER_CHANNEL_COUNTS["hex_v2"] == 40
        assert ENCODER_CHANNEL_COUNTS["hex_v3"] == 64
        assert ENCODER_CHANNEL_COUNTS["square"] == 56

    def test_base_channel_counts(self):
        """Verify BASE_CHANNEL_COUNTS matches registry."""
        assert BASE_CHANNEL_COUNTS["hex_v2"] == 10
        assert BASE_CHANNEL_COUNTS["hex_v3"] == 16
        assert BASE_CHANNEL_COUNTS["square"] == 14


class TestEncoderRegistryConsistency:
    """Tests verifying consistency across the registry."""

    def test_in_channels_equals_base_times_frames(self):
        """Verify in_channels = base_channels × frames for all configs."""
        for board_type, version, config in list_encoder_configs():
            expected = config.base_channels * config.frames
            assert config.in_channels == expected, (
                f"{board_type} {version}: in_channels={config.in_channels} "
                f"!= base({config.base_channels}) × frames({config.frames})"
            )

    def test_all_hex_boards_have_matching_channels(self):
        """Verify HEXAGONAL and HEX8 have same channel counts per version."""
        for version in ["v2", "v3"]:
            hex_config = get_encoder_config(BoardType.HEXAGONAL, version)
            hex8_config = get_encoder_config(BoardType.HEX8, version)
            assert hex_config.in_channels == hex8_config.in_channels
            assert hex_config.base_channels == hex8_config.base_channels
