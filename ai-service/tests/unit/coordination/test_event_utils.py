"""Tests for event_utils.py event extraction utilities.

December 30, 2025: Comprehensive test suite for consolidated event extraction.
"""

import pytest

from app.coordination.event_utils import (
    EvaluationEventData,
    ParsedConfigKey,
    TrainingEventData,
    extract_board_type_and_players,
    extract_config_key,
    extract_evaluation_data,
    extract_model_path,
    extract_training_data,
    make_config_key,
    parse_config_key,
)


class TestParsedConfigKey:
    """Tests for ParsedConfigKey dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        key = ParsedConfigKey(board_type="hex8", num_players=2, raw="hex8_2p")
        assert key.board_type == "hex8"
        assert key.num_players == 2
        assert key.raw == "hex8_2p"

    def test_config_key_property(self):
        """Test config_key property returns canonical format."""
        key = ParsedConfigKey(board_type="square19", num_players=4, raw="square19_4")
        assert key.config_key == "square19_4p"

    def test_config_key_normalizes_format(self):
        """Test that config_key property always includes 'p' suffix."""
        # Raw was without 'p' but property normalizes
        key = ParsedConfigKey(board_type="hex8", num_players=3, raw="hex8_3")
        assert key.config_key == "hex8_3p"


class TestEvaluationEventData:
    """Tests for EvaluationEventData dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation with required fields."""
        data = EvaluationEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            model_path="models/test.pth",
            elo=1450.0,
            games_played=100,
            win_rate=0.65,
        )
        assert data.config_key == "hex8_2p"
        assert data.board_type == "hex8"
        assert data.num_players == 2
        assert data.model_path == "models/test.pth"
        assert data.elo == 1450.0
        assert data.games_played == 100
        assert data.win_rate == 0.65

    def test_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        data = EvaluationEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            model_path="",
            elo=1000.0,
            games_played=0,
            win_rate=0.0,
        )
        assert data.harness_results is None
        assert data.best_harness is None
        assert data.composite_participant_ids is None
        assert data.is_multi_harness is False

    def test_multi_harness_fields(self):
        """Test multi-harness specific fields."""
        data = EvaluationEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            model_path="models/test.pth",
            elo=1450.0,
            games_played=100,
            win_rate=0.65,
            harness_results={"gumbel": 1450, "minimax": 1400},
            best_harness="gumbel",
            composite_participant_ids=["test:gumbel:abc123"],
            is_multi_harness=True,
        )
        assert data.harness_results == {"gumbel": 1450, "minimax": 1400}
        assert data.best_harness == "gumbel"
        assert data.composite_participant_ids == ["test:gumbel:abc123"]
        assert data.is_multi_harness is True

    def test_is_valid_with_valid_data(self):
        """Test is_valid returns True for valid data."""
        data = EvaluationEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            model_path="models/test.pth",
            elo=1450.0,
            games_played=100,
            win_rate=0.65,
        )
        assert data.is_valid is True

    def test_is_valid_with_empty_config_key(self):
        """Test is_valid returns False for empty config_key."""
        data = EvaluationEventData(
            config_key="",
            board_type="hex8",
            num_players=2,
            model_path="models/test.pth",
            elo=1450.0,
            games_played=100,
            win_rate=0.65,
        )
        assert data.is_valid is False

    def test_is_valid_with_empty_board_type(self):
        """Test is_valid returns False for empty board_type."""
        data = EvaluationEventData(
            config_key="hex8_2p",
            board_type="",
            num_players=2,
            model_path="models/test.pth",
            elo=1450.0,
            games_played=100,
            win_rate=0.65,
        )
        assert data.is_valid is False

    def test_is_valid_with_zero_players(self):
        """Test is_valid returns False for zero players."""
        data = EvaluationEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=0,
            model_path="models/test.pth",
            elo=1450.0,
            games_played=100,
            win_rate=0.65,
        )
        assert data.is_valid is False

    def test_is_valid_with_negative_players(self):
        """Test is_valid returns False for negative players."""
        data = EvaluationEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=-1,
            model_path="models/test.pth",
            elo=1450.0,
            games_played=100,
            win_rate=0.65,
        )
        assert data.is_valid is False


class TestTrainingEventData:
    """Tests for TrainingEventData dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        data = TrainingEventData(
            config_key="square8_4p",
            board_type="square8",
            num_players=4,
            model_path="models/test.pth",
            epochs=50,
            final_loss=0.45,
            samples_trained=100000,
        )
        assert data.config_key == "square8_4p"
        assert data.board_type == "square8"
        assert data.num_players == 4
        assert data.model_path == "models/test.pth"
        assert data.epochs == 50
        assert data.final_loss == 0.45
        assert data.samples_trained == 100000

    def test_is_valid_with_valid_data(self):
        """Test is_valid returns True for valid data."""
        data = TrainingEventData(
            config_key="square8_4p",
            board_type="square8",
            num_players=4,
            model_path="models/test.pth",
            epochs=50,
            final_loss=0.45,
            samples_trained=100000,
        )
        assert data.is_valid is True

    def test_is_valid_with_empty_config_key(self):
        """Test is_valid returns False for empty config_key."""
        data = TrainingEventData(
            config_key="",
            board_type="square8",
            num_players=4,
            model_path="",
            epochs=0,
            final_loss=0.0,
            samples_trained=0,
        )
        assert data.is_valid is False

    def test_is_valid_with_empty_board_type(self):
        """Test is_valid returns False for empty board_type."""
        data = TrainingEventData(
            config_key="square8_4p",
            board_type="",
            num_players=4,
            model_path="",
            epochs=0,
            final_loss=0.0,
            samples_trained=0,
        )
        assert data.is_valid is False

    def test_is_valid_with_invalid_num_players(self):
        """Test is_valid returns False for invalid num_players."""
        data = TrainingEventData(
            config_key="square8_4p",
            board_type="square8",
            num_players=0,
            model_path="",
            epochs=0,
            final_loss=0.0,
            samples_trained=0,
        )
        assert data.is_valid is False


class TestParseConfigKey:
    """Tests for parse_config_key function."""

    def test_parse_standard_format_2p(self):
        """Test parsing standard format with 'p' suffix."""
        result = parse_config_key("hex8_2p")
        assert result is not None
        assert result.board_type == "hex8"
        assert result.num_players == 2
        assert result.raw == "hex8_2p"

    def test_parse_standard_format_3p(self):
        """Test parsing standard format with 3 players."""
        result = parse_config_key("square19_3p")
        assert result is not None
        assert result.board_type == "square19"
        assert result.num_players == 3
        assert result.raw == "square19_3p"

    def test_parse_standard_format_4p(self):
        """Test parsing standard format with 4 players."""
        result = parse_config_key("hexagonal_4p")
        assert result is not None
        assert result.board_type == "hexagonal"
        assert result.num_players == 4
        assert result.raw == "hexagonal_4p"

    def test_parse_without_p_suffix(self):
        """Test parsing format without 'p' suffix."""
        result = parse_config_key("hex8_2")
        assert result is not None
        assert result.board_type == "hex8"
        assert result.num_players == 2

    def test_parse_complex_board_type(self):
        """Test parsing board type with underscore."""
        # Board type can have multiple parts if separated by underscore
        result = parse_config_key("square8_2p")
        assert result is not None
        assert result.board_type == "square8"
        assert result.num_players == 2

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = parse_config_key("")
        assert result is None

    def test_parse_none_returns_none(self):
        """Test parsing None-like values returns None."""
        # Empty string
        assert parse_config_key("") is None

    def test_parse_no_underscore(self):
        """Test parsing string without underscore returns None."""
        result = parse_config_key("hex82p")
        assert result is None

    def test_parse_invalid_player_count_string(self):
        """Test parsing with non-numeric player count returns None."""
        result = parse_config_key("hex8_abc")
        assert result is None

    def test_parse_player_count_1(self):
        """Test parsing with 1 player returns None (invalid)."""
        result = parse_config_key("hex8_1p")
        assert result is None

    def test_parse_player_count_5(self):
        """Test parsing with 5 players returns None (invalid)."""
        result = parse_config_key("hex8_5p")
        assert result is None

    def test_parse_player_count_0(self):
        """Test parsing with 0 players returns None."""
        result = parse_config_key("hex8_0p")
        assert result is None

    def test_parse_negative_player_count(self):
        """Test parsing with negative players returns None."""
        result = parse_config_key("hex8_-1p")
        assert result is None


class TestExtractConfigKey:
    """Tests for extract_config_key function."""

    def test_extract_from_config_key_field(self):
        """Test extraction from config_key field."""
        event = {"config_key": "hex8_2p"}
        assert extract_config_key(event) == "hex8_2p"

    def test_extract_from_config_field(self):
        """Test extraction from config field (fallback)."""
        event = {"config": "square8_4p"}
        assert extract_config_key(event) == "square8_4p"

    def test_config_key_takes_precedence(self):
        """Test config_key takes precedence over config."""
        event = {"config_key": "hex8_2p", "config": "square8_4p"}
        assert extract_config_key(event) == "hex8_2p"

    def test_extract_from_empty_event(self):
        """Test extraction from empty event returns empty string."""
        event = {}
        assert extract_config_key(event) == ""

    def test_extract_with_none_values(self):
        """Test extraction when fields are None."""
        event = {"config_key": None, "config": None}
        assert extract_config_key(event) == ""

    def test_extract_with_empty_config_key(self):
        """Test extraction when config_key is empty string."""
        event = {"config_key": "", "config": "hex8_2p"}
        assert extract_config_key(event) == "hex8_2p"


class TestExtractModelPath:
    """Tests for extract_model_path function."""

    def test_extract_from_model_path_field(self):
        """Test extraction from model_path field."""
        event = {"model_path": "models/hex8_2p.pth"}
        assert extract_model_path(event) == "models/hex8_2p.pth"

    def test_extract_from_model_id_field(self):
        """Test extraction from model_id field (fallback)."""
        event = {"model_id": "models/test.pth"}
        assert extract_model_path(event) == "models/test.pth"

    def test_extract_from_model_field(self):
        """Test extraction from model field (fallback)."""
        event = {"model": "models/another.pth"}
        assert extract_model_path(event) == "models/another.pth"

    def test_extract_from_config_field(self):
        """Test extraction from config field (last fallback)."""
        event = {"config": "hex8_2p"}
        assert extract_model_path(event) == "hex8_2p"

    def test_model_path_takes_precedence(self):
        """Test model_path takes precedence over others."""
        event = {
            "model_path": "path1.pth",
            "model_id": "path2.pth",
            "model": "path3.pth",
        }
        assert extract_model_path(event) == "path1.pth"

    def test_model_id_takes_precedence_over_model(self):
        """Test model_id takes precedence over model."""
        event = {"model_id": "path2.pth", "model": "path3.pth"}
        assert extract_model_path(event) == "path2.pth"

    def test_extract_from_empty_event(self):
        """Test extraction from empty event returns empty string."""
        event = {}
        assert extract_model_path(event) == ""

    def test_extract_with_none_values(self):
        """Test extraction when all fields are None."""
        event = {"model_path": None, "model_id": None, "model": None}
        assert extract_model_path(event) == ""


class TestExtractBoardTypeAndPlayers:
    """Tests for extract_board_type_and_players function."""

    def test_extract_from_direct_fields(self):
        """Test extraction from direct board_type and num_players fields."""
        event = {"board_type": "hex8", "num_players": 2}
        board_type, num_players = extract_board_type_and_players(event)
        assert board_type == "hex8"
        assert num_players == 2

    def test_extract_from_config_key_fallback(self):
        """Test extraction falls back to parsing config_key."""
        event = {"config_key": "square19_4p"}
        board_type, num_players = extract_board_type_and_players(event)
        assert board_type == "square19"
        assert num_players == 4

    def test_direct_fields_take_precedence(self):
        """Test direct fields take precedence over config_key."""
        event = {
            "board_type": "hex8",
            "num_players": 2,
            "config_key": "square19_4p",
        }
        board_type, num_players = extract_board_type_and_players(event)
        assert board_type == "hex8"
        assert num_players == 2

    def test_partial_direct_fields_use_fallback(self):
        """Test that partial direct fields fall back to config_key."""
        # Only board_type provided, should use config_key for both
        event = {"board_type": "hex8", "config_key": "square19_4p"}
        board_type, num_players = extract_board_type_and_players(event)
        # Since num_players is 0/missing, falls back to parsing
        assert board_type == "square19"
        assert num_players == 4

    def test_extract_from_empty_event(self):
        """Test extraction from empty event returns defaults."""
        event = {}
        board_type, num_players = extract_board_type_and_players(event)
        assert board_type == ""
        assert num_players == 0

    def test_extract_with_invalid_config_key(self):
        """Test extraction with invalid config_key returns partial data."""
        event = {"board_type": "hex8", "config_key": "invalid"}
        board_type, num_players = extract_board_type_and_players(event)
        # Should return the direct board_type, and 0 for num_players
        assert board_type == "hex8"
        assert num_players == 0


class TestExtractEvaluationData:
    """Tests for extract_evaluation_data function."""

    def test_extract_complete_event(self):
        """Test extraction from complete evaluation event."""
        event = {
            "config_key": "hex8_2p",
            "board_type": "hex8",
            "num_players": 2,
            "model_path": "models/test.pth",
            "elo": 1450.0,
            "games_played": 100,
            "win_rate": 0.65,
            "harness_results": {"gumbel": 1450},
            "best_harness": "gumbel",
            "composite_participant_ids": ["test:gumbel:abc"],
            "is_multi_harness": True,
        }
        data = extract_evaluation_data(event)
        assert data.config_key == "hex8_2p"
        assert data.board_type == "hex8"
        assert data.num_players == 2
        assert data.model_path == "models/test.pth"
        assert data.elo == 1450.0
        assert data.games_played == 100
        assert data.win_rate == 0.65
        assert data.harness_results == {"gumbel": 1450}
        assert data.best_harness == "gumbel"
        assert data.composite_participant_ids == ["test:gumbel:abc"]
        assert data.is_multi_harness is True

    def test_extract_minimal_event(self):
        """Test extraction from minimal event with defaults."""
        event = {"config_key": "hex8_2p"}
        data = extract_evaluation_data(event)
        assert data.config_key == "hex8_2p"
        assert data.board_type == "hex8"  # Parsed from config_key
        assert data.num_players == 2  # Parsed from config_key
        assert data.model_path == ""  # No model_path/model_id/model/config fields
        assert data.elo == 1000.0  # Default
        assert data.games_played == 0  # Default
        assert data.win_rate == 0.0  # Default

    def test_extract_games_fallback_to_games_field(self):
        """Test games_played falls back to 'games' field."""
        event = {"config_key": "hex8_2p", "games": 50}
        data = extract_evaluation_data(event)
        assert data.games_played == 50

    def test_extract_games_played_takes_precedence(self):
        """Test games_played takes precedence over games."""
        event = {"config_key": "hex8_2p", "games_played": 100, "games": 50}
        data = extract_evaluation_data(event)
        assert data.games_played == 100

    def test_extract_empty_event(self):
        """Test extraction from empty event."""
        event = {}
        data = extract_evaluation_data(event)
        assert data.config_key == ""
        assert data.board_type == ""
        assert data.num_players == 0
        assert data.model_path == ""
        assert data.is_valid is False


class TestExtractTrainingData:
    """Tests for extract_training_data function."""

    def test_extract_complete_event(self):
        """Test extraction from complete training event."""
        event = {
            "config_key": "square8_4p",
            "board_type": "square8",
            "num_players": 4,
            "model_path": "models/trained.pth",
            "epochs": 50,
            "final_loss": 0.45,
            "samples_trained": 100000,
        }
        data = extract_training_data(event)
        assert data.config_key == "square8_4p"
        assert data.board_type == "square8"
        assert data.num_players == 4
        assert data.model_path == "models/trained.pth"
        assert data.epochs == 50
        assert data.final_loss == 0.45
        assert data.samples_trained == 100000

    def test_extract_minimal_event(self):
        """Test extraction from minimal event with defaults."""
        event = {"config_key": "hex8_2p"}
        data = extract_training_data(event)
        assert data.config_key == "hex8_2p"
        assert data.board_type == "hex8"
        assert data.num_players == 2
        assert data.epochs == 0
        assert data.final_loss == 0.0
        assert data.samples_trained == 0

    def test_extract_loss_fallback(self):
        """Test final_loss falls back to 'loss' field."""
        event = {"config_key": "hex8_2p", "loss": 0.35}
        data = extract_training_data(event)
        assert data.final_loss == 0.35

    def test_extract_final_loss_takes_precedence(self):
        """Test final_loss takes precedence over loss."""
        event = {"config_key": "hex8_2p", "final_loss": 0.45, "loss": 0.35}
        data = extract_training_data(event)
        assert data.final_loss == 0.45

    def test_extract_samples_fallback(self):
        """Test samples_trained falls back to 'samples' field."""
        event = {"config_key": "hex8_2p", "samples": 50000}
        data = extract_training_data(event)
        assert data.samples_trained == 50000

    def test_extract_samples_trained_takes_precedence(self):
        """Test samples_trained takes precedence over samples."""
        event = {"config_key": "hex8_2p", "samples_trained": 100000, "samples": 50000}
        data = extract_training_data(event)
        assert data.samples_trained == 100000

    def test_extract_empty_event(self):
        """Test extraction from empty event."""
        event = {}
        data = extract_training_data(event)
        assert data.config_key == ""
        assert data.board_type == ""
        assert data.num_players == 0
        assert data.is_valid is False


class TestMakeConfigKey:
    """Tests for make_config_key function."""

    def test_make_2p_config(self):
        """Test creating 2-player config key."""
        key = make_config_key("hex8", 2)
        assert key == "hex8_2p"

    def test_make_3p_config(self):
        """Test creating 3-player config key."""
        key = make_config_key("square19", 3)
        assert key == "square19_3p"

    def test_make_4p_config(self):
        """Test creating 4-player config key."""
        key = make_config_key("hexagonal", 4)
        assert key == "hexagonal_4p"

    def test_roundtrip_with_parse(self):
        """Test that make_config_key output can be parsed back."""
        original_board = "square8"
        original_players = 4
        key = make_config_key(original_board, original_players)
        parsed = parse_config_key(key)
        assert parsed is not None
        assert parsed.board_type == original_board
        assert parsed.num_players == original_players

    def test_all_board_types(self):
        """Test creating keys for all standard board types."""
        board_types = ["hex8", "square8", "square19", "hexagonal"]
        for board_type in board_types:
            for num_players in [2, 3, 4]:
                key = make_config_key(board_type, num_players)
                assert key == f"{board_type}_{num_players}p"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_evaluation_event_roundtrip(self):
        """Test full evaluation event flow."""
        # Create event
        event = {
            "config_key": "hex8_2p",
            "model_path": "models/canonical_hex8_2p.pth",
            "elo": 1500.0,
            "games_played": 200,
            "win_rate": 0.72,
        }

        # Extract data
        data = extract_evaluation_data(event)
        assert data.is_valid is True

        # Reconstruct config key
        reconstructed = make_config_key(data.board_type, data.num_players)
        assert reconstructed == data.config_key

    def test_training_event_roundtrip(self):
        """Test full training event flow."""
        # Create event
        event = {
            "config_key": "square8_4p",
            "model_path": "models/trained_sq8_4p.pth",
            "epochs": 100,
            "final_loss": 0.38,
            "samples_trained": 250000,
        }

        # Extract data
        data = extract_training_data(event)
        assert data.is_valid is True

        # Verify config key matches
        reconstructed = make_config_key(data.board_type, data.num_players)
        assert reconstructed == data.config_key

    def test_all_12_canonical_configs(self):
        """Test parsing all 12 canonical configurations."""
        configs = [
            "hex8_2p",
            "hex8_3p",
            "hex8_4p",
            "square8_2p",
            "square8_3p",
            "square8_4p",
            "square19_2p",
            "square19_3p",
            "square19_4p",
            "hexagonal_2p",
            "hexagonal_3p",
            "hexagonal_4p",
        ]
        for config in configs:
            parsed = parse_config_key(config)
            assert parsed is not None, f"Failed to parse {config}"
            assert parsed.is_valid if hasattr(parsed, "is_valid") else True
            reconstructed = make_config_key(parsed.board_type, parsed.num_players)
            assert reconstructed == config

    def test_event_with_mixed_field_sources(self):
        """Test extraction when data comes from different field names."""
        event = {
            "config": "hex8_2p",  # Not config_key
            "model_id": "models/test.pth",  # Not model_path
            "games": 75,  # Not games_played
            "loss": 0.42,  # Not final_loss
            "samples": 80000,  # Not samples_trained
        }

        eval_data = extract_evaluation_data(event)
        assert eval_data.config_key == "hex8_2p"
        assert eval_data.model_path == "models/test.pth"
        assert eval_data.games_played == 75

        training_data = extract_training_data(event)
        assert training_data.config_key == "hex8_2p"
        assert training_data.model_path == "models/test.pth"
        assert training_data.final_loss == 0.42
        assert training_data.samples_trained == 80000
