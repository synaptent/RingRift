"""Tests for scripts/lib/cli.py module.

Tests cover:
- Argument helper functions
- Config key parsing
- Path validation
- Logging setup
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.cli import (
    BOARD_TYPES,
    VALID_PLAYER_COUNTS,
    add_board_args,
    add_common_args,
    add_config_arg,
    add_dry_run_arg,
    add_limit_arg,
    add_node_arg,
    add_output_arg,
    add_parallel_arg,
    add_timeout_arg,
    add_verbose_arg,
    confirm_action,
    create_subparser_with_common_args,
    get_config_key,
    parse_config_key,
    print_dry_run_notice,
    print_summary,
    setup_cli_logging,
    validate_path_arg,
)


class TestAddVerboseArg:
    """Tests for add_verbose_arg function."""

    def test_adds_verbose_flag(self):
        """Test that --verbose flag is added."""
        parser = argparse.ArgumentParser()
        add_verbose_arg(parser)

        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

    def test_short_flag(self):
        """Test that -v short flag works."""
        parser = argparse.ArgumentParser()
        add_verbose_arg(parser)

        args = parser.parse_args(["-v"])
        assert args.verbose is True

    def test_default_false(self):
        """Test that verbose defaults to False."""
        parser = argparse.ArgumentParser()
        add_verbose_arg(parser)

        args = parser.parse_args([])
        assert args.verbose is False


class TestAddDryRunArg:
    """Tests for add_dry_run_arg function."""

    def test_adds_dry_run_flag(self):
        """Test that --dry-run flag is added."""
        parser = argparse.ArgumentParser()
        add_dry_run_arg(parser)

        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_default_false(self):
        """Test that dry_run defaults to False."""
        parser = argparse.ArgumentParser()
        add_dry_run_arg(parser)

        args = parser.parse_args([])
        assert args.dry_run is False


class TestAddConfigArg:
    """Tests for add_config_arg function."""

    def test_adds_config_arg(self):
        """Test that --config argument is added."""
        parser = argparse.ArgumentParser()
        add_config_arg(parser, default="default.yaml")

        args = parser.parse_args(["--config", "custom.yaml"])
        assert args.config == "custom.yaml"

    def test_short_flag(self):
        """Test that -c short flag works."""
        parser = argparse.ArgumentParser()
        add_config_arg(parser)

        args = parser.parse_args(["-c", "test.yaml"])
        assert args.config == "test.yaml"

    def test_default_value(self):
        """Test default config value."""
        parser = argparse.ArgumentParser()
        add_config_arg(parser, default="default.yaml")

        args = parser.parse_args([])
        assert args.config == "default.yaml"


class TestAddNodeArg:
    """Tests for add_node_arg function."""

    def test_adds_node_id_arg(self):
        """Test that --node-id argument is added."""
        parser = argparse.ArgumentParser()
        add_node_arg(parser, required=False, default="test-node")

        args = parser.parse_args([])
        assert args.node_id == "test-node"

    def test_override_value(self):
        """Test overriding node ID."""
        parser = argparse.ArgumentParser()
        add_node_arg(parser, required=False, default="default")

        args = parser.parse_args(["--node-id", "custom-node"])
        assert args.node_id == "custom-node"


class TestAddBoardArgs:
    """Tests for add_board_args function."""

    def test_adds_board_and_players(self):
        """Test that board and num-players are added."""
        parser = argparse.ArgumentParser()
        add_board_args(parser)

        args = parser.parse_args(["--board", "hexagonal", "--num-players", "3"])
        assert args.board == "hexagonal"
        assert args.num_players == 3

    def test_default_values(self):
        """Test default board and players."""
        parser = argparse.ArgumentParser()
        add_board_args(parser)

        args = parser.parse_args([])
        assert args.board == "square8"
        assert args.num_players == 2

    def test_custom_defaults(self):
        """Test custom default values."""
        parser = argparse.ArgumentParser()
        add_board_args(parser, default_board="hexagonal", default_players=4)

        args = parser.parse_args([])
        assert args.board == "hexagonal"
        assert args.num_players == 4


class TestAddOutputArg:
    """Tests for add_output_arg function."""

    def test_adds_output_arg(self):
        """Test that --output argument is added."""
        parser = argparse.ArgumentParser()
        add_output_arg(parser)

        args = parser.parse_args(["--output", "/tmp/output"])
        assert args.output == "/tmp/output"


class TestAddLimitArg:
    """Tests for add_limit_arg function."""

    def test_adds_limit_arg(self):
        """Test that --limit argument is added."""
        parser = argparse.ArgumentParser()
        add_limit_arg(parser)

        args = parser.parse_args(["--limit", "100"])
        assert args.limit == 100


class TestAddParallelArg:
    """Tests for add_parallel_arg function."""

    def test_adds_jobs_arg(self):
        """Test that --jobs argument is added."""
        parser = argparse.ArgumentParser()
        add_parallel_arg(parser)

        args = parser.parse_args(["--jobs", "4"])
        assert args.jobs == 4


class TestAddTimeoutArg:
    """Tests for add_timeout_arg function."""

    def test_adds_timeout_arg(self):
        """Test that --timeout argument is added."""
        parser = argparse.ArgumentParser()
        add_timeout_arg(parser, default=60)

        args = parser.parse_args([])
        assert args.timeout == 60


class TestAddCommonArgs:
    """Tests for add_common_args function."""

    def test_adds_verbose_and_dry_run(self):
        """Test that common args are added."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["--verbose", "--dry-run"])
        assert args.verbose is True
        assert args.dry_run is True

    def test_with_config(self):
        """Test adding config argument."""
        parser = argparse.ArgumentParser()
        add_common_args(parser, config=True, config_default="test.yaml")

        args = parser.parse_args([])
        assert args.config == "test.yaml"

    def test_selective_args(self):
        """Test adding only specific args."""
        parser = argparse.ArgumentParser()
        add_common_args(parser, verbose=True, dry_run=False)

        args = parser.parse_args(["-v"])
        assert args.verbose is True
        assert not hasattr(args, "dry_run")


class TestSetupCliLogging:
    """Tests for setup_cli_logging function."""

    def test_returns_logger(self):
        """Test that logger is returned."""
        parser = argparse.ArgumentParser()
        add_verbose_arg(parser)
        args = parser.parse_args([])

        logger = setup_cli_logging("test_script", args)
        assert logger is not None


class TestGetConfigKey:
    """Tests for get_config_key function."""

    def test_standard_config(self):
        """Test standard config key generation."""
        assert get_config_key("square8", 2) == "square8_2p"
        assert get_config_key("hexagonal", 3) == "hexagonal_3p"
        assert get_config_key("square19", 4) == "square19_4p"


class TestParseConfigKey:
    """Tests for parse_config_key function."""

    def test_parse_valid_key(self):
        """Test parsing valid config keys."""
        board, players = parse_config_key("square8_2p")
        assert board == "square8"
        assert players == 2

        board, players = parse_config_key("hexagonal_4p")
        assert board == "hexagonal"
        assert players == 4

    def test_parse_invalid_format(self):
        """Test parsing invalid format."""
        with pytest.raises(ValueError):
            parse_config_key("invalid")

    def test_parse_invalid_players(self):
        """Test parsing invalid player count."""
        with pytest.raises(ValueError):
            parse_config_key("square8_xp")


class TestValidatePathArg:
    """Tests for validate_path_arg function."""

    def test_existing_file(self, tmp_path):
        """Test validating existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = validate_path_arg(test_file, must_exist=True, is_file=True)
        assert result == test_file

    def test_nonexistent_path(self, tmp_path):
        """Test validating nonexistent path."""
        with pytest.raises(argparse.ArgumentTypeError):
            validate_path_arg(tmp_path / "nonexistent", must_exist=True)

    def test_create_dir(self, tmp_path):
        """Test creating directory."""
        new_dir = tmp_path / "new_dir"
        result = validate_path_arg(new_dir, is_dir=True, create_dir=True)
        assert result.exists()
        assert result.is_dir()

    def test_wrong_type(self, tmp_path):
        """Test wrong path type."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with pytest.raises(argparse.ArgumentTypeError):
            validate_path_arg(test_file, is_dir=True)


class TestConfirmAction:
    """Tests for confirm_action function."""

    def test_skip_confirm(self):
        """Test skipping confirmation."""
        result = confirm_action("Continue?", skip_confirm=True)
        assert result is True

    @patch("builtins.input", return_value="y")
    def test_user_confirms(self, mock_input):
        """Test user confirming."""
        result = confirm_action("Continue?")
        assert result is True

    @patch("builtins.input", return_value="n")
    def test_user_denies(self, mock_input):
        """Test user denying."""
        result = confirm_action("Continue?")
        assert result is False

    @patch("builtins.input", return_value="")
    def test_default_true(self, mock_input):
        """Test default yes."""
        result = confirm_action("Continue?", default=True)
        assert result is True

    @patch("builtins.input", return_value="")
    def test_default_false(self, mock_input):
        """Test default no."""
        result = confirm_action("Continue?", default=False)
        assert result is False


class TestPrintDryRunNotice:
    """Tests for print_dry_run_notice function."""

    def test_prints_notice(self, capsys):
        """Test that notice is printed."""
        print_dry_run_notice()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_prints_summary(self, capsys):
        """Test that summary is printed."""
        print_summary("Test Summary", {"key1": "value1", "key2": 42})
        captured = capsys.readouterr()
        assert "Test Summary" in captured.out
        assert "key1: value1" in captured.out
        assert "key2: 42" in captured.out


class TestCreateSubparserWithCommonArgs:
    """Tests for create_subparser_with_common_args function."""

    def test_creates_subparser(self):
        """Test creating subparser with common args."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_subparser_with_common_args(
            subparsers, "test", "Test command", verbose=True, dry_run=True
        )

        args = parser.parse_args(["test", "--verbose", "--dry-run"])
        assert args.verbose is True
        assert args.dry_run is True


class TestConstants:
    """Tests for module constants."""

    def test_board_types(self):
        """Test board types list."""
        assert "square8" in BOARD_TYPES
        assert "hexagonal" in BOARD_TYPES

    def test_player_counts(self):
        """Test valid player counts."""
        assert 2 in VALID_PLAYER_COUNTS
        assert 3 in VALID_PLAYER_COUNTS
        assert 4 in VALID_PLAYER_COUNTS
