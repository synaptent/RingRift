"""Tests for scripts/lib/paths.py module.

Tests cover:
- Directory constants exist and are valid paths
- Path utility functions
- Directory creation utilities
"""

import pytest
from pathlib import Path

from scripts.lib.paths import (
    AI_SERVICE_ROOT,
    DATA_DIR,
    GAMES_DIR,
    SELFPLAY_DIR,
    TRAINING_DIR,
    MODELS_DIR,
    NNUE_MODELS_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    SCRIPTS_DIR,
    UNIFIED_ELO_DB,
    get_game_db_path,
    get_training_data_path,
    get_model_path,
    get_log_path,
    ensure_dir,
    ensure_parent_dir,
)


class TestDirectoryConstants:
    """Tests for directory constants."""

    def test_ai_service_root_is_path(self):
        """Test AI_SERVICE_ROOT is a Path."""
        assert isinstance(AI_SERVICE_ROOT, Path)

    def test_ai_service_root_exists(self):
        """Test AI_SERVICE_ROOT points to valid directory."""
        assert AI_SERVICE_ROOT.exists()
        assert AI_SERVICE_ROOT.is_dir()

    def test_data_dir_under_root(self):
        """Test DATA_DIR is under AI_SERVICE_ROOT."""
        assert DATA_DIR.parent == AI_SERVICE_ROOT
        assert DATA_DIR.name == "data"

    def test_games_dir_under_data(self):
        """Test GAMES_DIR is under DATA_DIR."""
        assert GAMES_DIR.parent == DATA_DIR
        assert GAMES_DIR.name == "games"

    def test_models_dir_under_root(self):
        """Test MODELS_DIR is under AI_SERVICE_ROOT."""
        assert MODELS_DIR.parent == AI_SERVICE_ROOT
        assert MODELS_DIR.name == "models"

    def test_logs_dir_under_root(self):
        """Test LOGS_DIR is under AI_SERVICE_ROOT."""
        assert LOGS_DIR.parent == AI_SERVICE_ROOT
        assert LOGS_DIR.name == "logs"

    def test_scripts_dir_under_root(self):
        """Test SCRIPTS_DIR is under AI_SERVICE_ROOT."""
        assert SCRIPTS_DIR.parent == AI_SERVICE_ROOT
        assert SCRIPTS_DIR.name == "scripts"

    def test_unified_elo_db_path(self):
        """Test UNIFIED_ELO_DB path is correct."""
        assert UNIFIED_ELO_DB.parent == DATA_DIR
        assert UNIFIED_ELO_DB.name == "unified_elo.db"


class TestGetGameDbPath:
    """Tests for get_game_db_path function."""

    def test_square8_2p(self):
        """Test path for square8_2p config."""
        path = get_game_db_path("square8_2p")
        assert path.parent == GAMES_DIR
        assert path.name == "selfplay_square8_2p.db"

    def test_hex7_3p(self):
        """Test path for hex7_3p config."""
        path = get_game_db_path("hex7_3p")
        assert path.name == "selfplay_hex7_3p.db"


class TestGetTrainingDataPath:
    """Tests for get_training_data_path function."""

    def test_default_suffix(self):
        """Test path with default .npz suffix."""
        path = get_training_data_path("square8_2p")
        assert path.parent == TRAINING_DIR
        assert path.name == "training_square8_2p.npz"

    def test_custom_suffix(self):
        """Test path with custom suffix."""
        path = get_training_data_path("hex7_3p", suffix=".hdf5")
        assert path.name == "training_hex7_3p.hdf5"


class TestGetModelPath:
    """Tests for get_model_path function."""

    def test_default_nnue(self):
        """Test default NNUE model path."""
        path = get_model_path("square8_2p")
        assert path.name == "nnue_square8_2p.pt"

    def test_policy_model(self):
        """Test policy model path."""
        path = get_model_path("square8_2p", model_type="policy")
        assert "policy" in str(path)

    def test_custom_filename(self):
        """Test with custom filename."""
        path = get_model_path("square8_2p", filename="custom.pt")
        assert path.name == "custom.pt"


class TestGetLogPath:
    """Tests for get_log_path function."""

    def test_log_path(self):
        """Test log path generation."""
        path = get_log_path("my_script")
        assert path.parent == LOGS_DIR
        assert path.name == "my_script.log"


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_creates_directory(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "new_dir"
        assert not new_dir.exists()

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        existing = tmp_path / "existing"
        existing.mkdir()

        result = ensure_dir(existing)

        assert result == existing
        assert existing.is_dir()

    def test_nested_directories(self, tmp_path):
        """Test creating nested directories."""
        nested = tmp_path / "a" / "b" / "c"

        result = ensure_dir(nested)

        assert nested.exists()
        assert result == nested


class TestEnsureParentDir:
    """Tests for ensure_parent_dir function."""

    def test_creates_parent(self, tmp_path):
        """Test parent directory creation."""
        file_path = tmp_path / "subdir" / "file.txt"
        assert not file_path.parent.exists()

        result = ensure_parent_dir(file_path)

        assert file_path.parent.exists()
        assert result == file_path

    def test_existing_parent(self, tmp_path):
        """Test with existing parent."""
        file_path = tmp_path / "file.txt"

        result = ensure_parent_dir(file_path)

        assert result == file_path
