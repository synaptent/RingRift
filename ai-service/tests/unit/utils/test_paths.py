"""Tests for app/utils/paths.py - centralized path definitions.

These tests verify the path infrastructure that underpins the entire AI service.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestProjectRoot:
    """Tests for project root detection."""

    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a Path object."""
        from app.utils.paths import get_project_root

        root = get_project_root()
        assert isinstance(root, Path)

    def test_get_project_root_is_ai_service(self):
        """Test that project root is the ai-service directory."""
        from app.utils.paths import get_project_root

        root = get_project_root()
        assert root.name == "ai-service"

    def test_ai_service_root_constant(self):
        """Test that AI_SERVICE_ROOT matches get_project_root."""
        from app.utils.paths import AI_SERVICE_ROOT, get_project_root

        assert AI_SERVICE_ROOT == get_project_root()

    def test_ai_service_root_exists(self):
        """Test that AI_SERVICE_ROOT directory exists."""
        from app.utils.paths import AI_SERVICE_ROOT

        assert AI_SERVICE_ROOT.exists()
        assert AI_SERVICE_ROOT.is_dir()


class TestPrimaryDirectories:
    """Tests for primary directory constants."""

    def test_data_dir_path(self):
        """Test DATA_DIR is correctly defined."""
        from app.utils.paths import AI_SERVICE_ROOT, DATA_DIR

        assert DATA_DIR == AI_SERVICE_ROOT / "data"

    def test_games_dir_path(self):
        """Test GAMES_DIR is correctly defined."""
        from app.utils.paths import DATA_DIR, GAMES_DIR

        assert GAMES_DIR == DATA_DIR / "games"

    def test_training_dir_path(self):
        """Test TRAINING_DIR is correctly defined."""
        from app.utils.paths import DATA_DIR, TRAINING_DIR

        assert TRAINING_DIR == DATA_DIR / "training"

    def test_models_dir_path(self):
        """Test MODELS_DIR is correctly defined."""
        from app.utils.paths import AI_SERVICE_ROOT, MODELS_DIR

        assert MODELS_DIR == AI_SERVICE_ROOT / "models"

    def test_logs_dir_path(self):
        """Test LOGS_DIR is correctly defined."""
        from app.utils.paths import AI_SERVICE_ROOT, LOGS_DIR

        assert LOGS_DIR == AI_SERVICE_ROOT / "logs"

    def test_config_dir_path(self):
        """Test CONFIG_DIR is correctly defined."""
        from app.utils.paths import AI_SERVICE_ROOT, CONFIG_DIR

        assert CONFIG_DIR == AI_SERVICE_ROOT / "config"

    def test_scripts_dir_path(self):
        """Test SCRIPTS_DIR is correctly defined."""
        from app.utils.paths import AI_SERVICE_ROOT, SCRIPTS_DIR

        assert SCRIPTS_DIR == AI_SERVICE_ROOT / "scripts"

    def test_coordination_dir_path(self):
        """Test COORDINATION_DIR is correctly defined."""
        from app.utils.paths import COORDINATION_DIR, DATA_DIR

        assert COORDINATION_DIR == DATA_DIR / "coordination"


class TestDatabasePaths:
    """Tests for database path constants."""

    def test_unified_elo_db_path(self):
        """Test UNIFIED_ELO_DB is correctly defined."""
        from app.utils.paths import DATA_DIR, UNIFIED_ELO_DB

        assert UNIFIED_ELO_DB == DATA_DIR / "unified_elo.db"

    def test_work_queue_db_path(self):
        """Test WORK_QUEUE_DB is correctly defined."""
        from app.utils.paths import DATA_DIR, WORK_QUEUE_DB

        assert WORK_QUEUE_DB == DATA_DIR / "work_queue.db"

    def test_training_metrics_db_path(self):
        """Test TRAINING_METRICS_DB is correctly defined."""
        from app.utils.paths import METRICS_DIR, TRAINING_METRICS_DB

        assert TRAINING_METRICS_DB == METRICS_DIR / "training_metrics.db"


class TestConfigurationFiles:
    """Tests for configuration file path constants."""

    def test_promotion_history_file_path(self):
        """Test PROMOTION_HISTORY_FILE is correctly defined."""
        from app.utils.paths import PROMOTION_DIR, PROMOTION_HISTORY_FILE

        assert PROMOTION_HISTORY_FILE == PROMOTION_DIR / "promoted_models.json"

    def test_gauntlet_results_file_path(self):
        """Test GAUNTLET_RESULTS_FILE is correctly defined."""
        from app.utils.paths import DATA_DIR, GAUNTLET_RESULTS_FILE

        assert GAUNTLET_RESULTS_FILE == DATA_DIR / "aggregated_gauntlet_results.json"

    def test_model_registry_file_path(self):
        """Test MODEL_REGISTRY_FILE is correctly defined."""
        from app.utils.paths import DATA_DIR, MODEL_REGISTRY_FILE

        assert MODEL_REGISTRY_FILE == DATA_DIR / "model_registry.json"


class TestHelperFunctions:
    """Tests for path helper functions."""

    def test_get_models_dir_no_board_type(self):
        """Test get_models_dir without board type."""
        from app.utils.paths import MODELS_DIR, get_models_dir

        assert get_models_dir() == MODELS_DIR

    def test_get_models_dir_with_board_type(self):
        """Test get_models_dir with board type."""
        from app.utils.paths import MODELS_DIR, get_models_dir

        result = get_models_dir("square8")
        assert result == MODELS_DIR / "square8"

    def test_get_data_dir_no_subdir(self):
        """Test get_data_dir without subdirectory."""
        from app.utils.paths import DATA_DIR, get_data_dir

        assert get_data_dir() == DATA_DIR

    def test_get_data_dir_with_subdir(self):
        """Test get_data_dir with subdirectory."""
        from app.utils.paths import DATA_DIR, get_data_dir

        result = get_data_dir("custom")
        assert result == DATA_DIR / "custom"

    def test_get_games_db_path(self):
        """Test get_games_db_path function."""
        from app.utils.paths import GAMES_DIR, get_games_db_path

        result = get_games_db_path("hex8_2p")
        assert result == GAMES_DIR / "hex8_2p.db"

    def test_get_selfplay_db_path(self):
        """Test get_selfplay_db_path function."""
        from app.utils.paths import SELFPLAY_DIR, get_selfplay_db_path

        result = get_selfplay_db_path("square8_4p")
        assert result == SELFPLAY_DIR / "selfplay_square8_4p.db"

    def test_get_training_npz_path(self):
        """Test get_training_npz_path function."""
        from app.utils.paths import TRAINING_DIR, get_training_npz_path

        result = get_training_npz_path("hexagonal_3p")
        assert result == TRAINING_DIR / "hexagonal_3p.npz"

    def test_get_model_path_no_board_type(self):
        """Test get_model_path without board type."""
        from app.utils.paths import MODELS_DIR, get_model_path

        result = get_model_path("best_model.pth")
        assert result == MODELS_DIR / "best_model.pth"

    def test_get_model_path_with_board_type(self):
        """Test get_model_path with board type."""
        from app.utils.paths import MODELS_DIR, get_model_path

        result = get_model_path("checkpoint.pth", "hex8")
        assert result == MODELS_DIR / "hex8" / "checkpoint.pth"

    def test_get_log_path_no_subdir(self):
        """Test get_log_path without subdirectory."""
        from app.utils.paths import LOGS_DIR, get_log_path

        result = get_log_path("app.log")
        assert result == LOGS_DIR / "app.log"

    def test_get_log_path_with_subdir(self):
        """Test get_log_path with subdirectory."""
        from app.utils.paths import LOGS_DIR, get_log_path

        result = get_log_path("train.log", "training")
        assert result == LOGS_DIR / "training" / "train.log"


class TestEnsureDirFunctions:
    """Tests for directory creation helpers."""

    def test_ensure_dir_creates_directory(self):
        """Test ensure_dir creates a directory."""
        from app.utils.paths import ensure_dir

        with tempfile.TemporaryDirectory() as tmp:
            test_dir = Path(tmp) / "new_dir"
            assert not test_dir.exists()

            result = ensure_dir(test_dir)

            assert test_dir.exists()
            assert test_dir.is_dir()
            assert result == test_dir  # Returns same path

    def test_ensure_dir_creates_nested_directories(self):
        """Test ensure_dir creates nested directories."""
        from app.utils.paths import ensure_dir

        with tempfile.TemporaryDirectory() as tmp:
            test_dir = Path(tmp) / "a" / "b" / "c"
            assert not test_dir.exists()

            ensure_dir(test_dir)

            assert test_dir.exists()

    def test_ensure_dir_idempotent(self):
        """Test ensure_dir is idempotent (doesn't fail on existing dir)."""
        from app.utils.paths import ensure_dir

        with tempfile.TemporaryDirectory() as tmp:
            test_dir = Path(tmp) / "existing"
            test_dir.mkdir()

            # Should not raise
            ensure_dir(test_dir)
            assert test_dir.exists()

    def test_ensure_parent_dir_creates_parent(self):
        """Test ensure_parent_dir creates parent directory."""
        from app.utils.paths import ensure_parent_dir

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "new_parent" / "file.txt"
            assert not file_path.parent.exists()

            result = ensure_parent_dir(file_path)

            assert file_path.parent.exists()
            assert not file_path.exists()  # File itself not created
            assert result == file_path  # Returns same path

    def test_ensure_parent_dir_nested(self):
        """Test ensure_parent_dir creates nested parents."""
        from app.utils.paths import ensure_parent_dir

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "a" / "b" / "c" / "file.txt"

            ensure_parent_dir(file_path)

            assert file_path.parent.exists()


class TestEnvironmentOverrides:
    """Tests for environment variable path overrides."""

    def test_get_env_path_with_env_set(self):
        """Test get_env_path returns env value when set."""
        from app.utils.paths import get_env_path

        with mock.patch.dict(os.environ, {"TEST_PATH": "/custom/path"}):
            result = get_env_path("TEST_PATH", Path("/default"))
            assert result == Path("/custom/path")

    def test_get_env_path_without_env_set(self):
        """Test get_env_path returns default when env not set."""
        from app.utils.paths import get_env_path

        # Ensure env var is not set
        os.environ.pop("NONEXISTENT_VAR", None)

        result = get_env_path("NONEXISTENT_VAR", Path("/default/path"))
        assert result == Path("/default/path")


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_defined(self):
        """Test that all items in __all__ are actually defined."""
        from app.utils import paths

        for name in paths.__all__:
            assert hasattr(paths, name), f"Missing export: {name}"

    def test_key_exports_present(self):
        """Test that key exports are in __all__."""
        from app.utils.paths import __all__

        key_exports = [
            "AI_SERVICE_ROOT",
            "DATA_DIR",
            "MODELS_DIR",
            "GAMES_DIR",
            "LOGS_DIR",
            "get_project_root",
            "get_models_dir",
            "ensure_dir",
        ]
        for name in key_exports:
            assert name in __all__, f"Missing from __all__: {name}"


class TestPathConsistency:
    """Tests for path relationships and consistency."""

    def test_all_data_subdirs_are_under_data_dir(self):
        """Test that all data subdirectories are children of DATA_DIR."""
        from app.utils.paths import (
            COORDINATION_DIR,
            DATA_DIR,
            GAMES_DIR,
            HOLDOUT_DIR,
            METRICS_DIR,
            QUARANTINE_DIR,
            SELFPLAY_DIR,
            TRAINING_DIR,
        )

        subdirs = [
            GAMES_DIR,
            TRAINING_DIR,
            SELFPLAY_DIR,
            METRICS_DIR,
            HOLDOUT_DIR,
            QUARANTINE_DIR,
            COORDINATION_DIR,
        ]

        for subdir in subdirs:
            assert DATA_DIR in subdir.parents or subdir.parent == DATA_DIR

    def test_all_log_subdirs_are_under_logs_dir(self):
        """Test that all log subdirectories are children of LOGS_DIR."""
        from app.utils.paths import (
            DEPLOYMENT_LOGS_DIR,
            LOGS_DIR,
            TRAINING_LOGS_DIR,
        )

        assert TRAINING_LOGS_DIR.parent == LOGS_DIR
        assert DEPLOYMENT_LOGS_DIR.parent == LOGS_DIR

    def test_all_model_subdirs_are_under_models_dir(self):
        """Test that all model subdirectories are children of MODELS_DIR."""
        from app.utils.paths import ARCHIVED_MODELS_DIR, MODELS_DIR

        assert ARCHIVED_MODELS_DIR.parent == MODELS_DIR
