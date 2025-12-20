"""Tests for model lifecycle management."""
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.training.model_lifecycle import (
    FullMaintenanceResult,
    MaintenanceResult,
    ModelLifecycleManager,
    RetentionPolicy,
)


@pytest.fixture
def temp_model_dir():
    """Create a temporary model directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def policy():
    """Create a test retention policy."""
    return RetentionPolicy(
        max_models_per_config=10,
        keep_top_by_elo=5,
        keep_latest_production=2,
        archive_after_days=7,
        delete_archived_after_days=14,
        min_models_to_keep=3,
        cooldown_hours=0.0,  # No cooldown for tests
    )


@pytest.fixture
def manager(temp_model_dir, policy):
    """Create a ModelLifecycleManager with temp directory."""
    return ModelLifecycleManager(
        model_dir=temp_model_dir,
        elo_db_path=temp_model_dir / "test_elo.db",
        policy=policy,
    )


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_default_values(self):
        """Default policy should have sensible values."""
        policy = RetentionPolicy()
        assert policy.max_models_per_config == 100
        assert policy.keep_top_by_elo == 25
        assert policy.keep_latest_production == 5
        assert policy.archive_after_days == 30
        assert policy.delete_archived_after_days == 90
        assert policy.min_models_to_keep == 10
        assert policy.cooldown_hours == 1.0

    def test_custom_values(self):
        """Custom policy values should be respected."""
        policy = RetentionPolicy(
            max_models_per_config=50,
            keep_top_by_elo=10,
            archive_after_days=14,
        )
        assert policy.max_models_per_config == 50
        assert policy.keep_top_by_elo == 10
        assert policy.archive_after_days == 14


class TestMaintenanceResult:
    """Tests for MaintenanceResult dataclass."""

    def test_basic_result(self):
        """Basic maintenance result creation."""
        result = MaintenanceResult(
            config_key="square8_2p",
            models_before=20,
            models_after=10,
            archived=8,
            deleted=2,
        )
        assert result.config_key == "square8_2p"
        assert result.models_before == 20
        assert result.models_after == 10
        assert result.archived == 8
        assert result.deleted == 2
        assert result.errors == []
        assert result.timestamp > 0

    def test_with_errors(self):
        """Maintenance result with errors."""
        result = MaintenanceResult(
            config_key="hex8_3p",
            models_before=5,
            models_after=5,
            archived=0,
            deleted=0,
            errors=["File not found", "Permission denied"],
        )
        assert len(result.errors) == 2


class TestFullMaintenanceResult:
    """Tests for FullMaintenanceResult dataclass."""

    def test_default_values(self):
        """Default full result should have zero values."""
        result = FullMaintenanceResult()
        assert result.total_archived == 0
        assert result.total_deleted == 0
        assert result.total_errors == 0
        assert result.per_config_results == {}
        assert result.duration_seconds == 0.0


class TestModelLifecycleManager:
    """Tests for ModelLifecycleManager class."""

    def test_initialization(self, manager, temp_model_dir, policy):
        """Manager should initialize with correct paths."""
        assert manager.model_dir == temp_model_dir
        assert manager.policy == policy
        assert (temp_model_dir / "archived").exists()

    def test_default_model_dir(self):
        """Manager should use default model dir if not specified."""
        with patch.dict("os.environ", {}, clear=True):
            manager = ModelLifecycleManager()
            assert manager.model_dir.name == "models"

    def test_env_model_dir(self, temp_model_dir):
        """Manager should use RINGRIFT_MODEL_DIR env var."""
        with patch.dict("os.environ", {"RINGRIFT_MODEL_DIR": str(temp_model_dir)}):
            manager = ModelLifecycleManager()
            assert manager.model_dir == temp_model_dir

    def test_get_models_for_config_empty(self, manager):
        """Should return empty list when no models exist."""
        models = manager.get_models_for_config("square8_2p")
        assert models == []

    def test_get_models_for_config_finds_models(self, manager, temp_model_dir):
        """Should find models matching config pattern."""
        # Create test model files
        (temp_model_dir / "model_square8_2p_gen1.pth").touch()
        (temp_model_dir / "model_square8_2p_gen2.pth").touch()
        (temp_model_dir / "model_hex8_2p_gen1.pth").touch()

        models = manager.get_models_for_config("square8_2p")
        assert len(models) == 2
        assert all("square8_2p" in str(m) for m in models)

    def test_get_models_normalizes_config_key(self, manager, temp_model_dir):
        """Should normalize config key aliases."""
        (temp_model_dir / "model_square8_2p_gen1.pth").touch()

        # Use alias
        models = manager.get_models_for_config("sq8_2p")
        assert len(models) == 1

    def test_get_archived_models_empty(self, manager):
        """Should return empty list when no archived models."""
        archived = manager.get_archived_models("square8_2p")
        assert archived == []

    def test_get_archived_models_finds_models(self, manager, temp_model_dir):
        """Should find archived models for config."""
        archive_dir = temp_model_dir / "archived" / "square8_2p"
        archive_dir.mkdir(parents=True)
        (archive_dir / "old_model.pth").touch()
        (archive_dir / "older_model.pth").touch()

        archived = manager.get_archived_models("square8_2p")
        assert len(archived) == 2

    def test_archive_model_success(self, manager, temp_model_dir):
        """Should successfully archive a model."""
        model_path = temp_model_dir / "model_to_archive.pth"
        model_path.touch()

        result = manager.archive_model(model_path, "square8_2p", "test")
        assert result is True
        assert not model_path.exists()
        assert (temp_model_dir / "archived" / "square8_2p" / "model_to_archive.pth").exists()

    def test_archive_model_not_found(self, manager, temp_model_dir):
        """Should return False for non-existent model."""
        model_path = temp_model_dir / "nonexistent.pth"
        result = manager.archive_model(model_path, "square8_2p", "test")
        assert result is False

    def test_delete_archived_model_success(self, manager, temp_model_dir):
        """Should successfully delete archived model."""
        archive_dir = temp_model_dir / "archived" / "square8_2p"
        archive_dir.mkdir(parents=True)
        model_path = archive_dir / "old_model.pth"
        model_path.touch()

        result = manager.delete_archived_model(model_path)
        assert result is True
        assert not model_path.exists()

    def test_delete_archived_model_not_found(self, manager, temp_model_dir):
        """Should return True for already deleted model."""
        model_path = temp_model_dir / "nonexistent.pth"
        result = manager.delete_archived_model(model_path)
        assert result is True

    def test_cleanup_old_archives(self, manager, temp_model_dir, policy):
        """Should delete archived models older than retention period."""
        archive_dir = temp_model_dir / "archived" / "square8_2p"
        archive_dir.mkdir(parents=True)

        # Create old archived model
        old_model = archive_dir / "old_model.pth"
        old_model.touch()
        # Set modification time to 30 days ago
        old_time = (datetime.now() - timedelta(days=30)).timestamp()
        import os
        os.utime(old_model, (old_time, old_time))

        # Create recent archived model
        recent_model = archive_dir / "recent_model.pth"
        recent_model.touch()

        deleted = manager.cleanup_old_archives("square8_2p")
        assert deleted == 1
        assert not old_model.exists()
        assert recent_model.exists()

    def test_check_config_respects_cooldown(self, temp_model_dir):
        """Should skip maintenance if within cooldown period."""
        policy = RetentionPolicy(cooldown_hours=24.0)
        manager = ModelLifecycleManager(
            model_dir=temp_model_dir,
            policy=policy,
        )

        # First run should work
        manager.check_config("square8_2p", force=False)

        # Second run should be skipped due to cooldown
        result2 = manager.check_config("square8_2p", force=False)
        assert result2.archived == 0
        assert result2.deleted == 0

    def test_check_config_force_ignores_cooldown(self, temp_model_dir):
        """Force flag should ignore cooldown."""
        policy = RetentionPolicy(cooldown_hours=24.0)
        manager = ModelLifecycleManager(
            model_dir=temp_model_dir,
            policy=policy,
        )

        # First run
        manager.check_config("square8_2p", force=False)

        # Force run should still work
        result = manager.check_config("square8_2p", force=True)
        # Won't archive/delete without models, but should run
        assert result.models_before == 0

    def test_run_maintenance_all_configs(self, manager):
        """Should run maintenance for all canonical configs."""
        result = manager.run_maintenance(force=True)

        # Should have results for all 12 configs
        assert len(result.per_config_results) == 12
        assert "square8_2p" in result.per_config_results
        assert "hexagonal_4p" in result.per_config_results
        assert result.duration_seconds >= 0

    def test_run_maintenance_specific_configs(self, manager):
        """Should only run maintenance for specified configs."""
        result = manager.run_maintenance(
            configs=["square8_2p", "hex8_3p"],
            force=True,
        )

        assert len(result.per_config_results) == 2
        assert "square8_2p" in result.per_config_results
        assert "hex8_3p" in result.per_config_results

    def test_get_status(self, manager, temp_model_dir):
        """Should return status for all configs."""
        # Create some models
        (temp_model_dir / "model_square8_2p_gen1.pth").touch()
        (temp_model_dir / "model_square8_2p_gen2.pth").touch()

        archive_dir = temp_model_dir / "archived" / "hex8_3p"
        archive_dir.mkdir(parents=True)
        (archive_dir / "archived_model.pth").touch()

        status = manager.get_status()

        assert len(status) == 12
        assert status["square8_2p"]["active_models"] == 2
        assert status["square8_2p"]["archived_models"] == 0
        assert status["hex8_3p"]["active_models"] == 0
        assert status["hex8_3p"]["archived_models"] == 1

    def test_get_status_needs_culling(self, manager, temp_model_dir, policy):
        """Should correctly identify configs needing culling."""
        # Create models over threshold
        for i in range(policy.max_models_per_config + 5):
            (temp_model_dir / f"model_square8_2p_gen{i}.pth").touch()

        status = manager.get_status()
        assert status["square8_2p"]["needs_culling"] is True
        assert status["hex8_2p"]["needs_culling"] is False


class TestIntegrationWithCuller:
    """Tests for integration with ModelCullingController."""

    def test_culler_import_failure_handled(self, manager, temp_model_dir):
        """Should handle missing ModelCullingController gracefully."""
        with patch.object(manager, "_get_culler", return_value=None):
            # Create enough models to trigger culling attempt
            for i in range(15):
                (temp_model_dir / f"model_square8_2p_gen{i}.pth").touch()

            # Should not raise even without culler
            result = manager.check_config("square8_2p", force=True)
            assert result.errors == []

    def test_culler_exception_handled(self, manager, temp_model_dir):
        """Should handle culler exceptions gracefully."""
        mock_culler = MagicMock()
        mock_culler.check_and_cull.side_effect = Exception("Culler error")

        with patch.object(manager, "_get_culler", return_value=mock_culler):
            for i in range(15):
                (temp_model_dir / f"model_square8_2p_gen{i}.pth").touch()

            result = manager.check_config("square8_2p", force=True)
            assert len(result.errors) == 1
            assert "Culling error" in result.errors[0]


class TestRunModelMaintenance:
    """Tests for run_model_maintenance convenience function."""

    def test_convenience_function(self, temp_model_dir):
        """Convenience function should work."""
        from app.training.model_lifecycle import run_model_maintenance

        result = run_model_maintenance(
            model_dir=str(temp_model_dir),
            force=True,
        )

        assert isinstance(result, FullMaintenanceResult)
        assert len(result.per_config_results) == 12
