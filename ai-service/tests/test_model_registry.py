"""Tests for app.training.model_registry module.

Tests model registration, lifecycle management, metrics updates,
and stage transitions.
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.training.model_registry import (
    ModelRegistry,
    ModelStage,
    ModelType,
    ModelMetrics,
    TrainingConfig,
    RegistryDatabase,
)


class TestRegistryDatabase:
    """Test the RegistryDatabase class."""

    def setup_method(self):
        """Create temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_registry.db"
        self.db = RegistryDatabase(self.db_path)

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization(self):
        """Test database tables are created."""
        # Tables should exist
        cursor = self.db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "models" in tables
        assert "versions" in tables
        assert "tags" in tables
        assert "stage_transitions" in tables
        assert "comparisons" in tables

    def test_create_model(self):
        """Test creating a model entry."""
        self.db.create_model(
            model_id="test_model",
            name="Test Model",
            model_type=ModelType.POLICY_VALUE,
            description="A test model"
        )

        assert self.db.model_exists("test_model")
        assert not self.db.model_exists("nonexistent")

    def test_version_numbering(self):
        """Test version numbering increments correctly."""
        self.db.create_model("test_model", "Test", ModelType.POLICY_VALUE)

        assert self.db.get_next_version("test_model") == 1

        # Create a version
        self.db.create_version(
            model_id="test_model",
            version=1,
            stage=ModelStage.DEVELOPMENT,
            file_path="/path/to/model.pt",
            file_hash="abc123",
            file_size=1024,
            metrics=ModelMetrics(),
            training_config=TrainingConfig()
        )

        assert self.db.get_next_version("test_model") == 2


class TestModelRegistry:
    """Test the ModelRegistry class."""

    def setup_method(self):
        """Create temporary registry for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(
            registry_dir=Path(self.temp_dir) / "registry"
        )
        # Create a dummy model file
        self.model_file = Path(self.temp_dir) / "dummy_model.pt"
        self.model_file.write_bytes(b"fake model weights" * 100)

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_model(self):
        """Test registering a new model."""
        model_id, version = self.registry.register_model(
            name="square8_2p",
            model_path=self.model_file,
            model_type=ModelType.POLICY_VALUE,
            description="Square 8 two-player model",
            initial_stage=ModelStage.DEVELOPMENT,
        )

        assert model_id == "square8_2p"
        assert version == 1

        # Retrieve and verify
        model = self.registry.get_model(model_id, version)
        assert model is not None
        assert model.name == "square8_2p"
        assert model.stage == ModelStage.DEVELOPMENT
        assert model.model_type == ModelType.POLICY_VALUE

    def test_register_model_with_metrics(self):
        """Test registering a model with metrics."""
        metrics = ModelMetrics(
            elo=1500.0,
            elo_uncertainty=50.0,
            win_rate=0.55,
            games_played=100
        )

        model_id, version = self.registry.register_model(
            name="test_model",
            model_path=self.model_file,
            metrics=metrics,
        )

        model = self.registry.get_model(model_id, version)
        assert model.metrics.elo == 1500.0
        assert model.metrics.win_rate == 0.55
        assert model.metrics.games_played == 100

    def test_register_multiple_versions(self):
        """Test registering multiple versions of same model."""
        # Version 1
        model_id1, v1 = self.registry.register_model(
            name="evolving_model",
            model_path=self.model_file,
            model_id="evolving_model"
        )
        assert v1 == 1

        # Create a new dummy file for v2
        model_file2 = Path(self.temp_dir) / "dummy_model_v2.pt"
        model_file2.write_bytes(b"updated model weights" * 100)

        # Version 2
        model_id2, v2 = self.registry.register_model(
            name="evolving_model",
            model_path=model_file2,
            model_id="evolving_model"
        )
        assert model_id2 == model_id1
        assert v2 == 2

    def test_promote_model(self):
        """Test promoting model through stages."""
        model_id, version = self.registry.register_model(
            name="promotable_model",
            model_path=self.model_file,
            initial_stage=ModelStage.DEVELOPMENT,
        )

        # Promote to staging
        self.registry.promote(model_id, version, ModelStage.STAGING, reason="Passed tests")

        model = self.registry.get_model(model_id, version)
        assert model.stage == ModelStage.STAGING

        # Promote to production
        self.registry.promote(model_id, version, ModelStage.PRODUCTION, reason="Evaluation passed")

        model = self.registry.get_model(model_id, version)
        assert model.stage == ModelStage.PRODUCTION

    def test_promote_invalid_transition(self):
        """Test that invalid stage transitions are rejected."""
        model_id, version = self.registry.register_model(
            name="test_model",
            model_path=self.model_file,
            initial_stage=ModelStage.DEVELOPMENT,
        )

        # Cannot go directly from DEVELOPMENT to PRODUCTION (must go through STAGING)
        # Actually, looking at the allowed transitions, DEVELOPMENT can only go to STAGING, ARCHIVED, or REJECTED
        with pytest.raises(ValueError, match="Cannot transition"):
            self.registry.promote(model_id, version, ModelStage.PRODUCTION)

    def test_update_metrics(self):
        """Test updating model metrics."""
        model_id, version = self.registry.register_model(
            name="test_model",
            model_path=self.model_file,
        )

        # Update metrics
        new_metrics = ModelMetrics(
            elo=1600.0,
            elo_uncertainty=30.0,
            win_rate=0.65,
            games_played=250
        )
        self.registry.update_metrics(model_id, version, new_metrics)

        model = self.registry.get_model(model_id, version)
        assert model.metrics.elo == 1600.0
        assert model.metrics.games_played == 250

    def test_get_latest_version(self):
        """Test getting latest version when version not specified."""
        model_id = "multi_version"

        # Create multiple versions
        self.registry.register_model(
            name="Multi Version",
            model_path=self.model_file,
            model_id=model_id
        )

        model_file2 = Path(self.temp_dir) / "dummy_v2.pt"
        model_file2.write_bytes(b"v2 weights" * 100)
        self.registry.register_model(
            name="Multi Version",
            model_path=model_file2,
            model_id=model_id
        )

        # Get without version should return latest (v2)
        model = self.registry.get_model(model_id)
        assert model.version == 2

    def test_list_models_by_stage(self):
        """Test listing models filtered by stage."""
        # Create models in different stages
        m1_id, m1_v = self.registry.register_model(
            name="dev_model",
            model_path=self.model_file,
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="dev_model"
        )

        model_file2 = Path(self.temp_dir) / "staging_model.pt"
        model_file2.write_bytes(b"staging weights" * 100)
        m2_id, m2_v = self.registry.register_model(
            name="staging_model",
            model_path=model_file2,
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="staging_model"
        )
        self.registry.promote(m2_id, m2_v, ModelStage.STAGING)

        # List by stage
        dev_models = self.registry.list_models(stage=ModelStage.DEVELOPMENT)
        staging_models = self.registry.list_models(stage=ModelStage.STAGING)

        assert len(dev_models) == 1
        assert dev_models[0]['model_id'] == "dev_model"

        assert len(staging_models) == 1
        assert staging_models[0]['model_id'] == "staging_model"

    def test_get_production_model(self):
        """Test getting the current production model."""
        # Initially no production model
        assert self.registry.get_production_model() is None

        # Create and promote to production
        model_id, version = self.registry.register_model(
            name="prod_model",
            model_path=self.model_file,
            initial_stage=ModelStage.DEVELOPMENT,
        )
        self.registry.promote(model_id, version, ModelStage.STAGING)
        self.registry.promote(model_id, version, ModelStage.PRODUCTION)

        prod = self.registry.get_production_model()
        assert prod is not None
        assert prod.model_id == "prod_model"
        assert prod.stage == ModelStage.PRODUCTION

    def test_production_replacement_archives_old(self):
        """Test that promoting new model to production archives old one."""
        # First production model
        m1_id, m1_v = self.registry.register_model(
            name="first_prod",
            model_path=self.model_file,
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="first_prod"
        )
        self.registry.promote(m1_id, m1_v, ModelStage.STAGING)
        self.registry.promote(m1_id, m1_v, ModelStage.PRODUCTION)

        # Second production model
        model_file2 = Path(self.temp_dir) / "second_prod.pt"
        model_file2.write_bytes(b"second prod weights" * 100)
        m2_id, m2_v = self.registry.register_model(
            name="second_prod",
            model_path=model_file2,
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="second_prod"
        )
        self.registry.promote(m2_id, m2_v, ModelStage.STAGING)
        self.registry.promote(m2_id, m2_v, ModelStage.PRODUCTION)

        # First should be archived
        first = self.registry.get_model(m1_id, m1_v)
        assert first.stage == ModelStage.ARCHIVED

        # Second should be production
        second = self.registry.get_model(m2_id, m2_v)
        assert second.stage == ModelStage.PRODUCTION


class TestModelMetrics:
    """Test ModelMetrics dataclass."""

    def test_metrics_defaults(self):
        """Test default metric values."""
        metrics = ModelMetrics()
        assert metrics.elo is None
        assert metrics.games_played == 0

    def test_metrics_roundtrip(self):
        """Test metrics serialization roundtrip."""
        metrics = ModelMetrics(
            elo=1500.0,
            elo_uncertainty=50.0,
            win_rate=0.55,
            draw_rate=0.1,
            games_played=100
        )

        d = metrics.to_dict()
        restored = ModelMetrics.from_dict(d)

        assert restored.elo == metrics.elo
        assert restored.win_rate == metrics.win_rate
        assert restored.games_played == metrics.games_played


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_config_defaults(self):
        """Test default training config values."""
        config = TrainingConfig()
        assert config.learning_rate == 0.001
        assert config.batch_size == 256
        assert config.optimizer == "adam"

    def test_config_roundtrip(self):
        """Test config serialization roundtrip."""
        config = TrainingConfig(
            learning_rate=0.0005,
            batch_size=512,
            epochs=200,
            num_residual_blocks=15
        )

        d = config.to_dict()
        restored = TrainingConfig.from_dict(d)

        assert restored.learning_rate == config.learning_rate
        assert restored.batch_size == config.batch_size
        assert restored.num_residual_blocks == config.num_residual_blocks
