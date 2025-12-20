"""Tests for the model registry module.

Comprehensive tests for:
- ModelStage, ValidationStatus, ModelType enums
- ModelMetrics and TrainingConfig dataclasses
- RegistryDatabase SQLite operations
- ModelRegistry lifecycle management
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


class TestEnums:
    """Test model registry enums."""

    def test_model_stage_values(self):
        """Test ModelStage enum values."""
        from app.training.model_registry import ModelStage

        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"
        assert ModelStage.REJECTED.value == "rejected"

    def test_model_stage_iteration(self):
        """Test that all ModelStage values can be iterated."""
        from app.training.model_registry import ModelStage

        stages = list(ModelStage)
        assert len(stages) == 5

    def test_validation_status_values(self):
        """Test ValidationStatus enum values."""
        from app.training.model_registry import ValidationStatus

        assert ValidationStatus.PENDING.value == "pending"
        assert ValidationStatus.QUEUED.value == "queued"
        assert ValidationStatus.RUNNING.value == "running"
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.SKIPPED.value == "skipped"

    def test_model_type_values(self):
        """Test ModelType enum values."""
        from app.training.model_registry import ModelType

        assert ModelType.POLICY_VALUE.value == "policy_value"
        assert ModelType.ENSEMBLE.value == "ensemble"
        assert ModelType.COMPRESSED.value == "compressed"
        assert ModelType.EXPERIMENTAL.value == "experimental"
        assert ModelType.HEURISTIC.value == "heuristic"


class TestModelMetrics:
    """Test ModelMetrics dataclass."""

    def test_default_values(self):
        """Test ModelMetrics default values."""
        from app.training.model_registry import ModelMetrics

        metrics = ModelMetrics()

        assert metrics.elo is None
        assert metrics.win_rate is None
        assert metrics.games_played == 0
        assert metrics.policy_accuracy is None

    def test_custom_values(self):
        """Test ModelMetrics with custom values."""
        from app.training.model_registry import ModelMetrics

        metrics = ModelMetrics(
            elo=1650.0,
            elo_uncertainty=50.0,
            win_rate=0.65,
            games_played=100,
            policy_accuracy=0.75,
        )

        assert metrics.elo == 1650.0
        assert metrics.elo_uncertainty == 50.0
        assert metrics.win_rate == 0.65
        assert metrics.games_played == 100
        assert metrics.policy_accuracy == 0.75

    def test_to_dict(self):
        """Test ModelMetrics serialization."""
        from app.training.model_registry import ModelMetrics

        metrics = ModelMetrics(elo=1600.0, games_played=50)
        d = metrics.to_dict()

        assert d["elo"] == 1600.0
        assert d["games_played"] == 50
        assert "win_rate" in d

    def test_from_dict(self):
        """Test ModelMetrics deserialization."""
        from app.training.model_registry import ModelMetrics

        d = {"elo": 1700.0, "games_played": 75, "win_rate": 0.60}
        metrics = ModelMetrics.from_dict(d)

        assert metrics.elo == 1700.0
        assert metrics.games_played == 75
        assert metrics.win_rate == 0.60


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test TrainingConfig default values."""
        from app.training.model_registry import TrainingConfig

        config = TrainingConfig()

        assert config.learning_rate == 0.001
        assert config.batch_size == 256
        assert config.epochs == 100
        assert config.optimizer == "adam"
        assert config.architecture == "resnet"

    def test_custom_values(self):
        """Test TrainingConfig with custom values."""
        from app.training.model_registry import TrainingConfig

        config = TrainingConfig(
            learning_rate=0.0005,
            batch_size=128,
            epochs=50,
            optimizer="sgd",
            num_residual_blocks=15,
        )

        assert config.learning_rate == 0.0005
        assert config.batch_size == 128
        assert config.epochs == 50
        assert config.optimizer == "sgd"
        assert config.num_residual_blocks == 15

    def test_to_dict(self):
        """Test TrainingConfig serialization."""
        from app.training.model_registry import TrainingConfig

        config = TrainingConfig(learning_rate=0.002)
        d = config.to_dict()

        assert d["learning_rate"] == 0.002
        assert "batch_size" in d
        assert "optimizer" in d

    def test_from_dict(self):
        """Test TrainingConfig deserialization."""
        from app.training.model_registry import TrainingConfig

        d = {"learning_rate": 0.003, "batch_size": 512, "epochs": 200}
        config = TrainingConfig.from_dict(d)

        assert config.learning_rate == 0.003
        assert config.batch_size == 512
        assert config.epochs == 200

    def test_from_dict_extra_fields(self):
        """Test TrainingConfig handles unknown fields gracefully."""
        from app.training.model_registry import TrainingConfig

        d = {
            "learning_rate": 0.001,
            "unknown_field": "value",
            "another_unknown": 123,
        }
        config = TrainingConfig.from_dict(d)

        assert config.learning_rate == 0.001
        assert "unknown_field" in config.extra_config
        assert config.extra_config["unknown_field"] == "value"


class TestRegistryDatabase:
    """Test RegistryDatabase SQLite operations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database."""
        from app.training.model_registry import RegistryDatabase

        db_path = tmp_path / "test_registry.db"
        return RegistryDatabase(db_path)

    def test_database_creation(self, tmp_path):
        """Test database file is created."""
        from app.training.model_registry import RegistryDatabase

        db_path = tmp_path / "new_registry.db"
        db = RegistryDatabase(db_path)

        assert db_path.exists()

    def test_tables_created(self, temp_db):
        """Test all required tables are created."""
        cursor = temp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "models" in tables
        assert "versions" in tables
        assert "tags" in tables
        assert "stage_transitions" in tables
        assert "comparisons" in tables


class TestModelRegistry:
    """Test ModelRegistry class."""

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create a temporary model registry."""
        from app.training.model_registry import ModelRegistry

        registry_dir = tmp_path / "registry"

        return ModelRegistry(registry_dir=registry_dir)

    def test_registry_creation(self, tmp_path):
        """Test ModelRegistry can be instantiated."""
        from app.training.model_registry import ModelRegistry

        registry_dir = tmp_path / "registry"

        registry = ModelRegistry(registry_dir=registry_dir)

        assert registry is not None
        assert registry_dir.exists()

    def test_list_models_empty(self, temp_registry):
        """Test listing models when registry is empty."""
        models = temp_registry.list_models()

        assert isinstance(models, list)
        assert len(models) == 0

    def test_get_nonexistent_model(self, temp_registry):
        """Test getting a model that doesn't exist."""
        model = temp_registry.get_model("nonexistent_id")

        assert model is None


class TestModelVersion:
    """Test ModelVersion dataclass."""

    def test_model_version_creation(self):
        """Test creating a ModelVersion."""
        from app.training.model_registry import (
            ModelVersion,
            ModelType,
            ModelStage,
            ModelMetrics,
            TrainingConfig,
        )

        now = datetime.now()
        version = ModelVersion(
            model_id="test_model_v1",
            version=1,
            name="Test Model",
            model_type=ModelType.POLICY_VALUE,
            stage=ModelStage.DEVELOPMENT,
            created_at=now,
            updated_at=now,
            file_path="/path/to/model.pt",
            file_hash="abc123",
            file_size_bytes=1024000,
            metrics=ModelMetrics(),
            training_config=TrainingConfig(),
        )

        assert version.model_id == "test_model_v1"
        assert version.version == 1
        assert version.stage == ModelStage.DEVELOPMENT

    def test_model_version_to_dict(self):
        """Test ModelVersion serialization."""
        from app.training.model_registry import (
            ModelVersion,
            ModelType,
            ModelStage,
            ModelMetrics,
            TrainingConfig,
        )

        now = datetime.now()
        version = ModelVersion(
            model_id="test_model_v1",
            version=1,
            name="Test Model",
            model_type=ModelType.POLICY_VALUE,
            stage=ModelStage.STAGING,
            created_at=now,
            updated_at=now,
            file_path="/path/to/model.pt",
            file_hash="abc123",
            file_size_bytes=1024000,
            metrics=ModelMetrics(elo=1600.0),
            training_config=TrainingConfig(),
        )

        d = version.to_dict()

        assert d["model_id"] == "test_model_v1"
        assert d["model_type"] == "policy_value"
        assert d["stage"] == "staging"
        assert isinstance(d["created_at"], str)  # ISO format


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_main_exports(self):
        """Test importing main exports from model_registry."""
        from app.training.model_registry import (
            ModelRegistry,
            ModelStage,
            ValidationStatus,
            ModelType,
            ModelMetrics,
            TrainingConfig,
            ModelVersion,
            RegistryDatabase,
        )

        assert ModelRegistry is not None
        assert ModelStage is not None
        assert ValidationStatus is not None
        assert ModelType is not None
        assert ModelMetrics is not None
        assert TrainingConfig is not None
        assert ModelVersion is not None
        assert RegistryDatabase is not None
