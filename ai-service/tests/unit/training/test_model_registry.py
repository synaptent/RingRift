"""Tests for Model Registry.

Tests the model registry, versioning, and auto-promotion functionality.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from app.training.model_registry import (
    AutoPromoter,
    ModelMetrics,
    ModelRegistry,
    ModelStage,
    ModelType,
    ModelVersion,
    RegistryDatabase,
    TrainingConfig,
)


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_default_values(self):
        """Should have None/0 defaults."""
        metrics = ModelMetrics()
        assert metrics.elo is None
        assert metrics.games_played == 0
        assert metrics.win_rate is None
        assert metrics.policy_accuracy is None

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = ModelMetrics(elo=1500, games_played=100, win_rate=0.55)
        d = metrics.to_dict()
        assert d["elo"] == 1500
        assert d["games_played"] == 100
        assert d["win_rate"] == 0.55

    def test_from_dict(self):
        """Should create from dictionary."""
        d = {"elo": 1600, "games_played": 200}
        metrics = ModelMetrics.from_dict(d)
        assert metrics.elo == 1600
        assert metrics.games_played == 200

    def test_from_dict_ignores_extra_keys(self):
        """Should ignore unknown keys."""
        d = {"elo": 1500, "unknown_field": "value"}
        metrics = ModelMetrics.from_dict(d)
        assert metrics.elo == 1500


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = TrainingConfig()
        assert config.learning_rate == 0.001
        assert config.batch_size == 256
        assert config.epochs == 100
        assert config.optimizer == "adam"
        assert config.architecture == "resnet"

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = TrainingConfig(learning_rate=0.01, batch_size=512)
        d = config.to_dict()
        assert d["learning_rate"] == 0.01
        assert d["batch_size"] == 512

    def test_from_dict(self):
        """Should create from dictionary."""
        d = {"learning_rate": 0.002, "num_residual_blocks": 20}
        config = TrainingConfig.from_dict(d)
        assert config.learning_rate == 0.002
        assert config.num_residual_blocks == 20

    def test_from_dict_extra_fields(self):
        """Should store extra fields in extra_config."""
        d = {"learning_rate": 0.001, "custom_param": "value"}
        config = TrainingConfig.from_dict(d)
        assert config.extra_config.get("custom_param") == "value"


class TestModelStage:
    """Tests for ModelStage enum."""

    def test_all_stages_exist(self):
        """Should have all expected stages."""
        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"
        assert ModelStage.REJECTED.value == "rejected"


class TestModelType:
    """Tests for ModelType enum."""

    def test_all_types_exist(self):
        """Should have all expected types."""
        assert ModelType.POLICY_VALUE.value == "policy_value"
        assert ModelType.ENSEMBLE.value == "ensemble"
        assert ModelType.COMPRESSED.value == "compressed"
        assert ModelType.EXPERIMENTAL.value == "experimental"
        assert ModelType.HEURISTIC.value == "heuristic"


class TestRegistryDatabase:
    """Tests for RegistryDatabase class."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database."""
        return RegistryDatabase(tmp_path / "test_registry.db")

    def test_init_creates_tables(self, db):
        """Should create database tables on init."""
        # Check tables exist by querying sqlite_master
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "models" in tables
        assert "versions" in tables
        assert "tags" in tables
        assert "stage_transitions" in tables
        assert "comparisons" in tables

    def test_create_model(self, db):
        """Should create a new model entry."""
        db.create_model("test_model", "Test Model", ModelType.POLICY_VALUE)
        assert db.model_exists("test_model")

    def test_model_not_exists(self, db):
        """Should return False for non-existent model."""
        assert db.model_exists("nonexistent") is False

    def test_get_next_version_initial(self, db):
        """Should return 1 for first version."""
        db.create_model("test", "Test", ModelType.POLICY_VALUE)
        assert db.get_next_version("test") == 1

    def test_create_version(self, db):
        """Should create a version entry."""
        db.create_model("test", "Test", ModelType.POLICY_VALUE)
        db.create_version(
            model_id="test",
            version=1,
            stage=ModelStage.DEVELOPMENT,
            file_path="/path/to/model.pt",
            file_hash="abc123",
            file_size=1000,
            metrics=ModelMetrics(elo=1500),
            training_config=TrainingConfig(),
        )

        version = db.get_version("test", 1)
        assert version is not None
        assert version["stage"] == "development"
        assert version["file_hash"] == "abc123"

    def test_update_stage(self, db):
        """Should update model stage."""
        db.create_model("test", "Test", ModelType.POLICY_VALUE)
        db.create_version(
            model_id="test",
            version=1,
            stage=ModelStage.DEVELOPMENT,
            file_path="/path/model.pt",
            file_hash="abc",
            file_size=100,
            metrics=ModelMetrics(),
            training_config=TrainingConfig(),
        )

        db.update_stage("test", 1, ModelStage.STAGING, "Testing")
        version = db.get_version("test", 1)
        assert version["stage"] == "staging"

    def test_add_and_search_tag(self, db):
        """Should add and search by tag."""
        db.create_model("test", "Test", ModelType.POLICY_VALUE)
        db.create_version(
            model_id="test",
            version=1,
            stage=ModelStage.DEVELOPMENT,
            file_path="/path/model.pt",
            file_hash="abc",
            file_size=100,
            metrics=ModelMetrics(),
            training_config=TrainingConfig(),
        )

        db.add_tag("test", 1, "baseline")
        results = db.search_by_tag("baseline")
        assert len(results) == 1
        assert results[0]["model_id"] == "test"

    def test_get_stage_history(self, db):
        """Should track stage transitions."""
        db.create_model("test", "Test", ModelType.POLICY_VALUE)
        db.create_version(
            model_id="test",
            version=1,
            stage=ModelStage.DEVELOPMENT,
            file_path="/path/model.pt",
            file_hash="abc",
            file_size=100,
            metrics=ModelMetrics(),
            training_config=TrainingConfig(),
        )
        db.update_stage("test", 1, ModelStage.STAGING)

        history = db.get_stage_history("test", 1)
        assert len(history) == 2  # Initial + update


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create a registry with temp directory."""
        return ModelRegistry(registry_dir=tmp_path / "registry")

    @pytest.fixture
    def model_file(self, tmp_path):
        """Create a dummy model file."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"dummy model data")
        return model_path

    def test_initialization(self, registry):
        """Should initialize directories."""
        assert registry.registry_dir.exists()
        assert registry.model_storage_dir.exists()

    def test_register_model(self, registry, model_file):
        """Should register a new model."""
        model_id, version = registry.register_model(
            name="Test Model",
            model_path=model_file,
        )
        assert model_id is not None
        assert version == 1

    def test_register_model_with_metrics(self, registry, model_file):
        """Should store metrics with model."""
        model_id, version = registry.register_model(
            name="Test Model",
            model_path=model_file,
            metrics=ModelMetrics(elo=1500, games_played=100),
        )

        model = registry.get_model(model_id, version)
        assert model.metrics.elo == 1500
        assert model.metrics.games_played == 100

    def test_register_model_not_found(self, registry):
        """Should raise for missing file."""
        with pytest.raises(FileNotFoundError):
            registry.register_model(
                name="Test",
                model_path=Path("/nonexistent/model.pt"),
            )

    def test_register_new_version(self, registry, model_file):
        """Should increment version for same model."""
        model_id, v1 = registry.register_model(
            name="Test",
            model_path=model_file,
        )
        _, v2 = registry.register_model(
            name="Test",
            model_path=model_file,
            model_id=model_id,
        )
        assert v2 == v1 + 1

    def test_get_model(self, registry, model_file):
        """Should retrieve registered model."""
        model_id, version = registry.register_model(
            name="Test Model",
            model_path=model_file,
            description="A test model",
        )

        model = registry.get_model(model_id, version)
        assert model is not None
        assert model.name == "Test Model"
        assert model.stage == ModelStage.DEVELOPMENT

    def test_get_model_latest_version(self, registry, model_file):
        """Should get latest version when version not specified."""
        model_id, _ = registry.register_model(name="Test", model_path=model_file)
        registry.register_model(name="Test", model_path=model_file, model_id=model_id)

        model = registry.get_model(model_id)  # No version
        assert model.version == 2

    def test_get_model_not_found(self, registry):
        """Should return None for unknown model."""
        model = registry.get_model("unknown")
        assert model is None

    def test_promote_to_staging(self, registry, model_file):
        """Should promote from development to staging."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
        )

        registry.promote(model_id, version, ModelStage.STAGING, "Testing")
        model = registry.get_model(model_id, version)
        assert model.stage == ModelStage.STAGING

    def test_promote_to_production(self, registry, model_file):
        """Should promote to production and archive old."""
        # Register and promote first model
        model_id, v1 = registry.register_model(name="Test", model_path=model_file)
        registry.promote(model_id, v1, ModelStage.STAGING)
        registry.promote(model_id, v1, ModelStage.PRODUCTION)

        # Register and promote second model
        _, v2 = registry.register_model(name="Test", model_path=model_file, model_id=model_id)
        registry.promote(model_id, v2, ModelStage.STAGING)
        registry.promote(model_id, v2, ModelStage.PRODUCTION)

        # Old version should be archived
        old_model = registry.get_model(model_id, v1)
        assert old_model.stage == ModelStage.ARCHIVED

    def test_promote_invalid_transition(self, registry, model_file):
        """Should reject invalid stage transitions."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
        )

        # Cannot go directly from development to production
        with pytest.raises(ValueError):
            registry.promote(model_id, version, ModelStage.PRODUCTION)

    def test_update_metrics(self, registry, model_file):
        """Should update model metrics."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
        )

        registry.update_metrics(
            model_id, version,
            ModelMetrics(elo=1600, games_played=200)
        )

        model = registry.get_model(model_id, version)
        assert model.metrics.elo == 1600

    def test_add_tag(self, registry, model_file):
        """Should add tags to model."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
        )

        registry.add_tag(model_id, version, "baseline")
        results = registry.search_by_tag("baseline")
        assert len(results) == 1

    def test_list_models(self, registry, model_file):
        """Should list all models."""
        registry.register_model(name="Model1", model_path=model_file)
        registry.register_model(name="Model2", model_path=model_file)

        models = registry.list_models()
        assert len(models) == 2

    def test_list_models_by_stage(self, registry, model_file):
        """Should filter by stage."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
        )
        registry.promote(model_id, version, ModelStage.STAGING)

        staging_models = registry.list_models(stage=ModelStage.STAGING)
        dev_models = registry.list_models(stage=ModelStage.DEVELOPMENT)

        assert len(staging_models) == 1
        assert len(dev_models) == 0

    def test_get_production_model(self, registry, model_file):
        """Should get current production model."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
        )
        registry.promote(model_id, version, ModelStage.STAGING)
        registry.promote(model_id, version, ModelStage.PRODUCTION)

        prod = registry.get_production_model()
        assert prod is not None
        assert prod.model_id == model_id

    def test_get_production_model_none(self, registry):
        """Should return None when no production model."""
        prod = registry.get_production_model()
        assert prod is None

    def test_compare_models(self, registry, model_file):
        """Should record model comparison."""
        id1, v1 = registry.register_model(name="Model1", model_path=model_file)
        id2, v2 = registry.register_model(name="Model2", model_path=model_file)

        result = registry.compare_models(
            model_a=(id1, v1),
            model_b=(id2, v2),
            games=100,
            a_wins=55,
            b_wins=40,
            draws=5,
            elo_diff=30.0,
        )

        assert result["a_win_rate"] == 0.55
        assert result["b_win_rate"] == 0.40
        assert result["elo_diff"] == 30.0

    def test_get_stage_history(self, registry, model_file):
        """Should return stage transition history."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
        )
        registry.promote(model_id, version, ModelStage.STAGING, "First promotion")
        registry.promote(model_id, version, ModelStage.PRODUCTION, "Second promotion")

        history = registry.get_stage_history(model_id, version)
        assert len(history) == 3  # Initial + 2 promotions

    def test_export_and_import_model(self, registry, model_file, tmp_path):
        """Should export and import models."""
        model_id, version = registry.register_model(
            name="Test Export",
            model_path=model_file,
            metrics=ModelMetrics(elo=1500),
        )

        # Export
        export_path = tmp_path / "exported_model.pt"
        registry.export_model(model_id, version, export_path)
        assert export_path.exists()

        # Create new registry and import
        new_registry = ModelRegistry(registry_dir=tmp_path / "new_registry")
        new_id, new_version = new_registry.import_model(
            export_path,
            model_id="imported_model",
        )

        imported = new_registry.get_model(new_id, new_version)
        assert imported is not None
        assert imported.metrics.elo == 1500


class TestAutoPromoter:
    """Tests for AutoPromoter class."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create a registry."""
        return ModelRegistry(registry_dir=tmp_path / "registry")

    @pytest.fixture
    def model_file(self, tmp_path):
        """Create a dummy model file."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"dummy model data")
        return model_path

    def test_initialization_defaults(self, registry):
        """Should initialize with defaults."""
        promoter = AutoPromoter(registry)
        assert promoter.min_elo_improvement == 25.0
        assert promoter.min_games == 50

    def test_initialization_custom(self, registry):
        """Should accept custom thresholds."""
        promoter = AutoPromoter(
            registry,
            min_elo_improvement=50.0,
            min_games=200,
        )
        assert promoter.min_elo_improvement == 50.0
        assert promoter.min_games == 200

    def test_evaluate_for_staging_insufficient_games(self, registry, model_file):
        """Should reject with insufficient games."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
            metrics=ModelMetrics(elo=1500, games_played=10),
        )

        promoter = AutoPromoter(registry, min_games=50)
        should_promote, reason = promoter.evaluate_for_staging(model_id, version)

        assert should_promote is False
        assert "Insufficient" in reason

    def test_evaluate_for_staging_no_elo(self, registry, model_file):
        """Should reject without Elo rating."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
            metrics=ModelMetrics(games_played=100),  # No elo
        )

        promoter = AutoPromoter(registry)
        should_promote, reason = promoter.evaluate_for_staging(model_id, version)

        assert should_promote is False
        assert "No Elo" in reason

    def test_evaluate_for_staging_success(self, registry, model_file):
        """Should approve with good metrics."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
            metrics=ModelMetrics(elo=1500, games_played=50),
        )

        promoter = AutoPromoter(registry, min_games=50)
        should_promote, _reason = promoter.evaluate_for_staging(model_id, version)

        assert should_promote is True

    def test_evaluate_for_production_no_baseline(self, registry, model_file):
        """Should approve when no production baseline."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
            metrics=ModelMetrics(elo=1500, games_played=100),
            initial_stage=ModelStage.STAGING,
        )

        promoter = AutoPromoter(registry)
        should_promote, reason = promoter.evaluate_for_production(model_id, version)

        assert should_promote is True
        assert "No current production" in reason

    def test_evaluate_for_production_insufficient_improvement(self, registry, model_file):
        """Should reject insufficient Elo improvement."""
        # Create production model
        prod_id, prod_v = registry.register_model(
            name="Prod",
            model_path=model_file,
            metrics=ModelMetrics(elo=1500, games_played=100),
        )
        registry.promote(prod_id, prod_v, ModelStage.STAGING)
        registry.promote(prod_id, prod_v, ModelStage.PRODUCTION)

        # Create staging model with small improvement
        stag_id, stag_v = registry.register_model(
            name="Staging",
            model_path=model_file,
            metrics=ModelMetrics(elo=1510, games_played=100),  # Only +10 Elo
            initial_stage=ModelStage.STAGING,
        )

        promoter = AutoPromoter(registry, min_elo_improvement=25.0)
        should_promote, reason = promoter.evaluate_for_production(stag_id, stag_v)

        assert should_promote is False
        assert "insufficient" in reason.lower()

    def test_auto_promote_development_to_staging(self, registry, model_file):
        """Should auto-promote from dev to staging."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
            metrics=ModelMetrics(elo=1500, games_played=50),
        )

        promoter = AutoPromoter(registry, min_games=50)
        new_stage = promoter.auto_promote(model_id, version)

        assert new_stage == ModelStage.STAGING
        model = registry.get_model(model_id, version)
        assert model.stage == ModelStage.STAGING

    def test_auto_promote_returns_none_when_not_ready(self, registry, model_file):
        """Should return None when criteria not met."""
        model_id, version = registry.register_model(
            name="Test",
            model_path=model_file,
            metrics=ModelMetrics(games_played=10),  # Insufficient
        )

        promoter = AutoPromoter(registry, min_games=50)
        new_stage = promoter.auto_promote(model_id, version)

        assert new_stage is None
