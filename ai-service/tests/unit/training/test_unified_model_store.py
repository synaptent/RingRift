"""Tests for unified_model_store.py - consolidated model lifecycle management."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestUnifiedModelStoreImports:
    """Test that unified_model_store provides correct imports."""

    def test_import_unified_api(self):
        """Test importing the unified API."""
        from app.training.unified_model_store import (
            UnifiedModelStore,
            ModelInfo,
            ModelStoreStage,
            ModelStoreType,
            get_model_store,
            register_model,
            get_production_model,
            promote_model,
        )
        assert UnifiedModelStore is not None
        assert ModelInfo is not None
        assert get_model_store is not None

    def test_import_legacy_reexports(self):
        """Test that legacy re-exports are available."""
        from app.training.unified_model_store import (
            ModelRegistry,
            ModelStage,
            ModelType,
            ModelMetrics,
            get_model_registry,
        )
        assert ModelRegistry is not None
        assert ModelStage is not None
        assert get_model_registry is not None

    def test_import_via_package(self):
        """Test importing via app.training package."""
        from app.training import (
            UnifiedModelStore,
            get_model_store,
            ModelInfo,
        )
        assert UnifiedModelStore is not None
        assert get_model_store is not None


class TestModelStoreStage:
    """Test ModelStoreStage enum."""

    def test_stage_values(self):
        """Test that expected stages exist."""
        from app.training.unified_model_store import ModelStoreStage

        assert hasattr(ModelStoreStage, 'DEVELOPMENT')
        assert hasattr(ModelStoreStage, 'STAGING')
        assert hasattr(ModelStoreStage, 'PRODUCTION')
        assert hasattr(ModelStoreStage, 'ARCHIVED')


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating a ModelInfo instance."""
        from app.training.unified_model_store import ModelInfo, ModelStoreStage, ModelStoreType

        info = ModelInfo(
            model_id="test-model-001",
            version=1,
            name="test_model",
            model_type=ModelStoreType.POLICY_VALUE,
            stage=ModelStoreStage.DEVELOPMENT,
            model_path="/path/to/model.pth",
        )

        assert info.model_id == "test-model-001"
        assert info.stage == ModelStoreStage.DEVELOPMENT
        assert info.version == 1

    def test_model_info_to_dict(self):
        """Test converting ModelInfo to dict."""
        from app.training.unified_model_store import ModelInfo, ModelStoreStage, ModelStoreType

        info = ModelInfo(
            model_id="test-model-002",
            version=2,
            name="production_model",
            model_type=ModelStoreType.POLICY_VALUE,
            stage=ModelStoreStage.PRODUCTION,
            model_path="/path/to/prod.pth",
        )

        data = info.to_dict()
        assert isinstance(data, dict)
        assert data["model_id"] == "test-model-002"


class TestUnifiedModelStore:
    """Test UnifiedModelStore class."""

    def test_singleton_pattern(self):
        """Test that get_model_store returns singleton."""
        from app.training.unified_model_store import get_model_store

        store1 = get_model_store()
        store2 = get_model_store()
        assert store1 is store2

    def test_store_has_required_methods(self):
        """Test that store has required interface methods."""
        from app.training.unified_model_store import get_model_store

        store = get_model_store()

        # Check key methods exist
        assert hasattr(store, 'register')
        assert hasattr(store, 'get')
        assert hasattr(store, 'promote')
        assert hasattr(store, 'get_production')
        assert hasattr(store, 'list_models')
        assert hasattr(store, 'get_stats')


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_register_model_function_exists(self):
        """Test that register_model function exists."""
        from app.training.unified_model_store import register_model
        assert callable(register_model)

    def test_get_production_model_function_exists(self):
        """Test that get_production_model function exists."""
        from app.training.unified_model_store import get_production_model
        assert callable(get_production_model)

    def test_promote_model_function_exists(self):
        """Test that promote_model function exists."""
        from app.training.unified_model_store import promote_model
        assert callable(promote_model)
