"""Tests for daemon_factory_implementations.py - Standalone daemon factory functions.

Tests cover:
- Factory function import and existence
- FACTORY_REGISTRY completeness
- get_factory() lookup behavior
- Factory function basic structure (async, error handling)
- ImportError handling in factories
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest

from app.coordination.daemon_factory_implementations import (
    create_auto_sync,
    create_model_distribution,
    create_npz_distribution,
    create_evaluation,
    create_auto_promotion,
    create_quality_monitor,
    create_feedback_loop,
    create_data_pipeline,
    FACTORY_REGISTRY,
    get_factory,
)


class TestFactoryRegistry:
    """Tests for FACTORY_REGISTRY dict."""

    def test_registry_contains_all_factories(self):
        """Test that registry contains all defined factory functions."""
        expected_factories = {
            "AUTO_SYNC",
            "MODEL_DISTRIBUTION",
            "NPZ_DISTRIBUTION",
            "EVALUATION",
            "AUTO_PROMOTION",
            "QUALITY_MONITOR",
            "FEEDBACK_LOOP",
            "DATA_PIPELINE",
        }
        assert set(FACTORY_REGISTRY.keys()) == expected_factories

    def test_registry_values_are_callables(self):
        """Test that all registry values are callable."""
        for name, factory in FACTORY_REGISTRY.items():
            assert callable(factory), f"{name} factory is not callable"

    def test_registry_values_are_coroutines(self):
        """Test that all registry values are async functions."""
        import inspect

        for name, factory in FACTORY_REGISTRY.items():
            assert inspect.iscoroutinefunction(factory), f"{name} is not async"


class TestGetFactory:
    """Tests for get_factory() lookup function."""

    def test_get_factory_returns_correct_function(self):
        """Test that get_factory returns the correct factory."""
        factory = get_factory("AUTO_SYNC")
        assert factory is create_auto_sync

    def test_get_factory_returns_none_for_unknown(self):
        """Test that get_factory returns None for unknown types."""
        factory = get_factory("NONEXISTENT_DAEMON")
        assert factory is None

    def test_get_factory_case_sensitive(self):
        """Test that get_factory is case-sensitive."""
        assert get_factory("auto_sync") is None
        assert get_factory("Auto_Sync") is None
        assert get_factory("AUTO_SYNC") is create_auto_sync

    def test_get_factory_all_registry_entries(self):
        """Test that get_factory works for all registry entries."""
        for name in FACTORY_REGISTRY.keys():
            factory = get_factory(name)
            assert factory is not None
            assert factory is FACTORY_REGISTRY[name]


class TestCreateAutoSync:
    """Tests for create_auto_sync factory."""

    @pytest.mark.asyncio
    async def test_creates_daemon_and_starts(self):
        """Test that factory creates and starts AutoSyncDaemon."""
        mock_daemon = MagicMock()
        mock_daemon.start = AsyncMock()
        mock_daemon._running = False  # Exit immediately

        # Patch at the source module level (where the import happens)
        with patch(
            "app.coordination.auto_sync_daemon.AutoSyncDaemon",
            return_value=mock_daemon,
        ):
            await create_auto_sync()

        mock_daemon.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_on_import_error(self):
        """Test that ImportError is propagated."""
        with patch.dict(
            "sys.modules",
            {"app.coordination.auto_sync_daemon": None},
        ):
            with pytest.raises((ImportError, TypeError)):
                await create_auto_sync()


class TestCreateModelDistribution:
    """Tests for create_model_distribution factory."""

    @pytest.mark.asyncio
    async def test_creates_daemon_with_model_type(self):
        """Test that factory creates daemon with MODEL data type."""
        mock_daemon = MagicMock()
        mock_daemon.start = AsyncMock()
        mock_daemon._running = False

        mock_data_type = MagicMock()
        mock_data_type.MODEL = "model"

        with patch(
            "app.coordination.unified_distribution_daemon.UnifiedDistributionDaemon",
            return_value=mock_daemon,
        ) as mock_class:
            with patch(
                "app.coordination.unified_distribution_daemon.DataType",
                mock_data_type,
            ):
                import sys
                mock_module = MagicMock()
                mock_module.UnifiedDistributionDaemon = mock_class
                mock_module.DataType = mock_data_type
                with patch.dict(sys.modules, {"app.coordination.unified_distribution_daemon": mock_module}):
                    await create_model_distribution()

        mock_daemon.start.assert_called_once()


class TestCreateNpzDistribution:
    """Tests for create_npz_distribution factory."""

    @pytest.mark.asyncio
    async def test_creates_daemon_with_npz_type(self):
        """Test that factory creates daemon with NPZ data type."""
        mock_daemon = MagicMock()
        mock_daemon.start = AsyncMock()
        mock_daemon._running = False

        mock_data_type = MagicMock()
        mock_data_type.NPZ = "npz"

        with patch(
            "app.coordination.unified_distribution_daemon.UnifiedDistributionDaemon",
            return_value=mock_daemon,
        ) as mock_class:
            with patch(
                "app.coordination.unified_distribution_daemon.DataType",
                mock_data_type,
            ):
                import sys
                mock_module = MagicMock()
                mock_module.UnifiedDistributionDaemon = mock_class
                mock_module.DataType = mock_data_type
                with patch.dict(sys.modules, {"app.coordination.unified_distribution_daemon": mock_module}):
                    await create_npz_distribution()

        mock_daemon.start.assert_called_once()


class TestCreateEvaluation:
    """Tests for create_evaluation factory."""

    @pytest.mark.asyncio
    async def test_creates_daemon_and_starts(self):
        """Test that factory creates and starts EvaluationDaemon."""
        mock_daemon = MagicMock()
        mock_daemon.start = AsyncMock()
        mock_daemon._running = False

        with patch(
            "app.coordination.evaluation_daemon.EvaluationDaemon",
            return_value=mock_daemon,
        ) as mock_class:
            import sys
            mock_module = MagicMock()
            mock_module.EvaluationDaemon = mock_class
            with patch.dict(sys.modules, {"app.coordination.evaluation_daemon": mock_module}):
                await create_evaluation()

        mock_daemon.start.assert_called_once()


class TestCreateAutoPromotion:
    """Tests for create_auto_promotion factory."""

    @pytest.mark.asyncio
    async def test_creates_controller_and_runs(self):
        """Test that factory creates and runs PromotionController."""
        mock_controller = MagicMock()
        mock_controller.run = AsyncMock()

        with patch(
            "app.training.promotion_controller.PromotionController",
            return_value=mock_controller,
        ) as mock_class:
            import sys
            mock_module = MagicMock()
            mock_module.PromotionController = mock_class
            with patch.dict(sys.modules, {"app.training.promotion_controller": mock_module}):
                await create_auto_promotion()

        mock_controller.run.assert_called_once()


class TestCreateQualityMonitor:
    """Tests for create_quality_monitor factory."""

    @pytest.mark.asyncio
    async def test_creates_daemon_and_starts(self):
        """Test that factory creates and starts QualityMonitorDaemon."""
        mock_daemon = MagicMock()
        mock_daemon.start = AsyncMock()
        mock_daemon._running = False

        with patch(
            "app.coordination.quality_monitor_daemon.QualityMonitorDaemon",
            return_value=mock_daemon,
        ) as mock_class:
            import sys
            mock_module = MagicMock()
            mock_module.QualityMonitorDaemon = mock_class
            with patch.dict(sys.modules, {"app.coordination.quality_monitor_daemon": mock_module}):
                await create_quality_monitor()

        mock_daemon.start.assert_called_once()


class TestCreateFeedbackLoop:
    """Tests for create_feedback_loop factory."""

    @pytest.mark.asyncio
    async def test_creates_controller_and_starts(self):
        """Test that factory creates and starts FeedbackLoopController."""
        mock_controller = MagicMock()
        mock_controller.start = AsyncMock()
        mock_controller._running = False

        with patch(
            "app.coordination.feedback_loop_controller.FeedbackLoopController",
            return_value=mock_controller,
        ) as mock_class:
            import sys
            mock_module = MagicMock()
            mock_module.FeedbackLoopController = mock_class
            with patch.dict(sys.modules, {"app.coordination.feedback_loop_controller": mock_module}):
                await create_feedback_loop()

        mock_controller.start.assert_called_once()


class TestCreateDataPipeline:
    """Tests for create_data_pipeline factory."""

    @pytest.mark.asyncio
    async def test_creates_orchestrator_and_starts(self):
        """Test that factory creates and starts DataPipelineOrchestrator."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.start = AsyncMock()
        mock_orchestrator._running = False

        with patch(
            "app.coordination.data_pipeline_orchestrator.DataPipelineOrchestrator",
            return_value=mock_orchestrator,
        ) as mock_class:
            import sys
            mock_module = MagicMock()
            mock_module.DataPipelineOrchestrator = mock_class
            with patch.dict(sys.modules, {"app.coordination.data_pipeline_orchestrator": mock_module}):
                await create_data_pipeline()

        mock_orchestrator.start.assert_called_once()


class TestFactoryAsyncBehavior:
    """Tests for async behavior of factory functions."""

    @pytest.mark.asyncio
    async def test_factory_functions_are_awaitable(self):
        """Test that all factory functions can be awaited."""
        import inspect

        for name, factory in FACTORY_REGISTRY.items():
            result = factory()
            assert inspect.iscoroutine(result), f"{name} factory does not return coroutine"
            # Clean up the coroutine to avoid warnings
            result.close()

    def test_factory_functions_return_coroutines(self):
        """Test that calling factory returns a coroutine object."""
        import inspect

        for name, factory in FACTORY_REGISTRY.items():
            result = factory()
            assert inspect.iscoroutine(result)
            result.close()


class TestFactoryImports:
    """Tests for factory function module availability."""

    def test_module_imports_successfully(self):
        """Test that the module can be imported."""
        import app.coordination.daemon_factory_implementations as module

        assert hasattr(module, "FACTORY_REGISTRY")
        assert hasattr(module, "get_factory")

    def test_all_factories_are_importable(self):
        """Test that all factory functions can be imported."""
        from app.coordination.daemon_factory_implementations import (
            create_auto_sync,
            create_model_distribution,
            create_npz_distribution,
            create_evaluation,
            create_auto_promotion,
            create_quality_monitor,
            create_feedback_loop,
            create_data_pipeline,
        )

        # All imports should be non-None
        factories = [
            create_auto_sync,
            create_model_distribution,
            create_npz_distribution,
            create_evaluation,
            create_auto_promotion,
            create_quality_monitor,
            create_feedback_loop,
            create_data_pipeline,
        ]
        for factory in factories:
            assert factory is not None
            assert callable(factory)


class TestFactoryDocstrings:
    """Tests for factory function documentation."""

    def test_all_factories_have_docstrings(self):
        """Test that all factory functions have docstrings."""
        for name, factory in FACTORY_REGISTRY.items():
            assert factory.__doc__ is not None, f"{name} factory has no docstring"
            assert len(factory.__doc__) > 10, f"{name} factory has empty docstring"

    def test_docstrings_describe_purpose(self):
        """Test that docstrings describe the daemon purpose."""
        # Each docstring should mention "daemon", "controller", or similar
        for name, factory in FACTORY_REGISTRY.items():
            doc = factory.__doc__.lower()
            has_keyword = any(
                kw in doc for kw in ["daemon", "controller", "orchestrator", "run"]
            )
            assert has_keyword, f"{name} docstring doesn't describe purpose"


class TestRegistryConsistency:
    """Tests for registry consistency with module exports."""

    def test_registry_matches_module_functions(self):
        """Test that registry entries match module-level functions."""
        import app.coordination.daemon_factory_implementations as module

        # Check that each registry value is actually in the module
        for name, factory in FACTORY_REGISTRY.items():
            # Find the function name from the factory object
            func_name = factory.__name__
            assert hasattr(module, func_name), f"{func_name} not in module"
            assert getattr(module, func_name) is factory

    def test_no_duplicate_factories(self):
        """Test that no factory is registered twice."""
        factory_ids = [id(f) for f in FACTORY_REGISTRY.values()]
        assert len(factory_ids) == len(set(factory_ids)), "Duplicate factories in registry"
