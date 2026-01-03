"""Tests for scripts.p2p.managers.manager_factory.

January 2026: Phase 4.4 testing.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.p2p.managers.manager_factory import (
    ManagerConfig,
    ManagerFactory,
    get_manager_factory,
    init_manager_factory,
    reset_manager_factory,
)


class TestManagerConfig:
    """Tests for ManagerConfig dataclass."""

    def test_default_values(self):
        """ManagerConfig should have sensible defaults."""
        config = ManagerConfig()
        assert config.db_path == Path("data/p2p.db")
        assert config.models_dir == Path("models")
        assert config.data_dir == Path("data")
        assert config.node_id == ""
        assert config.is_coordinator is False
        assert config.verbose is False
        assert config.dry_run is False
        assert config.port == 8770
        assert config.bind_address == "0.0.0.0"
        assert config.job_timeout == 3600
        assert config.sync_timeout == 300
        assert config.enable_selfplay is True
        assert config.enable_training is True
        assert config.enable_sync is True
        assert config.orchestrator is None

    def test_custom_values(self):
        """ManagerConfig should accept custom values."""
        config = ManagerConfig(
            db_path=Path("/custom/db.db"),
            node_id="test-node",
            is_coordinator=True,
            verbose=True,
            port=9999,
        )
        assert config.db_path == Path("/custom/db.db")
        assert config.node_id == "test-node"
        assert config.is_coordinator is True
        assert config.verbose is True
        assert config.port == 9999


class TestManagerFactory:
    """Tests for ManagerFactory class."""

    def setup_method(self):
        """Reset factory before each test."""
        reset_manager_factory()

    def teardown_method(self):
        """Reset factory after each test."""
        reset_manager_factory()

    def test_init_stores_config(self):
        """Factory should store configuration."""
        config = ManagerConfig(node_id="test")
        factory = ManagerFactory(config)
        assert factory.config is config
        assert factory.config.node_id == "test"

    def test_reset_clears_managers(self):
        """reset() should clear all managers."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Access a manager to initialize it
        with patch("scripts.p2p.managers.StateManager") as mock_sm:
            mock_sm.return_value = MagicMock()
            # Force evaluation of cached_property
            _ = factory.state_manager

        factory.reset()
        assert len(factory._managers) == 0
        assert len(factory._creating) == 0

    @patch("scripts.p2p.managers.StateManager")
    def test_state_manager_lazy_initialization(self, mock_sm):
        """state_manager should be lazily initialized."""
        mock_instance = MagicMock()
        mock_sm.return_value = mock_instance

        config = ManagerConfig(db_path=Path("/test/db.db"), verbose=True)
        factory = ManagerFactory(config)

        # Manager not created yet
        assert "state_manager" not in factory._managers

        # Access triggers creation
        result = factory.state_manager
        assert result is mock_instance
        mock_sm.assert_called_once_with(
            db_path=Path("/test/db.db"),
            verbose=True,
        )
        assert factory._managers["state_manager"] is mock_instance

    @patch("scripts.p2p.managers.StateManager")
    def test_state_manager_cached(self, mock_sm):
        """state_manager should be cached after first access."""
        mock_instance = MagicMock()
        mock_sm.return_value = mock_instance

        config = ManagerConfig()
        factory = ManagerFactory(config)

        result1 = factory.state_manager
        result2 = factory.state_manager

        assert result1 is result2
        mock_sm.assert_called_once()  # Only called once

    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_job_manager_depends_on_state_manager(self, mock_sm, mock_jm):
        """job_manager should depend on state_manager."""
        mock_state = MagicMock()
        mock_sm.return_value = mock_state
        mock_job = MagicMock()
        mock_jm.return_value = mock_job

        config = ManagerConfig(verbose=True)
        factory = ManagerFactory(config)

        result = factory.job_manager

        assert result is mock_job
        # Should have created state_manager first
        mock_sm.assert_called_once()
        mock_jm.assert_called_once()
        # job_manager should receive state_manager
        call_kwargs = mock_jm.call_args[1]
        assert call_kwargs["state_manager"] is mock_state

    @patch("scripts.p2p.managers.TrainingCoordinator")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_training_coordinator_dependencies(self, mock_sm, mock_jm, mock_tc):
        """training_coordinator should depend on state_manager and job_manager."""
        mock_state = MagicMock()
        mock_sm.return_value = mock_state
        mock_job = MagicMock()
        mock_jm.return_value = mock_job
        mock_training = MagicMock()
        mock_tc.return_value = mock_training

        config = ManagerConfig()
        factory = ManagerFactory(config)

        result = factory.training_coordinator

        assert result is mock_training
        call_kwargs = mock_tc.call_args[1]
        assert call_kwargs["state_manager"] is mock_state
        assert call_kwargs["job_manager"] is mock_job

    @patch("scripts.p2p.managers.SelfplayScheduler")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_selfplay_scheduler_dependencies(self, mock_sm, mock_jm, mock_sp):
        """selfplay_scheduler should depend on state_manager and job_manager."""
        mock_state = MagicMock()
        mock_sm.return_value = mock_state
        mock_job = MagicMock()
        mock_jm.return_value = mock_job
        mock_selfplay = MagicMock()
        mock_sp.return_value = mock_selfplay

        config = ManagerConfig()
        factory = ManagerFactory(config)

        result = factory.selfplay_scheduler

        assert result is mock_selfplay
        call_kwargs = mock_sp.call_args[1]
        assert call_kwargs["state_manager"] is mock_state
        assert call_kwargs["job_manager"] is mock_job

    @patch("scripts.p2p.managers.SyncPlanner")
    @patch("scripts.p2p.managers.StateManager")
    def test_sync_planner_dependencies(self, mock_sm, mock_sync):
        """sync_planner should depend on state_manager."""
        mock_state = MagicMock()
        mock_sm.return_value = mock_state
        mock_sync_instance = MagicMock()
        mock_sync.return_value = mock_sync_instance

        config = ManagerConfig()
        factory = ManagerFactory(config)

        result = factory.sync_planner

        assert result is mock_sync_instance
        call_kwargs = mock_sync.call_args[1]
        assert call_kwargs["state_manager"] is mock_state

    @patch("scripts.p2p.managers.NodeSelector")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_node_selector_dependencies(self, mock_sm, mock_jm, mock_ns):
        """node_selector should depend on state_manager and job_manager."""
        mock_state = MagicMock()
        mock_sm.return_value = mock_state
        mock_job = MagicMock()
        mock_jm.return_value = mock_job
        mock_node = MagicMock()
        mock_ns.return_value = mock_node

        config = ManagerConfig()
        factory = ManagerFactory(config)

        result = factory.node_selector

        assert result is mock_node
        call_kwargs = mock_ns.call_args[1]
        assert call_kwargs["state_manager"] is mock_state
        assert call_kwargs["job_manager"] is mock_job

    @patch("scripts.p2p.managers.StateManager")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.TrainingCoordinator")
    @patch("scripts.p2p.managers.SelfplayScheduler")
    @patch("scripts.p2p.managers.SyncPlanner")
    @patch("scripts.p2p.managers.NodeSelector")
    def test_get_all_managers(
        self, mock_ns, mock_sync, mock_sp, mock_tc, mock_jm, mock_sm
    ):
        """get_all_managers should return all managers."""
        # Setup mocks
        mock_sm.return_value = MagicMock()
        mock_jm.return_value = MagicMock()
        mock_tc.return_value = MagicMock()
        mock_sp.return_value = MagicMock()
        mock_sync.return_value = MagicMock()
        mock_ns.return_value = MagicMock()

        config = ManagerConfig()
        factory = ManagerFactory(config)

        managers = factory.get_all_managers()

        assert "state_manager" in managers
        assert "job_manager" in managers
        assert "training_coordinator" in managers
        assert "selfplay_scheduler" in managers
        assert "sync_planner" in managers
        assert "node_selector" in managers
        assert len(managers) == 6

    @patch("scripts.p2p.managers.StateManager")
    def test_get_initialized_managers(self, mock_sm):
        """get_initialized_managers should only return initialized managers."""
        mock_sm.return_value = MagicMock()

        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Initially empty
        assert len(factory.get_initialized_managers()) == 0

        # Access state_manager
        _ = factory.state_manager

        # Now should have one
        initialized = factory.get_initialized_managers()
        assert len(initialized) == 1
        assert "state_manager" in initialized


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        reset_manager_factory()

    def teardown_method(self):
        reset_manager_factory()

    def test_get_manager_factory_returns_none_initially(self):
        """get_manager_factory should return None when not initialized."""
        assert get_manager_factory() is None

    def test_init_manager_factory(self):
        """init_manager_factory should create and return factory."""
        config = ManagerConfig(node_id="test")
        factory = init_manager_factory(config)

        assert factory is not None
        assert factory.config.node_id == "test"
        assert get_manager_factory() is factory

    def test_reset_manager_factory(self):
        """reset_manager_factory should clear the global factory."""
        config = ManagerConfig()
        init_manager_factory(config)
        assert get_manager_factory() is not None

        reset_manager_factory()
        assert get_manager_factory() is None

    def test_init_replaces_existing(self):
        """init_manager_factory should replace existing factory."""
        config1 = ManagerConfig(node_id="first")
        factory1 = init_manager_factory(config1)

        config2 = ManagerConfig(node_id="second")
        factory2 = init_manager_factory(config2)

        assert factory1 is not factory2
        assert get_manager_factory() is factory2
        assert get_manager_factory().config.node_id == "second"


class TestCycleDetection:
    """Tests for circular dependency detection."""

    def setup_method(self):
        reset_manager_factory()

    def teardown_method(self):
        reset_manager_factory()

    def test_check_cycle_detects_simple_cycle(self):
        """_check_cycle should detect simple cycles."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Simulate a cycle: A is creating
        factory._creating.add("manager_a")

        # Trying to create A again should fail
        with pytest.raises(RuntimeError, match="Circular dependency"):
            factory._check_cycle("manager_a")

    def test_check_cycle_shows_path(self):
        """_check_cycle error should show dependency path."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Simulate: A -> B -> C -> A
        factory._creating.add("manager_a")
        factory._creating.add("manager_b")
        factory._creating.add("manager_c")

        with pytest.raises(RuntimeError) as exc_info:
            factory._check_cycle("manager_a")

        error_msg = str(exc_info.value)
        assert "manager_a" in error_msg
        assert "manager_b" in error_msg
        assert "manager_c" in error_msg

    def test_done_creating_removes_from_set(self):
        """_done_creating should remove from creating set."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        factory._creating.add("test_manager")
        assert "test_manager" in factory._creating

        factory._done_creating("test_manager")
        assert "test_manager" not in factory._creating

    def test_done_creating_safe_for_missing(self):
        """_done_creating should be safe for non-existent managers."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Should not raise
        factory._done_creating("nonexistent")
