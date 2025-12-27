"""Tests for app/coordination/coordinator_config.py - Coordinator configuration.

December 27, 2025: Created as part of high-risk module test coverage effort.
coordinator_config.py manages configuration lifecycle for all coordinators.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# =============================================================================
# Test Setup and Teardown
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global config state before and after each test."""
    from app.coordination.coordinator_config import reset_config, reset_exclusion_policy

    reset_config()
    reset_exclusion_policy()
    yield
    reset_config()
    reset_exclusion_policy()


# =============================================================================
# Individual Config Dataclass Tests
# =============================================================================


class TestTaskLifecycleConfig:
    """Tests for TaskLifecycleConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import TaskLifecycleConfig

        config = TaskLifecycleConfig()
        assert config.heartbeat_threshold_seconds == 60.0
        assert config.orphan_check_interval_seconds == 30.0
        assert config.max_history == 1000

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.coordination.coordinator_config import TaskLifecycleConfig

        config = TaskLifecycleConfig(
            heartbeat_threshold_seconds=120.0,
            orphan_check_interval_seconds=60.0,
            max_history=500,
        )
        assert config.heartbeat_threshold_seconds == 120.0
        assert config.orphan_check_interval_seconds == 60.0
        assert config.max_history == 500

    def test_frozen(self):
        """Test that TaskLifecycleConfig is immutable."""
        from app.coordination.coordinator_config import TaskLifecycleConfig

        config = TaskLifecycleConfig()
        with pytest.raises(AttributeError):
            config.heartbeat_threshold_seconds = 999.0


class TestSelfplayConfig:
    """Tests for SelfplayConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import SelfplayConfig

        config = SelfplayConfig()
        assert config.max_history == 500
        assert config.stats_window_seconds == 3600.0
        assert config.backpressure_threshold == 0.8

    def test_frozen(self):
        """Test that SelfplayConfig is immutable."""
        from app.coordination.coordinator_config import SelfplayConfig

        config = SelfplayConfig()
        with pytest.raises(AttributeError):
            config.max_history = 999


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import PipelineConfig

        config = PipelineConfig()
        assert config.max_history == 100
        assert config.auto_trigger is True  # Changed Dec 2025
        assert config.pause_on_critical_constraints is True
        assert config.constraint_stale_seconds == 60.0

    def test_per_stage_auto_trigger_defaults(self):
        """Test per-stage auto-trigger default values."""
        from app.coordination.coordinator_config import PipelineConfig

        config = PipelineConfig()
        assert config.auto_trigger_sync is True
        assert config.auto_trigger_export is True
        assert config.auto_trigger_training is True
        assert config.auto_trigger_evaluation is True
        assert config.auto_trigger_promotion is True

    def test_circuit_breaker_defaults(self):
        """Test circuit breaker default values."""
        from app.coordination.coordinator_config import PipelineConfig

        config = PipelineConfig()
        assert config.circuit_breaker_enabled is True
        assert config.circuit_breaker_failure_threshold == 3
        assert config.circuit_breaker_reset_timeout_seconds == 300.0
        assert config.circuit_breaker_half_open_max_requests == 1

    def test_training_defaults(self):
        """Test training configuration defaults."""
        from app.coordination.coordinator_config import PipelineConfig

        config = PipelineConfig()
        assert config.training_epochs == 50
        assert config.training_batch_size == 512
        assert config.training_model_version == "v2"

    def test_mutable(self):
        """Test that PipelineConfig is mutable."""
        from app.coordination.coordinator_config import PipelineConfig

        config = PipelineConfig()
        config.auto_trigger = False
        assert config.auto_trigger is False


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import OptimizationConfig

        config = OptimizationConfig()
        assert config.cmaes_cooldown_seconds == 3600.0
        assert config.nas_cooldown_seconds == 7200.0
        assert config.auto_trigger_on_plateau is True
        assert config.min_plateau_epochs_for_trigger == 15


class TestMetricsConfig:
    """Tests for MetricsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import MetricsConfig

        config = MetricsConfig()
        assert config.window_size == 100
        assert config.plateau_threshold == 0.001
        assert config.plateau_window == 10
        assert config.regression_threshold == 0.05
        assert config.anomaly_threshold == 3.0


class TestResourceConfig:
    """Tests for ResourceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import ResourceConfig

        config = ResourceConfig()
        assert config.monitoring_interval_seconds == 30.0
        assert config.memory_warning_threshold == 0.8
        assert config.memory_critical_threshold == 0.95
        assert config.gpu_memory_warning_threshold == 0.85
        assert config.gpu_memory_critical_threshold == 0.95


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import CacheConfig

        config = CacheConfig()
        assert config.invalidation_batch_size == 100
        assert config.max_cache_age_seconds == 3600.0
        assert config.auto_refresh is True


class TestHandlerResilienceConfig:
    """Tests for HandlerResilienceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import HandlerResilienceConfig

        config = HandlerResilienceConfig()
        assert config.timeout_seconds == 30.0
        assert config.emit_failure_events is True
        assert config.emit_timeout_events is True
        assert config.max_consecutive_failures == 5
        assert config.log_exceptions is True


class TestHeartbeatConfig:
    """Tests for HeartbeatConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import HeartbeatConfig

        config = HeartbeatConfig()
        assert config.interval_seconds == 30.0
        assert config.enabled is True


class TestEventBusConfig:
    """Tests for EventBusConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.coordinator_config import EventBusConfig

        config = EventBusConfig()
        assert config.max_history == 1000
        assert config.warn_unsubscribed is True
        assert config.max_latency_samples == 1000


# =============================================================================
# CoordinatorConfig Tests
# =============================================================================


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig unified config."""

    def test_default_nested_configs(self):
        """Test that all nested configs are initialized."""
        from app.coordination.coordinator_config import (
            CacheConfig,
            CoordinatorConfig,
            EventBusConfig,
            HandlerResilienceConfig,
            HeartbeatConfig,
            MetricsConfig,
            OptimizationConfig,
            PipelineConfig,
            ResourceConfig,
            SelfplayConfig,
            TaskLifecycleConfig,
        )

        config = CoordinatorConfig()
        assert isinstance(config.task_lifecycle, TaskLifecycleConfig)
        assert isinstance(config.selfplay, SelfplayConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.resources, ResourceConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.handler_resilience, HandlerResilienceConfig)
        assert isinstance(config.heartbeat, HeartbeatConfig)
        assert isinstance(config.event_bus, EventBusConfig)

    def test_to_dict(self):
        """Test to_dict serialization."""
        from app.coordination.coordinator_config import CoordinatorConfig

        config = CoordinatorConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "task_lifecycle" in d
        assert "pipeline" in d
        assert d["task_lifecycle"]["heartbeat_threshold_seconds"] == 60.0

    def test_from_dict_empty(self):
        """Test from_dict with empty dict."""
        from app.coordination.coordinator_config import CoordinatorConfig

        config = CoordinatorConfig.from_dict({})
        # Should use defaults
        assert config.task_lifecycle.heartbeat_threshold_seconds == 60.0

    def test_from_dict_partial(self):
        """Test from_dict with partial config."""
        from app.coordination.coordinator_config import CoordinatorConfig

        data = {
            "task_lifecycle": {"heartbeat_threshold_seconds": 120.0},
        }
        config = CoordinatorConfig.from_dict(data)

        assert config.task_lifecycle.heartbeat_threshold_seconds == 120.0
        assert config.pipeline.auto_trigger is True  # Default

    def test_from_dict_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        from app.coordination.coordinator_config import CoordinatorConfig

        original = CoordinatorConfig()
        d = original.to_dict()
        restored = CoordinatorConfig.from_dict(d)

        assert restored.task_lifecycle.heartbeat_threshold_seconds == original.task_lifecycle.heartbeat_threshold_seconds
        assert restored.pipeline.auto_trigger == original.pipeline.auto_trigger

    def test_from_environment_heartbeat_threshold(self):
        """Test from_environment with COORDINATOR_HEARTBEAT_THRESHOLD."""
        from app.coordination.coordinator_config import CoordinatorConfig

        with patch.dict(os.environ, {"COORDINATOR_HEARTBEAT_THRESHOLD": "120.0"}):
            config = CoordinatorConfig.from_environment()
            assert config.task_lifecycle.heartbeat_threshold_seconds == 120.0

    def test_from_environment_auto_trigger_pipeline(self):
        """Test from_environment with COORDINATOR_AUTO_TRIGGER_PIPELINE."""
        from app.coordination.coordinator_config import CoordinatorConfig

        with patch.dict(os.environ, {"COORDINATOR_AUTO_TRIGGER_PIPELINE": "false"}):
            config = CoordinatorConfig.from_environment()
            assert config.pipeline.auto_trigger is False

        with patch.dict(os.environ, {"COORDINATOR_AUTO_TRIGGER_PIPELINE": "true"}):
            config = CoordinatorConfig.from_environment()
            assert config.pipeline.auto_trigger is True

    def test_from_environment_handler_timeout(self):
        """Test from_environment with COORDINATOR_HANDLER_TIMEOUT."""
        from app.coordination.coordinator_config import CoordinatorConfig

        with patch.dict(os.environ, {"COORDINATOR_HANDLER_TIMEOUT": "60.0"}):
            config = CoordinatorConfig.from_environment()
            assert config.handler_resilience.timeout_seconds == 60.0

    def test_from_environment_heartbeat_enabled(self):
        """Test from_environment with COORDINATOR_HEARTBEAT_ENABLED."""
        from app.coordination.coordinator_config import CoordinatorConfig

        with patch.dict(os.environ, {"COORDINATOR_HEARTBEAT_ENABLED": "false"}):
            config = CoordinatorConfig.from_environment()
            assert config.heartbeat.enabled is False

    def test_from_environment_memory_thresholds(self):
        """Test from_environment with memory thresholds."""
        from app.coordination.coordinator_config import CoordinatorConfig

        with patch.dict(os.environ, {
            "COORDINATOR_MEMORY_WARNING_THRESHOLD": "0.7",
            "COORDINATOR_MEMORY_CRITICAL_THRESHOLD": "0.9",
        }):
            config = CoordinatorConfig.from_environment()
            assert config.resources.memory_warning_threshold == 0.7
            assert config.resources.memory_critical_threshold == 0.9


# =============================================================================
# Global Configuration Function Tests
# =============================================================================


class TestGlobalConfigFunctions:
    """Tests for global configuration functions."""

    def test_get_config_returns_singleton(self):
        """Test get_config returns same instance."""
        from app.coordination.coordinator_config import get_config

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_replaces_singleton(self):
        """Test set_config replaces the global config."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            PipelineConfig,
            get_config,
            set_config,
        )

        original = get_config()
        new_config = CoordinatorConfig(pipeline=PipelineConfig(max_history=999))
        set_config(new_config)

        current = get_config()
        assert current is new_config
        assert current.pipeline.max_history == 999

    def test_reset_config_clears_singleton(self):
        """Test reset_config clears the global config."""
        from app.coordination.coordinator_config import get_config, reset_config

        config1 = get_config()
        reset_config()
        config2 = get_config()

        # After reset, should get new instance
        assert config1 is not config2

    def test_update_config_modifies_values(self):
        """Test update_config modifies specific sections."""
        from app.coordination.coordinator_config import (
            PipelineConfig,
            get_config,
            update_config,
        )

        get_config()  # Initialize
        update_config(pipeline=PipelineConfig(max_history=200))

        config = get_config()
        assert config.pipeline.max_history == 200

    def test_update_config_unknown_key_warning(self):
        """Test update_config logs warning for unknown keys."""
        from app.coordination.coordinator_config import get_config, update_config

        get_config()  # Initialize
        # Should not raise, just log warning
        update_config(unknown_section="value")


# =============================================================================
# Configuration Validation Tests
# =============================================================================


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config_passes(self):
        """Test that default config is valid."""
        from app.coordination.coordinator_config import validate_config

        valid, issues = validate_config()
        assert valid is True
        assert issues == []

    def test_heartbeat_threshold_too_low(self):
        """Test validation catches low heartbeat threshold."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            TaskLifecycleConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            task_lifecycle=TaskLifecycleConfig(heartbeat_threshold_seconds=5.0)
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("heartbeat_threshold_seconds too low" in i for i in issues)

    def test_orphan_check_interval_too_low(self):
        """Test validation catches low orphan check interval."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            TaskLifecycleConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            task_lifecycle=TaskLifecycleConfig(orphan_check_interval_seconds=1.0)
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("orphan_check_interval_seconds too low" in i for i in issues)

    def test_handler_timeout_too_low(self):
        """Test validation catches low handler timeout."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            HandlerResilienceConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            handler_resilience=HandlerResilienceConfig(timeout_seconds=0.5)
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("handler timeout_seconds too low" in i for i in issues)

    def test_max_consecutive_failures_too_low(self):
        """Test validation catches low max consecutive failures."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            HandlerResilienceConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            handler_resilience=HandlerResilienceConfig(max_consecutive_failures=0)
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("max_consecutive_failures must be >= 1" in i for i in issues)

    def test_memory_threshold_out_of_range(self):
        """Test validation catches memory thresholds out of range."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            ResourceConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            resources=ResourceConfig(memory_warning_threshold=1.5)
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("memory_warning_threshold must be between 0 and 1" in i for i in issues)

    def test_memory_warning_exceeds_critical(self):
        """Test validation catches warning > critical threshold."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            ResourceConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            resources=ResourceConfig(
                memory_warning_threshold=0.9,
                memory_critical_threshold=0.8,
            )
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("memory_warning_threshold must be less than critical" in i for i in issues)

    def test_metrics_window_size_too_small(self):
        """Test validation catches small metrics window."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            MetricsConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            metrics=MetricsConfig(window_size=5)
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("metrics window_size too small" in i for i in issues)

    def test_plateau_threshold_not_positive(self):
        """Test validation catches non-positive plateau threshold."""
        from app.coordination.coordinator_config import (
            CoordinatorConfig,
            MetricsConfig,
            validate_config,
        )

        config = CoordinatorConfig(
            metrics=MetricsConfig(plateau_threshold=0.0)
        )
        valid, issues = validate_config(config)

        assert valid is False
        assert any("plateau_threshold must be positive" in i for i in issues)


# =============================================================================
# DaemonExclusionConfig Tests
# =============================================================================


class TestDaemonExclusionConfig:
    """Tests for DaemonExclusionConfig dataclass."""

    def test_default_excluded_nodes(self):
        """Test default excluded nodes are populated."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig()
        # Check some expected defaults
        assert "mbp-16gb" in config.excluded_nodes
        assert "mbp-64gb" in config.excluded_nodes
        assert "aws-proxy" in config.excluded_nodes

    def test_should_exclude_excluded_node(self):
        """Test should_exclude returns True for excluded nodes."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig()
        assert config.should_exclude("mbp-16gb") is True

    def test_should_exclude_retired_node(self):
        """Test should_exclude returns True for retired nodes."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig(retired_nodes={"old-node"})
        assert config.should_exclude("old-node") is True

    def test_should_exclude_normal_node(self):
        """Test should_exclude returns False for normal nodes."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig()
        assert config.should_exclude("training-node-1") is False

    def test_is_nfs_node(self):
        """Test is_nfs_node returns correct value."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig(nfs_nodes={"nfs-host-1", "nfs-host-2"})
        assert config.is_nfs_node("nfs-host-1") is True
        assert config.is_nfs_node("regular-host") is False

    def test_add_excluded_node(self):
        """Test add_excluded_node adds to set."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig()
        config.add_excluded_node("new-excluded-node")
        assert "new-excluded-node" in config.excluded_nodes
        assert config.should_exclude("new-excluded-node") is True

    def test_remove_excluded_node(self):
        """Test remove_excluded_node removes from set."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig()
        config.add_excluded_node("temp-node")
        config.remove_excluded_node("temp-node")
        assert "temp-node" not in config.excluded_nodes

    def test_remove_excluded_node_not_present(self):
        """Test remove_excluded_node handles missing node gracefully."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig()
        # Should not raise
        config.remove_excluded_node("never-existed")

    def test_min_disk_free_gb_default(self):
        """Test min_disk_free_gb has correct default."""
        from app.coordination.coordinator_config import DaemonExclusionConfig

        config = DaemonExclusionConfig()
        assert config.min_disk_free_gb == 50.0


class TestExclusionPolicyFunctions:
    """Tests for exclusion policy global functions."""

    def test_get_exclusion_policy_returns_singleton(self):
        """Test get_exclusion_policy returns same instance."""
        from app.coordination.coordinator_config import get_exclusion_policy

        policy1 = get_exclusion_policy()
        policy2 = get_exclusion_policy()
        assert policy1 is policy2

    def test_reset_exclusion_policy_clears_singleton(self):
        """Test reset_exclusion_policy clears the global policy."""
        from app.coordination.coordinator_config import (
            get_exclusion_policy,
            reset_exclusion_policy,
        )

        policy1 = get_exclusion_policy()
        reset_exclusion_policy()
        policy2 = get_exclusion_policy()

        # After reset, should get new instance
        assert policy1 is not policy2


# =============================================================================
# __all__ Export Tests
# =============================================================================


class TestExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from app.coordination import coordinator_config

        expected = {
            "CacheConfig",
            "CoordinatorConfig",
            "DaemonExclusionConfig",
            "EventBusConfig",
            "HandlerResilienceConfig",
            "HeartbeatConfig",
            "MetricsConfig",
            "OptimizationConfig",
            "PipelineConfig",
            "ResourceConfig",
            "SelfplayConfig",
            "TaskLifecycleConfig",
            "get_config",
            "get_exclusion_policy",
            "reset_config",
            "reset_exclusion_policy",
            "set_config",
            "update_config",
            "validate_config",
        }

        actual = set(coordinator_config.__all__)
        assert expected == actual

    def test_all_exports_importable(self):
        """Test that all exports are importable."""
        from app.coordination.coordinator_config import (
            CacheConfig,
            CoordinatorConfig,
            DaemonExclusionConfig,
            EventBusConfig,
            HandlerResilienceConfig,
            HeartbeatConfig,
            MetricsConfig,
            OptimizationConfig,
            PipelineConfig,
            ResourceConfig,
            SelfplayConfig,
            TaskLifecycleConfig,
            get_config,
            get_exclusion_policy,
            reset_config,
            reset_exclusion_policy,
            set_config,
            update_config,
            validate_config,
        )

        # Verify all are not None
        assert all([
            CacheConfig,
            CoordinatorConfig,
            DaemonExclusionConfig,
            EventBusConfig,
            HandlerResilienceConfig,
            HeartbeatConfig,
            MetricsConfig,
            OptimizationConfig,
            PipelineConfig,
            ResourceConfig,
            SelfplayConfig,
            TaskLifecycleConfig,
            get_config,
            get_exclusion_policy,
            reset_config,
            reset_exclusion_policy,
            set_config,
            update_config,
            validate_config,
        ])
