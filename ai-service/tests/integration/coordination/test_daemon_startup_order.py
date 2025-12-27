"""Integration tests for daemon startup order validation.

These tests verify that:
1. DAEMON_STARTUP_ORDER is respected during startup
2. Subscribers (DATA_PIPELINE, FEEDBACK_LOOP) start BEFORE emitters (AUTO_SYNC)
3. EVENT_ROUTER starts first (required for all event routing)
4. Critical daemons get faster health check intervals

Created: December 2025
Purpose: Prevent silent event loss due to startup race conditions
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
)
from app.coordination.daemon_types import (
    CRITICAL_DAEMONS,
    DAEMON_STARTUP_ORDER,
    DaemonType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager_config():
    """Create test config with short intervals for faster tests."""
    return DaemonManagerConfig(
        health_check_interval=0.1,
        shutdown_timeout=1.0,
        recovery_cooldown=0.2,
        auto_restart_failed=False,
    )


@pytest.fixture
def manager(manager_config):
    """Create fresh DaemonManager for each test."""
    DaemonManager.reset_instance()
    mgr = DaemonManager(manager_config)
    mgr._factories.clear()
    mgr._daemons.clear()
    yield mgr
    mgr._running = False
    if mgr._shutdown_event:
        mgr._shutdown_event.set()
    for info in list(mgr._daemons.values()):
        if info.task and not info.task.done():
            info.task.cancel()
    DaemonManager.reset_instance()


# =============================================================================
# Startup Order Definition Tests
# =============================================================================


class TestStartupOrderDefinition:
    """Tests for startup order constant definitions."""

    def test_event_router_is_first_in_startup_order(self):
        """EVENT_ROUTER must be first in startup order."""
        assert len(DAEMON_STARTUP_ORDER) > 0, "DAEMON_STARTUP_ORDER should not be empty"
        assert DAEMON_STARTUP_ORDER[0] == DaemonType.EVENT_ROUTER, (
            "EVENT_ROUTER must be first - all event routing depends on it"
        )

    def test_subscribers_before_emitters(self):
        """DATA_PIPELINE and FEEDBACK_LOOP must start before AUTO_SYNC."""
        # Get indices
        data_pipeline_idx = None
        feedback_loop_idx = None
        auto_sync_idx = None

        for i, daemon_type in enumerate(DAEMON_STARTUP_ORDER):
            if daemon_type == DaemonType.DATA_PIPELINE:
                data_pipeline_idx = i
            elif daemon_type == DaemonType.FEEDBACK_LOOP:
                feedback_loop_idx = i
            elif daemon_type == DaemonType.AUTO_SYNC:
                auto_sync_idx = i

        # Verify ordering
        if data_pipeline_idx is not None and auto_sync_idx is not None:
            assert data_pipeline_idx < auto_sync_idx, (
                f"DATA_PIPELINE (index {data_pipeline_idx}) must start before "
                f"AUTO_SYNC (index {auto_sync_idx}) to receive its events"
            )

        if feedback_loop_idx is not None and auto_sync_idx is not None:
            assert feedback_loop_idx < auto_sync_idx, (
                f"FEEDBACK_LOOP (index {feedback_loop_idx}) must start before "
                f"AUTO_SYNC (index {auto_sync_idx}) to receive its events"
            )

    def test_critical_daemons_have_required_types(self):
        """CRITICAL_DAEMONS should include essential startup order daemons."""
        required_critical = {
            DaemonType.EVENT_ROUTER,
            DaemonType.DAEMON_WATCHDOG,
            DaemonType.DATA_PIPELINE,
            DaemonType.AUTO_SYNC,
            DaemonType.FEEDBACK_LOOP,
        }

        missing = required_critical - CRITICAL_DAEMONS
        assert not missing, (
            f"CRITICAL_DAEMONS is missing essential daemons: {missing}. "
            "These are required for proper startup order and health monitoring."
        )

    def test_startup_order_has_minimum_daemons(self):
        """Startup order should have at least 6 critical daemons."""
        assert len(DAEMON_STARTUP_ORDER) >= 6, (
            f"DAEMON_STARTUP_ORDER has only {len(DAEMON_STARTUP_ORDER)} daemons, "
            "expected at least 6 for proper cluster operation"
        )

    def test_no_duplicates_in_startup_order(self):
        """Startup order should not have duplicate daemon types."""
        seen = set()
        duplicates = []
        for daemon_type in DAEMON_STARTUP_ORDER:
            if daemon_type in seen:
                duplicates.append(daemon_type)
            seen.add(daemon_type)

        assert not duplicates, f"DAEMON_STARTUP_ORDER has duplicates: {duplicates}"


# =============================================================================
# Startup Order Execution Tests
# =============================================================================


class TestStartupOrderExecution:
    """Tests for actual startup order behavior."""

    @pytest.mark.asyncio
    async def test_start_all_respects_startup_order(self, manager: DaemonManager):
        """start_all() should start daemons in DAEMON_STARTUP_ORDER."""
        started_order: list[DaemonType] = []

        async def track_start_factory(daemon_type: DaemonType):
            """Factory that tracks start order."""
            started_order.append(daemon_type)
            # Quick daemon that exits after tracking
            await asyncio.sleep(0.01)

        # Register factories for startup order daemons
        for daemon_type in DAEMON_STARTUP_ORDER[:4]:  # Test first 4 for speed
            manager.register_factory(daemon_type, lambda dt=daemon_type: track_start_factory(dt))

        # Start all
        await manager.start_all()
        await asyncio.sleep(0.1)  # Allow daemons to start

        # Verify order matches (for daemons we registered)
        expected_order = DAEMON_STARTUP_ORDER[:4]
        for i, (expected, actual) in enumerate(zip(expected_order, started_order)):
            assert actual == expected, (
                f"Startup order mismatch at position {i}: "
                f"expected {expected.name}, got {actual.name}"
            )

    @pytest.mark.asyncio
    async def test_individual_start_allows_out_of_order(self, manager: DaemonManager):
        """Individual start() calls should work regardless of order."""
        async def dummy_daemon():
            await asyncio.sleep(10)

        # Register daemons
        manager.register_factory(DaemonType.AUTO_SYNC, dummy_daemon)
        manager.register_factory(DaemonType.DATA_PIPELINE, dummy_daemon)

        # Start AUTO_SYNC first (out of order)
        await manager.start(DaemonType.AUTO_SYNC)
        await asyncio.sleep(0.05)

        assert manager._daemons[DaemonType.AUTO_SYNC].state == DaemonState.RUNNING

        # Then start DATA_PIPELINE (late)
        await manager.start(DaemonType.DATA_PIPELINE)
        await asyncio.sleep(0.05)

        assert manager._daemons[DaemonType.DATA_PIPELINE].state == DaemonState.RUNNING

        # Cleanup
        await manager.shutdown()


# =============================================================================
# Critical Daemon Health Check Tests
# =============================================================================


class TestCriticalDaemonHealthChecks:
    """Tests for critical daemon health check interval handling."""

    def test_critical_daemons_get_faster_health_checks(self, manager: DaemonManager):
        """Critical daemons should have shorter health check intervals."""
        # This tests the P11-HIGH-2 feature: critical daemons get 15s (3x health timeout)
        # vs 60s default for non-critical daemons
        from app.config.coordination_defaults import DaemonLoopDefaults
        from app.coordination.daemon_types import DaemonInfo

        # Critical daemons get 3x health timeout (15s = 5s * 3)
        critical_interval = DaemonLoopDefaults.HEALTH_CHECK_TIMEOUT * 3
        # Default health check interval from DaemonInfo
        default_interval = DaemonInfo(DaemonType.EVENT_ROUTER).health_check_interval

        # Verify critical interval is faster
        assert critical_interval < default_interval, (
            f"Critical interval ({critical_interval}s) should be less than "
            f"default interval ({default_interval}s)"
        )

    @pytest.mark.asyncio
    async def test_event_router_is_critical(self, manager: DaemonManager):
        """EVENT_ROUTER should be in CRITICAL_DAEMONS."""
        assert DaemonType.EVENT_ROUTER in CRITICAL_DAEMONS, (
            "EVENT_ROUTER must be critical - all coordination depends on it"
        )

    @pytest.mark.asyncio
    async def test_daemon_watchdog_is_critical(self, manager: DaemonManager):
        """DAEMON_WATCHDOG should be in CRITICAL_DAEMONS."""
        assert DaemonType.DAEMON_WATCHDOG in CRITICAL_DAEMONS, (
            "DAEMON_WATCHDOG must be critical - it restarts other failed daemons"
        )

    @pytest.mark.asyncio
    async def test_data_pipeline_is_critical(self, manager: DaemonManager):
        """DATA_PIPELINE should be in CRITICAL_DAEMONS."""
        assert DaemonType.DATA_PIPELINE in CRITICAL_DAEMONS, (
            "DATA_PIPELINE must be critical - it processes events from AUTO_SYNC"
        )


# =============================================================================
# Event Loss Prevention Tests
# =============================================================================


class TestEventLossPrevention:
    """Tests for preventing event loss due to startup order issues."""

    @pytest.mark.asyncio
    async def test_subscriber_ready_before_emitter(self, manager: DaemonManager):
        """Verify subscribers are ready to receive events before emitters start."""
        events_received: list[str] = []
        subscriber_ready = asyncio.Event()
        emitter_started = asyncio.Event()

        async def subscriber_daemon():
            """Simulates DATA_PIPELINE subscribing to events."""
            subscriber_ready.set()
            # Wait for emitter to start
            await emitter_started.wait()
            events_received.append("subscriber_received")
            await asyncio.sleep(0.1)

        async def emitter_daemon():
            """Simulates AUTO_SYNC emitting events."""
            # Wait for subscriber to be ready (this is what startup order ensures)
            await subscriber_ready.wait()
            emitter_started.set()
            events_received.append("emitter_sent")
            await asyncio.sleep(0.1)

        # Register in correct order
        manager.register_factory(DaemonType.DATA_PIPELINE, subscriber_daemon)
        manager.register_factory(DaemonType.AUTO_SYNC, emitter_daemon)

        # Start in correct order
        await manager.start(DaemonType.DATA_PIPELINE)
        await asyncio.sleep(0.05)
        await manager.start(DaemonType.AUTO_SYNC)
        await asyncio.sleep(0.15)

        # Verify subscriber was ready before emitter
        assert subscriber_ready.is_set(), "Subscriber should be ready before emitter"
        assert emitter_started.is_set(), "Emitter should have started"

        # Cleanup
        await manager.shutdown()


# =============================================================================
# Regression Tests
# =============================================================================


class TestStartupOrderRegressions:
    """Regression tests for startup order issues."""

    def test_startup_order_matches_documentation(self):
        """Startup order should match what's documented in daemon_types.py."""
        # This test will fail if someone changes the order without updating docs
        expected_early = [
            DaemonType.EVENT_ROUTER,  # Must be first
            DaemonType.DAEMON_WATCHDOG,  # Self-healing
            DaemonType.DATA_PIPELINE,  # Before sync
            DaemonType.FEEDBACK_LOOP,  # Before sync
            DaemonType.AUTO_SYNC,  # Emits events
        ]

        for i, expected in enumerate(expected_early):
            if i < len(DAEMON_STARTUP_ORDER):
                assert DAEMON_STARTUP_ORDER[i] == expected, (
                    f"Startup order position {i} should be {expected.name}, "
                    f"got {DAEMON_STARTUP_ORDER[i].name}"
                )

    def test_critical_daemons_subset_of_startup_order(self):
        """All startup order daemons should either be critical or optional."""
        # Core startup daemons should be in CRITICAL_DAEMONS
        core_startup = set(DAEMON_STARTUP_ORDER[:5])  # First 5 are core
        missing_critical = core_startup - CRITICAL_DAEMONS

        # Allow some non-critical (like TRAINING_TRIGGER which is optional)
        allowed_non_critical = {DaemonType.TRAINING_TRIGGER}
        unexpected = missing_critical - allowed_non_critical

        assert not unexpected, (
            f"Core startup daemons missing from CRITICAL_DAEMONS: {unexpected}"
        )
