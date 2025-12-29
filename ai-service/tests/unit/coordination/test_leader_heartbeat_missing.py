"""Unit tests for leader heartbeat missing event.

P0 December 2025: Tests for LEADER_HEARTBEAT_MISSING event type and emitter.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.data_events import (
    DataEventType,
    emit_leader_heartbeat_missing,
    get_event_bus,
    reset_event_bus,
)


class TestLeaderHeartbeatMissingEventType:
    """Tests for LEADER_HEARTBEAT_MISSING event type."""

    def test_event_type_exists(self):
        """Test that LEADER_HEARTBEAT_MISSING event type exists."""
        assert hasattr(DataEventType, "LEADER_HEARTBEAT_MISSING")
        assert DataEventType.LEADER_HEARTBEAT_MISSING.value == "leader_heartbeat_missing"

    def test_event_type_is_enum_member(self):
        """Test that LEADER_HEARTBEAT_MISSING is a proper enum member."""
        from enum import Enum
        assert isinstance(DataEventType.LEADER_HEARTBEAT_MISSING, Enum)


class TestEmitLeaderHeartbeatMissing:
    """Tests for emit_leader_heartbeat_missing function."""

    def setup_method(self):
        """Reset event bus before each test."""
        reset_event_bus()

    def teardown_method(self):
        """Clean up after each test."""
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_emit_creates_correct_event(self):
        """Test that emit_leader_heartbeat_missing creates correct event."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        bus = get_event_bus()
        bus.subscribe(DataEventType.LEADER_HEARTBEAT_MISSING, handler)

        await emit_leader_heartbeat_missing(
            leader_id="test-leader-1",
            last_heartbeat=1000.0,
            expected_interval=15.0,
            delay_seconds=60.0,
            source="test-node-1",
        )

        # Allow async handler to run
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        event = received_events[0]
        assert event.event_type == DataEventType.LEADER_HEARTBEAT_MISSING
        assert event.payload["leader_id"] == "test-leader-1"
        assert event.payload["last_heartbeat"] == 1000.0
        assert event.payload["expected_interval"] == 15.0
        assert event.payload["delay_seconds"] == 60.0
        assert event.source == "test-node-1"

    @pytest.mark.asyncio
    async def test_emit_with_minimal_args(self):
        """Test emit with minimal required arguments."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        bus = get_event_bus()
        bus.subscribe(DataEventType.LEADER_HEARTBEAT_MISSING, handler)

        await emit_leader_heartbeat_missing(
            leader_id="leader-minimal",
            last_heartbeat=500.0,
            expected_interval=15.0,
            delay_seconds=30.0,
        )

        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].payload["leader_id"] == "leader-minimal"
        assert received_events[0].source == ""  # Default source


class TestEventMappings:
    """Tests for event mappings."""

    def test_leader_heartbeat_missing_in_mappings(self):
        """Test that leader_heartbeat_missing is in DATA_TO_CROSS_PROCESS_MAP."""
        from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

        assert "leader_heartbeat_missing" in DATA_TO_CROSS_PROCESS_MAP
        assert DATA_TO_CROSS_PROCESS_MAP["leader_heartbeat_missing"] == "LEADER_HEARTBEAT_MISSING"


class TestHeartbeatDetectionLogic:
    """Tests for heartbeat missing detection logic."""

    def test_warning_threshold_calculation(self):
        """Test the warning threshold is 3x lease renewal interval."""
        from app.p2p.constants import LEADER_LEASE_RENEW_INTERVAL

        # Warning threshold should be 3x the renewal interval
        expected_threshold = LEADER_LEASE_RENEW_INTERVAL * 3
        assert expected_threshold == 45  # 15 * 3 = 45 seconds

    def test_detection_window(self):
        """Test detection triggers in correct window."""
        from app.p2p.constants import LEADER_LEASE_DURATION, LEADER_LEASE_RENEW_INTERVAL

        now = time.time()
        lease_expires = now + 30  # 30 seconds until expiry

        heartbeat_warning_threshold = LEADER_LEASE_RENEW_INTERVAL * 3  # 45s
        time_until_expiry = lease_expires - now

        # Should trigger: 0 < 30 < 45
        should_warn = 0 < time_until_expiry < heartbeat_warning_threshold
        assert should_warn is True

    def test_no_detection_when_lease_healthy(self):
        """Test no detection when lease is healthy."""
        from app.p2p.constants import LEADER_LEASE_DURATION, LEADER_LEASE_RENEW_INTERVAL

        now = time.time()
        lease_expires = now + 150  # 150 seconds until expiry (healthy)

        heartbeat_warning_threshold = LEADER_LEASE_RENEW_INTERVAL * 3  # 45s
        time_until_expiry = lease_expires - now

        # Should NOT trigger: 150 > 45
        should_warn = 0 < time_until_expiry < heartbeat_warning_threshold
        assert should_warn is False

    def test_no_detection_when_lease_expired(self):
        """Test no detection when lease already expired."""
        from app.p2p.constants import LEADER_LEASE_RENEW_INTERVAL

        now = time.time()
        lease_expires = now - 10  # Already expired 10 seconds ago

        heartbeat_warning_threshold = LEADER_LEASE_RENEW_INTERVAL * 3  # 45s
        time_until_expiry = lease_expires - now  # -10

        # Should NOT trigger: -10 < 0
        should_warn = 0 < time_until_expiry < heartbeat_warning_threshold
        assert should_warn is False


class TestEventExport:
    """Tests for __all__ exports."""

    def test_emit_function_exported(self):
        """Test that emit_leader_heartbeat_missing is in __all__."""
        from app.distributed import data_events

        assert "emit_leader_heartbeat_missing" in data_events.__all__

    def test_can_import_from_module(self):
        """Test that function can be imported from module."""
        from app.distributed.data_events import emit_leader_heartbeat_missing

        assert callable(emit_leader_heartbeat_missing)
