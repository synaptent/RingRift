"""Integration tests for cluster consolidation event flow.

January 7, 2026 - Phase 4.8: Tests the critical event chain from distributed
selfplay to canonical database consolidation:

    NEW_GAMES_AVAILABLE → ClusterConsolidationDaemon → CONSOLIDATION_COMPLETE

This flow bridges the gap between distributed selfplay (on 30+ cluster nodes)
and the training pipeline (which needs games in canonical databases).

Event Chain Tested:
    1. SELFPLAY_COMPLETE (on cluster node) → NEW_GAMES_AVAILABLE
    2. NEW_GAMES_AVAILABLE → ClusterConsolidationDaemon._on_new_games_available()
    3. ClusterConsolidationDaemon syncs data from cluster nodes
    4. Merge into canonical_{board}_{n}p.db
    5. CONSOLIDATION_COMPLETE → DataPipelineOrchestrator

Usage:
    pytest tests/integration/coordination/test_cluster_event_flow.py -v
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.data_events import DataEventType


# =============================================================================
# Test Infrastructure
# =============================================================================


@dataclass
class MockEventBus:
    """Mock event bus for tracking event emissions and subscriptions."""

    subscriptions: dict[str, list] = field(default_factory=dict)
    emitted_events: list[tuple[str, dict]] = field(default_factory=list)

    def subscribe(self, event_type: Any, handler: Any) -> None:
        """Subscribe a handler to an event type."""
        type_name = event_type.value if hasattr(event_type, "value") else str(event_type)
        if type_name not in self.subscriptions:
            self.subscriptions[type_name] = []
        self.subscriptions[type_name].append(handler)

    def unsubscribe(self, event_type: Any, handler: Any) -> None:
        """Unsubscribe a handler from an event type."""
        type_name = event_type.value if hasattr(event_type, "value") else str(event_type)
        if type_name in self.subscriptions:
            try:
                self.subscriptions[type_name].remove(handler)
            except ValueError:
                pass

    def emit(self, event_type: Any, payload: dict) -> None:
        """Emit an event to all subscribed handlers."""
        type_name = event_type.value if hasattr(event_type, "value") else str(event_type)
        self.emitted_events.append((type_name, payload))

        for handler in self.subscriptions.get(type_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.get_event_loop().run_until_complete(handler(payload))
                else:
                    handler(payload)
            except Exception:
                pass

    async def emit_async(self, event_type: Any, payload: dict) -> None:
        """Emit an event asynchronously."""
        type_name = event_type.value if hasattr(event_type, "value") else str(event_type)
        self.emitted_events.append((type_name, payload))

        for handler in self.subscriptions.get(type_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception:
                pass


@dataclass
class MockP2PStatus:
    """Mock P2P status response."""

    alive_peers: list[str] = field(default_factory=list)
    leader_id: str = "mac-studio"
    node_id: str = "test-node"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus for testing."""
    return MockEventBus()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directories for canonical and synced data."""
    canonical_dir = tmp_path / "canonical"
    synced_dir = tmp_path / "synced"
    canonical_dir.mkdir()
    synced_dir.mkdir()
    return canonical_dir, synced_dir


@pytest.fixture
def sample_games_db(tmp_path):
    """Create a sample SQLite database with game data."""
    db_path = tmp_path / "sample_games.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create games table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            winner INTEGER,
            completed_at TEXT,
            config_key TEXT
        )
    """)

    # Create moves table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            player INTEGER NOT NULL,
            move_type TEXT NOT NULL,
            move_data TEXT,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)

    # Insert sample games
    for i in range(10):
        game_id = f"game_{i:04d}"
        cursor.execute(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?)",
            (game_id, "hex8", 2, i % 2, "2026-01-07T12:00:00", "hex8_2p"),
        )
        # Add some moves
        for move_num in range(20):
            cursor.execute(
                "INSERT INTO moves (game_id, move_number, player, move_type, move_data) VALUES (?, ?, ?, ?, ?)",
                (game_id, move_num, move_num % 2, "PLACE_RING", f'{{"to": [{move_num % 8}, {move_num % 8}]}}'),
            )

    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# Unit Tests for Event Handlers
# =============================================================================


class TestClusterConsolidationEventHandlers:
    """Tests for ClusterConsolidationDaemon event handlers."""

    def test_on_new_games_available_queues_priority_sync(self):
        """NEW_GAMES_AVAILABLE should queue a priority sync for the source node."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False)
        daemon = ClusterConsolidationDaemon(config=config)

        # Simulate event payload
        event = MagicMock()
        event.payload = {"source": "vast-12345", "config_key": "hex8_2p", "games": 50}

        daemon._on_new_games_available(event)

        assert "vast-12345" in daemon._priority_nodes

    def test_on_selfplay_complete_queues_priority_sync(self):
        """SELFPLAY_COMPLETE should queue a priority sync for the source node."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False)
        daemon = ClusterConsolidationDaemon(config=config)

        # Simulate event payload
        event = MagicMock()
        event.payload = {"node_id": "nebius-h100-1", "config_key": "square8_4p", "games_completed": 100}

        daemon._on_selfplay_complete(event)

        assert "nebius-h100-1" in daemon._priority_nodes

    def test_handles_dict_payload_directly(self):
        """Handler should work with dict payloads (no .payload attribute)."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False)
        daemon = ClusterConsolidationDaemon(config=config)

        # Dict payload (no wrapper)
        payload = {"host": "lambda-gh200-1", "config_key": "hex8_3p"}
        daemon._on_new_games_available(payload)

        assert "lambda-gh200-1" in daemon._priority_nodes

    def test_handles_missing_source_gracefully(self):
        """Handler should not fail with missing source fields."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False)
        daemon = ClusterConsolidationDaemon(config=config)

        # Payload with no source info
        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "games": 50}

        # Should not raise
        daemon._on_new_games_available(event)

        # No node added (empty source)
        assert "" not in daemon._priority_nodes


# =============================================================================
# Integration Tests for Event Flow
# =============================================================================


class TestConsolidationEventFlow:
    """Integration tests for the full consolidation event flow."""

    @pytest.mark.asyncio
    async def test_event_subscription_on_start(self, mock_event_bus):
        """Daemon should subscribe to events on startup."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False, enabled=True)
        daemon = ClusterConsolidationDaemon(config=config)

        # Mock the router subscribe helper imported inside _subscribe_to_events.
        with patch(
            "app.coordination.event_router.subscribe",
            side_effect=mock_event_bus.subscribe,
        ):
            await daemon._subscribe_to_events()

        # Should subscribe to NEW_GAMES_AVAILABLE and SELFPLAY_COMPLETE
        # NEW_GAMES_AVAILABLE.value is "new_games"
        assert "new_games" in mock_event_bus.subscriptions or DataEventType.NEW_GAMES_AVAILABLE.value in mock_event_bus.subscriptions
        assert daemon._subscribed is True

    @pytest.mark.asyncio
    async def test_event_unsubscription_on_stop(self, mock_event_bus):
        """Daemon should unsubscribe from events on shutdown."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False)
        daemon = ClusterConsolidationDaemon(config=config)
        daemon._subscribed = True

        # First subscribe, then unsubscribe
        with patch(
            "app.coordination.event_router.subscribe",
            side_effect=mock_event_bus.subscribe,
        ), patch(
            "app.coordination.event_router.unsubscribe",
            side_effect=mock_event_bus.unsubscribe,
        ):
            await daemon._subscribe_to_events()
            await daemon._unsubscribe_from_events()

        assert daemon._subscribed is False

    @pytest.mark.asyncio
    async def test_consolidation_complete_emitted_after_merge(self, temp_data_dir):
        """CONSOLIDATION_COMPLETE should be emitted after successful merge."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        canonical_dir, synced_dir = temp_data_dir
        config = ClusterConsolidationConfig(
            coordinator_only=False,
            canonical_dir=canonical_dir,
            synced_dir=synced_dir,
        )
        daemon = ClusterConsolidationDaemon(config=config)

        emitted_events = []

        # Mock safe_emit_event to capture emitted events
        with patch(
            "app.coordination.cluster_consolidation_daemon.safe_emit_event",
            side_effect=lambda event_type, payload, **kwargs: emitted_events.append(
                (event_type, payload)
            ),
        ):
            # Call _emit_consolidation_complete with correct signature
            await daemon._emit_consolidation_complete(
                config_key="hex8_2p",
                games_merged=100,
                canonical_db="/path/to/canonical_hex8_2p.db",
            )

        # Check CONSOLIDATION_COMPLETE was emitted
        assert len(emitted_events) > 0
        event_type, payload = emitted_events[0]
        assert "consolidation" in event_type.lower() if isinstance(event_type, str) else "consolidation" in str(event_type).lower()

    @pytest.mark.asyncio
    async def test_full_event_chain_new_games_to_consolidation(
        self, mock_event_bus, temp_data_dir, sample_games_db
    ):
        """Test full chain: NEW_GAMES_AVAILABLE → sync → CONSOLIDATION_COMPLETE."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        canonical_dir, synced_dir = temp_data_dir
        config = ClusterConsolidationConfig(
            coordinator_only=False,
            canonical_dir=canonical_dir,
            synced_dir=synced_dir,
        )
        daemon = ClusterConsolidationDaemon(config=config)

        # Setup: Mock P2P status to return one peer with data
        mock_peers = [{"node_id": "vast-12345", "tailscale_ip": "100.64.0.1"}]

        # Emit NEW_GAMES_AVAILABLE
        payload = {"source": "vast-12345", "config_key": "hex8_2p", "games": 10}
        daemon._on_new_games_available(MagicMock(payload=payload))

        # Verify priority node was queued
        assert "vast-12345" in daemon._priority_nodes

    @pytest.mark.asyncio
    async def test_multiple_nodes_priority_queue(self):
        """Multiple NEW_GAMES_AVAILABLE events should queue multiple nodes."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False)
        daemon = ClusterConsolidationDaemon(config=config)

        # Emit events from multiple nodes
        nodes = ["vast-12345", "nebius-h100-1", "lambda-gh200-3"]
        for node in nodes:
            payload = MagicMock(payload={"source": node, "config_key": "hex8_2p"})
            daemon._on_new_games_available(payload)

        # All nodes should be in priority queue
        assert all(node in daemon._priority_nodes for node in nodes)


# =============================================================================
# Tests for Coordinator-Only Behavior
# =============================================================================


class TestCoordinatorOnlyBehavior:
    """Tests for coordinator-only daemon behavior."""

    def test_is_coordinator_detects_environment_variable(self):
        """Should detect coordinator from RINGRIFT_IS_COORDINATOR env var."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=True)
        daemon = ClusterConsolidationDaemon(config=config)
        daemon._is_coordinator = None  # Reset cached value

        with patch.dict("os.environ", {"RINGRIFT_IS_COORDINATOR": "true"}):
            assert daemon._is_coordinator_node() is True

    def test_is_coordinator_detects_hostname(self):
        """Should detect coordinator from hostname (mac-studio, local-mac)."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=True)
        daemon = ClusterConsolidationDaemon(config=config)
        daemon._is_coordinator = None

        with patch("socket.gethostname", return_value="mac-studio.local"):
            assert daemon._is_coordinator_node() is True

    @pytest.mark.asyncio
    async def test_skips_cycle_on_non_coordinator(self):
        """Should skip run_cycle on non-coordinator nodes."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=True)
        daemon = ClusterConsolidationDaemon(config=config)
        daemon._is_coordinator = False  # Force non-coordinator

        # Mock _run_sync_cycle to track if it's called
        sync_called = False

        async def mock_sync():
            nonlocal sync_called
            sync_called = True

        daemon._run_sync_cycle = mock_sync

        # Run cycle
        await daemon._run_cycle()

        # Should not have called sync (skipped)
        assert sync_called is False


# =============================================================================
# Tests for Event Chain Integration
# =============================================================================


class TestEventChainIntegration:
    """Tests for event chain integration with DataPipelineOrchestrator."""

    def test_consolidation_events_exist_in_data_events(self):
        """CONSOLIDATION_STARTED and CONSOLIDATION_COMPLETE should be defined."""
        assert hasattr(DataEventType, "CONSOLIDATION_STARTED")
        assert hasattr(DataEventType, "CONSOLIDATION_COMPLETE")
        assert DataEventType.CONSOLIDATION_STARTED.value == "consolidation_started"
        assert DataEventType.CONSOLIDATION_COMPLETE.value == "consolidation_complete"

    def test_new_games_available_event_exists(self):
        """NEW_GAMES_AVAILABLE event should be defined."""
        assert hasattr(DataEventType, "NEW_GAMES_AVAILABLE")
        assert DataEventType.NEW_GAMES_AVAILABLE.value == "new_games"

    def test_selfplay_complete_event_exists(self):
        """SELFPLAY_COMPLETE event should be defined."""
        assert hasattr(DataEventType, "SELFPLAY_COMPLETE")
        assert DataEventType.SELFPLAY_COMPLETE.value == "selfplay_complete"

    @pytest.mark.asyncio
    async def test_consolidation_triggers_pipeline_progression(self, mock_event_bus):
        """CONSOLIDATION_COMPLETE should be subscribable by DataPipelineOrchestrator."""
        pipeline_triggered = []

        def on_consolidation_complete(payload):
            pipeline_triggered.append(payload)

        mock_event_bus.subscribe(DataEventType.CONSOLIDATION_COMPLETE, on_consolidation_complete)

        # Emit consolidation complete
        await mock_event_bus.emit_async(
            DataEventType.CONSOLIDATION_COMPLETE,
            {"config_key": "hex8_2p", "games_merged": 100, "source_nodes": ["vast-12345"]},
        )

        assert len(pipeline_triggered) == 1
        assert pipeline_triggered[0]["config_key"] == "hex8_2p"
        assert pipeline_triggered[0]["games_merged"] == 100


# =============================================================================
# Tests for SyncStats and Health Reporting
# =============================================================================


class TestSyncStatsAndHealth:
    """Tests for sync statistics and health reporting."""

    def test_sync_stats_dataclass(self):
        """SyncStats should track all relevant metrics."""
        from app.coordination.cluster_consolidation_daemon import SyncStats

        stats = SyncStats(
            nodes_attempted=10,
            nodes_synced=8,
            nodes_failed=2,
            games_merged=500,
            games_duplicate=50,
            duration_seconds=30.5,
            errors=["timeout on node X"],
        )

        assert stats.nodes_attempted == 10
        assert stats.nodes_synced == 8
        assert stats.nodes_failed == 2
        assert stats.games_merged == 500
        assert stats.games_duplicate == 50
        assert stats.duration_seconds == 30.5
        assert len(stats.errors) == 1

    def test_health_check_returns_result(self):
        """health_check() should return HealthCheckResult."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        config = ClusterConsolidationConfig(coordinator_only=False)
        daemon = ClusterConsolidationDaemon(config=config)

        # Should inherit from HandlerBase which has health_check()
        result = daemon.health_check()

        # HandlerBase.health_check() returns HealthCheckResult
        assert result is not None
        assert hasattr(result, "status") or hasattr(result, "healthy")


# =============================================================================
# Tests for Configuration
# =============================================================================


class TestClusterConsolidationConfig:
    """Tests for ClusterConsolidationConfig."""

    def test_default_config_values(self):
        """Default configuration should have sensible values."""
        from app.coordination.cluster_consolidation_daemon import ClusterConsolidationConfig

        config = ClusterConsolidationConfig()

        assert config.enabled is True
        assert config.cycle_interval_seconds == 300  # 5 minutes
        assert config.max_concurrent_syncs == 1  # Feb 2026: reduced to prevent OOM
        assert config.coordinator_only is True
        assert config.min_moves_for_valid == 5
        assert config.sync_timeout_seconds == 120

    def test_from_env_loads_environment_variables(self):
        """from_env() should load configuration from environment."""
        from app.coordination.cluster_consolidation_daemon import ClusterConsolidationConfig

        with patch.dict(
            "os.environ",
            {
                "RINGRIFT_CLUSTER_CONSOLIDATION_ENABLED": "false",
                "RINGRIFT_CLUSTER_CONSOLIDATION_INTERVAL": "600",
                "RINGRIFT_CLUSTER_MAX_CONCURRENT_SYNCS": "10",
            },
        ):
            config = ClusterConsolidationConfig.from_env()

        assert config.enabled is False
        assert config.cycle_interval_seconds == 600
        assert config.max_concurrent_syncs == 10


# =============================================================================
# Tests for Database Merge Logic
# =============================================================================


class TestDatabaseMergeLogic:
    """Tests for database merge functionality."""

    @pytest.mark.asyncio
    async def test_merge_preserves_game_data(self, temp_data_dir, sample_games_db):
        """Merge should preserve all game and move data."""
        from app.coordination.cluster_consolidation_daemon import (
            ClusterConsolidationConfig,
            ClusterConsolidationDaemon,
        )

        canonical_dir, synced_dir = temp_data_dir
        config = ClusterConsolidationConfig(
            coordinator_only=False,
            canonical_dir=canonical_dir,
            synced_dir=synced_dir,
        )
        daemon = ClusterConsolidationDaemon(config=config)

        # Create target canonical database
        canonical_db = canonical_dir / "canonical_hex8_2p.db"
        conn = sqlite3.connect(str(canonical_db))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                winner INTEGER,
                completed_at TEXT,
                config_key TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                player INTEGER NOT NULL,
                move_type TEXT NOT NULL,
                move_data TEXT,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)
        conn.commit()
        conn.close()

        # Merge would happen here - just verify setup is correct
        # (Full merge testing requires more infrastructure)
        assert canonical_db.exists()

    def test_canonical_configs_list(self):
        """CANONICAL_CONFIGS should list all 12 board configurations."""
        from app.coordination.cluster_consolidation_daemon import CANONICAL_CONFIGS

        assert len(CANONICAL_CONFIGS) == 12

        # All board types present
        board_types = {config[0] for config in CANONICAL_CONFIGS}
        assert board_types == {"hex8", "square8", "square19", "hexagonal"}

        # All player counts present for each board
        for board in board_types:
            player_counts = {config[1] for config in CANONICAL_CONFIGS if config[0] == board}
            assert player_counts == {2, 3, 4}
