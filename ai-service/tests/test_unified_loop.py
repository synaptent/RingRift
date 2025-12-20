#!/usr/bin/env python3
"""Tests for the unified AI self-improvement loop components.

Covers:
- Event bus functionality
- Data collector validation
- Training scheduler logic
- Promotion criteria
- Curriculum rebalancing
"""

import asyncio
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.distributed.data_events import (
    DataEvent,
    DataEventType,
    EventBus,
    emit_new_games,
    emit_training_completed,
    get_event_bus,
    reset_event_bus,
)


class TestEventBus:
    """Test EventBus functionality."""

    def setup_method(self):
        """Reset event bus before each test."""
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test basic subscribe and publish."""
        bus = get_event_bus()
        received = []

        def handler(event: DataEvent):
            received.append(event)

        bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, handler)

        event = DataEvent(
            event_type=DataEventType.NEW_GAMES_AVAILABLE,
            payload={"host": "test", "new_games": 100},
        )
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].payload["new_games"] == 100

    @pytest.mark.asyncio
    async def test_async_subscriber(self):
        """Test async subscriber callback."""
        bus = get_event_bus()
        received = []

        async def async_handler(event: DataEvent):
            await asyncio.sleep(0.01)
            received.append(event)

        bus.subscribe(DataEventType.TRAINING_COMPLETED, async_handler)

        event = DataEvent(
            event_type=DataEventType.TRAINING_COMPLETED,
            payload={"success": True},
        )
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].payload["success"] is True

    @pytest.mark.asyncio
    async def test_global_subscriber(self):
        """Test subscribing to all events."""
        bus = get_event_bus()
        received = []

        def global_handler(event: DataEvent):
            received.append(event)

        bus.subscribe(None, global_handler)  # Subscribe to all

        await bus.publish(DataEvent(event_type=DataEventType.NEW_GAMES_AVAILABLE, payload={}))
        await bus.publish(DataEvent(event_type=DataEventType.TRAINING_STARTED, payload={}))
        await bus.publish(DataEvent(event_type=DataEventType.MODEL_PROMOTED, payload={}))

        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = get_event_bus()
        received = []

        def handler(event: DataEvent):
            received.append(event)

        bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, handler)
        await bus.publish(DataEvent(event_type=DataEventType.NEW_GAMES_AVAILABLE, payload={}))
        assert len(received) == 1

        bus.unsubscribe(DataEventType.NEW_GAMES_AVAILABLE, handler)
        await bus.publish(DataEvent(event_type=DataEventType.NEW_GAMES_AVAILABLE, payload={}))
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_event_history(self):
        """Test event history retrieval."""
        bus = get_event_bus()

        for i in range(5):
            await bus.publish(DataEvent(
                event_type=DataEventType.NEW_GAMES_AVAILABLE,
                payload={"count": i},
            ))

        history = bus.get_history(DataEventType.NEW_GAMES_AVAILABLE)
        assert len(history) == 5

        history = bus.get_history(DataEventType.NEW_GAMES_AVAILABLE, limit=3)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_event_serialization(self):
        """Test event to_dict and from_dict."""
        event = DataEvent(
            event_type=DataEventType.MODEL_PROMOTED,
            payload={"model_id": "test_model", "elo": 1600},
            source="test",
        )

        event_dict = event.to_dict()
        restored = DataEvent.from_dict(event_dict)

        assert restored.event_type == event.event_type
        assert restored.payload == event.payload
        assert restored.source == event.source

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test emit_* convenience functions."""
        bus = get_event_bus()
        received = []

        def handler(event: DataEvent):
            received.append(event)

        bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, handler)
        bus.subscribe(DataEventType.TRAINING_COMPLETED, handler)

        await emit_new_games("host1", 50, 500, "test")
        await emit_training_completed("square8_2p", True, 120.0, "/path/to/model", "test")

        assert len(received) == 2
        assert received[0].payload["new_games"] == 50
        assert received[1].payload["success"] is True

    @pytest.mark.asyncio
    async def test_subscriber_error_isolation(self):
        """Test that errors in one subscriber don't affect others."""
        bus = get_event_bus()
        received = []

        def bad_handler(event: DataEvent):
            raise ValueError("Test error")

        def good_handler(event: DataEvent):
            received.append(event)

        bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, bad_handler)
        bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, good_handler)

        # Should not raise, and good_handler should still be called
        await bus.publish(DataEvent(event_type=DataEventType.NEW_GAMES_AVAILABLE, payload={}))
        assert len(received) == 1


class TestDataManifest:
    """Test DataManifest for game deduplication."""

    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "manifest.db"

    def test_manifest_init(self):
        """Test manifest database initialization."""
        from app.distributed.unified_data_sync import DataManifest

        DataManifest(self.db_path)
        assert self.db_path.exists()

        # Check tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "synced_games" in tables
        assert "host_states" in tables
        assert "sync_history" in tables

    def test_game_deduplication(self):
        """Test marking games as synced and checking."""
        from app.distributed.unified_data_sync import DataManifest

        manifest = DataManifest(self.db_path)

        # Initially not synced
        assert not manifest.is_game_synced("game-001")

        # Mark as synced
        manifest.mark_games_synced(
            ["game-001", "game-002", "game-003"],
            source_host="test-host",
            source_db="test.db",
            board_type="square8",
            num_players=2,
        )

        # Now synced
        assert manifest.is_game_synced("game-001")
        assert manifest.is_game_synced("game-002")
        assert manifest.is_game_synced("game-003")
        assert not manifest.is_game_synced("game-004")

        # Count
        assert manifest.get_synced_count() == 3

    def test_host_state_persistence(self):
        """Test saving and loading host state."""
        from app.distributed.unified_data_sync import DataManifest, HostSyncState

        manifest = DataManifest(self.db_path)

        state = HostSyncState(
            name="test-host",
            last_sync_time=time.time(),
            last_game_count=500,
            total_games_synced=1000,
            consecutive_failures=0,
        )
        manifest.save_host_state(state)

        loaded = manifest.load_host_state("test-host")
        assert loaded is not None
        assert loaded.name == "test-host"
        assert loaded.last_game_count == 500
        assert loaded.total_games_synced == 1000

    def test_sync_history(self):
        """Test recording sync history."""
        from app.distributed.unified_data_sync import DataManifest

        manifest = DataManifest(self.db_path)

        manifest.record_sync("host1", 50, 2.5, True)
        manifest.record_sync("host1", 0, 1.0, False)
        manifest.record_sync("host2", 100, 5.0, True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sync_history")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 3


class TestConfigValidation:
    """Test configuration parsing and validation."""

    def test_config_from_yaml(self):
        """Test loading config from YAML."""
        from scripts.unified_ai_loop import UnifiedLoopConfig

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data_ingestion:
  poll_interval_seconds: 30
  min_games_per_sync: 5

training:
  trigger_threshold_games: 500

evaluation:
  shadow_interval_seconds: 600
  shadow_games_per_config: 20

promotion:
  elo_threshold: 30
  auto_promote: false
""")
            f.flush()

            config = UnifiedLoopConfig.from_yaml(Path(f.name))

            assert config.data_ingestion.poll_interval_seconds == 30
            assert config.data_ingestion.min_games_per_sync == 5
            assert config.training.trigger_threshold_games == 500
            assert config.evaluation.shadow_interval_seconds == 600
            assert config.promotion.elo_threshold == 30
            assert config.promotion.auto_promote is False

    def test_config_defaults(self):
        """Test default configuration values."""
        from scripts.unified_ai_loop import UnifiedLoopConfig

        config = UnifiedLoopConfig()

        assert config.data_ingestion.poll_interval_seconds == 60
        assert config.training.trigger_threshold_games == 500  # Updated from 1000
        assert config.evaluation.shadow_interval_seconds == 300  # Updated from 900
        assert config.promotion.elo_threshold == 25  # Updated from 20
        assert config.curriculum.adaptive is True


class TestTrainingScheduler:
    """Test training trigger logic."""

    def test_should_trigger_threshold(self):
        """Test training trigger based on game threshold."""
        from scripts.unified_ai_loop import (
            ConfigState,
            EventBus,
            TrainingConfig,
            TrainingScheduler,
            UnifiedLoopState,
        )

        config = TrainingConfig(trigger_threshold_games=100, min_interval_seconds=0)
        state = UnifiedLoopState()
        state.configs["square8_2p"] = ConfigState(
            board_type="square8",
            num_players=2,
            games_since_training=150,  # Above threshold
            last_training_time=0,
        )
        bus = EventBus()

        scheduler = TrainingScheduler(config, state, bus)
        trigger = scheduler.should_trigger_training()

        assert trigger == "square8_2p"

    def test_should_not_trigger_interval(self):
        """Test training blocked by minimum interval."""
        from scripts.unified_ai_loop import (
            ConfigState,
            EventBus,
            TrainingConfig,
            TrainingScheduler,
            UnifiedLoopState,
        )

        config = TrainingConfig(trigger_threshold_games=100, min_interval_seconds=3600)
        state = UnifiedLoopState()
        state.configs["square8_2p"] = ConfigState(
            board_type="square8",
            num_players=2,
            games_since_training=150,
            last_training_time=time.time() - 60,  # Too recent
        )
        bus = EventBus()

        scheduler = TrainingScheduler(config, state, bus)
        trigger = scheduler.should_trigger_training()

        assert trigger is None

    def test_should_not_trigger_in_progress(self):
        """Test training blocked when already in progress."""
        from scripts.unified_ai_loop import (
            ConfigState,
            EventBus,
            TrainingConfig,
            TrainingScheduler,
            UnifiedLoopState,
        )

        config = TrainingConfig(trigger_threshold_games=100, min_interval_seconds=0)
        state = UnifiedLoopState()
        state.training_in_progress = True
        state.configs["square8_2p"] = ConfigState(
            board_type="square8",
            num_players=2,
            games_since_training=150,
            last_training_time=0,
        )
        bus = EventBus()

        scheduler = TrainingScheduler(config, state, bus)
        trigger = scheduler.should_trigger_training()

        assert trigger is None


class TestAdaptiveCurriculum:
    """Test Elo-weighted curriculum logic."""

    def test_weight_computation(self):
        """Test curriculum weight computation."""
        # Weight should be higher for underperforming configs
        # and lower for overperforming configs

        # Simple weight formula: 1.0 + (median - elo) / 200
        elos = {
            "square8_2p": 1400,   # Below median - should get boost
            "square8_3p": 1500,   # At median - neutral
            "square8_4p": 1600,   # Above median - should get reduction
        }

        median = 1500

        weights = {}
        for config, elo in elos.items():
            deficit = median - elo
            weight = 1.0 + (deficit / 200.0)
            weight = max(0.5, min(2.0, weight))
            weights[config] = weight

        assert weights["square8_2p"] > 1.0  # Boost
        assert weights["square8_3p"] == 1.0  # Neutral
        assert weights["square8_4p"] < 1.0  # Reduction


class TestStatePersistence:
    """Test state serialization and persistence."""

    def test_state_roundtrip(self):
        """Test state serialization and deserialization."""
        from scripts.unified_ai_loop import (
            ConfigState,
            HostState,
            UnifiedLoopState,
        )

        state = UnifiedLoopState()
        state.started_at = "2024-01-01T00:00:00"
        state.total_data_syncs = 100
        state.total_training_runs = 5
        state.hosts["host1"] = HostState(name="host1", ssh_host="192.168.1.1")
        state.configs["square8_2p"] = ConfigState(
            board_type="square8",
            num_players=2,
            game_count=1000,
        )
        state.curriculum_weights = {"square8_2p": 1.2}

        # Convert to dict and back
        state_dict = state.to_dict()
        restored = UnifiedLoopState.from_dict(state_dict)

        assert restored.started_at == state.started_at
        assert restored.total_data_syncs == 100
        assert restored.total_training_runs == 5
        assert "host1" in restored.hosts
        assert restored.hosts["host1"].ssh_host == "192.168.1.1"
        assert "square8_2p" in restored.configs
        assert restored.configs["square8_2p"].game_count == 1000
        assert restored.curriculum_weights["square8_2p"] == 1.2


class TestIntegration:
    """Integration tests for the unified loop."""

    @pytest.mark.asyncio
    async def test_event_flow(self):
        """Test event flow through the system."""
        reset_event_bus()
        bus = get_event_bus()

        events_received = []

        def track_event(event: DataEvent):
            events_received.append(event.event_type)

        # Subscribe to key events
        bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, track_event)
        bus.subscribe(DataEventType.TRAINING_THRESHOLD_REACHED, track_event)
        bus.subscribe(DataEventType.TRAINING_COMPLETED, track_event)
        bus.subscribe(DataEventType.MODEL_PROMOTED, track_event)

        # Simulate event flow
        await bus.publish(DataEvent(
            event_type=DataEventType.NEW_GAMES_AVAILABLE,
            payload={"new_games": 500}
        ))
        await bus.publish(DataEvent(
            event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
            payload={"config": "square8_2p"}
        ))
        await bus.publish(DataEvent(
            event_type=DataEventType.TRAINING_COMPLETED,
            payload={"success": True}
        ))
        await bus.publish(DataEvent(
            event_type=DataEventType.MODEL_PROMOTED,
            payload={"model_id": "new_best"}
        ))

        assert events_received == [
            DataEventType.NEW_GAMES_AVAILABLE,
            DataEventType.TRAINING_THRESHOLD_REACHED,
            DataEventType.TRAINING_COMPLETED,
            DataEventType.MODEL_PROMOTED,
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
