"""Unit tests for event_driven_selfplay module.

Tests cover:
- SelfplayStats dataclass and computed properties
- EventDrivenSelfplay initialization and configuration
- Event subscription and model promotion handling
- Model hot-reload logic
- Statistics collection and reporting
- Lifecycle management (start/stop)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.training.event_driven_selfplay import (
    EventDrivenSelfplay,
    SelfplayStats,
    run_event_driven_selfplay,
)


# =============================================================================
# SelfplayStats Tests
# =============================================================================


class TestSelfplayStats:
    """Tests for SelfplayStats dataclass."""

    def test_default_initialization(self):
        """Test SelfplayStats with default values."""
        stats = SelfplayStats()
        assert stats.games_completed == 0
        assert stats.total_moves == 0
        assert stats.model_reloads == 0
        assert stats.current_model == ""
        assert isinstance(stats.start_time, float)
        assert stats.wins_by_player == {}

    def test_custom_initialization(self):
        """Test SelfplayStats with custom values."""
        wins = {1: 10, 2: 5}
        stats = SelfplayStats(
            games_completed=15,
            total_moves=450,
            model_reloads=2,
            current_model="models/test.pth",
            wins_by_player=wins,
        )
        assert stats.games_completed == 15
        assert stats.total_moves == 450
        assert stats.model_reloads == 2
        assert stats.current_model == "models/test.pth"
        assert stats.wins_by_player == wins

    def test_games_per_second_zero_games(self):
        """Test games_per_second with zero games."""
        stats = SelfplayStats()
        # With zero games, should return 0 (divided by time)
        assert stats.games_per_second == 0.0

    def test_games_per_second_calculation(self):
        """Test games_per_second calculation."""
        start_time = time.time() - 10.0  # 10 seconds ago
        stats = SelfplayStats(
            games_completed=50,
            start_time=start_time,
        )
        # 50 games in ~10 seconds = ~5 g/s
        assert 4.0 < stats.games_per_second < 6.0

    def test_elapsed_time(self):
        """Test elapsed_time property."""
        start_time = time.time() - 5.0
        stats = SelfplayStats(start_time=start_time)
        # Should be approximately 5 seconds
        assert 4.5 < stats.elapsed_time < 5.5

    def test_wins_by_player_modification(self):
        """Test modifying wins_by_player."""
        stats = SelfplayStats()
        stats.wins_by_player[1] = 10
        stats.wins_by_player[2] = 8
        assert stats.wins_by_player[1] == 10
        assert stats.wins_by_player[2] == 8


# =============================================================================
# EventDrivenSelfplay Initialization Tests
# =============================================================================


class TestEventDrivenSelfplayInit:
    """Tests for EventDrivenSelfplay initialization."""

    def test_basic_initialization(self, tmp_path):
        """Test basic initialization with required parameters."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path / "selfplay",
        )
        assert manager.board_type == "square8"
        assert manager.num_players == 2
        assert manager.batch_size == 16  # default
        assert manager.mcts_sims == 800  # default
        assert manager._config_key == "square8_2p"

    def test_custom_parameters(self, tmp_path):
        """Test initialization with custom parameters."""
        manager = EventDrivenSelfplay(
            board_type="hex8",
            num_players=4,
            batch_size=32,
            mcts_sims=400,
            max_moves=300,
            output_dir=tmp_path / "custom",
            prefer_nnue=False,
            use_gpu_mcts=True,
            gpu_device="cpu",
            gpu_eval_mode="nn",
        )
        assert manager.board_type == "hex8"
        assert manager.num_players == 4
        assert manager.batch_size == 32
        assert manager.mcts_sims == 400
        assert manager.max_moves == 300
        assert manager.prefer_nnue is False
        assert manager.use_gpu_mcts is True
        assert manager.gpu_device == "cpu"
        assert manager.gpu_eval_mode == "nn"
        assert manager._config_key == "hex8_4p"

    def test_output_directory_created(self, tmp_path):
        """Test that output directory is created on initialization."""
        output_dir = tmp_path / "new_selfplay_dir"
        assert not output_dir.exists()

        EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=output_dir,
        )

        assert output_dir.exists()

    def test_initial_state(self, tmp_path):
        """Test initial state of manager."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )
        assert manager._batched_mcts is None
        assert manager._current_model_path is None
        assert manager._model_update_pending is False
        assert manager._pending_model_path is None
        assert manager._running is False
        assert manager._event_subscription is None
        assert manager._callbacks == []


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription handling."""

    def test_subscribe_to_events_no_router(self, tmp_path):
        """Test subscription when event router is not available."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        with patch.dict("sys.modules", {"app.coordination.event_router": None}):
            # Should not raise - handles ImportError gracefully
            manager._subscribe_to_events()

        assert manager._event_subscription is None

    def test_subscribe_to_events_success(self, tmp_path):
        """Test successful event subscription."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        mock_router = MagicMock()
        mock_router.subscribe.return_value = "subscription_id"
        mock_get_router = MagicMock(return_value=mock_router)

        with patch(
            "app.coordination.event_router.get_event_router",
            mock_get_router,
        ):
            manager._subscribe_to_events()

        mock_router.subscribe.assert_called_once_with(
            "model_promoted",
            manager._on_model_promoted_sync,
        )
        assert manager._event_subscription == "subscription_id"

    def test_unsubscribe_from_events(self, tmp_path):
        """Test unsubscribing from events."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )
        manager._event_subscription = "test_subscription"

        mock_router = MagicMock()

        with patch(
            "app.coordination.event_router.get_event_router",
            return_value=mock_router,
        ):
            manager._unsubscribe_from_events()

        mock_router.unsubscribe.assert_called_once_with("test_subscription")
        assert manager._event_subscription is None


# =============================================================================
# Model Promotion Event Handling Tests
# =============================================================================


class TestModelPromotionHandling:
    """Tests for model promotion event handling."""

    def test_on_model_promoted_same_config(self, tmp_path):
        """Test handling promotion event for matching config."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "square8_2p",
            "model_path": "/path/to/model.pth",
        }

        manager._on_model_promoted_sync(event)

        assert manager._model_update_pending is True
        assert manager._pending_model_path == Path("/path/to/model.pth")

    def test_on_model_promoted_different_config(self, tmp_path):
        """Test ignoring promotion event for different config."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_4p",  # Different config
            "model_path": "/path/to/model.pth",
        }

        manager._on_model_promoted_sync(event)

        assert manager._model_update_pending is False
        assert manager._pending_model_path is None

    def test_on_model_promoted_dict_event(self, tmp_path):
        """Test handling dict event (no payload attribute)."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        # Event as dict (no payload attribute)
        event = {
            "config_key": "square8_2p",
            "model_path": "/path/to/model.pth",
        }

        manager._on_model_promoted_sync(event)

        assert manager._model_update_pending is True
        assert manager._pending_model_path == Path("/path/to/model.pth")

    def test_on_model_promoted_no_model_path(self, tmp_path):
        """Test handling event with no model path."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "square8_2p",
            # No model_path
        }

        manager._on_model_promoted_sync(event)

        # Should not set pending update without model path
        assert manager._model_update_pending is False


# =============================================================================
# Model Update Application Tests
# =============================================================================


class TestModelUpdateApplication:
    """Tests for applying model updates."""

    @pytest.mark.asyncio
    async def test_apply_model_update_no_pending(self, tmp_path):
        """Test apply_model_update with no pending update."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )
        manager._model_update_pending = True
        manager._pending_model_path = None

        await manager._apply_model_update()

        assert manager._model_update_pending is False

    @pytest.mark.asyncio
    async def test_apply_model_update_success(self, tmp_path):
        """Test successful model update application."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        # Set up pending update
        model_path = tmp_path / "new_model.pth"
        model_path.touch()
        manager._pending_model_path = model_path
        manager._model_update_pending = True

        # Mock neural net loading and MCTS
        mock_nn = MagicMock()
        manager._batched_mcts = MagicMock()

        with patch.object(
            manager,
            "_load_neural_net",
            new_callable=AsyncMock,
            return_value=mock_nn,
        ):
            await manager._apply_model_update()

        assert manager._model_update_pending is False
        assert manager._pending_model_path is None
        assert manager._current_model_path == model_path
        assert manager._stats.model_reloads == 1
        assert manager._batched_mcts.neural_net == mock_nn

    @pytest.mark.asyncio
    async def test_apply_model_update_callback(self, tmp_path):
        """Test that callbacks are called on model update."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        model_path = tmp_path / "callback_model.pth"
        model_path.touch()
        manager._pending_model_path = model_path
        manager._model_update_pending = True
        manager._batched_mcts = MagicMock()

        # Register callback
        callback = MagicMock()
        manager.on_model_update(callback)

        with patch.object(
            manager,
            "_load_neural_net",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ):
            await manager._apply_model_update()

        callback.assert_called_once_with(model_path)

    @pytest.mark.asyncio
    async def test_apply_model_update_load_failure(self, tmp_path):
        """Test handling failed model load."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        model_path = tmp_path / "bad_model.pth"
        manager._pending_model_path = model_path
        manager._model_update_pending = True
        manager._batched_mcts = MagicMock()

        with patch.object(
            manager,
            "_load_neural_net",
            new_callable=AsyncMock,
            return_value=None,  # Failed to load
        ):
            await manager._apply_model_update()

        # Should clear pending state even on failure
        assert manager._model_update_pending is False
        assert manager._pending_model_path is None
        # But stats should not update
        assert manager._stats.model_reloads == 0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_initializes_state(self, tmp_path):
        """Test that start() initializes manager state."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        mock_selector = MagicMock()
        mock_selector.get_current_model.return_value = None

        with patch(
            "app.training.selfplay_model_selector.SelfplayModelSelector",
            return_value=mock_selector,
        ), patch.object(
            manager, "_initialize_mcts", new_callable=AsyncMock
        ), patch.object(
            manager, "_subscribe_to_events"
        ):
            await manager.start()

        assert manager._running is True

    @pytest.mark.asyncio
    async def test_start_already_running(self, tmp_path, caplog):
        """Test that start() warns if already running."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )
        manager._running = True

        await manager.start()

        assert "already running" in caplog.text

    @pytest.mark.asyncio
    async def test_stop_clears_state(self, tmp_path):
        """Test that stop() clears running state."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )
        manager._running = True
        manager._stats.games_completed = 100

        with patch.object(manager, "_unsubscribe_from_events"):
            await manager.stop()

        assert manager._running is False


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics collection."""

    def test_get_stats(self, tmp_path):
        """Test get_stats() returns correct format."""
        manager = EventDrivenSelfplay(
            board_type="hex8",
            num_players=3,
            output_dir=tmp_path,
        )
        manager._stats.games_completed = 50
        manager._stats.total_moves = 1500
        manager._stats.model_reloads = 2
        manager._stats.current_model = "test_model.pth"
        manager._stats.wins_by_player = {1: 20, 2: 15, 3: 15}

        stats = manager.get_stats()

        assert stats["games_completed"] == 50
        assert stats["total_moves"] == 1500
        assert stats["model_reloads"] == 2
        assert stats["current_model"] == "test_model.pth"
        assert stats["wins_by_player"] == {1: 20, 2: 15, 3: 15}
        assert stats["config_key"] == "hex8_3p"
        assert "games_per_second" in stats
        assert "elapsed_time" in stats


# =============================================================================
# Callback Registration Tests
# =============================================================================


class TestCallbacks:
    """Tests for callback registration."""

    def test_on_model_update_registers_callback(self, tmp_path):
        """Test registering model update callback."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=2,
            output_dir=tmp_path,
        )

        callback1 = MagicMock()
        callback2 = MagicMock()

        manager.on_model_update(callback1)
        manager.on_model_update(callback2)

        assert len(manager._callbacks) == 2
        assert callback1 in manager._callbacks
        assert callback2 in manager._callbacks


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for run_event_driven_selfplay convenience function."""

    @pytest.mark.asyncio
    async def test_run_event_driven_selfplay_basic(self, tmp_path):
        """Test basic usage of convenience function."""
        with patch(
            "app.training.event_driven_selfplay.EventDrivenSelfplay"
        ) as MockManager:
            mock_instance = MagicMock()
            mock_instance.run_games = AsyncMock(return_value=[{"game": 1}])
            mock_instance.stop = AsyncMock()
            MockManager.return_value = mock_instance

            games = await run_event_driven_selfplay(
                board_type="square8",
                num_players=2,
                num_games=10,
            )

            assert games == [{"game": 1}]
            mock_instance.run_games.assert_called_once_with(10)
            mock_instance.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_event_driven_selfplay_custom_params(self, tmp_path):
        """Test convenience function with custom parameters."""
        with patch(
            "app.training.event_driven_selfplay.EventDrivenSelfplay"
        ) as MockManager:
            mock_instance = MagicMock()
            mock_instance.run_games = AsyncMock(return_value=[])
            mock_instance.stop = AsyncMock()
            MockManager.return_value = mock_instance

            await run_event_driven_selfplay(
                board_type="hex8",
                num_players=4,
                num_games=50,
                batch_size=32,
                mcts_sims=400,
            )

            MockManager.assert_called_once_with(
                board_type="hex8",
                num_players=4,
                batch_size=32,
                mcts_sims=400,
            )


# =============================================================================
# Board Type Mapping Tests
# =============================================================================


class TestBoardTypeMapping:
    """Tests for board type string to enum mapping."""

    @pytest.mark.parametrize("board_type,expected_key", [
        ("square8", "square8_2p"),
        ("SQUARE8", "square8_2p"),  # Case insensitive
        ("hex8", "hex8_2p"),
        ("hexagonal", "hexagonal_2p"),
        ("square19", "square19_2p"),
    ])
    def test_config_key_generation(self, tmp_path, board_type, expected_key):
        """Test config key generation for different board types."""
        manager = EventDrivenSelfplay(
            board_type=board_type,
            num_players=2,
            output_dir=tmp_path,
        )
        assert manager._config_key == expected_key

    @pytest.mark.parametrize("num_players,expected_suffix", [
        (2, "_2p"),
        (3, "_3p"),
        (4, "_4p"),
    ])
    def test_config_key_player_count(self, tmp_path, num_players, expected_suffix):
        """Test config key includes correct player count."""
        manager = EventDrivenSelfplay(
            board_type="square8",
            num_players=num_players,
            output_dir=tmp_path,
        )
        assert manager._config_key.endswith(expected_suffix)
