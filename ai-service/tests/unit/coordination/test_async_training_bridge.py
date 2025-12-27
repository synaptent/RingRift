"""Tests for async_training_bridge.py - Async/sync boundary for training.

Tests cover:
- TrainingProgressEvent dataclass
- AsyncTrainingBridge class methods
- Singleton pattern (get/reset)
- Progress callbacks
- Event emission on completion
- Convenience async functions
- Error handling in async wrappers
"""

import asyncio
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.coordination.async_training_bridge import (
    TrainingProgressEvent,
    AsyncTrainingBridge,
    get_training_bridge,
    reset_training_bridge,
    async_can_train,
    async_request_training,
    async_update_progress,
    async_complete_training,
    async_get_training_status,
)


class TestTrainingProgressEvent:
    """Tests for TrainingProgressEvent dataclass."""

    def test_creation(self):
        """Test creating a progress event."""
        event = TrainingProgressEvent(
            job_id="square8_2p_12345_678",
            board_type="square8",
            num_players=2,
            epochs_completed=10,
            best_val_loss=0.05,
            current_elo=1500.0,
        )
        assert event.job_id == "square8_2p_12345_678"
        assert event.board_type == "square8"
        assert event.num_players == 2
        assert event.epochs_completed == 10
        assert event.best_val_loss == 0.05
        assert event.current_elo == 1500.0

    def test_to_dict(self):
        """Test converting event to dict."""
        event = TrainingProgressEvent(
            job_id="hex8_4p_99999_111",
            board_type="hex8",
            num_players=4,
            epochs_completed=50,
            best_val_loss=0.02,
            current_elo=1650.0,
        )
        d = asdict(event)
        assert d["job_id"] == "hex8_4p_99999_111"
        assert d["epochs_completed"] == 50


class TestAsyncTrainingBridge:
    """Tests for AsyncTrainingBridge class."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock TrainingCoordinator."""
        coordinator = MagicMock()
        coordinator.can_start_training.return_value = True
        coordinator.start_training.return_value = "test_job_123"
        coordinator.update_progress.return_value = True
        coordinator.complete_training.return_value = True
        coordinator.get_active_jobs.return_value = []
        coordinator.get_status.return_value = {"active": 0, "pending": 0}
        return coordinator

    @pytest.fixture
    def mock_job(self):
        """Create a mock TrainingJob."""
        job = MagicMock()
        job.board_type = "square8"
        job.num_players = 2
        job.job_id = "square8_2p_12345_678"
        return job

    @pytest.fixture
    def bridge(self, mock_coordinator):
        """Create a bridge with mocked coordinator."""
        return AsyncTrainingBridge(coordinator=mock_coordinator, emit_events=False)

    @pytest.mark.asyncio
    async def test_can_start_training(self, bridge, mock_coordinator):
        """Test can_start_training async wrapper."""
        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = True

            result = await bridge.can_start_training("square8", 2)

            assert result is True
            mock_run.assert_called_once_with(
                mock_coordinator.can_start_training,
                "square8",
                2
            )

    @pytest.mark.asyncio
    async def test_request_training_slot_success(self, bridge, mock_coordinator):
        """Test successful training slot request."""
        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "job_12345"

            job_id = await bridge.request_training_slot("hex8", 4, "v2")

            assert job_id == "job_12345"
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_training_slot_no_slot(self, bridge, mock_coordinator):
        """Test when no training slot available."""
        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = None

            job_id = await bridge.request_training_slot("square8", 2)

            assert job_id is None

    @pytest.mark.asyncio
    async def test_update_progress(self, bridge, mock_coordinator, mock_job):
        """Test progress update."""
        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = True

            result = await bridge.update_progress(
                "square8_2p_12345",
                epochs_completed=20,
                best_val_loss=0.03,
                current_elo=1550.0,
            )

            assert result is True
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_progress_with_callback(self, bridge, mock_coordinator, mock_job):
        """Test progress update triggers callbacks."""
        callback_received = []

        def callback(event):
            callback_received.append(event)

        bridge.on_progress(callback)

        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = [True, mock_job]  # update_progress, get_job

            with patch.object(bridge, "get_job_by_id", new_callable=AsyncMock) as mock_get_job:
                mock_get_job.return_value = mock_job

                await bridge.update_progress(
                    "square8_2p_12345_678",
                    epochs_completed=30,
                    best_val_loss=0.02,
                    current_elo=1600.0,
                )

        assert len(callback_received) == 1
        assert callback_received[0].epochs_completed == 30

    @pytest.mark.asyncio
    async def test_complete_training_emits_event(self, mock_coordinator, mock_job):
        """Test that completing training emits event when enabled."""
        bridge = AsyncTrainingBridge(coordinator=mock_coordinator, emit_events=True)

        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = True

            with patch.object(bridge, "get_job_by_id", new_callable=AsyncMock) as mock_get_job:
                mock_get_job.return_value = mock_job

                with patch(
                    "app.coordination.async_training_bridge.emit_training_complete",
                    new_callable=AsyncMock,
                ) as mock_emit:
                    await bridge.complete_training(
                        "square8_2p_12345_678",
                        status="completed",
                        final_val_loss=0.02,
                        final_elo=1650.0,
                        model_path="/models/test.pth",
                    )

                    mock_emit.assert_called_once()
                    call_kwargs = mock_emit.call_args.kwargs
                    assert call_kwargs["job_id"] == "square8_2p_12345_678"
                    assert call_kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_complete_training_no_event_when_disabled(self, bridge, mock_coordinator, mock_job):
        """Test no event emitted when emit_events=False."""
        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = True

            with patch.object(bridge, "get_job_by_id", new_callable=AsyncMock) as mock_get_job:
                mock_get_job.return_value = mock_job

                with patch(
                    "app.coordination.async_training_bridge.emit_training_complete",
                    new_callable=AsyncMock,
                ) as mock_emit:
                    await bridge.complete_training("test_job", status="completed")

                    # Event should not be emitted when emit_events=False
                    mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_active_jobs(self, bridge, mock_coordinator):
        """Test getting active jobs."""
        mock_jobs = [MagicMock(job_id="job1"), MagicMock(job_id="job2")]

        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_jobs

            jobs = await bridge.get_active_jobs()

            assert len(jobs) == 2
            mock_run.assert_called_once_with(mock_coordinator.get_active_jobs)

    @pytest.mark.asyncio
    async def test_get_training_status(self, bridge, mock_coordinator):
        """Test getting training status."""
        expected_status = {"active": 3, "pending": 5}

        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = expected_status

            status = await bridge.get_training_status()

            assert status == expected_status
            mock_run.assert_called_once_with(mock_coordinator.get_status)

    @pytest.mark.asyncio
    async def test_get_job_by_id_parses_format(self, bridge, mock_coordinator, mock_job):
        """Test job ID parsing from format board_type_numplayersp_timestamp_pid."""
        with patch.object(bridge, "_run_sync", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_job

            job = await bridge.get_job_by_id("square8_2p_12345_678")

            assert job is mock_job
            mock_run.assert_called_once()
            # Should parse board_type="square8" and num_players=2
            call_args = mock_run.call_args
            assert call_args[0][1] == "square8"  # board_type
            assert call_args[0][2] == 2  # num_players

    @pytest.mark.asyncio
    async def test_get_job_by_id_invalid_format(self, bridge):
        """Test handling of invalid job ID format."""
        job = await bridge.get_job_by_id("invalid")

        # Should return None on parse error
        assert job is None

    def test_progress_callback_registration(self, bridge):
        """Test registering progress callbacks."""
        def callback(event):
            pass

        assert len(bridge._progress_callbacks) == 0

        bridge.on_progress(callback)
        assert len(bridge._progress_callbacks) == 1

        bridge.off_progress(callback)
        assert len(bridge._progress_callbacks) == 0

    def test_off_progress_nonexistent(self, bridge):
        """Test removing callback that does not exist."""
        def callback(event):
            pass

        # Should not raise
        bridge.off_progress(callback)
        assert len(bridge._progress_callbacks) == 0


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_training_bridge()

    def test_get_training_bridge_creates_singleton(self):
        """Test that get_training_bridge creates a singleton."""
        with patch(
            "app.coordination.async_training_bridge.get_training_coordinator"
        ) as mock_get_coord:
            mock_get_coord.return_value = MagicMock()

            bridge1 = get_training_bridge()
            bridge2 = get_training_bridge()

            assert bridge1 is bridge2

    def test_reset_training_bridge(self):
        """Test that reset clears the singleton."""
        with patch(
            "app.coordination.async_training_bridge.get_training_coordinator"
        ) as mock_get_coord:
            mock_get_coord.return_value = MagicMock()

            bridge1 = get_training_bridge()
            reset_training_bridge()
            bridge2 = get_training_bridge()

            assert bridge1 is not bridge2

    def test_get_training_bridge_with_custom_coordinator(self):
        """Test creating bridge with custom coordinator."""
        custom_coordinator = MagicMock()

        bridge = get_training_bridge(coordinator=custom_coordinator)

        assert bridge._coordinator is custom_coordinator

    def test_get_training_bridge_emit_events_param(self):
        """Test that emit_events parameter is passed."""
        with patch(
            "app.coordination.async_training_bridge.get_training_coordinator"
        ) as mock_get_coord:
            mock_get_coord.return_value = MagicMock()

            bridge = get_training_bridge(emit_events=False)

            assert bridge._emit_events is False


class TestConvenienceFunctions:
    """Tests for convenience async functions."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_training_bridge()

    @pytest.mark.asyncio
    async def test_async_can_train(self):
        """Test async_can_train convenience function."""
        with patch(
            "app.coordination.async_training_bridge.get_training_bridge"
        ) as mock_get:
            mock_bridge = MagicMock()
            mock_bridge.can_start_training = AsyncMock(return_value=True)
            mock_get.return_value = mock_bridge

            result = await async_can_train("hex8", 2)

            assert result is True
            mock_bridge.can_start_training.assert_called_once_with("hex8", 2)

    @pytest.mark.asyncio
    async def test_async_request_training(self):
        """Test async_request_training convenience function."""
        with patch(
            "app.coordination.async_training_bridge.get_training_bridge"
        ) as mock_get:
            mock_bridge = MagicMock()
            mock_bridge.request_training_slot = AsyncMock(return_value="job_abc")
            mock_get.return_value = mock_bridge

            result = await async_request_training("square8", 4, "v3", {"key": "val"})

            assert result == "job_abc"
            mock_bridge.request_training_slot.assert_called_once_with(
                "square8", 4, "v3", {"key": "val"}
            )

    @pytest.mark.asyncio
    async def test_async_update_progress(self):
        """Test async_update_progress convenience function."""
        with patch(
            "app.coordination.async_training_bridge.get_training_bridge"
        ) as mock_get:
            mock_bridge = MagicMock()
            mock_bridge.update_progress = AsyncMock(return_value=True)
            mock_get.return_value = mock_bridge

            result = await async_update_progress(
                "job_123",
                epochs_completed=15,
                best_val_loss=0.04,
                current_elo=1520.0,
            )

            assert result is True
            mock_bridge.update_progress.assert_called_once_with(
                "job_123", 15, 0.04, 1520.0
            )

    @pytest.mark.asyncio
    async def test_async_complete_training(self):
        """Test async_complete_training convenience function."""
        with patch(
            "app.coordination.async_training_bridge.get_training_bridge"
        ) as mock_get:
            mock_bridge = MagicMock()
            mock_bridge.complete_training = AsyncMock(return_value=True)
            mock_get.return_value = mock_bridge

            result = await async_complete_training(
                "job_456",
                status="completed",
                final_val_loss=0.02,
                final_elo=1650.0,
                model_path="/models/best.pth",
            )

            assert result is True
            mock_bridge.complete_training.assert_called_once_with(
                "job_456", "completed", 0.02, 1650.0, "/models/best.pth"
            )

    @pytest.mark.asyncio
    async def test_async_get_training_status(self):
        """Test async_get_training_status convenience function."""
        expected_status = {"active": 2, "queued": 5}

        with patch(
            "app.coordination.async_training_bridge.get_training_bridge"
        ) as mock_get:
            mock_bridge = MagicMock()
            mock_bridge.get_training_status = AsyncMock(return_value=expected_status)
            mock_get.return_value = mock_bridge

            result = await async_get_training_status()

            assert result == expected_status


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_exist(self):
        """Test that __all__ exports are valid."""
        from app.coordination import async_training_bridge

        for name in async_training_bridge.__all__:
            assert hasattr(async_training_bridge, name), f"{name} not in module"

    def test_imports(self):
        """Test that all expected items are importable."""
        from app.coordination.async_training_bridge import (
            AsyncTrainingBridge,
            TrainingProgressEvent,
            async_can_train,
            async_complete_training,
            async_get_training_status,
            async_request_training,
            async_update_progress,
            get_training_bridge,
            reset_training_bridge,
        )

        # All imports should be non-None
        assert AsyncTrainingBridge is not None
        assert TrainingProgressEvent is not None
        assert async_can_train is not None
        assert get_training_bridge is not None


class TestProgressCallbackErrors:
    """Tests for error handling in progress callbacks."""

    @pytest.fixture
    def bridge_with_coordinator(self):
        """Create bridge with mock coordinator."""
        mock_coordinator = MagicMock()
        mock_coordinator.update_progress.return_value = True
        bridge = AsyncTrainingBridge(coordinator=mock_coordinator, emit_events=False)
        return bridge

    @pytest.mark.asyncio
    async def test_callback_error_does_not_propagate(self, bridge_with_coordinator):
        """Test that callback errors do not break the update."""
        def bad_callback(event):
            raise ValueError("Callback error!")

        bridge_with_coordinator.on_progress(bad_callback)

        mock_job = MagicMock()
        mock_job.board_type = "hex8"
        mock_job.num_players = 2

        with patch.object(
            bridge_with_coordinator, "_run_sync", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = True

            with patch.object(
                bridge_with_coordinator, "get_job_by_id", new_callable=AsyncMock
            ) as mock_get_job:
                mock_get_job.return_value = mock_job

                # Should not raise despite callback error
                result = await bridge_with_coordinator.update_progress(
                    "hex8_2p_123_456",
                    epochs_completed=5,
                )

                assert result is True

    @pytest.mark.asyncio
    async def test_multiple_callbacks_continue_on_error(self, bridge_with_coordinator):
        """Test that other callbacks run even if one fails."""
        received = []

        def good_callback(event):
            received.append(event.epochs_completed)

        def bad_callback(event):
            raise RuntimeError("Boom!")

        bridge_with_coordinator.on_progress(bad_callback)
        bridge_with_coordinator.on_progress(good_callback)

        mock_job = MagicMock()
        mock_job.board_type = "square8"
        mock_job.num_players = 4

        with patch.object(
            bridge_with_coordinator, "_run_sync", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = True

            with patch.object(
                bridge_with_coordinator, "get_job_by_id", new_callable=AsyncMock
            ) as mock_get_job:
                mock_get_job.return_value = mock_job

                await bridge_with_coordinator.update_progress(
                    "square8_4p_999_888",
                    epochs_completed=25,
                )

        # Good callback should still have been called
        assert 25 in received
