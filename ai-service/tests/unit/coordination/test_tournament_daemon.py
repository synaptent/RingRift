"""Tests for TournamentDaemon.

Tests the automatic tournament scheduling and evaluation functionality:
- Event subscription (TRAINING_COMPLETED, MODEL_PROMOTED)
- Evaluation queue processing
- Gauntlet integration
- Health check compliance
- Singleton pattern

December 2025: Added as part of coordination test coverage improvement.
"""

import asyncio
import pytest
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.tournament_daemon import (
    TournamentDaemon,
    TournamentDaemonConfig,
    TournamentStats,
    get_tournament_daemon,
    reset_tournament_daemon,
)


@pytest.fixture
def mock_config():
    """Create a test configuration."""
    return TournamentDaemonConfig(
        trigger_on_training_completed=True,
        trigger_on_model_promoted=False,
        enable_periodic_ladder=False,  # Disable for faster tests
        ladder_interval_seconds=3600.0,
        games_per_evaluation=5,
        games_per_baseline=2,
        baselines=["random", "heuristic"],
        max_concurrent_games=2,
        game_timeout_seconds=30.0,
        evaluation_timeout_seconds=60.0,
    )


@pytest.fixture
def daemon(mock_config):
    """Create a test daemon instance."""
    reset_tournament_daemon()
    return TournamentDaemon(config=mock_config)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Reset singleton after each test."""
    yield
    reset_tournament_daemon()


class TestTournamentDaemonConfig:
    """Test TournamentDaemonConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TournamentDaemonConfig()
        assert config.trigger_on_training_completed is True
        assert config.trigger_on_model_promoted is False
        assert config.enable_periodic_ladder is True
        assert config.ladder_interval_seconds == 3600.0
        assert config.games_per_evaluation == 20
        assert config.games_per_baseline == 10
        assert "random" in config.baselines
        assert "heuristic" in config.baselines

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TournamentDaemonConfig(
            games_per_baseline=50,
            baselines=["random", "heuristic", "mcts"],
            enable_periodic_ladder=False,
        )
        assert config.games_per_baseline == 50
        assert len(config.baselines) == 3
        assert config.enable_periodic_ladder is False


class TestTournamentStats:
    """Test TournamentStats dataclass."""

    def test_stats_initialization(self):
        """Test stats initialize to zero."""
        stats = TournamentStats()
        assert stats.event_triggers == 0
        assert stats.evaluations_completed == 0
        assert stats.games_played == 0

    def test_record_tournament_success(self):
        """Test recording successful tournament."""
        stats = TournamentStats()
        stats.record_tournament_success(games=10)
        assert stats.evaluations_completed == 1
        assert stats.games_played == 10

    def test_record_tournament_failure(self):
        """Test recording failed tournament."""
        stats = TournamentStats()
        stats.record_tournament_failure("Test error")
        assert stats.evaluations_failed == 1
        assert stats.last_error == "Test error"

    def test_record_event_trigger(self):
        """Test recording event triggers."""
        stats = TournamentStats()
        stats.record_event_trigger()
        assert stats.event_triggers == 1
        assert stats.evaluations_triggered == 1

    def test_backward_compatibility_aliases(self):
        """Test backward compatibility property aliases."""
        stats = TournamentStats()
        stats.record_tournament_success(games=5)
        stats.last_evaluation_time = time.time()

        assert stats.tournaments_completed == stats.evaluations_completed
        assert stats.last_tournament_time == stats.last_evaluation_time

    def test_errors_property(self):
        """Test errors property returns list."""
        stats = TournamentStats()
        assert stats.errors == []

        stats.record_tournament_failure("Error 1")
        assert stats.errors == ["Error 1"]


class TestTournamentDaemon:
    """Test TournamentDaemon functionality."""

    def test_daemon_initialization(self, daemon):
        """Test daemon initializes correctly."""
        assert daemon.config.games_per_baseline == 2
        assert daemon._running is False
        assert daemon._subscribed is False
        assert daemon._evaluation_queue.empty()

    def test_is_running(self, daemon):
        """Test is_running property."""
        assert daemon.is_running() is False
        daemon._running = True
        assert daemon.is_running() is True

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, daemon):
        """Test daemon start and stop lifecycle."""
        with patch.object(daemon, "_subscribe_to_events"):
            await daemon.start()

        assert daemon._running is True
        assert daemon._evaluation_task is not None

        await daemon.stop()

        assert daemon._running is False
        assert daemon._evaluation_task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon):
        """Test that calling start twice is handled."""
        with patch.object(daemon, "_subscribe_to_events"):
            await daemon.start()
            await daemon.start()  # Should log warning but not fail

        assert daemon._running is True
        await daemon.stop()

    def test_parse_config_standard(self, daemon):
        """Test parsing standard config strings."""
        board, players = daemon._parse_config("hex8_2p")
        assert board == "hex8"
        assert players == 2

        board, players = daemon._parse_config("square8_4p")
        assert board == "square8"
        assert players == 4

    def test_parse_config_hexagonal(self, daemon):
        """Test parsing hexagonal config."""
        board, players = daemon._parse_config("hexagonal_3p")
        assert board == "hexagonal"
        assert players == 3

    def test_parse_config_square19(self, daemon):
        """Test parsing square19 config."""
        board, players = daemon._parse_config("square19_2p")
        assert board == "square19"
        assert players == 2

    def test_parse_config_invalid(self, daemon):
        """Test parsing invalid config strings."""
        board, players = daemon._parse_config("invalid")
        assert board is None
        assert players is None

    def test_on_training_completed_queues_evaluation(self, daemon):
        """Test that TRAINING_COMPLETED event queues evaluation."""
        event = SimpleNamespace(payload={
            "model_path": "/path/to/model.pth",
            "config": "hex8_2p",
            "success": True,
        })

        daemon._on_training_completed(event)

        assert not daemon._evaluation_queue.empty()
        request = daemon._evaluation_queue.get_nowait()
        assert request["model_path"] == "/path/to/model.pth"
        assert request["board_type"] == "hex8"
        assert request["num_players"] == 2
        assert request["trigger"] == "training_completed"

    def test_on_training_completed_skips_failed(self, daemon):
        """Test that failed training is skipped."""
        event = SimpleNamespace(payload={
            "model_path": "/path/to/model.pth",
            "config": "hex8_2p",
            "success": False,
        })

        daemon._on_training_completed(event)
        assert daemon._evaluation_queue.empty()

    def test_on_training_completed_missing_path(self, daemon):
        """Test handling of missing model_path."""
        event = SimpleNamespace(payload={
            "config": "hex8_2p",
            "success": True,
        })

        daemon._on_training_completed(event)
        assert daemon._evaluation_queue.empty()

    def test_on_training_completed_invalid_config(self, daemon):
        """Test handling of invalid config."""
        event = SimpleNamespace(payload={
            "model_path": "/path/to/model.pth",
            "config": "invalid_config",
            "success": True,
        })

        daemon._on_training_completed(event)
        assert daemon._evaluation_queue.empty()

    def test_on_model_promoted_records_trigger(self, daemon):
        """Test MODEL_PROMOTED event handling."""
        event = SimpleNamespace(payload={
            "model_id": "model_123",
            "config": "hex8_2p",
        })

        daemon._on_model_promoted(event)
        assert daemon._stats.event_triggers == 1

    def test_get_status(self, daemon):
        """Test status reporting."""
        daemon._running = True
        daemon._subscribed = True

        status = daemon.get_status()

        assert status["running"] is True
        assert status["subscribed"] is True
        assert "queue_size" in status
        assert "stats" in status
        assert "config" in status

    def test_health_check_not_running(self, daemon):
        """Test health check when not running."""
        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    def test_health_check_running(self, daemon):
        """Test health check when running."""
        daemon._running = True
        daemon._stats.games_played = 50

        result = daemon.health_check()

        assert result.healthy is True
        assert "50 games" in result.message

    def test_health_check_high_errors(self, daemon):
        """Test health check with high error count."""
        daemon._running = True
        # Simulate many errors (errors_count > 10 triggers degraded status)
        for i in range(15):
            daemon._stats.record_failure(f"Error {i}")

        result = daemon.health_check()

        assert result.healthy is False
        assert "15 errors" in result.message

    @pytest.mark.asyncio
    async def test_emit_evaluation_completed(self, daemon):
        """Test emission of EVALUATION_COMPLETED event."""
        results = {
            "model_path": "/path/to/model.pth",
            "board_type": "hex8",
            "num_players": 2,
            "success": True,
            "win_rates": {"random": 0.9, "heuristic": 0.7},
            "elo": 1500,
            "games_played": 20,
        }

        with patch("app.coordination.event_router.publish", new_callable=AsyncMock) as mock_publish:
            await daemon._emit_evaluation_completed(results)

            mock_publish.assert_called_once()
            call_args = mock_publish.call_args
            payload = call_args.kwargs.get("payload") or call_args[1].get("payload")
            assert payload["model_path"] == "/path/to/model.pth"
            assert payload["success"] is True

    @pytest.mark.asyncio
    async def test_run_basic_evaluation(self, daemon):
        """Test fallback basic evaluation."""
        with patch("app.models.BoardType") as mock_board_type:
            mock_board_type.return_value = MagicMock()

            with patch("app.tournament.scheduler.RoundRobinScheduler") as mock_scheduler:
                mock_scheduler.return_value.generate_matches.return_value = [
                    MagicMock(), MagicMock()
                ]

                results = await daemon._run_basic_evaluation(
                    model_path="/path/to/model.pth",
                    board_type="hex8",
                    num_players=2,
                )

        assert results["success"] is True
        assert results["scheduled_matches"] == 2

    @pytest.mark.asyncio
    async def test_run_evaluation_timeout(self, daemon):
        """Test evaluation timeout handling."""
        daemon.config.evaluation_timeout_seconds = 0.05  # Very short timeout

        with patch("app.training.game_gauntlet.run_baseline_gauntlet") as mock_gauntlet:
            import time as time_module
            def slow_gauntlet(*args, **kwargs):
                # Sync sleep (runs in thread via asyncio.to_thread)
                time_module.sleep(1.0)
                return MagicMock()

            mock_gauntlet.side_effect = slow_gauntlet

            with patch.object(daemon, "_emit_evaluation_completed", new_callable=AsyncMock):
                results = await daemon._run_evaluation(
                    model_path="/path/to/model.pth",
                    board_type="hex8",
                    num_players=2,
                )

        assert results["success"] is False
        assert results.get("error") == "timeout"


class TestTournamentDaemonSingleton:
    """Test singleton pattern."""

    def test_get_tournament_daemon_singleton(self):
        """Test that get_tournament_daemon returns singleton."""
        daemon1 = get_tournament_daemon()
        daemon2 = get_tournament_daemon()

        assert daemon1 is daemon2

    def test_reset_tournament_daemon(self):
        """Test that reset clears singleton."""
        daemon1 = get_tournament_daemon()
        reset_tournament_daemon()
        daemon2 = get_tournament_daemon()

        assert daemon1 is not daemon2

    def test_get_tournament_daemon_with_config(self):
        """Test singleton initialization with config."""
        config = TournamentDaemonConfig(games_per_baseline=100)
        daemon = get_tournament_daemon(config)

        assert daemon.config.games_per_baseline == 100


class TestTournamentDaemonSubscription:
    """Test event subscription."""

    def test_subscribe_to_events_training_completed(self, daemon):
        """Test subscription to TRAINING_COMPLETED."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            daemon._subscribe_to_events()

            assert daemon._subscribed is True
            mock_router.subscribe.assert_called()

    def test_subscribe_to_events_model_promoted(self, mock_config):
        """Test subscription to MODEL_PROMOTED when enabled."""
        mock_config.trigger_on_model_promoted = True
        daemon = TournamentDaemon(config=mock_config)

        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            daemon._subscribe_to_events()

            # Should have subscribed to both events
            assert mock_router.subscribe.call_count == 2

    def test_subscribe_to_events_idempotent(self, daemon):
        """Test that subscription is idempotent."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            daemon._subscribe_to_events()
            daemon._subscribe_to_events()  # Second call

            # Should only subscribe once
            assert mock_router.subscribe.call_count == 1


class TestTournamentDaemonEvaluationWorker:
    """Test evaluation worker."""

    @pytest.mark.asyncio
    async def test_evaluation_worker_processes_queue(self, daemon):
        """Test that worker processes queue items."""
        daemon._running = True

        # Queue an evaluation
        daemon._evaluation_queue.put_nowait({
            "model_path": "/path/to/model.pth",
            "board_type": "hex8",
            "num_players": 2,
            "trigger": "test",
        })

        with patch.object(daemon, "_run_evaluation", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = {"success": True}

            # Start worker and give it time to process
            worker_task = asyncio.create_task(daemon._evaluation_worker())
            await asyncio.sleep(0.1)

            # Stop worker
            daemon._running = False
            worker_task.cancel()

            try:
                await worker_task
            except asyncio.CancelledError:
                pass

            mock_eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_worker_handles_errors(self, daemon):
        """Test that worker handles evaluation errors."""
        daemon._running = True

        daemon._evaluation_queue.put_nowait({
            "model_path": "/path/to/model.pth",
            "board_type": "hex8",
            "num_players": 2,
            "trigger": "test",
        })

        with patch.object(daemon, "_run_evaluation", new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = Exception("Test error")

            worker_task = asyncio.create_task(daemon._evaluation_worker())
            await asyncio.sleep(0.1)

            daemon._running = False
            worker_task.cancel()

            try:
                await worker_task
            except asyncio.CancelledError:
                pass

            # Error should be recorded via record_failure()
            assert daemon._stats.errors_count > 0


class TestTournamentDaemonEloUpdate:
    """Test ELO update functionality."""

    @pytest.mark.asyncio
    async def test_update_elo_registers_model(self, daemon):
        """Test that ELO update registers model."""
        gauntlet_results = SimpleNamespace(
            estimated_elo=1500,
            opponent_results={"random": {"games": 10}}
        )

        with patch("app.training.elo_service.get_elo_service") as mock_get_elo:
            mock_elo = MagicMock()
            mock_get_elo.return_value = mock_elo

            await daemon._update_elo(
                model_path="/path/to/model.pth",
                board_type="hex8",
                num_players=2,
                gauntlet_results=gauntlet_results,
            )

            mock_elo.register_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_elo_handles_import_error(self, daemon):
        """Test graceful handling when EloService unavailable."""
        gauntlet_results = SimpleNamespace(estimated_elo=1500)

        with patch.dict("sys.modules", {"app.training.elo_service": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                # Should not raise
                await daemon._update_elo(
                    model_path="/path/to/model.pth",
                    board_type="hex8",
                    num_players=2,
                    gauntlet_results=gauntlet_results,
                )
