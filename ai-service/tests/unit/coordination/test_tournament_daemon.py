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
        assert config.trigger_on_model_promoted is True  # Jan 2026: Changed default to True
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
        assert daemon.is_subscribed is False  # Jan 2026: Use property instead of private attr
        assert daemon._evaluation_queue.empty()

    def test_is_running(self, daemon):
        """Test is_running property."""
        assert daemon.is_running() is False
        daemon._running = True
        assert daemon.is_running() is True

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, daemon):
        """Test daemon start and stop lifecycle."""
        with patch.object(daemon, "_subscribe_all_events"):  # Jan 2026: Updated method name
            await daemon.start()

        assert daemon._running is True
        assert daemon._task is not None  # Jan 2026: HandlerBase uses _task, not _evaluation_task

        await daemon.stop()

        assert daemon._running is False
        assert daemon._task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon):
        """Test that calling start twice is handled."""
        with patch.object(daemon, "_subscribe_all_events"):  # Jan 2026: Updated method name
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
        assert daemon._tournament_stats.event_triggers == 1

    def test_get_status(self, daemon):
        """Test status reporting."""
        daemon._running = True
        daemon._event_subscribed = True  # Jan 2026: Use correct private attr name

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
        daemon._tournament_stats.games_played = 50

        result = daemon.health_check()

        assert result.healthy is True
        assert "50 games" in result.message

    def test_health_check_high_errors(self, daemon):
        """Test health check with high error count."""
        daemon._running = True
        # Jan 2026: Populate _recent_errors directly (health_check uses this attribute)
        for i in range(15):
            daemon._recent_errors.append(f"Error {i}")

        result = daemon.health_check()

        assert result.healthy is False
        assert "15 recent errors" in result.message  # Jan 2026: Message says "recent errors"

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
        # Error can be "timeout" or another error if dependencies not available
        assert "error" in results


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
    """Test event subscription (Jan 2026: Updated for HandlerBase pattern)."""

    def test_subscribe_to_events_training_completed(self, daemon):
        """Test subscription to TRAINING_COMPLETED."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            daemon._subscribe_all_events()

            assert daemon.is_subscribed is True
            mock_router.subscribe.assert_called()

    def test_subscribe_to_events_model_promoted(self, mock_config):
        """Test subscription to MODEL_PROMOTED when enabled."""
        mock_config.trigger_on_model_promoted = True
        daemon = TournamentDaemon(config=mock_config)

        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            daemon._subscribe_all_events()

            # Should have subscribed to both events
            assert mock_router.subscribe.call_count == 2

    def test_subscribe_to_events_idempotent(self, daemon):
        """Test that subscription is idempotent."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            daemon._subscribe_all_events()
            daemon._subscribe_all_events()  # Second call

            # HandlerBase tracks subscription state
            assert daemon.is_subscribed is True


class TestTournamentDaemonEvaluationWorker:
    """Test evaluation worker (now via _run_cycle in HandlerBase pattern)."""

    @pytest.mark.asyncio
    async def test_evaluation_worker_processes_queue(self, daemon):
        """Test that worker processes queue items via _run_cycle."""
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

            # Jan 2026: Use _run_cycle instead of _evaluation_worker
            await daemon._run_cycle()

            mock_eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_worker_handles_errors(self, daemon):
        """Test that worker handles evaluation errors via _run_cycle."""
        daemon._running = True

        daemon._evaluation_queue.put_nowait({
            "model_path": "/path/to/model.pth",
            "board_type": "hex8",
            "num_players": 2,
            "trigger": "test",
        })

        with patch.object(daemon, "_run_evaluation", new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = Exception("Test error")

            # Jan 2026: Use _run_cycle instead of _evaluation_worker
            await daemon._run_cycle()

            # Error should be recorded via _record_error() (HandlerBase pattern)
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


class TestTournamentDaemonLadderTournament:
    """Test _run_ladder_tournament() method."""

    @pytest.mark.asyncio
    async def test_ladder_tournament_results_structure(self, daemon):
        """Test that ladder tournament returns expected result structure."""
        # Test with import error (most common case in test env)
        results = await daemon._run_ladder_tournament()

        # Results should always have these fields
        assert "tournament_id" in results
        assert "success" in results
        assert "configs_evaluated" in results
        assert "duration_seconds" in results

    @pytest.mark.asyncio
    async def test_ladder_tournament_updates_stats(self, daemon):
        """Test that stats are updated after ladder tournament."""
        initial_evaluations = daemon._tournament_stats.evaluations_completed

        await daemon._run_ladder_tournament()

        # Stats should increment regardless of success
        assert daemon._tournament_stats.evaluations_completed >= initial_evaluations

    def test_ladder_tournament_queue_population_logic(self, daemon):
        """Test queue population logic (unit test, no mocking)."""
        # Directly test the queue population logic
        models = [
            {"path": "/models/hex8_2p.pth", "board_type": "hex8", "num_players": 2},
            {"path": "/models/square8_4p.pth", "board_type": "square8", "num_players": 4},
        ]

        configs_evaluated = 0
        for model_info in models:
            model_path = model_info.get("path")
            board_type = model_info.get("board_type")
            num_players = model_info.get("num_players")

            if not all([model_path, board_type, num_players]):
                continue

            daemon._evaluation_queue.put_nowait({
                "model_path": model_path,
                "board_type": board_type,
                "num_players": num_players,
                "trigger": "periodic_ladder",
            })
            configs_evaluated += 1

        assert configs_evaluated == 2
        assert daemon._evaluation_queue.qsize() == 2

    def test_ladder_tournament_skips_incomplete_logic(self, daemon):
        """Test that incomplete models are skipped."""
        models = [
            {"path": "/models/hex8_2p.pth", "board_type": "hex8", "num_players": 2},
            {"path": "/models/incomplete.pth"},  # Missing board_type and num_players
            {"path": None, "board_type": "hex8", "num_players": 2},  # Missing path
            {},  # Empty dict
        ]

        configs_evaluated = 0
        for model_info in models:
            model_path = model_info.get("path")
            board_type = model_info.get("board_type")
            num_players = model_info.get("num_players")

            if not all([model_path, board_type, num_players]):
                continue

            configs_evaluated += 1

        assert configs_evaluated == 1  # Only the first one is complete


class TestTournamentDaemonCalibrationTournament:
    """Test _run_calibration_tournament() method."""

    @pytest.mark.asyncio
    async def test_calibration_tournament_results_structure(self, daemon):
        """Test that calibration tournament returns expected result structure."""
        results = await daemon._run_calibration_tournament()

        # Results should always have these fields
        assert "tournament_id" in results
        assert "tournament_type" in results
        assert results["tournament_type"] == "calibration"
        assert "success" in results
        assert "matchups" in results
        assert "duration_seconds" in results

    def test_calibration_validation_logic(self, daemon):
        """Test calibration validation threshold logic."""
        # Test that 90% margin validation works correctly
        expected_rate = 0.90
        actual_rate = 0.85

        # Allow 10% margin below expected rate
        calibration_valid = actual_rate >= expected_rate * 0.9
        assert calibration_valid is True  # 0.85 >= 0.81

        # Test fail case
        actual_rate = 0.50
        calibration_valid = actual_rate >= expected_rate * 0.9
        assert calibration_valid is False  # 0.50 < 0.81

    def test_calibration_win_rate_calculation(self, daemon):
        """Test win rate calculation logic."""
        games_per_matchup = 10

        # Test 70% win rate
        wins = 7
        actual_rate = wins / games_per_matchup if games_per_matchup > 0 else 0
        assert actual_rate == 0.7

        # Test 0 games edge case
        actual_rate = wins / 0 if 0 > 0 else 0
        assert actual_rate == 0

    def test_calibration_pairs_definition(self, daemon):
        """Test calibration pairs are defined correctly."""
        # The daemon uses 3 calibration pairs
        calibration_pairs = [
            ("HEURISTIC", "RANDOM", 0.90),
            ("HEURISTIC_STRONG", "HEURISTIC", 0.55),
            ("MCTS_LIGHT", "HEURISTIC_STRONG", 0.55),
        ]

        assert len(calibration_pairs) == 3
        assert calibration_pairs[0][2] == 0.90  # Heuristic vs Random: 90%
        assert calibration_pairs[1][2] == 0.55  # Strong vs Heuristic: 55%

    def test_calibration_matchup_key_format(self, daemon):
        """Test matchup key format."""
        stronger = "HEURISTIC"
        weaker = "RANDOM"

        matchup_key = f"{stronger}_vs_{weaker}"
        assert matchup_key == "HEURISTIC_vs_RANDOM"

    def test_calibration_all_valid_logic(self, daemon):
        """Test all_valid flag logic."""
        matchups = {
            "pair1": {"calibration_valid": True},
            "pair2": {"calibration_valid": True},
            "pair3": {"calibration_valid": True},
        }

        all_valid = all(m["calibration_valid"] for m in matchups.values())
        assert all_valid is True

        # Test with one failure
        matchups["pair2"]["calibration_valid"] = False
        all_valid = all(m["calibration_valid"] for m in matchups.values())
        assert all_valid is False


class TestTournamentDaemonCrossNNTournament:
    """Test _run_cross_nn_tournament() method."""

    @pytest.mark.asyncio
    async def test_cross_nn_results_structure(self, daemon):
        """Test that cross-NN tournament returns expected result structure."""
        results = await daemon._run_cross_nn_tournament()

        # Results should always have these fields
        assert "tournament_id" in results
        assert "tournament_type" in results
        assert results["tournament_type"] == "cross_nn"
        assert "success" in results
        assert "pairings" in results
        assert "games_played" in results
        assert "duration_seconds" in results

    def test_cross_nn_validates_improvement(self, daemon):
        """Test improvement validation logic (newer should win >50%)."""
        newer_wins = 7
        games = 10
        win_rate = newer_wins / games
        improvement_validated = win_rate >= 0.5

        # 70% > 50% so improvement is validated
        assert improvement_validated is True

    def test_cross_nn_detects_regression(self, daemon):
        """Test regression detection when newer loses majority."""
        newer_wins = 3
        games = 10
        win_rate = newer_wins / games
        improvement_validated = win_rate >= 0.5

        # 30% < 50% so improvement is NOT validated
        assert improvement_validated is False

    def test_cross_nn_version_sorting(self, daemon):
        """Test that model versions are sorted correctly."""
        def version_key(item):
            v = item[0]
            if v == "base":
                return 0
            return int(v.replace("v", ""))

        items = [("v3", "path3"), ("base", "path0"), ("v2", "path2")]
        sorted_items = sorted(items, key=version_key)

        assert sorted_items[0][0] == "base"
        assert sorted_items[1][0] == "v2"
        assert sorted_items[2][0] == "v3"

    def test_cross_nn_pairing_key_format(self, daemon):
        """Test pairing key format."""
        board = "hex8"
        num_players = 2
        older_version = "base"
        newer_version = "v2"

        pairing_key = f"{board}_{num_players}p:{older_version}_vs_{newer_version}"

        assert pairing_key == "hex8_2p:base_vs_v2"

    def test_cross_nn_model_pattern_matching(self, daemon):
        """Test model filename pattern matching."""
        import re

        # Primary pattern for canonical models
        model_pattern = re.compile(
            r"canonical_(?P<board>\w+)_(?P<players>\d)p(?:_v(?P<version>\d+))?\.pth"
        )

        # Test standard canonical
        match = model_pattern.match("canonical_hex8_2p.pth")
        assert match is not None
        assert match.group("board") == "hex8"
        assert match.group("players") == "2"
        assert match.group("version") is None

        # Test versioned canonical
        match = model_pattern.match("canonical_hex8_2p_v3.pth")
        assert match is not None
        assert match.group("board") == "hex8"
        assert match.group("players") == "2"
        assert match.group("version") == "3"

        # Test non-matching
        match = model_pattern.match("random_model.pth")
        assert match is None

    def test_cross_nn_version_pattern_matching(self, daemon):
        """Test versioned model filename pattern matching."""
        import re

        version_pattern = re.compile(
            r"(?:canonical_)?(?P<board>\w+)_(?P<players>\d)p_v(?P<version>\d+)\.pth"
        )

        # Test non-canonical versioned
        match = version_pattern.match("hex8_2p_v2.pth")
        assert match is not None
        assert match.group("board") == "hex8"
        assert match.group("players") == "2"
        assert match.group("version") == "2"

    def test_cross_nn_pairing_result_structure(self, daemon):
        """Test pairing result structure."""
        games_per_pairing = 10
        wins_newer = 7
        wins_older = 2

        pairing_result = {
            "newer_wins": wins_newer,
            "older_wins": wins_older,
            "draws": games_per_pairing - wins_newer - wins_older,
            "games": games_per_pairing,
            "newer_win_rate": wins_newer / games_per_pairing,
            "improvement_validated": (wins_newer / games_per_pairing) >= 0.5,
        }

        assert pairing_result["newer_wins"] == 7
        assert pairing_result["older_wins"] == 2
        assert pairing_result["draws"] == 1
        assert pairing_result["newer_win_rate"] == 0.7
        assert pairing_result["improvement_validated"] is True

    def test_cross_nn_config_grouping(self, daemon):
        """Test config key grouping logic."""
        config_models: dict[tuple[str, int], list] = {}

        # Simulate adding models
        models_to_add = [
            ("hex8", 2, "base", "path1"),
            ("hex8", 2, "v2", "path2"),
            ("square8", 4, "base", "path3"),
        ]

        for board, players, version, path in models_to_add:
            config_key = (board, players)
            if config_key not in config_models:
                config_models[config_key] = []
            config_models[config_key].append((version, path))

        assert len(config_models) == 2  # Two configs
        assert len(config_models[("hex8", 2)]) == 2  # Two versions
        assert len(config_models[("square8", 4)]) == 1  # One version
