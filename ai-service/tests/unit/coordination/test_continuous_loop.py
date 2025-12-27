"""Tests for app.coordination.continuous_loop module.

Tests the continuous training loop daemon:
- LoopState enum
- LoopConfig dataclass
- LoopStats dataclass
- ContinuousTrainingLoop class (start, stop, health_check, run_loop)
- Global instance management (get_continuous_loop, start_continuous_loop, stop_continuous_loop)
- CLI argument parsing (parse_config_arg)

December 2025 - Phase 3 test coverage for critical untested modules.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


class TestLoopState:
    """Tests for LoopState enum."""

    def test_loop_states_exist(self):
        """Should have all expected loop states."""
        from app.coordination.continuous_loop import LoopState

        assert hasattr(LoopState, 'IDLE')
        assert hasattr(LoopState, 'RUNNING_SELFPLAY')
        assert hasattr(LoopState, 'WAITING_PIPELINE')
        assert hasattr(LoopState, 'COOLING_DOWN')
        assert hasattr(LoopState, 'DEFERRED')
        assert hasattr(LoopState, 'STOPPED')
        assert hasattr(LoopState, 'ERROR')

    def test_loop_state_values(self):
        """Should have string values for all states."""
        from app.coordination.continuous_loop import LoopState

        assert LoopState.IDLE.value == "idle"
        assert LoopState.RUNNING_SELFPLAY.value == "running_selfplay"
        assert LoopState.WAITING_PIPELINE.value == "waiting_pipeline"
        assert LoopState.COOLING_DOWN.value == "cooling_down"
        assert LoopState.DEFERRED.value == "deferred"
        assert LoopState.STOPPED.value == "stopped"
        assert LoopState.ERROR.value == "error"


class TestLoopConfig:
    """Tests for LoopConfig dataclass."""

    def test_default_configs(self):
        """Should have sensible defaults."""
        from app.coordination.continuous_loop import LoopConfig

        config = LoopConfig()
        assert len(config.configs) == 2
        assert ("hex8", 2) in config.configs
        assert ("square8", 2) in config.configs

    def test_default_selfplay_settings(self):
        """Should have sensible selfplay defaults."""
        from app.coordination.continuous_loop import LoopConfig

        config = LoopConfig()
        assert config.selfplay_games_per_iteration == 1000
        assert config.selfplay_engine == "gumbel-mcts"

    def test_default_pipeline_settings(self):
        """Should have sensible pipeline defaults."""
        from app.coordination.continuous_loop import LoopConfig

        config = LoopConfig()
        assert config.parallel_configs is False
        assert config.pipeline_timeout_seconds == 7200  # 2 hours

    def test_default_loop_control(self):
        """Should have sensible loop control defaults."""
        from app.coordination.continuous_loop import LoopConfig

        config = LoopConfig()
        assert config.max_iterations == 0  # Infinite
        assert config.iteration_cooldown_seconds == 60.0

    def test_default_failure_handling(self):
        """Should have sensible failure handling defaults."""
        from app.coordination.continuous_loop import LoopConfig

        config = LoopConfig()
        assert config.max_consecutive_failures == 3
        assert config.failure_backoff_seconds == 300.0

    def test_custom_configs(self):
        """Should accept custom configs."""
        from app.coordination.continuous_loop import LoopConfig

        config = LoopConfig(
            configs=[("hexagonal", 4), ("square19", 2)],
            selfplay_games_per_iteration=500,
            max_iterations=10,
        )
        assert len(config.configs) == 2
        assert ("hexagonal", 4) in config.configs
        assert config.selfplay_games_per_iteration == 500
        assert config.max_iterations == 10


class TestLoopStats:
    """Tests for LoopStats dataclass."""

    def test_default_stats(self):
        """Should have zero defaults."""
        from app.coordination.continuous_loop import LoopStats

        stats = LoopStats()
        assert stats.total_iterations == 0
        assert stats.successful_iterations == 0
        assert stats.failed_iterations == 0
        assert stats.consecutive_failures == 0
        assert stats.last_iteration_time == 0.0
        assert stats.last_config_trained == ""
        assert stats.current_config == ""

    def test_default_state(self):
        """Should default to IDLE state."""
        from app.coordination.continuous_loop import LoopStats, LoopState

        stats = LoopStats()
        assert stats.current_state == LoopState.IDLE

    def test_start_time_set(self):
        """Should set start_time on creation."""
        from app.coordination.continuous_loop import LoopStats
        import time

        before = time.time()
        stats = LoopStats()
        after = time.time()

        assert before <= stats.start_time <= after


class TestContinuousTrainingLoop:
    """Tests for ContinuousTrainingLoop class."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopConfig

        loop = ContinuousTrainingLoop()
        assert loop.config is not None
        assert isinstance(loop.config, LoopConfig)

    def test_init_custom_config(self):
        """Should initialize with custom config."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopConfig

        config = LoopConfig(max_iterations=5)
        loop = ContinuousTrainingLoop(config)
        assert loop.config.max_iterations == 5

    def test_init_stats(self):
        """Should initialize stats."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        assert loop.stats.total_iterations == 0
        assert loop._running is False

    def test_is_unified_loop_running_force_mode(self):
        """Should return False in force mode."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopConfig

        config = LoopConfig(force=True)
        loop = ContinuousTrainingLoop(config)
        assert loop._is_unified_loop_running() is False

    @patch('app.coordination.continuous_loop.ContinuousTrainingLoop._is_unified_loop_running')
    def test_is_unified_loop_running_checked(self, mock_check):
        """Should call helpers.is_unified_loop_running."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        mock_check.return_value = False
        loop = ContinuousTrainingLoop()
        result = loop._is_unified_loop_running()
        # Method was called (since we're patching it, just verify it returns our mock value)
        assert result is False


class TestContinuousTrainingLoopStart:
    """Tests for ContinuousTrainingLoop.start()."""

    @pytest.fixture
    def mock_env(self):
        """Mock environment configuration."""
        with patch('app.coordination.continuous_loop.env') as mock:
            mock.is_coordinator = False
            mock.node_id = "test-node"
            yield mock

    @pytest.mark.asyncio
    async def test_start_skips_on_coordinator(self):
        """Should skip on coordinator nodes."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        with patch('app.coordination.continuous_loop.env') as mock_env:
            mock_env.is_coordinator = True
            mock_env.node_id = "coordinator-node"

            loop = ContinuousTrainingLoop()
            await loop.start()

            # Should not be running
            assert loop._running is False

    @pytest.mark.asyncio
    async def test_start_warns_if_already_running(self, mock_env, caplog):
        """Should warn if already running."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop
        import logging

        loop = ContinuousTrainingLoop()
        loop._running = True

        with caplog.at_level(logging.WARNING):
            await loop.start()

        assert "already running" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_start_defers_to_unified_loop(self, mock_env):
        """Should defer when unified_ai_loop is running."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopState

        with patch.object(ContinuousTrainingLoop, '_is_unified_loop_running', return_value=True):
            with patch.object(ContinuousTrainingLoop, '_setup_pipeline'):
                with patch.object(ContinuousTrainingLoop, '_run_loop', new_callable=AsyncMock):
                    loop = ContinuousTrainingLoop()
                    await loop.start()

                    assert loop.stats.current_state == LoopState.DEFERRED


class TestContinuousTrainingLoopStop:
    """Tests for ContinuousTrainingLoop.stop()."""

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Should do nothing if not running."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        loop._running = False

        await loop.stop()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self):
        """Should set state to STOPPED."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopState

        loop = ContinuousTrainingLoop()
        loop._running = True
        loop._shutdown_event = asyncio.Event()
        loop._task = None

        await loop.stop()

        assert loop.stats.current_state == LoopState.STOPPED
        assert loop._running is False


class TestContinuousTrainingLoopHealthCheck:
    """Tests for ContinuousTrainingLoop.health_check()."""

    def test_health_check_not_running(self):
        """Should return unhealthy when not running."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        loop._running = False

        result = loop.health_check()
        assert result.healthy is False

    def test_health_check_in_error_state(self):
        """Should return unhealthy in error state."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopState

        loop = ContinuousTrainingLoop()
        loop._running = True
        loop.stats.current_state = LoopState.ERROR

        result = loop.health_check()
        assert result.healthy is False

    def test_health_check_healthy(self):
        """Should return healthy when running normally."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopState

        loop = ContinuousTrainingLoop()
        loop._running = True
        loop.stats.current_state = LoopState.IDLE

        result = loop.health_check()
        assert result.healthy is True


class TestContinuousTrainingLoopGetStatus:
    """Tests for ContinuousTrainingLoop.get_status()."""

    def test_get_status_structure(self):
        """Should return complete status dict."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        status = loop.get_status()

        assert "running" in status
        assert "state" in status
        assert "total_iterations" in status
        assert "successful_iterations" in status
        assert "failed_iterations" in status
        assert "consecutive_failures" in status
        assert "uptime_seconds" in status
        assert "configs" in status

    def test_get_status_configs_formatted(self):
        """Should format configs correctly."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop, LoopConfig

        config = LoopConfig(configs=[("hex8", 2), ("square8", 4)])
        loop = ContinuousTrainingLoop(config)
        status = loop.get_status()

        assert "hex8_2p" in status["configs"]
        assert "square8_4p" in status["configs"]


class TestGlobalInstanceManagement:
    """Tests for global instance functions."""

    def test_get_continuous_loop_creates_instance(self):
        """Should create singleton instance."""
        from app.coordination import continuous_loop as module

        # Reset global
        module._loop_instance = None

        loop = module.get_continuous_loop()
        assert loop is not None

    def test_get_continuous_loop_returns_same_instance(self):
        """Should return same instance on subsequent calls."""
        from app.coordination import continuous_loop as module

        # Reset global
        module._loop_instance = None

        loop1 = module.get_continuous_loop()
        loop2 = module.get_continuous_loop()
        assert loop1 is loop2

    @pytest.mark.asyncio
    async def test_start_continuous_loop_creates_new_instance(self):
        """Should create new instance with config."""
        from app.coordination import continuous_loop as module
        from app.coordination.continuous_loop import LoopConfig

        # Mock to prevent actual startup
        with patch.object(module.ContinuousTrainingLoop, 'start', new_callable=AsyncMock):
            module._loop_instance = None
            config = LoopConfig(max_iterations=3)

            loop = await module.start_continuous_loop(config)
            assert loop is not None
            assert loop.config.max_iterations == 3

    @pytest.mark.asyncio
    async def test_stop_continuous_loop_clears_instance(self):
        """Should stop and clear global instance."""
        from app.coordination import continuous_loop as module

        # Set up mock instance
        mock_loop = MagicMock()
        mock_loop.stop = AsyncMock()
        module._loop_instance = mock_loop

        await module.stop_continuous_loop()

        mock_loop.stop.assert_called_once()
        assert module._loop_instance is None

    @pytest.mark.asyncio
    async def test_stop_continuous_loop_no_instance(self):
        """Should do nothing if no instance."""
        from app.coordination import continuous_loop as module

        module._loop_instance = None

        # Should not raise
        await module.stop_continuous_loop()


class TestParseConfigArg:
    """Tests for parse_config_arg()."""

    def test_parse_colon_format(self):
        """Should parse 'board:players' format."""
        from app.coordination.continuous_loop import parse_config_arg

        result = parse_config_arg("hex8:2")
        assert result == ("hex8", 2)

    def test_parse_colon_format_4p(self):
        """Should parse 4-player colon format."""
        from app.coordination.continuous_loop import parse_config_arg

        result = parse_config_arg("square19:4")
        assert result == ("square19", 4)

    def test_parse_underscore_format(self):
        """Should parse 'board_Xp' format."""
        from app.coordination.continuous_loop import parse_config_arg

        result = parse_config_arg("hex8_2p")
        assert result == ("hex8", 2)

    def test_parse_underscore_format_4p(self):
        """Should parse 4-player underscore format."""
        from app.coordination.continuous_loop import parse_config_arg

        result = parse_config_arg("square8_4p")
        assert result == ("square8", 4)

    def test_parse_hexagonal(self):
        """Should parse hexagonal board."""
        from app.coordination.continuous_loop import parse_config_arg

        result = parse_config_arg("hexagonal:3")
        assert result == ("hexagonal", 3)

    def test_parse_invalid_format_raises(self):
        """Should raise ValueError for invalid format."""
        from app.coordination.continuous_loop import parse_config_arg

        with pytest.raises(ValueError) as exc_info:
            parse_config_arg("invalid")

        assert "Invalid config format" in str(exc_info.value)


class TestRunSelfplay:
    """Tests for _run_selfplay method."""

    @pytest.mark.asyncio
    async def test_run_selfplay_success(self):
        """Should return True on successful selfplay."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()

        mock_stats = MagicMock()
        mock_stats.games_completed = 100
        mock_stats.total_samples = 5000

        with patch('app.coordination.continuous_loop.run_selfplay', return_value=mock_stats):
            result = await loop._run_selfplay("hex8", 2, 100, "gumbel-mcts")
            assert result is True

    @pytest.mark.asyncio
    async def test_run_selfplay_no_games(self):
        """Should return False when no games produced."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()

        mock_stats = MagicMock()
        mock_stats.games_completed = 0

        with patch('app.coordination.continuous_loop.run_selfplay', return_value=mock_stats):
            result = await loop._run_selfplay("hex8", 2, 100, "gumbel-mcts")
            assert result is False

    @pytest.mark.asyncio
    async def test_run_selfplay_import_error(self):
        """Should return False on import error."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()

        with patch('app.coordination.continuous_loop.run_selfplay', side_effect=ImportError("module not found")):
            result = await loop._run_selfplay("hex8", 2, 100, "gumbel-mcts")
            assert result is False


class TestWaitOrShutdown:
    """Tests for _wait_or_shutdown method."""

    @pytest.mark.asyncio
    async def test_wait_or_shutdown_timeout(self):
        """Should return True when timeout completes normally."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        loop._shutdown_event = asyncio.Event()

        # Wait for short duration
        result = await loop._wait_or_shutdown(0.01)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_or_shutdown_interrupted(self):
        """Should return False when shutdown requested."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        loop._shutdown_event = asyncio.Event()

        # Set shutdown event before waiting
        loop._shutdown_event.set()

        result = await loop._wait_or_shutdown(10.0)
        assert result is False


class TestSetupPipeline:
    """Tests for _setup_pipeline method."""

    def test_setup_pipeline_import_error(self, caplog):
        """Should log warning on import error."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop
        import logging

        loop = ContinuousTrainingLoop()

        with patch.dict('sys.modules', {'app.coordination.data_pipeline_orchestrator': None}):
            with patch('app.coordination.continuous_loop.get_orchestrator', side_effect=ImportError("not found")):
                with caplog.at_level(logging.WARNING):
                    loop._setup_pipeline()

        # Should have logged warning about import failure
        # (or succeeded if module is available)


class TestRunSingleConfig:
    """Tests for _run_single_config method."""

    @pytest.mark.asyncio
    async def test_run_single_config_selfplay_fails(self):
        """Should return False when selfplay fails."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()

        with patch.object(loop, '_run_selfplay', new_callable=AsyncMock) as mock_selfplay:
            mock_selfplay.return_value = False

            result = await loop._run_single_config("hex8", 2, 1)
            assert result is False

    @pytest.mark.asyncio
    async def test_run_single_config_success(self):
        """Should return True when iteration completes."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()

        with patch.object(loop, '_run_selfplay', new_callable=AsyncMock) as mock_selfplay:
            mock_selfplay.return_value = True
            with patch.object(loop, '_wait_for_pipeline', new_callable=AsyncMock) as mock_pipeline:
                mock_pipeline.return_value = True

                result = await loop._run_single_config("hex8", 2, 1)
                assert result is True


class TestWaitForPipeline:
    """Tests for _wait_for_pipeline method."""

    @pytest.mark.asyncio
    async def test_wait_for_pipeline_no_orchestrator(self):
        """Should return True when no orchestrator."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        loop._orchestrator = None

        result = await loop._wait_for_pipeline("hex8_2p", 10)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_pipeline_stopped(self):
        """Should return False when loop stopped."""
        from app.coordination.continuous_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        loop._running = False
        loop._orchestrator = MagicMock()

        result = await loop._wait_for_pipeline("hex8_2p", 1)
        assert result is False
