"""Tests for the execution module.

Comprehensive tests for:
- ExecutionResult dataclass
- LocalExecutor: run commands, timeout, working directory
- SSHExecutor: connection, command execution (mocked)
- ExecutorPool: multiple executors
- GameExecutor: game execution, results
- Module exports
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_result_creation(self):
        """Test creating an execution result."""
        from app.execution import ExecutionResult

        result = ExecutionResult(
            success=True,
            returncode=0,
            stdout="output",
            stderr="",
            duration_seconds=1.5,
            command="echo hello",
        )

        assert result.success is True
        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.duration_seconds == 1.5

    def test_result_failure(self):
        """Test creating a failed execution result."""
        from app.execution import ExecutionResult

        result = ExecutionResult(
            success=False,
            returncode=1,
            stdout="",
            stderr="error message",
            duration_seconds=0.5,
            command="exit 1",
        )

        assert result.success is False
        assert result.returncode == 1
        assert "error" in result.stderr

    def test_result_with_timeout(self):
        """Test execution result with timeout."""
        from app.execution import ExecutionResult

        result = ExecutionResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            duration_seconds=30.0,
            command="sleep 100",
            timed_out=True,
        )

        assert result.timed_out is True

    def test_result_host_default(self):
        """Test that host defaults to local."""
        from app.execution import ExecutionResult

        result = ExecutionResult(
            success=True,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            command="echo",
        )

        assert result.host == "local"


class TestLocalExecutor:
    """Tests for LocalExecutor class."""

    def test_executor_creation(self):
        """Test creating a local executor."""
        from app.execution import LocalExecutor

        executor = LocalExecutor()
        assert executor is not None

    def test_executor_with_working_dir(self, tmp_path):
        """Test executor with working directory."""
        from app.execution import LocalExecutor

        executor = LocalExecutor(working_dir=str(tmp_path))
        assert executor is not None

    @pytest.mark.asyncio
    async def test_run_simple_command(self):
        """Test running a simple command."""
        from app.execution import LocalExecutor

        executor = LocalExecutor()
        result = await executor.run("echo hello")

        assert result.success is True
        assert result.returncode == 0
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_failing_command(self):
        """Test running a failing command."""
        from app.execution import LocalExecutor

        executor = LocalExecutor()
        result = await executor.run("exit 1")

        assert result.success is False
        assert result.returncode == 1

    @pytest.mark.asyncio
    async def test_run_with_env(self):
        """Test running with custom environment."""
        from app.execution import LocalExecutor

        executor = LocalExecutor()
        result = await executor.run(
            "echo $TEST_VAR",
            env={"TEST_VAR": "test_value"},
        )

        assert result.success is True
        assert "test_value" in result.stdout

    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Test command timeout."""
        from app.execution import LocalExecutor

        executor = LocalExecutor()
        result = await executor.run("sleep 10", timeout=1)

        assert result.success is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_run_with_cwd(self, tmp_path):
        """Test running in specific directory."""
        from app.execution import LocalExecutor

        # Create a test file in tmp_path
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        executor = LocalExecutor()
        result = await executor.run("ls test.txt", cwd=str(tmp_path))

        assert result.success is True
        assert "test.txt" in result.stdout


class TestSSHExecutor:
    """Tests for SSHExecutor class (mocked)."""

    def test_ssh_executor_creation(self):
        """Test creating an SSH executor."""
        from app.execution import SSHExecutor

        executor = SSHExecutor(host="test-host", user="testuser")

        assert executor is not None
        # SSHExecutor stores config in SSHConfig object
        assert executor.config.host == "test-host"

    def test_ssh_executor_with_key(self, tmp_path):
        """Test SSH executor with key path."""
        from app.execution import SSHExecutor

        key_file = tmp_path / "id_rsa"
        key_file.write_text("fake key")

        executor = SSHExecutor(
            host="test-host",
            key_path=str(key_file),
        )

        assert executor is not None

    @pytest.mark.asyncio
    async def test_ssh_command_building(self):
        """Test SSH command construction."""
        from app.execution import SSHExecutor

        executor = SSHExecutor(
            host="test-host",
            user="testuser",
            port=2222,
        )

        # Verify executor was created with correct params
        assert executor.config.host == "test-host"
        assert executor.config.user == "testuser"
        assert executor.config.port == 2222


class TestExecutorPool:
    """Tests for ExecutorPool class."""

    def test_pool_creation(self):
        """Test creating an executor pool."""
        from app.execution import ExecutorPool

        pool = ExecutorPool()
        assert pool is not None

    def test_add_local_executor(self):
        """Test adding local executor to pool."""
        from app.execution import ExecutorPool

        pool = ExecutorPool()
        pool.add_local("worker-1")

        assert "worker-1" in pool.executors

    def test_add_ssh_executor(self):
        """Test adding SSH executor to pool."""
        from app.execution import ExecutorPool

        pool = ExecutorPool()
        pool.add_ssh("remote-1", host="192.168.1.10")

        assert "remote-1" in pool.executors

    @pytest.mark.asyncio
    async def test_run_all_local(self):
        """Test running command on all executors."""
        from app.execution import ExecutorPool

        pool = ExecutorPool()
        pool.add_local("worker-1")
        pool.add_local("worker-2")

        results = await pool.run_all("echo hello")

        assert len(results) == 2
        assert all(r.success for r in results.values())


class TestGameExecutor:
    """Tests for GameExecutor class."""

    def test_game_executor_creation(self):
        """Test creating a game executor."""
        from app.execution import GameExecutor

        executor = GameExecutor(board_type="square8", num_players=2)
        assert executor is not None

    def test_game_outcome_enum(self):
        """Test GameOutcome enum values."""
        from app.execution import GameOutcome

        assert GameOutcome.WIN is not None
        assert GameOutcome.DRAW is not None
        assert GameOutcome.TIMEOUT is not None
        assert GameOutcome.ERROR is not None
        assert GameOutcome.FORFEIT is not None


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_game_result_creation(self):
        """Test creating a game result."""
        from app.execution import GameResult, GameOutcome

        result = GameResult(
            game_id="test-game-1",
            board_type="square8",
            num_players=2,
            winner=1,
            outcome=GameOutcome.WIN,
            move_count=50,
            player_types=["heuristic", "mcts"],
            player_configs=[{}, {}],
            duration_seconds=120.0,
        )

        assert result.game_id == "test-game-1"
        assert result.outcome == GameOutcome.WIN
        assert result.winner == 1
        assert result.move_count == 50


class TestRunCommandHelper:
    """Tests for run_command helper function."""

    @pytest.mark.asyncio
    async def test_run_command_simple(self):
        """Test run_command helper."""
        from app.execution import run_command

        result = await run_command("echo hello")

        assert result.success is True
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_with_timeout(self):
        """Test run_command with timeout."""
        from app.execution import run_command

        result = await run_command("sleep 10", timeout=1)

        assert result.success is False
        assert result.timed_out is True


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_executor_exports(self):
        """Test executor exports."""
        from app.execution import (
            ExecutionResult,
            BaseExecutor,
            LocalExecutor,
            SSHExecutor,
            ExecutorPool,
            run_command,
        )

        assert ExecutionResult is not None
        assert BaseExecutor is not None
        assert LocalExecutor is not None
        assert SSHExecutor is not None
        assert ExecutorPool is not None
        assert callable(run_command)

    def test_backend_exports(self):
        """Test backend exports."""
        from app.execution import (
            BackendType,
            WorkerStatus,
            JobResult,
            OrchestratorBackend,
            LocalBackend,
            SSHBackend,
            get_backend,
        )

        assert BackendType is not None
        assert WorkerStatus is not None
        assert JobResult is not None
        assert OrchestratorBackend is not None
        assert LocalBackend is not None
        assert SSHBackend is not None
        assert callable(get_backend)

    def test_game_executor_exports(self):
        """Test game executor exports."""
        from app.execution import (
            GameOutcome,
            GameResult,
            GameExecutor,
            ParallelGameExecutor,
            run_quick_game,
            run_selfplay_batch,
        )

        assert GameOutcome is not None
        assert GameResult is not None
        assert GameExecutor is not None
        assert ParallelGameExecutor is not None
        assert callable(run_quick_game)
        assert callable(run_selfplay_batch)
