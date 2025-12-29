"""Tests for app.distributed.client module.

This module tests the distributed CMA-ES evaluation client.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from app.distributed.client import (
    BOARD_MEMORY_REQUIREMENTS,
    DistributedEvaluator,
    EvaluationStats,
    TaskResult,
    WorkerClient,
)


# =============================================================================
# TaskResult Tests
# =============================================================================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_fields(self):
        """Test TaskResult has expected fields."""
        result = TaskResult(
            task_id="task-123",
            candidate_id=5,
            fitness=0.75,
            games_played=24,
            evaluation_time_sec=12.5,
            worker_id="worker-1",
            status="success",
        )
        assert result.task_id == "task-123"
        assert result.candidate_id == 5
        assert result.fitness == 0.75
        assert result.games_played == 24
        assert result.evaluation_time_sec == 12.5
        assert result.worker_id == "worker-1"
        assert result.status == "success"
        assert result.error is None
        assert result.game_replays is None

    def test_task_result_with_error(self):
        """Test TaskResult with error."""
        result = TaskResult(
            task_id="task-456",
            candidate_id=3,
            fitness=0.0,
            games_played=0,
            evaluation_time_sec=0.0,
            worker_id="worker-2",
            status="error",
            error="Connection timeout",
        )
        assert result.status == "error"
        assert result.error == "Connection timeout"

    def test_task_result_with_game_replays(self):
        """Test TaskResult with game replays."""
        replays = [{"game_id": "g1", "moves": []}, {"game_id": "g2", "moves": []}]
        result = TaskResult(
            task_id="task-789",
            candidate_id=1,
            fitness=0.85,
            games_played=2,
            evaluation_time_sec=5.0,
            worker_id="worker-3",
            status="success",
            game_replays=replays,
        )
        assert result.game_replays == replays
        assert len(result.game_replays) == 2

    def test_from_dict_success(self):
        """Test TaskResult.from_dict with valid data."""
        data = {
            "task_id": "test-task",
            "candidate_id": 10,
            "fitness": 0.9,
            "games_played": 50,
            "evaluation_time_sec": 30.0,
            "worker_id": "worker-x",
            "status": "success",
        }
        result = TaskResult.from_dict(data)
        assert result.task_id == "test-task"
        assert result.candidate_id == 10
        assert result.fitness == 0.9
        assert result.status == "success"

    def test_from_dict_missing_fields(self):
        """Test TaskResult.from_dict with missing fields uses defaults."""
        data = {}
        result = TaskResult.from_dict(data)
        assert result.task_id == ""
        assert result.candidate_id == -1
        assert result.fitness == 0.0
        assert result.games_played == 0
        assert result.worker_id == "unknown"
        assert result.status == "error"

    def test_from_dict_with_replays(self):
        """Test TaskResult.from_dict preserves game replays."""
        data = {
            "task_id": "replay-task",
            "candidate_id": 1,
            "fitness": 0.8,
            "games_played": 10,
            "evaluation_time_sec": 5.0,
            "worker_id": "worker-y",
            "status": "success",
            "game_replays": [{"id": 1}],
        }
        result = TaskResult.from_dict(data)
        assert result.game_replays == [{"id": 1}]


# =============================================================================
# EvaluationStats Tests
# =============================================================================


class TestEvaluationStats:
    """Tests for EvaluationStats dataclass."""

    def test_default_values(self):
        """Test EvaluationStats default values."""
        stats = EvaluationStats()
        assert stats.total_candidates == 0
        assert stats.successful_evaluations == 0
        assert stats.failed_evaluations == 0
        assert stats.total_games == 0
        assert stats.total_time_sec == 0.0
        assert stats.worker_task_counts == {}
        assert stats.all_game_replays == []

    def test_add_result_success(self):
        """Test adding a successful result."""
        stats = EvaluationStats()
        result = TaskResult(
            task_id="t1",
            candidate_id=0,
            fitness=0.8,
            games_played=24,
            evaluation_time_sec=10.0,
            worker_id="worker-1",
            status="success",
        )
        stats.add_result(result)
        assert stats.total_candidates == 1
        assert stats.successful_evaluations == 1
        assert stats.failed_evaluations == 0
        assert stats.total_games == 24
        assert stats.worker_task_counts == {"worker-1": 1}

    def test_add_result_failure(self):
        """Test adding a failed result."""
        stats = EvaluationStats()
        result = TaskResult(
            task_id="t2",
            candidate_id=1,
            fitness=0.0,
            games_played=0,
            evaluation_time_sec=0.0,
            worker_id="worker-2",
            status="error",
            error="Timeout",
        )
        stats.add_result(result)
        assert stats.total_candidates == 1
        assert stats.successful_evaluations == 0
        assert stats.failed_evaluations == 1
        assert stats.total_games == 0

    def test_add_result_with_replays(self):
        """Test adding result with game replays."""
        stats = EvaluationStats()
        replays = [{"game": 1}, {"game": 2}]
        result = TaskResult(
            task_id="t3",
            candidate_id=0,
            fitness=0.9,
            games_played=2,
            evaluation_time_sec=5.0,
            worker_id="worker-1",
            status="success",
            game_replays=replays,
        )
        stats.add_result(result)
        assert len(stats.all_game_replays) == 2

    def test_add_multiple_results(self):
        """Test adding multiple results from different workers."""
        stats = EvaluationStats()

        for i in range(3):
            result = TaskResult(
                task_id=f"t{i}",
                candidate_id=i,
                fitness=0.5 + i * 0.1,
                games_played=10,
                evaluation_time_sec=5.0,
                worker_id=f"worker-{i % 2}",
                status="success",
            )
            stats.add_result(result)

        assert stats.total_candidates == 3
        assert stats.successful_evaluations == 3
        assert stats.total_games == 30
        assert stats.worker_task_counts["worker-0"] == 2
        assert stats.worker_task_counts["worker-1"] == 1


# =============================================================================
# WorkerClient Tests
# =============================================================================


class TestWorkerClient:
    """Tests for WorkerClient class."""

    def test_init_with_host_port(self):
        """Test initialization with host:port format."""
        client = WorkerClient("192.168.1.100:8080")
        assert client.worker_url == "192.168.1.100:8080"
        assert client._base_url == "http://192.168.1.100:8080"

    def test_init_with_http_url(self):
        """Test initialization with full HTTP URL."""
        client = WorkerClient("http://192.168.1.100:8080")
        assert client._base_url == "http://192.168.1.100:8080"

    def test_init_with_https_url(self):
        """Test initialization with HTTPS URL."""
        client = WorkerClient("https://secure.worker.io:443")
        assert client._base_url == "https://secure.worker.io:443"

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = WorkerClient("localhost:8080", timeout=600.0)
        assert client.timeout == 600.0

    @patch("app.distributed.client.urlopen")
    def test_health_check_success(self, mock_urlopen):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "healthy", "memory": {"total_gb": 32}}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = WorkerClient("localhost:8080")
        result = client.health_check()

        assert result["status"] == "healthy"
        assert result["memory"]["total_gb"] == 32

    @patch("app.distributed.client.urlopen")
    def test_health_check_http_error(self, mock_urlopen):
        """Test health check with HTTP error."""
        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:8080/health",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

        client = WorkerClient("localhost:8080")
        result = client.health_check()

        assert result["status"] == "error"
        assert "HTTP 500" in result["error"]

    @patch("app.distributed.client.urlopen")
    def test_health_check_url_error(self, mock_urlopen):
        """Test health check with URL error (connection refused)."""
        mock_urlopen.side_effect = URLError("Connection refused")

        client = WorkerClient("localhost:8080")
        result = client.health_check()

        assert result["status"] == "error"
        assert "URL error" in result["error"]

    @patch("app.distributed.client.urlopen")
    def test_health_check_timeout(self, mock_urlopen):
        """Test health check timeout."""
        mock_urlopen.side_effect = TimeoutError()

        client = WorkerClient("localhost:8080")
        result = client.health_check()

        assert result["status"] == "error"
        assert "timeout" in result["error"].lower()

    @patch("app.distributed.client.urlopen")
    def test_is_healthy_true(self, mock_urlopen):
        """Test is_healthy returns True for healthy worker."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "healthy"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = WorkerClient("localhost:8080")
        assert client.is_healthy() is True

    @patch("app.distributed.client.urlopen")
    def test_is_healthy_false(self, mock_urlopen):
        """Test is_healthy returns False for unhealthy worker."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "error"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = WorkerClient("localhost:8080")
        assert client.is_healthy() is False

    @patch("app.distributed.client.urlopen")
    def test_get_memory_info(self, mock_urlopen):
        """Test get_memory_info extracts memory data."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "healthy",
            "memory": {
                "total_gb": 64,
                "available_gb": 32,
                "eligible_boards": ["square8", "square19"],
            },
        }).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = WorkerClient("localhost:8080")
        memory = client.get_memory_info()

        assert memory["total_gb"] == 64
        assert memory["available_gb"] == 32
        assert "square8" in memory["eligible_boards"]

    def test_can_handle_board_eligible(self):
        """Test can_handle_board returns True for eligible boards."""
        client = WorkerClient("localhost:8080")

        with patch.object(client, "get_memory_info") as mock_get_memory:
            mock_get_memory.return_value = {
                "total_gb": 64,
                "eligible_boards": ["square8", "hex"],
            }
            assert client.can_handle_board("square8") is True

    def test_can_handle_board_fallback_memory_check(self):
        """Test can_handle_board falls back to memory check."""
        client = WorkerClient("localhost:8080")

        with patch.object(client, "get_memory_info") as mock_get_memory:
            mock_get_memory.return_value = {
                "total_gb": 64,
                "eligible_boards": [],  # Empty list
            }
            # 64GB >= 48GB required for hexagonal
            assert client.can_handle_board("hexagonal") is True

    def test_can_handle_board_insufficient_memory(self):
        """Test can_handle_board returns False for insufficient memory."""
        client = WorkerClient("localhost:8080")

        with patch.object(client, "get_memory_info") as mock_get_memory:
            mock_get_memory.return_value = {
                "total_gb": 16,
                "eligible_boards": [],
            }
            # 16GB < 48GB required for hexagonal
            assert client.can_handle_board("hexagonal") is False

    @patch("app.distributed.client.urlopen")
    def test_evaluate_success(self, mock_urlopen):
        """Test successful evaluation."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "task_id": "task-123",
            "candidate_id": 0,
            "fitness": 0.85,
            "games_played": 24,
            "evaluation_time_sec": 15.0,
            "worker_id": "worker-1",
            "status": "success",
        }).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = WorkerClient("localhost:8080")
        task = {"task_id": "task-123", "candidate_id": 0, "weights": {}}
        result = client.evaluate(task)

        assert result.status == "success"
        assert result.fitness == 0.85
        assert result.games_played == 24

    @patch("app.distributed.client.urlopen")
    def test_evaluate_http_error(self, mock_urlopen):
        """Test evaluation with HTTP error."""
        error = HTTPError(
            url="http://localhost:8080/evaluate",
            code=500,
            msg="Server Error",
            hdrs={},
            fp=MagicMock(read=lambda: b"Internal error"),
        )
        mock_urlopen.side_effect = error

        client = WorkerClient("localhost:8080")
        task = {"task_id": "task-456", "candidate_id": 1}
        result = client.evaluate(task)

        assert result.status == "error"
        assert "HTTP 500" in result.error

    @patch("app.distributed.client.urlopen")
    def test_evaluate_timeout(self, mock_urlopen):
        """Test evaluation timeout."""
        mock_urlopen.side_effect = TimeoutError()

        client = WorkerClient("localhost:8080")
        task = {"task_id": "task-789", "candidate_id": 2}
        result = client.evaluate(task)

        assert result.status == "error"
        assert "timeout" in result.error.lower()


# =============================================================================
# DistributedEvaluator Tests
# =============================================================================


class TestDistributedEvaluator:
    """Tests for DistributedEvaluator class."""

    def test_init(self):
        """Test initialization with worker list."""
        evaluator = DistributedEvaluator(
            workers=["worker1:8080", "worker2:8080"],
            board_type="square8",
            num_players=2,
            games_per_eval=24,
        )
        assert len(evaluator.workers) == 2
        assert evaluator.board_type == "square8"
        assert evaluator.num_players == 2
        assert evaluator.games_per_eval == 24

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        evaluator = DistributedEvaluator(
            workers=["worker:8080"],
            board_type="hexagonal",
            num_players=4,
            games_per_eval=50,
            eval_mode="initial-only",
            state_pool_id="v2",
            max_moves=500,
            eval_randomness=0.1,
            seed=42,
            timeout=600.0,
            max_retries=3,
            fallback_fitness=-1.0,
            record_games=True,
        )
        assert evaluator.board_type == "hexagonal"
        assert evaluator.num_players == 4
        assert evaluator.eval_mode == "initial-only"
        assert evaluator.max_retries == 3
        assert evaluator.record_games is True

    def test_clients_created(self):
        """Test that clients are created for each worker."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080", "w2:8080", "w3:8080"],
            board_type="square8",
            num_players=2,
        )
        assert len(evaluator._clients) == 3
        assert "w1:8080" in evaluator._clients
        assert "w2:8080" in evaluator._clients
        assert "w3:8080" in evaluator._clients

    def test_verify_workers(self):
        """Test verify_workers filters healthy workers."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080", "w2:8080"],
            board_type="square8",
            num_players=2,
        )

        # Mock client health checks
        evaluator._clients["w1:8080"].is_healthy = MagicMock(return_value=True)
        evaluator._clients["w2:8080"].is_healthy = MagicMock(return_value=False)
        evaluator._clients["w1:8080"].get_memory_info = MagicMock(return_value={
            "total_gb": 32,
            "eligible_boards": ["square8"],
        })

        healthy = evaluator.verify_workers(check_memory=True)
        assert len(healthy) == 1
        assert "w1:8080" in healthy

    def test_verify_workers_no_memory_check(self):
        """Test verify_workers without memory check."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080", "w2:8080"],
            board_type="square8",
            num_players=2,
        )

        evaluator._clients["w1:8080"].is_healthy = MagicMock(return_value=True)
        evaluator._clients["w2:8080"].is_healthy = MagicMock(return_value=True)

        healthy = evaluator.verify_workers(check_memory=False)
        assert len(healthy) == 2

    def test_get_next_worker_round_robin(self):
        """Test round-robin worker selection."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080", "w2:8080", "w3:8080"],
            board_type="square8",
            num_players=2,
        )
        evaluator._healthy_workers = ["w1:8080", "w2:8080", "w3:8080"]

        # Should cycle through workers
        assert evaluator._get_next_worker() == "w1:8080"
        assert evaluator._get_next_worker() == "w2:8080"
        assert evaluator._get_next_worker() == "w3:8080"
        assert evaluator._get_next_worker() == "w1:8080"

    def test_get_next_worker_no_healthy(self):
        """Test error when no healthy workers."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080"],
            board_type="square8",
            num_players=2,
        )
        evaluator._healthy_workers = []

        # Mock verify_workers to return empty
        evaluator.verify_workers = MagicMock(return_value=[])

        with pytest.raises(RuntimeError, match="No healthy workers"):
            evaluator._get_next_worker()

    def test_create_task(self):
        """Test task creation."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080"],
            board_type="hex8",
            num_players=3,
            games_per_eval=30,
            seed=123,
            record_games=True,
        )

        weights = {"weight1": 1.0}
        baseline = {"weight1": 0.5}
        task = evaluator._create_task(5, weights, baseline)

        assert "task_id" in task
        assert task["candidate_id"] == 5
        assert task["weights"] == weights
        assert task["baseline_weights"] == baseline
        assert task["board_type"] == "hex8"
        assert task["num_players"] == 3
        assert task["games_per_eval"] == 30
        assert task["seed"] == 123
        assert task["record_games"] is True

    def test_evaluate_population_empty(self):
        """Test evaluate_population with empty population."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080"],
            board_type="square8",
            num_players=2,
        )

        fitness, stats = evaluator.evaluate_population([])
        assert fitness == []
        assert stats.total_candidates == 0

    def test_evaluate_population_no_workers(self):
        """Test evaluate_population fails with no healthy workers."""
        evaluator = DistributedEvaluator(
            workers=["w1:8080"],
            board_type="square8",
            num_players=2,
        )
        evaluator.verify_workers = MagicMock(return_value=[])

        with pytest.raises(RuntimeError, match="No healthy workers"):
            evaluator.evaluate_population([{"w1": 1.0}])


# =============================================================================
# Board Memory Requirements Tests
# =============================================================================


class TestBoardMemoryRequirements:
    """Tests for BOARD_MEMORY_REQUIREMENTS constant."""

    def test_square8_requirement(self):
        """Test square8 memory requirement."""
        assert BOARD_MEMORY_REQUIREMENTS["square8"] == 8

    def test_square19_requirement(self):
        """Test square19 memory requirement."""
        assert BOARD_MEMORY_REQUIREMENTS["square19"] == 48

    def test_hexagonal_requirement(self):
        """Test hexagonal memory requirement."""
        assert BOARD_MEMORY_REQUIREMENTS["hexagonal"] == 48

    def test_hex_alias(self):
        """Test hex is an alias for hexagonal."""
        assert BOARD_MEMORY_REQUIREMENTS["hex"] == 48

    def test_full_hex_alias(self):
        """Test full_hex is an alias."""
        assert BOARD_MEMORY_REQUIREMENTS["full_hex"] == 48

    def test_all_keys_present(self):
        """Test all expected board types are present."""
        expected = ["square8", "square19", "hexagonal", "hex", "full_hex", "hex24"]
        for key in expected:
            assert key in BOARD_MEMORY_REQUIREMENTS


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_taskresult_exported(self):
        """Test TaskResult is exported."""
        from app.distributed.client import TaskResult
        assert TaskResult is not None

    def test_evaluationstats_exported(self):
        """Test EvaluationStats is exported."""
        from app.distributed.client import EvaluationStats
        assert EvaluationStats is not None

    def test_workerclient_exported(self):
        """Test WorkerClient is exported."""
        from app.distributed.client import WorkerClient
        assert WorkerClient is not None

    def test_distributedevaluator_exported(self):
        """Test DistributedEvaluator is exported."""
        from app.distributed.client import DistributedEvaluator
        assert DistributedEvaluator is not None

    def test_queuedistributedevaluator_exported(self):
        """Test QueueDistributedEvaluator is exported."""
        from app.distributed.client import QueueDistributedEvaluator
        assert QueueDistributedEvaluator is not None
