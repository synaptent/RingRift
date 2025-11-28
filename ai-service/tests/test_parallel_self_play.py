"""Tests for parallel self-play execution."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_parallel_self_play import (  # noqa: E402
    GameResult,
    WorkerTask,
    aggregate_results,
    create_ai_instance,
    create_worker_tasks,
    divide_memory_budget,
    play_single_game,
    run_parallel_self_play,
)


class TestMemoryBudgetDivision:
    """Test memory budget division across workers."""

    def test_divide_equal_workers(self):
        """Test dividing memory equally among workers."""
        total_memory = 8.0  # 8 GB
        num_workers = 4

        per_worker = divide_memory_budget(total_memory, num_workers)

        # Each worker should get 2 GB
        assert per_worker == 2.0

    def test_divide_uneven_workers(self):
        """Test dividing memory with uneven division."""
        total_memory = 10.0  # 10 GB
        num_workers = 3

        per_worker = divide_memory_budget(total_memory, num_workers)

        # Should be approximately 3.33 GB per worker
        assert abs(per_worker - (10.0 / 3.0)) < 0.01

    def test_divide_single_worker(self):
        """Test single worker gets all memory."""
        total_memory = 8.0  # 8 GB
        num_workers = 1

        per_worker = divide_memory_budget(total_memory, num_workers)

        assert per_worker == 8.0

    def test_divide_zero_memory(self):
        """Test zero memory budget."""
        total_memory = 0.0
        num_workers = 4

        per_worker = divide_memory_budget(total_memory, num_workers)

        assert per_worker == 0.0


class TestWorkerTaskCreation:
    """Test worker task creation and game distribution."""

    def test_create_worker_tasks_even_distribution(self):
        """Test even distribution of games across workers."""
        num_games = 12
        num_workers = 4
        per_worker_memory = 2.0
        base_seed = 42
        ai_type = "heuristic"

        tasks = create_worker_tasks(
            num_games=num_games,
            num_workers=num_workers,
            per_worker_memory_gb=per_worker_memory,
            base_seed=base_seed,
            ai_type=ai_type,
        )

        assert len(tasks) == 4
        # Each worker should get 3 games
        for task in tasks:
            assert len(task.game_indices) == 3
            assert task.memory_config_gb == 2.0
            assert task.ai_type == "heuristic"
            assert task.base_seed == 42

    def test_create_worker_tasks_uneven_distribution(self):
        """Test uneven distribution of games across workers."""
        num_games = 10
        num_workers = 3
        per_worker_memory = 2.0
        base_seed = 42
        ai_type = "heuristic"

        tasks = create_worker_tasks(
            num_games=num_games,
            num_workers=num_workers,
            per_worker_memory_gb=per_worker_memory,
            base_seed=base_seed,
            ai_type=ai_type,
        )

        assert len(tasks) == 3
        total_games = sum(len(t.game_indices) for t in tasks)
        assert total_games == 10

    def test_create_worker_tasks_more_workers_than_games(self):
        """Test when there are more workers than games."""
        num_games = 2
        num_workers = 4
        per_worker_memory = 2.0
        base_seed = 42
        ai_type = "heuristic"

        tasks = create_worker_tasks(
            num_games=num_games,
            num_workers=num_workers,
            per_worker_memory_gb=per_worker_memory,
            base_seed=base_seed,
            ai_type=ai_type,
        )

        # Some workers may have no games
        total_games = sum(len(t.game_indices) for t in tasks)
        assert total_games == 2


class TestAIInstanceCreation:
    """Test AI instance creation."""

    def test_create_heuristic_ai(self):
        """Test creating a heuristic AI instance."""
        ai = create_ai_instance("heuristic", 1, 42)
        assert ai is not None

    def test_create_random_ai(self):
        """Test creating a random AI instance."""
        ai = create_ai_instance("random", 1, 42)
        assert ai is not None

    def test_create_minimax_ai(self):
        """Test creating a minimax/descent AI instance."""
        ai = create_ai_instance("minimax", 1, 42)
        assert ai is not None

    def test_invalid_ai_type_raises(self):
        """Test that invalid AI type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown AI type"):
            create_ai_instance("invalid_type", 1, 42)


class TestGameResult:
    """Test GameResult dataclass."""

    def test_game_result_creation(self):
        """Test creating a GameResult instance."""
        result = GameResult(
            features=np.array([1, 2, 3]),
            globals=np.array([4, 5, 6]),
            values=np.array([0.5, -0.5, 0.0]),
            policy_indices=np.array([[0, 1], [2, 3], [4, 5]]),
            policy_values=np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]]),
            game_length=50,
            winner=1,
            seed=42,
            worker_id=0,
        )

        assert result.game_length == 50
        assert result.winner == 1
        assert result.seed == 42
        assert result.worker_id == 0


class TestResultAggregation:
    """Test result aggregation from workers."""

    def test_aggregate_empty_results(self):
        """Test aggregating empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stats = aggregate_results([], output_dir)

            assert stats["total_games"] == 0
            assert stats["total_positions"] == 0

    def test_aggregate_single_result(self):
        """Test aggregating a single result."""
        result = GameResult(
            features=np.array([]),
            globals=np.array([]),
            values=np.array([], dtype=np.float32),
            policy_indices=np.array([], dtype=np.int32),
            policy_values=np.array([], dtype=np.float32),
            game_length=50,
            winner=1,
            seed=42,
            worker_id=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stats = aggregate_results([result], output_dir)

            assert stats["total_games"] == 1
            assert stats["wins_p1"] == 1
            assert stats["wins_p2"] == 0
            assert "output_path" in stats

    def test_aggregate_multiple_results(self):
        """Test aggregating multiple results."""
        results = [
            GameResult(
                features=np.array([]),
                globals=np.array([]),
                values=np.array([], dtype=np.float32),
                policy_indices=np.array([], dtype=np.int32),
                policy_values=np.array([], dtype=np.float32),
                game_length=50,
                winner=1,
                seed=42,
                worker_id=0,
            ),
            GameResult(
                features=np.array([]),
                globals=np.array([]),
                values=np.array([], dtype=np.float32),
                policy_indices=np.array([], dtype=np.int32),
                policy_values=np.array([], dtype=np.float32),
                game_length=60,
                winner=2,
                seed=43,
                worker_id=0,
            ),
            GameResult(
                features=np.array([]),
                globals=np.array([]),
                values=np.array([], dtype=np.float32),
                policy_indices=np.array([], dtype=np.int32),
                policy_values=np.array([], dtype=np.float32),
                game_length=70,
                winner=None,
                seed=44,
                worker_id=1,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stats = aggregate_results(results, output_dir)

            assert stats["total_games"] == 3
            assert stats["wins_p1"] == 1
            assert stats["wins_p2"] == 1
            assert stats["draws"] == 1

    def test_aggregate_creates_output_file(self):
        """Test that aggregation creates an output file."""
        result = GameResult(
            features=np.array([]),
            globals=np.array([]),
            values=np.array([], dtype=np.float32),
            policy_indices=np.array([], dtype=np.int32),
            policy_values=np.array([], dtype=np.float32),
            game_length=50,
            winner=1,
            seed=42,
            worker_id=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stats = aggregate_results([result], output_dir)

            output_path = Path(stats["output_path"])
            assert output_path.exists()
            assert output_path.suffix == ".npz"


class TestPlaySingleGame:
    """Test single game execution."""

    def test_play_single_game_completes(self):
        """Test that a single game runs to completion."""
        game_length, winner, termination_reason = play_single_game(
            seed=42,
            ai_type="random",
            memory_config=None,
        )

        # Game should complete within move limit
        assert game_length > 0
        assert game_length <= 500  # Max moves limit
        # Winner should be 1, 2, or None (if abnormal termination)
        assert winner in [1, 2, None]
        # Termination reason should be set
        assert termination_reason is not None
        assert len(termination_reason) > 0

    def test_play_single_game_deterministic(self):
        """Test that same seed produces same game length."""
        game_length1, winner1, reason1 = play_single_game(
            seed=12345,
            ai_type="random",
            memory_config=None,
        )

        game_length2, winner2, reason2 = play_single_game(
            seed=12345,
            ai_type="random",
            memory_config=None,
        )

        # Same seed should produce identical results
        assert game_length1 == game_length2
        assert winner1 == winner2
        assert reason1 == reason2


class TestParallelExecution:
    """Test parallel execution of self-play games."""

    def test_run_parallel_small_count(self):
        """Test parallel execution with a small number of games."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            stats = run_parallel_self_play(
                num_games=4,
                num_workers=2,
                output_dir=output_dir,
                ai_type="random",
                base_seed=42,
                memory_budget_gb=0.0,  # No memory limit for test
            )

            assert stats["total_games"] == 4
            assert stats["wins_p1"] + stats["wins_p2"] + stats["draws"] == 4

    def test_run_parallel_single_worker(self):
        """Test parallel execution with single worker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            stats = run_parallel_self_play(
                num_games=2,
                num_workers=1,
                output_dir=output_dir,
                ai_type="random",
                base_seed=42,
                memory_budget_gb=0.0,
            )

            assert stats["total_games"] == 2

    def test_run_parallel_creates_output(self):
        """Test that parallel execution creates output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            stats = run_parallel_self_play(
                num_games=2,
                num_workers=1,
                output_dir=output_dir,
                ai_type="random",
                base_seed=42,
                memory_budget_gb=0.0,
            )

            output_path = Path(stats["output_path"])
            assert output_path.exists()

            # Verify npz contents
            data = np.load(output_path)
            assert "total_games" in data


class TestWorkerTask:
    """Test WorkerTask dataclass."""

    def test_worker_task_creation(self):
        """Test creating a WorkerTask instance."""
        task = WorkerTask(
            worker_id=0,
            game_indices=[0, 1, 2],
            base_seed=42,
            ai_type="heuristic",
            memory_config_gb=2.0,
        )

        assert task.worker_id == 0
        assert task.game_indices == [0, 1, 2]
        assert task.base_seed == 42
        assert task.ai_type == "heuristic"
        assert task.memory_config_gb == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])