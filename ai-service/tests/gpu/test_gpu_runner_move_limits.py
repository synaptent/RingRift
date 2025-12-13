import pytest
import torch


try:
    from app.ai.gpu_parallel_games import GameStatus, ParallelGameRunner

    GPU_MODULES_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import guard
    GPU_MODULES_AVAILABLE = False
    GPU_IMPORT_ERROR = str(exc)


pytestmark = pytest.mark.skipif(
    not GPU_MODULES_AVAILABLE,
    reason=f"GPU modules not available: {GPU_IMPORT_ERROR if not GPU_MODULES_AVAILABLE else ''}",
)


def test_gpu_run_games_max_moves_is_move_count_limit() -> None:
    """Regression guard: max_moves should apply to recorded move_count, not phase steps."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=4,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
    )

    max_moves = 5
    results = runner.run_games(max_moves=max_moves)

    assert len(results["status"]) == runner.batch_size
    assert len(results["move_counts"]) == runner.batch_size

    # At least one game should hit the move limit at this small threshold.
    assert any(status == GameStatus.MAX_MOVES for status in results["status"])

    for status, move_count in zip(results["status"], results["move_counts"]):
        if status == GameStatus.MAX_MOVES:
            assert move_count >= max_moves

