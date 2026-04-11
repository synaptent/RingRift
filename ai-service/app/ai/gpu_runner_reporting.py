"""Reporting helpers for GPU parallel game simulation."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("app.ai.gpu_parallel_games")


class GPURunnerReportingMixin:
    """Event/reporting helpers shared by ParallelGameRunner."""

    def _emit_gpu_selfplay_complete(
        self,
        games_count: int,
        elapsed_seconds: float,
        success: bool = True,
        task_id: str | None = None,
        iteration: int = 0,
        error: str = "",
    ) -> None:
        """Emit GPU selfplay completion to the coordination layer."""
        del elapsed_seconds
        try:
            import asyncio
            import socket

            from app.coordination.selfplay_orchestrator import emit_selfplay_completion

            node_id = socket.gethostname()
            if task_id is None:
                task_id = f"gpu_selfplay_{self.batch_size}_{int(time.time())}"

            board_type = self.board_type or f"square{self.board_size}"

            async def emit() -> None:
                await emit_selfplay_completion(
                    task_id=task_id,
                    board_type=board_type,
                    num_players=self.num_players,
                    games_generated=games_count,
                    success=success,
                    node_id=node_id,
                    selfplay_type="gpu_selfplay",
                    iteration=iteration,
                    error=error,
                )

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(emit())
            except RuntimeError:
                asyncio.run(emit())

            logger.debug(
                "Emitted GPU_SELFPLAY_COMPLETE: %s games, task_id=%s",
                games_count,
                task_id,
            )
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Failed to emit GPU_SELFPLAY_COMPLETE: %s", exc)

    def get_stats(self) -> dict[str, float]:
        """Get performance statistics."""
        return {
            "games_completed": self._games_completed,
            "total_moves": self._total_moves,
            "total_time_seconds": self._total_time,
            "games_per_second": (
                self._games_completed / self._total_time if self._total_time > 0 else 0
            ),
            "moves_per_second": (
                self._total_moves / self._total_time if self._total_time > 0 else 0
            ),
        }
