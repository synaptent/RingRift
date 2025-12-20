"""Background selfplay manager for overlapped pipeline execution.

This module enables running selfplay for iteration N+1 in background
while training/evaluation for iteration N is in progress.

Usage:
    manager = get_background_selfplay_manager()

    # Start selfplay in background
    task = manager.start_background_selfplay(config, iteration=1)

    # ... do training/evaluation for current iteration ...

    # Wait for background selfplay to complete
    success, staging_db_path, games = manager.wait_for_current()
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Get AI_SERVICE_ROOT
from app.utils.paths import AI_SERVICE_ROOT

# Import coordination for task limits
try:
    from app.coordination import (
        TaskType,
        register_running_task,
    )
    from app.coordination.helpers import can_spawn_safe
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    TaskType = None
    can_spawn_safe = None  # type: ignore[assignment]


@dataclass
class BackgroundSelfplayTask:
    """Track a background selfplay process."""

    iteration: int
    process: subprocess.Popen | None = None
    staging_db_path: Path | None = None
    games_requested: int = 0
    start_time: float = field(default_factory=time.time)
    board_type: str = "square8"
    num_players: int = 2

    def is_running(self) -> bool:
        """Check if the background process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def wait(self, timeout: float | None = None) -> tuple[bool, int]:
        """Wait for background process to complete.

        Returns: (success, return_code)
        """
        if self.process is None:
            return True, 0
        try:
            self.process.wait(timeout=timeout)
            return self.process.returncode == 0, self.process.returncode
        except subprocess.TimeoutExpired:
            return False, -1

    def terminate(self) -> None:
        """Terminate the background process."""
        if self.process is not None and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def elapsed_time(self) -> float:
        """Get elapsed time since task started."""
        return time.time() - self.start_time


class BackgroundSelfplayManager:
    """Manager for background selfplay processes.

    Enables overlapped pipeline execution where selfplay for iteration N+1
    runs while training/evaluation for iteration N is in progress.
    """

    def __init__(self, ai_service_root: Path | None = None):
        self._ai_service_root = ai_service_root or AI_SERVICE_ROOT
        self._current_task: BackgroundSelfplayTask | None = None
        self._history: list[BackgroundSelfplayTask] = []

    def start_background_selfplay(
        self,
        config: dict,
        iteration: int,
    ) -> BackgroundSelfplayTask | None:
        """Start selfplay in background.

        Args:
            config: Configuration dict with board, players, games_per_iter, etc.
            iteration: The iteration number for this selfplay run.

        Returns:
            The BackgroundSelfplayTask if started successfully, None otherwise.
        """
        # Cancel any existing task
        if self._current_task is not None and self._current_task.is_running():
            print(f"[background] Cancelling previous task for iteration {self._current_task.iteration}")
            self._current_task.terminate()

        board = str(config.get("board", "square8"))
        players = int(config.get("players", 2))
        games = int(config.get("games_per_iter", 100))
        max_moves = int(config.get("max_moves", 2000))  # Minimum 2000 for all boards

        # Check coordination before spawning (advisory)
        if HAS_COORDINATION and can_spawn_safe is not None:
            try:
                node_id = socket.gethostname()
                allowed, reason = can_spawn_safe(TaskType.SELFPLAY, node_id)
                if not allowed:
                    print(f"[background] Coordination warning: {reason}")
                    print("[background] Proceeding anyway (coordination is advisory)")
            except Exception as e:
                print(f"[background] Coordination check error: {e}")

        # Determine staging DB path
        staging_db_dir_raw = config.get("staging_db_dir", "data/games/staging")
        staging_db_dir = Path(staging_db_dir_raw)
        if not staging_db_dir.is_absolute():
            staging_db_dir = self._ai_service_root / staging_db_dir
        staging_db_dir.mkdir(parents=True, exist_ok=True)

        staging_db_name = f"staging_{board}_{players}p_iter{iteration}_{int(time.time())}.db"
        staging_db_path = staging_db_dir / staging_db_name

        # Build selfplay command
        cmd = [
            sys.executable,
            "scripts/run_self_play_soak.py",
            "--board", board,
            "--players", str(players),
            "--games", str(games),
            "--max-moves", str(max_moves),
            "--replay-db", str(staging_db_path),
        ]

        # Add optional parameters
        if config.get("selfplay_difficulty_band"):
            cmd.extend(["--difficulty-band", str(config["selfplay_difficulty_band"])])
        if config.get("selfplay_engine_mode"):
            cmd.extend(["--engine-mode", str(config["selfplay_engine_mode"])])
        if config.get("selfplay_nn_pool_size"):
            cmd.extend(["--nn-pool-size", str(config["selfplay_nn_pool_size"])])
        if config.get("selfplay_nn_pool_dir"):
            cmd.extend(["--nn-pool-dir", str(config["selfplay_nn_pool_dir"])])

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self._ai_service_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            task = BackgroundSelfplayTask(
                iteration=iteration,
                process=process,
                staging_db_path=staging_db_path,
                games_requested=games,
                board_type=board,
                num_players=players,
            )
            self._current_task = task
            print(f"[background] Started selfplay for iteration {iteration} (PID: {process.pid})")
            print(f"[background]   Board: {board}, Players: {players}, Games: {games}")
            print(f"[background]   Output: {staging_db_path}")
            return task
        except Exception as e:
            print(f"[background] Failed to start selfplay: {e}")
            return None

    def _emit_selfplay_complete(
        self,
        task: BackgroundSelfplayTask,
        success: bool,
        games_generated: int,
    ) -> None:
        """Emit selfplay completion event via SelfplayOrchestrator (December 2025)."""
        try:
            import asyncio
            import socket

            from app.coordination.selfplay_orchestrator import emit_selfplay_completion

            node_id = socket.gethostname()
            task_id = f"background_selfplay_{task.iteration}_{int(task.start_time)}"

            try:
                asyncio.get_running_loop()
                asyncio.create_task(emit_selfplay_completion(
                    task_id=task_id,
                    board_type=task.board_type,
                    num_players=task.num_players,
                    games_generated=games_generated,
                    success=success,
                    node_id=node_id,
                    selfplay_type="background",
                    iteration=task.iteration,
                ))
            except RuntimeError:
                # No event loop, run synchronously
                asyncio.run(emit_selfplay_completion(
                    task_id=task_id,
                    board_type=task.board_type,
                    num_players=task.num_players,
                    games_generated=games_generated,
                    success=success,
                    node_id=node_id,
                    selfplay_type="background",
                    iteration=task.iteration,
                ))

            print(f"[background] Emitted SELFPLAY_COMPLETE for iteration {task.iteration}")
        except ImportError:
            pass  # SelfplayOrchestrator not available
        except Exception as e:
            print(f"[background] Failed to emit SELFPLAY_COMPLETE: {e}")

    def get_current_task(self) -> BackgroundSelfplayTask | None:
        """Get the current background task."""
        return self._current_task

    def wait_for_current(
        self,
        timeout: float | None = None,
    ) -> tuple[bool, Path | None, int]:
        """Wait for current background task to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            Tuple of (success, staging_db_path, games_count)
        """
        if self._current_task is None:
            return True, None, 0

        task = self._current_task
        success, code = task.wait(timeout=timeout)

        if success:
            elapsed = task.elapsed_time()
            print(f"[background] Selfplay iteration {task.iteration} completed in {elapsed:.1f}s")
            staging_path = task.staging_db_path
            games = task.games_requested
        else:
            print(f"[background] Selfplay iteration {task.iteration} failed (code={code})")
            staging_path = None
            games = 0

        # Move to history
        self._history.append(task)
        self._current_task = None

        # Emit selfplay completion event (December 2025)
        self._emit_selfplay_complete(task, success, games)

        return success, staging_path, games

    def cancel_current(self) -> None:
        """Cancel the current background task."""
        if self._current_task is not None:
            print(f"[background] Cancelling selfplay iteration {self._current_task.iteration}")
            self._current_task.terminate()
            self._current_task = None

    def has_pending_task(self) -> bool:
        """Check if there's a pending background task."""
        return self._current_task is not None and self._current_task.is_running()

    def get_statistics(self) -> dict:
        """Get statistics about background selfplay."""
        completed = [t for t in self._history if t.process and t.process.returncode == 0]
        failed = [t for t in self._history if t.process and t.process.returncode != 0]

        return {
            "total_runs": len(self._history),
            "completed": len(completed),
            "failed": len(failed),
            "current_running": self.has_pending_task(),
            "current_iteration": self._current_task.iteration if self._current_task else None,
            "avg_elapsed_time": (
                sum(t.elapsed_time() for t in completed) / len(completed)
                if completed else 0.0
            ),
        }


# Global singleton manager
_background_selfplay_manager: BackgroundSelfplayManager | None = None


def get_background_selfplay_manager(
    ai_service_root: Path | None = None,
) -> BackgroundSelfplayManager:
    """Get or create the global background selfplay manager.

    Args:
        ai_service_root: Optional path to ai-service root directory.

    Returns:
        The singleton BackgroundSelfplayManager instance.
    """
    global _background_selfplay_manager
    if _background_selfplay_manager is None:
        _background_selfplay_manager = BackgroundSelfplayManager(ai_service_root)
    return _background_selfplay_manager


def reset_background_selfplay_manager() -> None:
    """Reset the global manager (useful for testing)."""
    global _background_selfplay_manager
    if _background_selfplay_manager is not None:
        _background_selfplay_manager.cancel_current()
        _background_selfplay_manager = None
