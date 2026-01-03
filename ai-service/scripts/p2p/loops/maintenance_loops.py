"""Maintenance Loops for P2P Orchestrator.

December 2025: Background loops for system maintenance tasks.

Loops:
- GitUpdateLoop: Periodically checks for and applies git updates with auto-restart

Usage:
    from scripts.p2p.loops import GitUpdateLoop, GitUpdateConfig

    git_loop = GitUpdateLoop(
        check_for_updates=orchestrator._check_for_updates,
        perform_update=orchestrator._perform_git_update,
        restart_orchestrator=orchestrator._restart_orchestrator,
    )
    await git_loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .base import BaseLoop

from .loop_constants import LoopIntervals, AUTO_UPDATE_ENABLED

logger = logging.getLogger(__name__)


# Backward-compat alias (Sprint 10: use LoopIntervals.GIT_UPDATE_CHECK instead)
GIT_UPDATE_CHECK_INTERVAL = LoopIntervals.GIT_UPDATE_CHECK
# AUTO_UPDATE_ENABLED is now imported from loop_constants


@dataclass
class GitUpdateConfig:
    """Configuration for git update loop.

    December 2025: Extracted from p2p_orchestrator._git_update_loop
    """

    check_interval_seconds: float = GIT_UPDATE_CHECK_INTERVAL
    error_retry_seconds: float = 60.0  # Wait before retry on error
    enabled: bool = AUTO_UPDATE_ENABLED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.error_retry_seconds <= 0:
            raise ValueError("error_retry_seconds must be > 0")


class GitUpdateLoop(BaseLoop):
    """Background loop to periodically check for and apply git updates.

    When updates are detected:
    1. Calculates commits behind remote
    2. Performs git pull
    3. Restarts orchestrator to apply changes

    December 2025: Extracted from p2p_orchestrator._git_update_loop
    """

    def __init__(
        self,
        check_for_updates: Callable[[], tuple[bool, str | None, str | None]],
        perform_update: Callable[[], Coroutine[Any, Any, tuple[bool, str]]],
        restart_orchestrator: Callable[[], Coroutine[Any, Any, None]],
        get_commits_behind: Callable[[str, str], int] | None = None,
        config: GitUpdateConfig | None = None,
    ):
        """Initialize git update loop.

        Args:
            check_for_updates: Callback returning (has_updates, local_commit, remote_commit)
            perform_update: Async callback to perform git pull, returns (success, message)
            restart_orchestrator: Async callback to restart the orchestrator
            get_commits_behind: Optional callback to count commits behind (local, remote) -> count
            config: Update configuration
        """
        self.config = config or GitUpdateConfig()
        super().__init__(
            name="git_update",
            interval=self.config.check_interval_seconds,
        )
        self._check_for_updates = check_for_updates
        self._perform_update = perform_update
        self._restart_orchestrator = restart_orchestrator
        self._get_commits_behind = get_commits_behind

        # Statistics
        self._checks_count = 0
        self._updates_found = 0
        self._updates_applied = 0
        self._update_failures = 0

    async def _on_start(self) -> None:
        """Check if auto-update is enabled."""
        if not self.config.enabled:
            logger.info("[GitUpdate] Auto-update disabled, loop will skip checks")
        else:
            logger.info(
                f"[GitUpdate] Auto-update loop started "
                f"(interval: {self.config.check_interval_seconds}s)"
            )

    async def _run_once(self) -> None:
        """Check for updates and apply if available."""
        if not self.config.enabled:
            return

        self._checks_count += 1

        try:
            # Check for updates
            has_updates, local_commit, remote_commit = self._check_for_updates()

            if has_updates and local_commit and remote_commit:
                self._updates_found += 1

                # Calculate commits behind (if callback provided)
                commits_behind = 0
                if self._get_commits_behind:
                    commits_behind = self._get_commits_behind(local_commit, remote_commit)
                    logger.info(f"[GitUpdate] Update available: {commits_behind} commits behind")

                logger.info(f"[GitUpdate] Local:  {local_commit[:8]}")
                logger.info(f"[GitUpdate] Remote: {remote_commit[:8]}")

                # Perform update
                success, message = await self._perform_update()

                if success:
                    self._updates_applied += 1
                    logger.info("[GitUpdate] Update successful, restarting...")
                    await self._restart_orchestrator()
                else:
                    self._update_failures += 1
                    logger.warning(f"[GitUpdate] Update failed: {message}")

        except Exception as e:
            logger.warning(f"[GitUpdate] Error in update check: {e}")
            await asyncio.sleep(self.config.error_retry_seconds)

    def get_update_stats(self) -> dict[str, Any]:
        """Get update statistics."""
        return {
            "enabled": self.config.enabled,
            "checks_count": self._checks_count,
            "updates_found": self._updates_found,
            "updates_applied": self._updates_applied,
            "update_failures": self._update_failures,
            "success_rate": (
                self._updates_applied / self._updates_found * 100
                if self._updates_found > 0
                else 100.0
            ),
            **self.stats.to_dict(),
        }


__all__ = [
    "GitUpdateConfig",
    "GitUpdateLoop",
]
