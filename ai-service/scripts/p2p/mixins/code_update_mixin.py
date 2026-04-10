"""Code Update Mixin - git/version detection and orchestrator restart helpers.

April 2026: Extracted from p2p_orchestrator.py (Phase 4 task 17).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.p2p.constants import (
    BUILD_VERSION_ENV,
    GIT_BRANCH_NAME,
    GIT_REMOTE_NAME,
    GRACEFUL_SHUTDOWN_BEFORE_UPDATE,
)
from scripts.p2p.p2p_mixin_base import P2PMixinBase

logger = logging.getLogger(__name__)


async def async_subprocess_run(
    cmd: list[str],
    cwd: str | Path | None = None,
    timeout: float = 30.0,
    capture_output: bool = True,
    text: bool = True,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run subprocess in a thread pool to avoid blocking the event loop."""
    def _run():
        return subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            env=env,
        )

    return await asyncio.to_thread(_run)


class CodeUpdateMixin(P2PMixinBase):
    """Mixin extracted from P2POrchestrator."""

    MIXIN_TYPE = "code_update"

    ringrift_path: str
    jobs_lock: Any
    local_jobs: dict[str, Any]

    def _detect_build_version(self) -> str:
        env_version = (os.environ.get(BUILD_VERSION_ENV, "") or "").strip()
        if env_version:
            return env_version

        commit = ""
        branch = ""
        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "--short", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError, AttributeError):
            commit = ""

        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "--abbrev-ref", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError, AttributeError):
            branch = ""

        if commit and branch:
            return f"{branch}@{commit}"
        return commit or "unknown"

    def _git_cmd(self, *args: str) -> list[str]:
        safe_dir = getattr(self, "_git_safe_directory", "") or os.path.abspath(self.ringrift_path)
        return ["git", "-c", f"safe.directory={safe_dir}", *args]

    def _detect_ringrift_path(self) -> str:
        """Detect the RingRift installation path."""
        # Try common locations
        candidates = [
            Path.home() / "Development" / "RingRift",
            Path.home() / "ringrift",
            Path("/home/ubuntu/ringrift"),
            Path("/root/ringrift"),
        ]
        for path in candidates:
            if (path / "ai-service").exists():
                return str(path)
        return str(Path(__file__).resolve().parents[4])

    def _get_ai_service_path(self) -> str:
        """Get the path to the ai-service directory.

        Handles both cases:
        - ringrift_path = /path/to/RingRift (root directory)
        - ringrift_path = /path/to/RingRift/ai-service (already ai-service)

        Returns:
            Path to ai-service directory.
        """
        if self.ringrift_path.rstrip("/").endswith("ai-service"):
            return self.ringrift_path
        return os.path.join(self.ringrift_path, "ai-service")

    async def _get_local_git_commit(self) -> str | None:
        """Get the current local git commit hash (async)."""
        try:
            result = await async_subprocess_run(
                self._git_cmd("rev-parse", "HEAD"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get local git commit: {e}")
        return None

    async def _get_local_git_branch(self) -> str | None:
        """Get the current local git branch name (async)."""
        try:
            result = await async_subprocess_run(
                self._git_cmd("rev-parse", "--abbrev-ref", "HEAD"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get local git branch: {e}")
        return None

    async def _get_remote_git_commit(self) -> str | None:
        """Fetch and get the remote branch's latest commit hash (async)."""
        try:
            # First fetch to update remote refs
            fetch_result = await async_subprocess_run(
                self._git_cmd("fetch", GIT_REMOTE_NAME, GIT_BRANCH_NAME),
                cwd=self.ringrift_path,
                timeout=60
            )
            if fetch_result.returncode != 0:
                logger.info(f"Git fetch failed: {fetch_result.stderr}")
                return None

            # Get remote branch commit
            result = await async_subprocess_run(
                self._git_cmd("rev-parse", f"{GIT_REMOTE_NAME}/{GIT_BRANCH_NAME}"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get remote git commit: {e}")
        return None

    async def _check_for_updates(self) -> tuple[bool, str | None, str | None]:
        """Check if there are updates available from GitHub (async).

        Returns: (has_updates, local_commit, remote_commit)
        """
        # Run both git queries in parallel
        local_commit, remote_commit = await asyncio.gather(
            self._get_local_git_commit(),
            self._get_remote_git_commit(),
        )

        if not local_commit or not remote_commit:
            return False, local_commit, remote_commit

        has_updates = local_commit != remote_commit
        return has_updates, local_commit, remote_commit

    async def _get_commits_behind(self, local_commit: str, remote_commit: str) -> int:
        """Get the number of commits the local branch is behind remote (async)."""
        try:
            result = await async_subprocess_run(
                self._git_cmd("rev-list", "--count", f"{local_commit}..{remote_commit}"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to count commits behind: {e}")
        return 0

    async def _check_local_changes(self) -> bool:
        """Check if there are uncommitted local changes (async).

        Notes:
        - Ignore untracked files by default. Cluster nodes often accumulate local
          artifacts (logs, data, env backups) that should not block git updates.
        - Still blocks on tracked/staged modifications to avoid stomping on
          local hotfixes.
        """
        try:
            result = await async_subprocess_run(
                self._git_cmd("status", "--porcelain", "--untracked-files=no"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                # If there's output, there are uncommitted changes
                return bool(result.stdout.strip())
            logger.error(
                "Failed to check local changes: git status returned %s (%s)",
                result.returncode,
                (result.stderr or "").strip(),
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to check local changes: {e}")
        # Fail closed: if we cannot confirm the tree is clean, do not auto-update.
        return True

    async def _stop_all_local_jobs(self) -> int:
        """Stop all local jobs gracefully before update.

        Returns: Number of jobs stopped
        """
        stopped = 0
        with self.jobs_lock:
            for job_id, job in list(self.local_jobs.items()):
                try:
                    if job.pid > 0:
                        os.kill(job.pid, signal.SIGTERM)
                        logger.info(f"Sent SIGTERM to job {job_id} (PID {job.pid})")
                        stopped += 1
                        job.status = "stopping"
                except ProcessLookupError:
                    # Process already gone
                    job.status = "stopped"
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to stop job {job_id}: {e}")

        # Wait for processes to terminate gracefully
        # GPU games can take 1-10 minutes, so use a longer timeout (Dec 2025 fix)
        grace_period = int(os.environ.get("RINGRIFT_JOB_GRACE_PERIOD", "60"))
        if stopped > 0:
            await asyncio.sleep(grace_period)

            # Force kill any remaining
            with self.jobs_lock:
                for job_id, job in list(self.local_jobs.items()):
                    if job.status == "stopping" and job.pid > 0:
                        try:
                            os.kill(job.pid, signal.SIGKILL)
                            logger.info(f"Force killed job {job_id}")
                        except OSError:
                            pass  # Process already dead
                        job.status = "stopped"

        return stopped

    async def _perform_git_update(self) -> tuple[bool, str]:
        """Perform git pull to update the codebase (async).

        Returns: (success, message)
        """
        # Check for local changes (async)
        if await self._check_local_changes():
            return False, "Local changes detected. Cannot auto-update. Please commit or stash changes."

        # Stop jobs if configured
        if GRACEFUL_SHUTDOWN_BEFORE_UPDATE:
            stopped = await self._stop_all_local_jobs()
            if stopped > 0:
                logger.info(f"Stopped {stopped} jobs before update")

        try:
            # Perform git pull (async - Jan 19, 2026)
            result = await async_subprocess_run(
                self._git_cmd("pull", GIT_REMOTE_NAME, GIT_BRANCH_NAME),
                cwd=self.ringrift_path,
                timeout=120
            )

            if result.returncode != 0:
                return False, f"Git pull failed: {result.stderr}"

            logger.info(f"Git pull successful: {result.stdout}")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            return False, "Git pull timed out"
        except Exception as e:  # noqa: BLE001
            return False, f"Git pull error: {e}"

    async def _restart_orchestrator(self):
        """Restart the orchestrator process after update."""
        logger.info("Restarting orchestrator to apply updates...")

        # Save state before restart
        self._save_state()

        # Get current script path and arguments
        script_path = Path(__file__).resolve().parents[2] / "p2p_orchestrator.py"
        args = sys.argv[1:]

        # Schedule restart
        await asyncio.sleep(2)

        # Use exec to replace current process
        os.execv(sys.executable, [sys.executable, str(script_path), *args])
