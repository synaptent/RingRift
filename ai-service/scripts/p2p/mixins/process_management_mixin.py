"""Process cleanup and subprocess utility helpers.

April 2026: Extracted from p2p_orchestrator.py (Part 3 Phase 2).
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class ProcessManagementMixin(P2PMixinBase):
    """Mixin for P2POrchestrator process cleanup and subprocess utility helpers."""

    MIXIN_TYPE = "process_management"

    def _run_subprocess_sync(self, cmd: list, timeout: int = 10) -> tuple[int, str, str]:
        """Run subprocess synchronously.

        IMPORTANT: This is a blocking operation. Call via asyncio.to_thread() from async code.
        Added Dec 2025 to fix P2P orchestrator CPU spikes from blocking subprocess in async loops.

        Returns: (return_code, stdout, stderr)
        """
        import subprocess
        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
            return (result.returncode, result.stdout or "", result.stderr or "")
        except subprocess.TimeoutExpired:
            return (-1, "", "timeout")
        except (OSError, subprocess.SubprocessError) as e:
            return (-1, "", str(e))

    async def _run_subprocess_async(self, cmd: list, timeout: int = 10) -> tuple[int, str, str]:
        """Run subprocess asynchronously via thread pool.

        Jan 2026: Added for Phase 1 multi-core parallelization.
        Uses asyncio.to_thread() to avoid blocking the event loop.

        Returns: (return_code, stdout, stderr)
        """
        return await asyncio.to_thread(self._run_subprocess_sync, cmd, timeout)

    def _get_max_selfplay_slots_for_node(self) -> int:
        """Get maximum selfplay slots based on GPU capability.

        Jan 2, 2026: Added for slot-based capacity management.
        This allows work queue claiming to coexist with legacy selfplay processes.

        The slot count is based on GPU type since different GPUs can handle
        different numbers of concurrent selfplay processes effectively.

        Returns:
            Maximum number of selfplay slots for this node.
        """
        import os

        # Check environment variable first (allows manual override)
        env_slots = os.environ.get("RINGRIFT_MAX_SELFPLAY_SLOTS")
        if env_slots:
            try:
                return int(env_slots)
            except ValueError:
                pass

        # Compute based on GPU name
        gpu_name = getattr(self.self_info, "gpu_name", "") or ""
        gpu_name_lower = gpu_name.lower()

        # High-end GPUs get more slots
        if "gh200" in gpu_name_lower or "h100" in gpu_name_lower:
            return 16
        elif "a100" in gpu_name_lower:
            return 12
        elif "5090" in gpu_name_lower or "4090" in gpu_name_lower:
            return 8
        elif "3090" in gpu_name_lower or "a40" in gpu_name_lower or "l40" in gpu_name_lower:
            return 6
        elif "4060" in gpu_name_lower or "3060" in gpu_name_lower:
            return 3
        elif self.self_info.has_gpu:
            return 4  # Default for other GPUs
        else:
            return 2  # CPU-only nodes

    def _reap_orphan_processes(self) -> int:
        """Kill orphan Python processes from previous P2P/master_loop runs.

        Mar 2026: When the P2P process restarts (via LaunchAgent KeepAlive or
        manual restart), child processes from the old run become orphans.
        These accumulate over time (148 zombies observed on mac-studio).

        This runs at startup and kills any Python processes that:
        1. Are children of PID 1 (reparented orphans)
        2. Match known RingRift process patterns (selfplay, train, gauntlet)
        3. Started before the current process

        Returns:
            Number of processes killed
        """
        import signal as _sig

        my_pid = os.getpid()
        my_start = os.path.getctime(f"/proc/{my_pid}") if os.path.exists(f"/proc/{my_pid}") else time.time()
        killed = 0

        try:
            import subprocess
            # Get all python processes with their PIDs and command lines
            result = subprocess.run(
                ["ps", "-eo", "pid,ppid,lstart,args"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return 0

            ringrift_patterns = [
                "selfplay", "train", "gauntlet", "export_replay",
                "gpu_parallel_games", "game_gauntlet",
            ]

            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                except ValueError:
                    continue

                if pid == my_pid:
                    continue

                cmd = " ".join(parts[4:]).lower()  # lstart takes 5 fields
                # Heuristic: skip lines where we can't find 'python' in the command
                # (ps output on macOS has variable-width lstart field)
                if "python" not in cmd:
                    continue

                # Only kill orphaned processes (ppid=1 means reparented)
                # and processes that match RingRift patterns
                is_orphan = ppid == 1
                is_ringrift = any(p in cmd for p in ringrift_patterns)

                if is_orphan and is_ringrift:
                    try:
                        os.kill(pid, _sig.SIGTERM)
                        killed += 1
                        logger.info(f"[OrphanReaper] Killed orphan PID {pid}: {cmd[:80]}")
                    except (OSError, ProcessLookupError):
                        pass

            if killed:
                logger.info(f"[OrphanReaper] Killed {killed} orphan processes at startup")
        except Exception as e:
            logger.debug(f"[OrphanReaper] Failed: {e}")

        return killed

    def _cleanup_stale_processes(self) -> int:
        """Kill processes that have been running too long.

        Jan 29, 2026: Delegated to JobOrchestrator.cleanup_stale_processes().
        """
        return self.jobs.cleanup_stale_processes()

    def _cleanup_orphan_gpu_processes(self) -> int:
        """Detect GPU processes not tracked in local_jobs and warn about them.

        Feb 2026: On P2P startup, previous sessions may have left training or
        selfplay processes that occupy GPU memory. We detect them via nvidia-smi
        and log warnings so operators can decide whether to kill them.

        Returns:
            Number of orphan GPU processes found.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except FileNotFoundError:
            return 0  # No nvidia-smi = no GPU
        except subprocess.TimeoutExpired:
            logger.warning("[P2P] nvidia-smi timed out during orphan detection")
            return 0

        if result.returncode != 0:
            return 0

        tracked_pids = set()
        for job in self.local_jobs.values():
            pid = getattr(job, "pid", 0) or 0
            if pid > 0:
                tracked_pids.add(pid)

        orphan_count = 0
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                gpu_pid = int(parts[0])
            except (ValueError, IndexError):
                continue
            proc_name = parts[1] if len(parts) > 1 else "unknown"
            mem_mb = parts[2] if len(parts) > 2 else "?"

            if gpu_pid not in tracked_pids:
                orphan_count += 1
                logger.warning(
                    f"[P2P] Orphan GPU process: PID={gpu_pid} "
                    f"name={proc_name} mem={mem_mb}MB (not tracked in local_jobs)"
                )

        if orphan_count > 0:
            logger.warning(
                f"[P2P] Found {orphan_count} orphan GPU processes. "
                "These may block work claiming due to GPU memory usage. "
                "Consider killing them manually if they're from previous sessions."
            )
        return orphan_count
