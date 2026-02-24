"""SSH Tournament HTTP Handlers Mixin.

Provides HTTP endpoints for SSH-distributed tournament execution.
Enables tournament matches to be executed directly on remote nodes
via SSH, bypassing the HTTP API for lower latency.

Usage:
    class P2POrchestrator(SSHTournamentHandlersMixin, ...):
        pass

Endpoints:
    POST /ssh-tournament/start - Start SSH-based tournament (leader only)
    GET /ssh-tournament/status - Get SSH tournament status
    POST /ssh-tournament/run - Execute tournament run on specific node
    GET /ssh-tournament/runs - List active SSH tournament runs
    POST /ssh-tournament/stop - Stop SSH tournament run

SSH Execution:
    Matches executed via SSH to remote nodes, allowing direct process
    control and output capture. Useful when HTTP API overhead is a concern
    or when nodes need to execute specific Python scripts directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from app.core.async_context import safe_create_task
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_TOURNAMENT

# January 2026: Use centralized timeouts from LoopTimeouts
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    LONG_RUNNING_JOB_TIMEOUT = int(LoopTimeouts.LONG_RUNNING_JOB)
except ImportError:
    LONG_RUNNING_JOB_TIMEOUT = 6 * 60 * 60  # 6 hours fallback

if TYPE_CHECKING:
    from scripts.p2p.models import NodeRole

logger = logging.getLogger(__name__)


class SSHTournamentHandlersMixin(BaseP2PHandler):
    """Mixin providing SSH tournament HTTP handlers.

    Inherits from BaseP2PHandler for common response formatting utilities
    (json_response, error_response).

    Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - leader_id: str
    - ringrift_path: str
    - ssh_tournament_runs: dict
    - ssh_tournament_lock: threading.Lock
    """

    # Type hints for IDE support
    node_id: str
    role: Any  # NodeRole
    leader_id: str
    ringrift_path: str
    ssh_tournament_runs: dict
    ssh_tournament_lock: Any

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_ssh_tournament_start(self, request: web.Request) -> web.Response:
        """Start an SSH-distributed difficulty-tier tournament (leader only).

        This is a thin wrapper that runs `scripts/run_ssh_distributed_tournament.py`
        as a subprocess and tracks its status locally on the leader node.
        """
        from scripts.p2p.constants import STATE_DIR
        from scripts.p2p.models import SSHTournamentRun
        from scripts.p2p.types import NodeRole

        try:
            if self.role != NodeRole.LEADER:
                return self.error_response(
                    "Only the leader can start SSH tournaments",
                    status=403,
                    details={"leader_id": self.leader_id},
                )

            data = await request.json()

            tiers = str(data.get("tiers") or "D1-D10")
            board = str(data.get("board") or data.get("board_type") or "square8").strip().lower()
            if board == "hexagonal":
                board = "hex"
            if board not in ("square8", "square19", "hex8", "hex"):
                return self.error_response(f"Invalid board: {board!r}", status=400)

            games_per_matchup = int(data.get("games_per_matchup", 50) or 50)
            seed = int(data.get("seed", 1) or 1)
            think_time_scale = float(data.get("think_time_scale", 1.0) or 1.0)
            max_moves = int(data.get("max_moves", 10000) or 10000)
            wilson_confidence = float(data.get("wilson_confidence", 0.95) or 0.95)
            nn_model_id = data.get("nn_model_id") or None
            config_path = data.get("config") or None
            include_nonready = bool(data.get("include_nonready", False))
            max_parallel_per_host = data.get("max_parallel_per_host")
            remote_output_dir = str(data.get("remote_output_dir") or "results/tournaments/ssh_shards")
            job_timeout_sec = int(data.get("job_timeout_sec", LONG_RUNNING_JOB_TIMEOUT) or LONG_RUNNING_JOB_TIMEOUT)
            retries = int(data.get("retries", 1) or 1)
            dry_run = bool(data.get("dry_run", False))

            requested_run_id = str(data.get("run_id") or "").strip()
            job_id = requested_run_id or f"ssh_tournament_{uuid.uuid4().hex[:8]}"
            run_id = job_id

            hosts = data.get("hosts")
            hosts_spec: str | None = None
            if isinstance(hosts, list):
                hosts_spec = ",".join(str(h).strip() for h in hosts if str(h).strip())
            elif isinstance(hosts, str) and hosts.strip():
                hosts_spec = hosts.strip()

            output_root = str(
                data.get("output_root") or f"results/tournaments/p2p_orchestrator/{run_id}"
            )

            report_path = str(Path(output_root) / f"report_{run_id}.json")
            checkpoint_path = str(Path(output_root) / f"tournament_{run_id}.json")
            manifest_path = str(Path(output_root) / "manifest.json")

            log_dir = STATE_DIR / "ssh_tournaments"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(log_dir / f"{run_id}.log")

            cmd: list[str] = [
                sys.executable,
                "scripts/run_ssh_distributed_tournament.py",
                "--tiers", tiers,
                "--board", board,
                "--games-per-matchup", str(games_per_matchup),
                "--seed", str(seed),
                "--think-time-scale", str(think_time_scale),
                "--max-moves", str(max_moves),
                "--wilson-confidence", str(wilson_confidence),
                "--remote-output-dir", remote_output_dir,
                "--job-timeout-sec", str(job_timeout_sec),
                "--retries", str(retries),
                "--run-id", run_id,
                "--output-root", output_root,
            ]
            if nn_model_id:
                cmd.extend(["--nn-model-id", str(nn_model_id)])
            if config_path:
                cmd.extend(["--config", str(config_path)])
            if hosts_spec:
                cmd.extend(["--hosts", hosts_spec])
            if include_nonready:
                cmd.append("--include-nonready")
            if max_parallel_per_host is not None:
                cmd.extend(["--max-parallel-per-host", str(int(max_parallel_per_host))])
            if dry_run:
                cmd.append("--dry-run")

            # Handle both root path and ai-service path (avoid doubling)
            ai_service_path = (
                self.ringrift_path
                if self.ringrift_path.rstrip("/").endswith("ai-service")
                else os.path.join(self.ringrift_path, "ai-service")
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = ai_service_path

            cwd = ai_service_path
            with open(log_path, "ab") as log_file:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    cwd=cwd,
                )

            run_state = SSHTournamentRun(
                job_id=job_id,
                run_id=run_id,
                tiers=tiers,
                board=board,
                games_per_matchup=games_per_matchup,
                pid=proc.pid,
                status="running",
                started_at=time.time(),
                output_root=output_root,
                manifest_path=manifest_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                log_path=log_path,
                command=cmd,
            )

            with self.ssh_tournament_lock:
                self.ssh_tournament_runs[job_id] = run_state

            safe_create_task(self._monitor_ssh_tournament_process(job_id, proc), name="ssh-tournament-monitor")

            return self.json_response({"success": True, "job": run_state.to_dict()})
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_ssh_tournament_status(self, request: web.Request) -> web.Response:
        """Get status of SSH-distributed tournaments."""
        try:
            job_id = request.query.get("job_id")

            with self.ssh_tournament_lock:
                if job_id:
                    job = self.ssh_tournament_runs.get(job_id)
                    if not job:
                        return self.error_response("Tournament not found", status=404)
                    return self.json_response(job.to_dict())

                return self.json_response({
                    jid: job.to_dict() for jid, job in self.ssh_tournament_runs.items()
                })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_ssh_tournament_cancel(self, request: web.Request) -> web.Response:
        """Cancel a running SSH tournament (best-effort)."""
        from scripts.p2p.types import NodeRole

        try:
            if self.role != NodeRole.LEADER:
                return self.error_response(
                    "Only the leader can cancel SSH tournaments",
                    status=403,
                    details={"leader_id": self.leader_id},
                )

            data = await request.json()
            job_id = data.get("job_id")
            if not job_id:
                return self.error_response("job_id is required", status=400)

            with self.ssh_tournament_lock:
                job = self.ssh_tournament_runs.get(job_id)
            if not job:
                return self.error_response("Tournament not found", status=404)

            if job.status != "running":
                return self.error_response(
                    f"Cannot cancel tournament in status: {job.status}",
                    status=400,
                    details={"success": False},
                )

            try:
                os.kill(job.pid, signal.SIGTERM)
            except Exception as e:
                return self.error_response(
                    f"Failed to signal process: {e}",
                    status=500,
                    details={"success": False},
                )

            with self.ssh_tournament_lock:
                job.status = "cancelled"
                job.completed_at = time.time()

            return self.json_response({"success": True, "job_id": job_id})
        except Exception as e:
            return self.error_response(str(e), status=500)

    async def _monitor_ssh_tournament_process(self, job_id: str, proc) -> None:
        """Monitor a tournament subprocess and update status."""
        try:
            return_code = await proc.wait()
            with self.ssh_tournament_lock:
                job = self.ssh_tournament_runs.get(job_id)
                if not job:
                    return
                job.return_code = return_code
                job.completed_at = time.time()
                if job.status != "cancelled":
                    job.status = "completed" if return_code == 0 else "failed"
                    if return_code != 0:
                        job.error_message = f"Process exited with code {return_code}"
        except Exception as e:
            with self.ssh_tournament_lock:
                job = self.ssh_tournament_runs.get(job_id)
                if job and job.status != "cancelled":
                    job.status = "failed"
                    job.completed_at = time.time()
                    job.error_message = str(e)
