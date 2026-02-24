"""Canonical Gate HTTP Handlers Mixin.

Provides HTTP endpoints for canonical selfplay gate management.
Handles health summaries, job tracking, log tailing, and gate job launching.

Usage:
    class P2POrchestrator(CanonicalGateHandlersMixin, ...):
        pass

Endpoints:
    GET /api/canonical/health - List canonical gate summary JSONs
    GET /api/canonical/jobs - List canonical gate jobs
    GET /api/canonical/jobs/{job_id} - Get job details
    GET /api/canonical/jobs/{job_id}/log - Tail job log
    GET /api/canonical/logs - List log files
    GET /api/canonical/logs/{log_name}/tail - Tail specific log file
    POST /api/canonical/generate - Start canonical selfplay+gate run
    POST /api/canonical/jobs/{job_id}/cancel - Cancel running job

Requires the implementing class to have:
    - ringrift_path: str (path to RingRift root)
    - node_id: str
    - canonical_gate_jobs: Dict[str, dict]
    - canonical_gate_jobs_lock: threading.Lock
    - _is_leader() method
    - _proxy_to_leader(request) method
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from app.core.async_context import safe_create_task
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_DELIVERY

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CanonicalGateHandlersMixin(BaseP2PHandler):
    """Mixin providing canonical gate HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - ringrift_path: str (path to RingRift root)
    - node_id: str
    - canonical_gate_jobs: Dict[str, dict]
    - canonical_gate_jobs_lock: threading.Lock
    - _is_leader() method
    - _proxy_to_leader(request) method
    """

    # Type hints for IDE support
    ringrift_path: str
    node_id: str
    canonical_gate_jobs: dict
    canonical_gate_jobs_lock: object  # threading.Lock

    def _canonical_slug_for_board(self, board_type: str) -> str:
        """Map board type to canonical slug used in database naming."""
        return {
            "square8": "square8",
            "square19": "square19",
            "hex8": "hex8",
            "hexagonal": "hex",
        }.get(board_type, board_type)

    def _canonical_gate_paths(self, board_type: str, num_players: int) -> tuple[Path, Path]:
        """Compute canonical DB + gate summary paths (leader-side conventions)."""
        slug = self._canonical_slug_for_board(board_type)
        suffix = "" if int(num_players) == 2 else f"_{int(num_players)}p"
        ai_root = Path(self.ringrift_path) / "ai-service"
        db_path = (ai_root / "data" / "games" / f"canonical_{slug}{suffix}.db").resolve()
        summary_path = (ai_root / "data" / "games" / f"db_health.canonical_{slug}{suffix}.json").resolve()
        return db_path, summary_path

    def _tail_text_file(self, path: Path, *, max_lines: int = 200, max_bytes: int = 256_000) -> str:
        """Best-effort tail of a potentially large log file."""
        try:
            if not path.exists():
                return ""
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                seek = max(0, size - int(max_bytes))
                f.seek(seek)
                data = f.read().decode("utf-8", errors="replace")
            lines = data.splitlines()
            return "\n".join(lines[-int(max_lines):])
        except Exception as e:  # noqa: BLE001
            return f"[tail_error] {e}"

    def _canonical_gate_log_dir(self) -> Path:
        """Return the directory for canonical gate log files."""
        return (Path(self.ringrift_path) / "ai-service" / "logs" / "canonical_gate").resolve()

    async def _monitor_canonical_gate_job(self, job_id: str, proc: asyncio.subprocess.Process, summary_path: Path) -> None:
        """Background task: wait for canonical gate to finish and record summary."""
        try:
            returncode = await proc.wait()
        except AttributeError:
            returncode = -1

        finished_at = time.time()
        gate_summary: dict[str, Any] | None = None
        try:
            if summary_path.exists():
                gate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, AttributeError):
            gate_summary = None

        with self.canonical_gate_jobs_lock:
            job = self.canonical_gate_jobs.get(job_id, {})
            prior_status = str(job.get("status") or "")
            if prior_status == "cancelling":
                status = "cancelled"
            else:
                status = "completed" if int(returncode) == 0 else "failed"
            job.update(
                {
                    "status": status,
                    "returncode": int(returncode),
                    "completed_at": finished_at,
                    "gate_summary": gate_summary,
                }
            )
            self.canonical_gate_jobs[job_id] = job

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_health(self, request: web.Request) -> web.Response:
        """GET /api/canonical/health - List canonical gate summary JSONs found on this node."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            ai_root = Path(self.ringrift_path) / "ai-service"
            games_dir = (ai_root / "data" / "games").resolve()
            summaries: list[dict[str, Any]] = []

            for path in sorted(games_dir.glob("db_health.canonical_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {"error": "failed_to_parse_json", "message": str(exc)}

                mtime = 0.0
                try:
                    mtime = float(path.stat().st_mtime)
                except (ValueError, AttributeError):
                    mtime = 0.0

                db_path_str = str(payload.get("db_path") or "")
                db_size_bytes = None
                if db_path_str:
                    try:
                        db_path = Path(db_path_str)
                        if not db_path.is_absolute():
                            db_path = (games_dir / db_path).resolve()
                        db_size_bytes = int(db_path.stat().st_size)
                    except (OSError, ValueError, AttributeError):
                        db_size_bytes = None

                summaries.append(
                    {
                        "path": str(path),
                        "modified_time": mtime,
                        "db_size_bytes": db_size_bytes,
                        "summary": payload,
                    }
                )

            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "summaries": summaries,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_jobs_list(self, request: web.Request) -> web.Response:
        """GET /api/canonical/jobs - List canonical gate jobs started from this node."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            with self.canonical_gate_jobs_lock:
                jobs = list(self.canonical_gate_jobs.values())
            jobs.sort(key=lambda j: float(j.get("started_at", 0.0) or 0.0), reverse=True)
            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "jobs": jobs[:100],
                    "timestamp": time.time(),
                }
            )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_job_get(self, request: web.Request) -> web.Response:
        """GET /api/canonical/jobs/{job_id} - Get details for a canonical gate job."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            job_id = (request.match_info.get("job_id") or "").strip()
            if not job_id:
                return web.json_response({"success": False, "error": "job_id is required"}, status=400)
            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id)
            if not job:
                return web.json_response({"success": False, "error": f"Job {job_id} not found"}, status=404)
            return web.json_response({"success": True, "job": job})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_job_log(self, request: web.Request) -> web.Response:
        """GET /api/canonical/jobs/{job_id}/log - Tail the log file for a canonical gate job."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            job_id = (request.match_info.get("job_id") or "").strip()
            if not job_id:
                return web.json_response({"success": False, "error": "job_id is required"}, status=400)
            tail_lines = int(request.query.get("tail", 200))
            tail_lines = max(10, min(tail_lines, 1000))

            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id)
            if not job:
                return web.json_response({"success": False, "error": f"Job {job_id} not found"}, status=404)

            log_path = Path(str(job.get("log_path") or ""))
            text = self._tail_text_file(log_path, max_lines=tail_lines)
            return web.json_response({"success": True, "job_id": job_id, "log_tail": text})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_logs_list(self, request: web.Request) -> web.Response:
        """GET /api/canonical/logs - List canonical gate log files on this node.

        Use ?local=1 to avoid proxying to the leader.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            logs_dir = self._canonical_gate_log_dir()
            entries: list[dict[str, Any]] = []
            if logs_dir.exists():
                paths = sorted(
                    logs_dir.glob("*.log"),
                    key=lambda p: float(p.stat().st_mtime),
                    reverse=True,
                )
                for path in paths[:200]:
                    try:
                        st = path.stat()
                        entries.append(
                            {
                                "name": path.name,
                                "path": str(path),
                                "size_bytes": int(st.st_size),
                                "modified_time": float(st.st_mtime),
                            }
                        )
                    except (ValueError, AttributeError):
                        continue

            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "log_dir": str(logs_dir),
                    "logs": entries,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_log_tail(self, request: web.Request) -> web.Response:
        """GET /api/canonical/logs/{log_name}/tail - Tail a specific canonical gate log file by name."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            log_name = (request.match_info.get("log_name") or "").strip()
            if not log_name:
                return web.json_response({"success": False, "error": "log_name is required"}, status=400)
            if any(token in log_name for token in ("..", "/", "\\")):
                return web.json_response({"success": False, "error": "Invalid log_name"}, status=400)

            tail_lines = int(request.query.get("tail", 200))
            tail_lines = max(10, min(tail_lines, 2000))

            logs_dir = self._canonical_gate_log_dir()
            log_path = (logs_dir / log_name).resolve()
            if log_path.parent != logs_dir:
                return web.json_response({"success": False, "error": "Invalid log_name"}, status=400)
            if not log_path.exists() or not log_path.is_file():
                return web.json_response({"success": False, "error": f"Log {log_name} not found"}, status=404)

            text = self._tail_text_file(log_path, max_lines=tail_lines)
            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "log_name": log_name,
                    "log_tail": text,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_generate(self, request: web.Request) -> web.Response:
        """POST /api/canonical/generate - Start a canonical selfplay+gate run (leader-only, dashboard-triggered).

        Request body:
            board_type: Board type (default: "square8")
            num_players: Number of players (default: 2)
            num_games: Number of games to generate (default: 0)
            difficulty_band: "light" or "canonical" (default: "light")
            reset_db: Reset database before running (default: false)
            hosts: Comma-separated list of hosts (optional)
            distributed_job_timeout_seconds: Timeout for distributed jobs (optional)
            distributed_fetch_timeout_seconds: Timeout for fetch operations (optional)

        Returns:
            success: True if job started
            job: Job details including job_id, status, config
        """
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {"success": False, "error": "Only leader can start canonical gate runs", "leader_id": getattr(self, 'leader_id', None)},
                status=403,
            )

        try:
            data = await request.json()
            board_type = str(data.get("board_type") or "square8")
            num_players = int(data.get("num_players") or 2)
            num_games = int(data.get("num_games") or 0)
            difficulty_band = str(data.get("difficulty_band") or "light")
            reset_db = bool(data.get("reset_db") or False)
            hosts = (str(data.get("hosts") or "").strip()) or None
            distributed_job_timeout_seconds = int(data.get("distributed_job_timeout_seconds") or 0)
            distributed_fetch_timeout_seconds = int(data.get("distributed_fetch_timeout_seconds") or 0)

            if board_type not in ("square8", "square19", "hex8", "hexagonal"):
                return web.json_response({"success": False, "error": f"Unsupported board_type: {board_type}"}, status=400)
            if num_players not in (2, 3, 4):
                return web.json_response({"success": False, "error": f"Unsupported num_players: {num_players}"}, status=400)
            if num_games < 0 or num_games > 250_000:
                return web.json_response({"success": False, "error": f"num_games out of range: {num_games}"}, status=400)
            if difficulty_band not in ("light", "canonical"):
                return web.json_response({"success": False, "error": f"Unsupported difficulty_band: {difficulty_band}"}, status=400)
            if hosts and any(c.isspace() for c in hosts):
                return web.json_response({"success": False, "error": "hosts must be comma-separated with no spaces"}, status=400)
            if distributed_job_timeout_seconds < 0 or distributed_job_timeout_seconds > 604_800:
                return web.json_response(
                    {"success": False, "error": f"distributed_job_timeout_seconds out of range: {distributed_job_timeout_seconds}"},
                    status=400,
                )
            if distributed_fetch_timeout_seconds < 0 or distributed_fetch_timeout_seconds > 86_400:
                return web.json_response(
                    {"success": False, "error": f"distributed_fetch_timeout_seconds out of range: {distributed_fetch_timeout_seconds}"},
                    status=400,
                )

            db_path, summary_path = self._canonical_gate_paths(board_type, num_players)

            job_id = f"canon_gate_{board_type}_{num_players}p_{int(time.time())}_{secrets.token_hex(4)}"
            ai_root = Path(self.ringrift_path) / "ai-service"
            log_dir = (ai_root / "logs" / "canonical_gate").resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = (log_dir / f"{job_id}.log").resolve()

            cmd = [
                sys.executable,
                "scripts/generate_canonical_selfplay.py",
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--difficulty-band", difficulty_band,
                "--db", str(db_path),
                "--summary", str(summary_path),
            ]
            if hosts:
                cmd.extend(["--hosts", hosts])
                if distributed_job_timeout_seconds > 0:
                    cmd.extend(
                        [
                            "--distributed-job-timeout-seconds",
                            str(distributed_job_timeout_seconds),
                        ]
                    )
                if distributed_fetch_timeout_seconds > 0:
                    cmd.extend(
                        [
                            "--distributed-fetch-timeout-seconds",
                            str(distributed_fetch_timeout_seconds),
                        ]
                    )
            if reset_db:
                cmd.append("--reset-db")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(ai_root)
            env.setdefault("RINGRIFT_JOB_ORIGIN", "dashboard")
            env.setdefault("PYTHONUNBUFFERED", "1")

            with log_path.open("a", encoding="utf-8") as log_handle:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(ai_root),
                    env=env,
                    stdout=log_handle,
                    stderr=log_handle,
                )

            job = {
                "job_id": job_id,
                "status": "running",
                "board_type": board_type,
                "num_players": num_players,
                "num_games": num_games,
                "difficulty_band": difficulty_band,
                "hosts": hosts,
                "reset_db": reset_db,
                "distributed_job_timeout_seconds": distributed_job_timeout_seconds,
                "distributed_fetch_timeout_seconds": distributed_fetch_timeout_seconds,
                "db_path": str(db_path),
                "summary_path": str(summary_path),
                "log_path": str(log_path),
                "pid": int(proc.pid),
                "started_at": time.time(),
            }

            with self.canonical_gate_jobs_lock:
                self.canonical_gate_jobs[job_id] = job

            safe_create_task(self._monitor_canonical_gate_job(job_id, proc, summary_path), name="canonical-gate-monitor")

            return web.json_response({"success": True, "job": job})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_api_canonical_job_cancel(self, request: web.Request) -> web.Response:
        """POST /api/canonical/jobs/{job_id}/cancel - Cancel a running canonical gate job."""
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {"success": False, "error": "Only leader can cancel canonical gate runs", "leader_id": getattr(self, 'leader_id', None)},
                status=403,
            )

        try:
            job_id = (request.match_info.get("job_id") or "").strip()
            if not job_id:
                return web.json_response({"success": False, "error": "job_id is required"}, status=400)

            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id)
            if not job:
                return web.json_response({"success": False, "error": f"Job {job_id} not found"}, status=404)

            pid = int(job.get("pid") or 0)
            if pid <= 0:
                return web.json_response({"success": False, "error": "No pid recorded for job"}, status=400)

            try:
                os.kill(pid, signal.SIGTERM)
            except Exception as exc:
                return web.json_response({"success": False, "error": f"Failed to signal pid {pid}: {exc}"}, status=500)

            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id, job)
                job["status"] = "cancelling"
                job["cancel_requested_at"] = time.time()
                self.canonical_gate_jobs[job_id] = job

            return web.json_response({"success": True, "message": f"Cancel signaled for {job_id}", "job": job})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)
