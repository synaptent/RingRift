"""P2P Pipeline HTTP handlers.

January 2026: Extracted from p2p_orchestrator.py to reduce file size (~244 LOC).

Endpoints:
- POST /pipeline/start - Start a canonical pipeline phase

The handler accesses orchestrator state via `self.*` since it's designed
as a mixin that gets inherited by P2POrchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web, ClientTimeout

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import NodeInfo

logger = logging.getLogger(__name__)


def get_client_session(timeout: ClientTimeout | None = None) -> aiohttp.ClientSession:
    """Create a client session with optional timeout."""
    return aiohttp.ClientSession(timeout=timeout or ClientTimeout(total=30))


class PipelineHandlersMixin:
    """Mixin providing pipeline phase HTTP handlers.

    Must be mixed into a class that provides:
    - self._is_leader()
    - self._proxy_to_leader()
    - self.leader_id
    - self.peers_lock, self.peers
    - self.self_info
    - self.node_id
    - self._get_ai_service_path()
    - self._urls_for_peer()
    - self._get_auth_headers()
    - self._enqueue_relay_command_for_peer()
    - self._pipeline_status
    """

    async def handle_pipeline_start(self, request: web.Request) -> web.Response:
        """POST /pipeline/start - Start a canonical pipeline phase.

        January 2026: Moved from p2p_orchestrator.py to PipelineHandlersMixin.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)
            if not self._is_leader():
                return web.json_response({
                    "success": False,
                    "error": "Only leader can start pipeline phases",
                    "leader_id": self.leader_id
                }, status=403)

            data = await request.json()
            phase = data.get("phase")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            if phase == "canonical_selfplay":
                result = await self._start_canonical_selfplay_pipeline(
                    board_type,
                    num_players,
                    data.get("games_per_node", 500),
                    data.get("seed", 0),
                    include_gpu_nodes=bool(data.get("include_gpu_nodes", False)),
                )
            elif phase == "parity_validation":
                result = await self._start_parity_validation_pipeline(
                    board_type, num_players, data.get("db_paths"))
            elif phase == "npz_export":
                result = await self._start_npz_export_pipeline(
                    board_type, num_players, data.get("output_dir", "data/training"))
            else:
                return web.json_response({
                    "success": False,
                    "error": f"Unknown phase: {phase}. Supported: canonical_selfplay, parity_validation, npz_export"
                }, status=400)

            return web.json_response(result)
        except Exception as e:  # noqa: BLE001
            logger.info(f"Pipeline start error: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def _start_canonical_selfplay_pipeline(
        self,
        board_type: str,
        num_players: int,
        games_per_node: int,
        seed: int,
        include_gpu_nodes: bool = False,
    ) -> dict[str, Any]:
        """Start canonical selfplay on healthy nodes in the cluster.

        Canonical selfplay is CPU-bound. By default, prefer CPU-only nodes so GPU
        machines remain available for GPU-utilizing tasks (training/hybrid selfplay).

        January 2026: Moved from p2p_orchestrator.py to PipelineHandlersMixin.
        """
        job_id = f"pipeline-selfplay-{int(time.time())}"
        healthy_nodes: list[tuple[str, "NodeInfo"]] = []

        with self.peers_lock:
            for peer_id, peer in self.peers.items():
                if peer.is_alive() and peer.is_healthy():
                    healthy_nodes.append((peer_id, peer))

        if self.self_info.is_healthy():
            healthy_nodes.append((self.node_id, self.self_info))

        if not include_gpu_nodes:
            cpu_nodes = [(nid, n) for nid, n in healthy_nodes if n.is_cpu_only_node()]
            if cpu_nodes:
                healthy_nodes = cpu_nodes

        # Load-balance: least-loaded nodes first.
        healthy_nodes.sort(key=lambda pair: pair[1].get_load_score())

        if not healthy_nodes:
            return {"success": False, "error": "No healthy nodes available"}

        logger.info(f"Starting canonical selfplay pipeline: {len(healthy_nodes)} nodes, {games_per_node} games/node")
        dispatched = 0

        for i, (node_id, node) in enumerate(healthy_nodes):
            node_seed = seed + i * 10000 + hash(node_id) % 10000

            if node_id == self.node_id:
                safe_create_task(self._run_local_canonical_selfplay(
                    f"{job_id}-{node_id}", board_type, num_players, games_per_node, node_seed), name="pipeline-local-selfplay")
                dispatched += 1
            else:
                try:
                    if getattr(node, "nat_blocked", False):
                        payload = {
                            "job_id": f"{job_id}-{node_id}",
                            "board_type": board_type,
                            "num_players": num_players,
                            "num_games": games_per_node,
                            "seed": node_seed,
                        }
                        cmd_id = await self._enqueue_relay_command_for_peer(node, "canonical_selfplay", payload)
                        if cmd_id:
                            dispatched += 1
                        else:
                            logger.info(f"Relay queue full; skipping canonical selfplay enqueue for {node_id}")
                    else:
                        payload = {
                            "job_id": f"{job_id}-{node_id}",
                            "board_type": board_type,
                            "num_players": num_players,
                            "num_games": games_per_node,
                            "seed": node_seed,
                        }
                        async with get_client_session(ClientTimeout(total=30)) as session:
                            for url in self._urls_for_peer(node, "/pipeline/selfplay_worker"):
                                try:
                                    async with session.post(url, json=payload, headers=self._get_auth_headers()) as resp:
                                        if resp.status == 200:
                                            dispatched += 1
                                            break
                                except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                                    continue
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to dispatch selfplay to {node_id}: {e}")

        self._pipeline_status = {
            "job_id": job_id,
            "phase": "canonical_selfplay",
            "status": "running",
            "dispatched_count": dispatched,
            "total_nodes": len(healthy_nodes),
            "board_type": board_type,
            "num_players": num_players,
            "games_per_node": games_per_node,
            "started_at": time.time(),
        }
        return {"success": True, "job_id": job_id, "dispatched_count": dispatched, "total_nodes": len(healthy_nodes)}

    async def _run_local_canonical_selfplay(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        num_games: int,
        seed: int,
    ) -> None:
        """Run canonical selfplay locally.

        January 2026: Moved from p2p_orchestrator.py to PipelineHandlersMixin.
        """
        try:
            db_file = os.path.join(
                self._get_ai_service_path(), "data", "games",
                f"canonical_{board_type}_{num_players}p_{self.node_id}.db"
            )
            log_file = os.path.join(
                self._get_ai_service_path(), "logs", "selfplay",
                f"canonical_{job_id}.jsonl"
            )
            os.makedirs(os.path.dirname(db_file), exist_ok=True)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            cmd = [
                sys.executable,
                os.path.join(self._get_ai_service_path(), "scripts", "run_self_play_soak.py"),
                "--num-games", str(num_games),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--max-moves", "10000",  # LEARNED LESSONS - Avoid draws due to move limit
                "--difficulty-band", "light",
                "--seed", str(seed),
                "--log-jsonl", log_file,
                "--record-db", db_file,
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = self._get_ai_service_path()
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            logger.info(f"Starting canonical selfplay job {job_id}: {num_games} games -> {db_file}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            _stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"Canonical selfplay job {job_id} completed successfully")
            else:
                logger.info(f"Canonical selfplay job {job_id} failed: {stderr.decode()[:500]}")
        except Exception as e:  # noqa: BLE001
            logger.info(f"Canonical selfplay job {job_id} error: {e}")

    async def _start_parity_validation_pipeline(
        self,
        board_type: str,
        num_players: int,
        db_paths: list[str] | None,
    ) -> dict[str, Any]:
        """Start parity validation on the leader node.

        January 2026: Moved from p2p_orchestrator.py to PipelineHandlersMixin.
        """
        job_id = f"pipeline-parity-{int(time.time())}"
        safe_create_task(self._run_parity_validation(job_id, board_type, num_players, db_paths), name="pipeline-parity-validation")
        self._pipeline_status = {
            "job_id": job_id,
            "phase": "parity_validation",
            "status": "running",
            "board_type": board_type,
            "num_players": num_players,
            "started_at": time.time(),
        }
        return {"success": True, "job_id": job_id, "message": "Parity validation started"}

    async def _run_parity_validation(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        db_paths: list[str] | None,
    ) -> None:
        """Run parity validation.

        January 2026: Moved from p2p_orchestrator.py to PipelineHandlersMixin.
        """
        try:
            if not db_paths:
                import glob
                db_paths = glob.glob(os.path.join(
                    self._get_ai_service_path(), "data", "games",
                    f"canonical_{board_type}_{num_players}p_*.db"
                ))

            if not db_paths:
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = "No databases found"
                return

            output_json = os.path.join(
                self._get_ai_service_path(), "data",
                f"parity_validation_{job_id}.json"
            )
            cmd = [
                sys.executable,
                os.path.join(self._get_ai_service_path(), "scripts", "run_parity_validation.py"),
                "--databases", *db_paths,
                "--mode", "canonical",
                "--output-json", output_json,
                "--progress-every", "100",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = self._get_ai_service_path()
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            logger.info(f"Starting parity validation job {job_id}: {len(db_paths)} databases")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            _stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"Parity validation job {job_id} completed successfully")
                self._pipeline_status["status"] = "completed"
                if os.path.exists(output_json):
                    with open(output_json) as f:
                        self._pipeline_status["results"] = json.load(f)
            else:
                logger.info(f"Parity validation job {job_id} failed: {stderr.decode()[:500]}")
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = stderr.decode()[:500]
        except Exception as e:  # noqa: BLE001
            logger.info(f"Parity validation job {job_id} error: {e}")
            self._pipeline_status["status"] = "failed"
            self._pipeline_status["error"] = str(e)

    async def _start_npz_export_pipeline(
        self,
        board_type: str,
        num_players: int,
        output_dir: str,
    ) -> dict[str, Any]:
        """Start NPZ export on the leader node.

        January 2026: Moved from p2p_orchestrator.py to PipelineHandlersMixin.
        """
        job_id = f"pipeline-npz-{int(time.time())}"
        safe_create_task(self._run_npz_export(job_id, board_type, num_players, output_dir), name="pipeline-npz-export")
        self._pipeline_status = {
            "job_id": job_id,
            "phase": "npz_export",
            "status": "running",
            "board_type": board_type,
            "num_players": num_players,
            "output_dir": output_dir,
            "started_at": time.time(),
        }
        return {"success": True, "job_id": job_id, "message": "NPZ export started"}

    async def _run_npz_export(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        output_dir: str,
    ) -> None:
        """Run NPZ export.

        January 2026: Moved from p2p_orchestrator.py to PipelineHandlersMixin.
        """
        try:
            import glob
            db_paths = glob.glob(os.path.join(
                self._get_ai_service_path(), "data", "games",
                f"canonical_{board_type}_{num_players}p_*.db"
            ))

            if not db_paths:
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = "No databases found"
                return

            full_output_dir = os.path.join(self._get_ai_service_path(), output_dir)
            os.makedirs(full_output_dir, exist_ok=True)
            output_file = os.path.join(
                full_output_dir,
                f"canonical_{board_type}_{num_players}p_{job_id}.npz"
            )

            cmd = [
                sys.executable,
                os.path.join(self._get_ai_service_path(), "scripts", "export_replay_dataset.py"),
                "--databases", *db_paths,
                "--output", output_file,
                "--board-type", board_type,
                "--num-players", str(num_players),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = self._get_ai_service_path()

            logger.info(f"Starting NPZ export job {job_id}: {len(db_paths)} databases -> {output_file}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            _stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"NPZ export job {job_id} completed successfully")
                self._pipeline_status["status"] = "completed"
                self._pipeline_status["output_file"] = output_file
            else:
                logger.info(f"NPZ export job {job_id} failed: {stderr.decode()[:500]}")
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = stderr.decode()[:500]
        except Exception as e:  # noqa: BLE001
            logger.info(f"NPZ export job {job_id} error: {e}")
            self._pipeline_status["status"] = "failed"
            self._pipeline_status["error"] = str(e)
