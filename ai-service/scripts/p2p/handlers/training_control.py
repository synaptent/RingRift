"""Training Control HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 3a

This mixin provides HTTP handlers for training job management,
training progress monitoring, and training trigger RPC endpoints.

Must be mixed into a class that provides:
- self.role: NodeRole
- self.node_id: str
- self.training_lock: threading.Lock
- self.training_jobs: dict
- self.training_thresholds: TrainingThresholds
- self.training_coordinator: TrainingCoordinator
- self.notifier: NotificationManager
- self.improvement_cycle_manager: Optional[ImprovementCycleManager]
- self._get_ai_service_path() -> str
- self.get_data_directory() -> Path
- self._save_state()
- self._monitor_training_process(job_id, proc, output_path)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from app.core.async_context import safe_create_task
from scripts.p2p.db_helpers import p2p_db_connection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import NodeRole
try:
    from scripts.p2p.node_types import NodeRole
except ImportError:
    # Fallback enum if not available
    class NodeRole:  # type: ignore[no-redef]
        LEADER = "leader"


class TrainingControlHandlersMixin:
    """Mixin providing training control HTTP handlers.

    Endpoints:
    - POST /training/start - Start a training job
    - GET /training/status - Get training job status
    - GET /training/progress - Get training progress with Elo deltas
    - POST /training/update - Update training job status from worker
    - POST /training/trigger - Trigger training for a config
    - GET /training/trigger-decision/{config_key} - Get training trigger decision
    - GET /training/trigger-configs - Get list of tracked training configs
    - POST /nnue/start - Start NNUE training subprocess
    """

    async def handle_training_start(self, request: web.Request) -> web.Response:
        """Handle request to start a training job (from external or leader)."""
        try:
            data = await request.json()
            job_type = data.get("job_type", "nnue")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "success": False,
                    "error": "Only leader can dispatch training jobs"
                })

            job_config = {
                "job_type": job_type,
                "board_type": board_type,
                "num_players": num_players,
                "config_key": f"{board_type}_{num_players}p",
                "total_games": data.get("total_games", 0),
            }

            job = await self.training_coordinator.dispatch_training_job(job_config)
            if job:
                return web.json_response({
                    "success": True,
                    "job_id": job.job_id,
                    "worker": job.worker_node,
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "No suitable worker available"
                })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})

    async def handle_training_status(self, request: web.Request) -> web.Response:
        """Return status of all training jobs."""
        with self.training_lock:
            jobs = [job.to_dict() for job in self.training_jobs.values()]

        return web.json_response({
            "success": True,
            "jobs": jobs,
            "thresholds": self.training_thresholds.to_dict(),
        })

    async def handle_training_progress(self, request: web.Request) -> web.Response:
        """Return training progress showing before/after Elo deltas.

        Jan 6, 2026: P4 - Training progress visibility.
        Shows before_elo and final_elo from training_history to demonstrate
        that training runs are producing model improvements.

        Query params:
            days: Number of days to look back (default: 7)
            config: Optional config filter (e.g., "hex8_2p")
        """
        try:
            # Parse query params
            days = float(request.query.get("days", "7"))
            config_key = request.query.get("config")

            # Use the training coordinator's database
            db_path = Path("data/training_coordination.db")
            if not db_path.exists():
                return web.json_response({
                    "success": False,
                    "error": f"Training database not found: {db_path}",
                })

            def _query_training_history():
                with p2p_db_connection(db_path, row_factory=True) as conn:

                    # Check if training_history table exists
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='training_history'"
                    )
                    if not cursor.fetchone():
                        return None, "training_history table not found"

                since = time.time() - (days * 86400)

                # Build query
                query = """
                    SELECT
                        job_id,
                        board_type,
                        num_players,
                        node_name,
                        started_at,
                        completed_at,
                        status,
                        final_val_loss,
                        final_elo,
                        before_elo,
                        epochs_completed
                    FROM training_history
                    WHERE completed_at > ?
                """
                params = [since]

                if config_key:
                    # Parse config_key (e.g., "hex8_2p" -> "hex8", 2)
                    parts = config_key.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].endswith("p"):
                        board_type = parts[0]
                        num_players = int(parts[1][:-1])
                        query += " AND board_type = ? AND num_players = ?"
                        params.extend([board_type, num_players])

                query += " ORDER BY completed_at DESC"

                cursor = conn.execute(query, params)
                rows = [dict(row) for row in cursor.fetchall()]
                return rows, None

            # Run blocking SQLite in thread
            rows, error = await asyncio.to_thread(_query_training_history)

            if error:
                return web.json_response({
                    "success": False,
                    "error": error,
                    "message": "Run a training job first to populate this data.",
                })

            if not rows:
                return web.json_response({
                    "success": True,
                    "runs": [],
                    "summary": {
                        "total_runs": 0,
                        "completed_count": 0,
                        "avg_elo_delta": None,
                        "improved_count": 0,
                        "improvement_rate": None,
                    },
                })

            # Compute stats
            total_delta = 0.0
            completed_count = 0
            improved_count = 0

            training_runs = []
            for row in rows:
                config = f"{row['board_type']}_{row['num_players']}p"
                before = row.get("before_elo") or 0.0
                after = row.get("final_elo") or 0.0
                delta = after - before if before > 0 and after > 0 else None

                training_runs.append({
                    "job_id": row["job_id"],
                    "config": config,
                    "status": row["status"] or "unknown",
                    "before_elo": before if before > 0 else None,
                    "final_elo": after if after > 0 else None,
                    "elo_delta": delta,
                    "epochs_completed": row.get("epochs_completed") or 0,
                    "node": row.get("node_name"),
                    "completed_at": row.get("completed_at"),
                })

                if row["status"] == "completed":
                    completed_count += 1
                    if delta is not None:
                        total_delta += delta
                        if delta > 0:
                            improved_count += 1

            avg_delta = total_delta / completed_count if completed_count > 0 else None
            improvement_rate = improved_count / completed_count if completed_count > 0 else None

            return web.json_response({
                "success": True,
                "runs": training_runs,
                "summary": {
                    "total_runs": len(rows),
                    "completed_count": completed_count,
                    "avg_elo_delta": round(avg_delta, 1) if avg_delta else None,
                    "improved_count": improved_count,
                    "improvement_rate": round(improvement_rate * 100, 1) if improvement_rate else None,
                },
                "is_improving": avg_delta is not None and avg_delta > 0,
                "days": days,
                "config_filter": config_key,
            })

        except (ValueError, TypeError) as e:
            return web.json_response({"success": False, "error": f"Invalid parameters: {e}"})
        except sqlite3.Error as e:
            return web.json_response({"success": False, "error": f"Database error: {e}"})

    async def handle_training_update(self, request: web.Request) -> web.Response:
        """Handle training progress/completion update from worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")

            should_trigger_eval = False

            with self.training_lock:
                job = self.training_jobs.get(job_id)
                if not job:
                    return web.json_response({
                        "success": False,
                        "error": f"Job {job_id} not found"
                    })

                # Update job status
                if data.get("status"):
                    job.status = data["status"]
                if data.get("completed"):
                    job.status = "completed"
                    job.completed_at = time.time()
                if data.get("output_model_path"):
                    job.output_model_path = data["output_model_path"]
                if data.get("final_loss"):
                    job.final_loss = data["final_loss"]
                if data.get("final_accuracy"):
                    job.final_accuracy = data["final_accuracy"]
                if data.get("error"):
                    job.status = "failed"
                    job.error_message = data["error"]
                    # ALERTING: Notify on training failure
                    safe_create_task(self.notifier.send(
                        title="Training Job Failed",
                        message=f"Training job {job.job_id} failed: {data['error'][:100]}",
                        level="error",
                        fields={
                            "Job ID": job.job_id,
                            "Type": job.job_type,
                            "Config": f"{job.board_type}_{job.num_players}p",
                            "Worker": job.worker_node or "unknown",
                            "Error": data["error"][:200],
                            "Checkpoint": job.checkpoint_path or "none",
                        },
                        node_id=self.node_id,
                    ), name="training-notify-failure")

                # TRAINING CHECKPOINTING: Track checkpoint progress
                if data.get("checkpoint_path"):
                    job.checkpoint_path = data["checkpoint_path"]
                    job.checkpoint_updated_at = time.time()
                if data.get("checkpoint_epoch"):
                    job.checkpoint_epoch = int(data["checkpoint_epoch"])
                if data.get("checkpoint_loss"):
                    job.checkpoint_loss = float(data["checkpoint_loss"])

                # Check if we should trigger evaluation after training
                should_trigger_eval = (
                    data.get("completed") and
                    job.output_model_path and
                    self.improvement_cycle_manager
                )

            self._save_state()

            # Auto-trigger tournament evaluation when training completes
            # Delegate to TrainingCoordinator (Phase 2B refactoring, Dec 2025)
            if should_trigger_eval:
                safe_create_task(self.training_coordinator.handle_training_job_completion(job), name="training-handle-completion")

            return web.json_response({"success": True})

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})

    async def handle_training_trigger(self, request: web.Request) -> web.Response:
        """Handle request to trigger training for a config (December 30, 2025).

        Endpoint: POST /training/trigger

        This provides programmatic control over training triggers, allowing
        operators to bypass the daemon's automatic trigger logic when needed.

        Request body:
            {
                "config_key": "hex8_2p",
                "priority": "normal",  // "low", "normal", "high", "urgent"
                "force": false,        // Skip freshness/cooldown checks
                "data_path": null      // Optional: override NPZ path
            }

        Returns:
            Success: job_id, worker_node, trigger_reason, data_info
            Failure: success=false with reason
        """
        try:
            data = await request.json()
            config_key = data.get("config_key")
            priority = data.get("priority", "normal")
            force = data.get("force", False)
            data_path = data.get("data_path")

            # Validate config_key format
            if not config_key or "_" not in config_key:
                return web.json_response({
                    "success": False,
                    "error": "Invalid config_key format (expected: board_Np, e.g., hex8_2p)"
                }, status=400)

            # Only leader can trigger training
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "success": False,
                    "error": "Only leader can trigger training"
                }, status=403)

            # Get training trigger daemon for decision checking
            try:
                from app.coordination.training_trigger_daemon import (
                    get_training_trigger_daemon,
                )
                daemon = get_training_trigger_daemon()
            except ImportError:
                daemon = None

            trigger_reason = "force=true" if force else "conditions met"
            decision_info = None

            # If not forcing, check conditions via daemon
            if not force and daemon:
                decision = await daemon.get_training_decision(config_key)
                decision_info = decision.to_dict()

                if not decision.can_trigger:
                    return web.json_response({
                        "success": False,
                        "triggered": False,
                        "reason": decision.reason,
                        "decision": decision_info,
                    })

                trigger_reason = decision.reason

            # Parse config_key to get board_type and num_players
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2:
                return web.json_response({
                    "success": False,
                    "error": f"Cannot parse config_key: {config_key}"
                }, status=400)

            board_type = parts[0]
            try:
                num_players = int(parts[1].rstrip("p"))
            except ValueError:
                return web.json_response({
                    "success": False,
                    "error": f"Invalid player count in config_key: {config_key}"
                }, status=400)

            # Dispatch training job
            job_config = {
                "job_type": "nnue",
                "board_type": board_type,
                "num_players": num_players,
                "config_key": config_key,
                "priority": priority,
                "data_path": data_path,
                "triggered_by": "rpc",
            }

            job = await self.training_coordinator.dispatch_training_job(job_config)

            if job:
                return web.json_response({
                    "success": True,
                    "triggered": True,
                    "job_id": job.job_id,
                    "worker_node": job.worker_node,
                    "trigger_reason": trigger_reason,
                    "decision": decision_info,
                })
            else:
                return web.json_response({
                    "success": False,
                    "triggered": False,
                    "error": "No suitable worker available",
                    "decision": decision_info,
                })

        except Exception as e:  # noqa: BLE001
            logger.exception(f"[P2POrchestrator] Training trigger failed: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def handle_training_trigger_decision(self, request: web.Request) -> web.Response:
        """Get training trigger decision for a config (December 30, 2025).

        Endpoint: GET /training/trigger-decision/{config_key}

        Returns full condition details explaining why training would or
        wouldn't trigger for the specified config.
        """
        try:
            config_key = request.match_info.get("config_key", "")

            if not config_key or "_" not in config_key:
                return web.json_response({
                    "success": False,
                    "error": "Invalid config_key format"
                }, status=400)

            # Get training trigger daemon
            try:
                from app.coordination.training_trigger_daemon import (
                    get_training_trigger_daemon,
                )
                daemon = get_training_trigger_daemon()
            except ImportError:
                return web.json_response({
                    "success": False,
                    "error": "TrainingTriggerDaemon not available"
                }, status=503)

            decision = await daemon.get_training_decision(config_key)

            return web.json_response({
                "success": True,
                **decision.to_dict(),
            })

        except Exception as e:  # noqa: BLE001
            logger.exception(f"[P2POrchestrator] Training trigger decision failed: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def handle_training_trigger_configs(self, request: web.Request) -> web.Response:
        """Get list of all tracked training configs (December 30, 2025).

        Endpoint: GET /training/trigger-configs

        Returns list of all config keys that the training trigger daemon is tracking.
        """
        try:
            from app.coordination.training_trigger_daemon import get_training_trigger_daemon
            daemon = get_training_trigger_daemon()
            configs = daemon.get_tracked_configs()

            return web.json_response({
                "success": True,
                "configs": configs,
                "count": len(configs),
            })

        except ImportError:
            return web.json_response({
                "success": False,
                "error": "TrainingTriggerDaemon not available"
            }, status=503)
        except Exception as e:  # noqa: BLE001
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def handle_nnue_start(self, request: web.Request) -> web.Response:
        """Handle NNUE training start request (worker endpoint)."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            epochs = data.get("epochs", 100)
            batch_size = data.get("batch_size", 4096)
            learning_rate = data.get("learning_rate", None)

            # Start NNUE training subprocess
            # December 29, 2025: Use canonical model paths for consistent naming
            output_path = os.path.join(
                self._get_ai_service_path(), "models",
                f"canonical_{board_type}_{num_players}p.pth"
            )

            # Collect local selfplay databases. The NNUE trainer requires at
            # least one DB (it can replay moves when snapshots are absent).
            data_dir = self.get_data_directory()
            board_tokens = [str(board_type).lower()]
            if "hex" in board_tokens[0]:
                board_tokens = ["hexagonal", "hex"]
            players_token = f"_{int(num_players)}p"

            candidate_dbs: list[Path] = []
            for pattern in ("selfplay/**/*.db", "games/**/*.db"):
                for db_path in data_dir.glob(pattern):
                    if not db_path.is_file():
                        continue
                    path_lower = str(db_path).lower()
                    if players_token not in path_lower:
                        continue
                    if not any(tok in path_lower for tok in board_tokens):
                        continue
                    candidate_dbs.append(db_path)

            # Fallback: if naming conventions differ, use any selfplay DBs.
            if not candidate_dbs:
                candidate_dbs = [p for p in data_dir.glob("selfplay/**/*.db") if p.is_file()]

            # De-dupe + prefer newest DBs (avoid overlong argv on large clusters).
            unique_dbs = list({p.resolve() for p in candidate_dbs})
            unique_dbs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
            max_dbs = 64
            unique_dbs = unique_dbs[:max_dbs]

            if not unique_dbs:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"No selfplay DBs found under {data_dir} for {board_type} {num_players}p",
                    },
                    status=400,
                )

            cmd = [
                sys.executable, "-m", "scripts.train_nnue",
                "--db", *[str(p) for p in unique_dbs],
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--save-path", output_path,
                # Jan 2026: Skip canonical registry check - cluster nodes may not have registry file
                "--allow-noncanonical",
                # Phase 1: Core Training Optimizations
                "--spectral-norm",  # Gradient stability
                "--cyclic-lr", "--cyclic-lr-period", "5",  # Cyclic LR
                "--mixed-precision", "--amp-dtype", "bfloat16",  # BF16 speed
                "--warmup-epochs", "3",  # LR warmup
                # Phase 2: Advanced Training
                "--value-whitening",  # Value head stability
                "--ema",  # Exponential Moving Average
                "--stochastic-depth", "--stochastic-depth-prob", "0.1",
                "--adaptive-warmup",  # Dataset-aware warmup
                "--hard-example-mining", "--hard-example-top-k", "0.3",
                # Phase 2: Optimizer Enhancements
                "--lookahead", "--lookahead-k", "5", "--lookahead-alpha", "0.5",
                "--adaptive-clip",  # Adaptive gradient clipping
                "--board-nas",  # Board-specific NAS
                "--online-bootstrap", "--bootstrap-temperature", "1.5",
                # Phase 2: Data Pipeline
                "--prefetch-gpu",  # GPU prefetching
                "--difficulty-curriculum",  # Curriculum learning
                "--quantized-eval",  # Fast validation
                # Phase 3: Advanced Learning
                "--grokking-detection",  # Detect delayed generalization
                "--policy-label-smoothing", "0.05",  # Prevent overconfidence
                "--sampling-weights", "victory_type",  # Balanced sampling
                # Phase 4: Training Stability (optional, enabled for production)
                "--adaptive-accumulation",  # Dynamic gradient accumulation
                # Phase 5: Production Optimization (selective)
                "--dynamic-loss-scaling",  # Adaptive FP16 loss scaling
                # Phase 6: Auto-Promotion (Jan 2026)
                "--auto-promote",  # Run gauntlet and promote if criteria met
                "--auto-promote-games", "30",  # Quick evaluation (30 games per opponent)
            ]
            # Add hex symmetry augmentation for hex boards (12x effective data)
            if board_type in ('hex8', 'hexagonal', 'hex'):
                cmd.append("--augment-hex-symmetry")
            # Add profiling for debug jobs
            if os.environ.get("RINGRIFT_PROFILE_TRAINING"):
                cmd.extend(["--profile", "--profile-dir", str(Path(output_path).parent / "profile")])
            if learning_rate is not None:
                cmd.extend(["--learning-rate", str(learning_rate)])

            env = os.environ.copy()
            env["PYTHONPATH"] = self._get_ai_service_path()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._get_ai_service_path(),
            )

            logger.info(f"Started NNUE training subprocess (PID {proc.pid}) for job {job_id}")

            # Don't wait - let it run in background
            safe_create_task(self._monitor_training_process(job_id, proc, output_path), name="training-monitor-nnue")

            return web.json_response({
                "success": True,
                "pid": proc.pid,
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})
