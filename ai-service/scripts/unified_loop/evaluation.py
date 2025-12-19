"""Unified Loop Evaluation Services.

This module contains evaluation-related services for the unified AI loop:
- ModelPruningService: Automated model evaluation and culling

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .config import DataEvent, DataEventType, ModelPruningConfig

if TYPE_CHECKING:
    from unified_ai_loop import EventBus, UnifiedLoopState

from app.utils.paths import AI_SERVICE_ROOT


class ModelPruningService:
    """Automated model pruning service - evaluates and culls models when count exceeds threshold."""

    def __init__(
        self,
        config: ModelPruningConfig,
        state: "UnifiedLoopState",
        event_bus: "EventBus",
    ):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self._last_check = 0.0
        self._last_prune = 0.0
        self._pruning_in_progress = False
        self._prune_process = None

    def _count_models(self) -> dict:
        """Count NN and NNUE models in the models directory."""
        models_dir = AI_SERVICE_ROOT / "models"
        nn_count = len(list(models_dir.glob("*.pth")))
        nnue_dir = models_dir / "nnue"
        nnue_count = len(list(nnue_dir.glob("*.pt"))) if nnue_dir.exists() else 0
        return {"nn": nn_count, "nnue": nnue_count, "total": nn_count + nnue_count}

    async def should_run(self) -> bool:
        """Check if model pruning should run."""
        if not self.config.enabled:
            return False
        if self._pruning_in_progress:
            return False

        now = time.time()
        if now - self._last_check < self.config.check_interval_seconds:
            return False

        self._last_check = now
        counts = self._count_models()
        return counts["total"] >= self.config.threshold

    async def run_pruning_cycle(self) -> Optional[str]:
        """Run model evaluation and pruning cycle."""
        if self._pruning_in_progress:
            return None

        # Use fast ELO-based culling if configured
        if getattr(self.config, 'use_elo_based_culling', False):
            result = await self.run_elo_based_culling()
            return str(result) if result else None

        self._pruning_in_progress = True
        try:
            counts = self._count_models()
            print(f"[ModelPruning] Starting evaluation cycle: {counts['total']} models "
                  f"(threshold={self.config.threshold})")

            # Build command to run distributed evaluator
            evaluator_script = AI_SERVICE_ROOT / "scripts" / "distributed_model_evaluator.py"
            cmd = [
                sys.executable,
                str(evaluator_script),
                "--run",
                "--force",
                "--fast",  # Use fast mode (depth=2) for quicker evaluation
                "--threshold", str(self.config.threshold),
                "--workers", str(self.config.parallel_workers),
                "--games", str(self.config.games_per_baseline),
            ]

            if self.config.dry_run:
                cmd.append("--dry-run")

            print(f"[ModelPruning] Running: {' '.join(cmd)}")

            # Run evaluator as subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(AI_SERVICE_ROOT),
            )
            self._prune_process = process

            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.evaluation_timeout_seconds,
                )
                output = stdout.decode() if stdout else ""

                if process.returncode == 0:
                    print(f"[ModelPruning] Completed successfully")
                    self._last_prune = time.time()
                    # Publish event
                    await self.event_bus.publish(DataEvent(
                        event_type=DataEventType.MODEL_PROMOTED,  # Reuse promotion event
                        source="model_pruning",
                        payload={"status": "completed", "models_before": counts["total"]},
                    ))
                    return output
                else:
                    print(f"[ModelPruning] Failed with exit code {process.returncode}")
                    print(output[:500] if output else "No output")
                    return None

            except asyncio.TimeoutError:
                print(f"[ModelPruning] Timed out after {self.config.evaluation_timeout_seconds}s")
                process.kill()
                return None

        except Exception as e:
            print(f"[ModelPruning] Error: {e}")
            return None
        finally:
            self._pruning_in_progress = False
            self._prune_process = None

    async def run_elo_based_culling(self) -> Optional[Dict[str, Any]]:
        """Run fast ELO-based culling using existing ratings.

        This is faster than game-based evaluation as it uses existing ELO
        ratings from the unified_elo.db instead of running new games.
        """
        if self._pruning_in_progress:
            return None

        self._pruning_in_progress = True
        try:
            # Import the ELO-based culler
            sys.path.insert(0, str(AI_SERVICE_ROOT))
            from app.tournament.model_culling import ModelCullingController

            culler = ModelCullingController(
                elo_db_path=str(AI_SERVICE_ROOT / "data" / "unified_elo.db"),
                model_dir=str(AI_SERVICE_ROOT / "models"),
            )

            results = {"configs_processed": [], "total_archived": 0}

            # Cull all configs that need it
            for config_key in ["square8_2p", "square8_3p", "square8_4p",
                               "square19_2p", "square19_3p", "square19_4p",
                               "hexagonal_2p", "hexagonal_3p", "hexagonal_4p"]:
                if culler.needs_culling(config_key):
                    count_before = culler.count_models(config_key)
                    culler.cull_all_configs()  # Will only cull configs that need it
                    count_after = culler.count_models(config_key)
                    archived = count_before - count_after
                    if archived > 0:
                        results["configs_processed"].append(config_key)
                        results["total_archived"] += archived
                        print(f"[ModelPruning] ELO-based cull {config_key}: "
                              f"{count_before} -> {count_after} ({archived} archived)")

            if results["total_archived"] > 0:
                self._last_prune = time.time()
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.MODEL_PROMOTED,
                    source="elo_based_culling",
                    payload=results,
                ))

            return results

        except Exception as e:
            print(f"[ModelPruning] ELO-based culling error: {e}")
            return None
        finally:
            self._pruning_in_progress = False

    def get_status(self) -> dict:
        """Get current pruning service status."""
        counts = self._count_models()
        return {
            "enabled": self.config.enabled,
            "model_count": counts["total"],
            "threshold": self.config.threshold,
            "pruning_in_progress": self._pruning_in_progress,
            "last_check": self._last_check,
            "last_prune": self._last_prune,
            "needs_pruning": counts["total"] >= self.config.threshold,
        }
