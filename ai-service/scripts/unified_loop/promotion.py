"""Unified Loop Promotion Services.

This module contains model promotion services for the unified AI loop:
- ModelPromoter: Automatic model promotion based on Elo with holdout validation

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .config import DataEvent, DataEventType, PromotionConfig

if TYPE_CHECKING:
    from unified_ai_loop import EventBus, UnifiedLoopState

# Path constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]

# Optional ELO service import
try:
    from app.training.elo_service import get_elo_service
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None

# Optional Prometheus metrics - avoid duplicate registration
try:
    from prometheus_client import Counter, Gauge, REGISTRY
    HAS_PROMETHEUS = True

    def _get_or_create_gauge(name, desc, labels):
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return Gauge(name, desc, labels)

    def _get_or_create_counter(name, desc, labels):
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return Counter(name, desc, labels)

    HOLDOUT_LOSS = _get_or_create_gauge('ringrift_holdout_loss', 'Holdout evaluation loss', ['config'])
    HOLDOUT_OVERFIT_GAP = _get_or_create_gauge('ringrift_holdout_overfit_gap', 'Gap between train and holdout loss', ['config'])
    HOLDOUT_EVALUATIONS = _get_or_create_counter('ringrift_holdout_evaluations_total', 'Total holdout evaluations', ['config', 'result'])
    PROMOTION_BLOCKED_OVERFIT = _get_or_create_counter('ringrift_promotion_blocked_overfit_total', 'Promotions blocked due to overfitting', ['config'])
except ImportError:
    HAS_PROMETHEUS = False
    HOLDOUT_LOSS = None
    HOLDOUT_OVERFIT_GAP = None
    HOLDOUT_EVALUATIONS = None
    PROMOTION_BLOCKED_OVERFIT = None

# Import holdout validation for overfitting detection during promotion
try:
    from scripts.holdout_validation import (
        evaluate_model_on_holdout,
        EvaluationResult,
        OVERFIT_THRESHOLD,
    )
    HAS_HOLDOUT_VALIDATION = True
except ImportError:
    HAS_HOLDOUT_VALIDATION = False
    evaluate_model_on_holdout = None
    EvaluationResult = None
    OVERFIT_THRESHOLD = 0.15  # Default fallback


class ModelPromoter:
    """Handles automatic model promotion based on Elo."""

    def __init__(self, config: PromotionConfig, state: "UnifiedLoopState", event_bus: "EventBus"):
        self.config = config
        self.state = state
        self.event_bus = event_bus

    async def check_promotion_candidates(self) -> List[Dict[str, Any]]:
        """Check for models that should be promoted."""
        if not self.config.auto_promote:
            return []

        candidates = []

        try:
            # Query Elo database for candidates via centralized service
            if get_elo_service is None:
                return []

            elo_svc = get_elo_service()

            # Find models that beat current best by threshold
            rows = elo_svc.execute_query("""
                SELECT participant_id, board_type, num_players, rating, games_played
                FROM elo_ratings
                WHERE games_played >= ?
                ORDER BY board_type, num_players, rating DESC
            """, (self.config.min_games,))

            # Group by config and find candidates
            by_config: Dict[str, List[tuple]] = {}
            for row in rows:
                config_key = f"{row[1]}_{row[2]}p"
                if config_key not in by_config:
                    by_config[config_key] = []
                by_config[config_key].append(row)

            for config_key, models in by_config.items():
                if len(models) < 2:
                    continue

                best = models[0]
                current_best_id = f"ringrift_best_{config_key.replace('_', '_')}"

                # Check if top model beats current best by threshold
                for model in models:
                    if model[0] == current_best_id:
                        continue
                    if model[3] - best[3] >= self.config.elo_threshold:
                        candidates.append({
                            "model_id": model[0],
                            "config": config_key,
                            "elo": model[3],
                            "games": model[4],
                            "elo_gain": model[3] - best[3],
                        })
                        break

            return candidates

        except Exception as e:
            print(f"[ModelPromoter] Error checking candidates: {e}")
            return []

    async def execute_promotion(self, candidate: Dict[str, Any]) -> bool:
        """Execute a model promotion with holdout validation gate."""
        try:
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.PROMOTION_CANDIDATE,
                payload=candidate
            ))

            # Holdout validation gate - check for overfitting before promotion
            if HAS_HOLDOUT_VALIDATION and evaluate_model_on_holdout is not None:
                config_key = candidate["config"]
                # Parse board_type and num_players from config key (e.g., "standard_2p")
                parts = config_key.rsplit("_", 1)
                board_type = parts[0] if len(parts) == 2 else config_key
                num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2

                # Get model path for evaluation
                model_path = AI_SERVICE_ROOT / "data" / "models" / f"{candidate['model_id']}.pt"
                if not model_path.exists():
                    # Try alternative path patterns
                    model_path = AI_SERVICE_ROOT / "models" / f"{candidate['model_id']}.pt"

                if model_path.exists():
                    try:
                        # Run holdout evaluation (synchronous call in async context)
                        eval_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: evaluate_model_on_holdout(
                                model_path=str(model_path),
                                board_type=board_type,
                                num_players=num_players,
                                train_loss=candidate.get("train_loss"),
                            )
                        )

                        # Emit metrics
                        if HAS_PROMETHEUS:
                            HOLDOUT_LOSS.labels(config=config_key).set(eval_result.holdout_loss)
                            if eval_result.overfit_gap is not None:
                                HOLDOUT_OVERFIT_GAP.labels(config=config_key).set(eval_result.overfit_gap)

                        # Check for overfitting
                        if eval_result.overfit_gap is not None and eval_result.overfit_gap > OVERFIT_THRESHOLD:
                            print(f"[ModelPromoter] Promotion BLOCKED for {candidate['model_id']}: "
                                  f"overfit_gap={eval_result.overfit_gap:.4f} > threshold={OVERFIT_THRESHOLD}")
                            if HAS_PROMETHEUS:
                                HOLDOUT_EVALUATIONS.labels(config=config_key, result='failed_overfit').inc()
                                PROMOTION_BLOCKED_OVERFIT.labels(config=config_key).inc()
                            return False

                        print(f"[ModelPromoter] Holdout validation PASSED for {candidate['model_id']}: "
                              f"holdout_loss={eval_result.holdout_loss:.4f}, gap={eval_result.overfit_gap}")
                        if HAS_PROMETHEUS:
                            HOLDOUT_EVALUATIONS.labels(config=config_key, result='passed').inc()

                    except Exception as e:
                        print(f"[ModelPromoter] Holdout validation error (proceeding anyway): {e}")
                        if HAS_PROMETHEUS:
                            HOLDOUT_EVALUATIONS.labels(config=config_key, result='skipped').inc()
                else:
                    print(f"[ModelPromoter] Model file not found for holdout validation: {model_path}")
                    if HAS_PROMETHEUS:
                        HOLDOUT_EVALUATIONS.labels(config=config_key, result='skipped').inc()

            # Run promotion script
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "auto_promote_best_models.py"),
                "--config", candidate["config"],
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            success = process.returncode == 0

            if success:
                self.state.total_promotions += 1

                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload=candidate
                ))

                # Sync to cluster if enabled
                if self.config.sync_to_cluster:
                    await self._sync_to_cluster(candidate)

            return success

        except Exception as e:
            print(f"[ModelPromoter] Error executing promotion: {e}")
            return False

    async def _sync_to_cluster(self, candidate: Dict[str, Any]):
        """Sync promoted model to cluster."""
        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "sync_models.py"),
                "--push-promoted",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            await asyncio.wait_for(process.communicate(), timeout=300)

        except Exception as e:
            print(f"[ModelPromoter] Error syncing to cluster: {e}")
