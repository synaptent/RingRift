"""Unified Loop Promotion Services.

This module contains model promotion services for the unified AI loop:
- ModelPromoter: Automatic model promotion based on Elo with holdout validation

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).

Integration with PromotionController:
The ModelPromoter now delegates to PromotionController for:
- Centralized promotion criteria (PromotionCriteria)
- Decision evaluation (PromotionDecision)
- Prometheus metrics emission

The existing logic (Elo queries, holdout validation, script execution) remains
but is enhanced with proper criteria checking and observability.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .config import DataEvent, DataEventType, PromotionConfig

if TYPE_CHECKING:
    from unified_ai_loop import EventBus, UnifiedLoopState

from app.utils.paths import AI_SERVICE_ROOT

# Board name abbreviations for champion model IDs (must match model_promotion_manager.py)
BOARD_ALIAS_TOKENS = {
    "square8": "sq8",
    "square19": "sq19",
    "hexagonal": "hex",
    "hex8": "hex8",  # Already short
}

# Optional ELO service import
try:
    from app.training.elo_service import get_elo_service
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None

# Optional PromotionController for centralized criteria and metrics
try:
    from app.training.promotion_controller import (
        PromotionController,
        PromotionCriteria,
        PromotionDecision,
        PromotionType,
    )
    HAS_PROMOTION_CONTROLLER = True
except ImportError:
    HAS_PROMOTION_CONTROLLER = False
    PromotionController = None
    PromotionCriteria = None
    PromotionDecision = None
    PromotionType = None

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

# Optional ModelLifecycleManager for unified lifecycle management
try:
    from app.training.model_lifecycle import ModelLifecycleManager
    HAS_LIFECYCLE_MANAGER = True
except ImportError:
    HAS_LIFECYCLE_MANAGER = False
    ModelLifecycleManager = None

# Optional ClusterHealthMonitor for checking cluster state before promotion
try:
    from app.monitoring.cluster_monitor import ClusterHealthMonitor, check_local_health
    from app.monitoring.base import HealthStatus
    HAS_CLUSTER_MONITOR = True
except ImportError:
    HAS_CLUSTER_MONITOR = False
    ClusterHealthMonitor = None
    check_local_health = None
    HealthStatus = None


class ModelPromoter:
    """Handles automatic model promotion based on Elo.

    Integrates with PromotionController for centralized criteria evaluation
    and Prometheus metrics emission. The PromotionController provides:
    - Standardized promotion criteria (min_elo_improvement, min_games_played, etc.)
    - Metrics emission for all promotion decisions
    - Support for different promotion types (staging, production, tier, rollback)
    """

    def __init__(self, config: PromotionConfig, state: "UnifiedLoopState", event_bus: "EventBus"):
        self.config = config
        self.state = state
        self.event_bus = event_bus

        # Initialize PromotionController with criteria from config
        self._promotion_controller = None
        if HAS_PROMOTION_CONTROLLER and PromotionController is not None:
            try:
                criteria = PromotionCriteria(
                    min_elo_improvement=config.elo_threshold,
                    min_games_played=config.min_games,
                )
                self._promotion_controller = PromotionController(criteria=criteria)
            except Exception as e:
                print(f"[ModelPromoter] Warning: Failed to initialize PromotionController: {e}")

        # Initialize ModelLifecycleManager for unified lifecycle tracking
        self._lifecycle_manager = None
        if HAS_LIFECYCLE_MANAGER and ModelLifecycleManager is not None:
            try:
                self._lifecycle_manager = ModelLifecycleManager()
                print("[ModelPromoter] ModelLifecycleManager initialized for unified lifecycle tracking")
            except Exception as e:
                print(f"[ModelPromoter] Warning: Failed to initialize ModelLifecycleManager: {e}")

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

                # Generate champion ID using board abbreviation (e.g., sq8 instead of square8)
                # to match the actual IDs in the database
                parts = config_key.rsplit("_", 1)
                board_type = parts[0] if len(parts) == 2 else config_key
                num_players_str = parts[1] if len(parts) == 2 else "2p"
                board_abbrev = BOARD_ALIAS_TOKENS.get(board_type, board_type)
                current_best_id = f"ringrift_best_{board_abbrev}_{num_players_str}"

                # Find the champion model's rating (ringrift_best_*) for comparison
                # Bug fix: previously compared against models[0] (top-rated) which meant
                # elo_gain was always <= 0. Now we correctly compare against the champion.
                champion_rating = None
                for m in models:
                    if m[0] == current_best_id:
                        champion_rating = m[3]
                        break

                # If champion not in ratings (e.g., never played), use top model as baseline
                # This allows initial promotion when no champion is established
                if champion_rating is None:
                    champion_rating = models[0][3]
                    print(f"[ModelPromoter] No champion found for {config_key}, using top model rating {champion_rating:.1f}")

                # Check if any model beats champion by threshold
                for model in models:
                    if model[0] == current_best_id:
                        continue

                    elo_gain = model[3] - champion_rating
                    if elo_gain >= self.config.elo_threshold:
                        candidate = {
                            "model_id": model[0],
                            "config": config_key,
                            "elo": model[3],
                            "games": model[4],
                            "elo_gain": elo_gain,
                        }

                        # Use PromotionController for centralized evaluation and metrics
                        if self._promotion_controller and HAS_PROMOTION_CONTROLLER:
                            parts = config_key.rsplit("_", 1)
                            board_type = parts[0] if len(parts) == 2 else config_key
                            num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2

                            decision = self._promotion_controller.evaluate_promotion(
                                model_id=model[0],
                                board_type=board_type,
                                num_players=num_players,
                                promotion_type=PromotionType.PRODUCTION,
                                baseline_model_id=current_best_id,
                            )

                            # Add decision details to candidate
                            candidate["promotion_decision"] = decision.to_dict()

                            if not decision.should_promote:
                                print(f"[ModelPromoter] Candidate {model[0]} rejected by PromotionController: {decision.reason}")
                                continue

                        candidates.append(candidate)
                        break

            return candidates

        except Exception as e:
            print(f"[ModelPromoter] Error checking candidates: {e}")
            return []

    async def execute_promotion(self, candidate: Dict[str, Any]) -> bool:
        """Execute a model promotion with holdout validation gate and cluster health check."""
        try:
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.PROMOTION_CANDIDATE,
                payload=candidate
            ))

            # Cluster health gate - defer promotion if cluster is unhealthy
            # This prevents deploying to a degraded cluster
            if HAS_CLUSTER_MONITOR and check_local_health is not None:
                try:
                    health_result = check_local_health()
                    if health_result.status != HealthStatus.HEALTHY:
                        print(f"[ModelPromoter] DEFERRING promotion of {candidate['model_id']}: "
                              f"cluster health is {health_result.status}")
                        await self.event_bus.publish(DataEvent(
                            event_type=DataEventType.PROMOTION_REJECTED,
                            payload={
                                **candidate,
                                "reason": f"cluster_unhealthy: {health_result.status}",
                                "alerts": [str(a) for a in health_result.alerts[:3]],
                            }
                        ))
                        return False
                except Exception as e:
                    # Don't block promotion on health check failure
                    print(f"[ModelPromoter] Warning: Cluster health check failed: {e}")

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
                            # Emit execution failure metrics
                            self._emit_execution_metrics(candidate, success=False)
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
            # Bug fix: auto_promote_best_models.py doesn't exist; use model_promotion_manager.py
            # which has proper --auto-promote support for Elo-based promotion
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "model_promotion_manager.py"),
                "--auto-promote",
                "--elo-threshold", str(self.config.elo_threshold),
                "--min-games", str(self.config.min_games),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            success = process.returncode == 0

            # Emit execution metrics via PromotionController
            self._emit_execution_metrics(candidate, success=success, dry_run=False)

            if success:
                self.state.total_promotions += 1

                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload=candidate
                ))

                # Sync to cluster if enabled
                if self.config.sync_to_cluster:
                    await self._sync_to_cluster(candidate)

                # Run lifecycle maintenance for this config (clean up old models)
                if self._lifecycle_manager:
                    try:
                        config_key = candidate["config"]
                        result = self._lifecycle_manager.check_config(config_key)
                        if result.archived > 0 or result.deleted > 0:
                            print(f"[ModelPromoter] Lifecycle maintenance for {config_key}: "
                                  f"archived={result.archived}, deleted={result.deleted}")
                    except Exception as e:
                        print(f"[ModelPromoter] Lifecycle maintenance error (non-blocking): {e}")

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

    def _emit_execution_metrics(
        self,
        candidate: Dict[str, Any],
        success: bool,
        dry_run: bool = False,
    ) -> None:
        """Emit Prometheus metrics for promotion execution.

        Uses the centralized record_promotion_execution helper from app.metrics.
        """
        try:
            from app.metrics import record_promotion_execution
            record_promotion_execution(
                promotion_type="production",  # ModelPromoter always does production promotions
                success=success,
                dry_run=dry_run,
            )
        except ImportError:
            pass
        except Exception as e:
            print(f"[ModelPromoter] Error emitting execution metrics: {e}")


async def verify_elo_promotion_pipeline() -> Dict[str, Any]:
    """Verify the Elo promotion pipeline is working end-to-end.

    This function checks all components of the promotion pipeline:
    1. EloService availability and database connectivity
    2. PromotionController initialization
    3. Model rating queries
    4. Champion model detection
    5. Promotion candidate evaluation

    Returns:
        Dict with verification results for each component.
    """
    results = {
        "elo_service_available": False,
        "elo_database_accessible": False,
        "promotion_controller_available": False,
        "model_ratings_count": 0,
        "champions_found": [],
        "promotion_candidates": [],
        "errors": [],
        "timestamp": None,
    }

    from datetime import datetime
    results["timestamp"] = datetime.now().isoformat()

    # Check EloService
    if not HAS_ELO_SERVICE or get_elo_service is None:
        results["errors"].append("EloService not available")
    else:
        try:
            elo_svc = get_elo_service()
            results["elo_service_available"] = True

            # Check database connectivity
            rows = elo_svc.execute_query(
                "SELECT COUNT(*) FROM elo_ratings"
            )
            if rows:
                results["elo_database_accessible"] = True
                results["model_ratings_count"] = rows[0][0]
        except Exception as e:
            results["errors"].append(f"EloService error: {str(e)}")

    # Check PromotionController
    if not HAS_PROMOTION_CONTROLLER:
        results["errors"].append("PromotionController not available")
    else:
        try:
            criteria = PromotionCriteria()
            controller = PromotionController(criteria=criteria)
            results["promotion_controller_available"] = True
        except Exception as e:
            results["errors"].append(f"PromotionController error: {str(e)}")

    # Check for champions in the database
    if results["elo_service_available"] and results["elo_database_accessible"]:
        try:
            elo_svc = get_elo_service()
            # Find champion models
            rows = elo_svc.execute_query("""
                SELECT participant_id, board_type, num_players, rating, games_played
                FROM elo_ratings
                WHERE participant_id LIKE 'ringrift_best_%'
                ORDER BY board_type, num_players
            """)
            for row in rows:
                results["champions_found"].append({
                    "model_id": row[0],
                    "board_type": row[1],
                    "num_players": row[2],
                    "rating": row[3],
                    "games_played": row[4],
                })
        except Exception as e:
            results["errors"].append(f"Champion query error: {str(e)}")

    # Check for promotion candidates
    if results["elo_service_available"] and results["elo_database_accessible"]:
        try:
            elo_svc = get_elo_service()
            # Find potential candidates (models with > 50 games that aren't champions)
            rows = elo_svc.execute_query("""
                SELECT participant_id, board_type, num_players, rating, games_played
                FROM elo_ratings
                WHERE games_played >= 50
                  AND participant_id NOT LIKE 'ringrift_best_%'
                ORDER BY rating DESC
                LIMIT 10
            """)
            for row in rows:
                results["promotion_candidates"].append({
                    "model_id": row[0],
                    "board_type": row[1],
                    "num_players": row[2],
                    "rating": row[3],
                    "games_played": row[4],
                })
        except Exception as e:
            results["errors"].append(f"Candidate query error: {str(e)}")

    # Summary
    results["pipeline_healthy"] = (
        results["elo_service_available"]
        and results["elo_database_accessible"]
        and results["promotion_controller_available"]
        and len(results["errors"]) == 0
    )

    return results


def print_promotion_verification(results: Dict[str, Any]) -> None:
    """Pretty print promotion verification results."""
    print("\n" + "=" * 60)
    print("ELO PROMOTION PIPELINE VERIFICATION")
    print("=" * 60)

    status = "HEALTHY" if results.get("pipeline_healthy") else "ISSUES DETECTED"
    print(f"\nOverall Status: {status}")
    print(f"Timestamp: {results.get('timestamp')}")

    print(f"\nComponent Status:")
    print(f"  EloService: {'OK' if results.get('elo_service_available') else 'MISSING'}")
    print(f"  Database: {'OK' if results.get('elo_database_accessible') else 'ERROR'}")
    print(f"  PromotionController: {'OK' if results.get('promotion_controller_available') else 'MISSING'}")

    print(f"\nModel Ratings: {results.get('model_ratings_count', 0)} models in database")

    if results.get("champions_found"):
        print(f"\nChampion Models ({len(results['champions_found'])}):")
        for c in results["champions_found"]:
            print(f"  {c['model_id']}: Elo={c['rating']:.1f}, games={c['games_played']}")
    else:
        print("\nNo champion models found (ringrift_best_*)")

    if results.get("promotion_candidates"):
        print(f"\nTop Promotion Candidates ({len(results['promotion_candidates'])}):")
        for c in results["promotion_candidates"][:5]:
            print(f"  {c['model_id']}: Elo={c['rating']:.1f}, games={c['games_played']}")

    if results.get("errors"):
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error}")

    print("\n" + "=" * 60)
