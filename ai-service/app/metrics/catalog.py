"""Metric Catalog - Central registry for all RingRift metrics.

This module provides a unified catalog of all Prometheus metrics used in
the RingRift AI service, enabling:
- Discovery of available metrics by category
- Documentation lookup
- Consistent naming conventions
- Metric type information

Usage:
    from app.metrics.catalog import (
        MetricCatalog,
        get_metric_catalog,
        MetricCategory,
        MetricInfo,
    )

    # Get catalog instance
    catalog = get_metric_catalog()

    # List metrics by category
    training_metrics = catalog.get_by_category(MetricCategory.TRAINING)
    for metric in training_metrics:
        print(f"{metric.name}: {metric.description}")

    # Get metric info
    info = catalog.get("ringrift_selfplay_games_total")
    print(f"Type: {info.metric_type}, Labels: {info.labels}")

    # Search metrics
    matches = catalog.search("elo")
    for m in matches:
        print(m.name)

Naming Conventions:
    - All metrics start with "ringrift_" prefix
    - Category is included in name: ringrift_{category}_{metric_name}
    - Use snake_case for names
    - Total counters end with "_total"
    - Histograms include unit: "_seconds", "_bytes", etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = [
    "MetricCatalog",
    "MetricCategory",
    "MetricInfo",
    "MetricType",
    "get_metric_catalog",
    "register_metric",
]


class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"    # Monotonically increasing
    GAUGE = "gauge"        # Can go up and down
    HISTOGRAM = "histogram"  # Distribution with buckets
    SUMMARY = "summary"    # Distribution with quantiles
    INFO = "info"          # Static info labels


class MetricCategory(Enum):
    """Categories of metrics in the RingRift system."""
    # Pipeline stages
    SELFPLAY = "selfplay"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"

    # Infrastructure
    SYNC = "sync"
    COORDINATOR = "coordinator"
    PIPELINE = "pipeline"

    # Application
    API = "api"
    AI = "ai"
    CACHE = "cache"

    # Quality
    DATA_QUALITY = "data_quality"

    # System
    SYSTEM = "system"
    HEALTH = "health"


@dataclass
class MetricInfo:
    """Information about a registered metric."""
    name: str
    description: str
    metric_type: MetricType
    category: MetricCategory
    labels: list[str] = field(default_factory=list)
    unit: str | None = None  # seconds, bytes, etc.
    buckets: list[float] | None = None  # For histograms
    module: str | None = None  # Source module

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def full_name(self) -> str:
        """Get the Prometheus-compatible full name."""
        return self.name

    @property
    def short_name(self) -> str:
        """Get name without ringrift_ prefix."""
        if self.name.startswith("ringrift_"):
            return self.name[9:]
        return self.name


class MetricCatalog:
    """Central registry and catalog for all metrics.

    Provides discovery, documentation, and lookup for all Prometheus
    metrics used in the RingRift AI service.
    """

    _instance: MetricCatalog | None = None

    def __init__(self):
        self._metrics: dict[str, MetricInfo] = {}
        self._by_category: dict[MetricCategory, set[str]] = {
            cat: set() for cat in MetricCategory
        }
        self._initialized = False

    @classmethod
    def get_instance(cls) -> MetricCatalog:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_all_metrics()
        return cls._instance

    def register(self, info: MetricInfo) -> None:
        """Register a metric in the catalog.

        Args:
            info: MetricInfo describing the metric
        """
        self._metrics[info.name] = info
        self._by_category[info.category].add(info.name)

    def get(self, name: str) -> MetricInfo | None:
        """Get metric info by name.

        Args:
            name: Full metric name

        Returns:
            MetricInfo or None if not found
        """
        return self._metrics.get(name)

    def get_by_category(self, category: MetricCategory) -> list[MetricInfo]:
        """Get all metrics in a category.

        Args:
            category: MetricCategory to filter by

        Returns:
            List of MetricInfo in the category
        """
        names = self._by_category.get(category, set())
        return [self._metrics[n] for n in names if n in self._metrics]

    def search(self, query: str) -> list[MetricInfo]:
        """Search metrics by name or description.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching MetricInfo
        """
        query_lower = query.lower()
        return [
            info for info in self._metrics.values()
            if query_lower in info.name.lower()
            or query_lower in info.description.lower()
        ]

    def list_all(self) -> list[MetricInfo]:
        """Get all registered metrics.

        Returns:
            List of all MetricInfo
        """
        return list(self._metrics.values())

    def list_names(self) -> list[str]:
        """Get all metric names.

        Returns:
            Sorted list of metric names
        """
        return sorted(self._metrics.keys())

    def get_documentation(self) -> str:
        """Generate documentation for all metrics.

        Returns:
            Markdown-formatted documentation string
        """
        lines = ["# RingRift Metrics Catalog\n"]

        for category in MetricCategory:
            metrics = self.get_by_category(category)
            if not metrics:
                continue

            lines.append(f"\n## {category.value.title()} Metrics\n")
            for metric in sorted(metrics, key=lambda m: m.name):
                lines.append(f"### `{metric.name}`\n")
                lines.append(f"- **Type**: {metric.metric_type.value}")
                lines.append(f"- **Description**: {metric.description}")
                if metric.labels:
                    lines.append(f"- **Labels**: {', '.join(metric.labels)}")
                if metric.unit:
                    lines.append(f"- **Unit**: {metric.unit}")
                lines.append("")

        return "\n".join(lines)

    def _register_all_metrics(self) -> None:
        """Register all known metrics from the codebase."""
        if self._initialized:
            return
        self._initialized = True

        # =================================================================
        # Selfplay Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_selfplay_games_total",
            description="Total selfplay games generated by orchestrators.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.SELFPLAY,
            labels=["board_type", "num_players", "orchestrator"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_selfplay_games_per_second",
            description="Current selfplay throughput in games per second.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.SELFPLAY,
            labels=["board_type", "num_players"],
            unit="games/second",
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_selfplay_batch_duration_seconds",
            description="Duration of selfplay batches in seconds.",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.SELFPLAY,
            labels=["board_type", "num_players"],
            unit="seconds",
            buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_selfplay_errors_total",
            description="Total errors during selfplay execution.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.SELFPLAY,
            labels=["board_type", "num_players", "error_type"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_selfplay_queue_size",
            description="Current size of the selfplay job queue.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.SELFPLAY,
            labels=["orchestrator"],
            module="app.metrics.orchestrator",
        ))

        # =================================================================
        # Training Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_orchestrator_training_runs_total",
            description="Total training runs completed.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.TRAINING,
            labels=["board_type", "num_players", "model_type"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_training_run_duration_seconds",
            description="Duration of training runs in seconds.",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.TRAINING,
            labels=["board_type", "num_players"],
            unit="seconds",
            buckets=[300, 600, 1200, 1800, 3600, 7200, 14400],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_orchestrator_training_loss",
            description="Current/final training loss.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.TRAINING,
            labels=["board_type", "num_players", "loss_type"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_training_accuracy",
            description="Current/final training accuracy.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.TRAINING,
            labels=["board_type", "num_players", "metric_type"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_training_samples_processed_total",
            description="Total training samples processed.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.TRAINING,
            labels=["board_type", "num_players"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_orchestrator_training_epochs_total",
            description="Total training epochs completed.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.TRAINING,
            labels=["board_type", "num_players"],
            module="app.metrics.orchestrator",
        ))

        # =================================================================
        # Evaluation Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_evaluation_games_total",
            description="Total evaluation games played.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.EVALUATION,
            labels=["board_type", "num_players", "eval_type"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_evaluation_elo_delta",
            description="Elo delta from evaluation (candidate - baseline).",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.EVALUATION,
            labels=["board_type", "num_players", "candidate_model"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_evaluation_win_rate",
            description="Win rate from evaluation.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.EVALUATION,
            labels=["board_type", "num_players", "candidate_model"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_evaluation_duration_seconds",
            description="Duration of evaluation runs in seconds.",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.EVALUATION,
            labels=["board_type", "num_players"],
            unit="seconds",
            module="app.metrics.orchestrator",
        ))

        # =================================================================
        # Promotion Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_model_promotions_total",
            description="Total model promotions executed.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.PROMOTION,
            labels=["board_type", "num_players", "tier"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_model_promotion_elo_gain",
            description="Elo gain from promoted model.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.PROMOTION,
            labels=["board_type", "num_players"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_model_promotion_rejections_total",
            description="Total model promotion rejections.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.PROMOTION,
            labels=["board_type", "num_players", "reason"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_current_model_elo",
            description="Current production model Elo rating.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.PROMOTION,
            labels=["board_type", "num_players"],
            module="app.metrics.orchestrator",
        ))

        # =================================================================
        # Sync Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_data_sync_duration_seconds",
            description="Duration of data sync operations.",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.SYNC,
            labels=["sync_type", "source"],
            unit="seconds",
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_data_sync_games_total",
            description="Total games synced from remote hosts.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.SYNC,
            labels=["source", "board_type"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_data_sync_errors_total",
            description="Total data sync errors.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.SYNC,
            labels=["error_type", "source"],
            module="app.metrics.orchestrator",
        ))

        # =================================================================
        # API Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ai_move_requests_total",
            description="Total number of /ai/move requests.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.API,
            labels=["ai_type", "difficulty", "outcome"],
            module="app.metrics",
        ))
        self.register(MetricInfo(
            name="ai_move_latency_seconds",
            description="Latency of /ai/move requests in seconds.",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.API,
            labels=["ai_type", "difficulty"],
            unit="seconds",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            module="app.metrics",
        ))
        self.register(MetricInfo(
            name="ringrift_ai_errors_total",
            description="Total AI error counts by type.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.AI,
            labels=["error_type", "difficulty", "board_type"],
            module="app.metrics",
        ))
        self.register(MetricInfo(
            name="ringrift_ai_fallbacks_total",
            description="Total AI fallback events when primary strategy fails.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.AI,
            labels=["ai_type", "fallback_method", "board_type"],
            module="app.metrics",
        ))

        # =================================================================
        # Pipeline Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_pipeline_stage_duration_seconds",
            description="Duration of pipeline stages in seconds.",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.PIPELINE,
            labels=["stage"],
            unit="seconds",
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_pipeline_iterations_total",
            description="Total pipeline iterations completed.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.PIPELINE,
            labels=["outcome"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_pipeline_errors_total",
            description="Total pipeline errors by type.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.PIPELINE,
            labels=["error_type", "stage"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_pipeline_state",
            description="Current pipeline state (0=idle, 1=selfplay, 2=training, 3=eval, 4=promo).",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.PIPELINE,
            labels=[],
            module="app.metrics.orchestrator",
        ))

        # =================================================================
        # Data Quality Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_training_data_quality_score",
            description="Quality score of training data (0-1).",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.DATA_QUALITY,
            labels=["board_type", "num_players"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_training_data_high_quality_count",
            description="Count of high-quality training samples.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.DATA_QUALITY,
            labels=["board_type", "num_players"],
            module="app.metrics.orchestrator",
        ))
        self.register(MetricInfo(
            name="ringrift_training_data_elo",
            description="Average Elo of training data games.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.DATA_QUALITY,
            labels=["board_type", "num_players"],
            module="app.metrics.orchestrator",
        ))

        # =================================================================
        # Coordinator Metrics
        # =================================================================
        self.register(MetricInfo(
            name="ringrift_coordinator_status",
            description="Current coordinator status (1=active, 0=inactive).",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.COORDINATOR,
            labels=["coordinator_type"],
            module="app.metrics.coordinator",
        ))
        self.register(MetricInfo(
            name="ringrift_coordinator_uptime_seconds",
            description="Coordinator uptime in seconds.",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.COORDINATOR,
            labels=["coordinator_type"],
            unit="seconds",
            module="app.metrics.coordinator",
        ))
        self.register(MetricInfo(
            name="ringrift_coordinator_operations_total",
            description="Total coordinator operations executed.",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.COORDINATOR,
            labels=["coordinator_type", "operation"],
            module="app.metrics.coordinator",
        ))


def get_metric_catalog() -> MetricCatalog:
    """Get the global metric catalog instance."""
    return MetricCatalog.get_instance()


def register_metric(
    name: str,
    description: str,
    metric_type: MetricType,
    category: MetricCategory,
    labels: list[str] | None = None,
    unit: str | None = None,
) -> MetricInfo:
    """Register a new metric in the catalog.

    Use this when defining new metrics to ensure they're cataloged.

    Args:
        name: Prometheus metric name
        description: Human-readable description
        metric_type: Type of metric (counter, gauge, etc.)
        category: Category for grouping
        labels: Label names
        unit: Unit of measurement

    Returns:
        MetricInfo for the registered metric
    """
    info = MetricInfo(
        name=name,
        description=description,
        metric_type=metric_type,
        category=category,
        labels=labels or [],
        unit=unit,
    )
    get_metric_catalog().register(info)
    return info
