"""Standardized AI Decision Logging.

Per Section 4.3 of the action plan, this module provides:
- Structured logging for AI decisions
- Integration with OpenTelemetry tracing
- Prometheus metrics export
- JSON-serializable log entries for analysis

Usage:
    from app.ai.decision_log import AIDecisionLog, log_ai_decision

    # Create a decision log entry
    decision = AIDecisionLog(
        game_id="game_123",
        move_number=5,
        difficulty=3,
        engine_type="mcts",
        simulations=800,
        time_ms=150.5,
        chosen_move="place(3,4)",
        move_score=0.65,
    )

    # Log the decision
    log_ai_decision(decision)

    # Or use the context manager
    with AIDecisionContext(game_id="game_123", difficulty=3) as ctx:
        move = ai.select_move(state)
        ctx.record_move(move, score=0.65)
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, REGISTRY

    def _safe_metric(metric_class, name, doc, **kwargs):
        """Create metric or get existing one."""
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return metric_class(name, doc, **kwargs)

    AI_DECISIONS = _safe_metric(
        Counter,
        'ringrift_ai_decisions_total',
        'Total AI decisions made',
        labelnames=['engine_type', 'difficulty', 'board_type']
    )

    AI_DECISION_TIME = _safe_metric(
        Histogram,
        'ringrift_ai_decision_time_seconds',
        'AI decision time in seconds',
        labelnames=['engine_type', 'difficulty'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    AI_CACHE_HITS = _safe_metric(
        Counter,
        'ringrift_ai_cache_hits_total',
        'AI cache hits',
        labelnames=['cache_type']
    )

    AI_SEARCH_DEPTH = _safe_metric(
        Histogram,
        'ringrift_ai_search_depth',
        'AI search depth',
        labelnames=['engine_type'],
        buckets=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    )

    AI_SIMULATIONS = _safe_metric(
        Histogram,
        'ringrift_ai_simulations',
        'MCTS simulations per move',
        labelnames=['difficulty'],
        buckets=[50, 100, 200, 400, 800, 1600, 3200, 6400]
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Optional OpenTelemetry tracing
try:
    from app.tracing import get_current_span, add_ai_move_attributes, is_tracing_enabled
    HAS_TRACING = True
except ImportError:
    HAS_TRACING = False
    get_current_span = lambda: None
    add_ai_move_attributes = lambda *args, **kwargs: None
    is_tracing_enabled = lambda: False


@dataclass
class AIDecisionLog:
    """Standardized AI decision log entry.

    Captures all relevant information about an AI move decision
    for debugging, analysis, and monitoring purposes.
    """

    # Context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    game_id: str = ""
    move_number: int = 0
    difficulty: int = 0
    board_type: str = "square8"
    num_players: int = 2

    # Engine info
    engine_type: str = ""  # mcts, minimax, heuristic, descent, policy, random
    model_version: str = ""

    # Search parameters
    search_depth: int = 0
    simulations: int = 0
    time_limit_ms: float = 0.0

    # Timing
    time_ms: float = 0.0
    time_breakdown: Dict[str, float] = field(default_factory=dict)

    # Move selection
    chosen_move: str = ""
    move_score: float = 0.0
    move_confidence: float = 0.0
    top_alternatives: List[Tuple[str, float]] = field(default_factory=list)

    # Diagnostics
    cache_hit: bool = False
    cache_type: str = ""  # transposition, policy, nnue
    nodes_evaluated: int = 0
    positions_searched: int = 0

    # State info
    game_phase: str = ""  # placement, movement, finished
    player_number: int = 0
    pieces_in_hand: int = 0

    # Fallback tracking
    used_fallback: bool = False
    fallback_reason: str = ""

    # Error tracking
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_structured_log(self) -> Dict[str, Any]:
        """Convert to structured log format for logging frameworks."""
        return {
            "event": "ai_decision",
            "level": "info" if not self.error else "error",
            **self.to_dict(),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [
            f"[{self.engine_type}]",
            f"move={self.chosen_move}",
            f"score={self.move_score:.3f}",
            f"time={self.time_ms:.1f}ms",
        ]
        if self.simulations > 0:
            parts.append(f"sims={self.simulations}")
        if self.search_depth > 0:
            parts.append(f"depth={self.search_depth}")
        if self.cache_hit:
            parts.append(f"cache={self.cache_type}")
        if self.used_fallback:
            parts.append(f"fallback={self.fallback_reason}")
        return " ".join(parts)


def log_ai_decision(decision: AIDecisionLog, log_level: int = logging.INFO) -> None:
    """Log an AI decision with metrics and tracing.

    Args:
        decision: The decision log entry
        log_level: Logging level (default: INFO)
    """
    # Structured logging
    logger.log(log_level, decision.summary(), extra=decision.to_structured_log())

    # Prometheus metrics
    if HAS_PROMETHEUS:
        AI_DECISIONS.labels(
            engine_type=decision.engine_type,
            difficulty=str(decision.difficulty),
            board_type=decision.board_type,
        ).inc()

        AI_DECISION_TIME.labels(
            engine_type=decision.engine_type,
            difficulty=str(decision.difficulty),
        ).observe(decision.time_ms / 1000.0)

        if decision.cache_hit and decision.cache_type:
            AI_CACHE_HITS.labels(cache_type=decision.cache_type).inc()

        if decision.search_depth > 0:
            AI_SEARCH_DEPTH.labels(engine_type=decision.engine_type).observe(decision.search_depth)

        if decision.simulations > 0:
            AI_SIMULATIONS.labels(difficulty=str(decision.difficulty)).observe(decision.simulations)

    # OpenTelemetry tracing
    if HAS_TRACING and is_tracing_enabled():
        span = get_current_span()
        if span:
            add_ai_move_attributes(
                span,
                board_type=decision.board_type,
                difficulty=decision.difficulty,
                engine_type=decision.engine_type,
                simulations=decision.simulations,
                depth=decision.search_depth,
                time_ms=decision.time_ms,
            )
            span.set_attribute("ai.move", decision.chosen_move)
            span.set_attribute("ai.score", decision.move_score)
            span.set_attribute("ai.cache_hit", decision.cache_hit)
            if decision.used_fallback:
                span.set_attribute("ai.fallback", decision.fallback_reason)


class AIDecisionContext:
    """Context manager for tracking AI decision timing and metadata.

    Usage:
        with AIDecisionContext(game_id="123", difficulty=3, engine_type="mcts") as ctx:
            move = ai.select_move(state)
            ctx.record_move(move, score=0.65)
        # Decision is automatically logged on exit
    """

    def __init__(
        self,
        game_id: str = "",
        move_number: int = 0,
        difficulty: int = 0,
        engine_type: str = "",
        board_type: str = "square8",
        num_players: int = 2,
        player_number: int = 0,
        model_version: str = "",
        auto_log: bool = True,
    ):
        self.decision = AIDecisionLog(
            game_id=game_id,
            move_number=move_number,
            difficulty=difficulty,
            engine_type=engine_type,
            board_type=board_type,
            num_players=num_players,
            player_number=player_number,
            model_version=model_version,
        )
        self.auto_log = auto_log
        self._start_time: Optional[float] = None
        self._time_marks: Dict[str, float] = {}

    def __enter__(self) -> 'AIDecisionContext':
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is not None:
            self.decision.time_ms = (time.perf_counter() - self._start_time) * 1000

        if exc_type is not None:
            self.decision.error = str(exc_val)

        if self.auto_log:
            log_level = logging.ERROR if self.decision.error else logging.INFO
            log_ai_decision(self.decision, log_level)

        return False  # Don't suppress exceptions

    def mark_time(self, label: str) -> None:
        """Mark a time checkpoint for breakdown analysis."""
        if self._start_time is not None:
            elapsed = (time.perf_counter() - self._start_time) * 1000
            self._time_marks[label] = elapsed
            self.decision.time_breakdown[label] = elapsed

    def record_move(
        self,
        move: Any,
        score: float = 0.0,
        confidence: float = 0.0,
        alternatives: Optional[List[Tuple[str, float]]] = None,
    ) -> None:
        """Record the chosen move and its evaluation.

        Args:
            move: The chosen move (will be converted to string)
            score: Move score/evaluation
            confidence: Confidence in the move (0-1)
            alternatives: List of (move_str, score) tuples for top alternatives
        """
        self.decision.chosen_move = str(move) if move else ""
        self.decision.move_score = score
        self.decision.move_confidence = confidence
        if alternatives:
            self.decision.top_alternatives = alternatives

    def record_search_stats(
        self,
        depth: int = 0,
        simulations: int = 0,
        nodes: int = 0,
        positions: int = 0,
    ) -> None:
        """Record search statistics.

        Args:
            depth: Search depth reached
            simulations: MCTS simulations performed
            nodes: Nodes evaluated
            positions: Positions searched
        """
        self.decision.search_depth = depth
        self.decision.simulations = simulations
        self.decision.nodes_evaluated = nodes
        self.decision.positions_searched = positions

    def record_cache_hit(self, cache_type: str) -> None:
        """Record a cache hit.

        Args:
            cache_type: Type of cache (transposition, policy, nnue)
        """
        self.decision.cache_hit = True
        self.decision.cache_type = cache_type

    def record_fallback(self, reason: str) -> None:
        """Record that a fallback was used.

        Args:
            reason: Reason for fallback (timeout, error, etc.)
        """
        self.decision.used_fallback = True
        self.decision.fallback_reason = reason

    def set_game_phase(self, phase: str, pieces_in_hand: int = 0) -> None:
        """Set game phase information.

        Args:
            phase: Current phase (placement, movement, finished)
            pieces_in_hand: Pieces remaining in hand
        """
        self.decision.game_phase = phase
        self.decision.pieces_in_hand = pieces_in_hand


@contextmanager
def track_ai_decision(
    game_id: str = "",
    move_number: int = 0,
    difficulty: int = 0,
    engine_type: str = "",
    board_type: str = "square8",
    **kwargs,
):
    """Convenience context manager for tracking AI decisions.

    Args:
        game_id: Game identifier
        move_number: Current move number
        difficulty: AI difficulty level
        engine_type: Type of AI engine
        board_type: Board type
        **kwargs: Additional AIDecisionContext arguments

    Yields:
        AIDecisionContext instance
    """
    ctx = AIDecisionContext(
        game_id=game_id,
        move_number=move_number,
        difficulty=difficulty,
        engine_type=engine_type,
        board_type=board_type,
        **kwargs,
    )
    with ctx:
        yield ctx


def create_decision_log_from_stats(
    move: Any,
    stats: Dict[str, Any],
    game_id: str = "",
    difficulty: int = 0,
) -> AIDecisionLog:
    """Create AIDecisionLog from a stats dictionary.

    Useful for integrating with existing AI implementations that
    return stats dictionaries.

    Args:
        move: The chosen move
        stats: Dictionary with search statistics
        game_id: Game identifier
        difficulty: AI difficulty

    Returns:
        AIDecisionLog instance
    """
    return AIDecisionLog(
        game_id=game_id,
        difficulty=difficulty,
        engine_type=stats.get('engine_type', stats.get('type', '')),
        chosen_move=str(move) if move else "",
        move_score=stats.get('score', stats.get('value', 0.0)),
        time_ms=stats.get('time_ms', stats.get('elapsed_ms', 0.0)),
        search_depth=stats.get('depth', 0),
        simulations=stats.get('simulations', stats.get('sims', 0)),
        nodes_evaluated=stats.get('nodes', stats.get('nodes_evaluated', 0)),
        cache_hit=stats.get('cache_hit', False),
        model_version=stats.get('model_version', stats.get('model', '')),
    )
