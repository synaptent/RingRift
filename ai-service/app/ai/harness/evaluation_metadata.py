"""Evaluation metadata for AI harness outputs.

This module defines the EvaluationMetadata dataclass that captures all relevant
information about a move selection, including search statistics and visit
distributions for training data enrichment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvaluationMetadata:
    """Metadata from AI move evaluation.

    This dataclass captures comprehensive information about how a move was
    selected, enabling:
    - Elo rating with harness-specific tracking
    - Visit distribution storage for soft policy targets
    - Search quality analysis and debugging

    Attributes:
        value_estimate: The AI's estimate of position value after the move.
            Range depends on harness: [-1, 1] for NN, [-100000, 100000] for heuristic.
        visit_distribution: MCTS visit counts per move as action_key -> visit_count.
            Only populated for tree search methods (Gumbel MCTS, etc.).
            Action keys are typically "{move_type}_{from}_{to}" format.
        policy_distribution: Raw policy logits/probabilities per move.
            Always populated for NN-based harnesses.
        search_depth: Maximum search depth reached.
            For MCTS: max tree depth explored.
            For Minimax/MaxN: configured search depth.
        nodes_visited: Total nodes evaluated during search.
        time_ms: Wall-clock time for move selection in milliseconds.
        harness_type: The harness used for this evaluation (e.g., "gumbel_mcts").
        model_type: The model type used ("nn", "nnue", or "heuristic").
        model_id: Model identifier for Elo tracking (e.g., "ringrift_v5_hex8_2p").
        config_hash: Hash of harness configuration for consistent Elo tracking.
        simulations: Number of MCTS simulations (for tree search harnesses).
        extra: Additional harness-specific metadata.
    """

    value_estimate: float = 0.0
    visit_distribution: dict[str, float] | None = None
    policy_distribution: dict[str, float] | None = None
    search_depth: int | None = None
    nodes_visited: int = 0
    time_ms: float = 0.0
    harness_type: str = ""
    model_type: str = ""
    model_id: str = ""
    config_hash: str = ""
    simulations: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Remove None values to reduce storage
        return {k: v for k, v in result.items() if v is not None}

    def to_json(self) -> str:
        """Serialize to JSON string for database storage."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> EvaluationMetadata:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationMetadata:
        """Create from dictionary."""
        # Filter to only known fields
        known_fields = {
            "value_estimate",
            "visit_distribution",
            "policy_distribution",
            "search_depth",
            "nodes_visited",
            "time_ms",
            "harness_type",
            "model_type",
            "model_id",
            "config_hash",
            "simulations",
            "extra",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def get_composite_id(self) -> str:
        """Generate composite participant ID for Elo tracking.

        Format: {model_id}:{harness_type}:{config_hash}
        Example: ringrift_v5_hex8_2p:gumbel_mcts:b200
        """
        parts = [
            self.model_id or "unknown",
            self.harness_type or "unknown",
            self.config_hash or "default",
        ]
        return ":".join(parts)

    def has_visit_distribution(self) -> bool:
        """Check if visit distribution is available for soft targets."""
        return self.visit_distribution is not None and len(self.visit_distribution) > 0

    def get_top_visits(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N moves by visit count.

        Useful for logging and debugging MCTS search.
        """
        if not self.visit_distribution:
            return []
        sorted_visits = sorted(
            self.visit_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_visits[:n]

    def merge(self, other: EvaluationMetadata) -> EvaluationMetadata:
        """Merge with another metadata instance (e.g., for ensemble methods).

        Takes the average of numeric values and unions of distributions.
        """
        merged_visits = None
        if self.visit_distribution and other.visit_distribution:
            merged_visits = {}
            all_keys = set(self.visit_distribution.keys()) | set(other.visit_distribution.keys())
            for k in all_keys:
                v1 = self.visit_distribution.get(k, 0.0)
                v2 = other.visit_distribution.get(k, 0.0)
                merged_visits[k] = (v1 + v2) / 2.0
        elif self.visit_distribution:
            merged_visits = self.visit_distribution.copy()
        elif other.visit_distribution:
            merged_visits = other.visit_distribution.copy()

        return EvaluationMetadata(
            value_estimate=(self.value_estimate + other.value_estimate) / 2.0,
            visit_distribution=merged_visits,
            policy_distribution=self.policy_distribution or other.policy_distribution,
            search_depth=max(self.search_depth or 0, other.search_depth or 0),
            nodes_visited=self.nodes_visited + other.nodes_visited,
            time_ms=self.time_ms + other.time_ms,
            harness_type=self.harness_type or other.harness_type,
            model_type=self.model_type or other.model_type,
            model_id=self.model_id or other.model_id,
            config_hash=self.config_hash or other.config_hash,
            simulations=(self.simulations or 0) + (other.simulations or 0),
            extra={**self.extra, **other.extra},
        )
