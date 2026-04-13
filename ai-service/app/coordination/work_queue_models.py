"""Shared work-queue models and constants.

This module holds data structures used by both ``work_queue`` and
``work_queue_storage`` so the storage mixin can avoid importing the concrete
queue implementation module.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from app.coordination.types import WorkStatus


class WorkQueueBackendType(str, Enum):
    """Available work queue backend types."""

    RAFT = "raft"
    SQLITE = "sqlite"


# Feb 28, 2026: Lowered from 7500/15000 — only ~7 GPU nodes can consume work,
# so 15K pending items = 2000+ per node = days of backlog. 500/1000 is ~2h drain time.
BACKPRESSURE_SOFT_LIMIT = int(os.environ.get("RINGRIFT_WORK_QUEUE_SOFT_LIMIT", "500"))
BACKPRESSURE_HARD_LIMIT = int(os.environ.get("RINGRIFT_WORK_QUEUE_HARD_LIMIT", "1000"))
BACKPRESSURE_RECOVERY_THRESHOLD = int(os.environ.get("RINGRIFT_WORK_QUEUE_RECOVERY", "200"))


class WorkType(str, Enum):
    """Types of work that can be queued."""

    TRAINING = "training"
    GPU_CMAES = "gpu_cmaes"
    CPU_CMAES = "cpu_cmaes"
    TOURNAMENT = "tournament"
    GAUNTLET = "gauntlet"
    SELFPLAY = "selfplay"
    DATA_MERGE = "data_merge"
    DATA_SYNC = "data_sync"
    VALIDATION = "validation"
    HYPERPARAM_SWEEP = "hyperparam_sweep"


@dataclass
class WorkItem:
    """A unit of work to be executed."""

    work_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    work_type: WorkType = WorkType.SELFPLAY
    priority: int = 50
    config: dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    claimed_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    status: WorkStatus = WorkStatus.PENDING
    claimed_by: str = ""
    attempts: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 3600.0

    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["work_type"] = self.work_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkItem:
        item = data.copy()
        item["work_type"] = WorkType(item.get("work_type", "selfplay"))
        item["status"] = WorkStatus(item.get("status", "pending"))
        if "depends_on" in item and isinstance(item["depends_on"], str):
            try:
                item["depends_on"] = json.loads(item["depends_on"]) if item["depends_on"] else []
            except (json.JSONDecodeError, TypeError):
                item["depends_on"] = []

        for float_field in ("created_at", "claimed_at", "started_at", "completed_at", "timeout_seconds"):
            if float_field in item and not isinstance(item[float_field], (int, float)):
                try:
                    item[float_field] = float(item[float_field])
                except (ValueError, TypeError):
                    item[float_field] = 0.0

        for int_field in ("priority", "attempts", "max_attempts"):
            if int_field in item and not isinstance(item[int_field], int):
                try:
                    item[int_field] = int(item[int_field])
                except (ValueError, TypeError):
                    item[int_field] = 0

        return cls(**item)

    def is_claimable(self) -> bool:
        """Check if this work can be claimed."""

        if self.status != WorkStatus.PENDING:
            return False
        return not self.attempts >= self.max_attempts

    def has_pending_dependencies(self, completed_ids: set[str]) -> bool:
        """Check if any dependencies are not yet completed."""

        if not self.depends_on:
            return False
        return any(dep_id not in completed_ids for dep_id in self.depends_on)

    def is_timed_out(self) -> bool:
        """Check if this work has timed out."""

        if self.status not in (WorkStatus.CLAIMED, WorkStatus.RUNNING):
            return False
        if self.claimed_at == 0:
            return False
        return time.time() - self.claimed_at > self.timeout_seconds


@dataclass
class ClaimRejectionStats:
    """Track why jobs are not being dispatched."""

    total_claim_attempts: int = 0
    rejected_by_circuit_breaker: int = 0
    rejected_by_capability: int = 0
    rejected_by_exclusion: int = 0
    rejected_by_target_node: int = 0
    rejected_by_target_node_expired: int = 0
    rejected_by_requires_gpu: int = 0
    rejected_by_policy: int = 0
    rejected_by_already_claimed: int = 0
    successful_claims: int = 0
    target_node_rejections: dict[str, int] = field(default_factory=dict)
    last_reset_at: float = field(default_factory=time.time)

    def increment_target_node_rejection(self, target_node: str) -> None:
        self.rejected_by_target_node += 1
        self.target_node_rejections[target_node] = (
            self.target_node_rejections.get(target_node, 0) + 1
        )

    def reset(self) -> None:
        self.total_claim_attempts = 0
        self.rejected_by_circuit_breaker = 0
        self.rejected_by_capability = 0
        self.rejected_by_exclusion = 0
        self.rejected_by_target_node = 0
        self.rejected_by_target_node_expired = 0
        self.rejected_by_requires_gpu = 0
        self.rejected_by_policy = 0
        self.rejected_by_already_claimed = 0
        self.successful_claims = 0
        self.target_node_rejections.clear()
        self.last_reset_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claim_attempts": self.total_claim_attempts,
            "rejected_by_circuit_breaker": self.rejected_by_circuit_breaker,
            "rejected_by_capability": self.rejected_by_capability,
            "rejected_by_exclusion": self.rejected_by_exclusion,
            "rejected_by_target_node": self.rejected_by_target_node,
            "rejected_by_target_node_expired": self.rejected_by_target_node_expired,
            "rejected_by_requires_gpu": self.rejected_by_requires_gpu,
            "rejected_by_policy": self.rejected_by_policy,
            "rejected_by_already_claimed": self.rejected_by_already_claimed,
            "successful_claims": self.successful_claims,
            "target_node_rejections": self.target_node_rejections.copy(),
            "last_reset_at": self.last_reset_at,
            "elapsed_seconds": time.time() - self.last_reset_at,
        }


__all__ = [
    "BACKPRESSURE_HARD_LIMIT",
    "BACKPRESSURE_RECOVERY_THRESHOLD",
    "BACKPRESSURE_SOFT_LIMIT",
    "ClaimRejectionStats",
    "WorkItem",
    "WorkQueueBackendType",
    "WorkType",
]
