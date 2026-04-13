"""Shared distribution types, transport settings, and remote-path cache."""

from __future__ import annotations

import threading
from dataclasses import dataclass

try:
    from app.distributed.circuit_breaker import (
        CircuitBreakerRegistry,
        CircuitState,
        get_adaptive_timeout,
    )

    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitBreakerRegistry = None  # type: ignore[assignment]
    CircuitState = None  # type: ignore[assignment]

    def get_adaptive_timeout(operation_type: str, host: str, default: float) -> float:
        return default

try:
    from app.config.coordination_defaults import build_ssh_options

    SSH_CONFIG_AVAILABLE = True
except ImportError:
    SSH_CONFIG_AVAILABLE = False
    build_ssh_options = None  # type: ignore[assignment]

from app.coordination.enums import DistributionDataType

DataType = DistributionDataType

REMOTE_PATH_PATTERNS: list[str] = [
    "/workspace/ringrift/ai-service",
    "~/ringrift/ai-service",
    "/root/ringrift/ai-service",
    "~/Development/RingRift/ai-service",
]

_remote_path_cache: dict[str, str] = {}
_remote_path_cache_lock = threading.Lock()


@dataclass
class DeliveryResult:
    """Result of delivering data to a single node."""

    node_id: str
    host: str
    data_path: str
    data_type: DataType
    success: bool
    checksum_verified: bool
    transfer_time_seconds: float
    error_message: str = ""
    method: str = "http"


__all__ = [
    "CIRCUIT_BREAKER_AVAILABLE",
    "CircuitBreakerRegistry",
    "CircuitState",
    "DataType",
    "DeliveryResult",
    "REMOTE_PATH_PATTERNS",
    "SSH_CONFIG_AVAILABLE",
    "_remote_path_cache",
    "_remote_path_cache_lock",
    "build_ssh_options",
    "get_adaptive_timeout",
]
