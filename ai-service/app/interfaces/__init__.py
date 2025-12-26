"""Shared interfaces and protocols for RingRift AI service.

This package defines Protocol classes and Abstract Base Classes that help
break circular dependencies between modules. Implementations live in their
respective packages, but depend on these interfaces.

Architecture:
    app.interfaces (this package) - no dependencies except typing/abc
    └── Used by: app.ai, app.game_engine, app.training, app.models

Protocols defined:
    - HashProvider: Interface for Zobrist-style position hashing
    - MoveCacheProvider: Interface for move caching
    - ModelProvider: Interface for loading AI models
    - EncodingProvider: Interface for move encoding/decoding
    - EventEmitter: Interface for event emission (Dec 2025)
    - EventSubscriber: Interface for event subscription (Dec 2025)
    - DaemonLifecycle: Interface for daemon management (Dec 2025)
"""

from .protocols import (
    DaemonLifecycle,
    EncodingProvider,
    EventEmitter,
    EventHandler,
    EventSubscriber,
    HashProvider,
    ModelProvider,
    MoveCacheProvider,
)

__all__ = [
    "DaemonLifecycle",
    "EncodingProvider",
    "EventEmitter",
    "EventHandler",
    "EventSubscriber",
    "HashProvider",
    "ModelProvider",
    "MoveCacheProvider",
]
