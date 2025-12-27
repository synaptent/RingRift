"""Queue populator module - DEPRECATED.

NOTE (December 2025): This module is DEPRECATED and re-exports from
`app.coordination.unified_queue_populator`. For new code, import directly from:

    from app.coordination.unified_queue_populator import (
        UnifiedQueuePopulator,
        QueuePopulatorConfig,
        get_queue_populator,
        # ...
    )

This re-export module maintains backward compatibility for existing code
that imports from `app.coordination.queue_populator`.

Class aliases:
    PopulatorConfig -> QueuePopulatorConfig
    QueuePopulator -> UnifiedQueuePopulator
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "app.coordination.queue_populator is deprecated. "
    "Use app.coordination.unified_queue_populator instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from unified module
from app.coordination.unified_queue_populator import (
    # Constants
    LARGE_BOARDS,
    # Enums
    BoardType,
    # Data classes
    ConfigTarget,
    QueuePopulatorConfig,
    UnifiedQueuePopulator,
    UnifiedQueuePopulatorDaemon,
    # Factory functions
    get_queue_populator,
    get_queue_populator_daemon,
    reset_queue_populator,
    wire_queue_populator_events,
    load_populator_config_from_yaml,
)

# Backward-compatible aliases (December 2025)
PopulatorConfig = QueuePopulatorConfig  # Alias for backward compatibility
QueuePopulator = UnifiedQueuePopulator  # Alias for backward compatibility

__all__ = [
    # Constants
    "LARGE_BOARDS",
    # Enums
    "BoardType",
    # Data classes - with aliases
    "ConfigTarget",
    "PopulatorConfig",  # Alias for QueuePopulatorConfig
    "QueuePopulator",   # Alias for UnifiedQueuePopulator
    "QueuePopulatorConfig",
    "UnifiedQueuePopulator",
    "UnifiedQueuePopulatorDaemon",
    # Factory functions
    "get_queue_populator",
    "get_queue_populator_daemon",
    "reset_queue_populator",
    "wire_queue_populator_events",
    "load_populator_config_from_yaml",
]
