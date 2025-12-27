"""Queue populator module - DEPRECATED.

NOTE (December 2025): This module is DEPRECATED and re-exports from
`app.coordination.unified_queue_populator`. For new code, import directly from:

    from app.coordination.unified_queue_populator import (
        UnifiedQueuePopulator,
        QueuePopulatorConfig,
        get_queue_populator,
        # ...
    )

This re-export module maintains backward compatibility for existing code.
"""

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
    BOARD_CONFIGS,
    DEFAULT_CURRICULUM_WEIGHTS,
    LARGE_BOARDS,
    # Enums
    BoardType,
    # Data classes - with aliases
    ConfigTarget,
    PopulatorConfig,
    QueuePopulator,
    QueuePopulatorConfig,
    # Main classes
    UnifiedQueuePopulator,
    UnifiedQueuePopulatorDaemon,
    # Singleton functions
    get_queue_populator,
    get_queue_populator_daemon,
    reset_queue_populator,
    start_queue_populator_daemon,
    # Utilities
    load_populator_config_from_yaml,
    wire_queue_populator_events,
)

__all__ = [
    # Constants
    "BOARD_CONFIGS",
    "LARGE_BOARDS",
    "DEFAULT_CURRICULUM_WEIGHTS",
    # Enums
    "BoardType",
    # Data classes
    "QueuePopulatorConfig",
    "PopulatorConfig",  # Alias
    "ConfigTarget",
    # Main classes
    "UnifiedQueuePopulator",
    "UnifiedQueuePopulatorDaemon",
    "QueuePopulator",  # Alias
    # Singleton functions
    "get_queue_populator",
    "get_queue_populator_daemon",
    "reset_queue_populator",
    "start_queue_populator_daemon",
    # Utilities
    "wire_queue_populator_events",
    "load_populator_config_from_yaml",
]
