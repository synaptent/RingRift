"""Backward-compatibility re-export for queue_populator module.

This module was replaced by unified_queue_populator.py.
Use the new module directly:

    from app.coordination.unified_queue_populator import (
        QueuePopulatorConfig,  # Was: PopulatorConfig
        UnifiedQueuePopulator,  # Was: QueuePopulator
        ConfigTarget,
        get_queue_populator,
        reset_queue_populator,
        load_populator_config_from_yaml,
    )

This shim will be removed in Q2 2026.
"""

import warnings

warnings.warn(
    "app.coordination.queue_populator is deprecated. "
    "Use app.coordination.unified_queue_populator instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from unified module with backward-compat aliases
from app.coordination.unified_queue_populator import (
    LARGE_BOARDS,
    ConfigTarget,
    QueuePopulatorConfig,
    UnifiedQueuePopulator,
    get_queue_populator,
    load_populator_config_from_yaml,
    reset_queue_populator,
    wire_queue_populator_events,
)

# Re-export BoardType from types module
from app.coordination.types import BoardType

# Backward-compat aliases
PopulatorConfig = QueuePopulatorConfig
QueuePopulator = UnifiedQueuePopulator

__all__ = [
    "LARGE_BOARDS",
    "BoardType",
    "ConfigTarget",
    "PopulatorConfig",
    "QueuePopulatorConfig",
    "QueuePopulator",
    "UnifiedQueuePopulator",
    "get_queue_populator",
    "load_populator_config_from_yaml",
    "reset_queue_populator",
    "wire_queue_populator_events",
]
