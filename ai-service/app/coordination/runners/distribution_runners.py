"""Distribution and Replication daemon runners.

February 2026: Extracted from daemon_runners.py.

Contains runners for:
- Distribution Daemons (model_sync, model_distribution, npz_distribution, data_server)
- Replication Daemons (replication_monitor, replication_repair)
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from app.coordination.runners._shared import wait_for_daemon

logger = logging.getLogger(__name__)


# =============================================================================
# Distribution Daemons
# =============================================================================


async def create_model_sync() -> None:
    """Create and run model sync daemon (December 2025)."""
    try:
        from app.coordination.unified_distribution_daemon import (
            create_model_distribution_daemon,
        )

        daemon = create_model_distribution_daemon()
        await daemon.start()
        await wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"Model sync daemon not available: {e}")
        raise


async def create_model_distribution() -> None:
    """Create and run model distribution daemon (December 2025)."""
    try:
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()
        await daemon.start()
        await wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedDistributionDaemon not available: {e}")
        raise


async def create_npz_distribution() -> None:
    """Deprecated. This daemon type has been removed.

    No ImportError path is needed because this runner is now a no-op.
    """
    logger.debug("Deprecated daemon runner called (no-op): NPZ_DISTRIBUTION")
    return


async def create_data_server() -> None:
    """Create and run data server daemon."""
    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        sync = SyncCoordinator.get_instance()
        # Jan 3, 2026: Fixed method name - SyncCoordinator has start_data_server(), not start_server()
        await sync.start_data_server()
        await wait_for_daemon(sync)
    except ImportError as e:
        logger.error(f"SyncCoordinator data server not available: {e}")
        raise


# =============================================================================
# Replication Daemons
# =============================================================================


async def create_replication_monitor() -> None:
    """Deprecated. This daemon type has been removed.

    No ImportError path is needed because this runner is now a no-op.
    """
    logger.debug("Deprecated daemon runner called (no-op): REPLICATION_MONITOR")
    return


async def create_replication_repair() -> None:
    """Deprecated. This daemon type has been removed.

    No ImportError path is needed because this runner is now a no-op.
    """
    logger.debug("Deprecated daemon runner called (no-op): REPLICATION_REPAIR")
    return
