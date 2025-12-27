"""P2P backend coordination (December 2025).

.. deprecated:: December 2025
    Import directly from app.coordination.p2p_backend instead.

Re-exports from p2p_backend.py for unified access.

Usage:
    # DEPRECATED:
    from app.coordination.cluster.p2p import P2PBackend

    # RECOMMENDED:
    from app.coordination.p2p_backend import P2PBackend
"""

import warnings

warnings.warn(
    "app.coordination.cluster.p2p is deprecated. "
    "Import directly from app.coordination.p2p_backend instead.",
    DeprecationWarning,
    stacklevel=2,
)

from app.coordination.p2p_backend import (
    P2PBackend,
    P2PNodeInfo,
    get_p2p_backend,
)

__all__ = [
    "P2PBackend",
    "P2PNodeInfo",
    "get_p2p_backend",
]
