"""Query builders for P2P orchestrator.

Phase 3.2 Code Quality Cleanup: Query method consolidation.

This module provides generic query infrastructure to consolidate
the 70+ _get_* methods in p2p_orchestrator.py into reusable patterns:

- BaseQueryBuilder: Thread-safe filtering, mapping, aggregation
- PeerQueryBuilder: Peer-specific queries (alive, healthy, by_role)
- JobQueryBuilder: Job/task queries (active, completed, by_status)
- MetricsBuilder: Summary aggregations for API responses

Usage:
    from scripts.p2p.query_builders import PeerQueryBuilder

    builder = PeerQueryBuilder(orchestrator.peers, orchestrator.peers_lock)
    alive_peers = builder.alive()
    summary = builder.summary()
"""

from scripts.p2p.query_builders.base_query_builder import (
    BaseQueryBuilder,
    QueryResult,
    SummaryResult,
)
from scripts.p2p.query_builders.peer_query_builder import PeerQueryBuilder

__all__ = [
    "BaseQueryBuilder",
    "QueryResult",
    "SummaryResult",
    "PeerQueryBuilder",
]
