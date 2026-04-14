"""Hashgraph consensus library for RingRift.

This module implements Hashgraph-inspired consensus mechanisms for:
- Multi-model evaluation consensus (Byzantine-tolerant Elo updates)
- Model promotion voting (BFT approval)
- Event ordering via gossip-about-gossip

Key Concepts:
- Gossip About Gossip: Each event includes parent hashes creating a DAG
- Virtual Voting: Nodes compute votes from DAG ancestry without messages
- Famous Witnesses: Events seen by 2/3+ of later witnesses achieve consensus

Usage:
    from app.coordination.hashgraph import (
        HashgraphEvent,
        HashgraphDAG,
        ConsensusEngine,
        EvaluationConsensus,
    )

    # Create event with parent references
    event = HashgraphEvent.create(
        creator="node-1",
        payload={"type": "evaluation", "model": "v5", "win_rate": 0.85},
        self_parent=my_last_event_hash,
        other_parent=received_event_hash,
    )

    # Add to DAG
    dag = HashgraphDAG()
    dag.add_event(event)

    # Get consensus ordering
    ordered_events = dag.get_consensus_order()

January 2026: Initial implementation for model evaluation consensus.
"""

from app.coordination.hashgraph.event import (
    HashgraphEvent,
    EventType,
    canonical_json,
    compute_event_hash,
)
from app.coordination.hashgraph.dag import (
    HashgraphDAG,
    DAGNode,
    AncestryResult,
)
from app.coordination.hashgraph.consensus import (
    ConsensusEngine,
    ConsensusResult,
    VirtualVote,
    StronglySeeingResult,
)
from app.coordination.hashgraph.famous_witnesses import (
    WitnessSelector,
    WitnessInfo,
    FameStatus,
    RoundInfo,
)
from app.coordination.hashgraph.evaluation_consensus import (
    EvaluationConsensusManager,
    EvaluationConsensusConfig,
    EvaluationResult,
    ConsensusEvaluationResult,
    EvaluationEventType,
    get_evaluation_consensus_manager,
    reset_evaluation_consensus_manager,
)
from app.coordination.hashgraph.gossip_ancestry import (
    GossipAncestryTracker,
    GossipAncestryConfig,
    AncestryEvent,
    ValidationResult,
    ValidationStatus,
    add_ancestry_to_payload,
    validate_ancestry,
    has_ancestry_fields,
    get_gossip_ancestry_tracker,
    reset_gossip_ancestry_tracker,
)
from app.coordination.hashgraph.promotion_consensus import (
    PromotionConsensusManager,
    PromotionConsensusConfig,
    PromotionProposal,
    PromotionVote,
    PromotionCertificate,
    PromotionConsensusResult,
    EvaluationEvidence,
    PromotionEventType,
    VoteType,
    get_promotion_consensus_manager,
    reset_promotion_consensus_manager,
)

__all__ = [
    # Event
    "HashgraphEvent",
    "EventType",
    "canonical_json",
    "compute_event_hash",
    # DAG
    "HashgraphDAG",
    "DAGNode",
    "AncestryResult",
    # Consensus
    "ConsensusEngine",
    "ConsensusResult",
    "VirtualVote",
    "StronglySeeingResult",
    # Witnesses
    "WitnessSelector",
    "WitnessInfo",
    "FameStatus",
    "RoundInfo",
    # Evaluation Consensus
    "EvaluationConsensusManager",
    "EvaluationConsensusConfig",
    "EvaluationResult",
    "ConsensusEvaluationResult",
    "EvaluationEventType",
    "get_evaluation_consensus_manager",
    "reset_evaluation_consensus_manager",
    # Gossip Ancestry
    "GossipAncestryTracker",
    "GossipAncestryConfig",
    "AncestryEvent",
    "ValidationResult",
    "ValidationStatus",
    "add_ancestry_to_payload",
    "validate_ancestry",
    "has_ancestry_fields",
    "get_gossip_ancestry_tracker",
    "reset_gossip_ancestry_tracker",
    # Promotion Consensus
    "PromotionConsensusManager",
    "PromotionConsensusConfig",
    "PromotionProposal",
    "PromotionVote",
    "PromotionCertificate",
    "PromotionConsensusResult",
    "EvaluationEvidence",
    "PromotionEventType",
    "VoteType",
    "get_promotion_consensus_manager",
    "reset_promotion_consensus_manager",
]


def __dir__() -> list[str]:
    """Expose the intended package surface for discoverability and tests."""

    return sorted(set(globals()) | set(__all__))
