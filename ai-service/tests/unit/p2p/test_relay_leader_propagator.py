"""Focused tests for relay leader propagation safety checks."""

from __future__ import annotations

from scripts.p2p.relay_leader_propagator import RelayLeaderPropagatorMixin


class _RelayHarness(RelayLeaderPropagatorMixin):
    """Minimal harness for leader reachability testing."""

    def __init__(self):
        self.node_id = "node-self"
        self.peers = {}
        self.peers_lock = None


def test_verify_leader_reachable_fails_closed_without_peers():
    """Missing peer inventory must deny leader adoption."""
    harness = _RelayHarness()
    harness.peers = None

    assert harness._verify_leader_reachable("leader-1") is False


def test_verify_leader_reachable_fails_closed_when_liveness_unknown():
    """Unknown peer liveness must deny leader adoption."""
    harness = _RelayHarness()
    harness.peers = {"leader-1": object()}

    assert harness._verify_leader_reachable("leader-1") is False
