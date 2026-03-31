"""Focused tests for election handler safety checks."""

from __future__ import annotations

from scripts.p2p.handlers.election import ElectionHandlersMixin


class _ElectionTestHarness(ElectionHandlersMixin):
    """Minimal harness for probation-check helper testing."""

    def __init__(self):
        self.peers = {}

    def _get_voter_config_version(self) -> int:
        return 10


def test_probation_check_rejects_on_internal_error():
    """Errors during probation checks must fail closed."""
    harness = _ElectionTestHarness()

    def _boom() -> int:
        raise RuntimeError("config unavailable")

    harness._get_voter_config_version = _boom  # type: ignore[method-assign]

    allowed, reason = harness._check_voter_config_probation("leader-1", {})

    assert allowed is False
    assert "check_error" in reason
