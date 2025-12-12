from __future__ import annotations

import pytest

from scripts.run_improvement_loop import _promotion_gate


def test_promotion_gate_requires_wilson_lower_bound() -> None:
    """High observed win rate should still fail without confidence."""
    result = _promotion_gate(
        wins=8,
        losses=2,
        draws=0,
        threshold=0.7,
        confidence=0.95,
    )
    assert result["win_rate"] == pytest.approx(0.8)
    assert result["win_rate_ci_low"] < 0.7
    assert result["promote"] is False


def test_promotion_gate_passes_when_ci_clears_threshold() -> None:
    """Promotion should pass once Wilson CI lower bound clears threshold."""
    result = _promotion_gate(
        wins=65,
        losses=35,
        draws=0,
        threshold=0.55,
        confidence=0.95,
    )
    assert result["win_rate"] == pytest.approx(0.65)
    assert result["win_rate_ci_low"] >= 0.55
    assert result["promote"] is True

