"""Tests for v5-heavy bootstrap compatibility."""

from __future__ import annotations

from scripts.bootstrap_v5_heavy import _create_v5_heavy_model


def test_bootstrap_v5_heavy_matches_runtime_hex_channels() -> None:
    """Bootstrap checkpoints must match the runtime v5-heavy loader."""
    model = _create_v5_heavy_model("hex8", 2)

    assert model.conv1.weight.shape[1] == 40
