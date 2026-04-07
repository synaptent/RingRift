"""Tests for scripts/export_for_all_architectures.py."""

from __future__ import annotations


def test_v5_heavy_expected_channels_are_board_aware():
    """Hex and square heavy exports must not share one hardcoded width."""
    from scripts.export_for_all_architectures import (
        get_expected_channels_for_architecture,
    )

    assert get_expected_channels_for_architecture("hex8", "v5-heavy") == 64
    assert get_expected_channels_for_architecture("hexagonal", "v5-heavy") == 64
    assert get_expected_channels_for_architecture("square8", "v5-heavy") == 56
    assert get_expected_channels_for_architecture("square19", "v5-heavy") == 56


def test_v4_expected_channels_remain_board_aware():
    """Standard encoder families should still resolve through the same helper."""
    from scripts.export_for_all_architectures import (
        get_expected_channels_for_architecture,
    )

    assert get_expected_channels_for_architecture("hex8", "v4") == 64
    assert get_expected_channels_for_architecture("square8", "v4") == 56
