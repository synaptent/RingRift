"""Tests for v5-heavy bootstrap compatibility.

The bootstrap script and the minimal AlphaZero loop must agree on the
state-encoder channel count.  Previously they disagreed — bootstrap
built a 40-channel (v2 encoder) model while the minimal loop's JSONL→NPZ
export used the v3 encoder (64 channels for hex, 56 for square).  That
meant iter 1 self-play could complete but the training step would
immediately fail with a shape mismatch at the first convolution,
wasting ~5h of GPU before the circuit breaker tripped.

These tests lock in the corrected alignment.
"""

from __future__ import annotations

import pytest

from scripts.bootstrap_v5_heavy import _create_v5_heavy_model, _v5_heavy_in_channels


class TestV5HeavyBootstrapChannels:
    """Bootstrap checkpoints must match what the minimal loop will export."""

    def test_hex8_bootstrap_has_64_input_channels(self) -> None:
        """hex boards use the v3 encoder: 16 base × 4 frames = 64."""
        model = _create_v5_heavy_model("hex8", 2)
        assert model.conv1.weight.shape[1] == 64

    def test_hexagonal_bootstrap_has_64_input_channels(self) -> None:
        model = _create_v5_heavy_model("hexagonal", 2)
        assert model.conv1.weight.shape[1] == 64

    def test_square8_bootstrap_has_56_input_channels(self) -> None:
        """square boards use 14 base × 4 frames = 56."""
        model = _create_v5_heavy_model("square8", 2)
        assert model.conv1.weight.shape[1] == 56

    def test_square19_bootstrap_has_56_input_channels(self) -> None:
        model = _create_v5_heavy_model("square19", 2)
        assert model.conv1.weight.shape[1] == 56


class TestV5HeavyInChannelsHelper:
    """_v5_heavy_in_channels is the single source of truth. Regression-guard
    the specific channel counts so a future edit to one side without the
    other is caught at test time."""

    @pytest.mark.parametrize(
        "board_type,expected",
        [
            ("hex8", 64),
            ("hexagonal", 64),
            ("square8", 56),
            ("square19", 56),
        ],
    )
    def test_channel_counts_are_stable(self, board_type, expected):
        assert _v5_heavy_in_channels(board_type) == expected


class TestBootstrapLoopAlignment:
    """The bootstrap channel count must match what jsonl_to_npz will emit
    for the same board under ``--encoder-version v3``. If this test fails,
    one side moved without the other and iter 1 training will crash."""

    @pytest.mark.parametrize(
        "board_type,num_players",
        [
            ("hex8", 2),
            ("hexagonal", 2),
            ("square8", 2),
            ("square19", 2),
        ],
    )
    def test_bootstrap_conv1_matches_v3_encoder_export(self, board_type, num_players):
        """For v5-heavy, bootstrap and minimal-loop-export must agree on
        total input-channel count."""
        model = _create_v5_heavy_model(board_type, num_players)
        bootstrap_channels = model.conv1.weight.shape[1]

        # Reconstruct what jsonl_to_npz would emit with encoder_version='v3'.
        # Source: scripts/jsonl_to_npz.py _generate_encoder_metadata.
        if board_type.startswith("hex") or board_type == "hexagonal":
            expected_export_channels = 16 * 4  # hex_v3 base × frames
        else:
            expected_export_channels = 14 * 4  # square v3 base × frames

        assert bootstrap_channels == expected_export_channels, (
            f"Bootstrap {board_type} has {bootstrap_channels} input channels "
            f"but minimal_alphazero_loop.py with --encoder-version v3 will "
            f"export {expected_export_channels}. iter 1 training would crash."
        )
