"""Tests for board_encoding_contract.py."""

from __future__ import annotations

from app.models import BoardType
from app.training.board_encoding_contract import get_expected_channels
from app.training.encoder_registry import get_expected_channels as get_registry_expected_channels


class TestBoardEncodingContract:
    """Keep the training contract table aligned with the live encoder registry."""

    def test_hex_v5_heavy_family_uses_64_channels(self):
        """Hex heavy families must stay aligned with the actual heavy encoder width."""
        for board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
            for version in (
                "v5",
                "v5-gnn",
                "v5-heavy",
                "v5-heavy-large",
                "v5-heavy-xl",
                "v6",
                "v6-xl",
            ):
                assert get_expected_channels(board_type, version) == 64

    def test_hex_heavy_aliases_match_encoder_registry(self):
        """Contract aliases should agree with the canonical encoder registry."""
        registry_version_map = {
            "v5": "v5-heavy",
            "v5-gnn": "v5-heavy",
            "v5-heavy": "v5-heavy",
            "v5-heavy-large": "v5-heavy-large",
            "v5-heavy-xl": "v5-heavy-xl",
            "v6": "v6",
            "v6-xl": "v6-xl",
        }
        for board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
            for contract_version, registry_version in registry_version_map.items():
                assert get_expected_channels(board_type, contract_version) == get_registry_expected_channels(
                    board_type,
                    registry_version,
                )

    def test_square_heavy_family_stays_56_channels(self):
        """Square heavy families must keep the 14x4 encoder contract."""
        for board_type in (BoardType.SQUARE8, BoardType.SQUARE19):
            for version in (
                "v5",
                "v5-gnn",
                "v5-heavy",
                "v5-heavy-large",
                "v5-heavy-xl",
                "v6",
                "v6-xl",
            ):
                assert get_expected_channels(board_type, version) == 56
