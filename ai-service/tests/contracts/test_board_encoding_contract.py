"""Contract tests for board encoding — prevents the 200+ GPU-hour mismatch bug class.

Each test corresponds to a specific manifestation of the encoding mismatch:
1. All 12 board configs have a contract entry
2. Channel counts are consistent (no hardcoded 40/56/64 disagreements)
3. Square boards get 56 channels (not 40 like hex)
4. Contract lookup works for every board/version combination
5. Invalid combinations fail fast with clear errors
"""
from __future__ import annotations

import pytest
from app.models import BoardType
from app.training.board_encoding_contract import (
    BoardEncodingContract,
    get_encoding_contract,
    get_expected_channels,
    is_valid_channel_count,
    infer_model_version_from_channels,
)


class TestContractCompleteness:
    """Every board type must have at least a v2 contract."""

    @pytest.mark.parametrize("board_type", list(BoardType))
    def test_all_board_types_have_v2_contract(self, board_type):
        contract = get_encoding_contract(board_type, "v2")
        assert contract.expected_in_channels > 0

    def test_all_12_configs_have_contracts(self):
        for bt in BoardType:
            contract = get_encoding_contract(bt, "v2")
            assert isinstance(contract, BoardEncodingContract)


class TestChannelConsistency:
    """Channel counts must match known encodings."""

    def test_hex_v2_is_40_channels(self):
        assert get_expected_channels(BoardType.HEX8, "v2") == 40
        assert get_expected_channels(BoardType.HEXAGONAL, "v2") == 40

    def test_square_v2_is_56_channels(self):
        """This was the root cause of the encoding mismatch — square != hex."""
        assert get_expected_channels(BoardType.SQUARE8, "v2") == 56
        assert get_expected_channels(BoardType.SQUARE19, "v2") == 56

    def test_square_channels_differ_from_hex(self):
        """Guard against someone 'fixing' square to match hex."""
        hex_ch = get_expected_channels(BoardType.HEX8, "v2")
        sq_ch = get_expected_channels(BoardType.SQUARE8, "v2")
        assert hex_ch != sq_ch, "Square and hex must have different channel counts"

    def test_known_valid_channel_counts(self):
        assert is_valid_channel_count(40)   # hex v2
        assert is_valid_channel_count(56)   # square v2, hex v5
        assert is_valid_channel_count(64)   # hex v3/v4
        assert not is_valid_channel_count(0)
        assert not is_valid_channel_count(48)


class TestInvalidCombinations:
    """Invalid board/model combinations must fail fast."""

    def test_unknown_model_version_raises(self):
        with pytest.raises(ValueError, match="No encoding contract"):
            get_encoding_contract(BoardType.HEX8, "v99")

    def test_error_message_lists_available_versions(self):
        with pytest.raises(ValueError, match="Available"):
            get_encoding_contract(BoardType.HEX8, "nonexistent")


class TestInference:
    """Model version inference from channels must be correct."""

    def test_infer_v2_from_40_channels_hex(self):
        assert infer_model_version_from_channels(40, BoardType.HEX8) == "v2"

    def test_infer_v2_from_56_channels_square(self):
        assert infer_model_version_from_channels(56, BoardType.SQUARE8) == "v2"

    def test_infer_v4_from_64_channels_hex(self):
        assert infer_model_version_from_channels(64, BoardType.HEX8) in ("v3", "v4")
