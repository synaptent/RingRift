"""Cross-module contract matrix test for channel counts.

Verifies that ALL channel-aware modules agree on the expected input channels
for every supported (board_type, model_version) pair. A single drift between
any two modules causes silent training corruption, so this test catches it
at CI time.

Modules checked:
  1. board_encoding_contract.py  (canonical source of truth)
  2. encoder_registry.py         (used by training data inference)
  3. architecture_registry.py    (used by MCTS / inference)
  4. train_model_factory.py      (implicit — model uses encoding_channels param)

April 2026: Created to eliminate the last known channel-inference bug family.
"""
from __future__ import annotations

import pytest
from app.models import BoardType
from app.training.board_encoding_contract import (
    _CONTRACTS,
    get_expected_channels as contract_get_channels,
)
from app.training.encoder_registry import (
    get_encoder_config,
    get_expected_channels as registry_get_channels,
)


# All (board_type, model_version) pairs registered in the encoding contract
def _all_contract_pairs() -> list[tuple[BoardType, str]]:
    """Return every (board_type, model_version) with a contract entry."""
    return sorted(_CONTRACTS.keys(), key=lambda x: (x[0].name, x[1]))


class TestCrossModuleChannelAgreement:
    """Every registered (board_type, model_version) must produce identical
    channel counts across board_encoding_contract and encoder_registry."""

    @pytest.mark.parametrize(
        "board_type,model_version",
        _all_contract_pairs(),
        ids=[f"{bt.name}_{mv}" for bt, mv in _all_contract_pairs()],
    )
    def test_contract_matches_encoder_registry(
        self, board_type: BoardType, model_version: str
    ) -> None:
        """board_encoding_contract and encoder_registry must agree."""
        contract_channels = contract_get_channels(board_type, model_version)

        # encoder_registry uses board_type.name (uppercase)
        try:
            registry_channels = registry_get_channels(
                board_type.name, model_version
            )
        except ValueError:
            # Some contract entries may not have a registry counterpart
            # (e.g., square v5-heavy-large). That is acceptable — the
            # contract is the superset.
            pytest.skip(
                f"encoder_registry has no entry for "
                f"{board_type.name}/{model_version}"
            )
            return

        assert contract_channels == registry_channels, (
            f"Channel MISMATCH for {board_type.name}/{model_version}: "
            f"contract={contract_channels}, encoder_registry={registry_channels}"
        )


class TestArchitectureRegistryConsistency:
    """architecture_registry channel->spec mapping must not contradict
    the encoding contract."""

    def test_architecture_registry_standard_channels_are_valid(self) -> None:
        """Standard channel counts (40, 56, 64) in architecture_registry must
        match at least one encoding contract entry. Lite variants (36, 44) are
        specialized and not covered by the standard contract."""
        from app.ai.neural_net.architecture_registry import ARCHITECTURE_REGISTRY

        # Lite variants are not part of the standard training pipeline
        _LITE_CHANNELS = {36, 44}

        for channels, spec in ARCHITECTURE_REGISTRY.items():
            if channels in _LITE_CHANNELS:
                continue
            # Find at least one contract that produces this channel count
            found = any(
                c.expected_in_channels == channels for c in _CONTRACTS.values()
            )
            assert found, (
                f"architecture_registry has {channels} channels "
                f"(spec={spec.description}) but no encoding contract "
                f"produces {channels} channels"
            )

    def test_40ch_is_v2(self) -> None:
        """40 channels must map to v2 in both registries."""
        from app.ai.neural_net.architecture_registry import (
            ARCHITECTURE_REGISTRY,
            ArchitectureVersion,
        )

        spec = ARCHITECTURE_REGISTRY[40]
        assert spec.version == ArchitectureVersion.V2

        for bt in (BoardType.HEX8, BoardType.HEXAGONAL):
            assert contract_get_channels(bt, "v2") == 40

    def test_64ch_is_v3_v4_family(self) -> None:
        """64 channels must map to v3/v4 (or v5-heavy) in both registries."""
        from app.ai.neural_net.architecture_registry import (
            ARCHITECTURE_REGISTRY,
            ArchitectureVersion,
        )

        spec = ARCHITECTURE_REGISTRY[64]
        assert spec.version in (
            ArchitectureVersion.V3,
            ArchitectureVersion.V4,
        )

        # Contract: hex v3, v4, and v5-heavy all produce 64ch
        for bt in (BoardType.HEX8, BoardType.HEXAGONAL):
            assert contract_get_channels(bt, "v3") == 64
            assert contract_get_channels(bt, "v4") == 64
            assert contract_get_channels(bt, "v5-heavy") == 64

    def test_56ch_is_square_v2(self) -> None:
        """56 channels must map to the square encoder family, not hex v5-heavy."""
        from app.ai.neural_net.architecture_registry import ARCHITECTURE_REGISTRY

        spec = ARCHITECTURE_REGISTRY[56]
        assert spec.encoder_name == "SquareStateEncoder"

        for bt in (BoardType.SQUARE8, BoardType.SQUARE19):
            assert contract_get_channels(bt, "v2") == 56

    def test_class_name_mapping_handles_hex_v5_heavy(self) -> None:
        """Class-name lookup must preserve the real hex heavy channel width."""
        from app.ai.neural_net.architecture_registry import get_architecture_from_class_name

        spec = get_architecture_from_class_name("HexNeuralNet_v5_Heavy")
        assert spec is not None
        assert spec.expected_channels == 64
        assert spec.encoder_name == "HexStateEncoderV3"

    def test_class_name_mapping_handles_square_cnn(self) -> None:
        """Square CNN class lookup must not borrow the hex heavy spec."""
        from app.ai.neural_net.architecture_registry import get_architecture_from_class_name

        spec = get_architecture_from_class_name("RingRiftCNN_v4")
        assert spec is not None
        assert spec.expected_channels == 56
        assert spec.encoder_name == "SquareStateEncoder"


class TestDetectModelVersionAmbiguity:
    """detect_model_version_from_channels must resolve the 64ch hex
    ambiguity correctly based on metadata."""

    def test_64ch_defaults_to_v4_without_metadata(self) -> None:
        from app.training.encoder_registry import detect_model_version_from_channels

        result = detect_model_version_from_channels(64, "hex8")
        assert result == "v4", "64ch without metadata should default to v4"

    def test_64ch_with_heuristic_metadata_returns_v5_heavy(self) -> None:
        from app.training.encoder_registry import detect_model_version_from_channels

        result = detect_model_version_from_channels(
            64, "hex8", has_heuristic_metadata=True
        )
        assert result == "v5-heavy", "64ch with heuristic metadata should be v5-heavy"

    def test_64ch_with_encoder_type_v5_heavy(self) -> None:
        from app.training.encoder_registry import detect_model_version_from_channels

        result = detect_model_version_from_channels(
            64, "hex8", npz_encoder_type="hex_v5_heavy"
        )
        assert result == "v5-heavy"

    def test_64ch_with_encoder_type_hex_v3(self) -> None:
        from app.training.encoder_registry import detect_model_version_from_channels

        result = detect_model_version_from_channels(
            64, "hex8", npz_encoder_type="hex_v3"
        )
        assert result == "v4"  # v3 and v4 share the same encoder

    def test_40ch_unambiguous_regardless_of_metadata(self) -> None:
        from app.training.encoder_registry import detect_model_version_from_channels

        assert detect_model_version_from_channels(40) == "v2"
        assert detect_model_version_from_channels(
            40, has_heuristic_metadata=True
        ) == "v2"

    def test_56ch_square_unambiguous(self) -> None:
        from app.training.encoder_registry import detect_model_version_from_channels

        assert detect_model_version_from_channels(56, "square8") == "v2"
        assert detect_model_version_from_channels(56, "square19") == "v2"


class TestContractCompleteness:
    """The contract must cover all board types and common model versions."""

    @pytest.mark.parametrize("board_type", list(BoardType))
    def test_every_board_type_has_v2_entry(self, board_type: BoardType) -> None:
        channels = contract_get_channels(board_type, "v2")
        assert channels > 0

    def test_hex_boards_have_v3_v4_v5_entries(self) -> None:
        for bt in (BoardType.HEX8, BoardType.HEXAGONAL):
            for mv in ("v3", "v4", "v5-heavy", "v5-heavy-large"):
                channels = contract_get_channels(bt, mv)
                assert channels == 64, (
                    f"{bt.name}/{mv} should be 64ch, got {channels}"
                )

    def test_square_boards_always_56ch(self) -> None:
        for bt in (BoardType.SQUARE8, BoardType.SQUARE19):
            for mv in ("v2", "v3", "v4", "v5-heavy"):
                channels = contract_get_channels(bt, mv)
                assert channels == 56, (
                    f"{bt.name}/{mv} should be 56ch, got {channels}"
                )
