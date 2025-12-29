"""Unit tests for DefaultRulesEngine.

Tests cover:
- DefaultRulesEngine initialization with validators and mutators
- Environment variable configuration handling
- validate_move() behavior for different move types
- get_valid_moves() behavior
- apply_move() with shadow contracts
- Helper methods (_describe_move, _diff_mapping_keys, _are_moves_equivalent)

Created: December 2025
"""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from app.rules.default_engine import DefaultRulesEngine
from app.rules.validators.placement import PlacementValidator
from app.rules.validators.movement import MovementValidator
from app.rules.validators.capture import CaptureValidator
from app.rules.validators.line import LineValidator
from app.rules.validators.territory import TerritoryValidator
from app.rules.mutators.placement import PlacementMutator
from app.rules.mutators.movement import MovementMutator
from app.rules.mutators.capture import CaptureMutator
from app.rules.mutators.line import LineMutator
from app.rules.mutators.territory import TerritoryMutator
from app.rules.mutators.turn import TurnMutator
from app.models import GamePhase, GameState, GameStatus, Move, MoveType


class TestDefaultRulesEngineInitialization:
    """Tests for DefaultRulesEngine initialization."""

    def test_default_initialization(self):
        """Test DefaultRulesEngine initializes with default configuration."""
        engine = DefaultRulesEngine()
        assert engine is not None

    def test_validators_list_created(self):
        """Test validators list is created with correct validators."""
        engine = DefaultRulesEngine()
        assert len(engine.validators) == 5
        assert isinstance(engine.validators[0], PlacementValidator)
        assert isinstance(engine.validators[1], MovementValidator)
        assert isinstance(engine.validators[2], CaptureValidator)
        assert isinstance(engine.validators[3], LineValidator)
        assert isinstance(engine.validators[4], TerritoryValidator)

    def test_mutators_list_created(self):
        """Test mutators list is created with correct mutators."""
        engine = DefaultRulesEngine()
        assert len(engine.mutators) == 6
        assert isinstance(engine.mutators[0], PlacementMutator)
        assert isinstance(engine.mutators[1], MovementMutator)
        assert isinstance(engine.mutators[2], CaptureMutator)
        assert isinstance(engine.mutators[3], LineMutator)
        assert isinstance(engine.mutators[4], TerritoryMutator)
        assert isinstance(engine.mutators[5], TurnMutator)

    def test_validators_have_validate_method(self):
        """Test all validators have a validate method (Validator interface)."""
        engine = DefaultRulesEngine()
        for validator in engine.validators:
            assert hasattr(validator, "validate")
            assert callable(validator.validate)

    def test_mutators_have_apply_method(self):
        """Test all mutators have an apply method (Mutator interface)."""
        engine = DefaultRulesEngine()
        for mutator in engine.mutators:
            assert hasattr(mutator, "apply")
            assert callable(mutator.apply)


class TestShadowContractsConfiguration:
    """Tests for shadow contracts configuration via env vars and constructor."""

    def test_skip_shadow_contracts_default_true(self):
        """Test skip_shadow_contracts defaults to True for performance."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var entirely
            os.environ.pop("RINGRIFT_SKIP_SHADOW_CONTRACTS", None)
            engine = DefaultRulesEngine()
            # Default env value is "true", so skip_shadow_contracts should be True
            assert engine._skip_shadow_contracts is True

    def test_skip_shadow_contracts_env_false(self):
        """Test skip_shadow_contracts respects env var set to false."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "false"}):
            engine = DefaultRulesEngine()
            assert engine._skip_shadow_contracts is False

    def test_skip_shadow_contracts_env_0(self):
        """Test skip_shadow_contracts respects env var set to 0."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "0"}):
            engine = DefaultRulesEngine()
            assert engine._skip_shadow_contracts is False

    def test_skip_shadow_contracts_env_no(self):
        """Test skip_shadow_contracts respects env var set to no."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "no"}):
            engine = DefaultRulesEngine()
            assert engine._skip_shadow_contracts is False

    def test_skip_shadow_contracts_env_off(self):
        """Test skip_shadow_contracts respects env var set to off."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "off"}):
            engine = DefaultRulesEngine()
            assert engine._skip_shadow_contracts is False

    def test_skip_shadow_contracts_constructor_override_true(self):
        """Test constructor argument overrides env var (True)."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "false"}):
            engine = DefaultRulesEngine(skip_shadow_contracts=True)
            assert engine._skip_shadow_contracts is True

    def test_skip_shadow_contracts_constructor_override_false(self):
        """Test constructor argument overrides env var (False)."""
        with patch.dict(os.environ, {"RINGRIFT_SKIP_SHADOW_CONTRACTS": "true"}):
            engine = DefaultRulesEngine(skip_shadow_contracts=False)
            assert engine._skip_shadow_contracts is False


class TestMutatorFirstConfiguration:
    """Tests for mutator-first mode configuration."""

    def test_mutator_first_default_not_set(self):
        """Test mutator_first is not enabled by default."""
        engine = DefaultRulesEngine()
        # _mutator_first_enabled is set based on constructor arg or env var
        # It may not exist as an attribute if not set, so check with getattr
        assert getattr(engine, "_mutator_first_enabled", False) is False

    def test_mutator_first_constructor_true_with_server_flag(self):
        """Test mutator_first can be enabled via constructor when server allows."""
        # Must set RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST to allow mutator-first
        with patch.dict(os.environ, {"RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST": "true"}):
            engine = DefaultRulesEngine(mutator_first=True)
            assert engine._mutator_first_enabled is True

    def test_mutator_first_constructor_blocked_without_server_flag(self):
        """Test mutator_first is blocked when server doesn't allow."""
        # Without RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST, mutator-first is hard-disabled
        with patch.dict(os.environ, {"RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST": ""}):
            engine = DefaultRulesEngine(mutator_first=True)
            assert engine._mutator_first_enabled is False

    def test_mutator_first_constructor_false(self):
        """Test mutator_first can be explicitly disabled via constructor."""
        engine = DefaultRulesEngine(mutator_first=False)
        assert engine._mutator_first_enabled is False


class TestValidateMoveMethods:
    """Tests for validate_move() method dispatching."""

    def test_validate_move_method_exists(self):
        """Test validate_move method exists."""
        engine = DefaultRulesEngine()
        assert hasattr(engine, "validate_move")
        assert callable(engine.validate_move)

    def test_no_placement_action_always_valid(self):
        """Test NO_PLACEMENT_ACTION is always valid (bookkeeping move)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.NO_PLACEMENT_ACTION

        result = engine.validate_move(mock_state, mock_move)
        assert result is True

    def test_no_line_action_always_valid(self):
        """Test NO_LINE_ACTION is always valid (bookkeeping move)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.NO_LINE_ACTION

        result = engine.validate_move(mock_state, mock_move)
        assert result is True

    def test_no_movement_action_always_valid(self):
        """Test NO_MOVEMENT_ACTION is always valid (bookkeeping move)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.NO_MOVEMENT_ACTION

        result = engine.validate_move(mock_state, mock_move)
        assert result is True

    def test_skip_capture_valid_in_capture_phase(self):
        """Test SKIP_CAPTURE is valid only in CAPTURE phase for current player."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_state.current_phase = GamePhase.CAPTURE
        mock_state.current_player = 1

        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.SKIP_CAPTURE
        mock_move.player = 1

        result = engine.validate_move(mock_state, mock_move)
        assert result is True

    def test_skip_capture_invalid_wrong_phase(self):
        """Test SKIP_CAPTURE is invalid in non-CAPTURE phases."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_state.current_phase = GamePhase.RING_PLACEMENT
        mock_state.current_player = 1

        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.SKIP_CAPTURE
        mock_move.player = 1

        result = engine.validate_move(mock_state, mock_move)
        assert result is False

    def test_skip_capture_invalid_wrong_player(self):
        """Test SKIP_CAPTURE is invalid for non-current player."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_state.current_phase = GamePhase.CAPTURE
        mock_state.current_player = 1

        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.SKIP_CAPTURE
        mock_move.player = 2  # Not the current player

        result = engine.validate_move(mock_state, mock_move)
        assert result is False

    def test_unknown_move_type_returns_false(self):
        """Test unknown move types return False."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        # Create a mock move type that doesn't match any dispatcher
        mock_move.type = MagicMock()
        mock_move.type.value = "UNKNOWN_MOVE"
        mock_move.player = 1

        result = engine.validate_move(mock_state, mock_move)
        assert result is False


class TestGetValidMovesMethods:
    """Tests for get_valid_moves() method."""

    def test_get_valid_moves_method_exists(self):
        """Test get_valid_moves method exists."""
        engine = DefaultRulesEngine()
        assert hasattr(engine, "get_valid_moves")
        assert callable(engine.get_valid_moves)


class TestApplyMoveMethods:
    """Tests for apply_move() method."""

    def test_apply_move_method_exists(self):
        """Test apply_move method exists."""
        engine = DefaultRulesEngine()
        assert hasattr(engine, "apply_move")
        assert callable(engine.apply_move)

    def test_apply_move_has_trace_mode_parameter(self):
        """Test apply_move accepts trace_mode parameter."""
        engine = DefaultRulesEngine()
        import inspect
        sig = inspect.signature(engine.apply_move)
        assert "trace_mode" in sig.parameters


class TestDescribeMoveHelper:
    """Tests for _describe_move static method."""

    def test_describe_move_method_exists(self):
        """Test _describe_move method exists."""
        assert hasattr(DefaultRulesEngine, "_describe_move")
        assert callable(DefaultRulesEngine._describe_move)

    def test_describe_move_basic(self):
        """Test _describe_move returns compact move description."""
        mock_move = MagicMock(spec=Move)
        mock_move.id = "move123"
        mock_move.type = MoveType.PLACE_RING
        mock_move.player = 1
        mock_move.from_pos = None
        mock_move.to = MagicMock()
        mock_move.to.to_key.return_value = "a1"

        result = DefaultRulesEngine._describe_move(mock_move)
        assert "move123" in result
        assert "PLACE_RING" in result
        assert "player=1" in result
        assert "to=a1" in result

    def test_describe_move_no_positions(self):
        """Test _describe_move handles moves without positions."""
        mock_move = MagicMock(spec=Move)
        mock_move.id = "move456"
        mock_move.type = MoveType.SKIP_PLACEMENT
        mock_move.player = 2
        mock_move.from_pos = None
        mock_move.to = None

        result = DefaultRulesEngine._describe_move(mock_move)
        assert "move456" in result
        assert "from=None" in result
        assert "to=None" in result

    def test_describe_move_with_from_and_to(self):
        """Test _describe_move handles moves with from and to positions."""
        mock_move = MagicMock(spec=Move)
        mock_move.id = "move789"
        mock_move.type = MoveType.MOVE_STACK
        mock_move.player = 1
        mock_move.from_pos = MagicMock()
        mock_move.from_pos.to_key.return_value = "b2"
        mock_move.to = MagicMock()
        mock_move.to.to_key.return_value = "c3"

        result = DefaultRulesEngine._describe_move(mock_move)
        assert "from=b2" in result
        assert "to=c3" in result


class TestDiffMappingKeysHelper:
    """Tests for _diff_mapping_keys static method."""

    def test_diff_mapping_keys_method_exists(self):
        """Test _diff_mapping_keys method exists."""
        assert hasattr(DefaultRulesEngine, "_diff_mapping_keys")
        assert callable(DefaultRulesEngine._diff_mapping_keys)

    def test_diff_mapping_keys_no_diff(self):
        """Test _diff_mapping_keys returns 'no_diff' when mappings are equal."""
        mut = {"a": 1, "b": 2}
        eng = {"a": 1, "b": 2}
        result = DefaultRulesEngine._diff_mapping_keys(mut, eng)
        assert result == "no_diff"

    def test_diff_mapping_keys_only_mut(self):
        """Test _diff_mapping_keys identifies keys only in mut."""
        mut = {"a": 1, "b": 2, "c": 3}
        eng = {"a": 1, "b": 2}
        result = DefaultRulesEngine._diff_mapping_keys(mut, eng)
        assert "only_mut" in result
        assert "'c'" in result

    def test_diff_mapping_keys_only_eng(self):
        """Test _diff_mapping_keys identifies keys only in eng."""
        mut = {"a": 1, "b": 2}
        eng = {"a": 1, "b": 2, "d": 4}
        result = DefaultRulesEngine._diff_mapping_keys(mut, eng)
        assert "only_eng" in result
        assert "'d'" in result

    def test_diff_mapping_keys_changed(self):
        """Test _diff_mapping_keys identifies changed values."""
        mut = {"a": 1, "b": 2}
        eng = {"a": 1, "b": 99}
        result = DefaultRulesEngine._diff_mapping_keys(mut, eng)
        assert "changed" in result
        assert "'b'" in result

    def test_diff_mapping_keys_truncation(self):
        """Test _diff_mapping_keys truncates when max_keys exceeded."""
        mut = {f"key{i}": i for i in range(10)}
        eng = {}
        result = DefaultRulesEngine._diff_mapping_keys(mut, eng, max_keys=3)
        assert "(+ more)" in result

    def test_diff_mapping_keys_empty_mappings(self):
        """Test _diff_mapping_keys handles empty mappings."""
        result = DefaultRulesEngine._diff_mapping_keys({}, {})
        assert result == "no_diff"

    def test_diff_mapping_keys_combined_differences(self):
        """Test _diff_mapping_keys handles multiple difference types."""
        mut = {"a": 1, "b": 2, "only_mut": 100}
        eng = {"a": 1, "b": 99, "only_eng": 200}
        result = DefaultRulesEngine._diff_mapping_keys(mut, eng)
        assert "only_mut" in result
        assert "only_eng" in result
        assert "changed" in result


class TestAreMoveEquivalentHelper:
    """Tests for _are_moves_equivalent method."""

    def test_are_moves_equivalent_method_exists(self):
        """Test _are_moves_equivalent method exists."""
        engine = DefaultRulesEngine()
        assert hasattr(engine, "_are_moves_equivalent")
        assert callable(engine._are_moves_equivalent)

    def test_equivalent_moves_same_type_and_player(self):
        """Test moves with same type and player are equivalent."""
        engine = DefaultRulesEngine()

        m1 = MagicMock(spec=Move)
        m1.type = MoveType.PLACE_RING
        m1.player = 1
        m1.to = MagicMock()
        m1.to.to_key.return_value = "a1"
        m1.from_pos = None

        m2 = MagicMock(spec=Move)
        m2.type = MoveType.PLACE_RING
        m2.player = 1
        m2.to = MagicMock()
        m2.to.to_key.return_value = "a1"
        m2.from_pos = None

        assert engine._are_moves_equivalent(m1, m2) is True

    def test_not_equivalent_different_type(self):
        """Test moves with different types are not equivalent."""
        engine = DefaultRulesEngine()

        m1 = MagicMock(spec=Move)
        m1.type = MoveType.PLACE_RING
        m1.player = 1
        m1.to = None
        m1.from_pos = None

        m2 = MagicMock(spec=Move)
        m2.type = MoveType.MOVE_STACK
        m2.player = 1
        m2.to = None
        m2.from_pos = None

        assert engine._are_moves_equivalent(m1, m2) is False

    def test_not_equivalent_different_player(self):
        """Test moves with different players are not equivalent."""
        engine = DefaultRulesEngine()

        m1 = MagicMock(spec=Move)
        m1.type = MoveType.PLACE_RING
        m1.player = 1
        m1.to = None
        m1.from_pos = None

        m2 = MagicMock(spec=Move)
        m2.type = MoveType.PLACE_RING
        m2.player = 2
        m2.to = None
        m2.from_pos = None

        assert engine._are_moves_equivalent(m1, m2) is False

    def test_not_equivalent_different_to_position(self):
        """Test moves with different to positions are not equivalent."""
        engine = DefaultRulesEngine()

        m1 = MagicMock(spec=Move)
        m1.type = MoveType.PLACE_RING
        m1.player = 1
        m1.to = MagicMock()
        m1.to.to_key.return_value = "a1"
        m1.from_pos = None

        m2 = MagicMock(spec=Move)
        m2.type = MoveType.PLACE_RING
        m2.player = 1
        m2.to = MagicMock()
        m2.to.to_key.return_value = "b2"
        m2.from_pos = None

        assert engine._are_moves_equivalent(m1, m2) is False

    def test_not_equivalent_one_to_none(self):
        """Test moves where one has to=None are not equivalent."""
        engine = DefaultRulesEngine()

        m1 = MagicMock(spec=Move)
        m1.type = MoveType.PLACE_RING
        m1.player = 1
        m1.to = MagicMock()
        m1.to.to_key.return_value = "a1"
        m1.from_pos = None

        m2 = MagicMock(spec=Move)
        m2.type = MoveType.PLACE_RING
        m2.player = 1
        m2.to = None
        m2.from_pos = None

        assert engine._are_moves_equivalent(m1, m2) is False

    def test_not_equivalent_different_from_position(self):
        """Test moves with different from positions are not equivalent."""
        engine = DefaultRulesEngine()

        m1 = MagicMock(spec=Move)
        m1.type = MoveType.MOVE_STACK
        m1.player = 1
        m1.to = MagicMock()
        m1.to.to_key.return_value = "c3"
        m1.from_pos = MagicMock()
        m1.from_pos.to_key.return_value = "a1"

        m2 = MagicMock(spec=Move)
        m2.type = MoveType.MOVE_STACK
        m2.player = 1
        m2.to = MagicMock()
        m2.to.to_key.return_value = "c3"
        m2.from_pos = MagicMock()
        m2.from_pos.to_key.return_value = "b2"

        assert engine._are_moves_equivalent(m1, m2) is False


class TestValidatorDispatch:
    """Tests for validator dispatch based on move type."""

    def test_place_ring_uses_placement_validator(self):
        """Test PLACE_RING dispatches to PlacementValidator (index 0)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.PLACE_RING

        with patch.object(engine.validators[0], 'validate', return_value=True) as mock_validate:
            result = engine.validate_move(mock_state, mock_move)
            mock_validate.assert_called_once_with(mock_state, mock_move)
            assert result is True

    def test_skip_placement_uses_placement_validator(self):
        """Test SKIP_PLACEMENT dispatches to PlacementValidator (index 0)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.SKIP_PLACEMENT

        with patch.object(engine.validators[0], 'validate', return_value=True) as mock_validate:
            result = engine.validate_move(mock_state, mock_move)
            mock_validate.assert_called_once_with(mock_state, mock_move)
            assert result is True

    def test_move_stack_uses_movement_validator(self):
        """Test MOVE_STACK dispatches to MovementValidator (index 1)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.MOVE_STACK

        with patch.object(engine.validators[1], 'validate', return_value=True) as mock_validate:
            result = engine.validate_move(mock_state, mock_move)
            mock_validate.assert_called_once_with(mock_state, mock_move)
            assert result is True

    def test_overtaking_capture_uses_capture_validator(self):
        """Test OVERTAKING_CAPTURE dispatches to CaptureValidator (index 2)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.OVERTAKING_CAPTURE

        with patch.object(engine.validators[2], 'validate', return_value=True) as mock_validate:
            result = engine.validate_move(mock_state, mock_move)
            mock_validate.assert_called_once_with(mock_state, mock_move)
            assert result is True

    def test_chain_capture_uses_capture_validator(self):
        """Test CHAIN_CAPTURE dispatches to CaptureValidator (index 2)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.CHAIN_CAPTURE

        with patch.object(engine.validators[2], 'validate', return_value=True) as mock_validate:
            result = engine.validate_move(mock_state, mock_move)
            mock_validate.assert_called_once_with(mock_state, mock_move)
            assert result is True

    def test_process_line_uses_line_validator(self):
        """Test PROCESS_LINE dispatches to LineValidator (index 3)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.PROCESS_LINE

        with patch.object(engine.validators[3], 'validate', return_value=True) as mock_validate:
            result = engine.validate_move(mock_state, mock_move)
            mock_validate.assert_called_once_with(mock_state, mock_move)
            assert result is True

    def test_territory_claim_uses_territory_validator(self):
        """Test TERRITORY_CLAIM dispatches to TerritoryValidator (index 4)."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.TERRITORY_CLAIM

        with patch.object(engine.validators[4], 'validate', return_value=True) as mock_validate:
            result = engine.validate_move(mock_state, mock_move)
            mock_validate.assert_called_once_with(mock_state, mock_move)
            assert result is True


class TestSwapSidesValidation:
    """Tests for SWAP_SIDES (pie rule) validation."""

    def test_swap_sides_requires_player_2(self):
        """Test SWAP_SIDES is only valid for player 2."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.SWAP_SIDES
        mock_move.player = 1  # Wrong player

        result = engine.validate_move(mock_state, mock_move)
        assert result is False

    def test_swap_sides_player_2_checks_valid_moves(self):
        """Test SWAP_SIDES for player 2 checks against GameEngine.get_valid_moves."""
        engine = DefaultRulesEngine()
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.SWAP_SIDES
        mock_move.player = 2

        # Mock GameEngine.get_valid_moves to return a SWAP_SIDES move
        swap_move = MagicMock()
        swap_move.type = MoveType.SWAP_SIDES
        swap_move.player = 2

        # GameEngine is imported inside the method, so patch at module level
        with patch('app.game_engine.GameEngine') as MockGameEngine:
            MockGameEngine.get_valid_moves.return_value = [swap_move]
            result = engine.validate_move(mock_state, mock_move)
            assert result is True


class TestForceBookkeepingMoves:
    """Tests for force_bookkeeping_moves configuration."""

    def test_force_bookkeeping_moves_default_false(self):
        """Test _force_bookkeeping_moves defaults to False."""
        engine = DefaultRulesEngine()
        # _force_bookkeeping_moves may be set based on env var or constructor
        # Check if attribute exists and is False or not set
        force_bm = getattr(engine, "_force_bookkeeping_moves", False)
        assert force_bm is False


class TestRulesEngineInterface:
    """Tests ensuring DefaultRulesEngine implements RulesEngine interface."""

    def test_derives_from_rules_engine(self):
        """Test DefaultRulesEngine derives from RulesEngine in MRO."""
        from app.rules.interfaces import RulesEngine
        # Protocol inheritance shows in __mro__
        mro_names = [cls.__name__ for cls in DefaultRulesEngine.__mro__]
        assert "RulesEngine" in mro_names

    def test_has_required_methods(self):
        """Test DefaultRulesEngine has all required RulesEngine methods."""
        engine = DefaultRulesEngine()
        # RulesEngine interface requires these methods
        assert hasattr(engine, "get_valid_moves")
        assert hasattr(engine, "apply_move")
        assert hasattr(engine, "validate_move")

    def test_methods_are_callable(self):
        """Test RulesEngine methods are callable."""
        engine = DefaultRulesEngine()
        assert callable(engine.get_valid_moves)
        assert callable(engine.apply_move)
        assert callable(engine.validate_move)

    def test_get_valid_moves_signature(self):
        """Test get_valid_moves has correct signature."""
        import inspect
        sig = inspect.signature(DefaultRulesEngine.get_valid_moves)
        params = list(sig.parameters.keys())
        assert "state" in params
        assert "player" in params

    def test_apply_move_signature(self):
        """Test apply_move has correct signature."""
        import inspect
        sig = inspect.signature(DefaultRulesEngine.apply_move)
        params = list(sig.parameters.keys())
        assert "state" in params
        assert "move" in params


class TestApplyMoveWithMutators:
    """Tests for _apply_move_with_mutators method."""

    def test_apply_move_with_mutators_method_exists(self):
        """Test _apply_move_with_mutators method exists."""
        engine = DefaultRulesEngine()
        assert hasattr(engine, "_apply_move_with_mutators")
        assert callable(engine._apply_move_with_mutators)


class TestShadowContractsBehavior:
    """Tests for shadow contracts behavior in apply_move."""

    def test_skip_shadow_contracts_returns_engine_result(self):
        """Test apply_move returns GameEngine result when skip_shadow_contracts is True."""
        engine = DefaultRulesEngine(skip_shadow_contracts=True)
        mock_state = MagicMock(spec=GameState)
        mock_move = MagicMock(spec=Move)
        mock_result = MagicMock(spec=GameState)

        # GameEngine is imported inside apply_move, so patch at module level
        with patch('app.game_engine.GameEngine') as MockGameEngine:
            MockGameEngine.apply_move.return_value = mock_result
            result = engine.apply_move(mock_state, mock_move)
            assert result is mock_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
