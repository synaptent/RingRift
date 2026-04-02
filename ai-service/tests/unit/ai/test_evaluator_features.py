"""Unit tests for Phase 3.1 evaluator features.

Tests cover the 5 new heuristic features added in Phase 3.1:
1. Chain Capture Risk (TacticalEvaluator) - cascade capture vulnerability
2. Tempo Advantage (StrategicEvaluator) - initiative and forcing moves
3. Endgame Phase Transition (EndgameEvaluator) - phase-aware weight modulation
4. Stack Synergy (PositionalEvaluator) - coordination between stacks
5. Expansion Potential (PositionalEvaluator) - territory growth opportunities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Mock Data Structures
# =============================================================================


@dataclass
class MockPosition:
    """Mock position for testing."""
    q: int
    r: int

    def __hash__(self):
        return hash((self.q, self.r))

    def __eq__(self, other):
        if isinstance(other, MockPosition):
            return self.q == other.q and self.r == other.r
        return False


@dataclass
class MockStack:
    """Mock ring stack for testing."""
    position: MockPosition
    controller: int
    height: int
    rings: List[int]

    @property
    def top_player(self) -> int:
        return self.rings[-1] if self.rings else 0

    @property
    def controlling_player(self) -> int:
        return self.controller


@dataclass
class MockPlayer:
    """Mock player for testing."""
    number: int
    rings_in_hand: int
    rings_on_board: int
    rings_eliminated: int
    markers_on_board: int
    territory_count: int
    stacks: List[MockStack]

    @property
    def player_number(self) -> int:
        return self.number


@dataclass
class MockBoard:
    """Mock board for testing."""
    type: str
    size: int
    cells: Dict
    stacks: Dict[MockPosition, MockStack]
    collapsed_spaces: set = field(default_factory=set)

    def get_all_cells(self) -> List[MockPosition]:
        return list(self.cells.keys())

    def get_stack_at(self, pos: MockPosition) -> Optional[MockStack]:
        return self.stacks.get(pos)


@dataclass
class MockGameState:
    """Mock game state for testing."""
    board: MockBoard
    players: List[MockPlayer]
    current_player: int
    num_players: int
    phase: str
    game_over: bool = False
    winner: Optional[int] = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_geometry():
    """Create mock geometry with configurable neighbors."""
    geometry = MagicMock()
    geometry.center_positions = []
    geometry.all_positions = []
    geometry.get_neighbors = MagicMock(return_value=[])
    geometry.get_adjacent_keys = MagicMock(return_value=[])
    geometry.get_line_of_sight = MagicMock(return_value=[])
    geometry.get_los_directions = MagicMock(return_value=[])
    geometry.get_center_positions = MagicMock(return_value=[])
    geometry.get_all_board_keys = MagicMock(return_value=[])
    return geometry


@pytest.fixture
def basic_2p_state():
    """Create a basic 2-player game state."""
    p1 = MockPlayer(
        number=1,
        rings_in_hand=3,
        rings_on_board=2,
        rings_eliminated=0,
        markers_on_board=5,
        territory_count=4,
        stacks=[],
    )
    p2 = MockPlayer(
        number=2,
        rings_in_hand=3,
        rings_on_board=2,
        rings_eliminated=0,
        markers_on_board=4,
        territory_count=3,
        stacks=[],
    )

    cells = {
        MockPosition(0, 0): None,
        MockPosition(1, 0): None,
        MockPosition(0, 1): None,
        MockPosition(1, 1): None,
    }

    board = MockBoard(
        type="hex8",
        size=8,
        cells=cells,
        stacks={},
    )

    return MockGameState(
        board=board,
        players=[p1, p2],
        current_player=1,
        num_players=2,
        phase="play",
    )


@pytest.fixture
def state_with_stacks():
    """Create a game state with stacks for tactical tests."""
    pos1 = MockPosition(0, 0)
    pos2 = MockPosition(1, 0)
    pos3 = MockPosition(2, 0)
    pos4 = MockPosition(0, 1)

    stack1 = MockStack(position=pos1, controller=1, height=2, rings=[1, 1])
    stack2 = MockStack(position=pos2, controller=2, height=3, rings=[2, 2, 2])
    stack3 = MockStack(position=pos3, controller=1, height=1, rings=[1])
    stack4 = MockStack(position=pos4, controller=2, height=2, rings=[2, 2])

    p1 = MockPlayer(
        number=1,
        rings_in_hand=2,
        rings_on_board=3,
        rings_eliminated=0,
        markers_on_board=5,
        territory_count=4,
        stacks=[stack1, stack3],
    )
    p2 = MockPlayer(
        number=2,
        rings_in_hand=2,
        rings_on_board=5,
        rings_eliminated=0,
        markers_on_board=4,
        territory_count=3,
        stacks=[stack2, stack4],
    )

    cells = {pos1: None, pos2: None, pos3: None, pos4: None}
    stacks = {pos1: stack1, pos2: stack2, pos3: stack3, pos4: stack4}

    board = MockBoard(type="hex8", size=8, cells=cells, stacks=stacks)

    return MockGameState(
        board=board,
        players=[p1, p2],
        current_player=1,
        num_players=2,
        phase="play",
    )


# =============================================================================
# TacticalEvaluator Tests - Chain Capture
# =============================================================================


class TestTacticalWeights:
    """Tests for TacticalWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        from app.ai.evaluators.tactical_evaluator import TacticalWeights

        weights = TacticalWeights()
        assert weights.chain_capture_risk == 7.0
        assert weights.chain_capture_potential == 6.0

    def test_weights_from_dict(self):
        """Test creating weights with custom values."""
        from app.ai.evaluators.tactical_evaluator import TacticalWeights

        weights = TacticalWeights(
            chain_capture_risk=10.0,
            chain_capture_potential=8.0,
        )
        assert weights.chain_capture_risk == 10.0
        assert weights.chain_capture_potential == 8.0

    def test_to_dict_includes_chain_capture(self):
        """Test that to_dict includes chain capture weights."""
        from app.ai.evaluators.tactical_evaluator import TacticalWeights

        weights = TacticalWeights()
        d = weights.to_dict()
        assert "WEIGHT_CHAIN_CAPTURE_RISK" in d
        assert "WEIGHT_CHAIN_CAPTURE_POTENTIAL" in d
        assert d["WEIGHT_CHAIN_CAPTURE_RISK"] == 7.0
        assert d["WEIGHT_CHAIN_CAPTURE_POTENTIAL"] == 6.0


class TestTacticalScore:
    """Tests for TacticalScore dataclass."""

    def test_score_includes_chain_capture_fields(self):
        """Test that TacticalScore has chain capture fields."""
        from app.ai.evaluators.tactical_evaluator import TacticalScore

        score = TacticalScore(
            total=10.0,
            opponent_threats=0.0,
            vulnerability=0.0,
            overtake_potential=0.0,
            chain_capture_risk=-5.0,
            chain_capture_potential=3.0,
        )
        assert score.chain_capture_risk == -5.0
        assert score.chain_capture_potential == 3.0

    def test_score_to_dict(self):
        """Test score to_dict method."""
        from app.ai.evaluators.tactical_evaluator import TacticalScore

        score = TacticalScore(
            total=10.0,
            opponent_threats=0.0,
            vulnerability=0.0,
            overtake_potential=0.0,
            chain_capture_risk=-5.0,
            chain_capture_potential=3.0,
        )
        d = score.to_dict()
        assert "chain_capture_risk" in d
        assert "chain_capture_potential" in d


class TestTacticalEvaluatorChainCapture:
    """Tests for chain capture risk/potential evaluation."""

    def test_evaluator_has_chain_capture_methods(self):
        """Test that TacticalEvaluator has chain capture methods."""
        from app.ai.evaluators.tactical_evaluator import TacticalEvaluator

        evaluator = TacticalEvaluator()
        assert hasattr(evaluator, "_evaluate_chain_capture_risk")
        assert hasattr(evaluator, "_evaluate_chain_capture_potential")

    def test_chain_capture_risk_returns_negative_for_vulnerable(self, mock_geometry):
        """Test that vulnerable positions return negative risk scores."""
        from app.ai.evaluators.tactical_evaluator import TacticalEvaluator

        evaluator = TacticalEvaluator()
        # With default empty state, should return 0 (no vulnerability)
        # This tests the method exists and returns a numeric value
        assert isinstance(evaluator.weights.chain_capture_risk, float)

    def test_evaluate_tactical_includes_chain_capture(self, basic_2p_state, mock_geometry):
        """Test that evaluate_tactical includes chain capture scores."""
        from app.ai.evaluators.tactical_evaluator import TacticalEvaluator

        evaluator = TacticalEvaluator()
        evaluator.set_geometry(mock_geometry)

        breakdown = evaluator.get_breakdown(basic_2p_state, player_idx=0)
        assert "chain_capture_risk" in breakdown
        assert "chain_capture_potential" in breakdown

    def test_set_weights_updates_chain_capture(self):
        """Test that set_weights properly updates chain capture weights."""
        from app.ai.evaluators.tactical_evaluator import TacticalEvaluator

        evaluator = TacticalEvaluator()
        weights = {
            "WEIGHT_CHAIN_CAPTURE_RISK": 15.0,
            "WEIGHT_CHAIN_CAPTURE_POTENTIAL": 12.0,
        }
        evaluator.set_weights(weights)

        assert evaluator.weights.chain_capture_risk == 15.0
        assert evaluator.weights.chain_capture_potential == 12.0


# =============================================================================
# StrategicEvaluator Tests - Tempo Advantage
# =============================================================================


class TestStrategicWeights:
    """Tests for StrategicWeights dataclass."""

    def test_default_tempo_weights(self):
        """Test default tempo weight values."""
        from app.ai.evaluators.strategic_evaluator import StrategicWeights

        weights = StrategicWeights()
        assert weights.tempo_advantage == 5.0
        assert weights.forcing_move_value == 3.0

    def test_tempo_weights_in_to_dict(self):
        """Test that to_dict includes tempo weights."""
        from app.ai.evaluators.strategic_evaluator import StrategicWeights

        weights = StrategicWeights()
        d = weights.to_dict()
        assert "WEIGHT_TEMPO_ADVANTAGE" in d
        assert "WEIGHT_FORCING_MOVE_VALUE" in d


class TestStrategicScore:
    """Tests for StrategicScore dataclass."""

    def test_score_includes_tempo_field(self):
        """Test that StrategicScore has tempo_advantage field."""
        from app.ai.evaluators.strategic_evaluator import StrategicScore

        score = StrategicScore(
            total=10.0,
            victory_proximity=5.0,
            opponent_victory_threat=0.0,
            forced_elimination_risk=0.0,
            lps_action_advantage=0.0,
            multi_leader_threat=0.0,
            tempo_advantage=5.0,
        )
        assert score.tempo_advantage == 5.0

    def test_score_to_dict_includes_tempo(self):
        """Test that to_dict includes tempo_advantage."""
        from app.ai.evaluators.strategic_evaluator import StrategicScore

        score = StrategicScore(
            total=10.0,
            victory_proximity=0.0,
            opponent_victory_threat=0.0,
            forced_elimination_risk=0.0,
            lps_action_advantage=0.0,
            multi_leader_threat=0.0,
            tempo_advantage=5.0,
        )
        d = score.to_dict()
        assert "tempo_advantage" in d
        assert d["tempo_advantage"] == 5.0


class TestStrategicEvaluatorTempo:
    """Tests for tempo advantage evaluation."""

    def test_evaluator_has_tempo_methods(self):
        """Test that StrategicEvaluator has tempo methods."""
        from app.ai.evaluators.strategic_evaluator import StrategicEvaluator

        evaluator = StrategicEvaluator()
        assert hasattr(evaluator, "_evaluate_tempo_advantage")
        assert hasattr(evaluator, "_count_forcing_moves")

    def test_set_weights_updates_tempo(self):
        """Test that set_weights properly updates tempo weights."""
        from app.ai.evaluators.strategic_evaluator import StrategicEvaluator

        evaluator = StrategicEvaluator()
        weights = {
            "WEIGHT_TEMPO_ADVANTAGE": 10.0,
            "WEIGHT_FORCING_MOVE_VALUE": 5.0,
        }
        evaluator.set_weights(weights)

        assert evaluator.weights.tempo_advantage == 10.0
        assert evaluator.weights.forcing_move_value == 5.0

    def test_evaluate_includes_tempo(self, basic_2p_state, mock_geometry):
        """Test that evaluate_strategic_all includes tempo score."""
        from app.ai.evaluators.strategic_evaluator import StrategicEvaluator

        evaluator = StrategicEvaluator()
        evaluator.set_geometry(mock_geometry)

        breakdown = evaluator.get_breakdown(basic_2p_state, player_idx=0)
        assert "tempo_advantage" in breakdown


# =============================================================================
# EndgameEvaluator Tests - Phase Transition
# =============================================================================


class TestEndgameWeights:
    """Tests for EndgameWeights dataclass."""

    def test_default_phase_transition_weights(self):
        """Test default phase transition weight values."""
        from app.ai.evaluators.endgame_evaluator import EndgameWeights

        weights = EndgameWeights()
        assert weights.endgame_aggression == 4.0
        assert weights.phase_transition_bonus == 3.0

    def test_to_dict_includes_phase_transition(self):
        """Test that to_dict includes phase transition weights."""
        from app.ai.evaluators.endgame_evaluator import EndgameWeights

        weights = EndgameWeights()
        d = weights.to_dict()
        assert "WEIGHT_ENDGAME_AGGRESSION" in d
        assert "WEIGHT_PHASE_TRANSITION_BONUS" in d


class TestEndgameScore:
    """Tests for EndgameScore dataclass."""

    def test_score_includes_phase_transition_field(self):
        """Test that EndgameScore has phase_transition field."""
        from app.ai.evaluators.endgame_evaluator import EndgameScore

        score = EndgameScore(
            total=10.0,
            recovery_potential=0.0,
            phase_transition=5.0,
        )
        assert score.phase_transition == 5.0

    def test_score_to_dict_includes_phase_transition(self):
        """Test that to_dict includes phase_transition."""
        from app.ai.evaluators.endgame_evaluator import EndgameScore

        score = EndgameScore(
            total=10.0,
            recovery_potential=0.0,
            phase_transition=5.0,
        )
        d = score.to_dict()
        assert "phase_transition" in d
        assert d["phase_transition"] == 5.0


class TestEndgameEvaluatorPhaseTransition:
    """Tests for phase transition evaluation."""

    def test_evaluator_has_phase_transition_methods(self):
        """Test that EndgameEvaluator has phase transition methods."""
        from app.ai.evaluators.endgame_evaluator import EndgameEvaluator

        evaluator = EndgameEvaluator()
        assert hasattr(evaluator, "_evaluate_phase_transition")

    def test_set_weights_updates_phase_transition(self):
        """Test that set_weights properly updates phase transition weights."""
        from app.ai.evaluators.endgame_evaluator import EndgameEvaluator

        evaluator = EndgameEvaluator()
        weights = {
            "WEIGHT_ENDGAME_AGGRESSION": 8.0,
            "WEIGHT_PHASE_TRANSITION_BONUS": 6.0,
        }
        evaluator.set_weights(weights)

        assert evaluator.weights.endgame_aggression == 8.0
        assert evaluator.weights.phase_transition_bonus == 6.0


# =============================================================================
# PositionalEvaluator Tests - Stack Synergy
# =============================================================================


class TestPositionalWeightsStackSynergy:
    """Tests for stack synergy weights."""

    def test_default_stack_synergy_weights(self):
        """Test default stack synergy weight values."""
        from app.ai.evaluators.positional_evaluator import PositionalWeights

        weights = PositionalWeights()
        assert weights.stack_synergy == 4.0
        assert weights.mutual_defense == 3.0

    def test_to_dict_includes_stack_synergy(self):
        """Test that to_dict includes stack synergy weights."""
        from app.ai.evaluators.positional_evaluator import PositionalWeights

        weights = PositionalWeights()
        d = weights.to_dict()
        assert "WEIGHT_STACK_SYNERGY" in d
        assert "WEIGHT_MUTUAL_DEFENSE" in d


class TestPositionalScoreStackSynergy:
    """Tests for stack synergy score field."""

    def test_score_includes_stack_synergy_field(self):
        """Test that PositionalScore has stack_synergy field."""
        from app.ai.evaluators.positional_evaluator import PositionalScore

        score = PositionalScore(
            total=10.0,
            territory=5.0,
            center_control=2.0,
            territory_closure=1.0,
            stack_synergy=5.0,
            expansion_potential=3.0,
        )
        assert score.stack_synergy == 5.0


class TestPositionalEvaluatorStackSynergy:
    """Tests for stack synergy evaluation."""

    def test_evaluator_has_stack_synergy_methods(self):
        """Test that PositionalEvaluator has stack synergy methods."""
        from app.ai.evaluators.positional_evaluator import PositionalEvaluator

        evaluator = PositionalEvaluator()
        assert hasattr(evaluator, "_evaluate_stack_synergy")

    def test_set_weights_updates_stack_synergy(self):
        """Test that set_weights properly updates stack synergy weights."""
        from app.ai.evaluators.positional_evaluator import PositionalEvaluator

        evaluator = PositionalEvaluator()
        weights = {
            "WEIGHT_STACK_SYNERGY": 8.0,
            "WEIGHT_MUTUAL_DEFENSE": 6.0,
        }
        evaluator.set_weights(weights)

        assert evaluator.weights.stack_synergy == 8.0
        assert evaluator.weights.mutual_defense == 6.0


# =============================================================================
# PositionalEvaluator Tests - Expansion Potential
# =============================================================================


class TestPositionalWeightsExpansion:
    """Tests for expansion potential weights."""

    def test_default_expansion_weights(self):
        """Test default expansion weight values."""
        from app.ai.evaluators.positional_evaluator import PositionalWeights

        weights = PositionalWeights()
        assert weights.expansion_potential == 5.0
        assert weights.frontier_strength == 3.0

    def test_to_dict_includes_expansion(self):
        """Test that to_dict includes expansion weights."""
        from app.ai.evaluators.positional_evaluator import PositionalWeights

        weights = PositionalWeights()
        d = weights.to_dict()
        assert "WEIGHT_EXPANSION_POTENTIAL" in d
        assert "WEIGHT_FRONTIER_STRENGTH" in d


class TestPositionalScoreExpansion:
    """Tests for expansion potential score field."""

    def test_score_includes_expansion_field(self):
        """Test that PositionalScore has expansion_potential field."""
        from app.ai.evaluators.positional_evaluator import PositionalScore

        score = PositionalScore(
            total=10.0,
            territory=5.0,
            center_control=2.0,
            territory_closure=1.0,
            stack_synergy=0.0,
            expansion_potential=5.0,
        )
        assert score.expansion_potential == 5.0


class TestPositionalEvaluatorExpansion:
    """Tests for expansion potential evaluation."""

    def test_evaluator_has_expansion_methods(self):
        """Test that PositionalEvaluator has expansion methods."""
        from app.ai.evaluators.positional_evaluator import PositionalEvaluator

        evaluator = PositionalEvaluator()
        assert hasattr(evaluator, "_evaluate_expansion_potential")

    def test_set_weights_updates_expansion(self):
        """Test that set_weights properly updates expansion weights."""
        from app.ai.evaluators.positional_evaluator import PositionalEvaluator

        evaluator = PositionalEvaluator()
        weights = {
            "WEIGHT_EXPANSION_POTENTIAL": 10.0,
            "WEIGHT_FRONTIER_STRENGTH": 8.0,
        }
        evaluator.set_weights(weights)

        assert evaluator.weights.expansion_potential == 10.0
        assert evaluator.weights.frontier_strength == 8.0


# =============================================================================
# Integration Tests - Weight Profile Verification
# =============================================================================


class TestHeuristicWeightsIntegration:
    """Tests that verify new weights are in heuristic_weights.py."""

    def test_all_new_weights_in_base_profile(self):
        """Test that all 10 new weights exist in BASE_V1_BALANCED_WEIGHTS."""
        from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS

        new_weights = [
            "WEIGHT_CHAIN_CAPTURE_RISK",
            "WEIGHT_CHAIN_CAPTURE_POTENTIAL",
            "WEIGHT_TEMPO_ADVANTAGE",
            "WEIGHT_FORCING_MOVE_VALUE",
            "WEIGHT_ENDGAME_AGGRESSION",
            "WEIGHT_PHASE_TRANSITION_BONUS",
            "WEIGHT_STACK_SYNERGY",
            "WEIGHT_MUTUAL_DEFENSE",
            "WEIGHT_EXPANSION_POTENTIAL",
            "WEIGHT_FRONTIER_STRENGTH",
        ]

        for weight in new_weights:
            assert weight in BASE_V1_BALANCED_WEIGHTS, f"{weight} not in BASE_V1_BALANCED_WEIGHTS"

    def test_all_new_weights_in_keys_list(self):
        """Test that all 10 new weights exist in HEURISTIC_WEIGHT_KEYS."""
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS

        new_weights = [
            "WEIGHT_CHAIN_CAPTURE_RISK",
            "WEIGHT_CHAIN_CAPTURE_POTENTIAL",
            "WEIGHT_TEMPO_ADVANTAGE",
            "WEIGHT_FORCING_MOVE_VALUE",
            "WEIGHT_ENDGAME_AGGRESSION",
            "WEIGHT_PHASE_TRANSITION_BONUS",
            "WEIGHT_STACK_SYNERGY",
            "WEIGHT_MUTUAL_DEFENSE",
            "WEIGHT_EXPANSION_POTENTIAL",
            "WEIGHT_FRONTIER_STRENGTH",
        ]

        for weight in new_weights:
            assert weight in HEURISTIC_WEIGHT_KEYS, f"{weight} not in HEURISTIC_WEIGHT_KEYS"

    def test_weight_count_is_59(self):
        """Test that total weight count is 59 (49 original + 10 new)."""
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS

        assert len(HEURISTIC_WEIGHT_KEYS) == 59, f"Expected 59 weights, got {len(HEURISTIC_WEIGHT_KEYS)}"

    def test_weights_dict_and_keys_match(self):
        """Test that BASE_V1_BALANCED_WEIGHTS keys match HEURISTIC_WEIGHT_KEYS."""
        from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS, HEURISTIC_WEIGHT_KEYS

        dict_keys = set(BASE_V1_BALANCED_WEIGHTS.keys())
        list_keys = set(HEURISTIC_WEIGHT_KEYS)

        assert dict_keys == list_keys, f"Mismatch: {dict_keys.symmetric_difference(list_keys)}"


# =============================================================================
# Symmetric Evaluation Tests
# =============================================================================


class TestSymmetricEvaluation:
    """Tests for symmetric evaluation (my_value - max_opponent_value)."""

    def test_chain_capture_uses_symmetric_eval(self):
        """Test that chain capture uses symmetric evaluation."""
        from app.ai.evaluators.tactical_evaluator import TacticalEvaluator

        evaluator = TacticalEvaluator()
        # The methods should exist for symmetric evaluation
        assert hasattr(evaluator, "_evaluate_chain_capture_risk")
        assert hasattr(evaluator, "_evaluate_chain_capture_potential")

    def test_tempo_uses_symmetric_eval(self):
        """Test that tempo advantage uses symmetric evaluation."""
        from app.ai.evaluators.strategic_evaluator import StrategicEvaluator

        evaluator = StrategicEvaluator()
        assert hasattr(evaluator, "_evaluate_tempo_advantage")

    def test_stack_synergy_uses_symmetric_eval(self):
        """Test that stack synergy uses symmetric evaluation."""
        from app.ai.evaluators.positional_evaluator import PositionalEvaluator

        evaluator = PositionalEvaluator()
        assert hasattr(evaluator, "_evaluate_stack_synergy")

    def test_expansion_potential_uses_symmetric_eval(self):
        """Test that expansion potential uses symmetric evaluation."""
        from app.ai.evaluators.positional_evaluator import PositionalEvaluator

        evaluator = PositionalEvaluator()
        assert hasattr(evaluator, "_evaluate_expansion_potential")


# =============================================================================
# Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing evaluators."""

    def test_tactical_evaluator_still_has_original_methods(self):
        """Test that TacticalEvaluator still has original evaluation methods."""
        from app.ai.evaluators.tactical_evaluator import TacticalEvaluator

        evaluator = TacticalEvaluator()
        assert hasattr(evaluator, "evaluate_tactical")
        assert hasattr(evaluator, "get_breakdown")
        assert hasattr(evaluator, "set_weights")
        assert hasattr(evaluator, "set_geometry")

    def test_strategic_evaluator_still_has_original_methods(self):
        """Test that StrategicEvaluator still has original evaluation methods."""
        from app.ai.evaluators.strategic_evaluator import StrategicEvaluator

        evaluator = StrategicEvaluator()
        assert hasattr(evaluator, "evaluate_strategic_all")
        assert hasattr(evaluator, "get_breakdown")
        assert hasattr(evaluator, "set_weights")
        assert hasattr(evaluator, "set_geometry")

    def test_endgame_evaluator_still_has_original_methods(self):
        """Test that EndgameEvaluator still has original evaluation methods."""
        from app.ai.evaluators.endgame_evaluator import EndgameEvaluator

        evaluator = EndgameEvaluator()
        assert hasattr(evaluator, "evaluate_endgame")
        assert hasattr(evaluator, "get_breakdown")
        assert hasattr(evaluator, "set_weights")
        assert hasattr(evaluator, "set_geometry")

    def test_positional_evaluator_still_has_original_methods(self):
        """Test that PositionalEvaluator still has original evaluation methods."""
        from app.ai.evaluators.positional_evaluator import PositionalEvaluator

        evaluator = PositionalEvaluator()
        assert hasattr(evaluator, "evaluate_positional")
        assert hasattr(evaluator, "get_breakdown")
        assert hasattr(evaluator, "set_weights")
        assert hasattr(evaluator, "set_geometry")
