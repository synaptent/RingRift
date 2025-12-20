"""Tests for gpu_heuristic module.

Tests heuristic position evaluation with 45 weights per RR-CANON rules.
"""

import torch
from dataclasses import dataclass

from app.ai.gpu_heuristic import evaluate_positions_batch


@dataclass
class MockBatchGameState:
    """Mock BatchGameState for testing heuristic evaluation.

    Provides minimal interface needed by evaluate_positions_batch.
    """

    batch_size: int
    board_size: int
    device: torch.device
    num_players: int = 2
    max_history_moves: int = 500

    # Core state tensors
    stack_owner: torch.Tensor = None
    stack_height: torch.Tensor = None
    cap_height: torch.Tensor = None
    marker_owner: torch.Tensor = None
    is_collapsed: torch.Tensor = None
    territory_owner: torch.Tensor = None

    # Player state
    current_player: torch.Tensor = None
    rings_in_hand: torch.Tensor = None
    eliminated_rings: torch.Tensor = None
    buried_rings: torch.Tensor = None
    rings_caused_eliminated: torch.Tensor = None
    territory_count: torch.Tensor = None

    # Game state
    game_over: torch.Tensor = None
    winner: torch.Tensor = None
    move_count: torch.Tensor = None

    # Constraints
    must_move_from_y: torch.Tensor = None
    must_move_from_x: torch.Tensor = None

    # History
    move_history: torch.Tensor = None

    def __post_init__(self):
        """Initialize tensors if not provided."""
        bs, bz = self.batch_size, self.board_size

        if self.stack_owner is None:
            self.stack_owner = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.stack_height is None:
            self.stack_height = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.cap_height is None:
            self.cap_height = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.marker_owner is None:
            self.marker_owner = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.is_collapsed is None:
            self.is_collapsed = torch.zeros((bs, bz, bz), dtype=torch.bool, device=self.device)
        if self.territory_owner is None:
            self.territory_owner = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)

        if self.current_player is None:
            self.current_player = torch.ones(bs, dtype=torch.int8, device=self.device)
        if self.rings_in_hand is None:
            self.rings_in_hand = torch.zeros((bs, self.num_players + 1), dtype=torch.int8, device=self.device)
            # Give each player some rings
            self.rings_in_hand[:, 1] = 10
            self.rings_in_hand[:, 2] = 10
        if self.eliminated_rings is None:
            self.eliminated_rings = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)
        if self.buried_rings is None:
            self.buried_rings = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)
        if self.rings_caused_eliminated is None:
            self.rings_caused_eliminated = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)
        if self.territory_count is None:
            self.territory_count = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)

        if self.game_over is None:
            self.game_over = torch.zeros(bs, dtype=torch.bool, device=self.device)
        if self.winner is None:
            self.winner = torch.zeros(bs, dtype=torch.int8, device=self.device)
        if self.move_count is None:
            self.move_count = torch.zeros(bs, dtype=torch.int32, device=self.device)

        if self.must_move_from_y is None:
            self.must_move_from_y = torch.full((bs,), -1, dtype=torch.int32, device=self.device)
        if self.must_move_from_x is None:
            self.must_move_from_x = torch.full((bs,), -1, dtype=torch.int32, device=self.device)

        if self.move_history is None:
            self.move_history = torch.zeros((bs, self.max_history_moves, 6), dtype=torch.int16, device=self.device)

    def get_active_mask(self) -> torch.Tensor:
        """Return mask of active (non-finished) games."""
        return ~self.game_over

    def place_stack(self, game_idx: int, y: int, x: int, owner: int, height: int, cap: int = None):
        """Helper to place a stack on the board."""
        if cap is None:
            cap = height
        self.stack_owner[game_idx, y, x] = owner
        self.stack_height[game_idx, y, x] = height
        self.cap_height[game_idx, y, x] = cap

    def place_marker(self, game_idx: int, y: int, x: int, owner: int):
        """Helper to place a marker on the board."""
        self.marker_owner[game_idx, y, x] = owner

    def add_territory(self, game_idx: int, y: int, x: int, owner: int):
        """Helper to add territory."""
        self.is_collapsed[game_idx, y, x] = True
        self.territory_owner[game_idx, y, x] = owner
        self.territory_count[game_idx, owner] += 1


# =============================================================================
# Test basic evaluation
# =============================================================================


class TestEvaluatePositionsBatch:
    """Tests for evaluate_positions_batch function."""

    def test_empty_board_symmetric_scores(self):
        """Test that empty board with equal rings gives equal scores."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        weights = {}  # Use defaults
        scores = evaluate_positions_batch(state, weights)

        # Scores should be equal for both players on empty symmetric board
        assert scores.shape == (1, 3)  # (batch, num_players+1)
        # Player 1 and 2 should have similar scores (both have 10 rings in hand)
        diff = abs(scores[0, 1].item() - scores[0, 2].item())
        assert diff < 1.0, f"Score difference {diff} too large for symmetric position"

    def test_returns_correct_shape(self):
        """Test that output has correct shape."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=4, board_size=8, device=device)

        weights = {}
        scores = evaluate_positions_batch(state, weights)

        assert scores.shape == (4, 3)  # (batch_size, num_players+1)
        assert scores.dtype == torch.float32

    def test_stack_control_increases_score(self):
        """Test that having more stacks increases score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Give player 1 more stacks
        state.place_stack(0, 3, 3, owner=1, height=2)
        state.place_stack(0, 3, 4, owner=1, height=2)
        state.place_stack(0, 4, 3, owner=2, height=1)

        weights = {"WEIGHT_STACK_CONTROL": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have higher score due to more stacks
        assert scores[0, 1].item() > scores[0, 2].item()

    def test_stack_height_increases_score(self):
        """Test that taller stacks increase score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Same stack count, different heights
        state.place_stack(0, 3, 3, owner=1, height=4)
        state.place_stack(0, 4, 4, owner=2, height=1)

        weights = {"WEIGHT_STACK_HEIGHT": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test territory scoring
# =============================================================================


class TestTerritoryScoring:
    """Tests for territory-related scoring."""

    def test_territory_increases_score(self):
        """Test that territory increases score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Give player 1 territory
        state.territory_count[0, 1] = 5
        state.territory_count[0, 2] = 1

        weights = {"WEIGHT_TERRITORY": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()

    def test_territory_closure_bonus(self):
        """Test that territory adjacent to stacks gets closure bonus."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Place stack and adjacent territory
        state.place_stack(0, 3, 3, owner=1, height=2)
        state.add_territory(0, 3, 4, owner=1)

        weights = {"WEIGHT_TERRITORY_CLOSURE": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should benefit from territory closure
        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test center control
# =============================================================================


class TestCenterControl:
    """Tests for center control scoring."""

    def test_center_control_bonus(self):
        """Test that stacks near center get bonus."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 at center, player 2 at corner
        state.place_stack(0, 4, 4, owner=1, height=1)  # Near center
        state.place_stack(0, 0, 0, owner=2, height=1)  # Corner

        weights = {"WEIGHT_CENTER_CONTROL": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have higher score due to center position
        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test line pattern detection
# =============================================================================


class TestLinePatterns:
    """Tests for line pattern detection."""

    def test_two_in_row_detection(self):
        """Test detection of 2-in-a-row patterns."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Horizontal 2-in-row
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 3, 4, owner=1, height=1)

        weights = {"WEIGHT_TWO_IN_ROW": 10.0, "WEIGHT_LINE_POTENTIAL": 1.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()

    def test_three_in_row_detection(self):
        """Test detection of 3-in-a-row patterns."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Horizontal 3-in-row
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 3, 4, owner=1, height=1)
        state.place_stack(0, 3, 5, owner=1, height=1)

        weights = {"WEIGHT_THREE_IN_ROW": 10.0, "WEIGHT_LINE_POTENTIAL": 1.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()

    def test_four_in_row_detection(self):
        """Test detection of 4-in-a-row patterns."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Vertical 4-in-row
        state.place_stack(0, 2, 3, owner=1, height=1)
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 4, 3, owner=1, height=1)
        state.place_stack(0, 5, 3, owner=1, height=1)

        weights = {"WEIGHT_FOUR_IN_ROW": 10.0, "WEIGHT_LINE_POTENTIAL": 1.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()

    def test_diagonal_line_detection(self):
        """Test detection of diagonal lines."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Diagonal 3-in-row
        state.place_stack(0, 2, 2, owner=1, height=1)
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 4, 4, owner=1, height=1)

        weights = {"WEIGHT_THREE_IN_ROW": 10.0, "WEIGHT_LINE_POTENTIAL": 1.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()

    def test_gap_potential_detection(self):
        """Test detection of gap patterns (stack-empty-stack)."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Pattern with gap: stack at (3,3), empty at (3,4), stack at (3,5)
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 3, 5, owner=1, height=1)

        weights = {"WEIGHT_GAP_POTENTIAL": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test opponent threat detection
# =============================================================================


class TestOpponentThreat:
    """Tests for opponent threat detection."""

    def test_opponent_lines_reduce_score(self):
        """Test that opponent lines reduce player score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Opponent has a 3-in-row threat
        state.place_stack(0, 3, 3, owner=2, height=1)
        state.place_stack(0, 3, 4, owner=2, height=1)
        state.place_stack(0, 3, 5, owner=2, height=1)

        weights = {"WEIGHT_OPPONENT_THREAT": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have lower score due to opponent threat
        assert scores[0, 1].item() < scores[0, 2].item()


# =============================================================================
# Test vulnerability scoring
# =============================================================================


class TestVulnerability:
    """Tests for vulnerability scoring."""

    def test_vulnerability_to_taller_stacks(self):
        """Test that being next to taller opponent stacks increases vulnerability."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 has short stack adjacent to tall opponent stack
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 3, 4, owner=2, height=3)

        weights = {"WEIGHT_VULNERABILITY": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have lower score due to vulnerability
        assert scores[0, 1].item() < scores[0, 2].item()


# =============================================================================
# Test overtake potential
# =============================================================================


class TestOvertakePotential:
    """Tests for overtake potential scoring."""

    def test_overtake_opportunity(self):
        """Test that having taller stacks adjacent to shorter opponent stacks increases score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 has tall stack adjacent to short opponent stack
        state.place_stack(0, 3, 3, owner=1, height=3)
        state.place_stack(0, 3, 4, owner=2, height=1)

        weights = {"WEIGHT_OVERTAKE_POTENTIAL": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have higher score due to overtake potential
        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test victory proximity
# =============================================================================


class TestVictoryProximity:
    """Tests for victory proximity scoring."""

    def test_near_victory_bonus(self):
        """Test that being close to victory increases score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 close to territory victory
        state.territory_count[0, 1] = 10

        weights = {"WEIGHT_VICTORY_PROXIMITY": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()

    def test_victory_threshold_bonus(self):
        """Test that being very close to victory gives threshold bonus."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 very close to territory victory (>80% progress)
        state.territory_count[0, 1] = 15

        weights = {"WEIGHT_VICTORY_THRESHOLD_BONUS": 100.0}
        scores = evaluate_positions_batch(state, weights)

        # Should have significant bonus
        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test eliminated rings scoring
# =============================================================================


class TestEliminatedRings:
    """Tests for eliminated rings scoring."""

    def test_eliminated_rings_increase_score(self):
        """Test that having eliminated rings increases score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 has eliminated some rings
        state.eliminated_rings[0, 1] = 5
        state.eliminated_rings[0, 2] = 0

        weights = {"WEIGHT_ELIMINATED_RINGS": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test marker scoring
# =============================================================================


class TestMarkerScoring:
    """Tests for marker scoring."""

    def test_markers_increase_score(self):
        """Test that having markers increases score."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 has markers
        state.place_marker(0, 3, 3, owner=1)
        state.place_marker(0, 3, 4, owner=1)

        weights = {"WEIGHT_MARKER_COUNT": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Test recovery scoring
# =============================================================================


class TestRecoveryScoring:
    """Tests for recovery-related scoring."""

    def test_buried_rings_value(self):
        """Test that buried rings have value."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        state.buried_rings[0, 1] = 3
        state.buried_rings[0, 2] = 0

        weights = {"WEIGHT_BURIED_RING_VALUE": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()

    def test_recovery_eligibility_bonus(self):
        """Test recovery eligibility increases score relative to without recovery weights."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1: no stacks, has marker, has buried rings = eligible for recovery
        state.place_marker(0, 3, 3, owner=1)
        state.buried_rings[0, 1] = 2
        state.rings_in_hand[0, 1] = 0

        # Evaluate with no recovery weights
        weights_no_recovery = {}
        scores_no_recovery = evaluate_positions_batch(state, weights_no_recovery)

        # Evaluate with recovery weights
        weights_with_recovery = {
            "WEIGHT_RECOVERY_ELIGIBILITY": 50.0,
            "WEIGHT_RECOVERY_POTENTIAL": 20.0,
        }
        scores_with_recovery = evaluate_positions_batch(state, weights_with_recovery)

        # Player 1's score should be higher with recovery weights
        p1_improvement = scores_with_recovery[0, 1].item() - scores_no_recovery[0, 1].item()
        assert p1_improvement > 0, f"Expected recovery bonus, got improvement {p1_improvement}"


# =============================================================================
# Test penalty scoring
# =============================================================================


class TestPenaltyScoring:
    """Tests for penalty/bonus scoring."""

    def test_no_stacks_penalty(self):
        """Test penalty for having no stacks."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 has no stacks, player 2 has stacks
        state.place_stack(0, 3, 3, owner=2, height=2)
        state.place_stack(0, 4, 4, owner=2, height=2)

        weights = {"WEIGHT_NO_STACKS_PENALTY": 50.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have much lower score
        assert scores[0, 1].item() < scores[0, 2].item()

    def test_blocked_stack_penalty(self):
        """Test penalty for blocked stacks (3+ adjacent occupied cells)."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 stack surrounded by opponent stacks
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 2, 3, owner=2, height=1)  # Above
        state.place_stack(0, 4, 3, owner=2, height=1)  # Below
        state.place_stack(0, 3, 2, owner=2, height=1)  # Left

        weights = {"WEIGHT_BLOCKED_STACK_PENALTY": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have penalty for blocked stack
        assert scores[0, 1].item() < scores[0, 2].item()


# =============================================================================
# Test elimination penalty
# =============================================================================


class TestEliminationPenalty:
    """Tests for permanent elimination penalty."""

    def test_elimination_penalty(self):
        """Test massive penalty for being permanently eliminated."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1: no stacks, no rings in hand, no buried rings = eliminated
        state.rings_in_hand[0, 1] = 0
        # Player 2: has material
        state.place_stack(0, 3, 3, owner=2, height=2)

        weights = {}
        scores = evaluate_positions_batch(state, weights)

        # Player 1 should have massive negative score
        assert scores[0, 1].item() < -1000
        assert scores[0, 1].item() < scores[0, 2].item()


# =============================================================================
# Test weight configuration
# =============================================================================


class TestWeightConfiguration:
    """Tests for weight configuration options."""

    def test_old_format_weights(self):
        """Test backward compatibility with old 8-weight format."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)
        state.place_stack(0, 3, 3, owner=1, height=2)

        old_weights = {
            "material_weight": 10.0,
            "ring_count_weight": 5.0,
            "territory_weight": 8.0,
            "center_control_weight": 3.0,
            "mobility_weight": 2.0,
            "line_potential_weight": 4.0,
        }
        scores = evaluate_positions_batch(state, old_weights)

        assert scores.shape == (1, 3)
        # Should produce valid scores
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_new_format_weights(self):
        """Test new 45-weight format."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)
        state.place_stack(0, 3, 3, owner=1, height=2)

        new_weights = {
            "WEIGHT_STACK_CONTROL": 10.0,
            "WEIGHT_STACK_HEIGHT": 5.0,
            "WEIGHT_TERRITORY": 8.0,
            "WEIGHT_CENTER_CONTROL": 3.0,
        }
        scores = evaluate_positions_batch(state, new_weights)

        assert scores.shape == (1, 3)
        assert not torch.isnan(scores).any()

    def test_missing_weights_use_defaults(self):
        """Test that missing weights use default values."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Empty weights dict - should use all defaults
        scores = evaluate_positions_batch(state, {})

        assert scores.shape == (1, 3)
        assert not torch.isnan(scores).any()


# =============================================================================
# Test multi-game batch processing
# =============================================================================


class TestBatchProcessing:
    """Tests for multi-game batch processing."""

    def test_multiple_games_independent(self):
        """Test that multiple games are evaluated independently."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=3, board_size=8, device=device)

        # Game 0: Player 1 advantage
        state.place_stack(0, 3, 3, owner=1, height=3)

        # Game 1: Player 2 advantage
        state.place_stack(1, 3, 3, owner=2, height=3)

        # Game 2: Equal
        state.place_stack(2, 3, 3, owner=1, height=2)
        state.place_stack(2, 4, 4, owner=2, height=2)

        weights = {"WEIGHT_STACK_HEIGHT": 10.0}
        scores = evaluate_positions_batch(state, weights)

        # Game 0: P1 > P2
        assert scores[0, 1].item() > scores[0, 2].item()
        # Game 1: P2 > P1
        assert scores[1, 2].item() > scores[1, 1].item()
        # Game 2: Roughly equal
        diff = abs(scores[2, 1].item() - scores[2, 2].item())
        assert diff < 5.0  # Allow some tolerance

    def test_large_batch_processing(self):
        """Test processing of large batch."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=64, board_size=8, device=device)

        # Add some positions
        for g in range(64):
            state.place_stack(g, 3, 3, owner=(g % 2) + 1, height=1)

        weights = {}
        scores = evaluate_positions_batch(state, weights)

        assert scores.shape == (64, 3)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


# =============================================================================
# Test adjacency scoring
# =============================================================================


class TestAdjacencyScoring:
    """Tests for adjacency bonus scoring."""

    def test_adjacent_stacks_bonus(self):
        """Test that adjacent owned stacks get bonus."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Player 1 has adjacent stacks
        state.place_stack(0, 3, 3, owner=1, height=1)
        state.place_stack(0, 3, 4, owner=1, height=1)
        # Player 2 has isolated stacks
        state.place_stack(0, 1, 1, owner=2, height=1)
        state.place_stack(0, 5, 5, owner=2, height=1)

        weights = {"WEIGHT_ADJACENCY": 10.0}
        scores = evaluate_positions_batch(state, weights)

        assert scores[0, 1].item() > scores[0, 2].item()


# =============================================================================
# Integration tests
# =============================================================================


class TestHeuristicIntegration:
    """Integration tests for heuristic evaluation."""

    def test_complex_position_evaluation(self):
        """Test evaluation of a complex game position."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        # Setup a complex position
        state.place_stack(0, 3, 3, owner=1, height=2)
        state.place_stack(0, 3, 4, owner=1, height=1)
        state.place_stack(0, 4, 3, owner=1, height=3)
        state.place_marker(0, 2, 3, owner=1)
        state.territory_count[0, 1] = 3

        state.place_stack(0, 5, 5, owner=2, height=2)
        state.place_stack(0, 5, 6, owner=2, height=2)
        state.territory_count[0, 2] = 2

        weights = {
            "WEIGHT_STACK_CONTROL": 5.0,
            "WEIGHT_STACK_HEIGHT": 3.0,
            "WEIGHT_TERRITORY": 8.0,
            "WEIGHT_ADJACENCY": 2.0,
        }
        scores = evaluate_positions_batch(state, weights)

        # Both players should have positive scores
        assert scores.shape == (1, 3)
        # Scores should be valid
        assert not torch.isnan(scores).any()

    def test_module_exports(self):
        """Test that module exports correctly."""
        from app.ai.gpu_heuristic import evaluate_positions_batch

        assert callable(evaluate_positions_batch)

    def test_different_board_sizes(self):
        """Test evaluation works with different board sizes."""
        device = torch.device('cpu')

        for board_size in [8, 9, 13, 19]:
            state = MockBatchGameState(batch_size=1, board_size=board_size, device=device)
            state.place_stack(0, board_size // 2, board_size // 2, owner=1, height=1)

            weights = {}
            scores = evaluate_positions_batch(state, weights)

            assert scores.shape == (1, 3)
            assert not torch.isnan(scores).any()


class TestSymmetricEvaluation:
    """Tests for symmetric evaluation (v2.0 requirement)."""

    def test_symmetric_empty_board(self):
        """Test that empty board gives symmetric scores."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=8, device=device)

        weights = {}
        scores = evaluate_positions_batch(state, weights)

        # Both players should have similar scores on empty board
        p1_score = scores[0, 1].item()
        p2_score = scores[0, 2].item()
        assert abs(p1_score - p2_score) < 1.0

    def test_symmetric_position_swap(self):
        """Test that swapping player positions swaps scores."""
        device = torch.device('cpu')

        # Position A: P1 has stack at (3,3), P2 has stack at (5,5)
        state_a = MockBatchGameState(batch_size=1, board_size=8, device=device)
        state_a.place_stack(0, 3, 3, owner=1, height=2)
        state_a.place_stack(0, 5, 5, owner=2, height=1)

        # Position B: Swapped - P2 has stack at (3,3), P1 has stack at (5,5)
        state_b = MockBatchGameState(batch_size=1, board_size=8, device=device)
        state_b.place_stack(0, 3, 3, owner=2, height=2)
        state_b.place_stack(0, 5, 5, owner=1, height=1)

        weights = {"WEIGHT_STACK_HEIGHT": 10.0}
        scores_a = evaluate_positions_batch(state_a, weights)
        scores_b = evaluate_positions_batch(state_b, weights)

        # Player advantages should roughly swap
        # P1 advantage in A should equal P2 advantage in B
        p1_adv_a = scores_a[0, 1].item() - scores_a[0, 2].item()
        p2_adv_b = scores_b[0, 2].item() - scores_b[0, 1].item()
        assert abs(p1_adv_a - p2_adv_b) < 1.0
