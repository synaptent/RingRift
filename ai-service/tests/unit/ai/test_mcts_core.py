"""Unit tests for MCTS core components.

These are fast unit tests that don't require neural network inference
or expensive tree search. They test the core MCTS logic in isolation.
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from app.ai.mcts_ai import (
    DynamicBatchSizer,
    MCTSNode,
    MCTSNodeLite,
    _move_key,
    _moves_match,
    _pos_key,
    _pos_seq_key,
)
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_game_state():
    """Create a minimal mock game state."""
    state = MagicMock(spec=GameState)
    state.board = MagicMock(spec=BoardState)
    state.board.board_type = BoardType.SQUARE8
    state.current_player = 1
    state.phase = GamePhase.MOVEMENT
    state.status = GameStatus.ACTIVE
    return state


@pytest.fixture
def sample_move():
    """Create a sample move for testing."""
    return Move(
        id="test_move",
        player=1,
        type=MoveType.MOVE_STACK,
        from_pos=Position(x=2, y=3),
        to=Position(x=4, y=5),
    )


@pytest.fixture
def root_node(mock_game_state):
    """Create a root MCTS node."""
    return MCTSNode(mock_game_state)


# =============================================================================
# Tests for MCTSNode
# =============================================================================

class TestMCTSNode:
    """Tests for MCTSNode class."""

    def test_initialization(self, mock_game_state):
        """Test node initialization."""
        node = MCTSNode(mock_game_state)

        assert node.game_state == mock_game_state
        assert node.parent is None
        assert node.move is None
        assert node.children == []
        assert node.wins == 0
        assert node.visits == 0
        assert node.prior == 0.0

    def test_initialization_with_parent(self, mock_game_state, sample_move):
        """Test node initialization with parent and move."""
        parent = MCTSNode(mock_game_state)
        child = MCTSNode(mock_game_state, parent=parent, move=sample_move)

        assert child.parent == parent
        assert child.move == sample_move

    def test_add_child(self, root_node, mock_game_state, sample_move):
        """Test adding a child node."""
        root_node.untried_moves = [sample_move]

        child = root_node.add_child(sample_move, mock_game_state, prior=0.5)

        assert child in root_node.children
        assert child.parent == root_node
        assert child.move == sample_move
        assert child.prior == 0.5
        assert sample_move not in root_node.untried_moves

    def test_update_increments_visits(self, root_node):
        """Test that update increments visits."""
        initial_visits = root_node.visits

        root_node.update(0.5)

        assert root_node.visits == initial_visits + 1

    def test_update_adds_wins(self, root_node):
        """Test that update adds to wins."""
        root_node.update(0.75)
        root_node.update(0.25)

        assert root_node.visits == 2
        assert root_node.wins == 1.0  # 0.75 + 0.25

    def test_uct_select_child_prefers_high_prior_unvisited(self, root_node, mock_game_state):
        """Test UCT selection prefers high-prior unvisited children."""
        # Add children with different priors
        moves = [
            Move(id=f"move_{i}", player=1, type=MoveType.MOVE_STACK,
                 from_pos=Position(x=i, y=0), to=Position(x=i, y=1))
            for i in range(3)
        ]
        root_node.untried_moves = moves.copy()

        root_node.add_child(moves[0], mock_game_state, prior=0.1)
        child2 = root_node.add_child(moves[1], mock_game_state, prior=0.8)
        root_node.add_child(moves[2], mock_game_state, prior=0.1)

        root_node.visits = 1  # Parent needs visits for UCT

        selected = root_node.uct_select_child(c_puct=1.0)

        # High prior child should be selected when all unvisited
        assert selected == child2

    def test_uct_select_child_balances_exploration(self, root_node, mock_game_state):
        """Test UCT balances exploration and exploitation."""
        moves = [
            Move(id=f"move_{i}", player=1, type=MoveType.MOVE_STACK,
                 from_pos=Position(x=i, y=0), to=Position(x=i, y=1))
            for i in range(2)
        ]
        root_node.untried_moves = moves.copy()

        child1 = root_node.add_child(moves[0], mock_game_state, prior=0.5)
        child2 = root_node.add_child(moves[1], mock_game_state, prior=0.5)

        # Child1 has many visits with low win rate
        child1.visits = 100
        child1.wins = 20  # 20% win rate

        # Child2 is unvisited
        child2.visits = 0

        root_node.visits = 100

        selected = root_node.uct_select_child(c_puct=1.0)

        # Unvisited child should get exploration bonus
        assert selected == child2


class TestMCTSNodeLite:
    """Tests for MCTSNodeLite class."""

    def test_initialization(self):
        """Test lightweight node initialization."""
        node = MCTSNodeLite(move=None, parent=None)

        assert node.move is None
        assert node.parent is None
        assert node.children == []
        assert node.wins == 0.0
        assert node.visits == 0

    def test_is_leaf(self):
        """Test is_leaf property."""
        node = MCTSNodeLite(move=None, parent=None)

        assert node.is_leaf()

        # Add a child
        child = MCTSNodeLite(move=MagicMock(), parent=node)
        node.children.append(child)

        assert not node.is_leaf()

    def test_is_fully_expanded(self):
        """Test is_fully_expanded property."""
        node = MCTSNodeLite(move=None, parent=None)
        node.untried_moves = [MagicMock(), MagicMock()]

        assert not node.is_fully_expanded()

        node.untried_moves = []
        assert node.is_fully_expanded()

    def test_add_child(self):
        """Test adding child to lite node."""
        parent = MCTSNodeLite(move=None, parent=None)
        move = MagicMock()
        parent.untried_moves = [move]

        child = parent.add_child(move, prior=0.7)

        assert child in parent.children
        assert child.move == move
        assert child.parent == parent
        assert child.prior == 0.7
        assert move not in parent.untried_moves

    def test_update(self):
        """Test update method."""
        node = MCTSNodeLite(move=None, parent=None)

        node.update(0.6)
        node.update(0.4)

        assert node.visits == 2
        assert node.wins == pytest.approx(1.0)


# =============================================================================
# Tests for DynamicBatchSizer
# =============================================================================

class TestDynamicBatchSizer:
    """Tests for DynamicBatchSizer class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        with patch('app.ai.mcts_ai.MemoryConfig') as mock_config:
            mock_config.from_env.return_value = MagicMock()
            sizer = DynamicBatchSizer()

            assert sizer.batch_size_min == 100
            assert sizer.batch_size_max == 1600
            assert sizer.memory_safety_margin == 0.8

    def test_initialization_custom(self):
        """Test custom initialization."""
        with patch('app.ai.mcts_ai.MemoryConfig') as mock_config:
            mock_config.from_env.return_value = MagicMock()
            sizer = DynamicBatchSizer(
                batch_size_min=50,
                batch_size_max=800,
                memory_safety_margin=0.9,
            )

            assert sizer.batch_size_min == 50
            assert sizer.batch_size_max == 800
            assert sizer.memory_safety_margin == 0.9

    def test_get_optimal_batch_size_clamped(self):
        """Test batch size is clamped to limits."""
        with patch('app.ai.mcts_ai.MemoryConfig') as mock_config, \
             patch('app.ai.mcts_ai.psutil') as mock_psutil:
            mock_mem_config = MagicMock()
            mock_mem_config.get_inference_limit_bytes.return_value = 1024 * 1024 * 1024  # 1GB
            mock_config.from_env.return_value = mock_mem_config

            mock_mem_info = MagicMock()
            mock_mem_info.available = 2 * 1024 * 1024 * 1024  # 2GB available
            mock_psutil.virtual_memory.return_value = mock_mem_info

            sizer = DynamicBatchSizer(batch_size_min=100, batch_size_max=500)
            batch_size = sizer.get_optimal_batch_size(current_node_count=0)

            # Should be clamped to max
            assert batch_size <= 500
            assert batch_size >= 100

    def test_stats(self):
        """Test stats method returns dict."""
        with patch('app.ai.mcts_ai.MemoryConfig') as mock_config:
            mock_config.from_env.return_value = MagicMock()
            sizer = DynamicBatchSizer()

            stats = sizer.stats()

            assert isinstance(stats, dict)
            assert 'batch_size_min' in stats
            assert 'batch_size_max' in stats
            assert 'adjustment_count' in stats


# =============================================================================
# Tests for Helper Functions
# =============================================================================

class TestMoveKey:
    """Tests for _move_key function."""

    def test_move_key_basic(self):
        """Test basic move key generation."""
        move = Move(
            id="test",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=2, y=3),
            to=Position(x=4, y=5),
        )

        key = _move_key(move)

        assert isinstance(key, tuple)
        assert MoveType.MOVE_STACK in key or "MOVEMENT" in str(key)

    def test_move_key_different_moves_different_keys(self):
        """Test different moves produce different keys."""
        move1 = Move(
            id="test1",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=2, y=3),
            to=Position(x=4, y=5),
        )
        move2 = Move(
            id="test2",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=0, y=0),
            to=Position(x=1, y=1),
        )

        assert _move_key(move1) != _move_key(move2)


class TestMovesMatch:
    """Tests for _moves_match function."""

    def test_identical_moves_match(self):
        """Test identical moves match."""
        move = Move(
            id="test",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=2, y=3),
            to=Position(x=4, y=5),
        )

        assert _moves_match(move, move)

    def test_same_content_different_id_match(self):
        """Test moves with same content but different IDs match."""
        move1 = Move(
            id="test1",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=2, y=3),
            to=Position(x=4, y=5),
        )
        move2 = Move(
            id="test2",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=2, y=3),
            to=Position(x=4, y=5),
        )

        assert _moves_match(move1, move2)

    def test_different_positions_dont_match(self):
        """Test moves with different positions don't match."""
        move1 = Move(
            id="test1",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=2, y=3),
            to=Position(x=4, y=5),
        )
        move2 = Move(
            id="test2",
            player=1,
            type=MoveType.MOVE_STACK,
            from_pos=Position(x=0, y=0),
            to=Position(x=1, y=1),
        )

        assert not _moves_match(move1, move2)


class TestPosKey:
    """Tests for _pos_key function."""

    def test_pos_key_creates_string(self):
        """Test _pos_key creates a string key."""
        pos = Position(x=3, y=7)

        key = _pos_key(pos)

        assert isinstance(key, str)
        assert "3" in key
        assert "7" in key

    def test_pos_key_none_returns_none(self):
        """Test _pos_key returns None for None input."""
        assert _pos_key(None) is None

    def test_pos_key_different_positions_different_keys(self):
        """Test different positions produce different keys."""
        pos1 = Position(x=1, y=2)
        pos2 = Position(x=3, y=4)

        assert _pos_key(pos1) != _pos_key(pos2)


class TestPosSeqKey:
    """Tests for _pos_seq_key function."""

    def test_pos_seq_key_tuple(self):
        """Test _pos_seq_key with position tuple."""
        positions = (Position(x=1, y=2), Position(x=3, y=4))

        key = _pos_seq_key(positions)

        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_pos_seq_key_none_returns_none(self):
        """Test _pos_seq_key returns None for None input."""
        assert _pos_seq_key(None) is None


# =============================================================================
# Tests for MCTS Pure Functions
# =============================================================================

class TestNormalizedEntropy:
    """Tests for _normalized_entropy method."""

    @pytest.fixture
    def mcts_ai(self):
        """Create MCTSAI instance for testing pure methods."""
        from app.ai.mcts_ai import MCTSAI
        from app.models import AIConfig

        with patch.object(MCTSAI, '__init__', lambda self, config: None):
            ai = MCTSAI.__new__(MCTSAI)
            ai.config = AIConfig(difficulty=5, think_time=1000)
            return ai

    def test_empty_list_returns_zero(self, mcts_ai):
        """Test empty list returns 0."""
        assert mcts_ai._normalized_entropy([]) == 0.0

    def test_single_element_returns_zero(self, mcts_ai):
        """Test single element returns 0 (no uncertainty)."""
        assert mcts_ai._normalized_entropy([1.0]) == 0.0

    def test_uniform_distribution_returns_one(self, mcts_ai):
        """Test uniform distribution returns 1 (max entropy)."""
        # Uniform distribution over 4 elements
        result = mcts_ai._normalized_entropy([0.25, 0.25, 0.25, 0.25])

        assert result == pytest.approx(1.0, rel=1e-5)

    def test_peaked_distribution_returns_low(self, mcts_ai):
        """Test peaked distribution returns low entropy."""
        # One dominant element
        result = mcts_ai._normalized_entropy([0.97, 0.01, 0.01, 0.01])

        assert result < 0.3

    def test_result_in_zero_one_range(self, mcts_ai):
        """Test result is always in [0, 1]."""
        test_cases = [
            [0.5, 0.5],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 0.0, 0.0],
        ]
        for priors in test_cases:
            result = mcts_ai._normalized_entropy(priors)
            assert 0.0 <= result <= 1.0


class TestDynamicCPuct:
    """Tests for _dynamic_c_puct method."""

    @pytest.fixture
    def mcts_ai(self):
        """Create MCTSAI instance for testing pure methods."""
        from app.ai.mcts_ai import MCTSAI
        from app.models import AIConfig

        with patch.object(MCTSAI, '__init__', lambda self, config: None):
            ai = MCTSAI.__new__(MCTSAI)
            ai.config = AIConfig(difficulty=5, think_time=1000)
            return ai

    def test_returns_float(self, mcts_ai):
        """Test method returns a float."""
        result = mcts_ai._dynamic_c_puct(100, [0.5, 0.5])

        assert isinstance(result, float)

    def test_result_in_valid_range(self, mcts_ai):
        """Test result is in valid range [0.25, 4.0]."""
        test_cases = [
            (0, []),
            (100, [0.5, 0.5]),
            (1000, [0.9, 0.05, 0.05]),
            (10, [0.25, 0.25, 0.25, 0.25]),
        ]
        for visits, priors in test_cases:
            result = mcts_ai._dynamic_c_puct(visits, priors)
            assert 0.25 <= result <= 4.0

    def test_higher_entropy_higher_exploration(self, mcts_ai):
        """Test high-entropy priors get higher exploration constant."""
        low_entropy = mcts_ai._dynamic_c_puct(100, [0.95, 0.05])
        high_entropy = mcts_ai._dynamic_c_puct(100, [0.5, 0.5])

        assert high_entropy >= low_entropy


class TestFpuReductionForPhase:
    """Tests for _fpu_reduction_for_phase method."""

    @pytest.fixture
    def mcts_ai(self):
        """Create MCTSAI instance for testing pure methods."""
        from app.ai.mcts_ai import MCTSAI
        from app.models import AIConfig

        with patch.object(MCTSAI, '__init__', lambda self, config: None):
            ai = MCTSAI.__new__(MCTSAI)
            ai.config = AIConfig(difficulty=5, think_time=1000)
            return ai

    def test_ring_placement_has_low_reduction(self, mcts_ai):
        """Test ring placement phase has low FPU reduction."""
        result = mcts_ai._fpu_reduction_for_phase(GamePhase.RING_PLACEMENT)

        assert result == 0.05

    def test_territory_processing_has_high_reduction(self, mcts_ai):
        """Test territory processing has higher FPU reduction."""
        result = mcts_ai._fpu_reduction_for_phase(GamePhase.TERRITORY_PROCESSING)

        assert result == 0.20

    def test_all_phases_return_float(self, mcts_ai):
        """Test all phases return a float."""
        phases = [
            GamePhase.RING_PLACEMENT,
            GamePhase.MOVEMENT,
            GamePhase.CAPTURE,
            GamePhase.LINE_PROCESSING,
        ]
        for phase in phases:
            result = mcts_ai._fpu_reduction_for_phase(phase)
            assert isinstance(result, float)
            assert result >= 0.0
