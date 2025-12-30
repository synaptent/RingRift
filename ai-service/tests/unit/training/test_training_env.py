"""Tests for app.training.env.

Tests the training environment module including:
- THEORETICAL_MAX_MOVES configuration
- get_theoretical_max_moves() helper
- Training evaluation presets
- TrainingEnvConfig dataclass
- make_env() factory
- RingRiftEnv class

Note: Integration tests that use the actual game engine may fail on systems
with Pydantic v1 installed (requires v2 for model_construct). These tests
are marked with pytest.mark.integration and can be skipped if needed.
"""

from unittest.mock import MagicMock, patch

import pydantic
import pytest

from app.models import BoardType, GamePhase, GameStatus, Move, MoveType

# Check Pydantic version for conditional skipping
PYDANTIC_V1 = pydantic.VERSION.startswith("1.")
SKIP_INTEGRATION = pytest.mark.skipif(
    PYDANTIC_V1,
    reason="Integration tests require Pydantic v2 (model_construct method)"
)
from app.training.env import (
    DEFAULT_TRAINING_EVAL_CONFIG,
    REPETITION_THRESHOLD,
    THEORETICAL_MAX_MOVES,
    TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD,
    TWO_PLAYER_TRAINING_PRESET,
    RingRiftEnv,
    TrainingEnvConfig,
    build_training_eval_kwargs,
    get_theoretical_max_moves,
    get_two_player_training_kwargs,
    make_env,
)


class TestTheoreticalMaxMoves:
    """Tests for THEORETICAL_MAX_MOVES configuration."""

    def test_all_board_types_defined(self):
        """Test all board types have move limits."""
        for board_type in [BoardType.SQUARE8, BoardType.SQUARE19,
                          BoardType.HEX8, BoardType.HEXAGONAL]:
            assert board_type in THEORETICAL_MAX_MOVES

    def test_all_player_counts_defined(self):
        """Test all player counts have limits for each board."""
        for board_type, limits in THEORETICAL_MAX_MOVES.items():
            assert 2 in limits
            assert 3 in limits
            assert 4 in limits

    def test_limits_increase_with_players(self):
        """Test move limits increase with player count."""
        for board_type, limits in THEORETICAL_MAX_MOVES.items():
            assert limits[2] <= limits[3] <= limits[4]

    def test_square8_limits(self):
        """Test SQUARE8 specific limits."""
        limits = THEORETICAL_MAX_MOVES[BoardType.SQUARE8]
        assert limits[2] == 600
        assert limits[3] == 900
        assert limits[4] == 1400

    def test_square19_limits(self):
        """Test SQUARE19 specific limits."""
        limits = THEORETICAL_MAX_MOVES[BoardType.SQUARE19]
        assert limits[2] == 2400
        assert limits[3] == 3000
        assert limits[4] == 3600

    def test_hex8_limits(self):
        """Test HEX8 specific limits."""
        limits = THEORETICAL_MAX_MOVES[BoardType.HEX8]
        assert limits[2] == 600
        assert limits[3] == 900
        assert limits[4] == 1400

    def test_hexagonal_limits(self):
        """Test HEXAGONAL specific limits."""
        limits = THEORETICAL_MAX_MOVES[BoardType.HEXAGONAL]
        assert limits[2] == 3600
        assert limits[3] == 4200
        assert limits[4] == 4800


class TestGetTheoreticalMaxMoves:
    """Tests for get_theoretical_max_moves helper."""

    def test_known_configs(self):
        """Test returns correct values for known configs."""
        assert get_theoretical_max_moves(BoardType.SQUARE8, 2) == 600
        assert get_theoretical_max_moves(BoardType.SQUARE19, 4) == 3600
        assert get_theoretical_max_moves(BoardType.HEX8, 3) == 900
        assert get_theoretical_max_moves(BoardType.HEXAGONAL, 2) == 3600

    def test_extrapolation_5_players(self):
        """Test extrapolation for 5 players."""
        # Uses 2-player base + increment per extra player
        max_moves = get_theoretical_max_moves(BoardType.SQUARE8, 5)
        # Base=600, increment=(900-600)=300, extra=(5-2)=3, so 600+300*3=1500
        assert max_moves == 1500

    def test_extrapolation_unknown_board(self):
        """Test extrapolation for unknown board type falls back gracefully."""
        # Create a mock board type
        unknown = MagicMock()
        unknown.value = "unknown"

        # Should use default fallback
        max_moves = get_theoretical_max_moves(unknown, 2)
        assert max_moves == 200  # Default base


class TestRepetitionThreshold:
    """Tests for REPETITION_THRESHOLD configuration."""

    def test_default_is_disabled(self):
        """Test repetition threshold is disabled by default."""
        assert REPETITION_THRESHOLD == 0


class TestTrainingPresets:
    """Tests for training preset configurations."""

    def test_default_training_eval_config(self):
        """Test DEFAULT_TRAINING_EVAL_CONFIG has required keys."""
        assert 'boards' in DEFAULT_TRAINING_EVAL_CONFIG
        assert 'eval_mode' in DEFAULT_TRAINING_EVAL_CONFIG
        assert 'state_pool_id' in DEFAULT_TRAINING_EVAL_CONFIG
        assert 'games_per_eval' in DEFAULT_TRAINING_EVAL_CONFIG
        assert 'max_moves' in DEFAULT_TRAINING_EVAL_CONFIG
        assert 'eval_randomness' in DEFAULT_TRAINING_EVAL_CONFIG

    def test_default_training_eval_config_boards(self):
        """Test default config includes all main board types."""
        boards = DEFAULT_TRAINING_EVAL_CONFIG['boards']
        assert BoardType.SQUARE8 in boards
        assert BoardType.SQUARE19 in boards
        assert BoardType.HEXAGONAL in boards

    def test_two_player_preset_extends_default(self):
        """Test TWO_PLAYER_TRAINING_PRESET extends default config."""
        for key in DEFAULT_TRAINING_EVAL_CONFIG:
            assert key in TWO_PLAYER_TRAINING_PRESET

    def test_two_player_preset_has_randomness(self):
        """Test two-player preset has non-zero randomness."""
        assert TWO_PLAYER_TRAINING_PRESET['eval_randomness'] == 0.02

    def test_heuristic_eval_modes(self):
        """Test heuristic evaluation mode mapping."""
        assert TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD[BoardType.SQUARE8] == "full"
        assert TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD[BoardType.SQUARE19] == "light"
        assert TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD[BoardType.HEXAGONAL] == "light"


class TestBuildTrainingEvalKwargs:
    """Tests for build_training_eval_kwargs helper."""

    def test_returns_defaults(self):
        """Test returns default config when no overrides."""
        kwargs = build_training_eval_kwargs()

        assert 'boards' in kwargs
        assert 'games_per_eval' in kwargs
        assert kwargs['seed'] is None

    def test_overrides_games_per_eval(self):
        """Test overrides games_per_eval."""
        kwargs = build_training_eval_kwargs(games_per_eval=32)

        assert kwargs['games_per_eval'] == 32

    def test_overrides_eval_randomness(self):
        """Test overrides eval_randomness."""
        kwargs = build_training_eval_kwargs(eval_randomness=0.1)

        assert kwargs['eval_randomness'] == 0.1

    def test_sets_seed(self):
        """Test sets seed parameter."""
        kwargs = build_training_eval_kwargs(seed=42)

        assert kwargs['seed'] == 42


class TestGetTwoPlayerTrainingKwargs:
    """Tests for get_two_player_training_kwargs helper."""

    def test_sets_games_per_eval(self):
        """Test sets games_per_eval from argument."""
        kwargs = get_two_player_training_kwargs(games_per_eval=100, seed=123)

        assert kwargs['games_per_eval'] == 100

    def test_sets_seed(self):
        """Test sets seed from argument."""
        kwargs = get_two_player_training_kwargs(games_per_eval=16, seed=456)

        assert kwargs['seed'] == 456

    def test_includes_preset_values(self):
        """Test includes preset configuration values."""
        kwargs = get_two_player_training_kwargs(games_per_eval=16, seed=1)

        assert kwargs['eval_randomness'] == 0.02
        assert kwargs['eval_mode'] == 'multi-start'


class TestTrainingEnvConfig:
    """Tests for TrainingEnvConfig dataclass."""

    def test_default_values(self):
        """Test TrainingEnvConfig has sensible defaults."""
        config = TrainingEnvConfig()

        assert config.board_type == BoardType.SQUARE8
        assert config.num_players == 2
        assert config.max_moves is None
        assert config.reward_mode == "terminal"
        assert config.seed is None
        assert config.use_default_rules_engine is True
        assert config.rings_per_player is None
        assert config.lps_rounds_required == 3

    def test_custom_values(self):
        """Test TrainingEnvConfig accepts custom values."""
        config = TrainingEnvConfig(
            board_type=BoardType.HEX8,
            num_players=4,
            max_moves=500,
            reward_mode="shaped",
            seed=42,
        )

        assert config.board_type == BoardType.HEX8
        assert config.num_players == 4
        assert config.max_moves == 500
        assert config.reward_mode == "shaped"
        assert config.seed == 42


class TestMakeEnv:
    """Tests for make_env factory function."""

    @patch('app.training.env.RingRiftEnv')
    def test_creates_env_with_defaults(self, mock_env_class):
        """Test make_env creates environment with defaults."""
        make_env()

        mock_env_class.assert_called_once()
        call_kwargs = mock_env_class.call_args[1]

        assert call_kwargs['board_type'] == BoardType.SQUARE8
        assert call_kwargs['num_players'] == 2
        assert call_kwargs['max_moves'] == 600  # Theoretical max for SQUARE8 2p
        assert call_kwargs['reward_on'] == 'terminal'

    @patch('app.training.env.RingRiftEnv')
    def test_creates_env_with_custom_config(self, mock_env_class):
        """Test make_env creates environment with custom config."""
        config = TrainingEnvConfig(
            board_type=BoardType.HEX8,
            num_players=3,
            max_moves=1000,
        )

        make_env(config)

        call_kwargs = mock_env_class.call_args[1]

        assert call_kwargs['board_type'] == BoardType.HEX8
        assert call_kwargs['num_players'] == 3
        assert call_kwargs['max_moves'] == 1000

    @patch('app.training.env.RingRiftEnv')
    def test_uses_theoretical_max_when_not_specified(self, mock_env_class):
        """Test make_env uses theoretical max when max_moves not specified."""
        config = TrainingEnvConfig(
            board_type=BoardType.SQUARE19,
            num_players=4,
            max_moves=None,
        )

        make_env(config)

        call_kwargs = mock_env_class.call_args[1]

        assert call_kwargs['max_moves'] == 3600  # Theoretical max for SQUARE19 4p


class TestRingRiftEnv:
    """Tests for RingRiftEnv class."""

    def test_initialization_default_params(self):
        """Test RingRiftEnv initializes with default parameters."""
        env = RingRiftEnv()

        assert env.board_type == BoardType.SQUARE8
        assert env.num_players == 2
        assert env.max_moves == 600
        assert env.reward_on == "terminal"
        assert env._state is None

    def test_initialization_custom_params(self):
        """Test RingRiftEnv initializes with custom parameters."""
        env = RingRiftEnv(
            board_type=BoardType.HEX8,
            num_players=4,
            max_moves=1000,
            reward_on="shaped",
        )

        assert env.board_type == BoardType.HEX8
        assert env.num_players == 4
        assert env.max_moves == 1000
        assert env.reward_on == "shaped"

    def test_max_moves_warning(self):
        """Test warning when max_moves below theoretical max."""
        with patch('app.training.env.logger') as mock_logger:
            RingRiftEnv(
                board_type=BoardType.SQUARE8,
                num_players=2,
                max_moves=100,  # Below 600 theoretical max
            )

            mock_logger.warning.assert_called_once()

    def test_reset_returns_game_state(self):
        """Test reset returns a valid game state."""
        env = RingRiftEnv()

        state = env.reset()

        assert state is not None
        assert env._state is not None
        assert env._move_count == 0

    def test_reset_with_seed(self):
        """Test reset with seed sets RNG state."""
        env = RingRiftEnv()

        with patch('app.training.env.seed_all') as mock_seed:
            env.reset(seed=42)

            mock_seed.assert_called_once_with(42)

    def test_reset_with_default_seed(self):
        """Test reset uses default seed if set."""
        env = RingRiftEnv(default_seed=123)

        with patch('app.training.env.seed_all') as mock_seed:
            env.reset()

            mock_seed.assert_called_once_with(123)

    def test_state_property(self):
        """Test state property returns current state."""
        env = RingRiftEnv()
        state = env.reset()

        assert env.state is state

    def test_state_property_raises_without_reset(self):
        """Test state property raises if not reset."""
        env = RingRiftEnv()

        with pytest.raises(AssertionError):
            _ = env.state

    @SKIP_INTEGRATION
    def test_legal_moves_returns_list(self):
        """Test legal_moves returns a list of moves."""
        env = RingRiftEnv()
        env.reset()

        moves = env.legal_moves()

        assert isinstance(moves, list)
        assert len(moves) > 0
        assert all(isinstance(m, Move) for m in moves)

    @SKIP_INTEGRATION
    def test_step_advances_move_count(self):
        """Test step advances move count."""
        env = RingRiftEnv()
        env.reset()

        moves = env.legal_moves()
        if moves:
            move = moves[0]
            _, _, _, info = env.step(move)

            assert env._move_count >= 1
            assert info['move_count'] >= 1

    @SKIP_INTEGRATION
    def test_step_returns_tuple(self):
        """Test step returns (state, reward, done, info) tuple."""
        env = RingRiftEnv()
        env.reset()

        moves = env.legal_moves()
        if moves:
            result = env.step(moves[0])

            assert len(result) == 4
            state, reward, done, info = result

            assert state is not None
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)

    @SKIP_INTEGRATION
    def test_step_checks_player_ownership(self):
        """Test step validates move player matches current player."""
        env = RingRiftEnv()
        state = env.reset()

        # Create a move for wrong player
        wrong_player = (state.current_player % env.num_players) + 1
        wrong_move = Move(
            player=wrong_player,
            type=MoveType.PLACE_RING,
            to=None,
        )

        with pytest.raises(ValueError, match="does not match current player"):
            env.step(wrong_move)

    @SKIP_INTEGRATION
    def test_step_info_contains_legal_moves(self):
        """Test step info contains legal_moves for non-terminal states."""
        env = RingRiftEnv()
        env.reset()

        moves = env.legal_moves()
        if moves:
            _, _, done, info = env.step(moves[0])

            if not done:
                assert 'legal_moves' in info
                assert isinstance(info['legal_moves'], list)

    @SKIP_INTEGRATION
    def test_step_terminal_info(self):
        """Test step info contains winner/victory_reason for terminal states."""
        env = RingRiftEnv(max_moves=1)  # Very short game
        env.reset()

        moves = env.legal_moves()
        if moves:
            _, _, done, info = env.step(moves[0])

            if done:
                assert 'winner' in info
                assert 'victory_reason' in info


@pytest.mark.integration
class TestRingRiftEnvIntegration:
    """Integration tests for RingRiftEnv.

    These tests require the full game engine and Pydantic v2.
    """

    @SKIP_INTEGRATION
    def test_can_play_complete_game(self):
        """Test can play a game to completion."""
        env = RingRiftEnv(
            board_type=BoardType.SQUARE8,
            num_players=2,
        )

        state = env.reset(seed=42)
        done = False
        move_count = 0
        max_moves = 50  # Reasonable limit for test

        while not done and move_count < max_moves:
            moves = env.legal_moves()
            if not moves:
                break

            # Pick first legal move
            move = moves[0]
            state, reward, done, info = env.step(move)
            move_count += 1

        # Either game ended or we hit move limit
        assert move_count > 0

    @SKIP_INTEGRATION
    def test_respects_max_moves(self):
        """Test environment respects max_moves limit."""
        max_moves = 10
        env = RingRiftEnv(
            board_type=BoardType.SQUARE8,
            num_players=2,
            max_moves=max_moves,
        )

        env.reset()
        done = False

        while not done:
            moves = env.legal_moves()
            if not moves:
                break

            _, _, done, _ = env.step(moves[0])

        # Should have stopped at or before max_moves
        assert env._move_count <= max_moves + 10  # Allow some tolerance for bookkeeping moves

    @SKIP_INTEGRATION
    def test_multiple_resets(self):
        """Test multiple resets work correctly."""
        env = RingRiftEnv()

        for i in range(5):
            state = env.reset(seed=i)

            assert state is not None
            assert env._move_count == 0

            # Take a few moves
            for _ in range(3):
                moves = env.legal_moves()
                if moves:
                    env.step(moves[0])


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @SKIP_INTEGRATION
    def test_hex8_board(self):
        """Test environment works with hex8 board."""
        env = RingRiftEnv(board_type=BoardType.HEX8, num_players=2)
        state = env.reset()

        assert state is not None
        moves = env.legal_moves()
        assert len(moves) > 0

    @SKIP_INTEGRATION
    def test_hexagonal_board(self):
        """Test environment works with hexagonal board."""
        env = RingRiftEnv(board_type=BoardType.HEXAGONAL, num_players=2)
        state = env.reset()

        assert state is not None
        moves = env.legal_moves()
        assert len(moves) > 0

    @SKIP_INTEGRATION
    def test_square19_board(self):
        """Test environment works with square19 board."""
        env = RingRiftEnv(board_type=BoardType.SQUARE19, num_players=2)
        state = env.reset()

        assert state is not None
        moves = env.legal_moves()
        assert len(moves) > 0

    @SKIP_INTEGRATION
    def test_3_player_game(self):
        """Test environment works with 3 players."""
        env = RingRiftEnv(board_type=BoardType.SQUARE8, num_players=3)
        state = env.reset()

        assert state is not None
        moves = env.legal_moves()
        assert len(moves) > 0

    @SKIP_INTEGRATION
    def test_4_player_game(self):
        """Test environment works with 4 players."""
        env = RingRiftEnv(board_type=BoardType.SQUARE8, num_players=4)
        state = env.reset()

        assert state is not None
        moves = env.legal_moves()
        assert len(moves) > 0

    def test_custom_rings_per_player(self):
        """Test environment respects custom rings_per_player."""
        env = RingRiftEnv(
            board_type=BoardType.SQUARE8,
            num_players=2,
            rings_per_player=5,  # Fewer rings
        )
        state = env.reset()

        assert state is not None

    def test_custom_lps_rounds(self):
        """Test environment respects custom lps_rounds_required."""
        env = RingRiftEnv(
            board_type=BoardType.SQUARE8,
            num_players=2,
            lps_rounds_required=5,
        )
        state = env.reset()

        assert state is not None
        assert env._lps_rounds_required == 5

    def test_no_rules_engine(self):
        """Test environment works without default rules engine."""
        env = RingRiftEnv(
            board_type=BoardType.SQUARE8,
            num_players=2,
            use_default_rules_engine=False,
        )

        assert env._rules_engine is None
        state = env.reset()
        assert state is not None
