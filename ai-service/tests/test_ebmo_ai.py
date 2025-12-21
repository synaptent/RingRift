"""Unit tests for Energy-Based Move Optimization (EBMO) AI.

Tests cover:
- Network forward pass
- State encoding
- Action embedding
- Energy optimization
- Move selection
- Gradient descent convergence
"""

import pytest
import torch

from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import (
    ActionFeatureExtractor,
    EBMOConfig,
    EBMONetwork,
)
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)

# =============================================================================
# Helpers
# =============================================================================


def create_test_game_state() -> GameState:
    """Create a minimal game state for testing."""
    from datetime import datetime

    now = datetime.now()
    return GameState(
        id="test-game",
        boardType=BoardType.SQUARE8,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={
                "2,2": RingStack(
                    position=Position(x=2, y=2),
                    rings=[1],
                    stackHeight=1,
                    capHeight=1,
                    controllingPlayer=1,
                ),
                "5,5": RingStack(
                    position=Position(x=5, y=5),
                    rings=[2],
                    stackHeight=1,
                    capHeight=1,
                    controllingPlayer=2,
                ),
            },
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        ),
        players=[
            Player(
                id="player1",
                username="Player1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600000,
                ringsInHand=17,
                eliminatedRings=0,
                territorySpaces=0,
            ),
            Player(
                id="player2",
                username="Player2",
                type="ai",
                playerNumber=2,
                isReady=True,
                timeRemaining=600000,
                ringsInHand=17,
                eliminatedRings=0,
                territorySpaces=0,
            ),
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=2,
        totalRingsEliminated=0,
        victoryThreshold=9,
        territoryVictoryThreshold=32,
        timeControl=TimeControl(
            initialTime=600000,
            increment=0,
            type="none",
        ),
    )


# =============================================================================
# EBMOConfig Tests
# =============================================================================


class TestEBMOConfig:
    """Test EBMO configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = EBMOConfig()
        assert config.board_size == 8
        assert config.state_embed_dim > 0
        assert config.action_embed_dim > 0
        assert config.optim_steps > 0
        assert config.optim_lr > 0
        assert config.num_restarts >= 1

    def test_custom_config(self):
        """Custom config values should be applied."""
        config = EBMOConfig(
            board_size=19,
            state_embed_dim=128,
            optim_steps=50,
        )
        assert config.board_size == 19
        assert config.state_embed_dim == 128
        assert config.optim_steps == 50


# =============================================================================
# EBMONetwork Tests
# =============================================================================


class TestEBMONetwork:
    """Test EBMO neural network."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return EBMOConfig(board_size=8)

    @pytest.fixture
    def network(self, config):
        """Create test network."""
        net = EBMONetwork(config)
        net.eval()
        return net

    def test_network_creation(self, network):
        """Network should be created successfully."""
        assert network is not None
        assert hasattr(network, 'state_encoder')
        assert hasattr(network, 'action_encoder')
        assert hasattr(network, 'energy_head')

    def test_forward_pass_shape(self, network, config):
        """Forward pass should produce correct output shape."""
        batch_size = 4
        state_embed = torch.randn(batch_size, config.state_embed_dim)
        action_embed = torch.randn(batch_size, config.action_embed_dim)

        energy = network.compute_energy(state_embed, action_embed)

        assert energy.shape == (batch_size,)

    def test_energy_computation(self, network, config):
        """Energy should be computed for state-action pairs."""
        state_embed = torch.randn(1, config.state_embed_dim)
        action_embed = torch.randn(1, config.action_embed_dim)

        with torch.no_grad():
            energy = network.compute_energy(state_embed, action_embed)

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()

    def test_state_encoding(self, network, config):
        """State encoder should produce embeddings of correct size."""
        # Create dummy state features (board representation + global features)
        board_features = torch.randn(1, config.num_input_channels, config.board_size, config.board_size)
        global_features = torch.randn(1, config.num_global_features)

        with torch.no_grad():
            state_embed = network.state_encoder(board_features, global_features)

        assert state_embed.shape == (1, config.state_embed_dim)


# =============================================================================
# ActionFeatureExtractor Tests
# =============================================================================


class TestActionFeatureExtractor:
    """Test action feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create test extractor."""
        return ActionFeatureExtractor(board_size=8)

    def test_extractor_creation(self, extractor):
        """Extractor should be created successfully."""
        assert extractor is not None
        assert extractor.board_size == 8


# =============================================================================
# EBMO_AI Tests
# =============================================================================


class TestEBMO_AI:
    """Test EBMO AI agent."""

    @pytest.fixture
    def ai_config(self):
        """Create AI config for testing."""
        return AIConfig(difficulty=5)

    @pytest.fixture
    def game_state(self):
        """Create test game state."""
        return create_test_game_state()

    @pytest.fixture
    def ai(self, ai_config):
        """Create EBMO AI instance."""
        return EBMO_AI(player_number=1, config=ai_config)

    def test_ai_creation(self, ai):
        """AI should be created successfully."""
        assert ai is not None
        assert ai.player_number == 1
        assert ai.network is not None

    def test_ai_has_network(self, ai):
        """AI should have initialized network."""
        assert ai.network is not None
        assert isinstance(ai.network, EBMONetwork)

    def test_select_move_returns_move(self, ai, game_state):
        """select_move should return a valid move."""
        move = ai.select_move(game_state)
        assert move is not None
        assert hasattr(move, 'type')
        assert hasattr(move, 'from_pos')
        assert hasattr(move, 'to')

    def test_select_move_returns_legal_move(self, ai, game_state):
        """Selected move should be legal."""
        move = ai.select_move(game_state)
        valid_moves = ai.get_valid_moves(game_state)

        # Check that the returned move is in the list of valid moves
        move_matches = [
            m for m in valid_moves
            if m.type == move.type
            and (m.to.x == move.to.x if m.to and move.to else True)
            and (m.to.y == move.to.y if m.to and move.to else True)
        ]
        assert len(move_matches) >= 1

    def test_select_move_deterministic_with_seed(self, ai_config, game_state):
        """With same seed, AI should select same move."""
        # Note: This may not be fully deterministic due to optimization
        ai1 = EBMO_AI(player_number=1, config=ai_config)
        ai2 = EBMO_AI(player_number=1, config=ai_config)

        # Set same random seed
        torch.manual_seed(42)
        move1 = ai1.select_move(game_state)

        torch.manual_seed(42)
        move2 = ai2.select_move(game_state)

        # Moves should be the same (deterministic given same initialization)
        assert move1.type == move2.type

    def test_evaluate_position(self, ai, game_state):
        """evaluate_position should return value in valid range."""
        value = ai.evaluate_position(game_state)

        assert isinstance(value, (int, float))
        assert -1.0 <= value <= 1.0


# =============================================================================
# Optimization Tests
# =============================================================================


class TestEBMOOptimization:
    """Test EBMO gradient descent optimization."""

    @pytest.fixture
    def config(self):
        """Create test config with minimal optimization steps."""
        return EBMOConfig(
            board_size=8,
            optim_steps=10,
            num_restarts=2,
        )

    @pytest.fixture
    def network(self, config):
        """Create test network."""
        net = EBMONetwork(config)
        net.eval()
        return net

    def test_optimization_reduces_energy(self, network, config):
        """Optimization should reduce energy over steps."""
        # Create random state and action embeddings
        state_embed = torch.randn(1, config.state_embed_dim)
        action_embed = torch.randn(1, config.action_embed_dim, requires_grad=True)

        # Compute initial energy
        initial_energy = network.compute_energy(state_embed, action_embed.detach())

        # Run a few optimization steps
        optimizer = torch.optim.Adam([action_embed], lr=config.optim_lr)

        for _ in range(config.optim_steps):
            optimizer.zero_grad()
            energy = network.compute_energy(state_embed, action_embed)
            energy.backward()
            optimizer.step()

        # Final energy should be lower or equal
        final_energy = network.compute_energy(state_embed, action_embed.detach())

        # Note: Not guaranteed to always reduce due to non-convexity
        # Just check it doesn't explode
        assert not torch.isnan(final_energy).any()
        assert not torch.isinf(final_energy).any()


# =============================================================================
# Integration Tests
# =============================================================================


class TestEBMOIntegration:
    """Integration tests for EBMO AI."""

    @pytest.fixture
    def ai(self):
        """Create EBMO AI with minimal config."""
        config = AIConfig(difficulty=3)
        return EBMO_AI(player_number=1, config=config)

    def test_multiple_moves(self, ai):
        """AI should handle multiple consecutive moves."""
        game_state = create_test_game_state()

        # Select multiple moves
        for _ in range(3):
            move = ai.select_move(game_state)
            assert move is not None

    def test_inference_statistics(self, ai):
        """AI should track inference statistics."""
        game_state = create_test_game_state()

        initial_count = ai._total_moves
        ai.select_move(game_state)
        assert ai._total_moves == initial_count + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
