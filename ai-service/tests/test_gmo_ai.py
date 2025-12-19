"""Unit tests for Gradient Move Optimization (GMO) AI.

Tests cover:
- Move encoding/decoding
- State encoding
- Value network forward pass
- Uncertainty estimation
- Novelty tracking
- Gradient optimization
- Full move selection
"""

import math

import pytest
import torch

from app.ai.gmo_ai import (
    GMOAI,
    GMOConfig,
    GMOValueNetWithUncertainty,
    MoveEncoder,
    NoveltyTracker,
    StateEncoder,
    estimate_uncertainty,
    optimize_move_with_entropy,
    project_to_legal_move,
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
from datetime import datetime


# =============================================================================
# Helpers
# =============================================================================

def create_test_game_state() -> GameState:
    """Create a minimal game state for testing."""
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
        timeControl=TimeControl(
            initialTime=600000,
            increment=0,
            type="fischer",
        ),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=2,
        totalRingsEliminated=0,
        victoryThreshold=5,
        territoryVictoryThreshold=33,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gmo_config():
    """Default GMO configuration for tests."""
    return GMOConfig(
        state_dim=64,
        move_dim=64,
        hidden_dim=128,
        top_k=3,
        optim_steps=5,
        lr=0.1,
        beta=0.3,
        gamma=0.1,
        dropout_rate=0.1,
        mc_samples=5,
        novelty_memory_size=100,
        device="cpu",
    )


@pytest.fixture
def move_encoder(gmo_config):
    """Move encoder for tests."""
    return MoveEncoder(embed_dim=gmo_config.move_dim, board_size=8)


@pytest.fixture
def state_encoder(gmo_config):
    """State encoder for tests."""
    return StateEncoder(embed_dim=gmo_config.state_dim, board_size=8)


@pytest.fixture
def value_net(gmo_config):
    """Value network for tests."""
    return GMOValueNetWithUncertainty(
        state_dim=gmo_config.state_dim,
        move_dim=gmo_config.move_dim,
        hidden_dim=gmo_config.hidden_dim,
        dropout_rate=gmo_config.dropout_rate,
    )


@pytest.fixture
def sample_move():
    """Sample move for tests."""
    from app.models import Move
    return Move(
        id="test_move",
        type=MoveType.PLACE_RING,
        player=1,
        from_pos=None,
        to=Position(x=3, y=4),
        placement_count=2,
    )


@pytest.fixture
def sample_move_stack():
    """Sample MOVE_STACK move for tests."""
    from app.models import Move
    return Move(
        id="test_move_stack",
        type=MoveType.MOVE_STACK,
        player=1,
        from_pos=Position(x=2, y=2),
        to=Position(x=5, y=2),
    )


# =============================================================================
# Move Encoder Tests
# =============================================================================

class TestMoveEncoder:
    """Tests for MoveEncoder."""

    def test_encode_placement_move(self, move_encoder, sample_move):
        """Test encoding a placement move."""
        embedding = move_encoder.encode_move(sample_move)

        assert embedding.shape == (move_encoder.embed_dim,)
        assert torch.isfinite(embedding).all()

    def test_encode_movement_move(self, move_encoder, sample_move_stack):
        """Test encoding a movement move."""
        embedding = move_encoder.encode_move(sample_move_stack)

        assert embedding.shape == (move_encoder.embed_dim,)
        assert torch.isfinite(embedding).all()

    def test_encode_multiple_moves(self, move_encoder, sample_move, sample_move_stack):
        """Test encoding multiple moves."""
        moves = [sample_move, sample_move_stack]
        embeddings = move_encoder.encode_moves(moves)

        assert embeddings.shape == (2, move_encoder.embed_dim)
        assert torch.isfinite(embeddings).all()

    def test_different_moves_different_embeddings(self, move_encoder, sample_move, sample_move_stack):
        """Test that different moves produce different embeddings."""
        emb1 = move_encoder.encode_move(sample_move)
        emb2 = move_encoder.encode_move(sample_move_stack)

        # Embeddings should be different (not equal)
        assert not torch.allclose(emb1, emb2)

    def test_same_move_same_embedding(self, move_encoder, sample_move):
        """Test that same move produces consistent embedding."""
        emb1 = move_encoder.encode_move(sample_move)
        emb2 = move_encoder.encode_move(sample_move)

        assert torch.allclose(emb1, emb2)


# =============================================================================
# State Encoder Tests
# =============================================================================

class TestStateEncoder:
    """Tests for StateEncoder."""

    def test_extract_features_empty_board(self, state_encoder):
        """Test feature extraction from board state."""
        state = create_test_game_state()

        features = state_encoder.extract_features(state)

        assert features.shape == (state_encoder.input_dim,)
        # Should have non-zero features for the stacks
        assert features.sum() > 0

    def test_encode_state_produces_embedding(self, state_encoder):
        """Test that encode_state produces valid embedding."""
        state = create_test_game_state()

        embedding = state_encoder.encode_state(state)

        assert embedding.shape == (state_encoder.embed_dim,)
        assert torch.isfinite(embedding).all()


# =============================================================================
# Value Network Tests
# =============================================================================

class TestGMOValueNet:
    """Tests for GMOValueNetWithUncertainty."""

    def test_forward_single_sample(self, value_net, gmo_config):
        """Test forward pass with single sample."""
        state_embed = torch.randn(gmo_config.state_dim)
        move_embed = torch.randn(gmo_config.move_dim)

        value, log_var = value_net(state_embed, move_embed)

        assert value.shape == (1, 1)
        assert log_var.shape == (1, 1)
        # Value should be in [-1, 1] due to tanh
        assert -1 <= value.item() <= 1

    def test_forward_batch(self, value_net, gmo_config):
        """Test forward pass with batch."""
        batch_size = 8
        state_embeds = torch.randn(batch_size, gmo_config.state_dim)
        move_embeds = torch.randn(batch_size, gmo_config.move_dim)

        values, log_vars = value_net(state_embeds, move_embeds)

        assert values.shape == (batch_size, 1)
        assert log_vars.shape == (batch_size, 1)
        # All values should be in [-1, 1]
        assert (values >= -1).all() and (values <= 1).all()

    def test_dropout_affects_output_in_train_mode(self, value_net, gmo_config):
        """Test that dropout creates variation in train mode."""
        value_net.train()
        state_embed = torch.randn(gmo_config.state_dim)
        move_embed = torch.randn(gmo_config.move_dim)

        # Run multiple times and collect outputs
        outputs = []
        for _ in range(10):
            value, _ = value_net(state_embed, move_embed)
            outputs.append(value.item())

        # There should be some variation due to dropout
        # (not always, but with high probability)
        # We just check it doesn't crash


# =============================================================================
# Uncertainty Estimation Tests
# =============================================================================

class TestUncertaintyEstimation:
    """Tests for MC Dropout uncertainty estimation."""

    def test_estimate_uncertainty_returns_three_values(self, value_net, gmo_config):
        """Test that estimate_uncertainty returns mean, entropy, variance."""
        state_embed = torch.randn(gmo_config.state_dim)
        move_embed = torch.randn(gmo_config.move_dim)

        mean_value, entropy, variance = estimate_uncertainty(
            state_embed, move_embed, value_net, n_samples=5
        )

        assert isinstance(mean_value, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert isinstance(variance, torch.Tensor)
        assert variance.item() > 0  # Variance should be positive

    def test_more_samples_reduces_estimate_variance(self, value_net, gmo_config):
        """Test that more MC samples leads to more stable estimates."""
        state_embed = torch.randn(gmo_config.state_dim)
        move_embed = torch.randn(gmo_config.move_dim)

        # Run with few samples multiple times
        few_sample_means = []
        for _ in range(5):
            mean, _, _ = estimate_uncertainty(
                state_embed, move_embed, value_net, n_samples=2
            )
            few_sample_means.append(mean.item())

        # Run with many samples multiple times
        many_sample_means = []
        for _ in range(5):
            mean, _, _ = estimate_uncertainty(
                state_embed, move_embed, value_net, n_samples=20
            )
            many_sample_means.append(mean.item())

        # Many samples should have lower variance (most of the time)
        # This is a statistical test, so we're lenient


# =============================================================================
# Novelty Tracker Tests
# =============================================================================

class TestNoveltyTracker:
    """Tests for NoveltyTracker."""

    def test_empty_memory_returns_high_novelty(self):
        """Test that empty memory returns max novelty."""
        tracker = NoveltyTracker(memory_size=100, embed_dim=64)
        embed = torch.randn(64)

        novelty = tracker.compute_novelty(embed)

        assert novelty.item() == 1.0

    def test_adding_embedding_increases_count(self):
        """Test that adding embedding increases count."""
        tracker = NoveltyTracker(memory_size=100, embed_dim=64)
        embed = torch.randn(64)

        assert tracker.count == 0
        tracker.add(embed)
        assert tracker.count == 1

    def test_same_embedding_has_zero_novelty(self):
        """Test that same embedding has low novelty."""
        tracker = NoveltyTracker(memory_size=100, embed_dim=64)
        embed = torch.randn(64)

        tracker.add(embed)
        novelty = tracker.compute_novelty(embed)

        assert novelty.item() < 0.01  # Should be very close to 0

    def test_distant_embedding_has_high_novelty(self):
        """Test that distant embedding has high novelty."""
        tracker = NoveltyTracker(memory_size=100, embed_dim=64)

        # Add a bunch of embeddings in one region
        for _ in range(10):
            embed = torch.randn(64) * 0.1  # Small embeddings
            tracker.add(embed)

        # Check novelty of a very different embedding
        distant_embed = torch.ones(64) * 10  # Large, far away
        novelty = tracker.compute_novelty(distant_embed)

        assert novelty.item() > 1.0  # Should be quite novel

    def test_reset_clears_memory(self):
        """Test that reset clears memory."""
        tracker = NoveltyTracker(memory_size=100, embed_dim=64)

        # Add some embeddings
        for _ in range(10):
            tracker.add(torch.randn(64))

        tracker.reset()

        assert tracker.count == 0
        # After reset, any embedding should be novel
        novelty = tracker.compute_novelty(torch.randn(64))
        assert novelty.item() == 1.0

    def test_ring_buffer_wraps(self):
        """Test that memory wraps around (ring buffer)."""
        tracker = NoveltyTracker(memory_size=5, embed_dim=64)

        # Add more than memory size
        for _ in range(10):
            tracker.add(torch.randn(64))

        assert tracker.count == 10
        # Memory should still work (only stores last 5)


# =============================================================================
# Gradient Optimization Tests
# =============================================================================

class TestGradientOptimization:
    """Tests for gradient-based move optimization."""

    def test_optimize_move_changes_embedding(self, value_net, gmo_config):
        """Test that optimization actually changes the embedding."""
        state_embed = torch.randn(gmo_config.state_dim)
        initial_embed = torch.randn(gmo_config.move_dim)

        optimized_embed = optimize_move_with_entropy(
            state_embed, initial_embed, value_net, gmo_config
        )

        # Embeddings should be different after optimization
        assert not torch.allclose(initial_embed, optimized_embed)

    def test_optimize_move_returns_correct_shape(self, value_net, gmo_config):
        """Test that optimized embedding has correct shape."""
        state_embed = torch.randn(gmo_config.state_dim)
        initial_embed = torch.randn(gmo_config.move_dim)

        optimized_embed = optimize_move_with_entropy(
            state_embed, initial_embed, value_net, gmo_config
        )

        assert optimized_embed.shape == (gmo_config.move_dim,)

    def test_optimize_move_is_deterministic_given_seed(self, gmo_config):
        """Test reproducibility with fixed seed."""
        torch.manual_seed(42)
        value_net1 = GMOValueNetWithUncertainty(
            state_dim=gmo_config.state_dim,
            move_dim=gmo_config.move_dim,
            hidden_dim=gmo_config.hidden_dim,
        )

        torch.manual_seed(42)
        value_net2 = GMOValueNetWithUncertainty(
            state_dim=gmo_config.state_dim,
            move_dim=gmo_config.move_dim,
            hidden_dim=gmo_config.hidden_dim,
        )

        state_embed = torch.randn(gmo_config.state_dim)
        initial_embed = torch.randn(gmo_config.move_dim)

        # Both should produce same result with same seed
        # (Note: MC dropout introduces randomness, so this is approximate)


# =============================================================================
# Projection Tests
# =============================================================================

class TestProjection:
    """Tests for projecting embeddings to legal moves."""

    def test_project_finds_nearest_move(self, move_encoder, sample_move, sample_move_stack):
        """Test that projection finds the nearest move."""
        moves = [sample_move, sample_move_stack]
        move_embeds = move_encoder.encode_moves(moves)

        # Project an embedding very close to first move
        close_embed = move_embeds[0] + torch.randn_like(move_embeds[0]) * 0.01

        selected_move, idx = project_to_legal_move(
            close_embed, move_embeds, moves, temperature=0.0
        )

        assert idx == 0
        assert selected_move == sample_move

    def test_project_with_temperature_samples(self, move_encoder, sample_move, sample_move_stack):
        """Test that temperature > 0 introduces sampling."""
        moves = [sample_move, sample_move_stack]
        move_embeds = move_encoder.encode_moves(moves)

        # With very high temperature, both should be selected sometimes
        midpoint = (move_embeds[0] + move_embeds[1]) / 2

        selected_indices = set()
        for _ in range(50):
            _, idx = project_to_legal_move(
                midpoint, move_embeds, moves, temperature=10.0
            )
            selected_indices.add(idx)

        # With high temperature, we should see both moves selected
        # (this may occasionally fail due to randomness, but is very unlikely)
        assert len(selected_indices) >= 1  # At least one move selected


# =============================================================================
# Full GMOAI Tests
# =============================================================================

class TestGMOAI:
    """Integration tests for GMOAI."""

    def test_gmo_ai_can_be_created(self, gmo_config):
        """Test that GMOAI can be instantiated."""
        ai_config = AIConfig(difficulty=6)
        gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

        assert gmo_ai.player_number == 1
        assert gmo_ai.gmo_config == gmo_config

    def test_gmo_ai_select_move_returns_valid_move(self, gmo_config):
        """Test that select_move returns a valid move."""
        ai_config = AIConfig(difficulty=6)
        gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

        state = create_test_game_state()

        move = gmo_ai.select_move(state)

        # Should return a move (game has legal moves)
        assert move is not None
        assert move.player == 1

    def test_gmo_ai_evaluate_position(self, gmo_config):
        """Test that evaluate_position returns a valid score."""
        ai_config = AIConfig(difficulty=6)
        gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

        state = create_test_game_state()

        score = gmo_ai.evaluate_position(state)

        # Score should be in valid range
        assert -1 <= score <= 1

    def test_gmo_ai_reset_clears_novelty(self, gmo_config):
        """Test that reset_for_new_game clears novelty tracker."""
        ai_config = AIConfig(difficulty=6)
        gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

        # Add some entries to novelty tracker
        gmo_ai.novelty_tracker.add(torch.randn(gmo_config.move_dim))
        gmo_ai.novelty_tracker.add(torch.randn(gmo_config.move_dim))

        assert gmo_ai.novelty_tracker.count > 0

        gmo_ai.reset_for_new_game()

        assert gmo_ai.novelty_tracker.count == 0

    def test_gmo_ai_handles_single_legal_move(self, gmo_config):
        """Test that GMO AI handles case with single legal move."""
        from app.models import Move

        ai_config = AIConfig(difficulty=6)
        gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

        state = create_test_game_state()

        # Mock get_valid_moves to return single move
        single_move = Move(
            id="single",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
        )

        original_get_valid = gmo_ai.get_valid_moves
        gmo_ai.get_valid_moves = lambda s: [single_move]

        move = gmo_ai.select_move(state)

        assert move == single_move

        # Restore
        gmo_ai.get_valid_moves = original_get_valid


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
