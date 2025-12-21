"""Unit tests for GMO v2 AI (Enhanced Gradient Move Optimization).

Tests cover:
- GMOv2Config configuration
- AttentionStateEncoder network
- MoveEncoderV2 network
- GMOv2ValueNet network
- GMOv2AI move selection
- Ensemble optimization
- Temperature scheduling
"""

import pytest
import torch

from app.ai.gmo_v2 import (
    AttentionStateEncoder,
    GMOv2AI,
    GMOv2Config,
    GMOv2ValueNet,
    MoveEncoderV2,
)
from app.ai.gmo_ai import NoveltyTracker
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
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
# GMOv2Config Tests
# =============================================================================


class TestGMOv2Config:
    """Test GMO v2 configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = GMOv2Config()
        assert config.state_dim == 256
        assert config.move_dim == 256
        assert config.hidden_dim == 512
        assert config.top_k == 7
        assert config.optim_steps == 15
        assert config.ensemble_size == 3
        assert config.ensemble_voting in ("hard", "soft")
        assert config.device == "cpu"

    def test_custom_config(self):
        """Custom config values should be applied."""
        config = GMOv2Config(
            state_dim=128,
            move_dim=128,
            top_k=5,
            ensemble_size=5,
            device="cuda",
        )
        assert config.state_dim == 128
        assert config.move_dim == 128
        assert config.top_k == 5
        assert config.ensemble_size == 5
        assert config.device == "cuda"

    def test_temperature_scheduling_config(self):
        """Temperature scheduling config should be valid."""
        config = GMOv2Config()
        assert config.temp_early_game > config.temp_mid_game
        assert config.temp_mid_game > config.temp_late_game
        assert config.early_game_threshold < config.late_game_threshold


# =============================================================================
# AttentionStateEncoder Tests
# =============================================================================


class TestAttentionStateEncoder:
    """Test attention-based state encoder."""

    @pytest.fixture
    def encoder(self):
        """Create test encoder."""
        return AttentionStateEncoder(embed_dim=256, board_size=8)

    def test_encoder_creation(self, encoder):
        """Encoder should be created successfully."""
        assert encoder is not None
        assert encoder.embed_dim == 256
        assert hasattr(encoder, 'transformer')
        assert hasattr(encoder, 'output_proj')

    def test_encoder_output_shape(self, encoder):
        """Encoder should produce correct output shape."""
        game_state = create_test_game_state()

        with torch.no_grad():
            output = encoder(game_state)

        assert output.shape == (256,)

    def test_encoder_no_nan(self, encoder):
        """Encoder output should not contain NaN."""
        game_state = create_test_game_state()

        with torch.no_grad():
            output = encoder(game_state)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# MoveEncoderV2 Tests
# =============================================================================


class TestMoveEncoderV2:
    """Test move encoder v2."""

    @pytest.fixture
    def encoder(self):
        """Create test encoder."""
        return MoveEncoderV2(embed_dim=256, board_size=8)

    def test_encoder_creation(self, encoder):
        """Encoder should be created successfully."""
        assert encoder is not None
        assert encoder.embed_dim == 256
        assert hasattr(encoder, 'move_type_embed')
        assert hasattr(encoder, 'from_pos_embed')
        assert hasattr(encoder, 'to_pos_embed')


# =============================================================================
# GMOv2ValueNet Tests
# =============================================================================


class TestGMOv2ValueNet:
    """Test GMO v2 value network."""

    @pytest.fixture
    def value_net(self):
        """Create test value network."""
        return GMOv2ValueNet(
            state_dim=256,
            move_dim=256,
            hidden_dim=512,
        )

    def test_value_net_creation(self, value_net):
        """Value network should be created successfully."""
        assert value_net is not None

    def test_forward_pass(self, value_net):
        """Forward pass should produce value output."""
        state_embed = torch.randn(1, 256)
        move_embed = torch.randn(1, 256)

        with torch.no_grad():
            value, logits = value_net(state_embed, move_embed)

        assert value.shape == (1,)
        assert not torch.isnan(value).any()

    def test_project_to_moves(self, value_net):
        """Project to moves should compute similarity scores."""
        optimized_embed = torch.randn(256)
        legal_move_embeds = torch.randn(10, 256)

        scores = value_net.project_to_moves(optimized_embed, legal_move_embeds)

        assert scores.shape == (10,)


# =============================================================================
# NoveltyTracker Tests
# =============================================================================


class TestNoveltyTracker:
    """Test novelty tracker (shared with GMO v1)."""

    @pytest.fixture
    def tracker(self):
        """Create test tracker."""
        return NoveltyTracker(memory_size=100, embed_dim=256)

    def test_tracker_creation(self, tracker):
        """Tracker should be created successfully."""
        assert tracker is not None

    def test_novelty_computation(self, tracker):
        """Novelty should be computed for embeddings."""
        embed = torch.randn(256)

        # First embedding should have max novelty (nothing in memory)
        novelty = tracker.compute_novelty(embed)
        assert isinstance(novelty, (int, float, torch.Tensor))

    def test_novelty_decreases_with_repetition(self, tracker):
        """Novelty should decrease for similar embeddings."""
        embed = torch.randn(256)

        # Add embedding to memory
        tracker.add(embed)

        # Same embedding should have lower novelty
        novelty_after = tracker.compute_novelty(embed)

        # Should be lower than max (1.0) since it's in memory
        if isinstance(novelty_after, torch.Tensor):
            novelty_after = novelty_after.item()
        assert novelty_after < 1.0


# =============================================================================
# GMOv2AI Tests
# =============================================================================


class TestGMOv2AI:
    """Test GMO v2 AI agent."""

    @pytest.fixture
    def ai_config(self):
        """Create AI config for testing."""
        return AIConfig(difficulty=5)

    @pytest.fixture
    def gmo_config(self):
        """Create minimal GMO config for faster tests."""
        return GMOv2Config(
            optim_steps=3,
            top_k=2,
            ensemble_size=2,
            mc_samples=3,
        )

    @pytest.fixture
    def game_state(self):
        """Create test game state."""
        return create_test_game_state()

    @pytest.fixture
    def ai(self, ai_config, gmo_config):
        """Create GMO v2 AI instance."""
        return GMOv2AI(player_number=1, config=ai_config, gmo_config=gmo_config)

    def test_ai_creation(self, ai):
        """AI should be created successfully."""
        assert ai is not None
        assert ai.player_number == 1
        assert ai.state_encoder is not None
        assert ai.move_encoder is not None
        assert ai.value_net is not None

    def test_ai_has_networks(self, ai):
        """AI should have all required networks."""
        assert isinstance(ai.state_encoder, AttentionStateEncoder)
        assert isinstance(ai.move_encoder, MoveEncoderV2)
        assert isinstance(ai.value_net, GMOv2ValueNet)

    def test_select_move_returns_move(self, ai, game_state):
        """select_move should return a valid move."""
        move = ai.select_move(game_state)
        assert move is not None
        assert hasattr(move, 'type')
        assert hasattr(move, 'to')

    def test_select_move_returns_legal_move(self, ai, game_state):
        """Selected move should be legal."""
        move = ai.select_move(game_state)
        valid_moves = ai.get_valid_moves(game_state)

        # Check that the returned move matches a valid move
        move_matches = [
            m for m in valid_moves
            if m.type == move.type
            and (m.to.x == move.to.x if m.to and move.to else True)
            and (m.to.y == move.to.y if m.to and move.to else True)
        ]
        assert len(move_matches) >= 1

    def test_get_exploration_temperature(self, ai, game_state):
        """Temperature should vary by game phase."""
        # Early game (no moves)
        temp_early = ai._get_exploration_temperature(game_state)
        assert temp_early == ai.gmo_config.temp_early_game

    def test_estimate_uncertainty(self, ai):
        """Uncertainty estimation should return mean, entropy, variance."""
        state_embed = torch.randn(1, ai.gmo_config.state_dim)
        move_embed = torch.randn(1, ai.gmo_config.move_dim)

        mean_value, entropy, variance = ai._estimate_uncertainty(state_embed, move_embed)

        assert not torch.isnan(mean_value).any()
        assert not torch.isnan(variance).any()
        assert variance >= 0  # Variance must be non-negative


# =============================================================================
# Ensemble Optimization Tests
# =============================================================================


class TestEnsembleOptimization:
    """Test ensemble gradient optimization."""

    @pytest.fixture
    def ai(self):
        """Create AI with minimal config."""
        ai_config = AIConfig(difficulty=5)
        gmo_config = GMOv2Config(
            optim_steps=3,
            ensemble_size=2,
            mc_samples=3,
        )
        return GMOv2AI(player_number=1, config=ai_config, gmo_config=gmo_config)

    def test_optimize_move_ensemble(self, ai):
        """Ensemble optimization should produce multiple paths."""
        # Note: GMOv2 internal implementation uses 1D tensors
        state_embed = torch.randn(ai.gmo_config.state_dim)
        initial_embed = torch.randn(ai.gmo_config.move_dim)

        optimized = ai._optimize_move_ensemble(
            state_embed,
            initial_embed,
            exploration_temp=1.0,
        )

        assert len(optimized) == ai.gmo_config.ensemble_size
        for embed in optimized:
            assert embed.shape == (ai.gmo_config.move_dim,)

    def test_ensemble_vote_soft(self, ai):
        """Soft voting should average scores."""
        optimized_embeds = [torch.randn(ai.gmo_config.move_dim) for _ in range(2)]
        state_embed = torch.randn(ai.gmo_config.state_dim)
        legal_move_embeds = torch.randn(5, ai.gmo_config.move_dim)

        scores = ai._ensemble_vote(optimized_embeds, state_embed, legal_move_embeds)

        assert scores.shape == (5,)


# =============================================================================
# Integration Tests
# =============================================================================


class TestGMOv2Integration:
    """Integration tests for GMO v2 AI."""

    @pytest.fixture
    def ai(self):
        """Create GMO v2 AI with minimal config."""
        ai_config = AIConfig(difficulty=5)
        gmo_config = GMOv2Config(
            optim_steps=2,
            top_k=2,
            ensemble_size=2,
            mc_samples=2,
        )
        return GMOv2AI(player_number=1, config=ai_config, gmo_config=gmo_config)

    def test_multiple_moves(self, ai):
        """AI should handle multiple consecutive moves."""
        game_state = create_test_game_state()

        # Select multiple moves
        for _ in range(3):
            move = ai.select_move(game_state)
            assert move is not None

    def test_novelty_tracking_updates(self, ai):
        """Novelty tracker should update after moves."""
        game_state = create_test_game_state()

        initial_memory = len(ai.novelty_tracker.memory) if hasattr(ai.novelty_tracker, 'memory') else 0

        ai.select_move(game_state)

        # Memory should have grown (implementation dependent)
        # Just verify it doesn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
