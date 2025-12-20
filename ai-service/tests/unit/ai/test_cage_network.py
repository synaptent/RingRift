"""Tests for CAGE (Constraint-Aware Graph Energy-Based) AI network."""

import pytest
import torch

from app.ai.cage_network import (
    CAGEConfig,
    CAGEEnergyHead,
    CAGENetwork,
    ConstraintNetwork,
    GraphAttentionLayer,
    GraphEncoder,
    board_to_graph,
)
from app.models import BoardType, GameState
from app.training.initial_state import create_initial_state


@pytest.fixture
def config():
    """Create a test configuration."""
    return CAGEConfig(
        board_size=8,
        board_type=BoardType.SQUARE8,
        gnn_num_layers=2,  # Smaller for faster tests
        num_energy_layers=2,
        optim_steps=10,  # Fewer steps for faster tests
    )


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("cpu")  # Use CPU for MPS compatibility
    return torch.device("cpu")


@pytest.fixture
def network(config, device):
    """Create a test network."""
    net = CAGENetwork(config)
    net.to(device)
    net.eval()
    return net


@pytest.fixture
def game_state():
    """Create a test game state."""
    return create_initial_state(board_type=BoardType.SQUARE8, num_players=2)


class TestCAGEConfig:
    """Tests for CAGEConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = CAGEConfig()

        assert config.board_size == 8
        assert config.gnn_hidden_dim == 128
        assert config.gnn_num_layers == 4
        assert config.optim_steps == 50
        assert config.constraint_penalty == 10.0

    def test_custom_config(self):
        """Should support custom configuration."""
        config = CAGEConfig(
            board_size=13,
            gnn_hidden_dim=256,
            optim_steps=100,
        )

        assert config.board_size == 13
        assert config.gnn_hidden_dim == 256
        assert config.optim_steps == 100


class TestGraphAttentionLayer:
    """Tests for GraphAttentionLayer."""

    def test_forward_shape(self, config, device):
        """Output should have correct shape."""
        layer = GraphAttentionLayer(
            in_features=config.node_feature_dim,
            out_features=config.gnn_hidden_dim,
            edge_features=config.edge_feature_dim,
            num_heads=config.gnn_num_heads,
        ).to(device)

        num_nodes = 64  # 8x8 board
        num_edges = 200  # Approximate edge count

        x = torch.randn(num_nodes, config.node_feature_dim, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        edge_attr = torch.randn(num_edges, config.edge_feature_dim, device=device)

        out = layer(x, edge_index, edge_attr)

        assert out.shape == (num_nodes, config.gnn_hidden_dim)

    def test_handles_empty_edges(self, config, device):
        """Should handle graph with no edges."""
        layer = GraphAttentionLayer(
            in_features=config.node_feature_dim,
            out_features=config.gnn_hidden_dim,
            edge_features=config.edge_feature_dim,
            num_heads=config.gnn_num_heads,
        ).to(device)

        num_nodes = 64
        x = torch.randn(num_nodes, config.node_feature_dim, device=device)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.zeros(0, config.edge_feature_dim, device=device)

        out = layer(x, edge_index, edge_attr)

        assert out.shape == (num_nodes, config.gnn_hidden_dim)


class TestGraphEncoder:
    """Tests for GraphEncoder."""

    def test_forward_shape(self, config, device):
        """Encoder should produce correct output shapes."""
        encoder = GraphEncoder(config).to(device)

        num_nodes = 64
        num_edges = 200

        x = torch.randn(num_nodes, config.node_feature_dim, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        edge_attr = torch.randn(num_edges, config.edge_feature_dim, device=device)

        node_embed, graph_embed = encoder(x, edge_index, edge_attr)

        assert node_embed.shape == (num_nodes, config.gnn_hidden_dim)
        assert graph_embed.shape == (1, config.gnn_hidden_dim)


class TestConstraintNetwork:
    """Tests for ConstraintNetwork."""

    def test_forward_shape(self, config, device):
        """Should produce constraint violation scores."""
        net = ConstraintNetwork(config).to(device)

        batch_size = 4
        state = torch.randn(batch_size, config.gnn_hidden_dim, device=device)
        action = torch.randn(batch_size, config.action_embed_dim, device=device)

        violations = net(state, action)

        assert violations.shape == (batch_size, config.num_constraint_types)

    def test_violations_positive(self, config, device):
        """Violations should be non-negative (ReLU activated)."""
        net = ConstraintNetwork(config).to(device)

        batch_size = 8
        state = torch.randn(batch_size, config.gnn_hidden_dim, device=device)
        action = torch.randn(batch_size, config.action_embed_dim, device=device)

        violations = net(state, action)

        assert (violations >= 0).all()


class TestCAGEEnergyHead:
    """Tests for CAGEEnergyHead."""

    def test_forward_shape(self, config, device):
        """Should produce scalar energy values and violations."""
        head = CAGEEnergyHead(config).to(device)

        batch_size = 4
        state = torch.randn(batch_size, config.gnn_hidden_dim, device=device)
        action = torch.randn(batch_size, config.action_embed_dim, device=device)

        energy, violations = head(state, action)

        assert energy.shape == (batch_size,)
        assert violations.shape == (batch_size, config.num_constraint_types)


class TestCAGENetwork:
    """Tests for the main CAGENetwork."""

    def test_initialization(self, config, device):
        """Network should initialize correctly."""
        net = CAGENetwork(config).to(device)

        assert net.config == config
        assert isinstance(net.graph_encoder, GraphEncoder)
        assert isinstance(net.energy_head, CAGEEnergyHead)
        # Constraint net is inside energy head
        assert isinstance(net.energy_head.constraint_net, ConstraintNetwork)

    def test_encode_graph(self, network, config, device):
        """encode_graph should produce embeddings."""
        num_nodes = 64
        num_edges = 200

        x = torch.randn(num_nodes, config.node_feature_dim, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        edge_attr = torch.randn(num_edges, config.edge_feature_dim, device=device)

        node_embed, graph_embed = network.encode_graph(x, edge_index, edge_attr)

        assert node_embed.shape == (num_nodes, config.gnn_hidden_dim)
        assert graph_embed.shape == (1, config.gnn_hidden_dim)

    def test_encode_action(self, network, config, device):
        """encode_action should produce action embeddings."""
        batch_size = 4
        # Action features: 14-dimensional (from_y, from_x, to_y, to_x, move_type, + extras)
        action_features = torch.randn(batch_size, 14, device=device)

        action_embed = network.encode_action(action_features)

        assert action_embed.shape == (batch_size, config.action_embed_dim)

    def test_compute_energy(self, network, config, device):
        """compute_energy should produce scalar energies."""
        batch_size = 4
        state = torch.randn(batch_size, config.gnn_hidden_dim, device=device)
        action = torch.randn(batch_size, config.action_embed_dim, device=device)

        energy = network.compute_energy(state, action)

        assert energy.shape == (batch_size,)

    def test_compute_energy_with_violations(self, network, config, device):
        """Should return both energy and violations."""
        batch_size = 4
        state = torch.randn(batch_size, config.gnn_hidden_dim, device=device)
        action = torch.randn(batch_size, config.action_embed_dim, device=device)

        energy, violations = network.compute_energy_with_violations(state, action)

        assert energy.shape == (batch_size,)
        assert violations.shape == (batch_size, config.num_constraint_types)

    def test_primal_dual_optimize(self, network, config, device):
        """Optimization should return best move index and energy."""
        num_moves = 10
        graph_embed = torch.randn(config.gnn_hidden_dim, device=device)
        action_embeds = torch.randn(num_moves, config.action_embed_dim, device=device)

        best_idx, best_energy = network.primal_dual_optimize(
            graph_embed, action_embeds, num_steps=5
        )

        assert 0 <= best_idx < num_moves
        assert isinstance(best_energy, float)


class TestBoardToGraph:
    """Tests for board_to_graph function."""

    def test_output_types(self, game_state, config):
        """Should return node features, edge index, and edge attributes."""
        node_feat, edge_index, edge_attr = board_to_graph(
            game_state, player_number=1, board_size=config.board_size
        )

        assert isinstance(node_feat, torch.Tensor)
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_attr, torch.Tensor)

    def test_node_count(self, game_state, config):
        """Should have correct number of nodes for board size."""
        node_feat, _edge_index, _edge_attr = board_to_graph(
            game_state, player_number=1, board_size=config.board_size
        )

        expected_nodes = config.board_size * config.board_size
        assert node_feat.shape[0] == expected_nodes

    def test_edge_index_shape(self, game_state, config):
        """Edge index should have 2 rows (source, target)."""
        _node_feat, edge_index, _edge_attr = board_to_graph(
            game_state, player_number=1, board_size=config.board_size
        )

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0  # Should have some edges

    def test_edge_attr_matches_edges(self, game_state, config):
        """Edge attributes should match edge count."""
        _node_feat, edge_index, edge_attr = board_to_graph(
            game_state, player_number=1, board_size=config.board_size
        )

        num_edges = edge_index.shape[1]
        assert edge_attr.shape[0] == num_edges


class TestIntegration:
    """Integration tests for CAGE network."""

    def test_full_forward_pass(self, game_state, config, device):
        """Full forward pass from game state to move selection."""
        # Create a fresh network for this test (primal_dual_optimize does internal backprop)
        network = CAGENetwork(config).to(device)
        network.eval()

        # Convert board to graph
        node_feat, edge_index, edge_attr = board_to_graph(
            game_state, player_number=1, board_size=config.board_size
        )
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        # Encode graph (with no_grad since we don't need gradients here)
        with torch.no_grad():
            _, graph_embed = network.encode_graph(node_feat, edge_index, edge_attr)
            graph_embed = graph_embed.squeeze(0)

            # Create dummy action features (5 moves, 14 features each)
            num_moves = 5
            action_features = torch.randn(num_moves, 14, device=device)
            action_embeds = network.encode_action(action_features)

        # Run optimization (this creates its own computation graph internally)
        best_idx, best_energy = network.primal_dual_optimize(
            graph_embed.detach(), action_embeds.detach(), num_steps=config.optim_steps
        )

        assert 0 <= best_idx < num_moves
        assert isinstance(best_energy, float)

    def test_batch_processing(self, network, config, device):
        """Should handle batched inputs efficiently."""
        batch_size = 4
        num_moves_per_game = 10

        # Batched state embeddings
        states = torch.randn(batch_size * num_moves_per_game, config.gnn_hidden_dim, device=device)
        actions = torch.randn(batch_size * num_moves_per_game, config.action_embed_dim, device=device)

        # Batch energy computation
        energies = network.compute_energy(states, actions)

        assert energies.shape == (batch_size * num_moves_per_game,)

    def test_gradient_flow(self, config, device):
        """Gradients should flow through the network."""
        net = CAGENetwork(config).to(device)
        net.train()

        # Forward pass with proper graph encoding
        num_nodes = 64
        num_edges = 200

        node_feat = torch.randn(num_nodes, config.node_feature_dim, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        edge_attr = torch.randn(num_edges, config.edge_feature_dim, device=device)

        _, graph_embed = net.encode_graph(node_feat, edge_index, edge_attr)

        # Action encoding
        action_feat = torch.randn(4, 14, device=device)
        action_embed = net.encode_action(action_feat)

        # Expand graph_embed to match batch size
        graph_embed_batch = graph_embed.expand(4, -1)

        energy = net.compute_energy(graph_embed_batch, action_embed)
        loss = energy.mean()
        loss.backward()

        # Check some model parameters have gradients
        has_grad = any(p.grad is not None for p in net.parameters() if p.requires_grad)
        assert has_grad, "No gradients found in model parameters"
