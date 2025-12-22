"""Graph Neural Network encoding for board states.

This module provides utilities for converting RingRift game states to graph
representations suitable for Graph Neural Networks (GNNs).

Extracted from app/ai/archive/cage_network.py (December 2025) to enable
GNN-based position evaluation approaches.

Key features:
- Node features: stack height, cap height, control, markers, collapsed spaces
- Edge connectivity: 4-adjacency for square boards, 6-adjacency for hex
- Extensible for custom graph architectures

Usage:
    from app.ai.neural_net.graph_encoding import board_to_graph, board_to_graph_hex

    # Square board
    node_features, edge_index, edge_attr = board_to_graph(game_state, player_number)

    # Hex board
    node_features, edge_index, edge_attr = board_to_graph_hex(game_state, player_number, radius=12)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from app.models import GameState


# Node feature indices for documentation
NODE_FEATURE_IDX = {
    "has_stack": 0,
    "stack_height_norm": 1,
    "cap_height_norm": 2,
    "controlled_by_player": 3,
    "controlled_by_opponent": 4,
    "has_marker": 5,
    "marker_is_player": 6,
    "marker_is_opponent": 7,
    "collapsed_space": 8,
    "collapsed_by_player": 9,
    "collapsed_by_opponent": 10,
    # Reserved 11-31 for future features
}

# Edge attribute indices
EDGE_ATTR_IDX = {
    "connected": 0,
    "direction_left": 1,
    "direction_right": 2,
    "direction_up": 3,
    "direction_down": 4,
    # Hex directions (5-10)
    "direction_hex_0": 5,
    "direction_hex_1": 6,
    "direction_hex_2": 7,
    "direction_hex_3": 8,
    "direction_hex_4": 9,
    "direction_hex_5": 10,
}


def board_to_graph(
    game_state: GameState,
    player_number: int,
    board_size: int = 8,
    node_feature_dim: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert square board state to graph representation.

    Creates a graph where each cell is a node and edges connect adjacent cells.
    Node features encode the game state from the perspective of `player_number`.

    Args:
        game_state: The current game state.
        player_number: The perspective player (1-indexed).
        board_size: Size of the square board (8 for square8, 19 for square19).
        node_feature_dim: Dimension of node feature vectors.

    Returns:
        Tuple of:
        - node_features: (num_nodes, node_feature_dim) tensor
        - edge_index: (2, num_edges) connectivity tensor
        - edge_attr: (num_edges, 8) edge attribute tensor
    """
    num_nodes = board_size * board_size
    node_features = torch.zeros(num_nodes, node_feature_dim)

    board = game_state.board

    def set_feature(x: int, y: int, idx: int, value: float = 1.0) -> None:
        """Set a feature for the node at position (x, y)."""
        if 0 <= x < board_size and 0 <= y < board_size:
            node_idx = y * board_size + x
            node_features[node_idx, idx] = value

    # Encode stacks
    for key, stack in board.stacks.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue

        height = float(getattr(stack, "stack_height", len(stack.rings or [])))
        cap = float(getattr(stack, "cap_height", 0))
        controller = int(getattr(stack, "controlling_player", 0))

        set_feature(x, y, NODE_FEATURE_IDX["has_stack"], 1.0)
        set_feature(x, y, NODE_FEATURE_IDX["stack_height_norm"], height / 5.0)
        set_feature(x, y, NODE_FEATURE_IDX["cap_height_norm"], cap / 5.0)

        if controller == player_number:
            set_feature(x, y, NODE_FEATURE_IDX["controlled_by_player"], 1.0)
        elif controller != 0:
            set_feature(x, y, NODE_FEATURE_IDX["controlled_by_opponent"], 1.0)

    # Encode markers
    for key, marker in board.markers.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue

        set_feature(x, y, NODE_FEATURE_IDX["has_marker"], 1.0)
        if marker.player == player_number:
            set_feature(x, y, NODE_FEATURE_IDX["marker_is_player"], 1.0)
        else:
            set_feature(x, y, NODE_FEATURE_IDX["marker_is_opponent"], 1.0)

    # Encode collapsed spaces
    for key, owner in board.collapsed_spaces.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue

        set_feature(x, y, NODE_FEATURE_IDX["collapsed_space"], 1.0)
        if owner == player_number:
            set_feature(x, y, NODE_FEATURE_IDX["collapsed_by_player"], 1.0)
        elif owner != 0:
            set_feature(x, y, NODE_FEATURE_IDX["collapsed_by_opponent"], 1.0)

    # Build edge index: 4-connectivity for square grids
    edges = []
    edge_attrs = []
    dir_map = {
        (-1, 0): EDGE_ATTR_IDX["direction_left"],
        (1, 0): EDGE_ATTR_IDX["direction_right"],
        (0, -1): EDGE_ATTR_IDX["direction_up"],
        (0, 1): EDGE_ATTR_IDX["direction_down"],
    }

    for y in range(board_size):
        for x in range(board_size):
            node_idx = y * board_size + x
            for (dx, dy), dir_idx in dir_map.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    neighbor_idx = ny * board_size + nx
                    edges.append([node_idx, neighbor_idx])
                    edge_attr = torch.zeros(12)  # Extended for hex directions
                    edge_attr[EDGE_ATTR_IDX["connected"]] = 1.0
                    edge_attr[dir_idx] = 1.0
                    edge_attrs.append(edge_attr)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs) if edge_attrs else torch.zeros(0, 12)

    return node_features, edge_index, edge_attr


def board_to_graph_hex(
    game_state: GameState,
    player_number: int,
    radius: int = 12,
    node_feature_dim: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert hexagonal board state to graph representation.

    Uses cube coordinates (x, y, z) where x + y + z = 0.
    Edges connect 6-adjacent hex cells.

    Args:
        game_state: The current game state.
        player_number: The perspective player (1-indexed).
        radius: Hex board radius (4 for hex8, 12 for hexagonal).
        node_feature_dim: Dimension of node feature vectors.

    Returns:
        Tuple of:
        - node_features: (num_nodes, node_feature_dim) tensor
        - edge_index: (2, num_edges) connectivity tensor
        - edge_attr: (num_edges, 12) edge attribute tensor
    """
    # Generate all valid hex positions using cube coordinates
    positions = []
    pos_to_idx = {}
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            z = -x - y
            if abs(z) <= radius:
                pos_to_idx[(x, y, z)] = len(positions)
                positions.append((x, y, z))

    num_nodes = len(positions)
    node_features = torch.zeros(num_nodes, node_feature_dim)

    board = game_state.board

    def parse_hex_key(key: str) -> tuple[int, int, int] | None:
        """Parse hex position key to cube coordinates."""
        try:
            parts = key.split(",")
            if len(parts) == 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                # Axial to cube conversion
                q, r = int(parts[0]), int(parts[1])
                return q, r, -q - r
        except (ValueError, IndexError):
            pass
        return None

    def set_feature(pos: tuple[int, int, int], idx: int, value: float = 1.0) -> None:
        """Set a feature for the node at hex position."""
        if pos in pos_to_idx:
            node_idx = pos_to_idx[pos]
            node_features[node_idx, idx] = value

    # Encode stacks
    for key, stack in board.stacks.items():
        pos = parse_hex_key(key)
        if pos is None:
            continue

        height = float(getattr(stack, "stack_height", len(stack.rings or [])))
        cap = float(getattr(stack, "cap_height", 0))
        controller = int(getattr(stack, "controlling_player", 0))

        set_feature(pos, NODE_FEATURE_IDX["has_stack"], 1.0)
        set_feature(pos, NODE_FEATURE_IDX["stack_height_norm"], height / 5.0)
        set_feature(pos, NODE_FEATURE_IDX["cap_height_norm"], cap / 5.0)

        if controller == player_number:
            set_feature(pos, NODE_FEATURE_IDX["controlled_by_player"], 1.0)
        elif controller != 0:
            set_feature(pos, NODE_FEATURE_IDX["controlled_by_opponent"], 1.0)

    # Encode markers
    for key, marker in board.markers.items():
        pos = parse_hex_key(key)
        if pos is None:
            continue

        set_feature(pos, NODE_FEATURE_IDX["has_marker"], 1.0)
        if marker.player == player_number:
            set_feature(pos, NODE_FEATURE_IDX["marker_is_player"], 1.0)
        else:
            set_feature(pos, NODE_FEATURE_IDX["marker_is_opponent"], 1.0)

    # Encode collapsed spaces
    for key, owner in board.collapsed_spaces.items():
        pos = parse_hex_key(key)
        if pos is None:
            continue

        set_feature(pos, NODE_FEATURE_IDX["collapsed_space"], 1.0)
        if owner == player_number:
            set_feature(pos, NODE_FEATURE_IDX["collapsed_by_player"], 1.0)
        elif owner != 0:
            set_feature(pos, NODE_FEATURE_IDX["collapsed_by_opponent"], 1.0)

    # Build edge index: 6-connectivity for hex grids
    # Cube coordinate directions for hex adjacency
    hex_directions = [
        (1, -1, 0),   # direction 0
        (1, 0, -1),   # direction 1
        (0, 1, -1),   # direction 2
        (-1, 1, 0),   # direction 3
        (-1, 0, 1),   # direction 4
        (0, -1, 1),   # direction 5
    ]

    edges = []
    edge_attrs = []

    for (x, y, z), node_idx in pos_to_idx.items():
        for dir_i, (dx, dy, dz) in enumerate(hex_directions):
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in pos_to_idx:
                neighbor_idx = pos_to_idx[neighbor]
                edges.append([node_idx, neighbor_idx])
                edge_attr = torch.zeros(12)
                edge_attr[EDGE_ATTR_IDX["connected"]] = 1.0
                edge_attr[EDGE_ATTR_IDX["direction_hex_0"] + dir_i] = 1.0
                edge_attrs.append(edge_attr)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs) if edge_attrs else torch.zeros(0, 12)

    return node_features, edge_index, edge_attr


__all__ = [
    "NODE_FEATURE_IDX",
    "EDGE_ATTR_IDX",
    "board_to_graph",
    "board_to_graph_hex",
]
