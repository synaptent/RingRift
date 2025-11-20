"""
Neural Network AI implementation for RingRift
Uses a simple feedforward neural network for move evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional
import random
from datetime import datetime

from .base import BaseAI
from ..models import GameState, Move


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class RingRiftCNN(nn.Module):
    def __init__(
        self, board_size=8, in_channels=10, global_features=10,
        num_res_blocks=10, num_filters=128, history_length=3
    ):
        super(RingRiftCNN, self).__init__()
        self.board_size = board_size
        
        # Input channels = base_channels * (history_length + 1)
        # Base channels = 10
        # History length = 3 (current + 3 previous states)
        # Total channels = 10 * 4 = 40
        self.total_in_channels = in_channels * (history_length + 1)

        # Initial convolution
        self.conv1 = nn.Conv2d(
            self.total_in_channels, num_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Adaptive Pooling to handle variable board sizes (e.g. 8x8, 19x19)
        # We pool to a fixed 4x4 grid before flattening, ensuring the FC layer input size is constant.
        # This allows the same model architecture to process different board sizes,
        # though retraining/finetuning is recommended for drastic size changes.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        # Input size is now num_filters * 4 * 4 (fixed)
        conv_out_size = num_filters * 4 * 4
        self.fc1 = nn.Linear(conv_out_size + global_features, 256)
        self.dropout = nn.Dropout(0.3)
        
        # Value head
        self.value_head = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        
        # Policy head
        # We use a large fixed size to accommodate up to 19x19 boards.
        # For smaller boards, we mask the invalid logits during inference/training.
        # Max size 19x19: ~55,000
        self.policy_size = 55000
        self.policy_head = nn.Linear(256, self.policy_size)

    def forward(self, x, globals):
        x = self.relu(self.bn1(self.conv1(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        value = self.tanh(self.value_head(x))  # Output between -1 and 1
        policy = self.policy_head(x) # Logits for CrossEntropyLoss
        
        return value, policy


class NeuralNetAI(BaseAI):
    """AI that uses a CNN to evaluate positions"""
    
    def __init__(self, player_number: int, config: Any):
        super().__init__(player_number, config)
        # Initialize model
        # Channels:
        # 0: My stacks (height normalized)
        # 1: Opponent stacks (height normalized)
        # 2: My markers
        # 3: Opponent markers
        # 4: My collapsed spaces
        # 5: Opponent collapsed spaces
        # 6: My liberties
        # 7: Opponent liberties
        # 8: My line potential
        # 9: Opponent line potential
        self.board_size = 8  # Default, will be updated based on game state
        self.history_length = 3
        self.state_history = [] # Stores last N feature planes
        
        # Initialize with default 8x8, but architecture is now adaptive
        self.model = RingRiftCNN(
            board_size=self.board_size, in_channels=10, global_features=10,
            num_res_blocks=10, num_filters=128, history_length=self.history_length
        )

        # Load weights if available
        import os
        # Use absolute path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models/ringrift_v1.pth")
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(
                    torch.load(model_path, weights_only=True)
                )
                self.model.eval()
            except RuntimeError as e:
                # Architecture mismatch
                print(f"Could not load model (architecture mismatch): {e}. Starting with fresh weights.")
                self.model.eval()
            except Exception as e:
                print(f"Could not load model (error): {e}. Starting with fresh weights.")
                self.model.eval()
        else:
            # No model found, start fresh (silent)
            self.model.eval()

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using neural network evaluation
        """
        self.simulate_thinking(min_ms=300, max_ms=1000)
        
        from ..game_engine import GameEngine
        valid_moves = GameEngine.get_valid_moves(
            game_state, self.player_number
        )

        if not valid_moves:
            return None
            
        if self.should_pick_random_move():
            selected = random.choice(valid_moves)
        else:
            best_move = None
            best_score = float('-inf')
            
            for move in valid_moves:
                # Simulate move to get next state
                next_state = GameEngine.apply_move(game_state, move)
                score = self.evaluate_position(next_state)
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            selected = best_move if best_move else random.choice(valid_moves)
            
        return selected

    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate position using neural network.

        This override matches BaseAI.evaluate_position by returning a single
        scalar score. Policy information remains available via
        evaluate_batch for training and analysis, but is not surfaced
        through this method.
        """
        values, _ = self.evaluate_batch([game_state])
        return values[0] if values else 0.0

    def evaluate_batch(self, game_states: list[GameState]) -> tuple[list[float], np.ndarray]:
        """
        Evaluate a batch of game states.

        Returns:
            A tuple of (values, policy_probs) where:
            - values is a list of scalar evaluations, one per state.
            - policy_probs is a NumPy array of shape (batch_size, policy_size).
        """
        if not game_states:
            empty_policy = np.zeros((0, self.model.policy_size), dtype=np.float32)
            return [], empty_policy

        batch_features = []
        batch_globals = []

        for state in game_states:
            features, globals = self._extract_features(state)
            # Stack features for history (simplified: repeat current state)
            stacked_features = np.concatenate([features] * (self.history_length + 1), axis=0)
            batch_features.append(stacked_features)
            batch_globals.append(globals)

        tensor_input = torch.FloatTensor(np.array(batch_features))
        globals_input = torch.FloatTensor(np.array(batch_globals))

        with torch.no_grad():
            values, policy_logits = self.model(tensor_input, globals_input)
            
            # Apply softmax to logits to get probabilities for MCTS
            policy_probs = torch.softmax(policy_logits, dim=1)

        return values.numpy().flatten().tolist(), policy_probs.numpy()

    def encode_move(self, move: Move, board_size: int) -> int:
        """
        Encode a move into a policy index.
        Supports variable board sizes up to 19x19.
        """
        MAX_N = 19  # Fixed max board size for encoding consistency
        
        if move.type == "place_ring":
            # Placement: 0 to 1082 (361 * 3)
            # Index = (y * MAX_N + x) * 3 + (count - 1)
            pos_idx = move.to.y * MAX_N + move.to.x
            count_idx = (move.placement_count or 1) - 1
            return pos_idx * 3 + count_idx
            
        elif move.type in ["move_stack", "move_ring", "overtaking_capture", "chain_capture"]:
            # Movement: 1083 to 53066
            # Base = 1083 (361 * 3)
            # Index = Base + (from_y * MAX_N + from_x) * (8 * (MAX_N-1)) + (dir_idx * (MAX_N-1)) + (dist - 1)
            if not move.from_pos:
                return 0
                
            from_idx = move.from_pos.y * MAX_N + move.from_pos.x
            
            dx = move.to.x - move.from_pos.x
            dy = move.to.y - move.from_pos.y
            
            # Handle Hexagonal Coordinates (Axial)
            # In axial, dist = (|dx| + |dy| + |dx+dy|) / 2
            # But here we use simple max(abs) for Chebyshev on square.
            # For hex, we need to map 6 directions.
            # Square has 8 directions.
            # We map hex directions to a subset of square directions for encoding simplicity.
            # Hex dirs: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
            
            dist = max(abs(dx), abs(dy))
            if dist == 0: return 0
            
            dir_x = dx // dist if dist > 0 else 0
            dir_y = dy // dist if dist > 0 else 0
            
            dirs = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1)
            ]
            try:
                dir_idx = dirs.index((dir_x, dir_y))
            except ValueError:
                # Fallback for hex directions that might not align perfectly if grid logic differs
                return 0
                
            base = 1083
            max_dist = MAX_N - 1
            return base + from_idx * (8 * max_dist) + dir_idx * max_dist + (dist - 1)
            
        elif move.type == "line_formation":
            # Line: 53067 to 54510
            # Base = 53067
            # Index = Base + (y * MAX_N + x) * 4 + dir_idx
            pos_idx = move.to.y * MAX_N + move.to.x
            return 53067 + pos_idx * 4
            
        elif move.type == "territory_claim":
            # Territory: 54511 to 54871
            # Base = 54511
            # Index = Base + (y * MAX_N + x)
            pos_idx = move.to.y * MAX_N + move.to.x
            return 54511 + pos_idx
            
        elif move.type == "skip_placement":
            return 54872
            
        return 0

    def decode_move(self, index: int, game_state: GameState) -> Optional[Move]:
        """
        Decode a policy index into a move.
        Supports variable board sizes up to 19x19.
        """
        MAX_N = 19
        
        if index < 1083:
            # Placement
            count_idx = index % 3
            pos_idx = index // 3
            y = pos_idx // MAX_N
            x = pos_idx % MAX_N
            
            if x >= self.board_size or y >= self.board_size:
                return None

            move_data = {
                "id": "decoded",
                "type": "place_ring",
                "player": game_state.current_player,
                "to": {"x": x, "y": y},
                "placementCount": count_idx + 1,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)
            
        elif index < 53067:
            # Movement
            base = 1083
            max_dist = MAX_N - 1
            offset = index - base
            
            dist_idx = offset % max_dist
            dist = dist_idx + 1
            offset //= max_dist
            
            dir_idx = offset % 8
            offset //= 8
            
            from_idx = offset
            from_y = from_idx // MAX_N
            from_x = from_idx % MAX_N
            
            if from_x >= self.board_size or from_y >= self.board_size:
                return None
            
            dirs = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1)
            ]
            dx, dy = dirs[dir_idx]
            
            to_x = from_x + dx * dist
            to_y = from_y + dy * dist
            
            if not (0 <= to_x < self.board_size and 0 <= to_y < self.board_size):
                return None

            move_data = {
                "id": "decoded",
                "type": "move_stack",
                "player": game_state.current_player,
                "from": {"x": from_x, "y": from_y},
                "to": {"x": to_x, "y": to_y},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)
            
        elif index == 54872:
            move_data = {
                "id": "decoded",
                "type": "skip_placement",
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)
            
        return None

    def _extract_features(
        self, game_state: GameState
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert game state to feature tensor for CNN and global features
        Returns: (board_features, global_features)
        """
        # Board features: 10 channels
        # 0-5: Existing features
        # 6: My liberties
        # 7: Opponent liberties
        # 8: My line potential
        # 9: Opponent line potential
        features = np.zeros(
            (10, self.board_size, self.board_size), dtype=np.float32
        )

        # Helper for liberties
        from .heuristic_ai import HeuristicAI
        # We can instantiate a temporary HeuristicAI to reuse helper methods
        # or just duplicate logic for speed. Duplicating logic is safer here to avoid circular deps or overhead.
        # Actually, we can just use BoardManager logic if available, or simple adjacency.
        
        # Simple adjacency logic (Square & Hex)
        def get_adjacent(x, y, size, is_hex=False):
            adj = []
            if is_hex:
                # Hexagonal directions (Axial coordinates)
                # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
                directions = [
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)
                ]
            else:
                # Square directions (8-way)
                directions = [
                    (-1, -1), (0, -1), (1, -1),
                    (-1, 0),           (1, 0),
                    (-1, 1),  (0, 1),  (1, 1)
                ]
                
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Bounds check depends on board type, but for feature extraction
                # we can just check if it's within the tensor size.
                # For hex, we map axial to tensor indices (offset or direct).
                # Here we assume direct mapping for simplicity, but hex grids
                # might need offset coordinates for rectangular storage.
                if 0 <= nx < size and 0 <= ny < size:
                    adj.append(f"{nx},{ny}")
            return adj

        is_hex = game_state.board.type == "hexagonal"

        for pos_key, stack in game_state.board.stacks.items():
            try:
                parts = pos_key.split(',')
                x, y = int(parts[0]), int(parts[1])
                
                # For hex, we might have z coordinate, but we use x,y for 2D tensor mapping.
                # Ensure x,y fit in board_size.
                if x >= self.board_size or y >= self.board_size:
                    continue

                val = min(stack.stack_height / 5.0, 1.0)  # Normalize height
                if stack.controlling_player == self.player_number:
                    features[0, x, y] = val
                else:
                    features[1, x, y] = val
            except ValueError:
                continue

        for pos_key, marker in game_state.board.markers.items():
            try:
                x, y = map(int, pos_key.split(',')[:2])
                if x >= self.board_size or y >= self.board_size:
                    continue

                if marker.player == self.player_number:
                    features[2, x, y] = 1.0
                else:
                    features[3, x, y] = 1.0
            except ValueError:
                continue

        for pos_key, player in game_state.board.collapsed_spaces.items():
            try:
                x, y = map(int, pos_key.split(',')[:2])
                if x >= self.board_size or y >= self.board_size:
                    continue

                if player == self.player_number:
                    features[4, x, y] = 1.0
                else:
                    features[5, x, y] = 1.0
            except ValueError:
                continue
                
        # Calculate liberties and line potential
        # This is a simplified version for feature extraction
        for pos_key, stack in game_state.board.stacks.items():
            try:
                x, y = map(int, pos_key.split(',')[:2])
                if x >= self.board_size or y >= self.board_size: continue
                
                # Liberties
                adj = get_adjacent(x, y, self.board_size, is_hex)
                liberties = 0
                for n_key in adj:
                    # Check if neighbor is occupied
                    # Note: n_key format "x,y" matches dictionary keys for square.
                    # For hex, keys might be "x,y,z". We need to handle that if we use exact key lookup.
                    # But here we are iterating adjacent coords.
                    # If the board uses "x,y,z" keys, we need to reconstruct z.
                    # z = -x - y
                    if is_hex:
                        nz = -int(n_key.split(',')[0]) - int(n_key.split(',')[1])
                        n_key_full = f"{n_key},{nz}"
                        if n_key_full in game_state.board.stacks or n_key_full in game_state.board.collapsed_spaces:
                            continue
                    else:
                        if n_key in game_state.board.stacks or n_key in game_state.board.collapsed_spaces:
                            continue
                            
                    liberties += 1
                
                max_libs = 6.0 if is_hex else 8.0
                val = min(liberties / max_libs, 1.0)
                if stack.controlling_player == self.player_number:
                    features[6, x, y] = val
                else:
                    features[7, x, y] = val
            except ValueError:
                continue
                
        # Line potential (simplified: markers with neighbors)
        for pos_key, marker in game_state.board.markers.items():
            try:
                x, y = map(int, pos_key.split(',')[:2])
                if x >= self.board_size or y >= self.board_size: continue
                
                adj = get_adjacent(x, y, self.board_size, is_hex)
                neighbors = 0
                for n_key in adj:
                    if is_hex:
                        nz = -int(n_key.split(',')[0]) - int(n_key.split(',')[1])
                        n_key_full = f"{n_key},{nz}"
                        if n_key_full in game_state.board.markers and game_state.board.markers[n_key_full].player == marker.player:
                            neighbors += 1
                    else:
                        if n_key in game_state.board.markers and game_state.board.markers[n_key].player == marker.player:
                            neighbors += 1
                
                max_neighbors = 6.0 if is_hex else 8.0
                val = min(neighbors / (max_neighbors / 2.0), 1.0)
                if marker.player == self.player_number:
                    features[8, x, y] = val
                else:
                    features[9, x, y] = val
            except ValueError:
                continue

        # Global features: 10 dims
        # Phase (5), Rings in hand (2), Eliminated rings (2), Turn (1)
        globals = np.zeros(10, dtype=np.float32)

        # Phase one-hot
        phases = [
            "ring_placement", "movement", "capture",
            "line_processing", "territory_processing"
        ]
        try:
            phase_idx = phases.index(game_state.current_phase.value)
            globals[phase_idx] = 1.0
        except ValueError:
            pass

        # Rings info
        my_player = next(
            (p for p in game_state.players
             if p.player_number == self.player_number),
            None
        )
        opp_player = next(
            (p for p in game_state.players
             if p.player_number != self.player_number),
            None
        )
        
        if my_player:
            globals[5] = my_player.rings_in_hand / 20.0
            globals[7] = my_player.eliminated_rings / 20.0
            
        if opp_player:
            globals[6] = opp_player.rings_in_hand / 20.0
            globals[8] = opp_player.eliminated_rings / 20.0
            
        # Is it my turn?
        if game_state.current_player == self.player_number:
            globals[9] = 1.0
            
        return features, globals

    # _evaluate_move_with_net is deprecated
