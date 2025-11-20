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
        
        # Fully connected layers
        conv_out_size = num_filters * board_size * board_size
        self.fc1 = nn.Linear(conv_out_size + global_features, 256)
        self.dropout = nn.Dropout(0.3)
        
        # Value head
        self.value_head = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        
        # Policy head
        # Output size: ~55,000 for 19x19 board support
        # Movement: 361 (from) * 8 (dir) * 18 (dist) = 51,984
        # Placement: 361 (pos) * 3 (count) = 1,083
        # Line: 361 (pos) * 4 (dir) = 1,444
        # Territory: 361 (pos) = 361
        # Special: 64 (buffer)
        # Total: 55,000 (approx)
        # Note: We keep this large size to support up to 19x19, even if current board is 8x8.
        # This allows the model architecture to remain consistent if board size changes,
        # though it is sparse for 8x8.
        self.policy_size = 55000
        self.policy_head = nn.Linear(256, self.policy_size)
        # self.softmax = nn.Softmax(dim=1) # Removed to output logits for CrossEntropyLoss

    def forward(self, x, globals):
        x = self.relu(self.bn1(self.conv1(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
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
        self.board_size = 8  # Default to 8x8 for now
        self.history_length = 3
        self.state_history = [] # Stores last N feature planes
        
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
                    torch.load(model_path)
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

    def evaluate_position(self, game_state: GameState) -> tuple[float, np.ndarray]:
        """
        Evaluate position using neural network
        Returns: (value, policy_probs)
        """
        value, policy_logits = self.evaluate_batch([game_state])
        return value[0], policy_logits[0]

    def evaluate_batch(self, game_states: list[GameState]) -> tuple[list[float], list[np.ndarray]]:
        """
        Evaluate a batch of game states
        Returns: (values, policy_probs)
        """
        if not game_states:
            return [], []

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
        Uses fixed offsets based on MAX_BOARD_SIZE=19 to ensure consistent indices across board sizes.
        """
        MAX_N = 19  # Fixed max board size for encoding consistency
        
        # Ensure coordinates are within bounds (sanity check)
        # Note: We use the actual board_size for bounds checking if needed,
        # but MAX_N for index calculation.
        
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
                return 0 # Should not happen
                
            from_idx = move.from_pos.y * MAX_N + move.from_pos.x
            
            dx = move.to.x - move.from_pos.x
            dy = move.to.y - move.from_pos.y
            
            dist = max(abs(dx), abs(dy)) # Chebyshev distance
            if dist == 0: return 0
            
            # Normalize direction
            dir_x = dx // dist if dist > 0 else 0
            dir_y = dy // dist if dist > 0 else 0
            
            # Map direction to 0-7
            # (-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)
            dirs = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1)
            ]
            try:
                dir_idx = dirs.index((dir_x, dir_y))
            except ValueError:
                return 0 # Invalid direction
                
            base = 1083
            max_dist = MAX_N - 1
            return base + from_idx * (8 * max_dist) + dir_idx * max_dist + (dist - 1)
            
        elif move.type == "line_formation":
            # Line: 53067 to 54510
            # Base = 53067
            # Index = Base + (y * MAX_N + x) * 4 + dir_idx
            # Directions: Horizontal(0), Vertical(1), Diagonal(2), Anti-Diagonal(3)
            # We need to infer direction from line info if available, or just use to_pos
            # For simplicity, let's assume to_pos is start and we need direction
            # This is tricky without extra info.
            # Placeholder: map to start pos
            pos_idx = move.to.y * MAX_N + move.to.x
            return 53067 + pos_idx * 4 # Default to dir 0
            
        elif move.type == "territory_claim":
            # Territory: 54511 to 54871
            # Base = 54511
            # Index = Base + (y * MAX_N + x)
            pos_idx = move.to.y * MAX_N + move.to.x
            return 54511 + pos_idx
            
        elif move.type == "skip_placement":
            return 54872
            
        return 0 # Unknown

    def decode_move(self, index: int, game_state: GameState) -> Optional[Move]:
        """
        Decode a policy index into a move.
        Uses fixed offsets based on MAX_BOARD_SIZE=19.
        """
        MAX_N = 19
        
        if index < 1083:
            # Placement
            count_idx = index % 3
            pos_idx = index // 3
            y = pos_idx // MAX_N
            x = pos_idx % MAX_N
            
            # Check bounds for actual board size
            if x >= self.board_size or y >= self.board_size:
                return None
                
            return Move(
                id="decoded", type="place_ring", player=game_state.current_player,
                to={"x": x, "y": y}, placement_count=count_idx + 1,
                timestamp=datetime.now(), think_time=0, move_number=0
            )
            
        elif index < 53067:
            # Movement
            # Index = Base + from_idx * (8 * max_dist) + dir_idx * max_dist + (dist - 1)
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
            
            # Check bounds for actual board size
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
            
            # Check bounds
            if not (0 <= to_x < self.board_size and 0 <= to_y < self.board_size):
                return None
                
            return Move(
                id="decoded", type="move_stack", player=game_state.current_player,
                from_pos={"x": from_x, "y": from_y}, to={"x": to_x, "y": to_y},
                timestamp=datetime.now(), think_time=0, move_number=0
            )
            
        elif index == 54872:
            return Move(
                id="decoded", type="skip_placement", player=game_state.current_player,
                to={"x": 0, "y": 0}, timestamp=datetime.now(), think_time=0, move_number=0
            )
            
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
        
        # Simple adjacency logic
        def get_adjacent(x, y, size):
            adj = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        adj.append(f"{nx},{ny}")
            return adj

        for pos_key, stack in game_state.board.stacks.items():
            try:
                x, y = map(int, pos_key.split(',')[:2])
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
                adj = get_adjacent(x, y, self.board_size)
                liberties = 0
                for n_key in adj:
                    if n_key not in game_state.board.stacks and n_key not in game_state.board.collapsed_spaces:
                        liberties += 1
                
                val = min(liberties / 8.0, 1.0)
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
                
                adj = get_adjacent(x, y, self.board_size)
                neighbors = 0
                for n_key in adj:
                    if n_key in game_state.board.markers and game_state.board.markers[n_key].player == marker.player:
                        neighbors += 1
                
                val = min(neighbors / 4.0, 1.0)
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
