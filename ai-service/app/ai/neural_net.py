"""
Neural Network AI implementation for RingRift
Uses a simple feedforward neural network for move evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional
import random
from datetime import datetime
import uuid

from .base import BaseAI
from ..models import GameState, Move, Position, BoardType

class RingRiftCNN(nn.Module):
    def __init__(self, board_size=8, in_channels=4):
        super(RingRiftCNN, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * board_size * board_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch_size, in_channels, board_size, board_size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.tanh(self.fc2(x)) # Output between -1 and 1
        return x

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
        self.board_size = 8 # Default to 8x8 for now
        self.model = RingRiftCNN(board_size=self.board_size, in_channels=4)
        
        # Load weights if available
        try:
            self.model.load_state_dict(torch.load("ai-service/app/models/ringrift_v1.pth"))
            self.model.eval()
        except FileNotFoundError:
            print("No pre-trained model found, using random weights")
            self.model.eval()

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using neural network evaluation
        """
        self.simulate_thinking(min_ms=300, max_ms=1000)
        
        from ..game_engine import GameEngine
        valid_moves = GameEngine.get_valid_moves(game_state, self.player_number)
        
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
        Evaluate position using neural network
        """
        features = self._extract_features(game_state)
        tensor_input = torch.FloatTensor(features).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            score = self.model(tensor_input).item()
        return score

    def _extract_features(self, game_state: GameState) -> np.ndarray:
        """
        Convert game state to feature tensor for CNN
        Shape: (4, board_size, board_size)
        """
        features = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        
        for pos_key, stack in game_state.board.stacks.items():
            try:
                x, y = map(int, pos_key.split(',')[:2])
                if x >= self.board_size or y >= self.board_size: continue
                
                val = stack.stack_height / 5.0 # Normalize height
                if stack.controlling_player == self.player_number:
                    features[0, x, y] = val
                else:
                    features[1, x, y] = val
            except ValueError:
                continue
                
        for pos_key, marker in game_state.board.markers.items():
            try:
                x, y = map(int, pos_key.split(',')[:2])
                if x >= self.board_size or y >= self.board_size: continue
                
                if marker.player == self.player_number:
                    features[2, x, y] = 1.0
                else:
                    features[3, x, y] = 1.0
            except ValueError:
                continue
                
        return features

    # _evaluate_move_with_net is deprecated
