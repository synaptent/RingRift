#!/usr/bin/env python3
"""
Ensemble model evaluation via gauntlet.

Combines multiple models at inference time by averaging their policy/value outputs.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import random
import logging

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rules.core import BoardType
from app.training.game_gauntlet import play_single_game, BaselineOpponent
from app.training.encoding import HexStateEncoder


def load_model(path: str, board_type: BoardType, num_players: int, device: str):
    """Load a single model using UnifiedModelLoader for automatic architecture detection."""
    from app.ai.unified_loader import UnifiedModelLoader

    loader = UnifiedModelLoader(device=device)
    loaded = loader.load(
        checkpoint_path=path,
        board_type=board_type,
        num_players=num_players,
        strict=False,
        allow_fresh=False,
    )
    return loaded.model


class EnsembleAI:
    """AI that combines multiple neural net models."""

    def __init__(self, models, board_type, device, weights=None):
        self.models = models
        self.board_type = board_type
        self.device = device
        self.weights = weights if weights else [1.0/len(models)] * len(models)

        # Configure encoder with correct board size
        from app.ai.neural_net.architecture_registry import get_encoder_class_for_channels, get_expected_channels_from_model
        from app.ai.neural_net import HEX8_BOARD_SIZE, HEX_BOARD_SIZE, POLICY_SIZE_HEX8, P_HEX

        # Detect expected channels from first model
        channels = get_expected_channels_from_model(models[0])
        encoder_class = get_encoder_class_for_channels(channels) if channels else None

        # Set correct board size and policy size for hex8 vs hexagonal
        if board_type == BoardType.HEX8:
            board_size = HEX8_BOARD_SIZE  # 9
            policy_size = POLICY_SIZE_HEX8  # 4500
        else:
            board_size = HEX_BOARD_SIZE  # 25
            policy_size = P_HEX  # 91876

        if encoder_class is not None:
            self.encoder = encoder_class(board_size=board_size, policy_size=policy_size)
        else:
            self.encoder = HexStateEncoder(board_size=board_size, policy_size=policy_size)

    def select_move(self, state):
        """Get best move using ensemble policy."""
        import numpy as np
        from app.rules.default_engine import DefaultRulesEngine

        engine = DefaultRulesEngine()
        # encode() returns (features, global_features) tuple
        features, globals_vec = self.encoder.encode(state)
        # Convert to tensors - features shape is (C, H, W), globals is (G,)
        features_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(self.device)
        globals_tensor = torch.from_numpy(np.asarray(globals_vec, dtype=np.float32)).unsqueeze(0).to(self.device)

        # Get ensemble prediction
        # HexNeuralNet_v2 returns (value, policy) - value first!
        with torch.no_grad():
            policies = []
            for model, w in zip(self.models, self.weights):
                _, policy = model(features_tensor, globals_tensor)
                policies.append(F.softmax(policy, dim=-1) * w)
            combined = sum(policies)[0].cpu().numpy()

        legal_moves = engine.get_valid_moves(state, state.current_player)
        if not legal_moves:
            return None

        # Find best legal move
        best_move = None
        best_prob = -1

        for move in legal_moves:
            try:
                move_idx = self.encoder.move_to_index(move)
                if 0 <= move_idx < len(combined) and combined[move_idx] > best_prob:
                    best_prob = combined[move_idx]
                    best_move = move
            except (AttributeError, KeyError, IndexError):
                pass

        return best_move if best_move else random.choice(legal_moves)


def main():
    parser = argparse.ArgumentParser(description="Run gauntlet on ensemble of models")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--weights", nargs="+", type=float, help="Model weights")
    parser.add_argument("--board-type", default="hex8")
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--games", type=int, default=15)
    args = parser.parse_args()

    # Normalize weights
    weights = args.weights
    if weights:
        total = sum(weights)
        weights = [w / total for w in weights]

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    board_type_map = {
        "hex8": BoardType.HEX8,
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map[args.board_type.lower()]

    # Load models
    print(f"\nLoading {len(args.models)} models...")
    models = []
    for path in args.models:
        print(f"  Loading: {Path(path).name}")
        models.append(load_model(path, board_type, args.num_players, device))

    ensemble_ai = EnsembleAI(models, board_type, device, weights)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE GAUNTLET: {len(models)} models")
    print(f"Board: {args.board_type}, Players: {args.num_players}")
    print(f"{'='*60}")

    results = {}
    total_wins = 0
    total_games = 0

    # Use create_baseline_ai for proper opponent construction
    from app.training.game_gauntlet import create_baseline_ai, BaselineOpponent

    opponents = [
        ("random", BaselineOpponent.RANDOM),
        ("heuristic", BaselineOpponent.HEURISTIC),
    ]

    for opponent_name, baseline_type in opponents:
        print(f"\nvs {opponent_name}:")
        wins = 0

        for i in range(args.games):
            # Create opponent AIs for each non-candidate player (important for multiplayer!)
            opponent_ais = {}
            for p in range(1, args.num_players + 1):
                if p != 1:  # candidate is player 1
                    opponent_ais[p] = create_baseline_ai(
                        baseline=baseline_type,
                        player=p,
                        board_type=board_type,
                        num_players=args.num_players,
                    )

            # For backwards compatibility, also provide fallback opponent_ai
            opponent_ai = opponent_ais.get(2, list(opponent_ais.values())[0])

            result = play_single_game(
                candidate_ai=ensemble_ai,
                opponent_ai=opponent_ai,
                board_type=board_type,
                num_players=args.num_players,
                candidate_player=1,
                opponent_ais=opponent_ais,
            )
            won = result.winner == 1
            wins += int(won)
            status = "WIN" if won else "LOSS"
            print(f"  Game {i+1}/{args.games}: {status} ({result.move_count} moves)")

        rate = wins / args.games
        results[opponent_name] = (wins, args.games, rate)
        total_wins += wins
        total_games += args.games
        print(f"  Result: {wins}/{args.games} ({rate*100:.1f}%)")

    # Summary
    overall = total_wins / total_games
    elo = 1000 + (overall - 0.5) * 400

    print(f"\n{'='*60}")
    print(f"ENSEMBLE SUMMARY")
    print(f"{'='*60}")
    for opp, (w, g, r) in results.items():
        print(f"  {opp}: {w}/{g} ({r*100:.1f}%)")
    print(f"  Overall: {total_wins}/{total_games} ({overall*100:.1f}%)")
    print(f"  Estimated Elo: {elo:.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
