#!/usr/bin/env python3
"""Model Ensembling - Average weights from top N models.

This script creates an ensemble model by averaging the weights of multiple
trained models. Ensembling often produces more robust models than any single
model, as it smooths out individual model quirks.

Usage:
    # Ensemble top 3 models for square8_2p
    python scripts/ensemble_models.py \
        --models models/square8_2p_v1.pt models/square8_2p_v2.pt models/square8_2p_v3.pt \
        --output models/square8_2p_ensemble.pt

    # Auto-detect top N models by Elo from database
    python scripts/ensemble_models.py \
        --board-type square8 --num-players 2 --top-n 5 \
        --output models/square8_2p_ensemble.pt

    # Weighted ensemble (by Elo)
    python scripts/ensemble_models.py \
        --board-type square8 --num-players 2 --top-n 5 --weighted \
        --output models/square8_2p_ensemble.pt
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def load_model_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        # May be raw state dict
        return checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")


def save_ensemble(
    state_dict: Dict[str, torch.Tensor],
    output_path: str,
    source_models: List[str],
    weights: Optional[List[float]] = None,
):
    """Save an ensemble model with metadata."""
    checkpoint = {
        "model_state_dict": state_dict,
        "ensemble_metadata": {
            "source_models": source_models,
            "weights": weights or [1.0 / len(source_models)] * len(source_models),
            "ensemble_type": "weighted_average" if weights else "simple_average",
        },
    }
    torch.save(checkpoint, output_path)
    print(f"Saved ensemble model to {output_path}")


def average_weights(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Average weights from multiple models.

    Args:
        model_paths: List of paths to model checkpoint files
        weights: Optional list of weights for weighted average (must sum to 1)

    Returns:
        Averaged state dict
    """
    if not model_paths:
        raise ValueError("No model paths provided")

    if weights:
        if len(weights) != len(model_paths):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(model_paths)})")
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]  # Normalize
    else:
        # Equal weights
        weights = [1.0 / len(model_paths)] * len(model_paths)

    print(f"Ensembling {len(model_paths)} models with weights: {[f'{w:.3f}' for w in weights]}")

    # Load first model as base
    ensemble_state = {}
    first_state = load_model_weights(model_paths[0])

    for key in first_state:
        ensemble_state[key] = first_state[key].float() * weights[0]

    # Add remaining models
    for i, model_path in enumerate(model_paths[1:], 1):
        state = load_model_weights(model_path)
        for key in ensemble_state:
            if key in state:
                ensemble_state[key] += state[key].float() * weights[i]
            else:
                print(f"Warning: Key {key} not found in model {model_path}")

    return ensemble_state


def get_top_models_from_db(
    db_path: str,
    board_type: str,
    num_players: int,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Get top N models by Elo from the training database.

    Returns list of (model_path, elo) tuples sorted by Elo descending.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Try to find models in training_history or elo tables
    try:
        cursor.execute("""
            SELECT model_path, elo
            FROM training_history
            WHERE board_type = ? AND num_players = ? AND model_path IS NOT NULL AND elo IS NOT NULL
            ORDER BY elo DESC
            LIMIT ?
        """, (board_type, num_players, top_n))
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        # Table doesn't have elo column, try different approach
        cursor.execute("""
            SELECT model_path
            FROM training_history
            WHERE board_type = ? AND num_players = ? AND model_path IS NOT NULL AND promoted = 1
            ORDER BY completed_at DESC
            LIMIT ?
        """, (board_type, num_players, top_n))
        rows = [(row[0], 1500.0) for row in cursor.fetchall()]  # Default Elo

    conn.close()

    # Filter to models that exist
    valid_models = []
    for model_path, elo in rows:
        if Path(model_path).exists():
            valid_models.append((model_path, elo))
        else:
            print(f"Warning: Model file not found: {model_path}")

    return valid_models


def main():
    parser = argparse.ArgumentParser(
        description="Create ensemble model by averaging weights from multiple models"
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        help="List of model checkpoint paths to ensemble"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path for ensemble model"
    )
    parser.add_argument(
        "--weights", "-w", type=float, nargs="+",
        help="Optional weights for each model (will be normalized)"
    )
    parser.add_argument(
        "--board-type", "-b",
        help="Board type for auto-detecting top models from DB"
    )
    parser.add_argument(
        "--num-players", "-p", type=int,
        help="Number of players for auto-detecting top models"
    )
    parser.add_argument(
        "--top-n", "-n", type=int, default=5,
        help="Number of top models to ensemble (default: 5)"
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Path to training database (default: auto-detect)"
    )
    parser.add_argument(
        "--weighted", action="store_true",
        help="Weight models by their Elo (higher Elo = higher weight)"
    )

    args = parser.parse_args()

    model_paths = args.models or []
    weights = args.weights

    # Auto-detect models from database if not provided
    if not model_paths and args.board_type and args.num_players:
        db_path = args.db_path
        if not db_path:
            # Try common locations
            for path in [
                "ai-service/logs/improvement_daemon/state.db",
                "logs/improvement_daemon/state.db",
                "../ai-service/logs/improvement_daemon/state.db",
            ]:
                if Path(path).exists():
                    db_path = path
                    break

        if not db_path:
            print("Error: Could not find training database. Please specify --db-path")
            return 1

        print(f"Loading top {args.top_n} models from {db_path}...")
        models_with_elo = get_top_models_from_db(
            db_path, args.board_type, args.num_players, args.top_n
        )

        if not models_with_elo:
            print(f"No models found for {args.board_type}_{args.num_players}p")
            return 1

        model_paths = [m[0] for m in models_with_elo]

        if args.weighted:
            # Weight by Elo - higher Elo = higher weight
            elos = [m[1] for m in models_with_elo]
            min_elo = min(elos)
            weights = [e - min_elo + 100 for e in elos]  # +100 to avoid zero weights

        print(f"Found {len(model_paths)} models:")
        for path, elo in models_with_elo:
            print(f"  - {Path(path).name}: Elo {elo:.0f}")

    if not model_paths:
        print("Error: No models specified. Use --models or --board-type/--num-players")
        return 1

    if len(model_paths) < 2:
        print("Error: Need at least 2 models to create an ensemble")
        return 1

    # Create ensemble
    try:
        ensemble_state = average_weights(model_paths, weights)
        save_ensemble(
            ensemble_state,
            args.output,
            model_paths,
            weights,
        )
        print(f"\nCreated ensemble from {len(model_paths)} models:")
        for i, path in enumerate(model_paths):
            w = weights[i] if weights else 1.0 / len(model_paths)
            print(f"  {w:.1%} - {Path(path).name}")

    except Exception as e:
        print(f"Error creating ensemble: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
