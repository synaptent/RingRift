#!/usr/bin/env python3
"""GPU-accelerated persona tournament runner.

This script runs tournaments between different heuristic personas using GPU
acceleration for high throughput (~10-100 games/sec vs ~0.1 games/sec CPU).

The key innovation is per-player weight support: each player in a game can
use different heuristic weights, enabling proper persona comparisons.

Usage:
    # Run tournament for square8 3-player
    python scripts/run_gpu_persona_tournament.py --board square8 --players 3

    # Run all tournaments
    python scripts/run_gpu_persona_tournament.py --all

    # Run with more games per matchup
    python scripts/run_gpu_persona_tournament.py --board hex8 --players 4 --games 50

Requires: CUDA GPU (or MPS on Apple Silicon, though slower)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import torch

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_batch_state import BatchGameState
from app.ai.gpu_batch import get_device
from app.ai.gpu_heuristic import evaluate_positions_batch
from app.ai.gpu_selection import select_moves_vectorized
from app.ai.gpu_game_types import GamePhase, GameStatus
from app.ai.heuristic_weights import (
    HEURISTIC_WEIGHT_PROFILES,
    HEURISTIC_V1_BALANCED,
    HEURISTIC_V1_AGGRESSIVE,
    HEURISTIC_V1_TERRITORIAL,
    HEURISTIC_V1_DEFENSIVE,
)
from app.models import BoardType
from app.rules.core import BOARD_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PersonaConfig:
    """Configuration for a persona in the tournament."""
    name: str
    weights: dict[str, float]
    is_trained: bool
    source: str


def load_trained_weights(weights_dir: Path) -> dict[str, PersonaConfig]:
    """Load all trained weight files from the optimized_weights directory."""
    trained = {}
    if not weights_dir.exists():
        logger.warning(f"Trained weights directory not found: {weights_dir}")
        return trained

    for path in weights_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)

            board_type = data.get("board_type", "unknown")
            num_players = data.get("num_players", 0)
            persona = data.get("persona", "unknown")
            weights = data.get("weights", {})

            if not weights:
                continue

            # Create identifier
            board_abbrev = {
                "square8": "sq8",
                "square19": "sq19",
                "hex8": "hex8",
                "hexagonal": "hex",
            }.get(board_type, board_type)

            key = f"{board_abbrev}_{num_players}p_{persona}"
            trained[key] = PersonaConfig(
                name=f"{persona}_trained",
                weights=weights,
                is_trained=True,
                source=str(path),
            )

        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    return trained


def get_personas_for_config(
    board_type: str,
    num_players: int,
    trained_weights: dict[str, PersonaConfig],
) -> list[PersonaConfig]:
    """Get all personas (trained + baseline) for a board/player config."""
    personas = []

    # Baseline personas (untrained)
    baseline_personas = [
        ("balanced_baseline", HEURISTIC_V1_BALANCED),
        ("aggressive_baseline", HEURISTIC_V1_AGGRESSIVE),
        ("territorial_baseline", HEURISTIC_V1_TERRITORIAL),
        ("defensive_baseline", HEURISTIC_V1_DEFENSIVE),
    ]

    for name, weights in baseline_personas:
        personas.append(PersonaConfig(
            name=name,
            weights=dict(weights),
            is_trained=False,
            source="baseline",
        ))

    # Trained personas
    board_abbrev = {
        "square8": "sq8",
        "square19": "sq19",
        "hex8": "hex8",
        "hexagonal": "hex",
    }.get(board_type, board_type)

    for persona_type in ["balanced", "aggressive", "territorial", "defensive"]:
        key = f"{board_abbrev}_{num_players}p_{persona_type}"
        if key in trained_weights:
            personas.append(trained_weights[key])

    return personas


def weights_to_selection_features(weights: dict[str, float]) -> dict[str, float]:
    """Convert full 45-weight profile to move selection feature weights.

    Maps strategic weight profiles (aggressive, territorial, etc.) to
    move selection features that influence which moves are chosen.

    The mapping captures strategic intent:
    - Aggressive: higher capture_value, lower adjacency
    - Territorial: higher adjacency/line_potential, lower capture_value
    - Defensive: balanced with higher noise for safety
    - Balanced: moderate all values
    """
    # Extract key weights that influence move selection
    stack_control = weights.get("WEIGHT_STACK_CONTROL", 9.39)
    territory = weights.get("WEIGHT_TERRITORY", 8.66)
    center = weights.get("WEIGHT_CENTER_CONTROL", 2.28)
    line_potential = weights.get("WEIGHT_LINE_POTENTIAL", 7.24)
    adjacency = weights.get("WEIGHT_ADJACENCY", 1.57)
    elim_rings = weights.get("WEIGHT_ELIMINATED_RINGS", 13.12)
    overtake = weights.get("WEIGHT_OVERTAKE_POTENTIAL", 5.96)

    # Normalize to reasonable feature weight ranges (0.5-8.0)
    # Higher stack_control/elim/overtake => more aggressive (higher capture value)
    aggression = (stack_control + elim_rings + overtake) / 30.0
    # Higher territory/adjacency => more territorial (higher adjacency bonus)
    territorial = (territory + adjacency + line_potential) / 20.0

    return {
        "center": max(0.5, min(5.0, center)),
        "capture_value": max(1.0, min(8.0, aggression * 6.0)),
        "adjacency": max(0.5, min(5.0, territorial * 3.0)),
        "line_potential": max(0.5, min(4.0, line_potential / 3.0)),
        "noise": 1.0,  # Keep some stochasticity
    }


class TournamentGameRunner(ParallelGameRunner):
    """Extended ParallelGameRunner with per-player weight support.

    This class enables running games where different players use different
    heuristic weight profiles, which is essential for tournament comparisons.

    Key differences from base ParallelGameRunner:
    - Stores a weight_bank of all persona profiles
    - Tracks player_persona_idx mapping players to personas
    - Overrides _select_moves to use per-game weights
    - Converts full 45-weight profiles to move selection features
    """

    def __init__(
        self,
        batch_size: int,
        board_size: int = 8,
        num_players: int = 2,
        device: str | None = None,
        weight_bank: list[dict[str, float]] | None = None,
        player_persona_idx: torch.Tensor | None = None,
        **kwargs,
    ):
        """Initialize tournament runner.

        Args:
            batch_size: Number of parallel games
            board_size: Board dimension
            num_players: Players per game
            device: Device to use (cuda/mps/cpu)
            weight_bank: List of weight dicts (personas) to choose from
            player_persona_idx: (batch_size, num_players) tensor mapping
                               player -> index into weight_bank
        """
        # Force heuristic selection for tournaments
        kwargs['use_heuristic_selection'] = True

        super().__init__(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=device,
            **kwargs,
        )

        self.weight_bank = weight_bank or [dict(HEURISTIC_V1_BALANCED)]
        self.player_persona_idx = player_persona_idx

        # Pre-compute selection features for each persona in weight_bank
        self.selection_features_bank = [
            weights_to_selection_features(w) for w in self.weight_bank
        ]

        # Cache for current weights (updated each step)
        self._current_selection_features: list[dict[str, float]] | None = None

    def _get_weights_for_current_players(self) -> list[dict[str, float]]:
        """Get the weight dict for each game's current player."""
        if self.player_persona_idx is None:
            # All players use the same weights
            return [self.weight_bank[0]] * self.batch_size

        weights_list = []
        current_players = self.state.current_player.cpu().tolist()
        player_idx = self.player_persona_idx.cpu().tolist()

        for g in range(self.batch_size):
            p = current_players[g]  # Current player (1-indexed)
            if p > 0 and p <= self.num_players:
                persona_idx = player_idx[g][p - 1]  # 0-indexed
                weights_list.append(self.weight_bank[persona_idx])
            else:
                weights_list.append(self.weight_bank[0])

        return weights_list

    def _get_selection_features_for_current_players(self) -> list[dict[str, float]]:
        """Get the move selection features for each game's current player."""
        if self.player_persona_idx is None:
            return [self.selection_features_bank[0]] * self.batch_size

        features_list = []
        current_players = self.state.current_player.cpu().tolist()
        player_idx = self.player_persona_idx.cpu().tolist()

        for g in range(self.batch_size):
            p = current_players[g]
            if p > 0 and p <= self.num_players:
                persona_idx = player_idx[g][p - 1]
                features_list.append(self.selection_features_bank[persona_idx])
            else:
                features_list.append(self.selection_features_bank[0])

        return features_list

    def _select_moves_with_per_game_weights(
        self,
        moves,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select moves using per-game weights from the current player's persona.

        This is a specialized version of _select_moves that uses different
        selection feature weights for each game based on the current player's
        assigned persona.
        """
        from app.ai.gpu_selection import select_moves_heuristic
        from app.ai.gpu_game_types import MoveType

        device = self.device
        batch_size = self.batch_size
        board_size = self.board_size

        # Initialize output
        selected = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

        if moves.total_moves == 0:
            return selected

        # Get per-game selection features
        features_list = self._get_selection_features_for_current_players()

        center = board_size // 2
        max_dist = center * 2.0

        # Extract move data
        game_idx = moves.game_idx.long()
        to_y = moves.to_y.long()
        to_x = moves.to_x.long()

        # Compute base features (same for all moves)
        dist_to_center = (
            (moves.to_y.float() - center).abs() +
            (moves.to_x.float() - center).abs()
        )

        # Target heights for capture scoring
        target_heights = self.state.stack_height[game_idx, to_y, to_x].float()
        is_capture = moves.move_type == MoveType.CAPTURE

        # Adjacency computation
        current_players = self.state.current_player[game_idx]
        so_padded = torch.nn.functional.pad(
            self.state.stack_owner.float(), (1, 1, 1, 1), value=0
        )
        adj_up = so_padded[game_idx, to_y, to_x + 1] == current_players.float()
        adj_down = so_padded[game_idx, to_y + 2, to_x + 1] == current_players.float()
        adj_left = so_padded[game_idx, to_y + 1, to_x] == current_players.float()
        adj_right = so_padded[game_idx, to_y + 1, to_x + 2] == current_players.float()
        adjacent_count = adj_up.float() + adj_down.float() + adj_left.float() + adj_right.float()

        # Line potential
        own_stacks = (self.state.stack_owner[game_idx] == current_players.view(-1, 1, 1))
        own_stacks_padded = torch.nn.functional.pad(own_stacks.float(), (1, 1, 1, 1), value=0)
        move_idx = torch.arange(moves.total_moves, device=device)
        h_left = own_stacks_padded[move_idx, to_y + 1, to_x]
        h_right = own_stacks_padded[move_idx, to_y + 1, to_x + 2]
        v_up = own_stacks_padded[move_idx, to_y, to_x + 1]
        v_down = own_stacks_padded[move_idx, to_y + 2, to_x + 1]
        line_count = h_left + h_right + v_up + v_down

        # Now apply per-game weights to compute final scores
        # Create per-move weight tensors by looking up each move's game's weights
        center_weights = torch.zeros(moves.total_moves, device=device)
        capture_weights = torch.zeros(moves.total_moves, device=device)
        adjacency_weights = torch.zeros(moves.total_moves, device=device)
        line_weights = torch.zeros(moves.total_moves, device=device)

        # Build weight tensors from per-game features
        game_idx_cpu = game_idx.cpu().tolist()
        center_w = [features_list[g].get("center", 3.0) for g in game_idx_cpu]
        capture_w = [features_list[g].get("capture_value", 5.0) for g in game_idx_cpu]
        adj_w = [features_list[g].get("adjacency", 2.0) for g in game_idx_cpu]
        line_w = [features_list[g].get("line_potential", 1.5) for g in game_idx_cpu]

        center_weights = torch.tensor(center_w, device=device, dtype=torch.float32)
        capture_weights = torch.tensor(capture_w, device=device, dtype=torch.float32)
        adjacency_weights = torch.tensor(adj_w, device=device, dtype=torch.float32)
        line_weights = torch.tensor(line_w, device=device, dtype=torch.float32)

        # Compute weighted scores
        center_score = (max_dist - dist_to_center) * center_weights
        capture_score = torch.where(
            is_capture,
            target_heights * capture_weights,
            torch.zeros_like(target_heights)
        )
        adjacency_score = adjacent_count * adjacency_weights
        line_score = line_count * line_weights
        noise = torch.rand(moves.total_moves, device=device)

        scores = center_score + capture_score + adjacency_score + line_score + noise

        # Segment-wise softmax sampling
        neg_inf = torch.full((batch_size,), float('-inf'), device=device)
        max_per_game = neg_inf.scatter_reduce(0, game_idx, scores, reduce='amax')
        scores_stable = scores - max_per_game[game_idx]
        exp_scores = torch.exp(scores_stable / self.temperature)

        sum_per_game = torch.zeros(batch_size, device=device)
        sum_per_game.scatter_add_(0, game_idx, exp_scores)

        probs = exp_scores / (sum_per_game[game_idx] + 1e-10)

        # Multinomial sampling per segment
        cumsum_probs = torch.cumsum(probs, dim=0)
        game_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
        game_starts[1:] = torch.searchsorted(game_idx, torch.arange(1, batch_size, device=device))

        rand_vals = torch.rand(batch_size, device=device)

        for g in torch.where(active_mask & (moves.moves_per_game > 0))[0].tolist():
            start = game_starts[g].item()
            count = moves.moves_per_game[g].item()
            if count == 0:
                continue

            seg_cumsum = cumsum_probs[start:start+count]
            if start > 0:
                seg_cumsum = seg_cumsum - cumsum_probs[start - 1]
            seg_cumsum = seg_cumsum / (seg_cumsum[-1] + 1e-10)

            local_idx = torch.searchsorted(seg_cumsum, rand_vals[g]).clamp(0, count - 1)
            selected[g] = local_idx

        return selected

    def _select_moves(
        self,
        moves,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Override base _select_moves to use per-game weights."""
        # Use our per-game weighted selection
        return self._select_moves_with_per_game_weights(moves, active_mask)

    def run_tournament_games(
        self,
        max_moves: int = 2000,
    ) -> dict[str, Any]:
        """Run tournament games with per-player weights.

        Returns:
            Dictionary with winners, move counts, etc.
        """
        self.reset_games()
        start_time = time.perf_counter()

        phase_steps = 0
        max_phase_steps = max_moves * 20

        while self.state.count_active() > 0 and phase_steps < max_phase_steps:
            active_mask = self.state.get_active_mask()

            # Enforce per-game move limit
            reached_limit = active_mask & (self.state.move_count >= max_moves)
            if reached_limit.any():
                self.state.game_status[reached_limit] = GameStatus.MAX_MOVES

            if self.state.count_active() == 0:
                break

            # Get per-player weights for current players
            weights_list = self._get_weights_for_current_players()

            # Step all games
            self._step_games(weights_list)
            phase_steps += 1

            # Check for victory
            self._check_victory_conditions()

        # Mark remaining active games as draws
        active_mask = self.state.get_active_mask()
        self.state.game_status[active_mask] = GameStatus.MAX_MOVES

        elapsed = time.perf_counter() - start_time

        # Extract results
        victory_types, _ = self.state.derive_victory_types_batch(max_moves)

        return {
            "winners": self.state.winner.cpu().tolist(),
            "move_counts": self.state.move_count.cpu().tolist(),
            "status": self.state.game_status.cpu().tolist(),
            "victory_types": victory_types,
            "elapsed_seconds": elapsed,
            "games_per_second": self.batch_size / elapsed if elapsed > 0 else 0,
        }


def run_gpu_round_robin(
    board_type: str,
    num_players: int,
    personas: list[PersonaConfig],
    games_per_matchup: int = 20,
    device: str | None = None,
) -> dict[str, Any]:
    """Run a GPU-accelerated round-robin tournament.

    For 2-player: each pair plays games_per_matchup times
    For 3-4 player: sample combinations of personas
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Get board configuration
    bt = BoardType(board_type)
    config = BOARD_CONFIGS[bt]
    board_size = config.size

    # Build weight bank (list of all persona weights)
    weight_bank = [p.weights for p in personas]
    persona_names = [p.name for p in personas]
    num_personas = len(personas)

    # Results tracking
    wins = {p.name: 0 for p in personas}
    games_played = {p.name: 0 for p in personas}
    head_to_head = {p.name: {q.name: 0 for q in personas} for p in personas}
    victories_by_type = {
        p.name: {"elimination": 0, "territory": 0, "lps": 0, "structural": 0, "draw": 0}
        for p in personas
    }

    # Generate matchups
    if num_players == 2:
        # Each pair plays games_per_matchup times
        matchups = list(combinations(range(num_personas), 2))
    else:
        # For 3-4 player, use combinations
        matchups = list(combinations(range(num_personas), num_players))
        # Limit to reasonable number
        if len(matchups) > 50:
            import random
            matchups = random.sample(matchups, 50)

    total_games = len(matchups) * games_per_matchup
    logger.info(f"\nRunning {total_games} GPU games ({len(matchups)} matchups x {games_per_matchup} games)")
    logger.info(f"Personas: {persona_names}")
    logger.info(f"Device: {device}")

    start_time = time.time()
    games_completed = 0

    # Process matchups in batches for efficiency
    batch_size = min(64, len(matchups) * games_per_matchup)

    # Create all games for this tournament
    all_game_assignments = []  # (persona_idx for each player)

    for matchup in matchups:
        for game_idx in range(games_per_matchup):
            # Rotate seats for fairness
            assignment = list(matchup)
            for _ in range(game_idx % len(assignment)):
                assignment.append(assignment.pop(0))
            all_game_assignments.append(assignment)

    # Process in batches
    for batch_start in range(0, len(all_game_assignments), batch_size):
        batch_end = min(batch_start + batch_size, len(all_game_assignments))
        batch_assignments = all_game_assignments[batch_start:batch_end]
        current_batch_size = len(batch_assignments)

        # Create player_persona_idx tensor for this batch
        player_persona_idx = torch.zeros(
            (current_batch_size, num_players),
            dtype=torch.long,
        )
        for g, assignment in enumerate(batch_assignments):
            for p_idx, persona_idx in enumerate(assignment):
                player_persona_idx[g, p_idx] = persona_idx

        # Create tournament runner
        runner = TournamentGameRunner(
            batch_size=current_batch_size,
            board_size=board_size,
            num_players=num_players,
            device=device,
            weight_bank=weight_bank,
            player_persona_idx=player_persona_idx,
        )

        # Run games
        results = runner.run_tournament_games(max_moves=2000)

        # Record results
        for g, assignment in enumerate(batch_assignments):
            winner_player = results["winners"][g]  # 1-indexed or 0 for draw
            victory_type = results["victory_types"][g]

            for p_idx, persona_idx in enumerate(assignment):
                persona_name = persona_names[persona_idx]
                games_played[persona_name] += 1

                if winner_player == p_idx + 1:  # Winner (1-indexed)
                    wins[persona_name] += 1
                    vt_key = victory_type if victory_type in victories_by_type[persona_name] else "structural"
                    victories_by_type[persona_name][vt_key] += 1

                    # Record head-to-head
                    for other_idx, other_persona_idx in enumerate(assignment):
                        if other_idx != p_idx:
                            other_name = persona_names[other_persona_idx]
                            head_to_head[persona_name][other_name] += 1
                elif winner_player == 0:
                    victories_by_type[persona_name]["draw"] += 1

        games_completed += current_batch_size
        elapsed = time.time() - start_time
        rate = games_completed / elapsed if elapsed > 0 else 0
        eta = (total_games - games_completed) / rate if rate > 0 else 0
        logger.info(f"  Progress: {games_completed}/{total_games} ({rate:.1f} games/sec, ETA {eta:.0f}s)")

    # Calculate win rates
    win_rates = {
        name: wins[name] / games_played[name] if games_played[name] > 0 else 0
        for name in wins
    }

    # Sort by win rate
    ranked = sorted(win_rates.items(), key=lambda x: -x[1])

    return {
        "board_type": board_type,
        "num_players": num_players,
        "personas": persona_names,
        "trained_personas": [p.name for p in personas if p.is_trained],
        "games_per_matchup": games_per_matchup,
        "total_games": games_completed,
        "wins": wins,
        "games_played": games_played,
        "win_rates": win_rates,
        "victories_by_type": victories_by_type,
        "head_to_head": head_to_head,
        "ranking": [name for name, _ in ranked],
        "duration_seconds": time.time() - start_time,
        "device": device,
    }


def print_results(results: dict[str, Any]) -> None:
    """Print formatted tournament results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"GPU TOURNAMENT RESULTS: {results['board_type']} {results['num_players']}p")
    logger.info(f"{'='*60}")

    logger.info(f"\nTotal games: {results['total_games']}")
    logger.info(f"Duration: {results['duration_seconds']:.1f}s ({results.get('device', 'unknown')})")

    logger.info(f"\n{'RANKING':-^60}")
    for rank, name in enumerate(results['ranking'], 1):
        win_rate = results['win_rates'][name]
        wins = results['wins'][name]
        games = results['games_played'][name]
        trained = " (TRAINED)" if name in results['trained_personas'] else ""
        logger.info(f"  {rank}. {name}: {win_rate:.1%} ({wins}/{games}){trained}")

    logger.info(f"\n{'VICTORY TYPES':-^60}")
    for name in results['ranking'][:4]:  # Top 4 only
        vt = results['victories_by_type'][name]
        logger.info(f"  {name}: elim={vt['elimination']}, terr={vt['territory']}, "
                   f"lps={vt['lps']}, struct={vt['structural']}")

    # Trained vs baseline summary
    trained = [n for n in results['ranking'] if n in results['trained_personas']]
    baseline = [n for n in results['ranking'] if n not in results['trained_personas']]

    if trained and baseline:
        logger.info(f"\n{'TRAINED vs BASELINE':-^60}")
        h2h = results['head_to_head']
        for t in trained[:2]:  # Top 2 trained
            for b in baseline[:2]:  # Top 2 baseline
                t_wins = h2h[t][b]
                b_wins = h2h[b][t]
                total = t_wins + b_wins
                if total > 0:
                    logger.info(f"  {t} vs {b}: {t_wins}-{b_wins} ({t_wins/total:.1%} trained)")


def main():
    parser = argparse.ArgumentParser(description="GPU persona tournament")
    parser.add_argument("--board", type=str, choices=["square8", "hex8", "square19", "hexagonal"],
                       help="Board type")
    parser.add_argument("--players", type=int, choices=[2, 3, 4], help="Number of players")
    parser.add_argument("--games", type=int, default=32, help="Games per matchup")
    parser.add_argument("--all", action="store_true", help="Run all configurations")
    parser.add_argument("--device", type=str, help="Device (cuda/mps/cpu)")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    # Check GPU availability
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        logger.warning("No GPU available, falling back to CPU (will be slow)")

    # Load trained weights
    weights_dir = ROOT / "data" / "optimized_weights"
    trained_weights = load_trained_weights(weights_dir)

    logger.info(f"Loaded {len(trained_weights)} trained weight profiles")

    # Determine which configs to run
    if args.all:
        configs = [
            ("square8", 2), ("square8", 3), ("square8", 4),
            ("hex8", 2), ("hex8", 3), ("hex8", 4),
        ]
    elif args.board and args.players:
        configs = [(args.board, args.players)]
    else:
        parser.error("Must specify --board and --players, or use --all")

    all_results = []

    for board_type, num_players in configs:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# GPU Tournament: {board_type} {num_players}p")
        logger.info(f"{'#'*60}")

        personas = get_personas_for_config(board_type, num_players, trained_weights)

        if len(personas) < 2:
            logger.warning(f"Not enough personas for {board_type} {num_players}p, skipping")
            continue

        results = run_gpu_round_robin(
            board_type, num_players, personas, args.games, device
        )
        print_results(results)
        all_results.append(results)

    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*60}")

    trained_better = 0
    baseline_better = 0

    for r in all_results:
        top = r['ranking'][0]
        is_trained = top in r['trained_personas']
        marker = "TRAINED" if is_trained else "BASELINE"
        logger.info(f"{r['board_type']} {r['num_players']}p: {top} ({marker}) - {r['win_rates'][top]:.1%}")

        if is_trained:
            trained_better += 1
        else:
            baseline_better += 1

    logger.info(f"\nTrained won {trained_better}/{len(all_results)} tournaments")
    logger.info(f"Baseline won {baseline_better}/{len(all_results)} tournaments")


if __name__ == "__main__":
    main()
