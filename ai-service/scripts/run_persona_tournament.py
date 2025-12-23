#!/usr/bin/env python3
"""Run round-robin tournaments comparing persona weight profiles.

This script pits trained CMA-ES optimized personas against untrained baseline
personas for each board type and player count configuration.

Usage:
    # Run tournament for square8 3-player
    python scripts/run_persona_tournament.py --board square8 --players 3

    # Run all tournaments
    python scripts/run_persona_tournament.py --all

    # Run specific board with more games
    python scripts/run_persona_tournament.py --board hex8 --players 4 --games 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import (
    HEURISTIC_WEIGHT_PROFILES,
    HEURISTIC_V1_BALANCED,
    HEURISTIC_V1_AGGRESSIVE,
    HEURISTIC_V1_TERRITORIAL,
    HEURISTIC_V1_DEFENSIVE,
)
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GamePhase, GameStatus

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PersonaConfig:
    """Configuration for a persona in the tournament."""
    name: str
    profile_id: str
    is_trained: bool
    source: str  # "baseline" or path to JSON file


def load_trained_weights(weights_dir: Path) -> dict[str, dict[str, Any]]:
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

            # Create profile ID
            board_abbrev = {
                "square8": "sq8",
                "square19": "sq19",
                "hex8": "hex8",
                "hexagonal": "hex",
            }.get(board_type, board_type)

            profile_id = f"trained_{board_abbrev}_{num_players}p_{persona}"
            trained[profile_id] = {
                "board_type": board_type,
                "num_players": num_players,
                "persona": persona,
                "weights": weights,
                "fitness": data.get("fitness", 0.0),
                "source": str(path),
            }

            # Register in global profile registry
            HEURISTIC_WEIGHT_PROFILES[profile_id] = weights

        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    return trained


def get_personas_for_config(
    board_type: str,
    num_players: int,
    trained_weights: dict[str, dict[str, Any]],
) -> list[PersonaConfig]:
    """Get all personas (trained + baseline) for a board/player config."""
    personas = []

    # Baseline personas (untrained)
    baseline_personas = [
        ("balanced", "heuristic_v1_balanced"),
        ("aggressive", "heuristic_v1_aggressive"),
        ("territorial", "heuristic_v1_territorial"),
        ("defensive", "heuristic_v1_defensive"),
    ]

    for name, profile_id in baseline_personas:
        personas.append(PersonaConfig(
            name=f"{name}_baseline",
            profile_id=profile_id,
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
        profile_id = f"trained_{board_abbrev}_{num_players}p_{persona_type}"
        if profile_id in trained_weights:
            info = trained_weights[profile_id]
            personas.append(PersonaConfig(
                name=f"{persona_type}_trained",
                profile_id=profile_id,
                is_trained=True,
                source=info["source"],
            ))

    return personas


def create_ai(player_number: int, profile_id: str, seed: int = 0) -> HeuristicAI:
    """Create a HeuristicAI with the specified weight profile."""
    config = AIConfig(
        difficulty=5,
        randomness=0.05,  # Small randomness for tie-breaking
        think_time=500,
        rngSeed=seed,
        heuristic_profile_id=profile_id,
    )
    return HeuristicAI(player_number, config)


def play_game(
    board_type: BoardType,
    num_players: int,
    personas: list[PersonaConfig],
    player_personas: dict[int, PersonaConfig],
    game_idx: int,
    max_moves: int = 2000,
) -> tuple[int | None, str]:
    """Play a single game and return (winner_player_number, victory_reason)."""
    engine = GameEngine()
    engine.initialize_game(
        board_type=board_type,
        num_players=num_players,
    )

    # Create AIs for each player
    ais = {}
    for player_num, persona in player_personas.items():
        seed = game_idx * 1000 + player_num
        ais[player_num] = create_ai(player_num, persona.profile_id, seed)

    moves = 0
    while engine.game_state.game_status == GameStatus.IN_PROGRESS and moves < max_moves:
        current_player = engine.game_state.current_player
        ai = ais.get(current_player)

        if ai is None:
            break

        try:
            move = ai.get_move(engine.game_state)
            if move is None:
                break
            engine.apply_move(move)
            moves += 1
        except Exception as e:
            logger.warning(f"Error in game: {e}")
            break

    # Determine winner
    if engine.game_state.winner is not None:
        winner = engine.game_state.winner
        # Infer victory reason
        state = engine.game_state
        elim = state.board.eliminated_rings.get(str(winner), 0)
        terr = sum(1 for p in state.board.collapsed_spaces.values() if p == winner)

        if elim >= state.victory_threshold:
            reason = "elimination"
        elif terr >= state.territory_victory_threshold:
            reason = "territory"
        elif state.lps_exclusive_player_for_completed_round == winner:
            reason = "lps"
        else:
            reason = "structural"

        return winner, reason

    return None, "draw"


def run_round_robin(
    board_type: str,
    num_players: int,
    personas: list[PersonaConfig],
    games_per_matchup: int = 20,
) -> dict[str, Any]:
    """Run a round-robin tournament between all personas."""
    bt = BoardType(board_type)

    # Results tracking
    wins: dict[str, int] = {p.name: 0 for p in personas}
    games_played: dict[str, int] = {p.name: 0 for p in personas}
    victories_by_type: dict[str, dict[str, int]] = {
        p.name: {"elimination": 0, "territory": 0, "lps": 0, "structural": 0, "draw": 0}
        for p in personas
    }

    # Head-to-head results
    head_to_head: dict[str, dict[str, int]] = {
        p.name: {q.name: 0 for q in personas} for p in personas
    }

    matchups = []

    # For 2-player: each pair plays games_per_matchup times
    if num_players == 2:
        for i, p1 in enumerate(personas):
            for p2 in personas[i+1:]:
                matchups.append((p1, p2))
    else:
        # For 3-4 player: each persona plays in seat rotations
        # Simplified: run games with different seat assignments
        import itertools
        for combo in itertools.combinations(personas, num_players):
            matchups.append(combo)

    total_games = len(matchups) * games_per_matchup
    logger.info(f"\nRunning {total_games} games ({len(matchups)} matchups x {games_per_matchup} games)")
    logger.info(f"Personas: {[p.name for p in personas]}")

    game_count = 0
    start_time = time.time()

    for matchup in matchups:
        for game_idx in range(games_per_matchup):
            # Rotate seat assignments
            if num_players == 2:
                p1, p2 = matchup
                if game_idx % 2 == 0:
                    player_personas = {1: p1, 2: p2}
                else:
                    player_personas = {1: p2, 2: p1}
            else:
                # Rotate seats for multiplayer
                rotated = list(matchup)
                for _ in range(game_idx % len(rotated)):
                    rotated.append(rotated.pop(0))
                player_personas = {i+1: p for i, p in enumerate(rotated)}

            winner, reason = play_game(bt, num_players, personas, player_personas, game_count)
            game_count += 1

            # Record results
            for player_num, persona in player_personas.items():
                games_played[persona.name] += 1
                if winner == player_num:
                    wins[persona.name] += 1
                    victories_by_type[persona.name][reason] += 1
                    # Record head-to-head
                    for other_num, other_persona in player_personas.items():
                        if other_num != player_num:
                            head_to_head[persona.name][other_persona.name] += 1
                elif winner is None:
                    victories_by_type[persona.name]["draw"] += 1

            # Progress update every 10 games
            if game_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = game_count / elapsed if elapsed > 0 else 0
                eta = (total_games - game_count) / rate if rate > 0 else 0
                logger.info(f"  Progress: {game_count}/{total_games} ({rate:.1f} games/sec, ETA {eta:.0f}s)")

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
        "personas": [p.name for p in personas],
        "trained_personas": [p.name for p in personas if p.is_trained],
        "games_per_matchup": games_per_matchup,
        "total_games": game_count,
        "wins": wins,
        "games_played": games_played,
        "win_rates": win_rates,
        "victories_by_type": victories_by_type,
        "head_to_head": head_to_head,
        "ranking": [name for name, _ in ranked],
        "duration_seconds": time.time() - start_time,
    }


def print_results(results: dict[str, Any]) -> None:
    """Print formatted tournament results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TOURNAMENT RESULTS: {results['board_type']} {results['num_players']}p")
    logger.info(f"{'='*60}")

    logger.info(f"\nTotal games: {results['total_games']}")
    logger.info(f"Duration: {results['duration_seconds']:.1f}s")

    logger.info(f"\n{'RANKING':-^60}")
    for rank, name in enumerate(results['ranking'], 1):
        win_rate = results['win_rates'][name]
        wins = results['wins'][name]
        games = results['games_played'][name]
        trained = " (TRAINED)" if name in results['trained_personas'] else ""
        logger.info(f"  {rank}. {name}: {win_rate:.1%} ({wins}/{games}){trained}")

    logger.info(f"\n{'VICTORY TYPES':-^60}")
    for name in results['ranking']:
        vt = results['victories_by_type'][name]
        logger.info(f"  {name}:")
        logger.info(f"    elimination: {vt['elimination']}, territory: {vt['territory']}, "
                   f"lps: {vt['lps']}, structural: {vt['structural']}")

    # Head-to-head summary for trained vs baseline
    trained = [n for n in results['ranking'] if n in results['trained_personas']]
    baseline = [n for n in results['ranking'] if n not in results['trained_personas']]

    if trained and baseline:
        logger.info(f"\n{'TRAINED vs BASELINE':-^60}")
        h2h = results['head_to_head']
        for t in trained:
            for b in baseline:
                t_wins = h2h[t][b]
                b_wins = h2h[b][t]
                total = t_wins + b_wins
                if total > 0:
                    logger.info(f"  {t} vs {b}: {t_wins}-{b_wins} ({t_wins/total:.1%} for trained)")


def main():
    parser = argparse.ArgumentParser(description="Run persona tournament")
    parser.add_argument("--board", type=str, choices=["square8", "hex8", "square19", "hexagonal"],
                       help="Board type")
    parser.add_argument("--players", type=int, choices=[2, 3, 4], help="Number of players")
    parser.add_argument("--games", type=int, default=20, help="Games per matchup")
    parser.add_argument("--all", action="store_true", help="Run all configurations")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    # Load trained weights
    weights_dir = ROOT / "data" / "optimized_weights"
    trained_weights = load_trained_weights(weights_dir)

    logger.info(f"Loaded {len(trained_weights)} trained weight profiles:")
    for pid, info in trained_weights.items():
        logger.info(f"  {pid}: fitness={info['fitness']:.1%}")

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
        logger.info(f"# Starting tournament: {board_type} {num_players}p")
        logger.info(f"{'#'*60}")

        personas = get_personas_for_config(board_type, num_players, trained_weights)

        if len(personas) < 2:
            logger.warning(f"Not enough personas for {board_type} {num_players}p, skipping")
            continue

        results = run_round_robin(board_type, num_players, personas, args.games)
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
