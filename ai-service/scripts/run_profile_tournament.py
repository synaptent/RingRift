#!/usr/bin/env python3
"""Tournament between two HeuristicAI weight profiles for multiplayer games.

For 3+ player games, properly tests all configurations:
- 1 AI with profile A vs 2 AIs with profile B
- 2 AIs with profile A vs 1 AI with profile B

Measures average win rate across all player positions.
"""

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_ai import HeuristicAI
from app.models import AIConfig, BoardType, GameState, GameStatus
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state
from scripts.lib.cli import BOARD_TYPE_MAP


def load_weight_profile(path: str) -> dict[str, float]:
    """Load weights from a JSON profile file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("weights", data)


def create_ai_with_weights(
    player_number: int,
    weights: dict[str, float],
    difficulty: int = 10,
    randomness: float = 0.02,
    rng_seed: int | None = None,
) -> HeuristicAI:
    """Create a HeuristicAI instance with custom weights applied."""
    ai = HeuristicAI(
        player_number,
        AIConfig(
            difficulty=difficulty,
            think_time=0,
            randomness=randomness,
            rngSeed=rng_seed,
            heuristic_profile_id=None,
        ),
    )
    # Override weights on the instance
    for name, value in weights.items():
        setattr(ai, name, value)
    return ai


def run_game(
    ais: list, board_type: BoardType, num_players: int, max_moves: int = 300, verbose: bool = False
) -> tuple[int | None, dict]:
    """
    Run a single game.
    Returns (winner player number (1-indexed) or None for draw, game_info dict).
    """
    game_state = create_initial_state(board_type, num_players)
    rules_engine = DefaultRulesEngine()
    move_count = 0
    move_types = []
    phases_seen = set()

    def _timeout_tiebreak_winner(final_state: GameState) -> int | None:
        """Deterministically select a winner for evaluation-only timeouts."""
        players = getattr(final_state, "players", None) or []
        if not players:
            return None

        board = getattr(final_state, "board", None)
        collapsed_spaces = {}
        markers = {}
        if board is not None:
            collapsed_spaces = (
                getattr(board, "collapsed_spaces", None)
                or getattr(board, "collapsedSpaces", None)
                or {}
            )
            markers = getattr(board, "markers", None) or {}

        territory_counts: dict[int, int] = {}
        try:
            for p_id in collapsed_spaces.values():
                territory_counts[int(p_id)] = territory_counts.get(int(p_id), 0) + 1
        except (ValueError, TypeError, KeyError, AttributeError):
            pass

        marker_counts: dict[int, int] = {int(p.player_number): 0 for p in players}
        try:
            for marker in markers.values():
                owner = int(getattr(marker, "player", getattr(marker, "player_number", 0)) or 0)
                if owner:
                    marker_counts[owner] = marker_counts.get(owner, 0) + 1
        except (ValueError, TypeError, KeyError, AttributeError):
            pass

        last_actor: int | None = None
        try:
            if final_state.move_history:
                last_actor = int(getattr(final_state.move_history[-1], "player", 0) or 0) or None
        except (ValueError, TypeError, IndexError, AttributeError):
            last_actor = None

        sorted_players = sorted(
            players,
            key=lambda p: (
                territory_counts.get(int(p.player_number), 0),
                int(getattr(p, "eliminated_rings", 0) or 0),
                marker_counts.get(int(p.player_number), 0),
                1 if last_actor == int(p.player_number) else 0,
                -int(p.player_number),
            ),
            reverse=True,
        )
        if not sorted_players:
            return None
        return int(sorted_players[0].player_number)

    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player_num = game_state.current_player
        current_ai = ais[current_player_num - 1]
        current_ai.player_number = current_player_num
        phases_seen.add(str(game_state.current_phase))

        try:
            move = current_ai.select_move(game_state)
        except Exception as e:
            print(f"Error in AI select_move: {e}")
            return None, {"error": str(e)}

        if not move:
            game_state.game_status = GameStatus.COMPLETED
            # Player with no moves loses - next player wins (simplified)
            game_state.winner = (current_player_num % num_players) + 1
            break

        move_types.append(move.type if hasattr(move, "type") else str(type(move).__name__))

        try:
            game_state = rules_engine.apply_move(game_state, move)
        except Exception as e:
            print(f"Error applying move: {e}")
            return None, {"error": str(e)}

        if game_state.game_status == GameStatus.COMPLETED:
            break

        move_count += 1

    # Collect game info for diagnostics
    game_info = {
        "move_count": move_count,
        "phases": list(phases_seen),
        "final_phase": str(game_state.current_phase),
        "final_status": str(game_state.game_status),
        "winner": game_state.winner,
        "players_eliminated": sum(1 for p in game_state.players if p.eliminated_rings >= 3),
        "total_rings_eliminated": game_state.total_rings_eliminated,
        "rings_in_play": game_state.total_rings_in_play,
    }

    # Add per-player stats
    for p in game_state.players:
        game_info[f"p{p.player_number}_rings_in_hand"] = p.rings_in_hand
        game_info[f"p{p.player_number}_eliminated_rings"] = p.eliminated_rings
        game_info[f"p{p.player_number}_territory"] = p.territory_spaces

    if verbose:
        print(
            f"  Game: {move_count} moves, winner=P{game_state.winner}, "
            f"eliminated={game_info['players_eliminated']}, "
            f"phases={game_info['phases']}"
        )

    if game_state.game_status == GameStatus.ACTIVE:
        winner = _timeout_tiebreak_winner(game_state)
        game_info["timeout_tiebreak_winner"] = winner
        return winner, game_info

    return game_state.winner, game_info


def generate_configurations(num_players: int) -> list[tuple[str, ...]]:
    """
    Generate all unique configurations for testing profile A vs profile B.

    For 3 players, generates:
    - (A, B, B) - A at position 1
    - (B, A, B) - A at position 2
    - (B, B, A) - A at position 3
    - (A, A, B) - B at position 3
    - (A, B, A) - B at position 2
    - (B, A, A) - B at position 1

    Returns list of tuples indicating profile assignment per position.
    """
    configs = []

    # Test "1 A vs rest B" - A at each position
    for a_pos in range(num_players):
        config = tuple("B" if i != a_pos else "A" for i in range(num_players))
        configs.append(config)

    # Test "1 B vs rest A" - B at each position
    for b_pos in range(num_players):
        config = tuple("A" if i != b_pos else "B" for i in range(num_players))
        configs.append(config)

    return configs


def main():
    parser = argparse.ArgumentParser(description="Run tournament between two weight profiles (multiplayer-aware)")
    parser.add_argument("--profile-a", type=str, required=True, help="Path to first weight profile JSON")
    parser.add_argument("--profile-b", type=str, required=True, help="Path to second weight profile JSON")
    parser.add_argument("--name-a", type=str, default="Profile_A", help="Display name for profile A")
    parser.add_argument("--name-b", type=str, default="Profile_B", help="Display name for profile B")
    parser.add_argument("--board", type=str, default="square8", choices=list(BOARD_TYPE_MAP.keys()), help="Board type")
    parser.add_argument("--num-players", type=int, default=3, help="Number of players (2-4)")
    parser.add_argument("--games-per-config", type=int, default=10, help="Number of games to play per configuration")
    parser.add_argument("--max-moves", type=int, default=10000, help="Maximum moves per game")
    parser.add_argument("--verbose", action="store_true", help="Print per-game diagnostics")
    parser.add_argument("--diagnostics", action="store_true", help="Print detailed game statistics at the end")

    args = parser.parse_args()

    # Load weight profiles
    weights_a = load_weight_profile(args.profile_a)
    weights_b = load_weight_profile(args.profile_b)

    board_type = BOARD_TYPE_MAP[args.board]
    num_players = args.num_players

    # Generate all configurations
    configs = generate_configurations(num_players)
    total_games = len(configs) * args.games_per_config

    print(f"\n{'='*60}")
    print(f"Tournament: {args.name_a} vs {args.name_b}")
    print(f"{'='*60}")
    print(f"Board: {args.board}")
    print(f"Players: {num_players}")
    print(f"Configurations: {len(configs)}")
    print(f"Games per config: {args.games_per_config}")
    print(f"Total games: {total_games}")
    print(f"Max moves: {args.max_moves}")
    print(f"{'='*60}")
    print("\nConfigurations to test:")
    for i, config in enumerate(configs):
        print(f"  {i+1}. {config}")
    print(f"{'='*60}\n")

    # Track wins for each profile
    total_a_wins = 0
    total_b_wins = 0
    total_draws = 0

    # Track per-configuration results
    config_results = {}

    # Track diagnostics
    all_game_infos = []
    wins_by_position = {i + 1: 0 for i in range(num_players)}

    for config_idx, config in enumerate(configs):
        config_a_wins = 0
        config_b_wins = 0
        config_draws = 0

        for game_num in range(args.games_per_config):
            # Create AIs with unique random seeds for each game
            game_seed = config_idx * 10000 + game_num * 100
            ais = []
            for p, profile in enumerate(config):
                # Each AI gets a unique seed based on game + player position
                ai_seed = game_seed + p
                if profile == "A":
                    ais.append(create_ai_with_weights(p + 1, weights_a, rng_seed=ai_seed))
                else:
                    ais.append(create_ai_with_weights(p + 1, weights_b, rng_seed=ai_seed))

            winner, game_info = run_game(ais, board_type, num_players, args.max_moves, verbose=args.verbose)
            game_info["config"] = config
            all_game_infos.append(game_info)

            if winner is None:
                config_draws += 1
                total_draws += 1
            else:
                wins_by_position[winner] += 1
                if config[winner - 1] == "A":
                    config_a_wins += 1
                    total_a_wins += 1
                else:
                    config_b_wins += 1
                    total_b_wins += 1

        config_results[config] = {"a_wins": config_a_wins, "b_wins": config_b_wins, "draws": config_draws}

        (config_idx + 1) * args.games_per_config
        print(
            f"Config {config_idx+1}/{len(configs)} {config}: "
            f"A={config_a_wins}, B={config_b_wins}, Draws={config_draws} "
            f"(Total: A={total_a_wins}, B={total_b_wins})"
        )

    print(f"\n{'='*60}")
    print("Final Results:")
    print(f"{'='*60}")

    # Show per-configuration breakdown
    print("\nPer-Configuration Results:")
    for config, results in config_results.items():
        total_in_config = results["a_wins"] + results["b_wins"] + results["draws"]
        a_pct = 100 * results["a_wins"] / total_in_config if total_in_config > 0 else 0
        b_pct = 100 * results["b_wins"] / total_in_config if total_in_config > 0 else 0
        print(
            f"  {config}: A={results['a_wins']} ({a_pct:.1f}%), "
            f"B={results['b_wins']} ({b_pct:.1f}%), Draws={results['draws']}"
        )

    print("\nOverall Results:")
    print(f"  {args.name_a}: {total_a_wins} wins ({100*total_a_wins/total_games:.1f}%)")
    print(f"  {args.name_b}: {total_b_wins} wins ({100*total_b_wins/total_games:.1f}%)")
    print(f"  Draws: {total_draws} ({100*total_draws/total_games:.1f}%)")
    print(f"{'='*60}")

    # Determine winner
    if total_a_wins > total_b_wins:
        margin = total_a_wins - total_b_wins
        print(f"\n>>> {args.name_a} is STRONGER (by {margin} games) <<<")
    elif total_b_wins > total_a_wins:
        margin = total_b_wins - total_a_wins
        print(f"\n>>> {args.name_b} is STRONGER (by {margin} games) <<<")
    else:
        print("\n>>> TIE <<<")

    # Print position-based wins
    print(f"\n{'='*60}")
    print("Wins by Player Position:")
    print(f"{'='*60}")
    for pos, wins in wins_by_position.items():
        pct = 100 * wins / total_games if total_games > 0 else 0
        print(f"  Player {pos}: {wins} wins ({pct:.1f}%)")

    # Print diagnostics if requested
    if args.diagnostics and all_game_infos:
        print(f"\n{'='*60}")
        print("Game Diagnostics:")
        print(f"{'='*60}")

        # Move count statistics
        move_counts = [g["move_count"] for g in all_game_infos]
        print("\nMove counts:")
        print(f"  Min: {min(move_counts)}")
        print(f"  Max: {max(move_counts)}")
        print(f"  Avg: {sum(move_counts)/len(move_counts):.1f}")
        print(f"  Unique values: {len(set(move_counts))}")

        # Check for identical games
        if len(set(move_counts)) == 1:
            print("  ⚠️  ALL GAMES SAME LENGTH - suspicious!")

        # Rings eliminated
        elim_counts = [g.get("total_rings_eliminated", 0) for g in all_game_infos]
        print("\nRings eliminated per game:")
        print(f"  Min: {min(elim_counts)}")
        print(f"  Max: {max(elim_counts)}")
        print(f"  Avg: {sum(elim_counts)/len(elim_counts):.1f}")
        print(f"  Unique values: {len(set(elim_counts))}")

        # Phases seen
        all_phases = set()
        for g in all_game_infos:
            all_phases.update(g.get("phases", []))
        print(f"\nGame phases seen: {sorted(all_phases)}")

        # Sample first 5 games
        print("\nSample games (first 5):")
        for i, g in enumerate(all_game_infos[:5]):
            print(
                f"  Game {i+1}: {g.get('move_count', '?')} moves, "
                f"winner=P{g.get('winner', '?')}, "
                f"elim={g.get('total_rings_eliminated', '?')}, "
                f"config={g.get('config', '?')}"
            )

        # Per-player stats from last game
        last = all_game_infos[-1]
        print("\nLast game player stats:")
        for p in range(1, num_players + 1):
            rih = last.get(f"p{p}_rings_in_hand", "?")
            elim = last.get(f"p{p}_eliminated_rings", "?")
            terr = last.get(f"p{p}_territory", "?")
            print(f"  P{p}: rings_in_hand={rih}, eliminated={elim}, territory={terr}")


if __name__ == "__main__":
    main()
