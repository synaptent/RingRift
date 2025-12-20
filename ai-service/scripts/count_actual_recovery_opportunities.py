#!/usr/bin/env python3
"""
Count actual recovery opportunities: states where player is eligible AND it's their turn.
"""

import json
from pathlib import Path

from app.game_engine import GameEngine
from app.models import GamePhase, GameState, MoveType
from app.rules.core import count_buried_rings, is_eligible_for_recovery, player_controls_any_stack, player_has_markers
from app.rules.recovery import get_expanded_recovery_moves
from app.training.generate_data import create_initial_state


def is_eligible_no_rings_requirement(state: GameState, player: int) -> bool:
    """Check recovery eligibility WITHOUT the rings_in_hand == 0 requirement."""
    board = state.board
    # Skip rings_in_hand check
    if player_controls_any_stack(board, player):
        return False
    if not player_has_markers(board, player):
        return False
    return not count_buried_rings(board, player) < 1


def analyze_game(game_data: dict, game_file: str, game_index: int) -> dict:
    """Analyze a single game for actual recovery opportunities."""
    moves_json = game_data.get('moves', [])
    board_type = game_data.get('board_type', 'square8')
    num_players = game_data.get('num_players', 2)

    state = create_initial_state(board_type=board_type, num_players=num_players)

    results = {
        'total_states': 0,
        'eligible_on_turn': 0,  # Player is eligible AND it's their turn
        'eligible_on_turn_in_movement': 0,  # Plus in movement phase
        'recovery_moves_available': 0,  # get_expanded_recovery_moves returns moves
        'actual_opportunities': [],  # Detailed list
        'eligible_but_not_movement': [],  # Eligible but in wrong phase
        # Alternate eligibility (without rings_in_hand requirement)
        'alt_eligible_on_turn': 0,
        'alt_eligible_on_turn_in_movement': 0,
    }

    for move_idx, m_json in enumerate(moves_json):
        move_type_str = m_json.get('type')

        try:
            move_type = MoveType(move_type_str)
        except ValueError:
            break

        # Check eligibility for current player BEFORE applying move
        current_player = state.current_player
        is_eligible = is_eligible_for_recovery(state, current_player)

        results['total_states'] += 1

        # Check alternate eligibility (no rings_in_hand requirement)
        alt_eligible = is_eligible_no_rings_requirement(state, current_player)
        if alt_eligible:
            results['alt_eligible_on_turn'] += 1
            if state.current_phase == GamePhase.MOVEMENT:
                results['alt_eligible_on_turn_in_movement'] += 1

        if is_eligible:
            results['eligible_on_turn'] += 1
            # Track what phase they're in when eligible
            phase_key = f'eligible_in_{state.current_phase.value}'
            results[phase_key] = results.get(phase_key, 0) + 1

            if state.current_phase == GamePhase.MOVEMENT:
                results['eligible_on_turn_in_movement'] += 1

                # Check if recovery moves are actually generated
                recovery_moves = get_expanded_recovery_moves(state, current_player)
                if recovery_moves:
                    results['recovery_moves_available'] += 1
                    results['actual_opportunities'].append({
                        'game_file': game_file,
                        'game_index': game_index,
                        'move_index': move_idx,
                        'player': current_player,
                        'phase': state.current_phase.value,
                        'num_recovery_moves': len(recovery_moves),
                    })
            else:
                # Track eligible but NOT in movement phase
                results['eligible_but_not_movement'].append({
                    'game_file': game_file,
                    'game_index': game_index,
                    'move_index': move_idx,
                    'player': current_player,
                    'phase': state.current_phase.value,
                })

        # Get valid moves and apply
        valid_moves = GameEngine.get_valid_moves(state, state.current_player)
        bookkeeping = None
        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req is not None:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(req, state)

        matched_move = None
        if bookkeeping and bookkeeping.type == move_type:
            matched_move = bookkeeping
        else:
            candidates = [vm for vm in valid_moves if vm.type == move_type]
            if m_json.get('to'):
                to_x, to_y = m_json['to']['x'], m_json['to']['y']
                candidates = [vm for vm in candidates if vm.to and vm.to.x == to_x and vm.to.y == to_y]
            if m_json.get('from'):
                from_x, from_y = m_json['from']['x'], m_json['from']['y']
                new_candidates = []
                for vm in candidates:
                    from_pos = getattr(vm, 'from_position', None) or getattr(vm, 'from_pos', None)
                    if from_pos and from_pos.x == from_x and from_pos.y == from_y:
                        new_candidates.append(vm)
                candidates = new_candidates
            if candidates:
                matched_move = candidates[0]

        if not matched_move:
            break

        try:
            state = GameEngine.apply_move(state, matched_move)
        except Exception:
            break

    return results


def main():
    # Find all selfplay game files
    selfplay_dirs = [
        Path("data/selfplay/recovery_analysis"),
        Path("data/selfplay/test_recovery"),
    ]

    all_games = []
    for d in selfplay_dirs:
        if d.exists():
            for f in d.glob("games.jsonl"):
                with open(f) as fp:
                    for line in fp:
                        line = line.strip()
                        if line:
                            try:
                                game = json.loads(line)
                                all_games.append((game, str(f)))
                            except json.JSONDecodeError:
                                pass

    print(f"Analyzing {len(all_games)} games...")

    totals = {
        'total_games': len(all_games),
        'total_states': 0,
        'eligible_on_turn': 0,
        'eligible_on_turn_in_movement': 0,
        'recovery_moves_available': 0,
        'actual_opportunities': [],
        'eligible_but_not_movement': [],
        'phase_breakdown': {},
        'alt_eligible_on_turn': 0,
        'alt_eligible_on_turn_in_movement': 0,
    }

    for i, (game, filepath) in enumerate(all_games):
        if i % 20 == 0:
            print(f"  Processing game {i}/{len(all_games)}...")

        results = analyze_game(game, filepath, i)
        totals['total_states'] += results['total_states']
        totals['eligible_on_turn'] += results['eligible_on_turn']
        totals['eligible_on_turn_in_movement'] += results['eligible_on_turn_in_movement']
        totals['recovery_moves_available'] += results['recovery_moves_available']
        totals['actual_opportunities'].extend(results['actual_opportunities'])
        totals['eligible_but_not_movement'].extend(results.get('eligible_but_not_movement', []))
        totals['alt_eligible_on_turn'] += results.get('alt_eligible_on_turn', 0)
        totals['alt_eligible_on_turn_in_movement'] += results.get('alt_eligible_on_turn_in_movement', 0)
        # Aggregate phase breakdown
        for key, val in results.items():
            if key.startswith('eligible_in_'):
                totals['phase_breakdown'][key] = totals['phase_breakdown'].get(key, 0) + val

    print(f"\n{'='*60}")
    print("ACTUAL RECOVERY OPPORTUNITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total games analyzed: {totals['total_games']}")
    print(f"Total game states checked: {totals['total_states']}")
    print("\nRecovery eligibility (player eligible AND it's their turn):")
    print(f"  Any phase: {totals['eligible_on_turn']} ({100*totals['eligible_on_turn']/totals['total_states']:.4f}%)")
    print(f"  Movement phase only: {totals['eligible_on_turn_in_movement']} ({100*totals['eligible_on_turn_in_movement']/totals['total_states']:.4f}%)")
    print(f"\nActual recovery moves available: {totals['recovery_moves_available']}")

    print(f"\n{'='*60}")
    print("ALTERNATE ELIGIBILITY (no rings_in_hand requirement)")
    print(f"{'='*60}")
    print(f"  Any phase: {totals['alt_eligible_on_turn']} ({100*totals['alt_eligible_on_turn']/totals['total_states']:.4f}%)")
    print(f"  Movement phase only: {totals['alt_eligible_on_turn_in_movement']} ({100*totals['alt_eligible_on_turn_in_movement']/totals['total_states']:.4f}%)")

    if totals['actual_opportunities']:
        print(f"\nDetailed opportunities ({len(totals['actual_opportunities'])} total):")
        for opp in totals['actual_opportunities'][:10]:
            print(f"  Game {opp['game_index']} move {opp['move_index']}: P{opp['player']} has {opp['num_recovery_moves']} recovery moves")

    # Save results
    output_file = Path("data/selfplay/actual_recovery_opportunities.json")
    with open(output_file, 'w') as f:
        json.dump(totals, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
