"""GPU heuristic evaluation for parallel games.

This module provides heuristic position evaluation for the GPU parallel games
system. Extracted from gpu_parallel_games.py for modularity.

December 2025: Extracted as part of R17 refactoring.

Implements comprehensive 45-weight heuristic evaluation per RR-CANON rules.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .gpu_parallel_games import BatchGameState


def evaluate_positions_batch(
    state: "BatchGameState",
    weights: Dict[str, float],
) -> torch.Tensor:
    """Evaluate all positions using comprehensive heuristic scoring.

    Implements all 45 heuristic weights from BASE_V1_BALANCED_WEIGHTS to match
    the CPU HeuristicAI evaluation. Weights are organized into categories:

    Core Position Weights:
    - WEIGHT_STACK_CONTROL: Number of controlled stacks
    - WEIGHT_STACK_HEIGHT: Total ring height on controlled stacks
    - WEIGHT_CAP_HEIGHT: Summed cap height (capture power)
    - WEIGHT_TERRITORY: Territory count
    - WEIGHT_RINGS_IN_HAND: Rings available to place
    - WEIGHT_CENTER_CONTROL: Stacks near board center
    - WEIGHT_ADJACENCY: Adjacency bonuses for stack clusters

    Threat/Defense Weights:
    - WEIGHT_OPPONENT_THREAT: Opponent line threats
    - WEIGHT_MOBILITY: Available movement options
    - WEIGHT_ELIMINATED_RINGS: Rings eliminated by this player
    - WEIGHT_VULNERABILITY: Exposure to capture

    Line/Victory Weights:
    - WEIGHT_LINE_POTENTIAL: 2/3/4-in-a-row patterns
    - WEIGHT_VICTORY_PROXIMITY: Distance to victory threshold
    - WEIGHT_LINE_CONNECTIVITY: Connected line structures

    Advanced Weights (v1.1+):
    - WEIGHT_MARKER_COUNT: Board markers controlled
    - WEIGHT_OVERTAKE_POTENTIAL: Capture opportunities
    - WEIGHT_TERRITORY_CLOSURE: Enclosed territory potential
    - WEIGHT_TERRITORY_SAFETY: Protected territory
    - WEIGHT_STACK_MOBILITY: Per-stack movement freedom
    - WEIGHT_OPPONENT_VICTORY_THREAT: Opponent's progress to victory
    - WEIGHT_FORCED_ELIMINATION_RISK: Risk of forced elimination
    - WEIGHT_LPS_ACTION_ADVANTAGE: Last Player Standing advantage
    - WEIGHT_MULTI_LEADER_THREAT: Multiple opponents with lead

    Penalty/Bonus Weights (v1.1 refactor):
    - WEIGHT_NO_STACKS_PENALTY: Massive penalty for no controlled stacks
    - WEIGHT_SINGLE_STACK_PENALTY: Penalty for single stack vulnerability
    - WEIGHT_STACK_DIVERSITY_BONUS: Spread vs concentration
    - WEIGHT_SAFE_MOVE_BONUS: Bonus for safe moves available
    - WEIGHT_NO_SAFE_MOVES_PENALTY: Penalty for no safe moves
    - WEIGHT_VICTORY_THRESHOLD_BONUS: Near-victory bonus

    Line Pattern Weights:
    - WEIGHT_TWO_IN_ROW, WEIGHT_THREE_IN_ROW, WEIGHT_FOUR_IN_ROW
    - WEIGHT_CONNECTED_NEIGHBOR, WEIGHT_GAP_POTENTIAL
    - WEIGHT_BLOCKED_STACK_PENALTY

    Swap/Opening Weights (v1.2-v1.4):
    - WEIGHT_SWAP_OPENING_CENTER, WEIGHT_SWAP_OPENING_ADJACENCY
    - WEIGHT_SWAP_OPENING_HEIGHT, WEIGHT_SWAP_CORNER_PENALTY
    - WEIGHT_SWAP_EDGE_BONUS, WEIGHT_SWAP_DIAGONAL_BONUS
    - WEIGHT_SWAP_OPENING_STRENGTH, WEIGHT_SWAP_EXPLORATION_TEMPERATURE

    Recovery Weights (v1.5):
    - WEIGHT_RECOVERY_POTENTIAL, WEIGHT_RECOVERY_ELIGIBILITY
    - WEIGHT_BURIED_RING_VALUE, WEIGHT_RECOVERY_THREAT

    Args:
        state: BatchGameState to evaluate
        weights: Heuristic weight dictionary (can use either old 8-weight format
                 or full 45-weight format; missing weights use defaults)

    Returns:
        Tensor of scores (batch_size, num_players) for each player
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    num_players = state.num_players
    center = board_size // 2

    scores = torch.zeros(batch_size, num_players + 1, dtype=torch.float32, device=device)

    # Pre-compute center distance matrix
    y_coords = torch.arange(board_size, device=device).view(-1, 1).expand(board_size, board_size)
    x_coords = torch.arange(board_size, device=device).view(1, -1).expand(board_size, board_size)
    center_dist = ((y_coords - center).abs() + (x_coords - center).abs()).float()
    max_dist = center_dist.max()
    center_bonus = (max_dist - center_dist) / max_dist  # 1.0 at center, 0.0 at corners

    # Canonical victory thresholds (RR-CANON-R061/R062-v2).
    # Keep this in sync with app.rules.core.BOARD_CONFIGS.
    from app.models import BoardType
    from app.rules.core import get_territory_victory_minimum, get_victory_threshold

    board_type_map = {
        8: BoardType.SQUARE8,
        9: BoardType.HEX8,
        19: BoardType.SQUARE19,
        13: BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(board_size, BoardType.SQUARE8)
    # Per RR-CANON-R062-v2: Use player-count-aware minimum threshold
    territory_victory_minimum = get_territory_victory_minimum(board_type, num_players)
    ring_victory_threshold = get_victory_threshold(board_type, num_players)

    # Weight mapping: support both old 8-weight format and new 45-weight format
    def get_weight(new_key: str, old_key: str = None, default: float = 0.0) -> float:
        """Get weight from dict, checking both new and old key formats."""
        if new_key in weights:
            return weights[new_key]
        if old_key and old_key in weights:
            return weights[old_key]
        return default

    for p in range(1, num_players + 1):
        # === CORE POSITION METRICS ===
        player_stacks = (state.stack_owner == p)
        stack_count = player_stacks.sum(dim=(1, 2)).float()

        # Total rings on controlled stacks
        player_heights = state.stack_height * player_stacks.int()
        total_ring_count = player_heights.sum(dim=(1, 2)).float()

        # Cap height (sum of stack heights, reflects capture power)
        cap_height = total_ring_count.clone()  # Same as ring count for now

        # Rings in hand
        rings_in_hand = state.rings_in_hand[:, p].float()

        # Territory count
        territory = state.territory_count[:, p].float()

        # Center control: weighted sum of positions near center
        center_control = (center_bonus.unsqueeze(0) * player_stacks.float()).sum(dim=(1, 2))

        # === STACK HEIGHT METRICS ===
        # Average stack height
        avg_height = total_ring_count / (stack_count + 1e-6)

        # Tall stacks bonus (height 3+)
        tall_stacks = ((state.stack_height >= 3) & player_stacks).sum(dim=(1, 2)).float()

        # === ADJACENCY METRICS ===
        # Count adjacent pairs of controlled stacks (vectorized)
        # Check horizontal adjacency: player_stacks[:, :, :-1] AND player_stacks[:, :, 1:]
        player_stacks_float = player_stacks.float()
        horizontal_adj = (player_stacks_float[:, :, :-1] * player_stacks_float[:, :, 1:]).sum(dim=(1, 2))
        # Check vertical adjacency: player_stacks[:, :-1, :] AND player_stacks[:, 1:, :]
        vertical_adj = (player_stacks_float[:, :-1, :] * player_stacks_float[:, 1:, :]).sum(dim=(1, 2))
        adjacency_score = horizontal_adj + vertical_adj

        # === MARKER METRICS ===
        marker_count = (state.marker_owner == p).sum(dim=(1, 2)).float()

        # === ELIMINATED RINGS METRICS ===
        eliminated_rings = state.eliminated_rings[:, p].float()

        # === BURIED RINGS METRICS ===
        buried_rings = state.buried_rings[:, p].float()

        # === MOBILITY METRICS (simplified) ===
        # Approximate mobility by stack count and territory
        mobility = stack_count * 4.0 + territory * 0.5  # Each stack ~4 moves avg

        # === LINE POTENTIAL METRICS (VECTORIZED) ===
        # Track 2/3/4-in-a-row patterns using tensor operations instead of per-game loops
        # This provides ~10x speedup over the naive O(batch * board^2 * directions) approach

        # HORIZONTAL patterns: check consecutive columns
        # 2-in-row: stack at (y, x) AND stack at (y, x+1)
        h2 = (player_stacks_float[:, :, :-1] * player_stacks_float[:, :, 1:]).sum(dim=(1, 2))
        # 3-in-row: (y, x) AND (y, x+1) AND (y, x+2)
        h3 = (player_stacks_float[:, :, :-2] * player_stacks_float[:, :, 1:-1] * player_stacks_float[:, :, 2:]).sum(dim=(1, 2))
        # 4-in-row
        h4 = (player_stacks_float[:, :, :-3] * player_stacks_float[:, :, 1:-2] * player_stacks_float[:, :, 2:-1] * player_stacks_float[:, :, 3:]).sum(dim=(1, 2))

        # VERTICAL patterns: check consecutive rows
        v2 = (player_stacks_float[:, :-1, :] * player_stacks_float[:, 1:, :]).sum(dim=(1, 2))
        v3 = (player_stacks_float[:, :-2, :] * player_stacks_float[:, 1:-1, :] * player_stacks_float[:, 2:, :]).sum(dim=(1, 2))
        v4 = (player_stacks_float[:, :-3, :] * player_stacks_float[:, 1:-2, :] * player_stacks_float[:, 2:-1, :] * player_stacks_float[:, 3:, :]).sum(dim=(1, 2))

        # DIAGONAL patterns (down-right): check (y,x), (y+1,x+1), etc.
        d1_2 = (player_stacks_float[:, :-1, :-1] * player_stacks_float[:, 1:, 1:]).sum(dim=(1, 2))
        d1_3 = (player_stacks_float[:, :-2, :-2] * player_stacks_float[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, 2:]).sum(dim=(1, 2))
        d1_4 = (player_stacks_float[:, :-3, :-3] * player_stacks_float[:, 1:-2, 1:-2] * player_stacks_float[:, 2:-1, 2:-1] * player_stacks_float[:, 3:, 3:]).sum(dim=(1, 2))

        # ANTI-DIAGONAL patterns (down-left): check (y,x), (y+1,x-1), etc.
        d2_2 = (player_stacks_float[:, :-1, 1:] * player_stacks_float[:, 1:, :-1]).sum(dim=(1, 2))
        d2_3 = (player_stacks_float[:, :-2, 2:] * player_stacks_float[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, :-2]).sum(dim=(1, 2))
        d2_4 = (player_stacks_float[:, :-3, 3:] * player_stacks_float[:, 1:-2, 2:-1] * player_stacks_float[:, 2:-1, 1:-2] * player_stacks_float[:, 3:, :-3]).sum(dim=(1, 2))

        # Sum across all directions
        two_in_row = h2 + v2 + d1_2 + d2_2
        three_in_row = h3 + v3 + d1_3 + d2_3
        four_in_row = h4 + v4 + d1_4 + d2_4

        # Connected neighbors: count of adjacent pairs (same as two_in_row)
        connected_neighbors = two_in_row

        # Gap potential: simplified - check patterns like [stack, empty, stack]
        # Horizontal gaps: stack at x, empty at x+1, stack at x+2
        empty_cells = (state.stack_owner == 0).float()
        h_gap = (player_stacks_float[:, :, :-2] * empty_cells[:, :, 1:-1] * player_stacks_float[:, :, 2:]).sum(dim=(1, 2))
        v_gap = (player_stacks_float[:, :-2, :] * empty_cells[:, 1:-1, :] * player_stacks_float[:, 2:, :]).sum(dim=(1, 2))
        d1_gap = (player_stacks_float[:, :-2, :-2] * empty_cells[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, 2:]).sum(dim=(1, 2))
        d2_gap = (player_stacks_float[:, :-2, 2:] * empty_cells[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, :-2]).sum(dim=(1, 2))
        gap_potential = (h_gap + v_gap + d1_gap + d2_gap) * 0.5

        # === OPPONENT THREAT METRICS (VECTORIZED) ===
        opponent_threat = torch.zeros(batch_size, device=device)
        opponent_victory_threat = torch.zeros(batch_size, device=device)
        blocking_score = torch.zeros(batch_size, device=device)

        for opponent in range(1, num_players + 1):
            if opponent == p:
                continue

            opp_stacks = (state.stack_owner == opponent).float()
            opp_territory = state.territory_count[:, opponent].float()
            opp_eliminated = state.eliminated_rings[:, opponent].float()

            # Victory proximity threat
            opp_territory_progress = opp_territory / territory_victory_minimum
            opp_elim_progress = opp_eliminated / ring_victory_threshold
            opponent_victory_threat += torch.max(opp_territory_progress, opp_elim_progress)

            # Vectorized line threat detection (same as line potential but for opponent)
            # HORIZONTAL opponent lines
            opp_h2 = (opp_stacks[:, :, :-1] * opp_stacks[:, :, 1:]).sum(dim=(1, 2))
            opp_h3 = (opp_stacks[:, :, :-2] * opp_stacks[:, :, 1:-1] * opp_stacks[:, :, 2:]).sum(dim=(1, 2))
            opp_h4 = (opp_stacks[:, :, :-3] * opp_stacks[:, :, 1:-2] * opp_stacks[:, :, 2:-1] * opp_stacks[:, :, 3:]).sum(dim=(1, 2))

            # VERTICAL opponent lines
            opp_v2 = (opp_stacks[:, :-1, :] * opp_stacks[:, 1:, :]).sum(dim=(1, 2))
            opp_v3 = (opp_stacks[:, :-2, :] * opp_stacks[:, 1:-1, :] * opp_stacks[:, 2:, :]).sum(dim=(1, 2))
            opp_v4 = (opp_stacks[:, :-3, :] * opp_stacks[:, 1:-2, :] * opp_stacks[:, 2:-1, :] * opp_stacks[:, 3:, :]).sum(dim=(1, 2))

            # DIAGONAL opponent lines (down-right)
            opp_d1_2 = (opp_stacks[:, :-1, :-1] * opp_stacks[:, 1:, 1:]).sum(dim=(1, 2))
            opp_d1_3 = (opp_stacks[:, :-2, :-2] * opp_stacks[:, 1:-1, 1:-1] * opp_stacks[:, 2:, 2:]).sum(dim=(1, 2))
            opp_d1_4 = (opp_stacks[:, :-3, :-3] * opp_stacks[:, 1:-2, 1:-2] * opp_stacks[:, 2:-1, 2:-1] * opp_stacks[:, 3:, 3:]).sum(dim=(1, 2))

            # ANTI-DIAGONAL opponent lines (down-left)
            opp_d2_2 = (opp_stacks[:, :-1, 1:] * opp_stacks[:, 1:, :-1]).sum(dim=(1, 2))
            opp_d2_3 = (opp_stacks[:, :-2, 2:] * opp_stacks[:, 1:-1, 1:-1] * opp_stacks[:, 2:, :-2]).sum(dim=(1, 2))
            opp_d2_4 = (opp_stacks[:, :-3, 3:] * opp_stacks[:, 1:-2, 2:-1] * opp_stacks[:, 2:-1, 1:-2] * opp_stacks[:, 3:, :-3]).sum(dim=(1, 2))

            # Weight threats by line length (longer = more dangerous)
            opp_two = opp_h2 + opp_v2 + opp_d1_2 + opp_d2_2
            opp_three = opp_h3 + opp_v3 + opp_d1_3 + opp_d2_3
            opp_four = opp_h4 + opp_v4 + opp_d1_4 + opp_d2_4
            opponent_threat += opp_two * 1.0 + opp_three * 1.5 + opp_four * 2.0

            # Blocking score: count our stacks adjacent to opponent stacks
            # Horizontal blocking
            block_h = (player_stacks_float[:, :, :-1] * opp_stacks[:, :, 1:]).sum(dim=(1, 2))
            block_h += (player_stacks_float[:, :, 1:] * opp_stacks[:, :, :-1]).sum(dim=(1, 2))
            # Vertical blocking
            block_v = (player_stacks_float[:, :-1, :] * opp_stacks[:, 1:, :]).sum(dim=(1, 2))
            block_v += (player_stacks_float[:, 1:, :] * opp_stacks[:, :-1, :]).sum(dim=(1, 2))
            blocking_score += block_h + block_v

        # === VULNERABILITY METRICS (VECTORIZED) ===
        # Check how many of our stacks could be captured by taller adjacent opponent stacks
        # Create padded versions for neighbor checking (pad with 0s)
        heights = state.stack_height.float()
        owners = state.stack_owner

        # Opponent mask: owned by non-zero and not by player p
        opponent_mask = (owners != 0) & (owners != p)
        opponent_heights = heights * opponent_mask.float()

        # For each of 4 directions, check if opponent neighbor is >= our height
        # Pad player_stacks and opponent_heights to handle boundary
        ps_pad = torch.nn.functional.pad(player_stacks_float, (1, 1, 1, 1), value=0)
        oh_pad = torch.nn.functional.pad(opponent_heights, (1, 1, 1, 1), value=0)
        h_pad = torch.nn.functional.pad(heights, (1, 1, 1, 1), value=0)
        om_pad = torch.nn.functional.pad(opponent_mask.float(), (1, 1, 1, 1), value=0)
        own_pad = torch.nn.functional.pad((owners != 0).float(), (1, 1, 1, 1), value=0)

        # Our stack positions (in padded coords, offset by 1)
        # Check each direction: up(-1,0), down(+1,0), left(0,-1), right(0,+1)
        # For vulnerability: opponent neighbor height >= our height at (y,x)
        vuln_up = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, :-2, 1:-1] *
                   (oh_pad[:, :-2, 1:-1] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vuln_down = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 2:, 1:-1] *
                     (oh_pad[:, 2:, 1:-1] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vuln_left = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, :-2] *
                     (oh_pad[:, 1:-1, :-2] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vuln_right = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, 2:] *
                      (oh_pad[:, 1:-1, 2:] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vulnerability = vuln_up + vuln_down + vuln_left + vuln_right

        # Blocked stacks: count adjacent occupied cells per stack, blocked if >= 3
        adj_up = ps_pad[:, 1:-1, 1:-1] * own_pad[:, :-2, 1:-1]
        adj_down = ps_pad[:, 1:-1, 1:-1] * own_pad[:, 2:, 1:-1]
        adj_left = ps_pad[:, 1:-1, 1:-1] * own_pad[:, 1:-1, :-2]
        adj_right = ps_pad[:, 1:-1, 1:-1] * own_pad[:, 1:-1, 2:]
        adj_count = adj_up + adj_down + adj_left + adj_right
        blocked_stacks = (adj_count >= 3).float().sum(dim=(1, 2))

        # === OVERTAKE POTENTIAL (VECTORIZED) ===
        # Count opponent stacks we could capture (our taller stacks adjacent to shorter opponent)
        # Our height > opponent neighbor height
        over_up = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, :-2, 1:-1] *
                   (h_pad[:, 1:-1, 1:-1] > oh_pad[:, :-2, 1:-1]).float()).sum(dim=(1, 2))
        over_down = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 2:, 1:-1] *
                     (h_pad[:, 1:-1, 1:-1] > oh_pad[:, 2:, 1:-1]).float()).sum(dim=(1, 2))
        over_left = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, :-2] *
                     (h_pad[:, 1:-1, 1:-1] > oh_pad[:, 1:-1, :-2]).float()).sum(dim=(1, 2))
        over_right = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, 2:] *
                      (h_pad[:, 1:-1, 1:-1] > oh_pad[:, 1:-1, 2:]).float()).sum(dim=(1, 2))
        overtake_potential = over_up + over_down + over_left + over_right

        # === TERRITORY METRICS (VECTORIZED) ===
        # Territory closure: territory cells adjacent to our stacks
        player_territory = (state.territory_owner == p).float()
        pt_pad = torch.nn.functional.pad(player_territory, (1, 1, 1, 1), value=0)

        # For each territory cell, check if any adjacent cell has our stack
        # Neighbor our-stack adjacency
        terr_adj_up = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, :-2, 1:-1]
        terr_adj_down = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, 2:, 1:-1]
        terr_adj_left = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, 1:-1, :-2]
        terr_adj_right = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, 1:-1, 2:]
        territory_adj = terr_adj_up + terr_adj_down + terr_adj_left + terr_adj_right
        territory_closure = territory_adj.sum(dim=(1, 2)) * 0.5
        territory_safety = territory_closure  # Same metric for now

        # === STACK MOBILITY ===
        # Per-stack movement freedom (simplified)
        stack_mobility = stack_count * 3.0  # Avg 3 directions per stack

        # === VICTORY PROXIMITY ===
        # How close to winning (normalized 0-1)
        territory_progress = territory / territory_victory_minimum
        elim_progress = eliminated_rings / ring_victory_threshold
        victory_proximity = torch.max(territory_progress, elim_progress)

        # === FORCED ELIMINATION RISK ===
        # Risk of being forced to eliminate (few stacks, surrounded)
        forced_elim_risk = torch.where(
            stack_count <= 2,
            vulnerability * 2.0,
            vulnerability * 0.5
        )

        # === LPS ACTION ADVANTAGE ===
        # Bonus for having more moves available than opponents
        lps_advantage = mobility / (mobility.sum() / num_players + 1e-6)

        # === MULTI-LEADER THREAT ===
        # Multiple opponents ahead
        multi_leader = opponent_victory_threat / (num_players - 1 + 1e-6)

        # === RECOVERY METRICS ===
        # Per RR-CANON-R110: eligible iff controls no stacks, has a marker,
        # and has at least one buried ring. Rings in hand do not affect eligibility.
        has_buried = buried_rings > 0
        no_controlled = stack_count == 0
        has_markers = marker_count > 0
        recovery_eligible = has_buried & no_controlled & has_markers
        recovery_potential = buried_rings * recovery_eligible.float()

        # === PENALTY/BONUS FLAGS (SYMMETRIC) ===
        # v2.0: Made symmetric to match CPU heuristic evaluation.
        # CPU computes (my_diversity - opp_diversity), so GPU must do the same.
        # This ensures parity and avoids divergence at game start when all players
        # have 0 stacks (penalties should cancel out to 0, not apply absolutely).

        # Compute total opponent stack count
        opponent_stack_count = torch.zeros(batch_size, device=device)
        for opp in range(1, num_players + 1):
            if opp != p:
                opp_stacks = (state.stack_owner == opp).sum(dim=(1, 2)).float()
                opponent_stack_count += opp_stacks
        # Average opponent stack count for fair comparison
        opponent_stack_count = opponent_stack_count / max(1, num_players - 1)

        # Compute diversity scores (mirroring CPU diversification_score function)
        # CPU: if stacks == 0: return -penalty; elif stacks == 1: return -single_penalty; else: return stacks * bonus
        no_stacks_penalty = get_weight("WEIGHT_NO_STACKS_PENALTY", None, 51.02)
        single_stack_penalty = get_weight("WEIGHT_SINGLE_STACK_PENALTY", None, 10.53)
        diversity_bonus = get_weight("WEIGHT_STACK_DIVERSITY_BONUS", None, -0.74)

        # My diversity score
        my_diversity = torch.where(
            stack_count == 0,
            -torch.full((batch_size,), no_stacks_penalty, device=device),
            torch.where(
                stack_count == 1,
                -torch.full((batch_size,), single_stack_penalty, device=device),
                stack_count * diversity_bonus
            )
        )

        # Opponent diversity score (using average opponent stack count)
        opp_diversity = torch.where(
            opponent_stack_count < 0.5,  # Effectively 0 stacks
            -torch.full((batch_size,), no_stacks_penalty, device=device),
            torch.where(
                opponent_stack_count < 1.5,  # Effectively 1 stack
                -torch.full((batch_size,), single_stack_penalty, device=device),
                opponent_stack_count * diversity_bonus
            )
        )

        # Relative diversity advantage (symmetric like CPU)
        diversity_advantage = my_diversity - opp_diversity

        near_victory = (victory_proximity > 0.8).float()
        # Note: stack_diversity bonus now computed as part of diversity_advantage

        # === COMPUTE OPPONENT METRICS FOR SYMMETRIC EVALUATION ===
        # v2.0: CPU heuristic computes (my_value - opponent_value) for most features.
        # To achieve parity, we compute average/max opponent values and use relative differences.
        opp_stack_count = torch.zeros(batch_size, device=device)
        opp_ring_count = torch.zeros(batch_size, device=device)
        opp_cap_height = torch.zeros(batch_size, device=device)
        opp_territory = torch.zeros(batch_size, device=device)
        max_opp_rings_in_hand = torch.zeros(batch_size, device=device)
        opp_center_control = torch.zeros(batch_size, device=device)
        opp_eliminated_rings = torch.zeros(batch_size, device=device)

        for opp in range(1, num_players + 1):
            if opp != p:
                opp_stacks_mask = (state.stack_owner == opp)
                opp_stack_count += opp_stacks_mask.sum(dim=(1, 2)).float()
                opp_heights = state.stack_height * opp_stacks_mask.int()
                opp_ring_count += opp_heights.sum(dim=(1, 2)).float()
                opp_cap_height += opp_ring_count  # Same as ring count for now
                opp_territory += state.territory_count[:, opp].float()
                max_opp_rings_in_hand = torch.max(max_opp_rings_in_hand, state.rings_in_hand[:, opp].float())
                opp_center_control += (center_bonus.unsqueeze(0) * opp_stacks_mask.float()).sum(dim=(1, 2))
                opp_eliminated_rings += state.eliminated_rings[:, opp].float()

        # Average opponent values (for symmetric comparison)
        num_opps = max(1, num_players - 1)
        opp_stack_count_avg = opp_stack_count / num_opps
        opp_ring_count_avg = opp_ring_count / num_opps
        opp_cap_height_avg = opp_cap_height / num_opps
        opp_territory_avg = opp_territory / num_opps
        opp_center_control_avg = opp_center_control / num_opps
        opp_eliminated_rings_avg = opp_eliminated_rings / num_opps

        # === COMBINE ALL COMPONENTS (SYMMETRIC) ===
        # v2.0: Use relative differences like CPU heuristic for parity.
        # CPU: score += (my_value - opponent_value) * weight
        score = torch.zeros(batch_size, device=device)
        score += (stack_count - opp_stack_count_avg) * get_weight("WEIGHT_STACK_CONTROL", "material_weight", 9.39)
        score += (total_ring_count - opp_ring_count_avg) * get_weight("WEIGHT_STACK_HEIGHT", "ring_count_weight", 6.81)
        score += (cap_height - opp_cap_height_avg) * get_weight("WEIGHT_CAP_HEIGHT", None, 4.82)
        score += (territory - opp_territory_avg) * get_weight("WEIGHT_TERRITORY", "territory_weight", 8.66)
        score += (rings_in_hand - max_opp_rings_in_hand) * get_weight("WEIGHT_RINGS_IN_HAND", None, 5.17)
        score += (center_control - opp_center_control_avg) * get_weight("WEIGHT_CENTER_CONTROL", "center_control_weight", 2.28)
        score += adjacency_score * get_weight("WEIGHT_ADJACENCY", None, 1.57)  # Keep absolute (CPU doesn't compare)

        # Threat/defense weights
        score -= opponent_threat * get_weight("WEIGHT_OPPONENT_THREAT", None, 6.11)
        score += mobility * get_weight("WEIGHT_MOBILITY", "mobility_weight", 5.31)
        # Eliminated rings: relative advantage over opponents
        score += (eliminated_rings - opp_eliminated_rings_avg) * get_weight("WEIGHT_ELIMINATED_RINGS", None, 13.12)
        score -= vulnerability * get_weight("WEIGHT_VULNERABILITY", None, 9.32)

        # Line/victory weights
        line_potential = (
            two_in_row * get_weight("WEIGHT_TWO_IN_ROW", None, 4.25) +
            three_in_row * get_weight("WEIGHT_THREE_IN_ROW", None, 2.13) +
            four_in_row * get_weight("WEIGHT_FOUR_IN_ROW", None, 4.36)
        )
        score += line_potential * get_weight("WEIGHT_LINE_POTENTIAL", "line_potential_weight", 7.24)
        score += victory_proximity * get_weight("WEIGHT_VICTORY_PROXIMITY", None, 20.94)
        score += connected_neighbors * get_weight("WEIGHT_CONNECTED_NEIGHBOR", None, 2.21)
        score += gap_potential * get_weight("WEIGHT_GAP_POTENTIAL", None, 0.03)

        # Advanced weights
        score += marker_count * get_weight("WEIGHT_MARKER_COUNT", None, 3.76)
        score += overtake_potential * get_weight("WEIGHT_OVERTAKE_POTENTIAL", None, 5.96)
        score += territory_closure * get_weight("WEIGHT_TERRITORY_CLOSURE", None, 11.56)
        score += connected_neighbors * get_weight("WEIGHT_LINE_CONNECTIVITY", None, 5.65)
        score += territory_safety * get_weight("WEIGHT_TERRITORY_SAFETY", None, 2.83)
        score += stack_mobility * get_weight("WEIGHT_STACK_MOBILITY", None, 1.11)

        # Opponent threat weights
        score -= opponent_victory_threat * get_weight("WEIGHT_OPPONENT_VICTORY_THREAT", None, 5.21)
        score -= forced_elim_risk * get_weight("WEIGHT_FORCED_ELIMINATION_RISK", None, 2.89)
        score += lps_advantage * get_weight("WEIGHT_LPS_ACTION_ADVANTAGE", None, 0.99)
        score -= multi_leader * get_weight("WEIGHT_MULTI_LEADER_THREAT", None, 1.03)

        # Penalty/bonus weights (v2.0: symmetric diversity scoring)
        # Use relative diversity advantage instead of absolute penalties
        # This matches CPU: score += (my_diversity - opp_diversity)
        score += diversity_advantage
        score -= blocked_stacks * get_weight("WEIGHT_BLOCKED_STACK_PENALTY", None, 4.57)
        score += near_victory * get_weight("WEIGHT_VICTORY_THRESHOLD_BONUS", None, 998.52)

        # Defensive bonus (backward compat)
        score += blocking_score * get_weight("defensive_weight", None, 0.3)

        # Recovery weights (v1.5)
        score += recovery_potential * get_weight("WEIGHT_RECOVERY_POTENTIAL", None, 6.0)
        score += recovery_eligible.float() * get_weight("WEIGHT_RECOVERY_ELIGIBILITY", None, 8.0)
        score += buried_rings * get_weight("WEIGHT_BURIED_RING_VALUE", None, 3.0)

        # === ELIMINATION PENALTY ===
        # Player with no stacks, no rings in hand, and no buried rings is eliminated
        has_material = (stack_count > 0) | (rings_in_hand > 0) | (buried_rings > 0)
        score = torch.where(
            ~has_material,
            score - 10000.0,  # Massive penalty for permanent elimination
            score
        )

        scores[:, p] = score

    return scores


__all__ = [
    'evaluate_positions_batch',
]
