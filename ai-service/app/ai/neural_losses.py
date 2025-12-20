"""Loss functions for neural network training.

This module contains loss functions used for training RingRift neural networks,
extracted from neural_net.py for better modularity.

December 2025: Extracted from neural_net.py as part of N7 refactoring.
"""

from __future__ import annotations

import torch

# Maximum number of players for multi-player value head
MAX_PLAYERS = 4


def multi_player_value_loss(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    num_players: int | torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss for multi-player value predictions.

    Only computes loss over active players (slots 0 to num_players-1).
    Inactive slots (for games with fewer than MAX_PLAYERS) are masked out.

    Parameters
    ----------
    pred_values : torch.Tensor
        Predicted values of shape (batch, MAX_PLAYERS).
    target_values : torch.Tensor
        Target values of shape (batch, MAX_PLAYERS).
    num_players : int | torch.Tensor
        Either a single integer active player count (2, 3, or 4), or a
        per-sample tensor of shape (batch,) with values in [1, MAX_PLAYERS].

    Returns
    -------
    torch.Tensor
        Scalar MSE loss averaged over active players.
    """
    if pred_values.shape != target_values.shape:
        raise ValueError(
            "multi_player_value_loss expects pred_values and target_values to "
            f"share the same shape; got pred_values={tuple(pred_values.shape)} "
            f"target_values={tuple(target_values.shape)}."
        )
    if pred_values.ndim != 2:
        raise ValueError(
            "multi_player_value_loss expects 2D tensors of shape "
            "(batch, max_players); got "
            f"pred_values.ndim={pred_values.ndim}."
        )

    batch_size, max_players = target_values.shape

    # Create mask for active players.
    if isinstance(num_players, int):
        n = int(num_players)
        if n < 1 or n > max_players:
            raise ValueError(
                f"num_players must be in [1, {max_players}], got {n}."
            )
        mask = torch.zeros_like(target_values)
        mask[:, :n] = 1.0
    else:
        num_players_tensor = num_players.to(
            device=target_values.device,
            dtype=torch.long,
        )
        if num_players_tensor.ndim == 0:
            n = None
            if n < 1 or n > max_players:
                raise ValueError(
                    f"num_players must be in [1, {max_players}], got {n}."
                )
            mask = torch.zeros_like(target_values)
            mask[:, :n] = 1.0
        elif num_players_tensor.ndim == 1:
            if int(num_players_tensor.shape[0]) != int(batch_size):
                raise ValueError(
                    "Per-sample num_players tensor must have shape (batch,), "
                    f"got {tuple(num_players_tensor.shape)} for batch_size={batch_size}."
                )
            if torch.any(num_players_tensor < 1) or torch.any(num_players_tensor > max_players):
                raise ValueError(
                    "Per-sample num_players tensor contains values outside "
                    f"[1, {max_players}]."
                )

            player_idx = torch.arange(
                max_players,
                device=target_values.device,
            ).unsqueeze(0)
            mask = (
                player_idx < num_players_tensor.unsqueeze(1)
            ).to(dtype=target_values.dtype)
        else:
            raise ValueError(
                "Per-sample num_players tensor must be a scalar or 1D tensor; "
                f"got ndim={num_players_tensor.ndim}."
            )

    # Compute masked MSE
    squared_errors = ((pred_values - target_values) ** 2) * mask
    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError(
            "multi_player_value_loss mask has zero active entries; check num_players."
        )
    loss = squared_errors.sum() / denom

    return loss


def rank_distribution_loss(
    pred_rank_dist: torch.Tensor,
    target_ranks: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """
    Cross-entropy loss for rank distribution predictions.

    For each player, computes the cross-entropy loss between the predicted
    rank probability distribution and the actual rank (one-hot encoded).

    Parameters
    ----------
    pred_rank_dist : torch.Tensor
        Predicted rank distributions of shape (batch, MAX_PLAYERS, MAX_PLAYERS).
        pred_rank_dist[b, p, r] = P(player p finishes at rank r).
        Must be probability distributions (sum to 1 over rank dimension).
    target_ranks : torch.Tensor
        Target rank indices of shape (batch, MAX_PLAYERS).
        target_ranks[b, p] = actual rank of player p (0 = 1st place, 1 = 2nd, etc.)
        Values should be in range [0, num_players-1] for active players.
        Inactive player slots (index >= num_players) are ignored.
    num_players : int
        Number of active players in the game (2, 3, or 4).

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss averaged over active players.

    Example
    -------
    For a 3-player game where player 0 won (rank 0), player 1 came 2nd (rank 1),
    and player 2 came last (rank 2):
        target_ranks = [0, 1, 2, -1]  # -1 for inactive player 4
        pred_rank_dist = [[0.8, 0.15, 0.05, 0],  # Player 0's rank dist
                          [0.1, 0.7, 0.2, 0],    # Player 1's rank dist
                          [0.1, 0.15, 0.75, 0],  # Player 2's rank dist
                          [0.25, 0.25, 0.25, 0.25]]  # Inactive, ignored
    """
    batch_size = pred_rank_dist.size(0)
    max_players = pred_rank_dist.size(1)

    # Clamp predictions to avoid log(0)
    pred_rank_dist = torch.clamp(pred_rank_dist, min=1e-8)

    # Create mask for active players
    player_mask = torch.zeros(batch_size, max_players, device=pred_rank_dist.device)
    player_mask[:, :num_players] = 1.0

    # Compute cross-entropy for each player
    # For each player p, we want: -log(pred_rank_dist[b, p, target_ranks[b, p]])
    # Use gather to select the predicted probability for the target rank
    target_ranks_clamped = target_ranks.clamp(0, num_players - 1)  # Clamp to valid range
    target_ranks_expanded = target_ranks_clamped.unsqueeze(-1)  # [B, P, 1]
    pred_at_target = pred_rank_dist.gather(dim=2, index=target_ranks_expanded).squeeze(-1)  # [B, P]

    # Negative log likelihood
    nll = -torch.log(pred_at_target)  # [B, P]

    # Apply mask and compute mean
    masked_nll = nll * player_mask
    loss = masked_nll.sum() / player_mask.sum()

    return loss


def ranks_from_game_result(
    winner: int,
    num_players: int,
    player_territories: list[int] | None = None,
    player_eliminated_rings: list[int] | None = None,
    player_markers_on_board: list[int] | None = None,
    elimination_order: list[int] | None = None,
) -> torch.Tensor:
    """
    Compute rank indices from game result using canonical ranking rules.

    Follows Section 8 of ringrift_compact_rules.md:
    1. Winner gets rank 0 (1st place)
    2. Remaining players ranked by: territory -> eliminated rings -> markers -> elimination order

    Parameters
    ----------
    winner : int
        Index of winning player (0-indexed).
    num_players : int
        Number of active players (2, 3, or 4).
    player_territories : Optional[list[int]]
        Territory count per player, or None if not available.
    player_eliminated_rings : Optional[list[int]]
        Total eliminated rings per player, or None.
    player_markers_on_board : Optional[list[int]]
        Markers remaining on board per player, or None.
    elimination_order : Optional[list[int]]
        Order of elimination (later = better), or None.

    Returns
    -------
    torch.Tensor
        Rank indices of shape (MAX_PLAYERS,) where ranks[p] = rank of player p.
        Inactive players get rank num_players (outside valid range).
    """
    ranks = torch.full((MAX_PLAYERS,), MAX_PLAYERS, dtype=torch.long)

    # Winner gets rank 0
    ranks[winner] = 0

    # For remaining players, assign ranks 1, 2, ...
    remaining = [p for p in range(num_players) if p != winner]

    if len(remaining) == 0:
        return ranks

    # Build scoring tuples for sorting (higher is better)
    def score(p: int) -> tuple:
        territory = player_territories[p] if player_territories else 0
        elim_rings = player_eliminated_rings[p] if player_eliminated_rings else 0
        markers = player_markers_on_board[p] if player_markers_on_board else 0
        elim_order = elimination_order.index(p) if elimination_order and p in elimination_order else -1
        return (territory, elim_rings, markers, elim_order)

    # Sort remaining players by score (descending)
    remaining_sorted = sorted(remaining, key=score, reverse=True)

    # Assign ranks
    for rank_idx, player in enumerate(remaining_sorted, start=1):
        ranks[player] = rank_idx

    return ranks


def masked_policy_kl(
    policy_log_probs: torch.Tensor,
    policy_targets: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence loss while ignoring samples with empty policy targets.

    This handles cases where some samples in a batch may have no valid policy
    targets (all zeros), such as terminal positions or positions where no moves
    were legal.

    Parameters
    ----------
    policy_log_probs : torch.Tensor
        Log probabilities of shape (batch, policy_size) from the model.
    policy_targets : torch.Tensor
        Target probability distributions of shape (batch, policy_size).
        Samples with all-zero targets are masked out.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence loss averaged over valid samples.
        Returns 0.0 if no valid samples exist.
    """
    target_sums = policy_targets.sum(dim=1)
    valid_mask = target_sums > 0
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=policy_log_probs.device)

    targets = policy_targets[valid_mask]
    log_probs = policy_log_probs[valid_mask]
    log_targets = torch.log(targets.clamp_min(1e-12))
    loss_terms = torch.where(
        targets > 0,
        targets * (log_targets - log_probs),
        torch.zeros_like(targets),
    )
    return loss_terms.sum(dim=1).mean()


def build_rank_targets(
    values_mp: torch.Tensor,
    num_players: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-player rank distributions from value vectors.

    Converts raw value predictions into rank probability distributions,
    handling ties by distributing probability mass evenly.

    Parameters
    ----------
    values_mp : torch.Tensor
        Value predictions of shape (batch, max_players).
    num_players : int | torch.Tensor
        Either a single integer for uniform player count, or a
        per-sample tensor of shape (batch,) with player counts.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        rank_targets: [B, P, P] distribution over ranks per player
        active_mask: [B, P] True for active player slots
    """
    batch_size, max_players = values_mp.shape
    if isinstance(num_players, int):
        num_players_tensor = torch.full(
            (batch_size,),
            int(num_players),
            device=values_mp.device,
            dtype=torch.long,
        )
    else:
        num_players_tensor = num_players.to(
            device=values_mp.device,
            dtype=torch.long,
        )
        if num_players_tensor.ndim == 0:
            num_players_tensor = num_players_tensor.repeat(batch_size)

    rank_targets = torch.zeros(
        (batch_size, max_players, max_players),
        device=values_mp.device,
        dtype=values_mp.dtype,
    )
    active_mask = torch.zeros(
        (batch_size, max_players),
        device=values_mp.device,
        dtype=torch.bool,
    )

    for b in range(batch_size):
        n = int(num_players_tensor[b].item())
        n = max(1, min(n, max_players))
        vals = values_mp[b, :n]
        active_mask[b, :n] = True
        for p in range(n):
            v = vals[p]
            higher = int((vals > v).sum().item())
            tie = int((vals == v).sum().item())
            if tie <= 0:
                continue
            start = higher
            end = higher + tie
            rank_targets[b, p, start:end] = 1.0 / float(tie)

    return rank_targets, active_mask


__all__ = [
    'MAX_PLAYERS',
    'build_rank_targets',
    'masked_policy_kl',
    'multi_player_value_loss',
    'rank_distribution_loss',
    'ranks_from_game_result',
]
