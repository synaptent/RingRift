"""Loss functions for neural network training.

This module contains loss functions used for training RingRift neural networks,
extracted from neural_net.py for better modularity.

December 2025: Extracted from neural_net.py as part of N7 refactoring.
"""

from __future__ import annotations

import logging
import warnings

import torch

logger = logging.getLogger(__name__)

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

    Follows Section 8 of docs/rules/COMPACT_RULES.md:
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

    Raises
    ------
    ValueError
        If inputs contain NaN values or targets are severely denormalized.
    """
    # Phase 3 Sanity Checks: Validate inputs
    if torch.any(torch.isnan(policy_log_probs)):
        raise ValueError(
            f"NaN detected in policy_log_probs! "
            f"Count: {torch.isnan(policy_log_probs).sum().item()}"
        )
    if torch.any(torch.isnan(policy_targets)):
        raise ValueError(
            f"NaN detected in policy_targets! "
            f"Count: {torch.isnan(policy_targets).sum().item()}"
        )

    target_sums = policy_targets.sum(dim=1)
    valid_mask = target_sums > 0
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=policy_log_probs.device)

    targets = policy_targets[valid_mask]
    log_probs = policy_log_probs[valid_mask]

    # Phase 3 Sanity Check: targets should sum to ~1.0
    valid_target_sums = targets.sum(dim=1)
    expected_ones = torch.ones(targets.size(0), device=targets.device)
    if not torch.allclose(valid_target_sums, expected_ones, atol=1e-4):
        min_sum = valid_target_sums.min().item()
        max_sum = valid_target_sums.max().item()
        if min_sum < 0.5 or max_sum > 1.5:
            raise ValueError(
                f"Policy targets severely denormalized in loss computation. "
                f"Sums range: [{min_sum:.4f}, {max_sum:.4f}]"
            )
        else:
            warnings.warn(
                f"Policy targets slightly denormalized. "
                f"Sums range: [{min_sum:.6f}, {max_sum:.6f}]"
            )

    log_targets = torch.log(targets.clamp_min(1e-12))
    loss_terms = torch.where(
        targets > 0,
        targets * (log_targets - log_probs),
        torch.zeros_like(targets),
    )

    per_sample_loss = loss_terms.sum(dim=1)

    # Phase 3 Sanity Check: loss should be reasonable
    # For a policy with ~4500 actions, max reasonable loss is -log(1e-43) ≈ 100
    # Loss > 1000 indicates severe numerical issues
    max_reasonable_loss = 1000.0
    max_loss = per_sample_loss.max().item()
    if max_loss > max_reasonable_loss:
        # Log detailed diagnostics for debugging
        worst_idx = per_sample_loss.argmax()
        worst_target = targets[worst_idx]
        worst_log_prob = log_probs[worst_idx]
        target_positions = (worst_target > 0).nonzero(as_tuple=True)[0]

        warnings.warn(
            f"Extreme policy loss detected: max={max_loss:.2e}. "
            f"This indicates model outputs extremely negative logits at target positions. "
            f"Target positions: {target_positions[:5].tolist()}..., "
            f"Log probs at targets: {worst_log_prob[target_positions[:3]].tolist()}"
        )

    return per_sample_loss.mean()


def masked_log_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    fill_value: float = -float("inf"),
) -> torch.Tensor:
    """Compute log_softmax only over valid positions for spatial policy heads.

    Spatial policy heads (V3/V4 architectures) scatter logits into a flat policy
    vector, initializing invalid positions to large negative values (-1e4 or -1e9).
    Standard log_softmax applied to these vectors causes numerical instability:
    the softmax denominator is dominated by masked entries, attenuating valid
    action log-probabilities.

    This function computes softmax normalization only over valid action positions,
    avoiding the instability.

    Parameters
    ----------
    logits : torch.Tensor
        Raw policy logits of shape (batch, policy_size).
    valid_mask : torch.Tensor
        Boolean mask of shape (batch, policy_size) or (policy_size,).
        True indicates valid action positions.
    fill_value : float
        Value to fill for invalid positions in output. Default -inf produces
        probability 0 when exp() is applied.

    Returns
    -------
    torch.Tensor
        Log probabilities of shape (batch, policy_size).
        Invalid positions are filled with fill_value.

    Notes
    -----
    The standard log_softmax formula is:
        log_softmax(x)_i = x_i - log(sum_j exp(x_j))

    Masked version:
        log_softmax(x)_i = x_i - log(sum_{j in valid} exp(x_j))  for i in valid
                         = fill_value                             otherwise

    Example
    -------
    >>> logits = torch.tensor([[1.0, 2.0, -1e4, -1e4]])
    >>> valid_mask = torch.tensor([[True, True, False, False]])
    >>> log_probs = masked_log_softmax(logits, valid_mask)
    >>> torch.exp(log_probs[0, :2]).sum()  # Should be ~1.0
    tensor(1.0000)
    """
    # Expand mask to batch dimension if needed
    if valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0).expand(logits.size(0), -1)

    # Ensure mask is boolean
    valid_mask = valid_mask.bool()

    # Create output tensor filled with fill_value
    output = torch.full_like(logits, fill_value)

    # Set invalid positions to -inf before softmax so they don't contribute to denominator
    masked_logits = logits.masked_fill(~valid_mask, -float("inf"))

    # Compute log_softmax - now only valid positions contribute to denominator
    # The -inf positions contribute 0 to sum(exp(x)), so they're excluded
    log_probs = torch.log_softmax(masked_logits, dim=1)

    # Copy valid positions to output (invalid remain at fill_value)
    output = torch.where(valid_mask, log_probs, output)

    return output


def uses_spatial_policy_head(model) -> bool:
    """Check if a model uses spatial policy heads with masking.

    Spatial policy models (V3/V4) initialize invalid positions to large
    negative values via _scatter_policy_logits(). These require masked
    log_softmax to avoid numerical instability during training.

    Parameters
    ----------
    model : nn.Module
        The neural network model to check.

    Returns
    -------
    bool
        True if the model uses spatial policy heads requiring masking.
    """
    model_name = model.__class__.__name__

    # Models with spatial policy heads
    spatial_models = {
        "RingRiftCNN_v3",
        "RingRiftCNN_v3_Lite",
        "RingRiftCNN_v4",
        "HexNeuralNet_v3",
        "HexNeuralNet_v3_Lite",
        "HexNeuralNet_v4",
    }

    # Explicit flat policy variants (V3 backbone with flat heads)
    flat_variants = {
        "RingRiftCNN_v3_Flat",
        "HexNeuralNet_v3_Flat",
    }

    return model_name in spatial_models and model_name not in flat_variants


def detect_masked_policy_output(policy_pred: torch.Tensor, threshold: float = -1e3) -> bool:
    """Detect if policy prediction contains masked positions from spatial policy heads.

    This function checks if the policy logits contain large negative values
    characteristic of spatial policy head masking (-1e4 or -1e9).

    Parameters
    ----------
    policy_pred : torch.Tensor
        Policy logits of shape (batch, policy_size).
    threshold : float
        Values below this threshold are considered masked. Default -1e3.

    Returns
    -------
    bool
        True if masked positions are detected.
    """
    return bool((policy_pred.min() < threshold).item())


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


def validate_hex_policy_indices(
    policy_indices: torch.Tensor | "np.ndarray",
    board_size: int,
    hex_radius: int | None = None,
    sample_size: int = 1000,
) -> tuple[bool, list[str]]:
    """Validate that hex policy indices fall within valid hex cells.

    This validation gate catches encoding mismatches before training starts.
    It decodes a sample of policy indices and checks that the referenced
    positions are within the valid hex region (not in corners of bounding box).

    Parameters
    ----------
    policy_indices : torch.Tensor | np.ndarray
        Policy indices of shape (N, K) where K is the number of sparse entries.
    board_size : int
        Hex bounding box size (e.g., 9 for hex8, 25 for hexagonal).
    hex_radius : int | None
        Hex radius. If None, computed as (board_size - 1) // 2.
    sample_size : int
        Number of samples to validate. Set to -1 for all.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages). Empty list if valid.

    Example
    -------
    >>> import numpy as np
    >>> indices = np.array([[24], [4060], [100]])  # 4060 is invalid for hex8
    >>> valid, errors = validate_hex_policy_indices(indices, board_size=9)
    >>> valid
    False
    >>> errors[0]
    'Policy index 4060 references cell (7,8) outside hex radius 4'
    """
    import numpy as np

    if hex_radius is None:
        hex_radius = (board_size - 1) // 2

    # Layout constants (must match ActionEncoderHex and model architecture)
    num_ring_counts = 3
    num_directions = 6
    max_dist = board_size - 1
    placement_span = board_size * board_size * num_ring_counts
    movement_span = board_size * board_size * num_directions * max_dist
    special_base = placement_span + movement_span

    # Convert to numpy if tensor
    if hasattr(policy_indices, 'numpy'):
        policy_indices = policy_indices.cpu().numpy()

    # Sample if dataset is large
    n_samples = len(policy_indices)
    if sample_size > 0 and n_samples > sample_size:
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        policy_indices = policy_indices[sample_idx]

    center = board_size // 2
    errors: list[str] = []
    invalid_count = 0

    for sample_indices in policy_indices:
        for idx in sample_indices:
            idx = int(idx)
            if idx < 0:
                continue  # INVALID_MOVE_INDEX, skip

            # Decode the policy index to check validity
            if idx < placement_span:
                # Placement: idx = pos_idx * 3 + ring_offset
                pos_idx = idx // 3
                row = pos_idx // board_size
                col = pos_idx % board_size
            elif idx < special_base:
                # Movement: decode from_cell
                movement_idx = idx - placement_span
                from_cell = movement_idx // (num_directions * max_dist)
                row = from_cell // board_size
                col = from_cell % board_size
            else:
                # Special action (skip_placement), always valid
                continue

            # Check hex validity
            q = col - center
            r = row - center
            hex_dist = max(abs(q), abs(r), abs(q + r))

            if hex_dist > hex_radius:
                invalid_count += 1
                if len(errors) < 5:  # Limit error messages
                    errors.append(
                        f"Policy index {idx} references cell ({col},{row}) "
                        f"outside hex radius {hex_radius} (hex_dist={hex_dist})"
                    )

    if invalid_count > 0:
        errors.insert(0, f"Found {invalid_count} policy indices referencing invalid hex cells")

    return len(errors) == 0, errors


__all__ = [
    'MAX_PLAYERS',
    'build_rank_targets',
    'masked_policy_kl',
    'multi_player_value_loss',
    'rank_distribution_loss',
    'ranks_from_game_result',
    'validate_hex_policy_indices',
]
