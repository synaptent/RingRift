"""Board encoding helpers for GPU parallel game simulation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from app.models import BoardType

logger = logging.getLogger("app.ai.gpu_parallel_games")


class GPUBoardEncodingMixin:
    """Feature extraction helpers shared by ParallelGameRunner."""

    def _extract_features_batched(
        self,
        game_indices: torch.Tensor,
        board_type: BoardType,
        feature_dim: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Extract NNUE features for multiple games using vectorized operations."""
        del board_type
        try:
            num_games = len(game_indices)
            board_size = self.board_size
            num_positions = board_size * board_size
            features = torch.zeros(
                (num_games, feature_dim),
                dtype=torch.float32,
                device=device,
            )

            current_player = self.state.current_player[game_indices]
            current_player = torch.where(
                current_player < 1,
                torch.ones_like(current_player),
                current_player,
            )

            stack_owner = self.state.stack_owner[game_indices]
            stack_height = self.state.stack_height[game_indices]
            territory_owner = self.state.territory_owner[game_indices]

            y_coords = torch.arange(board_size, device=device).view(-1, 1).expand(
                board_size,
                board_size,
            )
            x_coords = torch.arange(board_size, device=device).view(1, -1).expand(
                board_size,
                board_size,
            )
            pos_indices = (y_coords * board_size + x_coords).flatten()
            current_player_np = current_player.cpu().numpy()

            for local_index in range(num_games):
                current = int(current_player_np[local_index])
                owner_slice = stack_owner[local_index].flatten()
                height_slice = stack_height[local_index].flatten()

                occupied = (owner_slice > 0) & (height_slice > 0)
                occupied_idx = torch.where(occupied)[0]
                if len(occupied_idx) > 0:
                    owners = owner_slice[occupied_idx]
                    heights = height_slice[occupied_idx]
                    positions = pos_indices[occupied_idx]
                    plane_offsets = ((owners - current) % self.num_players).long()

                    ring_indices = plane_offsets * num_positions + positions
                    valid_ring = ring_indices < feature_dim
                    features[local_index].scatter_(
                        0,
                        ring_indices[valid_ring],
                        torch.ones_like(
                            ring_indices[valid_ring],
                            dtype=torch.float32,
                        ),
                    )

                    stack_indices = (4 + plane_offsets) * num_positions + positions
                    valid_stack = stack_indices < feature_dim
                    heights_scaled = torch.clamp(heights.float() / 5.0, 0.0, 1.0)
                    features[local_index].scatter_(
                        0,
                        stack_indices[valid_stack],
                        heights_scaled[valid_stack],
                    )

                territory_slice = territory_owner[local_index].flatten()
                territory_occupied = territory_slice > 0
                territory_idx = torch.where(territory_occupied)[0]
                if len(territory_idx) > 0:
                    territory_owners = territory_slice[territory_idx]
                    territory_positions = pos_indices[territory_idx]
                    territory_offsets = (
                        (territory_owners - current) % self.num_players
                    ).long()
                    territory_plane_indices = (
                        (8 + territory_offsets) * num_positions + territory_positions
                    )
                    valid_territory = territory_plane_indices < feature_dim
                    features[local_index].scatter_(
                        0,
                        territory_plane_indices[valid_territory],
                        torch.ones_like(
                            territory_plane_indices[valid_territory],
                            dtype=torch.float32,
                        ),
                    )

            return features
        except Exception as exc:
            logger.debug("Batched feature extraction failed: %s", exc)
            return None

    def _extract_features_for_game(
        self,
        game_idx: int,
        board_type: BoardType,
    ) -> np.ndarray | None:
        """Extract NNUE features from batch state for a single game."""
        try:
            from .nnue import get_feature_dim

            feature_dim = get_feature_dim(board_type)
            features = np.zeros(feature_dim, dtype=np.float32)
            board_size = self.board_size
            num_positions = board_size * board_size

            current_player = int(self.state.current_player[game_idx].cpu().numpy())
            if current_player < 1:
                current_player = 1

            stack_owner_np = self.state.stack_owner[game_idx].cpu().numpy()
            stack_height_np = self.state.stack_height[game_idx].cpu().numpy()
            territory_owner_np = self.state.territory_owner[game_idx].cpu().numpy()

            for row in range(board_size):
                for col in range(board_size):
                    pos_idx = row * board_size + col
                    owner = int(stack_owner_np[row, col])
                    height = int(stack_height_np[row, col])
                    if owner > 0 and height > 0:
                        plane_offset = (owner - current_player) % self.num_players
                        ring_plane = plane_offset * num_positions + pos_idx
                        stack_plane = (4 + plane_offset) * num_positions + pos_idx
                        if ring_plane < feature_dim:
                            features[ring_plane] = 1.0
                        if stack_plane < feature_dim:
                            features[stack_plane] = min(float(height) / 5.0, 1.0)

                    territory_owner = int(territory_owner_np[row, col])
                    if territory_owner > 0:
                        plane_offset = (
                            territory_owner - current_player
                        ) % self.num_players
                        territory_plane = (8 + plane_offset) * num_positions + pos_idx
                        if territory_plane < feature_dim:
                            features[territory_plane] = 1.0

            return features
        except Exception as exc:
            logger.debug("Feature extraction failed for game %s: %s", game_idx, exc)
            return None
