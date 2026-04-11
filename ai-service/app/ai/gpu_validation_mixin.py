"""Validation helpers for GPU parallel game simulation."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .gpu_move_generation import BatchMoves

logger = logging.getLogger("app.ai.gpu_parallel_games")


class GPUValidationMixin:
    """Shadow/state validation helpers shared by ParallelGameRunner."""

    def get_validation_reports(self) -> dict[str, Any]:
        """Get validation reports from both shadow and state validators."""
        reports: dict[str, Any] = {}
        if self.shadow_validator:
            reports["shadow_validation"] = self.shadow_validator.get_report()
        if self.state_validator:
            reports["state_validation"] = self.state_validator.get_report()

        all_passed = True
        if (
            self.shadow_validator
            and self.shadow_validator.stats.divergence_rate
            > self.shadow_validator.threshold
        ):
            all_passed = False
        if (
            self.state_validator
            and self.state_validator.stats.divergence_rate
            > self.state_validator.threshold
        ):
            all_passed = False

        reports["combined_status"] = "PASS" if all_passed else "FAIL"
        return reports

    def reset_validation_stats(self) -> None:
        """Reset all validation statistics."""
        if self.shadow_validator:
            self.shadow_validator.reset_stats()
        if self.state_validator:
            self.state_validator.reset_stats()

    def _validate_placement_moves_sample(
        self,
        moves: BatchMoves,
        mask: torch.Tensor,
    ) -> None:
        """Shadow validate a sample of placement moves against CPU rules."""
        if self.shadow_validator is None:
            return

        game_indices = torch.where(mask)[0].tolist()
        move_offsets_np = moves.move_offsets.cpu().numpy()
        moves_per_game_np = moves.moves_per_game.cpu().numpy()
        from_y_np = moves.from_y.cpu().numpy()
        from_x_np = moves.from_x.cpu().numpy()
        current_player_np = self.state.current_player.cpu().numpy()

        for game_idx in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            move_start = int(move_offsets_np[game_idx])
            move_count = int(moves_per_game_np[game_idx])
            if move_count == 0:
                continue

            gpu_positions = []
            for offset in range(move_count):
                move_index = move_start + offset
                row = int(from_y_np[move_index])
                col = int(from_x_np[move_index])
                if self.board_type and self.board_type.lower() in (
                    "hexagonal",
                    "hex",
                ):
                    center = self.board_size // 2
                    x = col - center
                    y = row - center
                else:
                    x = col
                    y = row
                gpu_positions.append((x, y))

            cpu_state = self.state.to_game_state(game_idx)
            player = int(current_player_np[game_idx])
            self.shadow_validator.validate_placement_moves(
                gpu_positions,
                cpu_state,
                player,
            )

    def _validate_movement_moves_sample(
        self,
        movement_moves: BatchMoves,
        capture_moves: BatchMoves,
        mask: torch.Tensor,
    ) -> None:
        """Shadow validate a sample of movement/capture moves."""
        if self.shadow_validator is None:
            return

        game_indices = torch.where(mask)[0].tolist()
        mv_offsets_np = movement_moves.move_offsets.cpu().numpy()
        mv_counts_np = movement_moves.moves_per_game.cpu().numpy()
        mv_from_y_np = movement_moves.from_y.cpu().numpy()
        mv_from_x_np = movement_moves.from_x.cpu().numpy()
        mv_to_y_np = movement_moves.to_y.cpu().numpy()
        mv_to_x_np = movement_moves.to_x.cpu().numpy()
        cap_offsets_np = capture_moves.move_offsets.cpu().numpy()
        cap_counts_np = capture_moves.moves_per_game.cpu().numpy()
        cap_from_y_np = capture_moves.from_y.cpu().numpy()
        cap_from_x_np = capture_moves.from_x.cpu().numpy()
        cap_to_y_np = capture_moves.to_y.cpu().numpy()
        cap_to_x_np = capture_moves.to_x.cpu().numpy()
        current_player_np = self.state.current_player.cpu().numpy()

        is_hex = self.board_type and self.board_type.lower() in (
            "hexagonal",
            "hex",
            "hex8",
        )
        hex_center = self.board_size // 2 if is_hex else 0

        def to_cpu_coords(row: int, col: int) -> tuple[int, int]:
            if is_hex:
                return col - hex_center, row - hex_center
            return col, row

        for game_idx in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            move_start = int(mv_offsets_np[game_idx])
            move_count = int(mv_counts_np[game_idx])
            gpu_movement = []
            for offset in range(move_count):
                move_index = move_start + offset
                from_x, from_y = to_cpu_coords(
                    int(mv_from_y_np[move_index]),
                    int(mv_from_x_np[move_index]),
                )
                to_x, to_y = to_cpu_coords(
                    int(mv_to_y_np[move_index]),
                    int(mv_to_x_np[move_index]),
                )
                gpu_movement.append(((from_x, from_y), (to_x, to_y)))

            cap_start = int(cap_offsets_np[game_idx])
            cap_count = int(cap_counts_np[game_idx])
            gpu_captures = []
            for offset in range(cap_count):
                move_index = cap_start + offset
                from_x, from_y = to_cpu_coords(
                    int(cap_from_y_np[move_index]),
                    int(cap_from_x_np[move_index]),
                )
                to_x, to_y = to_cpu_coords(
                    int(cap_to_y_np[move_index]),
                    int(cap_to_x_np[move_index]),
                )
                gpu_captures.append(((from_x, from_y), (to_x, to_y)))

            cpu_state = self.state.to_game_state(game_idx)
            player = int(current_player_np[game_idx])
            if gpu_movement:
                self.shadow_validator.validate_movement_moves(
                    gpu_movement,
                    cpu_state,
                    player,
                )
            if gpu_captures:
                self.shadow_validator.validate_capture_moves(
                    gpu_captures,
                    cpu_state,
                    player,
                )

    def get_shadow_validation_report(self) -> dict[str, Any] | None:
        """Get shadow validation statistics if enabled."""
        if self.shadow_validator is None:
            return None
        return self.shadow_validator.get_report()
