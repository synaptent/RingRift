"""Persona and heuristic-weight helpers for GPU parallel game simulation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("app.ai.gpu_parallel_games")


class GPUPersonaMixin:
    """Persona/weight helpers shared by ParallelGameRunner."""

    TRAINING_MATCHUPS: dict[str, list[str]] = {
        "aggressive_vs_defensive": ["aggressive", "defensive"],
        "territorial_vs_aggressive": ["territorial", "aggressive"],
        "balanced_vs_aggressive": ["balanced", "aggressive"],
        "balanced_vs_defensive": ["balanced", "defensive"],
        "balanced_vs_territorial": ["balanced", "territorial"],
        "defensive_vs_territorial": ["defensive", "territorial"],
        "aggressive_mirror": ["aggressive", "aggressive"],
        "defensive_mirror": ["defensive", "defensive"],
        "balanced_mirror": ["balanced", "balanced"],
        "territorial_mirror": ["territorial", "territorial"],
        "3p_balanced": ["balanced", "balanced", "balanced"],
        "3p_mixed": ["aggressive", "defensive", "territorial"],
        "3p_aggressive": ["aggressive", "aggressive", "aggressive"],
        "3p_defensive": ["defensive", "defensive", "defensive"],
        "3p_territorial": ["territorial", "territorial", "territorial"],
        "3p_agg_def_bal": ["aggressive", "defensive", "balanced"],
        "3p_ter_agg_bal": ["territorial", "aggressive", "balanced"],
        "4p_balanced": ["balanced", "balanced", "balanced", "balanced"],
        "4p_mixed": ["aggressive", "defensive", "territorial", "balanced"],
        "4p_aggressive": ["aggressive", "aggressive", "aggressive", "aggressive"],
        "4p_defensive": ["defensive", "defensive", "defensive", "defensive"],
        "4p_territorial": ["territorial", "territorial", "territorial", "territorial"],
        "4p_agg_vs_def": ["aggressive", "aggressive", "defensive", "defensive"],
        "4p_ter_vs_bal": ["territorial", "territorial", "balanced", "balanced"],
    }

    def _default_weights(self) -> dict[str, float]:
        """Load best heuristic weights for the current board configuration."""
        board_type = self.board_type or "square8"
        if self.board_size == 19:
            board_type = "square19"
        elif self.board_size == 8 and board_type not in ("square8", "hex8"):
            board_type = "square8"

        try:
            from app.training.cmaes_registry_integration import (
                load_heuristic_weights_from_registry,
            )

            for stage in ("production", "staging"):
                weights = load_heuristic_weights_from_registry(
                    board_type=board_type,
                    num_players=self.num_players,
                    stage=stage,
                )
                if weights:
                    logger.info(
                        "Loaded %s weights from registry for %s_%sp",
                        stage,
                        board_type,
                        self.num_players,
                    )
                    return weights
        except Exception as exc:
            logger.debug("Registry weight loading failed: %s", exc)

        try:
            from .heuristic_weights import get_weights_for_board

            weights = get_weights_for_board(board_type, self.num_players)
            if weights:
                logger.debug(
                    "Using profile weights for %s_%sp",
                    board_type,
                    self.num_players,
                )
                return weights
        except Exception as exc:
            logger.debug("Profile weight loading failed: %s", exc)

        return {
            "stack_count": 1.0,
            "territory_count": 2.0,
            "rings_penalty": 0.1,
            "center_control": 0.3,
        }

    def _apply_weight_noise(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply multiplicative noise to weights for training diversity."""
        if self.weight_noise <= 0:
            return weights.copy()

        import random

        noisy_weights = {}
        for key, value in weights.items():
            noise_factor = 1.0 + random.uniform(-self.weight_noise, self.weight_noise)
            noisy_weights[key] = value * noise_factor
        return noisy_weights

    def _resolve_persona_weights(self, persona_id: str) -> dict[str, float]:
        """Resolve a persona ID to its weight dictionary."""
        from .heuristic_weights import get_weights

        full_id = persona_id
        if not persona_id.startswith("heuristic_"):
            full_id = f"heuristic_v1_{persona_id}"

        persona_weights = get_weights(full_id)
        if not persona_weights:
            logger.warning("Persona '%s' not found, using default", persona_id)
            return self._default_weights()
        return persona_weights

    def _generate_weights_list(self) -> list[list[dict[str, float]]]:
        """Generate per-game, per-player heuristic weights."""
        import random

        weights_list: list[list[dict[str, float]]] = []
        if self.per_player_personas:
            player_weights = [
                self._resolve_persona_weights(persona_id)
                for persona_id in self.per_player_personas
            ]
            for _ in range(self.batch_size):
                game_weights = []
                for player_idx in range(self.num_players):
                    player_weight = player_weights[player_idx].copy()
                    if self.weight_noise > 0:
                        player_weight = self._apply_weight_noise(player_weight)
                    game_weights.append(player_weight)
                weights_list.append(game_weights)
            return weights_list

        if self.persona_pool:
            for _ in range(self.batch_size):
                persona_id = random.choice(self.persona_pool)
                persona_weights = self._resolve_persona_weights(persona_id)
                if self.weight_noise > 0:
                    persona_weights = self._apply_weight_noise(persona_weights)
                weights_list.append([persona_weights] * self.num_players)
            return weights_list

        base_weights = self._default_weights()
        if self.weight_noise <= 0:
            player_weights = [base_weights] * self.num_players
            return [player_weights] * self.batch_size

        for _ in range(self.batch_size):
            noisy = self._apply_weight_noise(base_weights)
            weights_list.append([noisy] * self.num_players)
        return weights_list

    def get_weights_for_current_players(
        self,
        weights_list: list[list[dict[str, float]]] | list[dict[str, float]] | None,
    ) -> list[dict[str, float]] | None:
        """Get weights for the current player of each game."""
        if not weights_list:
            return None

        first_elem = weights_list[0]
        if isinstance(first_elem, list):
            current_players = self.state.current_player.cpu().numpy()
            return [
                weights_list[game_idx][int(current_players[game_idx]) - 1]
                for game_idx in range(self.batch_size)
            ]
        if isinstance(first_elem, dict):
            if not first_elem or len(weights_list) == 1:
                return None
            return weights_list
        return None

    @staticmethod
    def get_available_personas() -> list[str]:
        """Get persona IDs available for training variety."""
        return ["balanced", "aggressive", "territorial", "defensive"]

    @staticmethod
    def get_all_persona_profiles() -> dict[str, dict[str, float]]:
        """Get all persona profiles with their full weight dictionaries."""
        from .heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        personas = {}
        for short_name in GPUPersonaMixin.get_available_personas():
            full_id = f"heuristic_v1_{short_name}"
            if full_id in HEURISTIC_WEIGHT_PROFILES:
                personas[short_name] = HEURISTIC_WEIGHT_PROFILES[full_id]
        return personas

    @classmethod
    def get_training_matchup(cls, matchup_name: str) -> list[str]:
        """Get a predefined training matchup configuration."""
        if matchup_name not in cls.TRAINING_MATCHUPS:
            available = list(cls.TRAINING_MATCHUPS.keys())
            raise ValueError(
                f"Unknown matchup '{matchup_name}'. Available: {available}"
            )
        return cls.TRAINING_MATCHUPS[matchup_name]

    @classmethod
    def get_all_training_matchups(cls) -> list[str]:
        """Get all available training matchup names."""
        return list(cls.TRAINING_MATCHUPS.keys())

    @classmethod
    def create_with_matchup(
        cls,
        matchup_name: str,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> "ParallelGameRunner":
        """Factory method to create a runner with a predefined matchup."""
        matchup = cls.get_training_matchup(matchup_name)
        return cls(
            batch_size=batch_size,
            num_players=len(matchup),
            per_player_personas=matchup,
            use_heuristic_selection=True,
            **kwargs,
        )

    @classmethod
    def run_matchup_tournament(
        cls,
        matchups: list[str] | None = None,
        games_per_matchup: int = 100,
        max_moves: int = 200,
        **runner_kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        """Run a tournament across multiple matchup configurations."""
        if matchups is None:
            matchups = cls.get_all_training_matchups()

        tournament_results = {}
        for matchup_name in matchups:
            runner = cls.create_with_matchup(
                matchup_name,
                batch_size=games_per_matchup,
                **runner_kwargs,
            )
            game_results = runner.run_games(max_moves=max_moves)
            winners = game_results.get("winners", [])
            move_counts = game_results.get("move_counts", [])
            p1_wins = sum(1 for winner in winners if winner == 1)
            p2_wins = sum(1 for winner in winners if winner == 2)
            draws = sum(1 for winner in winners if winner == 0 or winner is None)

            tournament_results[matchup_name] = {
                "p1_wins": p1_wins,
                "p2_wins": p2_wins,
                "draws": draws,
                "total_games": len(winners),
                "avg_game_length": (
                    sum(move_counts) / len(move_counts) if move_counts else 0
                ),
                "personas": cls.get_training_matchup(matchup_name),
            }

        return tournament_results
