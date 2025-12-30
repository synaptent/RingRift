"""Concrete harness implementations.

This module contains the actual harness classes that wrap the underlying
AI implementations and adapt them to the unified AIHarness interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base_harness import AIHarness, HarnessConfig, ModelType

if TYPE_CHECKING:
    from ...models import GameState, Move

logger = logging.getLogger(__name__)


class GumbelMCTSHarness(AIHarness):
    """Harness for Gumbel MCTS with Sequential Halving."""

    supports_nn = True
    supports_nnue = False
    requires_policy_head = True

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..gumbel_mcts_ai import GumbelMCTSAI

        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            simulations=self.config.simulations,
            **self.config.extra,
        )
        ai = GumbelMCTSAI(player_number, ai_config, model_path=self.config.model_path)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        # Ensure AI is for correct player
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        # Extract visit distribution from tree
        self._last_visit_distribution = self._extract_visit_distribution()
        self._last_policy_distribution = self._extract_policy_distribution()

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_value', 0.0),
            "nodes_visited": getattr(self._underlying_ai, 'nodes_visited', 0),
            "search_depth": getattr(self._underlying_ai, 'max_depth_reached', None),
            "simulations": self.config.simulations,
        }
        return move, metadata

    def _extract_visit_distribution(self) -> dict[str, float] | None:
        """Extract visit distribution from Gumbel MCTS tree."""
        if not hasattr(self._underlying_ai, '_root_visits'):
            return None
        root_visits = getattr(self._underlying_ai, '_root_visits', None)
        if root_visits is None:
            return None
        return {str(k): float(v) for k, v in root_visits.items()}

    def _extract_policy_distribution(self) -> dict[str, float] | None:
        """Extract policy distribution from neural network."""
        if not hasattr(self._underlying_ai, '_root_policy'):
            return None
        root_policy = getattr(self._underlying_ai, '_root_policy', None)
        if root_policy is None:
            return None
        return {str(k): float(v) for k, v in root_policy.items()}


class GPUGumbelHarness(AIHarness):
    """Harness for GPU-accelerated Gumbel MCTS."""

    supports_nn = True
    supports_nnue = False
    requires_policy_head = True

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..tensor_gumbel_tree import GPUGumbelMCTS

        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            simulations=self.config.simulations,
            **self.config.extra,
        )
        # GPUGumbelMCTS needs neural net loaded
        from ..neural_net import NeuralNetAI
        nn_ai = NeuralNetAI(player_number, ai_config, model_path=self.config.model_path)
        ai = GPUGumbelMCTS(
            neural_net=nn_ai,
            budget=self.config.simulations,
            **self.config.extra,
        )
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        move = self._underlying_ai.search(game_state)

        # GPU Gumbel provides visit distribution directly
        visit_dist = self._underlying_ai.get_root_visit_distribution()
        if visit_dist:
            self._last_visit_distribution = {
                str(k): float(v) for k, v in visit_dist.items()
            }

        metadata = {
            "value_estimate": getattr(self._underlying_ai, 'root_value', 0.0),
            "nodes_visited": getattr(self._underlying_ai, 'total_nodes', 0),
            "simulations": self.config.simulations,
        }
        return move, metadata


class MinimaxHarness(AIHarness):
    """Harness for Minimax with alpha-beta pruning."""

    supports_nn = True
    supports_nnue = True
    requires_policy_head = False

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..minimax_ai import MinimaxAI

        use_nn = self.config.model_type in (ModelType.NEURAL_NET, ModelType.NNUE)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            use_neural_net=use_nn,
            nn_model_id=self.config.model_id if use_nn else None,
            **self.config.extra,
        )
        ai = MinimaxAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_eval', 0.0),
            "nodes_visited": self._underlying_ai.nodes_visited,
            "search_depth": self._underlying_ai._get_max_depth(),
        }
        return move, metadata


class MaxNHarness(AIHarness):
    """Harness for Max-N multiplayer search."""

    supports_nn = True
    supports_nnue = True
    requires_policy_head = False

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..maxn_ai import MaxNAI

        use_nn = self.config.model_type in (ModelType.NEURAL_NET, ModelType.NNUE)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            use_neural_net=use_nn,
            **self.config.extra,
        )
        ai = MaxNAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": 0.0,  # Max-N returns score vectors
            "nodes_visited": self._underlying_ai.nodes_visited,
            "search_depth": self._underlying_ai._get_max_depth(),
        }
        return move, metadata


class BRSHarness(AIHarness):
    """Harness for Best-Reply Search."""

    supports_nn = True
    supports_nnue = True
    requires_policy_head = False

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..maxn_ai import BRSAI

        use_nn = self.config.model_type in (ModelType.NEURAL_NET, ModelType.NNUE)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            use_neural_net=use_nn,
            **self.config.extra,
        )
        ai = BRSAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": 0.0,
            "nodes_visited": self._underlying_ai.nodes_visited,
            "search_depth": self._underlying_ai._get_lookahead_rounds(),
        }
        return move, metadata


class PolicyOnlyHarness(AIHarness):
    """Harness for direct policy sampling (no search)."""

    supports_nn = True
    supports_nnue = True  # If NNUE has policy head
    requires_policy_head = True

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..policy_only_ai import PolicyOnlyAI

        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            **self.config.extra,
        )
        ai = PolicyOnlyAI(player_number, ai_config, model_path=self.config.model_path)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        # Get policy distribution
        policy = getattr(self._underlying_ai, '_last_policy', None)
        if policy is not None:
            self._last_policy_distribution = {
                str(k): float(v) for k, v in policy.items()
            }

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_value', 0.0),
            "nodes_visited": 1,  # Single evaluation
        }
        return move, metadata


class DescentHarness(AIHarness):
    """Harness for gradient descent move selection."""

    supports_nn = True
    supports_nnue = False
    requires_policy_head = True

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..descent_ai import DescentAI

        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            **self.config.extra,
        )
        ai = DescentAI(player_number, ai_config, model_path=self.config.model_path)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_value', 0.0),
            "nodes_visited": getattr(self._underlying_ai, 'iterations', 1),
        }
        return move, metadata


class HeuristicHarness(AIHarness):
    """Harness for pure heuristic evaluation (no neural network)."""

    supports_nn = False
    supports_nnue = False
    requires_policy_head = False

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..heuristic_ai import HeuristicAI

        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            **self.config.extra,
        )
        ai = HeuristicAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_eval', 0.0),
            "nodes_visited": 1,
        }
        return move, metadata
