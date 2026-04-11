"""Shared MCTS tree structures and search-support helpers.

This module holds the reusable tree-node representations plus the tuning,
self-play, and root-prior helpers that were previously embedded directly in
``mcts_ai.py``. ``MCTSAI`` remains the public AI wrapper while importing and
re-exporting these internals for backward compatibility with tests and helper
scripts.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..models import BoardType, GamePhase, GameState, Move
from ..rules.mutable_state import MoveUndo
from .game_state_utils import infer_num_players

logger = logging.getLogger("app.ai.mcts_ai")

# Module-level cache for NNUE policy models to avoid reloading per MCTSAI
# instance. Key: (board_type.value, num_players) -> RingRiftNNUEWithPolicy model.
_NNUE_POLICY_CACHE: dict[tuple[str, int], Any] = {}
_NNUE_POLICY_CACHE_LOCK = None


def _get_cached_nnue_policy(board_type: BoardType, num_players: int) -> Any | None:
    """Get cached NNUE policy model or load and cache it."""
    global _NNUE_POLICY_CACHE_LOCK

    board_type_value = (
        board_type.value if hasattr(board_type, "value") else str(board_type).lower()
    )
    cache_key = (board_type_value, num_players)

    if cache_key in _NNUE_POLICY_CACHE:
        return _NNUE_POLICY_CACHE[cache_key]

    if _NNUE_POLICY_CACHE_LOCK is None:
        import threading

        _NNUE_POLICY_CACHE_LOCK = threading.Lock()

    with _NNUE_POLICY_CACHE_LOCK:
        if cache_key in _NNUE_POLICY_CACHE:
            return _NNUE_POLICY_CACHE[cache_key]

        try:
            from .nnue_policy import RingRiftNNUEWithPolicy, prepare_policy_checkpoint

            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "models",
                "nnue",
                f"nnue_policy_{board_type_value}_{num_players}p.pt",
            )
            model_path = os.path.normpath(model_path)

            if os.path.exists(model_path):
                from app.utils.torch_utils import safe_load_checkpoint

                checkpoint = safe_load_checkpoint(
                    model_path,
                    map_location="cpu",
                    warn_on_unsafe=False,
                )
                state_dict, hidden_dim, num_hidden_layers = prepare_policy_checkpoint(
                    checkpoint,
                    board_type,
                )

                model = RingRiftNNUEWithPolicy(
                    board_type=board_type,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                )
                model.load_state_dict(state_dict)
                model.eval()

                _NNUE_POLICY_CACHE[cache_key] = model
                logger.info(
                    "NNUE Policy Cache: Loaded model for %s_%sp (hidden=%s, layers=%s)",
                    board_type_value,
                    num_players,
                    hidden_dim,
                    num_hidden_layers,
                )
                return model

            _NNUE_POLICY_CACHE[cache_key] = None
            logger.debug("NNUE Policy Cache: No model at %s", model_path)
            return None
        except Exception as exc:
            logger.warning("NNUE Policy Cache: Failed to load model: %s", exc)
            _NNUE_POLICY_CACHE[cache_key] = None
            return None


def _pos_key(pos: Any | None) -> str | None:
    """Convert a position object to a hashable string key for caching."""
    if pos is None:
        return None
    to_key = getattr(pos, "to_key", None)
    if callable(to_key):
        return cast(str, to_key())
    x = getattr(pos, "x", None)
    y = getattr(pos, "y", None)
    z = getattr(pos, "z", None)
    if x is None or y is None:
        return None
    return f"{x},{y},{z}" if z is not None else f"{x},{y}"


def _pos_seq_key(seq: tuple[Any, ...] | None) -> tuple[str, ...] | None:
    """Convert a sequence of positions to a hashable tuple of string keys."""
    if not seq:
        return None
    return tuple(k for k in (_pos_key(p) for p in seq) if k is not None)


def _move_key(move: Move) -> tuple:
    """Return a stable, hashable key for comparing AI moves."""
    move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
    return (
        move_type,
        int(move.player),
        _pos_key(getattr(move, "from_pos", None)),
        _pos_key(getattr(move, "to", None)),
        _pos_key(getattr(move, "capture_target", None)),
        getattr(move, "placement_count", None),
        getattr(move, "placed_on_stack", None),
        getattr(move, "line_index", None),
        _pos_seq_key(getattr(move, "collapsed_markers", None)),
        _pos_seq_key(getattr(move, "collapse_positions", None)),
        tuple(getattr(move, "extraction_stacks", None) or ()),
        getattr(move, "recovery_option", None),
        getattr(move, "recovery_mode", None),
        getattr(move, "elimination_context", None),
        _pos_seq_key(getattr(move, "capture_chain", None)),
        tuple(getattr(move, "overtaken_rings", None) or ()),
    )


def _moves_match(m1: Move, m2: Move) -> bool:
    """Check if two moves match by semantic identity, not timing metadata."""
    if m1.type != m2.type or m1.player != m2.player:
        return False
    if m1.from_pos != m2.from_pos:
        return False
    if m1.to != m2.to:
        return False
    if m1.capture_target != m2.capture_target:
        return False
    if m1.placement_count != m2.placement_count:
        return False
    if m1.placed_on_stack != m2.placed_on_stack:
        return False
    if m1.line_index != m2.line_index:
        return False
    if m1.collapsed_markers != m2.collapsed_markers:
        return False
    if m1.collapse_positions != m2.collapse_positions:
        return False
    if m1.extraction_stacks != m2.extraction_stacks:
        return False
    if m1.recovery_option != m2.recovery_option:
        return False
    if m1.recovery_mode != m2.recovery_mode:
        return False
    if m1.elimination_context != m2.elimination_context:
        return False
    if m1.capture_chain != m2.capture_chain:
        return False
    return m1.overtaken_rings == m2.overtaken_rings


class MCTSNode:
    """MCTS tree node for legacy (immutable) search."""

    def __init__(
        self,
        game_state: GameState,
        parent: MCTSNode | None = None,
        move: Move | None = None,
    ) -> None:
        self.game_state: GameState = game_state
        self.parent: MCTSNode | None = parent
        self.move: Move | None = move
        self.children: list[MCTSNode] = []
        self.wins = 0
        self.visits = 0
        self.amaf_wins = 0
        self.amaf_visits = 0
        self.untried_moves: list[Move] = []
        self.prior = 0.0
        self.policy_map: dict[str, float] = {}
        self.to_move_is_root: bool = True

    def uct_select_child(
        self,
        *,
        c_puct: float = 1.0,
        rave_k: float = 1000.0,
        fpu_reduction: float = 0.0,
    ) -> MCTSNode:
        """Select child using PUCT formula with RAVE."""

        def puct_value(child: MCTSNode) -> float:
            parent_is_root = bool(getattr(self, "to_move_is_root", True))
            child_is_root = bool(getattr(child, "to_move_is_root", parent_is_root))
            flip = parent_is_root != child_is_root

            parent_q = self.wins / self.visits if self.visits > 0 else 0.0
            if child.visits == 0:
                q_value = parent_q - float(fpu_reduction)
            else:
                q_value = child.wins / child.visits
                if flip:
                    q_value = -q_value

            if child.amaf_visits == 0:
                amaf_value = 0.0
            else:
                amaf_value = child.amaf_wins / child.amaf_visits
                if flip:
                    amaf_value = -amaf_value

            beta = 0.0
            if rave_k > 0:
                beta = math.sqrt(float(rave_k) / (3 * self.visits + float(rave_k)))

            combined_value = (1 - beta) * q_value + beta * amaf_value
            prior = getattr(child, "prior", 1.0 / len(self.children))
            u_value = c_puct * prior * math.sqrt(self.visits) / (1 + child.visits)
            return combined_value + u_value

        return max(self.children, key=puct_value)

    def add_child(
        self,
        move: Move,
        game_state: GameState,
        prior: float | None = None,
    ) -> MCTSNode:
        """Add a new child node."""
        child = MCTSNode(game_state, parent=self, move=move)
        if prior is not None:
            child.prior = prior
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float, played_moves: list[Move] | None = None) -> None:
        """Update node stats."""
        self.visits += 1
        self.wins += result

        if played_moves and self.move:
            for move in played_moves:
                if _moves_match(move, self.move):
                    self.amaf_visits += 1
                    self.amaf_wins += result
                    break


class MCTSNodeLite:
    """Lightweight MCTS tree node for incremental (mutable) search."""

    __slots__ = [
        "amaf_visits",
        "amaf_wins",
        "children",
        "move",
        "parent",
        "policy_map",
        "prior",
        "to_move_is_root",
        "untried_moves",
        "visits",
        "wins",
    ]

    def __init__(
        self,
        parent: MCTSNodeLite | None = None,
        move: Move | None = None,
        to_move_is_root: bool = True,
    ) -> None:
        self.parent: MCTSNodeLite | None = parent
        self.move: Move | None = move
        self.children: list[MCTSNodeLite] = []
        self.wins = 0.0
        self.visits = 0
        self.amaf_wins = 0.0
        self.amaf_visits = 0
        self.untried_moves: list[Move] = []
        self.prior = 0.0
        self.policy_map: dict[str, float] = {}
        self.to_move_is_root: bool = bool(to_move_is_root)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def uct_select_child(
        self,
        *,
        c_puct: float = 1.0,
        rave_k: float = 1000.0,
        fpu_reduction: float = 0.0,
    ) -> MCTSNodeLite:
        """Select child using PUCT formula with RAVE."""

        def puct_value(child: MCTSNodeLite) -> float:
            parent_is_root = bool(getattr(self, "to_move_is_root", True))
            child_is_root = bool(getattr(child, "to_move_is_root", parent_is_root))
            flip = parent_is_root != child_is_root

            parent_q = self.wins / self.visits if self.visits > 0 else 0.0
            if child.visits == 0:
                q_value = parent_q - float(fpu_reduction)
            else:
                q_value = child.wins / child.visits
                if flip:
                    q_value = -q_value

            if child.amaf_visits == 0:
                amaf_value = 0.0
            else:
                amaf_value = child.amaf_wins / child.amaf_visits
                if flip:
                    amaf_value = -amaf_value

            beta = 0.0
            if rave_k > 0:
                beta = math.sqrt(float(rave_k) / (3 * self.visits + float(rave_k)))
            combined_value = (1 - beta) * q_value + beta * amaf_value

            num_children = max(1, len(self.children))
            prior = child.prior if child.prior > 0 else 1.0 / num_children
            sqrt_visits = math.sqrt(self.visits)
            u_value = c_puct * prior * sqrt_visits / (1 + child.visits)
            return combined_value + u_value

        return max(self.children, key=puct_value)

    def add_child(
        self,
        move: Move,
        prior: float | None = None,
        to_move_is_root: bool | None = None,
    ) -> MCTSNodeLite:
        """Add a new child node."""
        child = MCTSNodeLite(
            parent=self,
            move=move,
            to_move_is_root=(
                bool(to_move_is_root)
                if to_move_is_root is not None
                else bool(getattr(self, "to_move_is_root", True))
            ),
        )
        if prior is not None:
            child.prior = prior
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float, played_moves: list[Move] | None = None) -> None:
        """Update node stats."""
        self.visits += 1
        self.wins += result

        if played_moves and self.move:
            for move in played_moves:
                if _moves_match(move, self.move):
                    self.amaf_visits += 1
                    self.amaf_wins += result
                    break


@dataclass
class _EvalBatchLegacy:
    """Batch evaluation state for legacy (immutable) MCTS search."""

    leaves: list[tuple[MCTSNode, GameState, list[Move]]]
    states: list[GameState]
    cached_results: list[tuple[int, float, Any]]
    uncached_indices: list[int]
    uncached_states: list[GameState]
    use_hex_nn: bool


@dataclass
class _EvalBatchIncremental:
    """Batch evaluation state for incremental (mutable) MCTS search."""

    leaves: list[tuple[MCTSNodeLite, list[MoveUndo], list[Move]]]
    states: list[GameState]
    cached_results: list[tuple[int, float, Any]]
    uncached_indices: list[int]
    uncached_states: list[GameState]
    use_hex_nn: bool


class MCTSSearchSupportMixin:
    """Shared search-support helpers used by MCTSAI."""

    def _extract_visit_dist_legacy(
        self,
        root: MCTSNode,
    ) -> tuple[list[Move], list[float]]:
        if not root.children:
            return [], []

        total_visits = sum(child.visits for child in root.children)
        if total_visits == 0:
            return [], []

        moves: list[Move] = []
        probs: list[float] = []
        for child in root.children:
            if child.move is not None and child.visits > 0:
                moves.append(child.move)
                probs.append(child.visits / total_visits)
        return moves, probs

    def _extract_visit_dist_lite(
        self,
        root: MCTSNodeLite,
    ) -> tuple[list[Move], list[float]]:
        if not root.children:
            return [], []

        total_visits = sum(child.visits for child in root.children)
        if total_visits == 0:
            return [], []

        moves: list[Move] = []
        probs: list[float] = []
        for child in root.children:
            if child.move is not None and child.visits > 0:
                moves.append(child.move)
                probs.append(child.visits / total_visits)
        return moves, probs

    def _normalized_entropy(self, priors: list[float]) -> float:
        if not priors:
            return 0.0
        total = float(sum(priors))
        if total <= 0.0:
            return 1.0
        if len(priors) <= 1:
            return 0.0
        inv_total = 1.0 / total
        ent = 0.0
        for prior in priors:
            if prior <= 0.0:
                continue
            normalized_prior = float(prior) * inv_total
            ent -= normalized_prior * math.log(normalized_prior)
        denom = math.log(len(priors))
        if denom <= 0.0:
            return 0.0
        return max(0.0, min(1.0, ent / denom))

    def _dynamic_c_puct(self, parent_visits: int, priors: list[float]) -> float:
        entropy = self._normalized_entropy(priors)
        visit_term = min(1.0, math.log1p(max(0, int(parent_visits))) / 6.0)
        cpuct = 1.0 + 0.8 * entropy + 0.4 * visit_term
        return float(max(0.25, min(4.0, cpuct)))

    def _rave_k_for_node(self, parent_visits: int, priors: list[float]) -> float:
        entropy = self._normalized_entropy(priors)
        difficulty = int(getattr(self.config, "difficulty", 5))
        difficulty_scale = max(0.2, 1.0 - 0.12 * max(0, difficulty - 5))
        visit_scale = 1.0 / (1.0 + max(0, int(parent_visits)) / 200.0)
        entropy_scale = 0.5 + 0.5 * entropy
        base_k = 1000.0
        return float(max(0.0, base_k * difficulty_scale * visit_scale * entropy_scale))

    def _fpu_reduction_for_phase(self, phase: GamePhase) -> float:
        phase_map = {
            GamePhase.RING_PLACEMENT: 0.05,
            GamePhase.MOVEMENT: 0.10,
            GamePhase.CAPTURE: 0.12,
            GamePhase.CHAIN_CAPTURE: 0.12,
            GamePhase.LINE_PROCESSING: 0.16,
            GamePhase.TERRITORY_PROCESSING: 0.20,
            GamePhase.FORCED_ELIMINATION: 0.22,
        }
        return float(phase_map.get(phase, 0.10))

    def _puct_params_for_node(
        self,
        node: Any,
        phase: GamePhase,
    ) -> tuple[float, float, float]:
        children = getattr(node, "children", None) or []
        if not children:
            priors: list[float] = []
        else:
            uniform = 1.0 / max(1, len(children))
            priors = [
                float(getattr(child, "prior", 0.0) or 0.0) or uniform
                for child in children
            ]

        visits = int(getattr(node, "visits", 0) or 0)
        c_puct = self._dynamic_c_puct(visits, priors)
        rave_k = self._rave_k_for_node(visits, priors)
        fpu_reduction = self._fpu_reduction_for_phase(phase)
        return c_puct, rave_k, fpu_reduction

    def _default_dirichlet_alpha(self, board_type: BoardType) -> float:
        if board_type == BoardType.SQUARE8:
            return 0.3
        return 0.15

    def _maybe_apply_root_dirichlet_noise(
        self,
        node: Any,
        board_type: BoardType,
    ) -> None:
        if not self.self_play or self._dirichlet_applied_this_search:
            return
        if getattr(node, "parent", None) is not None:
            return
        if not getattr(node, "policy_map", None):
            return

        keys = list(node.policy_map.keys())
        if len(keys) <= 1 or self.root_noise_fraction <= 0:
            self._dirichlet_applied_this_search = True
            return

        alpha = self.root_dirichlet_alpha or self._default_dirichlet_alpha(board_type)
        epsilon = self.root_noise_fraction
        seed = int(self.rng.randrange(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        noise = rng.dirichlet([alpha] * len(keys))

        for index, key in enumerate(keys):
            prior = float(node.policy_map[key])
            node.policy_map[key] = (1.0 - epsilon) * prior + epsilon * float(noise[index])

        total = float(sum(node.policy_map.values()))
        if total > 0:
            for key in node.policy_map:
                node.policy_map[key] /= total

        self._dirichlet_applied_this_search = True

    def _get_selfplay_temperature(self, game_state: GameState) -> float:
        if self.temperature_override is not None:
            return float(self.temperature_override)

        board_type = game_state.board.type
        cutoff = self.temperature_cutoff_moves
        if cutoff is None:
            cutoff = 24 if board_type == BoardType.SQUARE8 else 40

        move_index = len(game_state.move_history)
        if move_index < cutoff:
            return 1.0
        if move_index < cutoff * 2:
            return 0.5
        return 0.1

    def _default_leaf_batch_size(self) -> int:
        env_val = os.environ.get("RINGRIFT_MCTS_LEAF_BATCH_SIZE")
        if env_val:
            try:
                parsed = int(env_val)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass

        if not self.neural_net:
            return 8

        device = getattr(self.neural_net, "device", "cpu")
        dev_str = device if isinstance(device, str) else getattr(device, "type", "cpu")
        if dev_str == "cuda":
            return 32
        if dev_str == "mps":
            return 16
        if dev_str == "cpu":
            return 8
        return 16

    def _maybe_seed_root_priors(self, root: Any, game_state: GameState) -> None:
        board_type = game_state.board.type
        use_progressive = self._use_progressive_widening(board_type)
        if not use_progressive and (self.neural_net or self.nnue_policy_model is None):
            return

        existing_map = getattr(root, "policy_map", None)
        if isinstance(existing_map, dict) and existing_map:
            return

        if not self.neural_net:
            if self.nnue_policy_model is not None:
                try:
                    valid_moves = self.rules_engine.get_valid_moves(
                        game_state,
                        game_state.current_player,
                    )
                    nnue_policy = self._compute_nnue_policy(valid_moves, game_state)
                    if nnue_policy:
                        root.policy_map = nnue_policy
                        root.untried_moves = list(valid_moves)
                        root.untried_moves.sort(
                            key=lambda move: root.policy_map.get(str(move), 0.0),
                            reverse=True,
                        )
                        for child in getattr(root, "children", []):
                            move = getattr(child, "move", None)
                            if move is None:
                                continue
                            prior = root.policy_map.get(str(move))
                            if prior is not None:
                                child.prior = float(prior)
                        max_p = max(nnue_policy.values()) if nnue_policy else 0.0
                        min_p = min(nnue_policy.values()) if nnue_policy else 0.0
                        logger.debug(
                            "Seeded NNUE root priors for %s: %d moves, temp=%.1f, max_prior=%.3f, min_prior=%.4f",
                            board_type.value,
                            len(nnue_policy),
                            self.policy_temperature,
                            max_p,
                            min_p,
                        )
                except (ValueError, TypeError, KeyError, AttributeError):
                    logger.debug("Failed to seed NNUE root priors", exc_info=True)
            return

        try:
            use_hex_nn = (
                self.hex_model is not None
                and self.hex_encoder is not None
                and board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
            )
            use_vector_head = (
                self.use_vector_value_head
                and not use_hex_nn
                and infer_num_players(game_state) > 2
            )
            value_head = (self.player_number - 1) if use_vector_head else None

            if use_hex_nn:
                eval_values, eval_policies = self._evaluate_hex_batch([game_state])
                policy_vec = eval_policies[0]
                value = float(eval_values[0]) if eval_values else 0.0
            else:
                eval_values, policy_batch = (
                    self.nn_batcher.evaluate([game_state], value_head=value_head)
                    if self.nn_batcher is not None
                    else self.neural_net.evaluate_batch([game_state], value_head=value_head)
                )
                policy_vec = policy_batch[0]
                value = float(eval_values[0]) if eval_values else 0.0

            if isinstance(root, MCTSNode):
                self._update_node_policy_legacy(
                    root,
                    game_state,
                    policy_vec,
                    bool(use_hex_nn),
                )
            else:
                self._update_node_policy_lite(
                    cast(MCTSNodeLite, root),
                    game_state,
                    policy_vec,
                    bool(use_hex_nn),
                )

            for child in getattr(root, "children", []):
                move = getattr(child, "move", None)
                if move is None:
                    continue
                prior = getattr(root, "policy_map", {}).get(str(move))
                if prior is not None:
                    child.prior = float(prior)

            state_hash = game_state.zobrist_hash or 0
            if self.transposition_table.get(state_hash) is None:
                self.transposition_table.put(state_hash, (value, policy_vec))
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            logger.debug("Failed to seed root priors", exc_info=True)

    def _sample_child_by_temperature(
        self,
        children: list[Any],
        temperature: float,
    ) -> Any:
        if temperature <= 0 or len(children) == 1:
            return max(children, key=lambda child: child.visits)

        visits = np.array(
            [max(0.0, float(child.visits)) for child in children],
            dtype=np.float64,
        )
        if visits.sum() <= 0:
            probs = np.ones_like(visits) / len(visits)
        else:
            probs = visits / visits.sum()

        if temperature != 1.0:
            probs = probs ** (1.0 / float(temperature))
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs /= prob_sum

        index = self.rng.choices(
            list(range(len(children))),
            weights=probs.tolist(),
            k=1,
        )[0]
        return children[index]

    def _use_progressive_widening(self, board_type: BoardType) -> bool:
        return board_type in (BoardType.SQUARE19, BoardType.HEXAGONAL)

    def _max_children_allowed(self, visits: int, board_type: BoardType) -> int:
        if not self._use_progressive_widening(board_type):
            return 1_000_000_000

        min_children = 8 if board_type == BoardType.SQUARE19 else 10
        return max(min_children, int(2.0 * (max(1, int(visits)) ** 0.5)))

    def _can_expand_node(self, node: Any, board_type: BoardType) -> bool:
        if not self._use_progressive_widening(board_type):
            return True
        visits = int(getattr(node, "visits", 0))
        children = getattr(node, "children", [])
        return len(children) < self._max_children_allowed(visits, board_type)

    def _select_untried_move(self, node: Any, board_type: BoardType) -> Move:
        del board_type
        moves: list[Move] = list(getattr(node, "untried_moves", []))
        if not moves:
            raise ValueError("No untried moves to select")
        policy_map = getattr(node, "policy_map", None)
        if isinstance(policy_map, dict) and policy_map:
            return max(moves, key=lambda move: policy_map.get(str(move), 0.0))
        return cast(Move, self.get_random_element(moves))

    def _log_stats(self) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        tt_stats = self.transposition_table.stats()
        logger.debug(
            "MCTS transposition table stats: entries=%d/%d, hits=%d, misses=%d, hit_rate=%.2f%%, evictions=%d, est_memory=%.2fMB",
            tt_stats["entries"],
            tt_stats["max_entries"],
            tt_stats["hits"],
            tt_stats["misses"],
            tt_stats["hit_rate"] * 100,
            tt_stats["evictions"],
            tt_stats["estimated_memory_mb"],
        )

        if self.enable_dynamic_batching and self.dynamic_sizer is not None:
            ds_stats = self.dynamic_sizer.stats()
            logger.debug(
                "MCTS dynamic batch sizer stats: batch_size=%d, node_estimate=%d bytes, adjustments=%d",
                ds_stats["current_batch_size"],
                ds_stats["node_size_estimate"],
                ds_stats["adjustment_count"],
            )

    def clear_tree(self) -> None:
        """Clear the cached legacy tree root to free memory."""
        if hasattr(self, "last_root"):
            self.last_root = None


__all__ = [
    "MCTSNode",
    "MCTSNodeLite",
    "MCTSSearchSupportMixin",
    "_EvalBatchIncremental",
    "_EvalBatchLegacy",
    "_get_cached_nnue_policy",
    "_move_key",
    "_moves_match",
    "_pos_key",
    "_pos_seq_key",
]
