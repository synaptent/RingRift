import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Literal

from app.models import (
    GameState,
    Move,
    BoardType,
    GameStatus,
    GamePhase,
    MoveType,
)
from app.game_engine import GameEngine, PhaseRequirementType
from app.rules.default_engine import DefaultRulesEngine
from app.rules.fsm import TurnFSM, get_fsm_mode
from app.training.seed_utils import seed_all
from app.training.tournament import infer_victory_reason

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Theoretical Maximum Moves by Board Type and Player Count
# -----------------------------------------------------------------------------
# These represent upper bounds on game length based on board geometry.
# A game exceeding these thresholds is anomalous and should be logged/flagged.
#
# Derivation:
# - Square8: 64 cells, 18 rings/player (2p) = 36 total rings
#   Max placements ~36, max movements ~36*10, captures reduce rings
#   Conservative estimate: 150 moves for 2p, +50 per additional player
#
# - Square19: 361 cells, 48 rings/player (2p) = 96 total rings
#   Much larger board, longer games expected
#   Conservative estimate: 400 moves for 2p, +100 per additional player
#
# - Hexagonal: 469 cells (radius 12), 60 rings/player (2p) = 120 total rings
#   Similar to Square19 scale
#   Conservative estimate: 400 moves for 2p, +100 per additional player
#
# Games reaching these limits without a winner indicate potential bugs.
#
# NOTE: With canonical recording (RR-CANON-R075), each turn generates multiple
# moves: RING_PLACEMENT/NO_PLACEMENT_ACTION, MOVEMENT/NO_MOVEMENT_ACTION,
# LINE_PROCESSING/NO_LINE_ACTION, TERRITORY_PROCESSING/NO_TERRITORY_ACTION,
# plus any captures. Typical games show ~4-5 moves per turn, so limits must
# account for this multiplier.
# -----------------------------------------------------------------------------

THEORETICAL_MAX_MOVES: Dict[BoardType, Dict[int, int]] = {
    BoardType.SQUARE8: {
        2: 400,   # ~80 turns * 5 moves/turn
        3: 500,
        4: 600,
    },
    BoardType.SQUARE19: {
        2: 2000,  # larger board, more phases
        3: 2500,
        4: 3000,
    },
    BoardType.HEXAGONAL: {
        2: 2000,
        3: 2500,
        4: 3000,
    },
}


def get_theoretical_max_moves(board_type: BoardType, num_players: int) -> int:
    """Return theoretical maximum moves for a board type and player count.

    If the exact player count isn't defined, extrapolate from the pattern.
    """
    board_limits = THEORETICAL_MAX_MOVES.get(board_type, {})
    if num_players in board_limits:
        return board_limits[num_players]
    # Extrapolate: use 2-player base + increment per extra player
    base = board_limits.get(2, 200)
    increment = board_limits.get(3, base + 50) - base
    return base + increment * (num_players - 2)


# Canonical default evaluation configuration for heuristic
# training (GA, CMA-ES). This configuration is intended for
# multi-board, multi-start evaluation runs using the CMA-ES
# and GA heuristic scripts under ``ai-service/scripts``.
# It serves as a single source of truth for default board
# sets and evaluation kwargs.
DEFAULT_TRAINING_EVAL_CONFIG: Dict[str, Any] = {
    "boards": [
        BoardType.SQUARE8,
        BoardType.SQUARE19,
        BoardType.HEXAGONAL,
    ],
    "eval_mode": "multi-start",
    "state_pool_id": "v1",
    "games_per_eval": 16,
    "max_moves": 400,  # Increased for canonical recording (~4-5 moves/turn)
    "eval_randomness": 0.0,
    # RNG seed is supplied by calling code; see `build_training_eval_kwargs`.
}

# 2-player training preset for heuristic optimisation (CMA-ES / GA).
#
# This preset encodes the recommended "serious" 2-player training regime:
#
# - Multi-board evaluation on Square8, Square19, and Hexagonal boards.
# - Multi-start evaluation from fixed mid/late-game state pools ("v1").
# - A small amount of evaluation-time randomness to break symmetry and
#   avoid degenerate 0.5 plateaus while remaining reproducible when a
#   seed is provided.
#
# Callers that need a simple 2-player training configuration should prefer
# this preset (via `get_two_player_training_kwargs`) over re-encoding
# boards / eval_mode / state_pool_id / eval_randomness by hand.
TWO_PLAYER_TRAINING_PRESET: Dict[str, Any] = {
    **DEFAULT_TRAINING_EVAL_CONFIG,
    "eval_randomness": 0.02,
}

# Canonical heuristic evaluation mode mapping for training/soaks.
#
# Training harnesses (CMA-ES, GA, self-play soaks) should consult this
# mapping when constructing HeuristicAI instances so that:
#
# - Square8 uses the full structural evaluator (historical behaviour).
# - Square19 and Hexagonal boards default to the lightweight evaluator
#   that omits Tier-2 structural/global features for better throughput.
TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD: Dict[BoardType, str] = {
    BoardType.SQUARE8: "full",
    BoardType.SQUARE19: "light",
    BoardType.HEXAGONAL: "light",
}


def build_training_eval_kwargs(
    games_per_eval: Optional[int] = None,
    eval_randomness: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build canonical kwargs for heuristic training evaluation.

    This helper centralises defaults for multi-board, multi-start
    heuristic evaluation, returning a dict that can be splatted into
    calls such as ``evaluate_fitness_over_boards(...)`` and
    higher-level GA / CMA-ES harnesses.

    It is not a mandatory API; scripts may construct kwargs manually,
    but should keep semantics aligned with DEFAULT_TRAINING_EVAL_CONFIG
    to ensure comparable results across runs.
    """
    cfg: Dict[str, Any] = dict(DEFAULT_TRAINING_EVAL_CONFIG)
    if games_per_eval is not None:
        cfg["games_per_eval"] = games_per_eval
    if eval_randomness is not None:
        cfg["eval_randomness"] = eval_randomness
    # Always thread the seed explicitly so callers can control reproducibility.
    cfg["seed"] = seed
    return cfg


def get_two_player_training_kwargs(
    games_per_eval: int,
    seed: int,
) -> Dict[str, Any]:
    """Return canonical kwargs for 2-player heuristic training evaluation.

    The returned dict is suitable for splatting into
    :func:`evaluate_fitness_over_boards` and related helpers. It encodes the
    recommended 2-player CMA-ES / GA configuration:

    - multi-board evaluation over Square8, Square19, and Hexagonal boards;
    - ``eval_mode='multi-start'`` from the ``'v1'`` mid/late-game pools; and
    - a small, non-zero ``eval_randomness`` for symmetry breaking.

    Callers remain free to override any of the keys (for example,
    ``eval_mode`` or ``eval_randomness``) after calling this helper.
    """
    cfg: Dict[str, Any] = dict(TWO_PLAYER_TRAINING_PRESET)
    cfg["games_per_eval"] = games_per_eval
    cfg["seed"] = seed
    return cfg


@dataclass
class TrainingEnvConfig:
    """Configuration for the canonical RingRift training environment.

    This dataclass captures the knobs that must be kept stable across
    training, evaluation, and tournament tooling. It is intentionally
    small and versioned via code rather than ad-hoc keyword arguments.

    Attributes
    ----------
    board_type:
        Logical board geometry. The calibration environment uses
        ``BoardType.SQUARE8``.
    num_players:
        Number of active players (2–4). The calibration environment
        uses 2-player games.
    max_moves:
        Hard upper bound on episode length. When ``None`` a sensible
        default is chosen for the given board/player combination
        (see :func:`get_theoretical_max_moves`). Hitting this limit
        is treated as a host-level cutoff and surfaced as the
        ``"max_moves"`` victory_reason in ``info``.
    reward_mode:
        Reward shaping mode used by :meth:`RingRiftEnv.step`:

        * ``"terminal"`` – +1 / −1 / 0 only at terminal states.
        * ``"shaped"`` – delegates to ``calculate_outcome`` from
          :mod:`app.training.generate_data` at terminal states.

        Non-terminal steps always return reward 0.0.
    seed:
        Optional default RNG seed. If set, :meth:`RingRiftEnv.reset`
        will seed all supported RNGs via :func:`seed_all` whenever it
        is called without an explicit ``seed`` argument.
    use_default_rules_engine:
        When True (default) the environment applies moves via
        :class:`app.rules.default_engine.DefaultRulesEngine`, which
        in turn delegates to :class:`app.game_engine.GameEngine` while
        running optional shadow contracts. When False, the environment
        falls back to calling :func:`GameEngine.apply_move` directly.
    """

    board_type: BoardType = BoardType.SQUARE8
    num_players: int = 2
    max_moves: Optional[int] = None
    reward_mode: Literal["terminal", "shaped"] = "terminal"
    seed: Optional[int] = None
    use_default_rules_engine: bool = True


def make_env(config: Optional[TrainingEnvConfig] = None) -> "RingRiftEnv":
    """Construct the canonical RingRift training environment.

    This helper centralises how environments are created so that all
    callers (self-play generators, evaluation scripts, tournaments,
    and future RL loops) share a single, well-documented interface.

    Callers should prefer this factory over instantiating
    :class:`RingRiftEnv` directly.

    Parameters
    ----------
    config:
        :class:`TrainingEnvConfig` specifying board geometry,
        player count, reward mode, and move budget. When ``None``
        the calibration configuration for 2-player square8 is used.

    Returns
    -------
    RingRiftEnv
        A freshly constructed environment instance.
    """
    if config is None:
        config = TrainingEnvConfig()

    # Derive a concrete max_moves bound.
    if config.max_moves is not None:
        max_moves = config.max_moves
    else:
        # Fall back to the theoretical limit for the given board / player count.
        max_moves = get_theoretical_max_moves(
            config.board_type,
            config.num_players,
        )

    return RingRiftEnv(
        board_type=config.board_type,
        max_moves=max_moves,
        reward_on=config.reward_mode,
        num_players=config.num_players,
        default_seed=config.seed,
        use_default_rules_engine=config.use_default_rules_engine,
    )


class RingRiftEnv:
    """Canonical training environment for RingRift AI (gym-like API).

    This environment wraps the Python rules engine and exposes a small,
    stable surface suitable for training loops, evaluation harnesses,
    and tournament tooling.

    Core semantics
    --------------

    * **Observation**
        The observation returned from :meth:`reset` and
        :meth:`step` is a :class:`GameState` instance mirroring the
        shared TypeScript ``GameState`` type (see ``app.models``).

        Callers are expected to convert this to tensors using the
        existing encoders in :mod:`app.training.encoding` and
        :mod:`app.training.heuristic_features` rather than inventing
        new representations.

    * **Action**
        :meth:`step` accepts a fully-specified :class:`Move` object.
        When integrating with neural-network policies, actions should
        be encoded/decoded using the existing policy-head encoders
        (for example :meth:`NeuralNetAI.encode_move` and the helpers
        in :mod:`app.training.encoding`). The environment itself does
        not define a new integer action space.

    * **Reward**
        By default (``reward_on='terminal'``) the reward is:

        * +1.0 for a win for the player who just moved,
        * −1.0 for a loss for that player,
        * 0.0 for a draw/structural stalemate/max-moves cutoff.

        Rewards are only emitted when ``done`` is ``True``; all
        intermediate steps return 0.0. When ``reward_on='shaped'``,
        the terminal reward is delegated to
        :func:`app.training.generate_data.calculate_outcome`.

    * **Termination and info**
        ``done`` is ``True`` when either the underlying GameState
        becomes non-ACTIVE (the rules engine has finished the game)
        or the environment's ``max_moves`` budget is reached.

        When ``done`` is ``True`` the ``info`` mapping contains at
        least:

        * ``'winner'`` – winning player number or ``None``.
        * ``'victory_reason'`` – canonical result string, one of:

          ``'ring_elimination'``, ``'territory_control'``,
          ``'last_player_standing'``, ``'structural_stalemate'``,
          ``'max_moves'``, or ``'unknown'``.

        * ``'rings_eliminated'`` – mapping from causing player
          number to the number of rings they have eliminated.
        * ``'territory_spaces'`` – mapping from player number to
          their ``territory_spaces`` counter.
        * ``'moves_played'`` – total number of moves in the episode.

        The ``info`` dict always contains ``'move_count'`` with the
        current move index; additional keys may be added in future
        but existing keys are kept stable.

    See also
    --------
    ``docs/ai/AI_DIFFICULTY_SPEC.md`` for calibration and difficulty
    ladder details.
    """

    def __init__(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        max_moves: int = 400,  # Increased for canonical recording (~4-5 moves/turn)
        reward_on: str = "terminal",  # "terminal" or "shaped"
        num_players: int = 2,
        *,
        default_seed: Optional[int] = None,
        use_default_rules_engine: bool = True,
    ):
        self.board_type = board_type
        self.max_moves = max_moves
        self.reward_on = reward_on
        self.num_players = num_players
        self._default_seed = default_seed
        self._state: Optional[GameState] = None
        self._move_count: int = 0

        self._use_default_rules_engine = use_default_rules_engine
        self._rules_engine: Optional[DefaultRulesEngine] = None
        if use_default_rules_engine:
            # Shadow-contract / mutator-first behaviour is controlled by
            # environment flags (see DefaultRulesEngine); we do not override
            # those here.
            self._rules_engine = DefaultRulesEngine()
        # Force host-side bookkeeping synthesis when requested (used by
        # canonical generators to mirror TS orchestration).
        self._force_bookkeeping_moves = os.getenv(
            "RINGRIFT_FORCE_BOOKKEEPING_MOVES", ""
        ).lower() in {"1", "true", "yes", "on"}

        # FSM validation for phase-to-move contract enforcement.
        # Mode is controlled by RINGRIFT_FSM_VALIDATION_MODE environment variable:
        # - "off" (default): No validation
        # - "shadow": Logs violations but continues processing
        # - "active": Raises FSMValidationError on violations (fail-fast)
        self._fsm: Optional[TurnFSM] = None
        fsm_mode = get_fsm_mode()
        if fsm_mode != "off":
            self._fsm = TurnFSM(mode=fsm_mode)
            logger.info(f"FSM validation enabled in '{fsm_mode}' mode")

    def reset(self, seed: Optional[int] = None) -> GameState:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        seed:
            Optional RNG seed. When ``None`` and a ``default_seed`` was
            supplied at construction time, that default is used. The
            seed is threaded through :func:`seed_all`, which initialises
            Python ``random``, NumPy, and torch RNGs in a consistent way
            for training/evaluation runs.

        Returns
        -------
        GameState
            Fresh initial state for the configured board and player
            count.
        """
        if seed is None:
            seed = self._default_seed
        if seed is not None:
            # Use the central training seeding utility so that Python
            # random, NumPy, and torch (including CUDA/cuDNN flags) are
            # all initialised consistently for this environment.
            seed_all(seed)

        # Reuse a shared helper from generate_data.
        # Avoid circular import by importing inside method.
        from app.training.generate_data import create_initial_state

        self._state = create_initial_state(
            board_type=self.board_type,
            num_players=self.num_players,
        )
        self._move_count = 0

        # Reset FSM state for the new episode.
        if self._fsm is not None:
            self._fsm.reset()

        return self._state

    @property
    def state(self) -> GameState:
        """Return the current GameState.

        This property is mainly provided for convenience in tests and
        diagnostic tooling; training loops should treat the object
        returned from :meth:`reset` / :meth:`step` as the canonical
        observation.
        """
        assert self._state is not None, "Call reset() before using env"
        return self._state

    def legal_moves(self) -> List[Move]:
        """Return legal moves for the current player.

        Per RR-CANON-R076, the core rules layer (GameEngine.get_valid_moves)
        returns ONLY interactive moves. When there are no interactive moves,
        this host-level method checks for phase requirements and synthesizes
        the appropriate bookkeeping move (no_*_action, forced_elimination).

        The returned list contains fully-specified :class:`Move`
        instances and matches the behaviour of the TypeScript shared
        engine for the same state.
        """
        # Get interactive moves from the core rules layer.
        if self._rules_engine is not None:
            moves = self._rules_engine.get_valid_moves(
                self.state,
                self.state.current_player,
            )
        else:
            moves = GameEngine.get_valid_moves(
                self.state,
                self.state.current_player,
            )

        # If no interactive moves, check for phase requirements (R076)
        if not moves:
            requirement = GameEngine.get_phase_requirement(
                self.state,
                self.state.current_player,
            )
            if requirement is not None:
                # Host synthesizes the bookkeeping move
                bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                    requirement,
                    self.state,
                )
                moves = [bookkeeping_move]

        # Defensive phase/move invariant: every move returned by this
        # host-level surface must be legal for the current phase.
        for move in moves:
            GameEngine._assert_phase_move_invariant(self.state, move)

        # Defensive phase-requirement consistency check: if the core
        # engine reports a pending phase requirement, ensure we surfaced
        # exactly one corresponding bookkeeping move.
        requirement = GameEngine.get_phase_requirement(
            self.state,
            self.state.current_player,
        )
        if requirement is not None:
            expected = GameEngine.synthesize_bookkeeping_move(
                requirement,
                self.state,
            )
            if not moves:
                raise RuntimeError(
                    "RingRiftEnv.legal_moves: phase requirement exists "
                    f"({requirement.type.value}) but no legal moves were "
                    "returned"
                )
            if len(moves) != 1 or moves[0].type != expected.type:
                raise RuntimeError(
                    "RingRiftEnv.legal_moves: inconsistent bookkeeping move "
                    f"for requirement {requirement.type.value}: "
                    f"got {moves[0].type.value}, expected {expected.type.value}"
                )

        return moves

    def _infer_canonical_victory_reason(
        self,
        terminated_by_budget_only: bool,
    ) -> str:
        """Map engine-level termination state to a canonical result string.

        This helper keeps the mapping between Python/TS result enums
        and the canonical categories used by training and evaluation.
        """
        if (
            terminated_by_budget_only
            and self._state is not None
            and self._state.game_status == GameStatus.ACTIVE
        ):
            return "max_moves"

        if self._state is None:
            return "unknown"

        engine_reason = infer_victory_reason(self._state)
        mapping = {
            "elimination": "ring_elimination",
            "territory": "territory_control",
            "last_player_standing": "last_player_standing",
            "structural": "structural_stalemate",
            "unknown": "unknown",
        }
        return mapping.get(engine_reason, engine_reason)

    def step(
        self, move: Move
    ) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """Apply a move and advance the environment.

        Parameters
        ----------
        move:
            A legal :class:`Move` for the current player. Legality is
            not re-validated here; callers are expected to pass moves
            drawn from :meth:`legal_moves` or validated by the rules
            engine.

        Returns
        -------
        observation:
            The updated :class:`GameState`.
        reward:
            Scalar reward from the perspective of ``move.player``
            according to the configured ``reward_on`` mode.
        done:
            ``True`` if the episode has terminated either because the
            rules engine finished the game or because ``max_moves`` was
            reached.
        info:
            A dictionary with episode metadata; see class docstring for
            the stable fields.
        """
        # Enforce canonical phase→MoveType mapping at the host boundary
        # before delegating to the underlying rules engine. This mirrors
        # the guard in GameEngine.apply_move and ensures that any client
        # attempting to apply an illegal move/phase combination fails
        # early during training and self-play.
        GameEngine._assert_phase_move_invariant(self.state, move)

        # FSM validation: validate user-provided move against phase-to-move contract.
        # In shadow mode, violations are logged but processing continues.
        # In active mode, FSMValidationError is raised for fail-fast debugging.
        if self._fsm is not None:
            self._fsm.validate_and_send(
                self.state.current_phase, move, self.state
            )

        # Enforce turn ownership at the host boundary. This prevents
        # mis-attributed moves (wrong player) from being applied or
        # recorded during self-play and training.
        if self.state.game_status == GameStatus.ACTIVE and move.player != self.state.current_player:
            raise ValueError(
                f"Move player {move.player} does not match current player {self.state.current_player}"
            )

        # Note: Move history length could be tracked here to detect any
        # additional bookkeeping moves (e.g., no_territory_action) appended
        # by the rules engine, but is currently unused.

        actor_player = self._state.current_player

        # Apply move via the canonical Python rules engine.
        # Use trace_mode=True to prevent automatic phase skipping (e.g., jumping
        # from MOVEMENT directly to TERRITORY_PROCESSING when there are no lines).
        # This ensures all bookkeeping moves (no_line_action, no_territory_action,
        # forced_elimination) are explicitly emitted and recorded for replay parity.
        if self._rules_engine is not None:
            self._state = self._rules_engine.apply_move(self.state, move)
        else:
            self._state = GameEngine.apply_move(self.state, move, trace_mode=True)

        self._move_count += 1

        # Auto-satisfy any pending phase requirements (no_*_action / FE) the
        # host must emit per RR-CANON-R075/R076. This mirrors the TS
        # orchestrator, preventing the turn from rotating without recording
        # the required bookkeeping move.
        auto_generated_moves = []
        while (
            self._state.game_status == GameStatus.ACTIVE
            and self._state.current_player == actor_player
        ):
            requirement = GameEngine.get_phase_requirement(
                self._state,
                self._state.current_player,
            )
            if requirement is None:
                break
            auto_move = GameEngine.synthesize_bookkeeping_move(
                requirement,
                self._state,
            )
            # FSM validation for auto-generated bookkeeping moves.
            if self._fsm is not None:
                self._fsm.validate_and_send(
                    self._state.current_phase, auto_move, self._state
                )
            # Apply the synthesized move and continue checking for chained
            # requirements (e.g., entering territory_processing).
            self._state = (
                self._rules_engine.apply_move(self._state, auto_move)
                if self._rules_engine is not None
                else GameEngine.apply_move(self._state, auto_move, trace_mode=True)
            )
            self._move_count += 1
            auto_generated_moves.append(auto_move)

        # If we're still in a bookkeeping-only phase for the same actor, and
        # the only legal move is a required no-op (no_line_action /
        # no_territory_action), apply it automatically to avoid rotating turn
        # order without recording the canonical phase visit. This mirrors the
        # TS TurnOrchestrator behaviour where these bookkeeping moves are
        # forced before advancing.
        while (
            self._state.game_status == GameStatus.ACTIVE
            and self._state.current_player == actor_player
            and self._state.current_phase
            in (GamePhase.LINE_PROCESSING, GamePhase.TERRITORY_PROCESSING)
            and self._force_bookkeeping_moves
        ):
            # Prefer explicit phase requirement if the core reports one.
            requirement = GameEngine.get_phase_requirement(
                self._state,
                actor_player,
            )
            forced_move = None
            if requirement is not None:
                if requirement.type in (
                    PhaseRequirementType.NO_LINE_ACTION_REQUIRED,
                    PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED,
                ):
                    forced_move = GameEngine.synthesize_bookkeeping_move(
                        requirement,
                        self._state,
                    )
            else:
                forced_moves = GameEngine.get_valid_moves(
                    self._state,
                    actor_player,
                )
                if not forced_moves:
                    # No interactive moves and no explicit requirement – synthesize the
                    # mandatory no-op for this bookkeeping phase.
                    req_type = (
                        PhaseRequirementType.NO_LINE_ACTION_REQUIRED
                        if self._state.current_phase == GamePhase.LINE_PROCESSING
                        else PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED
                    )
                    req = GameEngine.PhaseRequirement(  # type: ignore[attr-defined]
                        type=req_type,
                        player=actor_player,
                        eligible_positions=[],
                    )
                    forced_move = GameEngine.synthesize_bookkeeping_move(
                        req,
                        self._state,
                    )
                elif (
                    len(forced_moves) == 1
                    and forced_moves[0].type
                    in (MoveType.NO_LINE_ACTION, MoveType.NO_TERRITORY_ACTION)
                ):
                    forced_move = forced_moves[0]

            if forced_move is None:
                break

            # FSM validation for forced bookkeeping moves.
            if self._fsm is not None:
                self._fsm.validate_and_send(
                    self._state.current_phase, forced_move, self._state
                )

            self._state = (
                self._rules_engine.apply_move(self._state, forced_move)
                if self._rules_engine is not None
                else GameEngine.apply_move(self._state, forced_move, trace_mode=True)
            )
            self._move_count += 1
            auto_generated_moves.append(forced_move)

        terminated_by_rules = self._state.game_status != GameStatus.ACTIVE
        terminated_by_budget = self._move_count >= self.max_moves
        done = terminated_by_rules or terminated_by_budget

        reward = 0.0
        info: Dict[str, Any] = {
            "winner": self._state.winner,
            "move_count": self._move_count,
            "auto_generated_moves": auto_generated_moves,
        }

        # Log warning/error for games that hit max_moves without a winner.
        if terminated_by_budget and not terminated_by_rules:
            theoretical_max = get_theoretical_max_moves(
                self.board_type,
                self.num_players,
            )
            if self._move_count >= theoretical_max:
                # Game exceeded the theoretical maximum number of moves.
                logger.error(
                    "GAME_NON_TERMINATION_ERROR: exceeded theoretical "
                    "maximum moves without a conclusion. board_type=%s, "
                    "num_players=%d, move_count=%d, max_moves=%d, "
                    "theoretical_max=%d, game_status=%s, winner=%s",
                    self.board_type.value,
                    self.num_players,
                    self._move_count,
                    self.max_moves,
                    theoretical_max,
                    self._state.game_status.value,
                    self._state.winner,
                )
            else:
                # Hit configured max_moves but not theoretical maximum.
                logger.warning(
                    "GAME_MAX_MOVES_CUTOFF: hit max_moves without a winner. "
                    "board_type=%s, num_players=%d, move_count=%d, "
                    "max_moves=%d, theoretical_max=%d, game_status=%s, "
                    "winner=%s",
                    self.board_type.value,
                    self.num_players,
                    self._move_count,
                    self.max_moves,
                    theoretical_max,
                    self._state.game_status.value,
                    self._state.winner,
                )
            info["termination_anomaly"] = True
            info["theoretical_max_moves"] = theoretical_max

        # Terminal rewards.
        if done:
            if self.reward_on == "terminal":
                # Perspective: player who just moved is move.player.
                perspective = move.player
                if self._state.winner is None:
                    reward = 0.0
                elif self._state.winner == perspective:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                # Reuse calculate_outcome-like shaping.
                from app.training.generate_data import calculate_outcome

                reward = calculate_outcome(
                    self._state,
                    player_number=move.player,
                    depth=self._move_count,
                )

            # Canonical victory_reason and per-player stats.
            victory_reason = self._infer_canonical_victory_reason(
                terminated_by_budget_only=terminated_by_budget
                and not terminated_by_rules,
            )
            info["victory_reason"] = victory_reason
            # Raw engine-level category for debugging / compatibility.
            info["engine_victory_reason"] = infer_victory_reason(self._state)

            # Rings eliminated are keyed by causing player id as strings
            # in GameState; expose a simpler int-keyed mapping.
            rings_eliminated: Dict[int, int] = {}
            for pid_str, count in self._state.board.eliminated_rings.items():
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    continue
                rings_eliminated[pid] = count
            info["rings_eliminated"] = rings_eliminated

            # Territory spaces per player from the Player models.
            territory_spaces: Dict[int, int] = {}
            for player in self._state.players:
                territory_spaces[player.player_number] = (
                    player.territory_spaces
                )
            info["territory_spaces"] = territory_spaces

            info["moves_played"] = self._move_count
        else:
            # For non-terminal steps expose the next state's legal moves so
            # callers do not need to re-query the engine.
            info["legal_moves"] = self.legal_moves()

        return self._state, reward, done, info
