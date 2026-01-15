#!/usr/bin/env python3
"""Generate high-quality selfplay data using Gumbel MCTS.

This script generates training data using Gumbel AlphaZero-style MCTS with
Sequential Halving for sample-efficient search. Designed to produce data
for training models to exceed 2000 Elo.

Key features:
1. Gumbel MCTS with GPU acceleration for 5-50x speedup
2. MCTS policy distribution recording for soft policy targets
3. Support for all 12 board/player configurations
4. Canonical move recording for RR-CANON compliance
5. Optional TypeScript parity validation

Usage:
    # Generate 100 games on square8 2-player
    python scripts/generate_gumbel_selfplay.py \
        --board square8 \
        --num-players 2 \
        --num-games 100 \
        --output data/selfplay/gumbel_square8_2p.jsonl

    # Generate for all 12 configurations
    python scripts/generate_gumbel_selfplay.py \
        --all-configs \
        --num-games 50 \
        --output-dir data/selfplay/gumbel_all

    # With parity validation (slower but ensures correctness)
    python scripts/generate_gumbel_selfplay.py \
        --board hexagonal \
        --num-players 2 \
        --num-games 20 \
        --validate-parity
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Disable torch.compile on GH200 nodes (driver incompatibility)
# Set this before any torch imports to prevent compile() errors
if not os.environ.get('RINGRIFT_DISABLE_TORCH_COMPILE'):
    os.environ['RINGRIFT_DISABLE_TORCH_COMPILE'] = '1'

# Ensure app imports resolve
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import AIConfig, AIType, BoardType, GamePhase, GameState, GameStatus, Move
from app.rules.default_engine import DefaultRulesEngine
from app.training.env import RingRiftEnv, TrainingEnvConfig, make_env

# Optional imports for advanced features
try:
    from app.training.mcts_labeling import build_move_key, extract_mcts_quality
    HAS_MCTS_LABELING = True
except ImportError:
    HAS_MCTS_LABELING = False

try:
    from app.db.game_replay import GameReplayDB
    HAS_GAME_REPLAY_DB = True
except ImportError:
    HAS_GAME_REPLAY_DB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# All 12 board/player configurations
ALL_CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


def _emit_gumbel_selfplay_complete(
    config: "GumbelSelfplayConfig",
    games_generated: int,
    duration_seconds: float,
    success: bool = True,
    error: str = "",
) -> None:
    """Emit selfplay completion event for pipeline coordination (December 2025).

    This notifies downstream pipeline stages that Gumbel MCTS selfplay has completed.
    Uses gpu_selfplay type since Gumbel MCTS leverages GPU acceleration.

    Args:
        config: Selfplay configuration
        games_generated: Number of games successfully generated
        duration_seconds: Total duration in seconds
        success: Whether selfplay completed successfully
        error: Error message if failed
    """
    try:
        import asyncio
        import socket

        from app.coordination.selfplay_orchestrator import emit_selfplay_completion

        node_id = socket.gethostname()
        task_id = f"gumbel_selfplay_{config.board_type}_{config.num_players}p_{int(time.time())}"

        async def emit():
            return await emit_selfplay_completion(
                task_id=task_id,
                board_type=config.board_type,
                num_players=config.num_players,
                games_generated=games_generated,
                success=success,
                node_id=node_id,
                selfplay_type="gpu_selfplay",  # Gumbel MCTS uses GPU acceleration
                error=error,
            )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(emit())
        except RuntimeError:
            asyncio.run(emit())

        logger.debug(
            f"Emitted GPU_SELFPLAY_COMPLETE: {games_generated} games, "
            f"{duration_seconds:.2f}s, task_id={task_id}"
        )
    except ImportError:
        pass  # SelfplayOrchestrator not available
    except Exception as e:
        logger.debug(f"Failed to emit GPU_SELFPLAY_COMPLETE: {e}")


@dataclass
class GumbelSelfplayConfig:
    """Configuration for Gumbel MCTS selfplay generation."""
    board_type: str = "square8"
    num_players: int = 2
    num_games: int = 100
    simulation_budget: int = 800  # Increased from 150 for quality data (AlphaZero minimum)
    think_time_ms: int = 0  # 0 = use simulation budget, >0 = time limit
    temperature: float = 1.0  # Policy temperature for exploration
    temperature_threshold: int = 30  # Move after which to use greedy (temp=0)
    use_temperature_curriculum: bool = True  # Enable move-based temperature decay
    max_moves: int = 0  # 0 = auto-derive from board/players
    output_path: str = ""
    output_dir: str = ""
    db_path: str = ""  # Optional: save to GameReplayDB
    validate_parity: bool = False
    seed: int = 0
    verbose: bool = False
    # Neural net settings
    nn_model_id: str = ""  # Empty = use default for board
    use_gpu: bool = True
    allow_fresh_weights: bool = False  # Allow random weights if no checkpoint
    # GPU tree provides 10-20x speedup for MCTS search
    use_gpu_tree: bool = True  # RR-GPU-TREE-001: defensive validation added
    # Architecture version for model selection (v2, v4, v5, v5-heavy, etc.)
    # Jan 5, 2026: Added for architecture selection feedback loop
    # Jan 12, 2026: Changed default to "v2" - the NeuralNetAI loader currently
    # expects HexNeuralNet_v2 architecture. Using v5-heavy checkpoints (v5.1.0-hex)
    # triggers VersionMismatchError since the weights are incompatible.
    model_version: str = "v2"

    def get_temperature_for_move(self, move_number: int) -> float:
        """Get temperature for a specific move number.

        Implements AlphaZero-style temperature scheduling:
        - Full exploration (temp=1.0) for early moves
        - Greedy selection (temp→0) after threshold

        Returns:
            Temperature value for the move (0.0-1.0+)
        """
        if not self.use_temperature_curriculum:
            return self.temperature

        if move_number >= self.temperature_threshold:
            return 0.1  # Near-greedy after threshold

        # Linear decay from temperature to 0.1 over threshold moves
        progress = move_number / self.temperature_threshold
        return self.temperature * (1.0 - 0.9 * progress)


@dataclass
class GameResult:
    """Result of a single selfplay game."""
    game_id: str
    board_type: str
    num_players: int
    winner: int | None
    status: str
    num_moves: int
    duration_ms: float
    moves: list[dict[str, Any]] = field(default_factory=list)
    initial_state: GameState | None = None
    final_state: GameState | None = None
    parity_ok: bool = True
    parity_error: str = ""


def parse_board_type(board_str: str) -> BoardType:
    """Parse board type string to enum."""
    board_str = board_str.lower()
    if "square8" in board_str or "sq8" in board_str:
        return BoardType.SQUARE8
    elif "square19" in board_str or "sq19" in board_str:
        return BoardType.SQUARE19
    elif "hex8" in board_str:
        return BoardType.HEX8
    elif "hex" in board_str:
        return BoardType.HEXAGONAL
    return BoardType.SQUARE8


def get_max_moves(board_type: str, num_players: int) -> int:
    """Get reasonable max moves for board/player combo.

    Note: These limits are for script-side move counting. Each script move
    may generate multiple env moves (due to auto-bookkeeping), so these limits
    should be ~2-3x the env's theoretical max to account for the multiplier.

    For Gumbel games specifically, the env.max_moves is set to 2x theoretical
    max, so these script limits provide additional headroom.
    """
    base = {
        "square8": 800,   # 2x+ theoretical (600) for Gumbel's slower progression
        "square19": 3000,
        "hex8": 800,      # 2x+ theoretical
        "hexagonal": 5000,
    }
    # More players = potentially longer games
    multiplier = 1.0 + (num_players - 2) * 0.25
    return int(base.get(board_type, 800) * multiplier)


# Stalemate detection parameters
STALEMATE_THRESHOLD = 30  # Consecutive moves without S-invariant progress
EXPLORATION_NOISE_PROB = 0.05  # 5% chance of random move for exploration


def create_gumbel_ai(
    player: int,
    board_type: BoardType,
    config: GumbelSelfplayConfig,
) -> Any:
    """Create a GumbelMCTSAI instance for selfplay."""
    from app.ai.gumbel_mcts_ai import GumbelMCTSAI

    ai_config = AIConfig(
        difficulty=9,  # Master level (Gumbel MCTS)
        randomness=0.0,
        think_time_ms=config.think_time_ms if config.think_time_ms > 0 else None,
        use_neural_net=True,
        gumbel_simulation_budget=config.simulation_budget,
        allow_fresh_weights=config.allow_fresh_weights,
        use_gpu_tree=config.use_gpu_tree,  # 170x speedup with GPU tensor tree
    )

    if config.nn_model_id:
        ai_config.nn_model_id = config.nn_model_id

    return GumbelMCTSAI(player, ai_config, board_type)


def serialize_move(
    move: Move,
    mcts_policy: dict[str, float] | None = None,
    value: float | None = None,
    phase: str | None = None,
    move_number: int | None = None,
    search_stats: dict | None = None,
) -> dict[str, Any]:
    """Serialize a Move to JSON-compatible dict with optional MCTS info."""
    move_data = move.model_dump(by_alias=True, exclude_none=True, mode="json")
    if phase and "phase" not in move_data:
        move_data["phase"] = phase

    # Add move_number for proper training sample ordering (critical for NNUE)
    if move_number is not None:
        move_data["moveNumber"] = move_number

    # Add MCTS policy distribution for training
    if mcts_policy:
        move_data["mcts_policy"] = mcts_policy

    # Add value estimate for training
    if value is not None:
        move_data["value"] = value

    # Add rich search statistics for auxiliary training (Q-values, uncertainty, etc.)
    if search_stats:
        move_data["search_stats"] = search_stats

    return move_data


def extract_policy_from_gumbel(ai: Any, legal_moves: list[Move]) -> dict[str, float]:
    """Extract policy distribution from Gumbel MCTS search.

    Returns a dict mapping move keys to visit fractions.
    """
    if not hasattr(ai, '_last_search_actions') or ai._last_search_actions is None:
        return {}

    total_visits = sum(a.visit_count for a in ai._last_search_actions)
    if total_visits == 0:
        return {}

    policy = {}
    for action in ai._last_search_actions:
        if action.visit_count > 0:
            # Build move key for this action
            move = action.move
            key = move.type.value
            if move.from_pos:
                key += f"_{move.from_pos.x},{move.from_pos.y}"
            if move.to:
                key += f"_{move.to.x},{move.to.y}"
            policy[key] = action.visit_count / total_visits

    return policy


def generate_game(
    env: RingRiftEnv,
    ai_players: dict[int, Any],
    config: GumbelSelfplayConfig,
    game_idx: int,
) -> GameResult:
    """Generate a single selfplay game with Gumbel MCTS."""
    from app.rules.core import compute_progress_snapshot

    game_id = str(uuid.uuid4())
    start_time = time.time()

    # Derive unique RNG seed per game for reproducibility and diversity
    base_seed = config.seed if config.seed else random.randint(0, 0xFFFFFFFF)
    game_seed = (base_seed + game_idx * 1_000_003) & 0xFFFFFFFF

    # Reset AI players with unique per-game, per-player seeds
    for player, ai in ai_players.items():
        player_seed = (game_seed + player * 97_911) & 0xFFFFFFFF
        if hasattr(ai, 'reset_for_new_game'):
            ai.reset_for_new_game(rng_seed=player_seed)

    # Reset environment with game seed
    state = env.reset(seed=game_seed)
    # Use .copy() for Pydantic v1 compatibility
    initial_state = state.copy(deep=True) if hasattr(state, "copy") else state

    moves_data = []
    max_moves = config.max_moves or get_max_moves(config.board_type, config.num_players)

    # S-invariant tracking for non-termination diagnosis and violation detection
    s_history = []
    initial_snapshot = compute_progress_snapshot(state)
    s_history.append(initial_snapshot['S'])
    prev_s = initial_snapshot['S']
    moves_without_s_progress = 0
    s_violations = []  # Track any S-invariant violations
    stalemate_detected = False

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player
        ai = ai_players.get(current_player)

        if ai is None:
            logger.error(f"No AI for player {current_player}")
            break

        # Update AI's player number for current player (critical for correct move generation)
        ai.player_number = current_player

        # Get legal moves
        legal_moves = env.legal_moves()
        if not legal_moves:
            logger.warning(f"No legal moves at move {move_count}")
            break

        # Exploration noise: occasionally pick random move to prevent ultra-defensive play
        use_random = random.random() < EXPLORATION_NOISE_PROB
        if use_random and len(legal_moves) > 1:
            selected_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
            mcts_policy = {}  # No MCTS policy for random moves
            value = None
            search_stats = None  # No search stats for random moves
        else:
            # AI selects move (Gumbel MCTS search)
            selected_move = ai.select_move(state)
            if selected_move is None:
                logger.warning(f"AI returned None move at move {move_count}")
                break

            # Validate selected move is legal; fall back to random if not
            if selected_move not in legal_moves:
                logger.warning(f"AI selected invalid move at move {move_count}, falling back to random")
                selected_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

            # Extract MCTS policy distribution for training
            mcts_policy = extract_policy_from_gumbel(ai, legal_moves)

            # Extract value estimate if available
            value = None
            if hasattr(ai, '_last_root_value'):
                value = ai._last_root_value

            # Extract rich search statistics (Q-values, uncertainty, etc.)
            search_stats = None
            if hasattr(ai, 'get_search_stats'):
                search_stats = ai.get_search_stats()

        # Serialize move with MCTS info
        phase = (
            state.current_phase.value
            if hasattr(state.current_phase, "value")
            else str(state.current_phase)
        )
        # move_count is 0-indexed during loop, but moveNumber should be 1-indexed for consistency
        move_data = serialize_move(
            selected_move, mcts_policy, value, phase=phase, move_number=move_count + 1,
            search_stats=search_stats
        )
        moves_data.append(move_data)

        # Apply move
        state, _, done, _ = env.step(selected_move)
        move_count += 1

        # Track S-invariant progression and detect violations
        snapshot = compute_progress_snapshot(state)
        current_s = snapshot['S']
        s_history.append(current_s)

        # S-invariant violation detection: S should never decrease
        if current_s < prev_s:
            violation_info = {
                'move': move_count,
                'prev_s': prev_s,
                'current_s': current_s,
                'delta': current_s - prev_s,
                'phase': state.current_phase.value,
                'move_type': selected_move.type.value if hasattr(selected_move.type, 'value') else str(selected_move.type),
            }
            s_violations.append(violation_info)
            logger.error(
                f"S-INVARIANT VIOLATION at move {move_count}: S decreased from {prev_s} to {current_s} "
                f"(delta={current_s - prev_s}), phase={state.current_phase.value}, "
                f"move_type={violation_info['move_type']}"
            )

        # Stalemate detection: if S hasn't increased for STALEMATE_THRESHOLD moves
        if current_s == prev_s:
            moves_without_s_progress += 1
            if moves_without_s_progress >= STALEMATE_THRESHOLD:
                stalemate_detected = True
                logger.warning(
                    f"STALEMATE_DETECTED at move {move_count}: S={current_s} unchanged for "
                    f"{moves_without_s_progress} consecutive moves. Forcing game end."
                )
                break
        else:
            moves_without_s_progress = 0

        prev_s = current_s

        if done:
            break

        if config.verbose and move_count % 50 == 0:
            logger.info(f"Game {game_idx}: move {move_count}")

    # Determine winner
    winner = None
    termination_reason = None

    if state.game_status == GameStatus.COMPLETED:
        winner = state.winner
        termination_reason = "victory"
    elif stalemate_detected:
        # Stalemate: determine winner by tiebreaker (most rings remaining)
        termination_reason = "stalemate"
        rings_per_player = {}
        for p in state.players:
            rings_on_board = sum(1 for s in state.board.stacks.values() for r in s.rings if r == p.player_number)
            rings_per_player[p.player_number] = rings_on_board + p.rings_in_hand

        # Winner is player with most rings
        if rings_per_player:
            winner = max(rings_per_player, key=rings_per_player.get)
            logger.info(
                f"Stalemate resolved: winner={winner} (rings={rings_per_player})"
            )
    else:
        # Game did not complete - log diagnostic info
        termination_reason = "timeout"
        final_snapshot = compute_progress_snapshot(state)
        s_delta = s_history[-1] - s_history[0] if len(s_history) > 1 else 0
        s_rate = s_delta / move_count if move_count > 0 else 0

        # Count rings per player
        rings_per_player = {}
        for p in state.players:
            rings_on_board = sum(1 for s in state.board.stacks.values() for r in s.rings if r == p.player_number)
            rings_per_player[p.player_number] = rings_on_board + p.rings_in_hand

        logger.warning(
            f"GAME_NON_TERMINATION_DIAGNOSTIC: game={game_idx}, seed={game_seed}, "
            f"moves={move_count}, status={state.game_status.value}, "
            f"S={final_snapshot['S']} (m={final_snapshot['markers']}, "
            f"c={final_snapshot['collapsed']}, e={final_snapshot['eliminated']}), "
            f"S_rate={s_rate:.3f}/move, rings={rings_per_player}, "
            f"phase={state.current_phase.value}, "
            f"s_violations={len(s_violations)}, stalemate={stalemate_detected}"
        )

        # Log any S-invariant violations
        if s_violations:
            logger.error(
                f"S-INVARIANT VIOLATIONS in game {game_idx}: {len(s_violations)} violations detected. "
                f"First violation: {s_violations[0]}"
            )

    duration_ms = (time.time() - start_time) * 1000

    return GameResult(
        game_id=game_id,
        board_type=config.board_type,
        num_players=config.num_players,
        winner=winner,
        status=state.game_status.value,
        num_moves=move_count,
        duration_ms=duration_ms,
        moves=moves_data,
        initial_state=initial_state,
        final_state=state,
    )


def run_parity_validation(result: GameResult, config: GumbelSelfplayConfig) -> bool:
    """Validate game against TypeScript engine for parity.

    Returns True if parity check passes.
    """
    try:
        from app.ai.gpu_canonical_export import validate_game_parity
        errors = validate_game_parity(
            game_id=result.game_id,
            board_type=config.board_type,
            num_players=config.num_players,
            moves=[m for m in result.moves],
        )
        if errors:
            result.parity_ok = False
            result.parity_error = "; ".join(errors[:3])  # First 3 errors
            return False
        return True
    except ImportError:
        logger.warning("Parity validation not available")
        return True
    except Exception as e:
        result.parity_ok = False
        result.parity_error = str(e)
        return False


def save_game_to_jsonl(result: GameResult, output_path: Path) -> None:
    """Append game result to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize initial_state for NPZ conversion compatibility
    initial_state_dict = None
    if result.initial_state is not None:
        initial_state_dict = result.initial_state.model_dump(
            by_alias=True, exclude_none=True, mode="json"
        )

    game_data = {
        "game_id": result.game_id,
        "board_type": result.board_type,
        "num_players": result.num_players,
        "winner": result.winner,
        "status": result.status,
        "num_moves": result.num_moves,
        "duration_ms": result.duration_ms,
        "moves": result.moves,
        "parity_ok": result.parity_ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "initial_state": initial_state_dict,
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(game_data) + "\n")


def save_game_to_db(result: GameResult, db: Any) -> None:
    """Save game to GameReplayDB with soft policy targets (move_probs).
    
    This function saves the game to the database using incremental recording
    to properly store move_probs (MCTS policy distributions) for each move.
    """
    if not HAS_GAME_REPLAY_DB:
        return

    if result.initial_state is None or result.final_state is None:
        logger.warning("Missing initial/final state; skipping DB write")
        return

    from app.db.game_replay import GameReplayDB
    from app.db.unified_recording import GameRecorder
    from app.game_engine import GameEngine
    from app.models import Move as MoveModel

    metadata = {
        "source": "gumbel_selfplay",
        "board_type": result.board_type,
        "num_players": result.num_players,
        "winner": result.winner,
        "engine_mode": "gumbel-mcts",
    }

    # Use incremental recording to properly store move_probs and search_stats
    with GameRecorder(db, result.initial_state, result.game_id) as recorder:
        state = result.initial_state
        for idx, move_data in enumerate(result.moves):
            # Extract move_probs (MCTS policy) from the move data
            move_probs = move_data.get("mcts_policy")

            # Extract rich search statistics for auxiliary training
            search_stats = move_data.get("search_stats")

            # Create Move object from the serialized data (excluding extra fields)
            move_dict = {k: v for k, v in move_data.items()
                         if k not in ("mcts_policy", "value", "moveNumber", "search_stats")}
            move_dict.setdefault("id", f"{result.game_id}:{idx}")
            move = MoveModel.model_validate(move_dict)

            # Apply move to get state_after
            state_after = GameEngine.apply_move(state, move, trace_mode=True)

            # Record the move with soft policy targets and search stats
            recorder.add_move(
                move,
                state_after=state_after,
                state_before=state,
                move_probs=move_probs,
                search_stats=search_stats,
            )

            state = state_after

        # Finalize with the final state and metadata
        recorder.finalize(result.final_state, metadata)


def run_selfplay(config: GumbelSelfplayConfig) -> list[GameResult]:
    """Run Gumbel MCTS selfplay for specified configuration."""
    board_type_enum = parse_board_type(config.board_type)

    logger.info(
        f"Starting Gumbel MCTS selfplay: {config.board_type} {config.num_players}P, "
        f"{config.num_games} games, budget={config.simulation_budget}"
    )

    # Create environment with increased max_moves for Gumbel games.
    # Gumbel MCTS plays more defensively than heuristic AI, requiring more
    # moves to reach a decisive outcome. The default theoretical max (600 for
    # square8 2p) is based on heuristic games; Gumbel games may need 50-100%
    # more moves due to slower S-invariant progression.
    from app.training.env import get_theoretical_max_moves
    theoretical_max = get_theoretical_max_moves(board_type_enum, config.num_players)
    gumbel_max_moves = int(theoretical_max * 1.5)  # 50% increase for Gumbel

    env_config = TrainingEnvConfig(
        board_type=board_type_enum,
        num_players=config.num_players,
        max_moves=gumbel_max_moves,
    )
    env = make_env(env_config)
    logger.info(f"Env max_moves set to {gumbel_max_moves} (1.5x theoretical max {theoretical_max})")

    # Create AI players (same AI for all players in selfplay)
    ai_players = {}
    for player in range(1, config.num_players + 1):
        ai_players[player] = create_gumbel_ai(player, board_type_enum, config)

    # Determine output path
    if config.output_path:
        output_path = Path(config.output_path)
    elif config.output_dir:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"gumbel_{config.board_type}_{config.num_players}p.jsonl"
    else:
        output_path = Path(f"data/selfplay/gumbel_{config.board_type}_{config.num_players}p.jsonl")

    # Optional: GameReplayDB
    # Note: enforce_canonical_history=False allows recording moves that may not
    # perfectly match phase expectations (needed for Gumbel MCTS which may have
    # slight state tracking differences)
    db = None
    if config.db_path and HAS_GAME_REPLAY_DB:
        db = GameReplayDB(config.db_path, enforce_canonical_history=False)

    results = []
    parity_failures = 0

    for game_idx in range(config.num_games):
        try:
            result = generate_game(env, ai_players, config, game_idx)

            # Optional parity validation
            if config.validate_parity:
                if not run_parity_validation(result, config):
                    parity_failures += 1
                    logger.warning(f"Game {game_idx} failed parity: {result.parity_error}")

            # Jan 14, 2026: Save games even if they hit max_moves without winner.
            # These games contain valid training data (MCTS policies, value estimates).
            # Skipping them was causing 0 games to be saved on GH200 nodes.
            if result.winner is None and result.status != "completed":
                logger.warning(
                    f"Game {game_idx} did not complete naturally: status={result.status}, "
                    f"moves={result.num_moves}. Saving anyway (valid training data)."
                )
                # Continue to save - don't skip

            # Save game
            save_game_to_jsonl(result, output_path)
            if db:
                save_game_to_db(result, db)

            results.append(result)

            if (game_idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {game_idx + 1}/{config.num_games} games, "
                    f"avg moves: {sum(r.num_moves for r in results) / len(results):.1f}"
                )

        except Exception as e:
            # Jan 15, 2026: Use logger.exception() for safe exception logging
            # (avoids RingRiftError.__str__() crash when error code is misconfigured)
            logger.exception(f"Game {game_idx} failed with {type(e).__name__}")
            continue

    # Summary
    if results:
        avg_moves = sum(r.num_moves for r in results) / len(results)
        avg_duration = sum(r.duration_ms for r in results) / len(results)
        total_duration_sec = sum(r.duration_ms for r in results) / 1000.0
        winners = [r.winner for r in results if r.winner is not None]
        win_dist = {p: winners.count(p) for p in range(1, config.num_players + 1)}

        logger.info(f"\nSummary for {config.board_type} {config.num_players}P:")
        logger.info(f"  Games: {len(results)}/{config.num_games}")
        logger.info(f"  Avg moves: {avg_moves:.1f}")
        logger.info(f"  Avg duration: {avg_duration:.0f}ms")
        logger.info(f"  Win distribution: {win_dist}")
        if config.validate_parity:
            logger.info(f"  Parity failures: {parity_failures}")
        logger.info(f"  Output: {output_path}")

        # Emit selfplay completion event for pipeline coordination (December 2025)
        _emit_gumbel_selfplay_complete(
            config=config,
            games_generated=len(results),
            duration_seconds=total_duration_sec,
            success=True,
        )

    return results


def run_all_configs(config: GumbelSelfplayConfig) -> dict[str, list[GameResult]]:
    """Run Gumbel MCTS selfplay for all 12 board/player configurations."""
    all_results = {}

    for board_type, num_players in ALL_CONFIGS:
        # Create config for this combo
        combo_config = GumbelSelfplayConfig(
            board_type=board_type,
            num_players=num_players,
            num_games=config.num_games,
            simulation_budget=config.simulation_budget,
            think_time_ms=config.think_time_ms,
            temperature=config.temperature,
            output_dir=config.output_dir or "data/selfplay/gumbel_all",
            validate_parity=config.validate_parity,
            seed=config.seed,
            verbose=config.verbose,
            nn_model_id=config.nn_model_id,
            use_gpu=config.use_gpu,
            allow_fresh_weights=config.allow_fresh_weights,
        )

        try:
            results = run_selfplay(combo_config)
            all_results[f"{board_type}_{num_players}p"] = results
        except Exception as e:
            logger.error(f"Failed {board_type} {num_players}P: {e}")
            all_results[f"{board_type}_{num_players}p"] = []

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Generate selfplay data using Gumbel MCTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--board", "-b",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=100,
        help="Number of games to generate (default: 100)",
    )
    parser.add_argument(
        "--simulation-budget",
        type=int,
        default=150,
        help="Gumbel MCTS simulation budget per move (default: 150)",
    )
    parser.add_argument(
        "--think-time-ms",
        type=int,
        default=0,
        help="Think time limit in ms (0 = use simulation budget)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (for --all-configs)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="",
        help="Optional GameReplayDB path for canonical storage",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run for all 12 board/player configurations",
    )
    parser.add_argument(
        "--validate-parity",
        action="store_true",
        help="Validate games against TypeScript engine",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (0 = random)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Neural network model ID to use",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v2",
        help="Architecture version (v2, v4, v5, v5-heavy, etc.) for model selection. "
             "Default v2 to match NeuralNetAI's HexNeuralNet_v2 architecture.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--allow-fresh-weights",
        action="store_true",
        help="Allow random/fresh weights if no trained model checkpoint exists",
    )
    parser.add_argument(
        "--use-gpu-tree",
        action="store_true",
        default=True,
        help="Use GPU-accelerated tensor tree MCTS (170x faster, default: enabled)",
    )
    parser.add_argument(
        "--no-gpu-tree",
        action="store_true",
        help="Disable GPU tensor tree MCTS and use CPU implementation",
    )

    args = parser.parse_args()

    # Jan 5, 2026: Resolve model path using get_model_for_config() for architecture selection
    # If no explicit model_id provided, use version-specific model from selector
    nn_model_path = args.model_id
    if not nn_model_path and args.model_version:
        try:
            from app.training.selfplay_model_selector import get_model_for_config
            resolved_path = get_model_for_config(
                args.board,
                args.num_players,
                prefer_nnue=False,
                model_version=args.model_version,
            )
            if resolved_path:
                nn_model_path = str(resolved_path)
                logger.info(f"Using model version {args.model_version}: {nn_model_path}")
        except ImportError:
            logger.debug("selfplay_model_selector not available, using default model resolution")

    config = GumbelSelfplayConfig(
        board_type=args.board,
        num_players=args.num_players,
        num_games=args.num_games,
        simulation_budget=args.simulation_budget,
        think_time_ms=args.think_time_ms,
        output_path=args.output,
        output_dir=args.output_dir,
        db_path=args.db,
        validate_parity=args.validate_parity,
        seed=args.seed,
        verbose=args.verbose,
        nn_model_id=nn_model_path,
        use_gpu=not args.no_gpu,
        allow_fresh_weights=args.allow_fresh_weights,
        use_gpu_tree=args.use_gpu_tree and not args.no_gpu_tree,  # 170x faster
        model_version=args.model_version,
    )

    if args.all_configs:
        results = run_all_configs(config)
        total_games = sum(len(r) for r in results.values())
        logger.info(f"\nTotal: {total_games} games across {len(results)} configurations")
    else:
        run_selfplay(config)


if __name__ == "__main__":
    main()
