#!/usr/bin/env python
"""Run (parallel) AI strength evaluation tournament for difficulty tiers.

This script orchestrates AI-vs-AI matches across multiple hosts to empirically
measure the relative strength of different AI configurations. It uses the
canonical difficulty ladder configuration and runs hundreds of games in
parallel on a single machine.

Note: Despite the filename, this harness currently runs locally with a thread
pool. For SSH-based multi-host distribution, see
`scripts/run_ssh_distributed_tournament.py`.

Features:
- Supports canonical difficulty tiers (D1-D10)
- Calculates Elo ratings from match results (deterministic replay for reporting)
- Reports per-matchup win/loss/draw + Wilson intervals (decisive games)
- Generates comprehensive strength reports
- Fault-tolerant with automatic retry

Usage:
    # Full tournament across all difficulty tiers
    python scripts/run_distributed_tournament.py --games-per-matchup 50

    # Quick validation run
    python scripts/run_distributed_tournament.py --games-per-matchup 10 --tiers D1,D2,D3

    # Resume from previous run
    python scripts/run_distributed_tournament.py --resume results/tournament_20251211.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# JSON Serialization Helpers
# ============================================================================

class GameRecordEncoder(json.JSONEncoder):
    """Custom JSON encoder for game records with non-serializable types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "value"):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return super().default(obj)


def serialize_game_state(state: Any) -> dict[str, Any]:
    """Serialize a GameState to a JSON-compatible dict."""
    if hasattr(state, "model_dump"):
        return state.model_dump(mode="json")
    if hasattr(state, "dict"):
        raw = state.dict()
        # Round-trip through JSON to ensure all nested objects are serializable
        return json.loads(json.dumps(raw, cls=GameRecordEncoder))
    return {}

# Add parent to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.descent_ai import DescentAI
from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.mcts_ai import MCTSAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.random_ai import RandomAI
from app.config.ladder_config import get_ladder_tier_config
from app.config.thresholds import INITIAL_ELO_RATING
from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType, GameStatus
from app.training.initial_state import create_initial_state
from app.training.significance import wilson_score_interval
from scripts.lib.logging_config import setup_script_logging
from scripts.lib.resilience import exponential_backoff_delay

logger = setup_script_logging("run_distributed_tournament")

# Timeout handling constants
DEFAULT_MATCHUP_TIMEOUT = 600  # 10 minutes per matchup
DEFAULT_TOURNAMENT_TIMEOUT = 7200  # 2 hours for entire tournament
HEARTBEAT_INTERVAL = 60  # Log progress every 60 seconds


# ============================================================================
# Device Detection
# ============================================================================

def get_compute_device() -> str:
    """Detect available compute device: cuda, mps, or cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_safe_max_workers(requested_workers: int) -> tuple[int, str | None]:
    """Get safe number of workers based on device.

    MPS (Apple Silicon) has threading issues with PyTorch - multiple threads
    accessing the MPS context can cause deadlocks. CUDA handles threading fine.

    Returns:
        Tuple of (safe_worker_count, warning_message_or_none)
    """
    device = get_compute_device()

    if device == "mps" and requested_workers > 1:
        return 1, (
            f"MPS device detected - limiting workers to 1 (requested: {requested_workers}). "
            "MPS has threading limitations with PyTorch. Use CUDA for parallel execution."
        )

    return requested_workers, None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MatchResult:
    """Result of a single game between two AI configurations."""
    tier_a: str
    tier_b: str
    winner: int | None  # 1 for A, 2 for B, None for draw
    game_length: int
    duration_sec: float
    worker: str
    game_id: str
    timestamp: str
    seed: int | None = None
    game_index: int | None = None
    # Optional full game record for training data export
    game_record: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "tier_a": self.tier_a,
            "tier_b": self.tier_b,
            "winner": self.winner,
            "game_length": self.game_length,
            "duration_sec": self.duration_sec,
            "worker": self.worker,
            "game_id": self.game_id,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "game_index": self.game_index,
        }
        # Don't include game_record in basic dict (it's large)
        return result

    def to_training_record(self) -> dict[str, Any] | None:
        """Return full game record for training data export."""
        return self.game_record


@dataclass
class TierStats:
    """Statistics for a single difficulty tier."""
    tier: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_played: int = 0
    elo: float = INITIAL_ELO_RATING

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games_played": self.games_played,
            "win_rate": self.win_rate,
            "elo": self.elo,
        }


@dataclass
class TournamentState:
    """Full tournament state for checkpointing and resumption."""
    tournament_id: str
    started_at: str
    board_type: str
    games_per_matchup: int
    tiers: list[str]
    base_seed: int = 1
    think_time_scale: float = 1.0
    nn_model_id: str | None = None
    matches: list[MatchResult] = field(default_factory=list)
    tier_stats: dict[str, TierStats] = field(default_factory=dict)
    completed_matchups: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tournament_id": self.tournament_id,
            "started_at": self.started_at,
            "board_type": self.board_type,
            "games_per_matchup": self.games_per_matchup,
            "tiers": self.tiers,
            "base_seed": self.base_seed,
            "think_time_scale": self.think_time_scale,
            "nn_model_id": self.nn_model_id,
            "matches": [m.to_dict() for m in self.matches],
            "tier_stats": {k: v.to_dict() for k, v in self.tier_stats.items()},
            "completed_matchups": self.completed_matchups,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TournamentState:
        state = cls(
            tournament_id=data["tournament_id"],
            started_at=data["started_at"],
            board_type=data["board_type"],
            games_per_matchup=data["games_per_matchup"],
            tiers=data["tiers"],
            base_seed=int(data.get("base_seed", 1)),
            think_time_scale=float(data.get("think_time_scale", 1.0)),
            nn_model_id=data.get("nn_model_id"),
            completed_matchups=[tuple(m) for m in data.get("completed_matchups", [])],
        )
        state.matches = [
            MatchResult(**m) for m in data.get("matches", [])
        ]
        for tier, stats in data.get("tier_stats", {}).items():
            # Filter out computed properties like 'win_rate' that aren't constructor args
            filtered_stats = {k: v for k, v in stats.items() if k != 'win_rate'}
            state.tier_stats[tier] = TierStats(**filtered_stats)
        return state


# ============================================================================
# Elo Rating System
# ============================================================================

# Import unified Elo database for persistent tracking
try:
    from app.training.elo_service import get_elo_service
    ELO_SERVICE_AVAILABLE = True
except ImportError:
    ELO_SERVICE_AVAILABLE = False

try:
    from app.tournament import get_elo_database
    UNIFIED_ELO_AVAILABLE = True
except ImportError:
    UNIFIED_ELO_AVAILABLE = False


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Update Elo ratings after a match (local calculation only)."""
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)

    return new_a, new_b


def _tier_to_participant_id(tier: str, board_type: str, num_players: int) -> str:
    """Convert tier name to a descriptive participant ID.

    This ensures tier-based participants have meaningful IDs that include
    context about what AI type they represent (random, heuristic, mcts).

    Args:
        tier: Tier name like "D1", "D2", etc.
        board_type: Board type for config context
        num_players: Number of players for config context

    Returns:
        Descriptive participant ID like "tier_D1_random_sq8_2p"
    """
    tier_upper = tier.upper().strip()
    difficulty = int(tier_upper[1:]) if tier_upper.startswith("D") else 1

    # Map difficulty to AI type description
    if difficulty == 1:
        ai_desc = "random"
    elif difficulty <= 6:
        ai_desc = f"heuristic_d{difficulty}"
    elif difficulty <= 8:
        ai_desc = f"mcts_d{difficulty}"
    else:
        ai_desc = f"gumbel_d{difficulty}"

    # Create descriptive ID
    config = f"{board_type}_{num_players}p"
    return f"tier_{tier_upper}_{ai_desc}_{config}"


def persist_match_to_unified_elo(
    tier_a: str,
    tier_b: str,
    winner: int | None,
    board_type: str,
    num_players: int,
    tournament_id: str,
    game_length: int = 0,
    duration_sec: float = 0.0,
) -> None:
    """Persist match result to unified Elo database."""
    # Convert tier names to descriptive participant IDs
    participant_a = _tier_to_participant_id(tier_a, board_type, num_players)
    participant_b = _tier_to_participant_id(tier_b, board_type, num_players)
    winner_id = participant_a if winner == 1 else participant_b if winner == 2 else None

    if ELO_SERVICE_AVAILABLE:
        try:
            svc = get_elo_service()
            svc.record_match(
                participant_a,
                participant_b,
                winner=winner_id,
                board_type=board_type,
                num_players=num_players,
                game_length=game_length,
                duration_sec=duration_sec,
                tournament_id=tournament_id,
            )
            return
        except Exception as e:
            logger.warning(f"Failed to persist match to Elo service: {e}")

    if not UNIFIED_ELO_AVAILABLE:
        return

    try:
        db = get_elo_database()

        # Determine rankings based on winner
        if winner == 1:
            rankings = [0, 1]  # participant_a won
        elif winner == 2:
            rankings = [1, 0]  # participant_b won
        else:
            rankings = [0, 0]  # draw

        db.record_match_and_update(
            participant_ids=[participant_a, participant_b],
            rankings=rankings,
            board_type=board_type,
            num_players=num_players,
            tournament_id=tournament_id,
            game_length=game_length,
            duration_sec=duration_sec,
        )
    except Exception as e:
        logger.warning(f"Failed to persist match to unified Elo: {e}")


# ============================================================================
# Game Runner
# ============================================================================

AI_CLASSES = {
    AIType.RANDOM: RandomAI,
    AIType.HEURISTIC: HeuristicAI,
    AIType.MINIMAX: MinimaxAI,
    AIType.MCTS: MCTSAI,
    AIType.DESCENT: DescentAI,
    AIType.GUMBEL_MCTS: GumbelMCTSAI,
}

def _tier_to_difficulty(tier: str) -> int:
    cleaned = tier.strip().upper()
    if cleaned.startswith("D"):
        cleaned = cleaned[1:]
    if not cleaned.isdigit():
        raise ValueError(f"Invalid tier label: {tier!r}")
    return int(cleaned)


def _scaled_think_time_ms(think_time_ms: int, scale: float) -> int:
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        factor = 1.0
    if factor <= 0.0:
        factor = 0.0
    return max(0, round(think_time_ms * factor))


def create_ai_for_tier(
    tier: str,
    player_number: int,
    seed: int,
    *,
    board_type: BoardType,
    num_players: int,
    think_time_scale: float = 1.0,
    nn_model_id: str | None = None,
) -> Any:
    """Create an AI instance for the given difficulty tier.

    Uses `LadderTierConfig` (board- and player-count-aware) for tiers >= 2.
    Difficulty 1 is a fixed random baseline.
    """
    difficulty = _tier_to_difficulty(tier)

    if difficulty == 1:
        ai_type = AIType.RANDOM
        randomness = 0.5
        think_time_ms = 150
        use_neural_net = False
        heuristic_profile_id: str | None = None
        ladder_model_id: str | None = None
    else:
        ladder = get_ladder_tier_config(difficulty, board_type, num_players)
        ai_type = ladder.ai_type
        randomness = ladder.randomness
        think_time_ms = ladder.think_time_ms
        use_neural_net = bool(ladder.use_neural_net)
        heuristic_profile_id = ladder.heuristic_profile_id
        ladder_model_id = ladder.model_id

    scaled_think_time = _scaled_think_time_ms(think_time_ms, think_time_scale)

    # CNN policy/value nets are only used by MCTS/Descent. Minimax's NNUE
    # evaluator selects checkpoints via its own board-aware default path.
    effective_nn_model_id: str | None = None
    if use_neural_net and ai_type in {AIType.MCTS, AIType.DESCENT}:
        effective_nn_model_id = nn_model_id or ladder_model_id

    config = AIConfig(
        difficulty=difficulty,
        randomness=randomness,
        think_time=scaled_think_time,
        rng_seed=seed,
        heuristic_profile_id=heuristic_profile_id,
        use_neural_net=use_neural_net,
        nn_model_id=effective_nn_model_id,
        allow_fresh_weights=False,
    )

    ai_class = AI_CLASSES[ai_type]
    return ai_class(player_number, config)


def run_single_game(
    tier_a: str,
    tier_b: str,
    board_type: BoardType,
    seed: int,
    max_moves: int = 10000,
    worker_name: str = "local",
    game_index: int | None = None,
    think_time_scale: float = 1.0,
    nn_model_id: str | None = None,
    fail_fast: bool = False,
    num_players: int = 2,
    filler_ai_type: str = "Random",
    filler_difficulty: int = 1,
    record_training_data: bool = False,
    record_replay_db: bool = True,
) -> MatchResult:
    """Run a single game between two AI tiers (with optional filler AIs for 3-4 player games).

    In multiplayer games:
    - Player 1: tier_a
    - Player 2: tier_b
    - Players 3-4: filler_ai_type (Random/Heuristic)

    Winner is reported as 1 if tier_a won, 2 if tier_b won, None for filler/draw.

    Args:
        record_training_data: If True, capture full game history for training export.
        record_replay_db: If True, record canonical replay data to GameReplayDB.
    """
    from contextlib import ExitStack

    from app.db.unified_recording import (
        RecordSource,
        RecordingConfig,
        UnifiedGameRecorder,
        is_recording_enabled,
    )
    from app.quality import compute_game_quality, get_quality_category
    from app.rules.history_contract import derive_phase_from_move_type, phase_move_contract
    def _tiebreak_winner(final_state: Any) -> int | None:
        players = getattr(final_state, "players", None) or []
        if not players:
            return None

        territory_counts: dict[int, int] = {}
        for p_id in final_state.board.collapsed_spaces.values():
            territory_counts[int(p_id)] = territory_counts.get(int(p_id), 0) + 1

        marker_counts: dict[int, int] = {int(p.player_number): 0 for p in players}
        for marker in final_state.board.markers.values():
            owner = int(marker.player)
            marker_counts[owner] = marker_counts.get(owner, 0) + 1

        last_actor = final_state.move_history[-1].player if final_state.move_history else None
        sorted_players = sorted(
            players,
            key=lambda p: (
                territory_counts.get(int(p.player_number), 0),
                -int(p.eliminated_rings),  # FEWER lost rings = better (negate for reverse sort)
                marker_counts.get(int(p.player_number), 0),
                1 if last_actor == p.player_number else 0,
            ),
            reverse=True,
        )
        return int(sorted_players[0].player_number) if sorted_players else None

    game_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    state = create_initial_state(board_type, num_players=num_players)
    engine = GameEngine()

    # Capture initial state for training data if requested
    initial_state_snapshot = None
    move_history_list = []
    if record_training_data:
        try:
            initial_state_snapshot = serialize_game_state(state)
        except (TypeError, AttributeError, json.JSONDecodeError):
            initial_state_snapshot = None

    # Create main competing AIs (tier_a as P1, tier_b as P2)
    ai_a = create_ai_for_tier(
        tier_a,
        1,
        seed,
        board_type=board_type,
        num_players=num_players,
        think_time_scale=think_time_scale,
        nn_model_id=nn_model_id,
    )
    ai_b = create_ai_for_tier(
        tier_b,
        2,
        seed + 1,
        board_type=board_type,
        num_players=num_players,
        think_time_scale=think_time_scale,
        nn_model_id=nn_model_id,
    )

    # Map player numbers to their AIs
    ai_map: dict[int, Any] = {1: ai_a, 2: ai_b}

    # Create filler AIs for 3-4 player games
    filler_ai_class = RandomAI if filler_ai_type == "Random" else HeuristicAI
    for p_num in range(3, num_players + 1):
        filler_config = AIConfig(
            difficulty=filler_difficulty,
            randomness=0.1 if filler_ai_type == "Random" else 0.05,
            think_time=0,
            rng_seed=seed + p_num,
        )
        ai_map[p_num] = filler_ai_class(p_num, filler_config)

    recorded_move_types: list[str] = []
    recording_enabled = record_replay_db and is_recording_enabled()
    board_type_value = board_type.value if hasattr(board_type, "value") else str(board_type)

    move_count = 0
    winner_override: int | None = None

    duration = 0.0
    actual_winner: int | None = None
    tier_winner: int | None = None
    game_record: dict[str, Any] | None = None

    with ExitStack() as stack:
        recorder = None
        if recording_enabled:
            recording_config = RecordingConfig(
                board_type=board_type_value,
                num_players=num_players,
                source=RecordSource.TOURNAMENT,
                engine_mode="tournament",
                db_prefix="tournament",
                db_dir="data/games",
            )
            recorder = stack.enter_context(
                UnifiedGameRecorder(recording_config, state, game_id=game_id)
            )

        while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
            current_player = state.current_player
            current_ai = ai_map.get(current_player)

            if current_ai is None:
                logger.error(f"No AI for player {current_player}")
                break

            try:
                move = current_ai.select_move(state)
                if move is None:
                    requirement = GameEngine.get_phase_requirement(state, current_player)
                    if requirement is not None:
                        move = GameEngine.synthesize_bookkeeping_move(requirement, state)

                if move is None:
                    # Current player cannot move - they lose, pick another winner
                    # For simplicity in multiplayer, just use tiebreak
                    winner_override = _tiebreak_winner(state)
                    break

                # Record move for training data if requested
                if record_training_data and move is not None:
                    try:
                        if hasattr(move, "model_dump"):
                            move_record = move.model_dump(mode="json")
                        elif hasattr(move, "dict"):
                            raw = move.dict()
                            move_record = json.loads(json.dumps(raw, cls=GameRecordEncoder))
                        else:
                            move_record = {"raw": str(move)}
                        move_history_list.append(move_record)
                    except (TypeError, AttributeError, json.JSONDecodeError):
                        pass

                state_before = state
                state = engine.apply_move(state, move, trace_mode=True)
                move_count += 1
                recorded_move_types.append(move.type.value)
                if recorder is not None:
                    recorder.add_move(
                        move,
                        state_after=state,
                        state_before=state_before,
                        available_moves_count=None,
                    )
            except Exception as e:
                if fail_fast:
                    raise
                logger.warning(f"Error in game {game_id}: {e}")
                winner_override = _tiebreak_winner(state)
                break

        duration = time.time() - start_time

        # Determine the actual winner player number
        actual_winner = winner_override
        if actual_winner is None and state.winner is not None:
            actual_winner = int(state.winner)
        if actual_winner is None:
            actual_winner = _tiebreak_winner(state)

        # Convert to tier winner (1=tier_a, 2=tier_b, None=filler/draw)
        if actual_winner == 1:
            tier_winner = 1  # tier_a won
        elif actual_winner == 2:
            tier_winner = 2  # tier_b won
        # else: filler AI won or draw - tier_winner stays None

        # Build game record for training if requested
        if record_training_data:
            winner_label = "draw"
            if tier_winner == 1:
                winner_label = "tier_a"
            elif tier_winner == 2:
                winner_label = "tier_b"

            game_record = {
                "game_id": game_id,
                "board_type": board_type.name.lower(),
                "num_players": num_players,
                "winner": winner_label,
                "winner_player": actual_winner,
                "game_length": move_count,
                "duration_sec": duration,
                "moves": move_history_list,
                "initial_state": initial_state_snapshot,
                "tier_a": tier_a,
                "tier_b": tier_b,
                "source": "run_distributed_tournament",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
            }

        if recorder is not None:
            unique_move_types = set(recorded_move_types)
            phase_labels = {
                phase
                for phase in (
                    derive_phase_from_move_type(mt) for mt in recorded_move_types
                )
                if phase
            }
            phase_count = len(phase_move_contract()) or 1
            phase_balance_score = min(1.0, len(phase_labels) / phase_count)
            diversity_score = min(1.0, len(unique_move_types) / 6.0) if unique_move_types else 0.0

            quality = compute_game_quality(
                {
                    "game_id": game_id,
                    "move_count": move_count,
                    "board_type": board_type_value,
                    "num_players": num_players,
                    "winner": actual_winner,
                    "termination_reason": state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status),
                    "source": "run_distributed_tournament",
                    "phase_balance_score": phase_balance_score,
                    "diversity_score": diversity_score,
                }
            )

            recorder.finalize(
                state,
                extra_metadata={
                    "tournament_id": f"{tier_a}_vs_{tier_b}",
                    "match_id": game_id,
                    "tier_a": tier_a,
                    "tier_b": tier_b,
                    "winner_player": actual_winner,
                    "winner_label": "tier_a" if tier_winner == 1 else ("tier_b" if tier_winner == 2 else "draw"),
                    "game_length": move_count,
                    "duration_sec": duration,
                    "seed": seed,
                    "phase_balance_score": phase_balance_score,
                    "diversity_score": diversity_score,
                    "quality_score": quality.quality_score,
                    "quality_category": get_quality_category(quality.quality_score).value,
                    "training_weight": quality.training_weight,
                    "sync_priority": quality.sync_priority,
                },
            )

    return MatchResult(
        tier_a=tier_a,
        tier_b=tier_b,
        winner=tier_winner,
        game_length=move_count,
        duration_sec=duration,
        worker=worker_name,
        game_id=game_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        game_index=game_index,
        game_record=game_record,
    )


# ============================================================================
# Distributed Tournament Runner
# ============================================================================

class DistributedTournament:
    """Orchestrates distributed AI strength evaluation tournament."""

    def __init__(
        self,
        tiers: list[str],
        games_per_matchup: int = 50,
        board_type: BoardType = BoardType.SQUARE8,
        max_workers: int = 8,
        output_dir: str = "results/tournaments",
        resume_file: str | None = None,
        checkpoint_path: str | None = None,
        nn_model_id: str | None = None,
        base_seed: int = 1,
        think_time_scale: float = 1.0,
        max_moves: int = 10000,
        confidence: float = 0.95,
        report_path: str | None = None,
        worker_label: str | None = None,
        fail_fast: bool = False,
        tournament_id: str | None = None,
        num_players: int = 2,
        filler_ai_type: str = "Random",
        filler_difficulty: int = 1,
        record_replay_db: bool = True,
        game_retries: int = 3,
        matchup_timeout: int = DEFAULT_MATCHUP_TIMEOUT,
        tournament_timeout: int = DEFAULT_TOURNAMENT_TIMEOUT,
    ):
        self.tiers = sorted(tiers, key=lambda t: int(t[1:]))
        self.game_retries = game_retries
        self.matchup_timeout = matchup_timeout
        self.tournament_timeout = tournament_timeout
        self.games_per_matchup = games_per_matchup
        self.board_type = board_type

        # Check device and limit workers if needed (MPS has threading issues)
        safe_workers, worker_warning = get_safe_max_workers(max_workers)
        if worker_warning:
            logger.warning(worker_warning)
        self.max_workers = safe_workers

        self.output_dir = Path(output_dir)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.nn_model_id = nn_model_id
        self.think_time_scale = think_time_scale
        self.max_moves = max_moves
        self.confidence = confidence
        self.report_path = Path(report_path) if report_path else None
        self.worker_label = worker_label
        self.fail_fast = fail_fast
        self.num_players = num_players
        self.filler_ai_type = filler_ai_type
        self.filler_difficulty = filler_difficulty
        self.record_replay_db = record_replay_db
        self.record_training_data = False
        self.training_output_path: Path | None = None
        self.training_records: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if resume_file and os.path.exists(resume_file):
            with open(resume_file) as f:
                self.state = TournamentState.from_dict(json.load(f))
            logger.info(f"Resumed tournament {self.state.tournament_id}")
            self.nn_model_id = self.state.nn_model_id
            self.think_time_scale = self.state.think_time_scale
            if self.checkpoint_path is None:
                self.checkpoint_path = Path(resume_file)
        else:
            self.state = TournamentState(
                tournament_id=tournament_id or str(uuid.uuid4())[:8],
                started_at=datetime.now(timezone.utc).isoformat(),
                board_type=board_type.value,
                games_per_matchup=games_per_matchup,
                tiers=tiers,
                base_seed=int(base_seed),
                think_time_scale=float(think_time_scale),
                nn_model_id=nn_model_id,
            )
            for tier in tiers:
                self.state.tier_stats[tier] = TierStats(tier=tier)

        self.all_matchups = []
        for i, tier_a in enumerate(self.tiers):
            for tier_b in self.tiers[i+1:]:
                self.all_matchups.append((tier_a, tier_b))

    def _get_pending_matchups(self) -> list[tuple[str, str]]:
        completed = set(self.state.completed_matchups)
        return [m for m in self.all_matchups if m not in completed]

    def _save_checkpoint(self) -> None:
        path = (
            self.checkpoint_path
            if self.checkpoint_path is not None
            else self.output_dir / f"tournament_{self.state.tournament_id}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.info(f"Saved checkpoint: {path}")

    def _export_training_data(self) -> None:
        """Export collected training records to JSONL file."""
        if not self.training_records:
            return

        if self.training_output_path is not None:
            output_path = self.training_output_path
        else:
            # Use data/tournaments/ to match ingestion pipeline patterns
            training_dir = Path("data/tournaments")
            training_dir.mkdir(parents=True, exist_ok=True)
            output_path = training_dir / f"tier_tournament_{self.state.tournament_id}.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in self.training_records:
                f.write(json.dumps(record, cls=GameRecordEncoder) + "\n")

        logger.info(
            f"Exported {len(self.training_records)} training records to {output_path}"
        )

    def _update_stats(self, result: MatchResult) -> None:
        tier_a = result.tier_a
        tier_b = result.tier_b

        stats_a = self.state.tier_stats[tier_a]
        stats_b = self.state.tier_stats[tier_b]

        stats_a.games_played += 1
        stats_b.games_played += 1

        if result.winner == 1:
            stats_a.wins += 1
            stats_b.losses += 1
        elif result.winner == 2:
            stats_a.losses += 1
            stats_b.wins += 1
        else:
            stats_a.draws += 1
            stats_b.draws += 1

        self.state.matches.append(result)

        # Persist to unified Elo database
        persist_match_to_unified_elo(
            tier_a=tier_a,
            tier_b=tier_b,
            winner=result.winner,
            board_type=self.state.board_type,
            num_players=self.num_players,
            tournament_id=self.state.tournament_id,
            game_length=result.game_length,
            duration_sec=result.duration_sec,
        )

    def run_matchup(
        self,
        tier_a: str,
        tier_b: str,
        worker_name: str = "local",
    ) -> list[MatchResult]:
        results = []
        base_seed = hash(
            (tier_a, tier_b, self.state.base_seed, self.board_type.value)
        ) & 0xFFFFFFFF

        for game_idx in range(self.games_per_matchup):
            if game_idx % 2 == 0:
                actual_a, actual_b = tier_a, tier_b
            else:
                actual_a, actual_b = tier_b, tier_a

            seed = base_seed + game_idx

            # Run game with retry logic
            result = None
            last_error = None
            for attempt in range(self.game_retries):
                try:
                    result = run_single_game(
                        actual_a, actual_b,
                        self.board_type, seed,
                        max_moves=self.max_moves,
                        worker_name=worker_name,
                        game_index=game_idx,
                        think_time_scale=self.think_time_scale,
                        nn_model_id=self.nn_model_id,
                        fail_fast=self.fail_fast,
                        num_players=self.num_players,
                        filler_ai_type=self.filler_ai_type,
                        filler_difficulty=self.filler_difficulty,
                        record_training_data=self.record_training_data,
                        record_replay_db=self.record_replay_db,
                    )
                    break  # Success
                except Exception as e:
                    last_error = e
                    if attempt < self.game_retries - 1:
                        delay = exponential_backoff_delay(attempt, base_delay=0.5, max_delay=5.0)
                        logger.warning(
                            f"Game {game_idx} failed (attempt {attempt + 1}/{self.game_retries}): {e}, "
                            f"retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Game {game_idx} failed after {self.game_retries} attempts: {e}")
                        if self.fail_fast:
                            raise

            # If all retries failed, skip this game
            if result is None:
                logger.warning(f"Skipping game {game_idx} after all retries failed")
                continue

            if game_idx % 2 == 1:
                result = MatchResult(
                    tier_a=tier_a,
                    tier_b=tier_b,
                    winner=3 - result.winner if result.winner else None,
                    game_length=result.game_length,
                    duration_sec=result.duration_sec,
                    worker=result.worker,
                    game_id=result.game_id,
                    timestamp=result.timestamp,
                    seed=result.seed,
                    game_index=result.game_index,
                    game_record=result.game_record,
                )

            results.append(result)
            self._update_stats(result)

            # Collect training records if enabled
            if self.record_training_data and result.game_record is not None:
                self.training_records.append(result.game_record)

            if (game_idx + 1) % 10 == 0:
                logger.info(
                    f"  {tier_a} vs {tier_b}: {game_idx + 1}/{self.games_per_matchup} games"
                )

        return results

    def run(self) -> dict[str, Any]:
        pending = self._get_pending_matchups()
        total_matchups = len(pending)

        logger.info(f"Tournament {self.state.tournament_id}")
        logger.info(f"  Tiers: {', '.join(self.tiers)}")
        logger.info(f"  Games per matchup: {self.games_per_matchup}")
        logger.info(f"  Pending matchups: {total_matchups}")
        logger.info(f"  Total games: {total_matchups * self.games_per_matchup}")

        start_time = time.time()

        # Heartbeat tracking for stuck detection
        last_completion_time = time.time()
        completed = 0
        matchups_timed_out = 0
        heartbeat_stop = threading.Event()

        def heartbeat_thread():
            """Log progress periodically to detect stuck tournaments."""
            while not heartbeat_stop.is_set():
                heartbeat_stop.wait(HEARTBEAT_INTERVAL)
                if heartbeat_stop.is_set():
                    break
                elapsed = time.time() - start_time
                since_last = time.time() - last_completion_time
                remaining_timeout = self.tournament_timeout - elapsed
                logger.info(
                    f"[Heartbeat] Tournament running for {elapsed:.0f}s, "
                    f"{completed}/{total_matchups} matchups done, "
                    f"{since_last:.0f}s since last completion, "
                    f"timeout in {remaining_timeout:.0f}s"
                )
                if since_last > self.matchup_timeout:
                    logger.warning(f"No matchup completed in {since_last:.0f}s (timeout: {self.matchup_timeout}s)")

        heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
        heartbeat.start()

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}

                for tier_a, tier_b in pending:
                    worker_name = f"worker_{len(futures) % self.max_workers}"
                    if self.worker_label:
                        worker_name = f"{self.worker_label}:{worker_name}"
                    future = executor.submit(
                        self.run_matchup, tier_a, tier_b, worker_name
                    )
                    futures[future] = (tier_a, tier_b)

                pending_futures = set(futures.keys())
                tournament_deadline = start_time + self.tournament_timeout

                while pending_futures:
                    # Calculate remaining time
                    remaining_time = tournament_deadline - time.time()
                    if remaining_time <= 0:
                        logger.error(f"Tournament timeout ({self.tournament_timeout}s) exceeded")
                        for f in pending_futures:
                            f.cancel()
                        break

                    try:
                        done_iter = as_completed(pending_futures, timeout=min(self.matchup_timeout, remaining_time))
                        for future in done_iter:
                            pending_futures.discard(future)
                            last_completion_time = time.time()
                            tier_a, tier_b = futures[future]

                            try:
                                results = future.result(timeout=5)
                                self.state.completed_matchups.append((tier_a, tier_b))
                                completed += 1

                                wins_a = sum(1 for r in results if r.winner == 1)
                                wins_b = sum(1 for r in results if r.winner == 2)
                                draws = sum(1 for r in results if r.winner is None)
                                logger.info(
                                    f"Completed {completed}/{total_matchups}: "
                                    f"{tier_a} vs {tier_b} = {wins_a}-{wins_b}-{draws}"
                                )

                                if completed % 5 == 0:
                                    self._save_checkpoint()

                            except Exception as e:
                                logger.error(f"Matchup {tier_a} vs {tier_b} failed: {e}")
                                if self.fail_fast:
                                    raise

                    except FuturesTimeoutError:
                        matchups_timed_out += 1
                        logger.warning(f"Matchup batch timed out after {self.matchup_timeout}s ({matchups_timed_out} timeouts)")
                        if matchups_timed_out >= 3:
                            logger.warning(f"{matchups_timed_out} timeouts - cancelling stuck futures")
                            for f in list(pending_futures)[:self.max_workers]:
                                f.cancel()
                                pending_futures.discard(f)

                if pending_futures:
                    logger.warning(f"{len(pending_futures)} matchups did not complete (cancelled/timed out)")

        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)

        duration = time.time() - start_time

        self._save_checkpoint()

        # Export training data if enabled
        if self.record_training_data and self.training_records:
            self._export_training_data()

        report = self.generate_report(duration)

        report_path = (
            self.report_path
            if self.report_path is not None
            else self.output_dir / f"report_{self.state.tournament_id}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved: {report_path}")

        return report

    def generate_report(self, duration: float) -> dict[str, Any]:
        tiers = list(self.tiers)
        matches = list(self.state.matches)

        def _canonical_match_key(result: MatchResult) -> tuple[str, str, int | None]:
            """Return ordered matchup key and winner in that orientation."""
            a, b = result.tier_a, result.tier_b
            if _tier_to_difficulty(a) <= _tier_to_difficulty(b):
                return (a, b, result.winner)
            winner = result.winner
            if winner == 1:
                winner = 2
            elif winner == 2:
                winner = 1
            return (b, a, winner)

        # Aggregate W/L/D per tier (order-independent).
        stats_by_tier: dict[str, TierStats] = {t: TierStats(tier=t) for t in tiers}
        for m in matches:
            stats_a = stats_by_tier.setdefault(m.tier_a, TierStats(tier=m.tier_a))
            stats_b = stats_by_tier.setdefault(m.tier_b, TierStats(tier=m.tier_b))
            stats_a.games_played += 1
            stats_b.games_played += 1
            if m.winner == 1:
                stats_a.wins += 1
                stats_b.losses += 1
            elif m.winner == 2:
                stats_a.losses += 1
                stats_b.wins += 1
            else:
                stats_a.draws += 1
                stats_b.draws += 1

        # Deterministic Elo: replay results in a fixed order, independent of
        # worker scheduling. (We keep k=32 to match the legacy harness.)
        elo_by_tier: dict[str, float] = dict.fromkeys(tiers, 1500.0)
        matchup_results: dict[tuple[str, str], list[MatchResult]] = {}
        for m in matches:
            a, b, winner = _canonical_match_key(m)
            canonical = MatchResult(
                tier_a=a,
                tier_b=b,
                winner=winner,
                game_length=m.game_length,
                duration_sec=m.duration_sec,
                worker=m.worker,
                game_id=m.game_id,
                timestamp=m.timestamp,
                seed=m.seed,
                game_index=m.game_index,
            )
            matchup_results.setdefault((a, b), []).append(canonical)

        for i, tier_a in enumerate(tiers):
            for tier_b in tiers[i + 1 :]:
                games = matchup_results.get((tier_a, tier_b), [])
                games.sort(
                    key=lambda r: (
                        r.game_index if r.game_index is not None else 1_000_000_000,
                        r.seed if r.seed is not None else 1_000_000_000_000_000_000,
                        r.game_id,
                    )
                )
                for r in games:
                    score_a = 0.5
                    if r.winner == 1:
                        score_a = 1.0
                    elif r.winner == 2:
                        score_a = 0.0
                    new_a, new_b = update_elo(
                        elo_by_tier.get(tier_a, 1500.0),
                        elo_by_tier.get(tier_b, 1500.0),
                        score_a,
                    )
                    elo_by_tier[tier_a] = new_a
                    elo_by_tier[tier_b] = new_b

        for tier, stats in stats_by_tier.items():
            stats.elo = float(elo_by_tier.get(tier, 1500.0))

        sorted_tiers = sorted(
            stats_by_tier.values(),
            key=lambda s: s.elo,
            reverse=True,
        )

        def _count_pair(tier: str, opp: str) -> tuple[int, int, int]:
            """Return (wins, losses, draws) for `tier` vs `opp`."""
            wins = sum(
                1
                for m in matches
                if (m.tier_a == tier and m.tier_b == opp and m.winner == 1)
                or (m.tier_a == opp and m.tier_b == tier and m.winner == 2)
            )
            losses = sum(
                1
                for m in matches
                if (m.tier_a == tier and m.tier_b == opp and m.winner == 2)
                or (m.tier_a == opp and m.tier_b == tier and m.winner == 1)
            )
            draws = sum(
                1
                for m in matches
                if (
                    (m.tier_a == tier and m.tier_b == opp)
                    or (m.tier_a == opp and m.tier_b == tier)
                )
                and m.winner is None
            )
            return (wins, losses, draws)

        h2h: dict[str, dict[str, str]] = {}
        for tier in tiers:
            h2h[tier] = {}
            for opp in tiers:
                if tier == opp:
                    h2h[tier][opp] = "-"
                    continue
                wins, losses, draws = _count_pair(tier, opp)
                h2h[tier][opp] = f"{wins}-{losses}-{draws}"

        matchup_stats: list[dict[str, Any]] = []
        for i, tier_a in enumerate(tiers):
            for tier_b in tiers[i + 1 :]:
                wins_a, losses_a, draws = _count_pair(tier_a, tier_b)
                wins_b = losses_a
                decisive = wins_a + wins_b
                win_rate_b = wins_b / decisive if decisive else 0.0
                ci_low, ci_high = wilson_score_interval(
                    wins_b,
                    decisive,
                    confidence=self.confidence,
                )
                games = matchup_results.get((tier_a, tier_b), [])
                avg_len = (
                    sum(g.game_length for g in games) / len(games)
                    if games
                    else 0.0
                )
                avg_dur = (
                    sum(g.duration_sec for g in games) / len(games)
                    if games
                    else 0.0
                )
                matchup_stats.append(
                    {
                        "tier_a": tier_a,
                        "tier_b": tier_b,
                        "wins_a": wins_a,
                        "wins_b": wins_b,
                        "draws": draws,
                        "decisive_games": decisive,
                        "decisive_win_rate_b": round(win_rate_b, 4),
                        "decisive_wilson_ci_b": [ci_low, ci_high],
                        "avg_game_length": round(avg_len, 2),
                        "avg_duration_sec": round(avg_dur, 3),
                    }
                )

        elo_gaps = []
        for i in range(len(sorted_tiers) - 1):
            gap = sorted_tiers[i].elo - sorted_tiers[i + 1].elo
            elo_gaps.append(
                {
                    "from": sorted_tiers[i].tier,
                    "to": sorted_tiers[i + 1].tier,
                    "gap": round(gap, 1),
                }
            )

        total_games = len(matches)
        avg_game_length = (
            sum(m.game_length for m in matches) / total_games
            if total_games
            else 0.0
        )
        avg_game_duration = (
            sum(m.duration_sec for m in matches) / total_games
            if total_games
            else 0.0
        )

        return {
            "tournament_id": self.state.tournament_id,
            "started_at": self.state.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(duration, 1),
            "config": {
                "board_type": self.board_type.value,
                "games_per_matchup": self.games_per_matchup,
                "tiers": tiers,
                "base_seed": self.state.base_seed,
                "think_time_scale": self.think_time_scale,
                "max_moves": self.max_moves,
                "nn_model_id": self.nn_model_id,
                "wilson_confidence": self.confidence,
            },
            "summary": {
                "total_games": total_games,
                "total_matchups": len(self.all_matchups),
                "avg_game_length": round(avg_game_length, 1),
                "avg_game_duration_sec": round(avg_game_duration, 3),
            },
            "rankings": [
                {
                    "rank": i + 1,
                    "tier": s.tier,
                    "elo": round(s.elo, 1),
                    "wins": s.wins,
                    "losses": s.losses,
                    "draws": s.draws,
                    "win_rate": round(s.win_rate * 100, 1),
                }
                for i, s in enumerate(sorted_tiers)
            ],
            "elo_gaps": elo_gaps,
            "head_to_head": h2h,
            "matchup_stats": matchup_stats,
        }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run distributed AI strength evaluation tournament"
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=50,
        help="Number of games per matchup (default: 50)",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="D1,D2,D3,D4,D5,D6",
        help="Comma-separated list of tiers to include (default: D1-D6)",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tournaments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        default=None,
        help="Optional explicit path for the checkpoint JSON (default: output-dir/tournament_<id>.json)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from previous tournament checkpoint",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 games per matchup, tiers D1-D4 only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed for deterministic match seeding (default: 1)",
    )
    parser.add_argument(
        "--tournament-id",
        type=str,
        default=None,
        help="Optional explicit tournament id (default: random 8-char id).",
    )
    parser.add_argument(
        "--think-time-scale",
        type=float,
        default=1.0,
        help="Multiply ladder think_time_ms by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=10000,
        help="Max moves per game before declaring draw (default: 10000)",
    )
    parser.add_argument(
        "--wilson-confidence",
        type=float,
        default=0.95,
        help="Wilson CI confidence for decisive matchups (default: 0.95)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional explicit path for the JSON report (default: output-dir/report_<id>.json)",
    )
    parser.add_argument(
        "--worker-label",
        type=str,
        default=None,
        help="Optional prefix for MatchResult.worker (useful for multi-host orchestration)",
    )
    parser.add_argument(
        "--nn-model-id",
        type=str,
        default=None,
        help=(
            "Optional override for the CNN model id used by MCTS/Descent tiers "
            "(defaults to LadderTierConfig.model_id; leave unset to use ladder defaults)."
        ),
    )
    parser.add_argument(
        "--require-neural-net",
        action="store_true",
        help=(
            "Fail fast if neural checkpoints cannot be loaded for neural tiers. "
            "Sets RINGRIFT_REQUIRE_NEURAL_NET=1 and implies --fail-fast."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the tournament on the first matchup/game exception.",
    )
    parser.add_argument(
        "--game-retries",
        type=int,
        default=3,
        help="Number of retries for failed individual games (default: 3).",
    )
    parser.add_argument(
        "--matchup-timeout",
        type=int,
        default=DEFAULT_MATCHUP_TIMEOUT,
        help=f"Timeout per matchup in seconds (default: {DEFAULT_MATCHUP_TIMEOUT})",
    )
    parser.add_argument(
        "--tournament-timeout",
        type=int,
        default=DEFAULT_TOURNAMENT_TIMEOUT,
        help=f"Timeout for entire tournament in seconds (default: {DEFAULT_TOURNAMENT_TIMEOUT})",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players per game (default: 2). For 3-4 players, filler slots use --filler-ai.",
    )
    parser.add_argument(
        "--filler-ai",
        type=str,
        default="Random",
        choices=["Random", "Heuristic"],
        help="AI type for extra player slots in 3-4 player games (default: Random).",
    )
    parser.add_argument(
        "--filler-difficulty",
        type=int,
        default=1,
        help="Difficulty level for filler AI (default: 1).",
    )
    parser.add_argument(
        "--record-training-data",
        action="store_true",
        help="Record full game history for training data export to JSONL.",
    )
    parser.add_argument(
        "--skip-replay-db",
        action="store_true",
        help="Skip recording games to GameReplayDB (canonical replay).",
    )
    parser.add_argument(
        "--training-output",
        type=str,
        default=None,
        help="Output path for training JSONL (default: data/tournaments/tier_tournament_{id}.jsonl).",
    )
    parser.add_argument(
        "--ingest-training",
        action="store_true",
        help="Ingest tournament replay DB into the canonical training pool (includes tournament sources).",
    )
    parser.add_argument(
        "--training-output-db",
        type=str,
        default=None,
        help="Output training pool DB (default: data/games/canonical_<board>[_<players>p].db).",
    )
    parser.add_argument(
        "--training-holdout-db",
        type=str,
        default=None,
        help="Optional holdout DB path for non-training sources.",
    )
    parser.add_argument(
        "--training-quarantine-db",
        type=str,
        default=None,
        help="Optional quarantine DB path for failed games.",
    )
    parser.add_argument(
        "--training-report-json",
        type=str,
        default=None,
        help="Optional JSON report path for the training ingest gate summary.",
    )
    parser.add_argument(
        "--training-parallel-workers",
        type=int,
        default=4,
        help="Parallel workers for training pool validation (default: 4).",
    )
    parser.add_argument(
        "--export-npz",
        action="store_true",
        help="Export an NPZ dataset after ingesting the training pool.",
    )
    parser.add_argument(
        "--export-npz-output",
        type=str,
        default=None,
        help="Output NPZ path (default: data/training/tournament_<board>_<players>p.npz).",
    )
    return parser.parse_args()


def _preflight_neural_checkpoints(
    tiers: list[str],
    board_type: BoardType,
    *,
    nn_model_id: str | None,
    think_time_scale: float,
) -> None:
    """Fail-fast preflight to ensure neural tiers can load their checkpoints."""
    neural_ids: set[str] = set()
    for tier in tiers:
        difficulty = _tier_to_difficulty(tier)
        if difficulty <= 1:
            continue
        ladder = get_ladder_tier_config(difficulty, board_type, num_players=2)
        if not ladder.use_neural_net or ladder.ai_type not in {AIType.MCTS, AIType.DESCENT}:
            continue
        effective = nn_model_id or ladder.model_id
        if effective:
            neural_ids.add(effective)

    if not neural_ids:
        return

    from app.ai.neural_net import NeuralNetAI

    logger.info(
        "Preflight: validating %d neural checkpoint id(s): %s",
        len(neural_ids),
        ", ".join(sorted(neural_ids)),
    )

    for model_id in sorted(neural_ids):
        cfg = AIConfig(
            difficulty=6,
            randomness=0.0,
            think_time=_scaled_think_time_ms(1_000, think_time_scale),
            rng_seed=123,
            heuristic_profile_id=None,
            use_neural_net=True,
            nn_model_id=model_id,
            allow_fresh_weights=False,
        )
        nn = NeuralNetAI(player_number=1, config=cfg, board_type=board_type)

        # Extra sanity check: verify that the runtime move encoder produces
        # indices that are in-range for the loaded policy head. This catches
        # subtle incompatibilities where a checkpoint loads but uses a
        # different policy layout (e.g. square8 policy_size=7000 vs legacy
        # MAX_N layouts).
        if board_type in {BoardType.SQUARE8, BoardType.SQUARE19} and nn.model is not None:
            from app.models.core import BoardState, Move, Position

            board_size = 8 if board_type == BoardType.SQUARE8 else 19
            board_state = BoardState(
                type=board_type,
                size=board_size,
                stacks={},
                markers={},
                collapsedSpaces={},
            )
            probe_move = Move(
                id="probe",
                type="move_stack",
                player=1,
                from_pos=Position(x=0, y=0),
                to=Position(x=1, y=0),
                timestamp=datetime.now(timezone.utc),
                thinkTime=0,
                moveNumber=0,
            )
            idx = nn.encode_move(probe_move, board_state)
            policy_size = int(getattr(nn.model, "policy_size", 0) or 0)
            if idx < 0 or idx >= policy_size:
                raise RuntimeError(
                    "Neural checkpoint policy layout mismatch: "
                    f"nn_model_id={model_id!r} board={board_type.value} "
                    f"probe_idx={idx} policy_size={policy_size}"
                )


def main() -> None:
    args = parse_args()

    if args.quick:
        tiers = ["D1", "D2", "D3", "D4"]
        games_per_matchup = 10
    else:
        tiers = [t.strip().upper() for t in args.tiers.split(",")]
        games_per_matchup = args.games_per_matchup

    board_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex": BoardType.HEXAGONAL,
    }
    board_type = board_map[args.board]

    nn_model_id = args.nn_model_id or None
    fail_fast = bool(args.fail_fast)

    if args.require_neural_net:
        os.environ["RINGRIFT_REQUIRE_NEURAL_NET"] = "1"
        fail_fast = True
        _preflight_neural_checkpoints(
            tiers,
            board_type,
            nn_model_id=nn_model_id,
            think_time_scale=args.think_time_scale,
        )

    tournament = DistributedTournament(
        tiers=tiers,
        games_per_matchup=games_per_matchup,
        board_type=board_type,
        max_workers=args.workers,
        output_dir=args.output_dir,
        resume_file=args.resume,
        checkpoint_path=args.output_checkpoint,
        nn_model_id=nn_model_id,
        base_seed=args.seed,
        think_time_scale=args.think_time_scale,
        max_moves=args.max_moves,
        confidence=args.wilson_confidence,
        report_path=args.output_report,
        worker_label=args.worker_label,
        fail_fast=fail_fast,
        tournament_id=args.tournament_id,
        num_players=args.num_players,
        filler_ai_type=args.filler_ai,
        filler_difficulty=args.filler_difficulty,
        record_replay_db=not args.skip_replay_db,
        game_retries=args.game_retries,
        matchup_timeout=args.matchup_timeout,
        tournament_timeout=args.tournament_timeout,
    )

    # Configure training data export if requested
    if args.record_training_data:
        tournament.record_training_data = True
        if args.training_output:
            tournament.training_output_path = Path(args.training_output)

    report = tournament.run()

    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"\nTotal games: {report['summary']['total_games']}")
    print(f"Duration: {report['duration_sec']:.1f} seconds")
    print("\nRANKINGS BY ELO:")
    print("-" * 50)
    print(f"{'Rank':<6} {'Tier':<6} {'Elo':<8} {'W-L-D':<12} {'Win%':<8}")
    print("-" * 50)
    for r in report["rankings"]:
        wld = f"{r['wins']}-{r['losses']}-{r['draws']}"
        print(f"{r['rank']:<6} {r['tier']:<6} {r['elo']:<8.1f} {wld:<12} {r['win_rate']:.1f}%")

    print("\nELO GAPS:")
    print("-" * 30)
    for gap in report["elo_gaps"]:
        print(f"{gap['from']} -> {gap['to']}: {gap['gap']:+.0f}")

    report_path = (
        args.output_report
        if args.output_report
        else f"{args.output_dir}/report_{report['tournament_id']}.json"
    )
    print(f"\nFull report: {report_path}")

    if args.ingest_training:
        if args.skip_replay_db:
            logger.warning("Skipping training ingest because --skip-replay-db was set.")
            return

        from app.db.unified_recording import RecordingConfig, RecordSource

        recording_config = RecordingConfig(
            board_type=board_type.value,
            num_players=args.num_players,
            source=RecordSource.TOURNAMENT,
            db_prefix="tournament",
            db_dir="data/games",
        )
        tournament_db = Path(recording_config.get_db_path()).resolve()
        if not tournament_db.exists():
            logger.warning("Tournament replay DB not found: %s", tournament_db)
            return

        board_token = board_type.value
        suffix = "" if int(args.num_players) == 2 else f"_{int(args.num_players)}p"
        default_output_db = Path("data/games") / f"canonical_{board_token}{suffix}.db"
        output_db = Path(args.training_output_db) if args.training_output_db else default_output_db
        report_json = (
            Path(args.training_report_json)
            if args.training_report_json
            else Path(args.output_dir) / f"training_ingest_{report['tournament_id']}.json"
        )

        cmd = [
            sys.executable,
            "scripts/build_canonical_training_pool_db.py",
            "--output-db",
            str(output_db),
            "--board-type",
            board_type.value,
            "--num-players",
            str(args.num_players),
            "--require-completed",
            "--include-tournament",
            "--report-json",
            str(report_json),
        ]
        if args.training_parallel_workers and args.training_parallel_workers > 1:
            cmd += ["--parallel-workers", str(args.training_parallel_workers)]
        if args.training_holdout_db:
            cmd += ["--holdout-db", args.training_holdout_db]
        if args.training_quarantine_db:
            cmd += ["--quarantine-db", args.training_quarantine_db]
        cmd += ["--input-db", str(tournament_db)]

        logger.info("Ingesting tournament DB into training pool: %s", " ".join(cmd))
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            PROJECT_ROOT
            if not existing_pythonpath
            else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
        )
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)

        if args.export_npz:
            default_npz = Path("data/training") / f"tournament_{board_token}_{int(args.num_players)}p.npz"
            export_output = Path(args.export_npz_output) if args.export_npz_output else default_npz
            export_output.parent.mkdir(parents=True, exist_ok=True)

            export_cmd = [
                sys.executable,
                "scripts/export_replay_dataset.py",
                "--db",
                str(output_db),
                "--board-type",
                board_type.value,
                "--num-players",
                str(args.num_players),
                "--output",
                str(export_output),
            ]
            logger.info("Exporting training dataset: %s", " ".join(export_cmd))
            # Feb 2026: Best-effort cross-process export coordination
            _config_key = f"{board_type.value}_{int(args.num_players)}p"
            _release_slot = False
            try:
                from app.coordination.export_coordinator import get_export_coordinator
                _coord = get_export_coordinator()
                if _coord.try_acquire(_config_key):
                    _release_slot = True
            except Exception:
                pass
            try:
                subprocess.run(export_cmd, cwd=PROJECT_ROOT, env=env, check=False)
            finally:
                if _release_slot:
                    try:
                        _coord.release(_config_key)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
