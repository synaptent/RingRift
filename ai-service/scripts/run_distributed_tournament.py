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
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import AIConfig, AIType, BoardType, GameStatus
from app.game_engine import GameEngine
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.mcts_ai import MCTSAI
from app.ai.descent_ai import DescentAI
from app.config.ladder_config import get_ladder_tier_config
from app.training.significance import wilson_score_interval
from app.training.generate_data import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MatchResult:
    """Result of a single game between two AI configurations."""
    tier_a: str
    tier_b: str
    winner: Optional[int]  # 1 for A, 2 for B, None for draw
    game_length: int
    duration_sec: float
    worker: str
    game_id: str
    timestamp: str
    seed: Optional[int] = None
    game_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
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


@dataclass
class TierStats:
    """Statistics for a single difficulty tier."""
    tier: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_played: int = 0
    elo: float = 1500.0

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played

    def to_dict(self) -> Dict[str, Any]:
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
    tiers: List[str]
    base_seed: int = 1
    think_time_scale: float = 1.0
    nn_model_id: Optional[str] = None
    matches: List[MatchResult] = field(default_factory=list)
    tier_stats: Dict[str, TierStats] = field(default_factory=dict)
    completed_matchups: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "TournamentState":
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
            state.tier_stats[tier] = TierStats(**stats)
        return state


# ============================================================================
# Elo Rating System
# ============================================================================

def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> Tuple[float, float]:
    """Update Elo ratings after a match."""
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)

    return new_a, new_b


# ============================================================================
# Game Runner
# ============================================================================

AI_CLASSES = {
    AIType.RANDOM: RandomAI,
    AIType.HEURISTIC: HeuristicAI,
    AIType.MINIMAX: MinimaxAI,
    AIType.MCTS: MCTSAI,
    AIType.DESCENT: DescentAI,
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
    return max(0, int(round(think_time_ms * factor)))


def create_ai_for_tier(
    tier: str,
    player_number: int,
    seed: int,
    *,
    board_type: BoardType,
    num_players: int,
    think_time_scale: float = 1.0,
    nn_model_id: Optional[str] = None,
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
        heuristic_profile_id: Optional[str] = None
        ladder_model_id: Optional[str] = None
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
    effective_nn_model_id: Optional[str] = None
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
    max_moves: int = 300,
    worker_name: str = "local",
    game_index: Optional[int] = None,
    think_time_scale: float = 1.0,
    nn_model_id: Optional[str] = None,
    fail_fast: bool = False,
) -> MatchResult:
    """Run a single game between two AI tiers."""
    game_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    state = create_initial_state(board_type, num_players=2)
    engine = GameEngine()

    ai_a = create_ai_for_tier(
        tier_a,
        1,
        seed,
        board_type=board_type,
        num_players=2,
        think_time_scale=think_time_scale,
        nn_model_id=nn_model_id,
    )
    ai_b = create_ai_for_tier(
        tier_b,
        2,
        seed + 1,
        board_type=board_type,
        num_players=2,
        think_time_scale=think_time_scale,
        nn_model_id=nn_model_id,
    )

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_ai = ai_a if state.current_player == 1 else ai_b

        try:
            move = current_ai.select_move(state)
            if move is None:
                break
            state = engine.apply_move(state, move)
            move_count += 1
        except Exception as e:
            if fail_fast:
                raise
            logger.warning(f"Error in game {game_id}: {e}")
            break

    duration = time.time() - start_time

    winner = None
    if state.winner == 1:
        winner = 1
    elif state.winner == 2:
        winner = 2

    return MatchResult(
        tier_a=tier_a,
        tier_b=tier_b,
        winner=winner,
        game_length=move_count,
        duration_sec=duration,
        worker=worker_name,
        game_id=game_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        game_index=game_index,
    )


# ============================================================================
# Distributed Tournament Runner
# ============================================================================

class DistributedTournament:
    """Orchestrates distributed AI strength evaluation tournament."""

    def __init__(
        self,
        tiers: List[str],
        games_per_matchup: int = 50,
        board_type: BoardType = BoardType.SQUARE8,
        max_workers: int = 8,
        output_dir: str = "results/tournaments",
        resume_file: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        nn_model_id: Optional[str] = None,
        base_seed: int = 1,
        think_time_scale: float = 1.0,
        max_moves: int = 300,
        confidence: float = 0.95,
        report_path: Optional[str] = None,
        worker_label: Optional[str] = None,
        fail_fast: bool = False,
    ):
        self.tiers = sorted(tiers, key=lambda t: int(t[1:]))
        self.games_per_matchup = games_per_matchup
        self.board_type = board_type
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.nn_model_id = nn_model_id
        self.think_time_scale = think_time_scale
        self.max_moves = max_moves
        self.confidence = confidence
        self.report_path = Path(report_path) if report_path else None
        self.worker_label = worker_label
        self.fail_fast = fail_fast
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
                tournament_id=str(uuid.uuid4())[:8],
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

    def _get_pending_matchups(self) -> List[Tuple[str, str]]:
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

    def run_matchup(
        self,
        tier_a: str,
        tier_b: str,
        worker_name: str = "local",
    ) -> List[MatchResult]:
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
            result = run_single_game(
                actual_a, actual_b,
                self.board_type, seed,
                max_moves=self.max_moves,
                worker_name=worker_name,
                game_index=game_idx,
                think_time_scale=self.think_time_scale,
                nn_model_id=self.nn_model_id,
                fail_fast=self.fail_fast,
            )

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
                )

            results.append(result)
            self._update_stats(result)

            if (game_idx + 1) % 10 == 0:
                logger.info(
                    f"  {tier_a} vs {tier_b}: {game_idx + 1}/{self.games_per_matchup} games"
                )

        return results

    def run(self) -> Dict[str, Any]:
        pending = self._get_pending_matchups()
        total_matchups = len(pending)

        logger.info(f"Tournament {self.state.tournament_id}")
        logger.info(f"  Tiers: {', '.join(self.tiers)}")
        logger.info(f"  Games per matchup: {self.games_per_matchup}")
        logger.info(f"  Pending matchups: {total_matchups}")
        logger.info(f"  Total games: {total_matchups * self.games_per_matchup}")

        start_time = time.time()

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

            completed = 0
            for future in as_completed(futures):
                tier_a, tier_b = futures[future]
                try:
                    results = future.result()
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

        duration = time.time() - start_time

        self._save_checkpoint()

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

    def generate_report(self, duration: float) -> Dict[str, Any]:
        tiers = list(self.tiers)
        matches = list(self.state.matches)

        def _canonical_match_key(result: MatchResult) -> Tuple[str, str, Optional[int]]:
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
        stats_by_tier: Dict[str, TierStats] = {t: TierStats(tier=t) for t in tiers}
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
        elo_by_tier: Dict[str, float] = {t: 1500.0 for t in tiers}
        matchup_results: Dict[Tuple[str, str], List[MatchResult]] = {}
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

        def _count_pair(tier: str, opp: str) -> Tuple[int, int, int]:
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

        h2h: Dict[str, Dict[str, str]] = {}
        for tier in tiers:
            h2h[tier] = {}
            for opp in tiers:
                if tier == opp:
                    h2h[tier][opp] = "-"
                    continue
                wins, losses, draws = _count_pair(tier, opp)
                h2h[tier][opp] = f"{wins}-{losses}-{draws}"

        matchup_stats: List[Dict[str, Any]] = []
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
        "--think-time-scale",
        type=float,
        default=1.0,
        help="Multiply ladder think_time_ms by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        help="Max moves per game before declaring draw (default: 300)",
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
    return parser.parse_args()


def _preflight_neural_checkpoints(
    tiers: List[str],
    board_type: BoardType,
    *,
    nn_model_id: Optional[str],
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
    )

    report = tournament.run()

    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"\nTotal games: {report['summary']['total_games']}")
    print(f"Duration: {report['duration_sec']:.1f} seconds")
    print(f"\nRANKINGS BY ELO:")
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


if __name__ == "__main__":
    main()
