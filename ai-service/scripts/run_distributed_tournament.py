#!/usr/bin/env python
"""Run distributed AI strength evaluation tournament across cloud instances.

This script orchestrates AI-vs-AI matches across multiple hosts to empirically
measure the relative strength of different AI configurations. It uses the
existing cluster infrastructure (distributed_hosts.yaml) and can run hundreds
of games in parallel.

Features:
- Distributes games across all available cloud workers
- Supports all difficulty tiers (D1-D10)
- Calculates Elo ratings from match results
- Generates comprehensive strength reports
- Fault-tolerant with automatic retry

Usage:
    # Full tournament across all difficulty tiers
    python scripts/run_distributed_tournament.py --games-per-matchup 50

    # Quick validation run
    python scripts/run_distributed_tournament.py --games-per-matchup 10 --tiers D1,D2,D3

    # Use specific hosts only
    python scripts/run_distributed_tournament.py --hosts aws-staging,lambda-gpu

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

# Canonical difficulty profiles from main.py
DIFFICULTY_PROFILES = {
    "D1": {"ai_type": AIType.RANDOM, "randomness": 0.5, "think_time_ms": 150},
    "D2": {"ai_type": AIType.HEURISTIC, "randomness": 0.3, "think_time_ms": 200},
    "D3": {"ai_type": AIType.MINIMAX, "randomness": 0.15, "think_time_ms": 1800, "use_neural_net": False},
    "D4": {"ai_type": AIType.MINIMAX, "randomness": 0.08, "think_time_ms": 2800, "use_neural_net": True},
    "D5": {"ai_type": AIType.MCTS, "randomness": 0.05, "think_time_ms": 4000, "use_neural_net": False},
    "D6": {"ai_type": AIType.MCTS, "randomness": 0.02, "think_time_ms": 5500, "use_neural_net": True},
    "D7": {"ai_type": AIType.MCTS, "randomness": 0.0, "think_time_ms": 7500, "use_neural_net": True},
    "D8": {"ai_type": AIType.MCTS, "randomness": 0.0, "think_time_ms": 9600, "use_neural_net": True},
    "D9": {"ai_type": AIType.DESCENT, "randomness": 0.0, "think_time_ms": 12600, "use_neural_net": True},
    "D10": {"ai_type": AIType.DESCENT, "randomness": 0.0, "think_time_ms": 16000, "use_neural_net": True},
}

AI_CLASSES = {
    AIType.RANDOM: RandomAI,
    AIType.HEURISTIC: HeuristicAI,
    AIType.MINIMAX: MinimaxAI,
    AIType.MCTS: MCTSAI,
    AIType.DESCENT: DescentAI,
}


def get_best_nn_model_id() -> str:
    """Find the best available neural network model.

    Preference order:
    1. ringrift_v4_sq8_2p (canonical, stable id)
    2. ringrift_v3_sq8_2p (fallback)
    3. sq8_2p_nn_baseline (legacy fallback; may have reduced policy head)

    Notes:
      - "v4" here refers to the model *ID/lineage* (checkpoint naming), not a
        separate architecture class. The checkpoint metadata declares the
        actual model class (e.g., RingRiftCNN_v2).
      - We intentionally do NOT fall back to deprecated ringrift_v1/v1_mps ids
        because they are often missing and silently disable neural evaluation.
    """
    import glob
    import os

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    def _is_usable(path: str) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            if os.path.getsize(path) <= 1000:
                return False
            # Avoid specialized artifacts unless explicitly requested.
            if "h100" in os.path.basename(path).lower():
                return False
            return True
        except OSError:
            return False

    # Prefer canonical v4 checkpoints when present.
    v4_candidates = [
        os.path.join(models_dir, "ringrift_v4_sq8_2p.pth"),
        *sorted(glob.glob(os.path.join(models_dir, "ringrift_v4_sq8_2p_*.pth"))),
    ]
    if any(_is_usable(p) for p in v4_candidates):
        return "ringrift_v4_sq8_2p"

    # Check for v3 models first (prefer latest timestamp)
    v3_patterns = [
        os.path.join(models_dir, "ringrift_v3_sq8_2p_*.pth"),
    ]
    for pattern in v3_patterns:
        matches = sorted(glob.glob(pattern))
        # Filter out empty files and get latest
        valid_matches = [m for m in matches if _is_usable(m)]
        if valid_matches:
            # Return model ID without path and extension
            basename = os.path.basename(valid_matches[-1])
            # Extract the base model ID (before any timestamp suffix)
            return "ringrift_v3_sq8_2p"

    # Fallback to v2 baseline
    v2_patterns = [
        os.path.join(models_dir, "sq8_2p_nn_baseline_*.pth"),
    ]
    for pattern in v2_patterns:
        matches = sorted(glob.glob(pattern))
        valid_matches = [m for m in matches if _is_usable(m)]
        if valid_matches:
            return "sq8_2p_nn_baseline"

    raise RuntimeError(
        "No usable neural network checkpoints found under ai-service/models/. "
        "Expected at least one of: ringrift_v4_sq8_2p*.pth, ringrift_v3_sq8_2p*.pth. "
        "Provide --nn-model-id explicitly, or run training to produce a checkpoint."
    )


def create_ai_for_tier(
    tier: str,
    player_number: int,
    seed: int,
    *,
    nn_model_id: Optional[str] = None,
) -> Any:
    """Create an AI instance for the given difficulty tier."""
    profile = DIFFICULTY_PROFILES[tier]
    difficulty = int(tier[1:])
    use_nn = profile.get("use_neural_net", False)

    # Auto-select best available CNN model for neural-enabled tiers that use it.
    # Note: D4 (Minimax) uses NNUE, not the CNN policy/value net; do not
    # override nn_model_id there unless explicitly provided for NNUE.
    effective_nn_model_id = None
    if use_nn and profile["ai_type"] in {AIType.MCTS, AIType.DESCENT}:
        effective_nn_model_id = nn_model_id or get_best_nn_model_id()
        logger.debug("Using neural model %s for tier %s", effective_nn_model_id, tier)

    config = AIConfig(
        difficulty=difficulty,
        randomness=profile["randomness"],
        think_time=profile["think_time_ms"],
        rng_seed=seed,
        use_neural_net=use_nn,
        nn_model_id=effective_nn_model_id,
        allow_fresh_weights=False,
    )

    ai_class = AI_CLASSES[profile["ai_type"]]
    return ai_class(player_number, config)


def run_single_game(
    tier_a: str,
    tier_b: str,
    board_type: BoardType,
    seed: int,
    max_moves: int = 300,
    worker_name: str = "local",
    nn_model_id: Optional[str] = None,
) -> MatchResult:
    """Run a single game between two AI tiers."""
    game_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    state = create_initial_state(board_type, num_players=2)
    engine = GameEngine()

    ai_a = create_ai_for_tier(tier_a, 1, seed, nn_model_id=nn_model_id)
    ai_b = create_ai_for_tier(tier_b, 2, seed + 1, nn_model_id=nn_model_id)

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
        nn_model_id: Optional[str] = None,
    ):
        self.tiers = sorted(tiers, key=lambda t: int(t[1:]))
        self.games_per_matchup = games_per_matchup
        self.board_type = board_type
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        # CNN policy/value model id used by neural tiers (MCTS/Descent).
        # When omitted, get_best_nn_model_id() picks a canonical v4 checkpoint.
        self.nn_model_id = nn_model_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if resume_file and os.path.exists(resume_file):
            with open(resume_file) as f:
                self.state = TournamentState.from_dict(json.load(f))
            logger.info(f"Resumed tournament {self.state.tournament_id}")
        else:
            self.state = TournamentState(
                tournament_id=str(uuid.uuid4())[:8],
                started_at=datetime.now(timezone.utc).isoformat(),
                board_type=board_type.value,
                games_per_matchup=games_per_matchup,
                tiers=tiers,
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
        path = self.output_dir / f"tournament_{self.state.tournament_id}.json"
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
            score_a = 1.0
        elif result.winner == 2:
            stats_a.losses += 1
            stats_b.wins += 1
            score_a = 0.0
        else:
            stats_a.draws += 1
            stats_b.draws += 1
            score_a = 0.5

        new_a, new_b = update_elo(stats_a.elo, stats_b.elo, score_a)
        stats_a.elo = new_a
        stats_b.elo = new_b

        self.state.matches.append(result)

    def run_matchup(
        self,
        tier_a: str,
        tier_b: str,
        worker_name: str = "local",
    ) -> List[MatchResult]:
        results = []
        base_seed = hash((tier_a, tier_b, self.state.tournament_id)) & 0xFFFFFFFF

        for game_idx in range(self.games_per_matchup):
            if game_idx % 2 == 0:
                actual_a, actual_b = tier_a, tier_b
            else:
                actual_a, actual_b = tier_b, tier_a

            seed = base_seed + game_idx
            result = run_single_game(
                actual_a, actual_b,
                self.board_type, seed,
                worker_name=worker_name,
                nn_model_id=self.nn_model_id,
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

        duration = time.time() - start_time

        self._save_checkpoint()

        report = self.generate_report(duration)

        report_path = self.output_dir / f"report_{self.state.tournament_id}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved: {report_path}")

        return report

    def generate_report(self, duration: float) -> Dict[str, Any]:
        sorted_tiers = sorted(
            self.state.tier_stats.values(),
            key=lambda s: s.elo,
            reverse=True,
        )

        h2h = {}
        for tier in self.tiers:
            h2h[tier] = {}
            for opp in self.tiers:
                if tier == opp:
                    h2h[tier][opp] = "-"
                else:
                    wins = sum(
                        1 for m in self.state.matches
                        if (m.tier_a == tier and m.tier_b == opp and m.winner == 1)
                        or (m.tier_a == opp and m.tier_b == tier and m.winner == 2)
                    )
                    losses = sum(
                        1 for m in self.state.matches
                        if (m.tier_a == tier and m.tier_b == opp and m.winner == 2)
                        or (m.tier_a == opp and m.tier_b == tier and m.winner == 1)
                    )
                    h2h[tier][opp] = f"{wins}-{losses}"

        elo_gaps = []
        for i in range(len(sorted_tiers) - 1):
            gap = sorted_tiers[i].elo - sorted_tiers[i + 1].elo
            elo_gaps.append({
                "from": sorted_tiers[i].tier,
                "to": sorted_tiers[i + 1].tier,
                "gap": round(gap, 1),
            })

        return {
            "tournament_id": self.state.tournament_id,
            "started_at": self.state.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(duration, 1),
            "config": {
                "board_type": self.board_type.value,
                "games_per_matchup": self.games_per_matchup,
                "tiers": self.tiers,
            },
            "summary": {
                "total_games": len(self.state.matches),
                "total_matchups": len(self.all_matchups),
                "avg_game_length": round(
                    sum(m.game_length for m in self.state.matches)
                    / max(1, len(self.state.matches)), 1
                ),
                "avg_game_duration_sec": round(
                    sum(m.duration_sec for m in self.state.matches)
                    / max(1, len(self.state.matches)), 2
                ),
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
        "--nn-model-id",
        type=str,
        default="auto",
        help=(
            "CNN model id/prefix for neural tiers (MCTS/Descent). "
            "Use 'auto' to select the best available local checkpoint "
            "(default: auto)."
        ),
    )
    return parser.parse_args()


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

    nn_model_id: Optional[str] = None
    needs_cnn_model = any(
        DIFFICULTY_PROFILES[t].get("use_neural_net", False)
        and DIFFICULTY_PROFILES[t]["ai_type"] in {AIType.MCTS, AIType.DESCENT}
        for t in tiers
    )
    if needs_cnn_model:
        if args.nn_model_id and args.nn_model_id.lower() != "auto":
            nn_model_id = args.nn_model_id
        else:
            nn_model_id = get_best_nn_model_id()
        logger.info("Using CNN nn_model_id=%s for neural tiers", nn_model_id)

    tournament = DistributedTournament(
        tiers=tiers,
        games_per_matchup=games_per_matchup,
        board_type=board_type,
        max_workers=args.workers,
        output_dir=args.output_dir,
        resume_file=args.resume,
        nn_model_id=nn_model_id,
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

    print(f"\nFull report: {args.output_dir}/report_{report['tournament_id']}.json")


if __name__ == "__main__":
    main()
