#!/usr/bin/env python3
"""Unified Distributed Tournament System for RingRift AI Evaluation.

This script combines all tournament functionality into a single unified system:
- All board types (square8, square19, hexagonal)
- All player counts (2p, 3p, 4p)
- All difficulty tiers (D1-D10)
- All AI types (Random, Heuristic, Minimax, MCTS, Descent)
- Neural network model support
- Distributed execution across SSH cluster
- Persistent Elo tracking in SQLite
- Continuous/periodic tournament scheduling

Usage:
    # Full tournament across all configurations
    python scripts/run_unified_tournament.py --all-configs --distributed

    # Specific board/players
    python scripts/run_unified_tournament.py --board square8 --players 2 3 4

    # Run specific difficulty tiers
    python scripts/run_unified_tournament.py --tiers D1-D10 --board square8

    # Run with specific AI types only
    python scripts/run_unified_tournament.py --ai-types MCTS Descent --neural-only

    # Continuous mode (runs every N hours)
    python scripts/run_unified_tournament.py --all-configs --interval 4

    # View current Elo leaderboard
    python scripts/run_unified_tournament.py --leaderboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sqlite3
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import AIConfig, AIType, BoardType, GameStatus
from app.game_engine import GameEngine
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.mcts_ai import MCTSAI
from app.ai.descent_ai import DescentAI
from app.config.ladder_config import get_ladder_tier_config, LadderTierConfig
from app.training.generate_data import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

AI_CLASSES = {
    AIType.RANDOM: RandomAI,
    AIType.HEURISTIC: HeuristicAI,
    AIType.MINIMAX: MinimaxAI,
    AIType.MCTS: MCTSAI,
    AIType.DESCENT: DescentAI,
}

AI_TYPE_MAP = {
    "random": AIType.RANDOM,
    "heuristic": AIType.HEURISTIC,
    "minimax": AIType.MINIMAX,
    "mcts": AIType.MCTS,
    "descent": AIType.DESCENT,
}

BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
    "hex": BoardType.HEXAGONAL,
}

# Default games per matchup based on complexity
DEFAULT_GAMES_PER_MATCHUP = {
    (BoardType.SQUARE8, 2): 20,
    (BoardType.SQUARE8, 3): 15,
    (BoardType.SQUARE8, 4): 10,
    (BoardType.SQUARE19, 2): 10,
    (BoardType.SQUARE19, 3): 8,
    (BoardType.SQUARE19, 4): 6,
    (BoardType.HEXAGONAL, 2): 15,
    (BoardType.HEXAGONAL, 3): 10,
    (BoardType.HEXAGONAL, 4): 8,
}

# Think time scaling per board (larger boards need more time)
THINK_TIME_SCALE = {
    BoardType.SQUARE8: 1.0,
    BoardType.SQUARE19: 1.5,
    BoardType.HEXAGONAL: 1.2,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Participant:
    """A tournament participant (AI configuration)."""
    id: str
    name: str
    ai_type: AIType
    difficulty: int
    use_neural_net: bool
    model_id: Optional[str] = None
    think_time_ms: int = 1000
    randomness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "ai_type": self.ai_type.value,
            "difficulty": self.difficulty,
            "use_neural_net": self.use_neural_net,
            "model_id": self.model_id,
            "think_time_ms": self.think_time_ms,
            "randomness": self.randomness,
        }


@dataclass
class MatchResult:
    """Result of a single match."""
    match_id: str
    participant_ids: List[str]  # All participants in order of player number
    winner_id: Optional[str]  # None for draw
    game_length: int
    duration_sec: float
    board_type: str
    num_players: int
    timestamp: str
    worker: str = "local"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EloRating:
    """Elo rating for a participant."""
    participant_id: str
    board_type: str
    num_players: int
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


@dataclass
class ClusterHost:
    """A host in the distributed cluster."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ringrift_path: str = "~/ringrift/ai-service"
    status: str = "ready"

    def ssh_cmd(self, command: str) -> List[str]:
        return [
            "ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
            "-p", str(self.ssh_port),
            f"{self.ssh_user}@{self.ssh_host}",
            command,
        ]


# =============================================================================
# Elo Database
# =============================================================================

class EloDatabase:
    """SQLite database for persistent Elo tracking."""

    def __init__(self, db_path: str = "data/unified_elo.db"):
        self.db_path = Path(PROJECT_ROOT) / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS participants (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    ai_type TEXT,
                    difficulty INTEGER,
                    use_neural_net INTEGER,
                    model_id TEXT,
                    created_at REAL
                );

                CREATE TABLE IF NOT EXISTS elo_ratings (
                    participant_id TEXT,
                    board_type TEXT,
                    num_players INTEGER,
                    rating REAL DEFAULT 1500.0,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    last_update REAL,
                    PRIMARY KEY (participant_id, board_type, num_players)
                );

                CREATE TABLE IF NOT EXISTS match_history (
                    id TEXT PRIMARY KEY,
                    participant_ids TEXT,  -- JSON array
                    winner_id TEXT,
                    game_length INTEGER,
                    duration_sec REAL,
                    board_type TEXT,
                    num_players INTEGER,
                    timestamp TEXT,
                    worker TEXT,
                    tournament_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_elo_rating
                ON elo_ratings(board_type, num_players, rating DESC);

                CREATE INDEX IF NOT EXISTS idx_match_timestamp
                ON match_history(timestamp DESC);
            """)

    def register_participant(self, p: Participant):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO participants
                (id, name, ai_type, difficulty, use_neural_net, model_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (p.id, p.name, p.ai_type.value, p.difficulty,
                  int(p.use_neural_net), p.model_id, time.time()))

    def get_rating(self, participant_id: str, board_type: str, num_players: int) -> EloRating:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT rating, games_played, wins, losses, draws
                FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (participant_id, board_type, num_players)).fetchone()

            if row:
                return EloRating(
                    participant_id=participant_id,
                    board_type=board_type,
                    num_players=num_players,
                    rating=row[0],
                    games_played=row[1],
                    wins=row[2],
                    losses=row[3],
                    draws=row[4],
                )
            return EloRating(participant_id, board_type, num_players)

    def update_ratings(self, ratings: List[EloRating]):
        with sqlite3.connect(self.db_path) as conn:
            for r in ratings:
                conn.execute("""
                    INSERT OR REPLACE INTO elo_ratings
                    (participant_id, board_type, num_players, rating, games_played,
                     wins, losses, draws, last_update)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (r.participant_id, r.board_type, r.num_players, r.rating,
                      r.games_played, r.wins, r.losses, r.draws, time.time()))

    def record_match(self, result: MatchResult, tournament_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO match_history
                (id, participant_ids, winner_id, game_length, duration_sec,
                 board_type, num_players, timestamp, worker, tournament_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (result.match_id, json.dumps(result.participant_ids),
                  result.winner_id, result.game_length, result.duration_sec,
                  result.board_type, result.num_players, result.timestamp,
                  result.worker, tournament_id))

    def get_leaderboard(self, board_type: Optional[str] = None,
                        num_players: Optional[int] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT e.*, p.name, p.ai_type, p.difficulty, p.use_neural_net, p.model_id
                FROM elo_ratings e
                JOIN participants p ON e.participant_id = p.id
                WHERE e.games_played > 0
            """
            params = []

            if board_type:
                query += " AND e.board_type = ?"
                params.append(board_type)
            if num_players:
                query += " AND e.num_players = ?"
                params.append(num_players)

            query += " ORDER BY e.rating DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]


# =============================================================================
# Participant Generation
# =============================================================================

def generate_tier_participants(
    board_type: BoardType,
    num_players: int,
    tiers: List[int],
) -> List[Participant]:
    """Generate participants from difficulty tiers using ladder config."""
    participants = []

    for tier in tiers:
        try:
            if tier == 1:
                # D1 is always Random baseline
                p = Participant(
                    id=f"D1_{board_type.value}_{num_players}p",
                    name=f"Random_D1",
                    ai_type=AIType.RANDOM,
                    difficulty=1,
                    use_neural_net=False,
                    think_time_ms=100,
                    randomness=1.0,
                )
            else:
                ladder = get_ladder_tier_config(tier, board_type, num_players)
                p = Participant(
                    id=f"D{tier}_{board_type.value}_{num_players}p",
                    name=f"{ladder.ai_type.value}_D{tier}",
                    ai_type=ladder.ai_type,
                    difficulty=tier,
                    use_neural_net=ladder.use_neural_net,
                    model_id=ladder.model_id,
                    think_time_ms=ladder.think_time_ms,
                    randomness=ladder.randomness,
                )
            participants.append(p)
        except Exception as e:
            logger.warning(f"No ladder config for D{tier} {board_type.value} {num_players}p: {e}")

    return participants


def generate_ai_type_participants(
    board_type: BoardType,
    num_players: int,
    ai_types: List[AIType],
    include_neural: bool = True,
    difficulties: List[int] = [3, 5, 7],
) -> List[Participant]:
    """Generate participants for specific AI types at various difficulties."""
    participants = []

    for ai_type in ai_types:
        for diff in difficulties:
            # Non-neural version
            p_id = f"{ai_type.value}_D{diff}_{board_type.value}_{num_players}p"
            participants.append(Participant(
                id=p_id,
                name=f"{ai_type.value}_D{diff}",
                ai_type=ai_type,
                difficulty=diff,
                use_neural_net=False,
                think_time_ms=1000 * diff,
                randomness=max(0, 0.3 - diff * 0.03),
            ))

            # Neural version (for MCTS and Descent only)
            if include_neural and ai_type in {AIType.MCTS, AIType.DESCENT}:
                p_id_nn = f"{ai_type.value}_NN_D{diff}_{board_type.value}_{num_players}p"
                model_id = f"ringrift_best_{board_type.value}_{num_players}p"
                participants.append(Participant(
                    id=p_id_nn,
                    name=f"{ai_type.value}_NN_D{diff}",
                    ai_type=ai_type,
                    difficulty=diff,
                    use_neural_net=True,
                    model_id=model_id,
                    think_time_ms=1000 * diff,
                    randomness=max(0, 0.2 - diff * 0.02),
                ))

    return participants


# =============================================================================
# Game Execution
# =============================================================================

def create_ai_from_participant(
    participant: Participant,
    player_number: int,
    board_type: BoardType,
    seed: int,
    think_time_scale: float = 1.0,
) -> Any:
    """Create an AI instance from a participant definition."""
    scaled_think_time = int(participant.think_time_ms * think_time_scale)

    config = AIConfig(
        difficulty=participant.difficulty,
        randomness=participant.randomness,
        think_time=scaled_think_time,
        rng_seed=seed,
        use_neural_net=participant.use_neural_net,
        nn_model_id=participant.model_id if participant.use_neural_net else None,
        allow_fresh_weights=False,
    )

    ai_class = AI_CLASSES[participant.ai_type]
    return ai_class(player_number, config)


def run_match(
    participants: List[Participant],
    board_type: BoardType,
    seed: int,
    max_moves: int = 10000,
    worker: str = "local",
) -> MatchResult:
    """Run a single match between participants."""
    match_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    num_players = len(participants)

    # Create game state
    state = create_initial_state(board_type, num_players=num_players)
    engine = GameEngine()

    # Create AIs
    think_scale = THINK_TIME_SCALE.get(board_type, 1.0)
    ais = {}
    for i, p in enumerate(participants):
        player_num = i + 1
        ais[player_num] = create_ai_from_participant(
            p, player_num, board_type, seed + i, think_scale
        )

    # Run game
    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player
        ai = ais.get(current_player)

        if ai is None:
            break

        try:
            move = ai.select_move(state)
            if move is None:
                break
            state = engine.apply_move(state, move)
            move_count += 1
        except Exception as e:
            logger.warning(f"Move error in match {match_id}: {e}")
            break

    duration = time.time() - start_time

    # Determine winner
    winner_id = None
    if state.winner is not None:
        winner_idx = state.winner - 1  # Convert player number to index
        if 0 <= winner_idx < len(participants):
            winner_id = participants[winner_idx].id

    return MatchResult(
        match_id=match_id,
        participant_ids=[p.id for p in participants],
        winner_id=winner_id,
        game_length=move_count,
        duration_sec=duration,
        board_type=board_type.value,
        num_players=num_players,
        timestamp=datetime.now(timezone.utc).isoformat(),
        worker=worker,
    )


# =============================================================================
# Elo Calculation
# =============================================================================

def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo_2p(
    rating_a: float, rating_b: float,
    score_a: float, k: float = 32.0,
) -> Tuple[float, float]:
    """Update Elo ratings for a 2-player match."""
    expected_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - expected_a))
    return new_a, new_b


def update_elo_multiplayer(
    ratings: List[float],
    winner_idx: Optional[int],
    k: float = 32.0,
) -> List[float]:
    """Update Elo ratings for multiplayer match.

    Uses average opponent rating approach.
    """
    n = len(ratings)
    if n < 2:
        return ratings

    new_ratings = list(ratings)

    for i in range(n):
        # Calculate average opponent rating
        opp_ratings = [ratings[j] for j in range(n) if j != i]
        avg_opp = sum(opp_ratings) / len(opp_ratings)

        # Score: 1 for winner, 0.5 for draw, 0 otherwise
        if winner_idx is None:
            score = 0.5
        elif winner_idx == i:
            score = 1.0
        else:
            score = 0.0

        expected = expected_score(ratings[i], avg_opp)
        new_ratings[i] = ratings[i] + k * (score - expected)

    return new_ratings


# =============================================================================
# Tournament Runner
# =============================================================================

class UnifiedTournament:
    """Main tournament orchestrator."""

    def __init__(
        self,
        db: EloDatabase,
        board_types: List[BoardType],
        player_counts: List[int],
        participants_per_config: Optional[Dict[Tuple[BoardType, int], List[Participant]]] = None,
        games_per_matchup: int = 10,
        max_workers: int = 4,
    ):
        self.db = db
        self.board_types = board_types
        self.player_counts = player_counts
        self.participants = participants_per_config or {}
        self.games_per_matchup = games_per_matchup
        self.max_workers = max_workers
        self.tournament_id = str(uuid.uuid4())[:8]
        self.results: List[MatchResult] = []

    def add_participants(self, board_type: BoardType, num_players: int,
                         participants: List[Participant]):
        """Add participants for a specific configuration."""
        key = (board_type, num_players)
        if key not in self.participants:
            self.participants[key] = []

        for p in participants:
            self.db.register_participant(p)
            if p not in self.participants[key]:
                self.participants[key].append(p)

    def generate_matchups(
        self, board_type: BoardType, num_players: int
    ) -> List[List[Participant]]:
        """Generate all matchups for a configuration."""
        key = (board_type, num_players)
        participants = self.participants.get(key, [])

        if len(participants) < num_players:
            return []

        matchups = []

        if num_players == 2:
            # Round-robin for 2-player
            for i, p1 in enumerate(participants):
                for p2 in participants[i+1:]:
                    for _ in range(self.games_per_matchup):
                        # Alternate colors
                        if len(matchups) % 2 == 0:
                            matchups.append([p1, p2])
                        else:
                            matchups.append([p2, p1])
        else:
            # Sample combinations for multiplayer
            import itertools
            all_combos = list(itertools.combinations(participants, num_players))
            random.shuffle(all_combos)

            # Take subset if too many
            max_combos = min(len(all_combos), len(participants) * 5)
            for combo in all_combos[:max_combos]:
                combo_list = list(combo)
                for _ in range(self.games_per_matchup):
                    random.shuffle(combo_list)
                    matchups.append(combo_list.copy())

        return matchups

    def run_config(self, board_type: BoardType, num_players: int) -> List[MatchResult]:
        """Run tournament for a single configuration."""
        matchups = self.generate_matchups(board_type, num_players)

        if not matchups:
            logger.warning(f"No matchups for {board_type.value} {num_players}p")
            return []

        logger.info(f"Running {len(matchups)} matches for {board_type.value} {num_players}p")
        results = []

        def run_single(args):
            idx, matchup = args
            seed = self.tournament_id.__hash__() + idx
            return run_match(matchup, board_type, seed)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(run_single, (i, m)): i
                for i, m in enumerate(matchups)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.db.record_match(result, self.tournament_id)

                    # Update Elo ratings
                    self._update_ratings(result)

                    if len(results) % 10 == 0:
                        logger.info(f"  Completed {len(results)}/{len(matchups)} matches")

                except Exception as e:
                    logger.error(f"Match failed: {e}")

        return results

    def _update_ratings(self, result: MatchResult):
        """Update Elo ratings after a match."""
        # Get current ratings
        ratings = []
        for pid in result.participant_ids:
            elo = self.db.get_rating(pid, result.board_type, result.num_players)
            ratings.append(elo)

        # Determine winner index
        winner_idx = None
        if result.winner_id:
            try:
                winner_idx = result.participant_ids.index(result.winner_id)
            except ValueError:
                pass

        # Calculate new ratings
        old_values = [r.rating for r in ratings]

        if result.num_players == 2:
            score_a = 1.0 if winner_idx == 0 else (0.5 if winner_idx is None else 0.0)
            new_a, new_b = update_elo_2p(old_values[0], old_values[1], score_a)
            new_values = [new_a, new_b]
        else:
            new_values = update_elo_multiplayer(old_values, winner_idx)

        # Update ratings
        for i, elo in enumerate(ratings):
            elo.rating = new_values[i]
            elo.games_played += 1
            if winner_idx is None:
                elo.draws += 1
            elif winner_idx == i:
                elo.wins += 1
            else:
                elo.losses += 1

        self.db.update_ratings(ratings)

    def run_all(self) -> List[MatchResult]:
        """Run tournament across all configurations."""
        all_results = []

        for board_type in self.board_types:
            for num_players in self.player_counts:
                results = self.run_config(board_type, num_players)
                all_results.extend(results)

        self.results = all_results
        return all_results

    def print_summary(self):
        """Print tournament summary."""
        print("\n" + "=" * 70)
        print("TOURNAMENT SUMMARY")
        print("=" * 70)
        print(f"Tournament ID: {self.tournament_id}")
        print(f"Total matches: {len(self.results)}")

        # Group by config
        by_config: Dict[Tuple[str, int], List[MatchResult]] = {}
        for r in self.results:
            key = (r.board_type, r.num_players)
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(r)

        for (bt, np), matches in sorted(by_config.items()):
            draws = sum(1 for m in matches if m.winner_id is None)
            decisive = len(matches) - draws
            avg_len = sum(m.game_length for m in matches) / len(matches)
            print(f"\n{bt} {np}p: {len(matches)} matches, {decisive} decisive, avg {avg_len:.0f} moves")


# =============================================================================
# Cluster Distribution
# =============================================================================

def load_cluster_hosts() -> List[ClusterHost]:
    """Load available cluster hosts."""
    # Known Lambda GPU hosts
    lambda_hosts = [
        ("192.222.51.167", "lambda-gh200-a"),
        ("192.222.51.162", "lambda-gh200-b"),
        ("192.222.57.162", "lambda-gh200-c"),
        ("192.222.57.178", "lambda-gh200-d"),
        ("192.222.50.112", "lambda-gh200-e"),
        ("192.222.50.210", "lambda-gh200-f"),
        ("192.222.51.29", "lambda-gh200-g"),
    ]

    hosts = []
    for ip, name in lambda_hosts:
        hosts.append(ClusterHost(
            name=name,
            ssh_host=ip,
            ssh_user="ubuntu",
            ringrift_path="~/ringrift/ai-service",
        ))

    return hosts


async def check_host(host: ClusterHost) -> bool:
    """Check if host is reachable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *host.ssh_cmd("echo ok"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0 and b"ok" in stdout
    except Exception:
        return False


async def get_available_hosts() -> List[ClusterHost]:
    """Get list of available cluster hosts."""
    hosts = load_cluster_hosts()

    logger.info(f"Checking {len(hosts)} hosts...")
    tasks = [check_host(h) for h in hosts]
    results = await asyncio.gather(*tasks)

    available = [h for h, ok in zip(hosts, results) if ok]
    logger.info(f"  {len(available)} hosts available")

    return available


# =============================================================================
# CLI
# =============================================================================

def parse_tiers(spec: str) -> List[int]:
    """Parse tier specification like 'D1-D10' or 'D2,D4,D6'."""
    spec = spec.strip().upper()

    if "-" in spec and "," not in spec:
        # Range: D1-D10
        parts = spec.split("-")
        start = int(parts[0].replace("D", ""))
        end = int(parts[1].replace("D", ""))
        return list(range(start, end + 1))
    else:
        # List: D2,D4,D6
        return [int(t.strip().replace("D", "")) for t in spec.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Unified Distributed Tournament System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration
    parser.add_argument("--board", "-b", type=str, nargs="+",
                        default=["square8"],
                        help="Board types (square8, square19, hexagonal)")
    parser.add_argument("--players", "-p", type=int, nargs="+",
                        default=[2],
                        help="Player counts (2, 3, 4)")
    parser.add_argument("--tiers", "-t", type=str, default="D1-D10",
                        help="Difficulty tiers (D1-D10 or D2,D4,D6)")
    parser.add_argument("--ai-types", type=str, nargs="+",
                        help="Specific AI types (Random, Heuristic, Minimax, MCTS, Descent)")
    parser.add_argument("--neural-only", action="store_true",
                        help="Only include neural network variants")
    parser.add_argument("--all-configs", action="store_true",
                        help="Run across all board/player configurations")

    # Execution
    parser.add_argument("--games", "-g", type=int, default=10,
                        help="Games per matchup")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel workers")
    parser.add_argument("--distributed", action="store_true",
                        help="Distribute across cluster")
    parser.add_argument("--interval", type=float,
                        help="Run continuously every N hours")

    # Output
    parser.add_argument("--db", type=str, default="data/unified_elo.db",
                        help="Elo database path")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Show leaderboard and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run")

    args = parser.parse_args()

    # Initialize database
    db = EloDatabase(args.db)

    # Show leaderboard
    if args.leaderboard:
        print("\n" + "=" * 80)
        print("ELO LEADERBOARD")
        print("=" * 80)

        for bt in ["square8", "square19", "hexagonal"]:
            for np in [2, 3, 4]:
                leaders = db.get_leaderboard(bt, np, limit=15)
                if leaders:
                    print(f"\n{bt} {np}-player:")
                    print("-" * 70)
                    print(f"{'Rank':<5} {'Name':<30} {'Elo':>8} {'Games':>7} {'Win%':>7}")
                    print("-" * 70)
                    for i, row in enumerate(leaders, 1):
                        win_rate = row['wins'] / row['games_played'] * 100 if row['games_played'] > 0 else 0
                        print(f"{i:<5} {row['name']:<30} {row['rating']:>8.1f} {row['games_played']:>7} {win_rate:>6.1f}%")
        return

    # Determine configurations
    if args.all_configs:
        board_types = [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]
        player_counts = [2, 3, 4]
    else:
        board_types = [BOARD_TYPE_MAP[b.lower()] for b in args.board]
        player_counts = args.players

    tiers = parse_tiers(args.tiers)

    # Create tournament
    tournament = UnifiedTournament(
        db=db,
        board_types=board_types,
        player_counts=player_counts,
        games_per_matchup=args.games,
        max_workers=args.workers,
    )

    # Generate participants
    for bt in board_types:
        for np in player_counts:
            if args.ai_types:
                # Specific AI types
                ai_types = [AI_TYPE_MAP[t.lower()] for t in args.ai_types]
                participants = generate_ai_type_participants(
                    bt, np, ai_types,
                    include_neural=not args.neural_only or True,
                )
            else:
                # Use ladder tiers
                participants = generate_tier_participants(bt, np, tiers)

            tournament.add_participants(bt, np, participants)
            logger.info(f"Added {len(participants)} participants for {bt.value} {np}p")

    # Dry run
    if args.dry_run:
        print("\nDRY RUN - Would execute:")
        for bt in board_types:
            for np in player_counts:
                matchups = tournament.generate_matchups(bt, np)
                print(f"  {bt.value} {np}p: {len(matchups)} matches")
        return

    # Run tournament(s)
    def run_once():
        logger.info(f"Starting tournament {tournament.tournament_id}")
        tournament.run_all()
        tournament.print_summary()

    if args.interval:
        # Continuous mode
        while True:
            run_once()
            logger.info(f"Sleeping {args.interval} hours until next run...")
            time.sleep(args.interval * 3600)
            tournament.tournament_id = str(uuid.uuid4())[:8]
    else:
        run_once()


if __name__ == "__main__":
    main()
