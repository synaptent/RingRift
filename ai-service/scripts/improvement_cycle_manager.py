#!/usr/bin/env python3
"""Improvement Cycle Manager - Bridges P2P orchestrator with AI training loop.

This module integrates with the p2p_orchestrator to coordinate:
1. Diverse selfplay scheduling with varied AI opponents
2. Training job triggers based on data thresholds
3. Tournament scheduling between diverse AI types
4. Model promotion and baseline management
5. CMA-ES weight integration for new agent variations

The manager ensures maximum positive feedback for AI self-improvement by:
- Scheduling imbalanced games (strong vs weak) for diverse training data
- Mixing AI types (MCTS, Minimax, NN, Heuristic) for varied strategies
- Tracking data distribution and prioritizing underserved configurations
- Coordinating distributed training across cluster nodes
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import BoardType


# =============================================================================
# Configuration
# =============================================================================

# Diverse AI opponent configurations for varied training data
DIVERSE_AI_CONFIGS = [
    # Type, difficulty, engine_mode, description
    {"ai_type": "random", "difficulty": 1, "engine_mode": "random-only", "weight": 0.05},
    {"ai_type": "heuristic", "difficulty": 2, "engine_mode": "heuristic-only", "weight": 0.10},
    {"ai_type": "heuristic", "difficulty": 3, "engine_mode": "heuristic-only", "weight": 0.10},
    {"ai_type": "minimax", "difficulty": 4, "engine_mode": "minimax-only", "weight": 0.10},
    {"ai_type": "minimax", "difficulty": 5, "engine_mode": "minimax-only", "weight": 0.10},
    {"ai_type": "minimax", "difficulty": 6, "engine_mode": "minimax-only", "weight": 0.10},
    {"ai_type": "mcts", "difficulty": 7, "engine_mode": "mcts-only", "weight": 0.15},
    {"ai_type": "mcts", "difficulty": 8, "engine_mode": "mcts-only", "weight": 0.10},
    {"ai_type": "descent", "difficulty": 9, "engine_mode": "descent-only", "weight": 0.10},
    {"ai_type": "neural", "difficulty": 10, "engine_mode": "nn-only", "weight": 0.10},
]

# Asymmetric game configurations for diverse search space
# Strong player index, weak player config - creates imbalanced games
ASYMMETRIC_CONFIGS = [
    # Strong MCTS vs weak Heuristic - teaches aggressive play
    {"strong": {"engine_mode": "mcts-only", "difficulty": 8},
     "weak": {"engine_mode": "heuristic-only", "difficulty": 2},
     "weight": 0.15},
    # Strong Descent vs weak Minimax - teaches deep search
    {"strong": {"engine_mode": "descent-only", "difficulty": 9},
     "weak": {"engine_mode": "minimax-only", "difficulty": 4},
     "weight": 0.15},
    # Strong NN vs weak MCTS - tests learned patterns vs search
    {"strong": {"engine_mode": "nn-only", "difficulty": 10},
     "weak": {"engine_mode": "mcts-only", "difficulty": 6},
     "weight": 0.10},
    # Medium vs Random - basic winning patterns
    {"strong": {"engine_mode": "minimax-only", "difficulty": 5},
     "weak": {"engine_mode": "random-only", "difficulty": 1},
     "weight": 0.10},
]

# Board type configurations with priority (higher = more important)
BOARD_CONFIGS = [
    # High priority: multiplayer hex and square19 (underserved)
    {"board_type": "hexagonal", "num_players": 4, "priority": 3.0, "min_games": 5000},
    {"board_type": "hexagonal", "num_players": 3, "priority": 3.0, "min_games": 5000},
    {"board_type": "square19", "num_players": 4, "priority": 3.0, "min_games": 5000},
    {"board_type": "square19", "num_players": 3, "priority": 3.0, "min_games": 5000},
    # Medium priority: multiplayer square8
    {"board_type": "square8", "num_players": 4, "priority": 2.0, "min_games": 3000},
    {"board_type": "square8", "num_players": 3, "priority": 2.0, "min_games": 3000},
    # Lower priority: 2-player (more data available)
    {"board_type": "hexagonal", "num_players": 2, "priority": 1.5, "min_games": 10000},
    {"board_type": "square19", "num_players": 2, "priority": 1.5, "min_games": 8000},
    {"board_type": "square8", "num_players": 2, "priority": 1.0, "min_games": 15000},
]

# Training thresholds
MIN_NEW_GAMES_FOR_TRAINING = 2000
TRAINING_COOLDOWN_SECONDS = 1800  # 30 min between training runs
TOURNAMENT_GAMES_PER_MATCHUP = 20
PROMOTION_THRESHOLD = 0.55  # 55% win rate for promotion

# CMA-ES integration
CMAES_TRIGGER_INTERVAL = 3600 * 6  # Run CMA-ES every 6 hours
CMAES_MIN_GAMES_SINCE_LAST = 10000  # Need this many games before next CMA-ES


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CycleState:
    """State for a specific board/player configuration."""
    board_type: str
    num_players: int
    total_games: int = 0
    games_since_last_training: int = 0
    last_training_time: float = 0.0
    last_cmaes_time: float = 0.0
    current_iteration: int = 0
    best_model_path: Optional[str] = None
    pending_training: bool = False
    pending_evaluation: bool = False


@dataclass
class AgentVariation:
    """A specific AI agent configuration for tournaments."""
    agent_id: str
    ai_type: str  # random, heuristic, minimax, mcts, descent, neural
    difficulty: int
    weights: Optional[Dict[str, float]] = None  # Custom heuristic weights
    model_path: Optional[str] = None  # NN model path
    elo_rating: float = 1500.0
    games_played: int = 0


@dataclass
class ManagerState:
    """Complete manager state."""
    cycles: Dict[str, CycleState] = field(default_factory=dict)
    agents: Dict[str, AgentVariation] = field(default_factory=dict)
    last_update: float = 0.0
    total_games_scheduled: int = 0
    total_training_triggered: int = 0
    total_tournaments_run: int = 0
    last_cmaes_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)


# =============================================================================
# Improvement Cycle Manager
# =============================================================================

class ImprovementCycleManager:
    """Manages AI improvement cycles across distributed cluster.

    Integrates with p2p_orchestrator to schedule diverse training workloads.
    """

    def __init__(self, db_path: Path, ringrift_path: Path):
        """Initialize manager with persistent storage.

        Args:
            db_path: Path to SQLite database for state persistence.
            ringrift_path: Root path of RingRift project.
        """
        self.db_path = Path(db_path)
        self.ringrift_path = Path(ringrift_path)
        self.ai_service_path = self.ringrift_path / "ai-service"
        self.models_path = self.ai_service_path / "models"
        self.data_path = self.ai_service_path / "data"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.state = self._load_state()

        # Register built-in agent variations
        self._register_builtin_agents()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_counts (
                board_type TEXT,
                num_players INTEGER,
                engine_mode TEXT,
                difficulty INTEGER,
                game_count INTEGER,
                last_updated REAL,
                PRIMARY KEY (board_type, num_players, engine_mode, difficulty)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_type TEXT,
                num_players INTEGER,
                iteration INTEGER,
                started_at REAL,
                completed_at REAL,
                games_used INTEGER,
                model_path TEXT,
                promoted INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournament_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_a TEXT,
                agent_b TEXT,
                board_type TEXT,
                num_players INTEGER,
                wins_a INTEGER,
                wins_b INTEGER,
                draws INTEGER,
                played_at REAL
            )
        """)

        conn.commit()
        conn.close()

    def _load_state(self) -> ManagerState:
        """Load state from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM state WHERE key = 'manager_state'")
        row = cursor.fetchone()
        conn.close()

        if row:
            try:
                data = json.loads(row[0])
                state = ManagerState()
                state.last_update = data.get("last_update", 0.0)
                state.total_games_scheduled = data.get("total_games_scheduled", 0)
                state.total_training_triggered = data.get("total_training_triggered", 0)
                state.total_tournaments_run = data.get("total_tournaments_run", 0)
                state.last_cmaes_weights = data.get("last_cmaes_weights", {})

                for key, cycle_data in data.get("cycles", {}).items():
                    state.cycles[key] = CycleState(**cycle_data)

                for agent_id, agent_data in data.get("agents", {}).items():
                    state.agents[agent_id] = AgentVariation(**agent_data)

                return state
            except Exception as e:
                print(f"[ImprovementManager] Error loading state: {e}")

        return ManagerState()

    def _save_state(self):
        """Save state to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = {
            "last_update": time.time(),
            "total_games_scheduled": self.state.total_games_scheduled,
            "total_training_triggered": self.state.total_training_triggered,
            "total_tournaments_run": self.state.total_tournaments_run,
            "last_cmaes_weights": self.state.last_cmaes_weights,
            "cycles": {k: asdict(v) for k, v in self.state.cycles.items()},
            "agents": {k: asdict(v) for k, v in self.state.agents.items()},
        }

        cursor.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            ("manager_state", json.dumps(data))
        )

        conn.commit()
        conn.close()

    def _register_builtin_agents(self):
        """Register built-in agent variations for diverse tournaments."""
        builtin = [
            AgentVariation("random_d1", "random", 1),
            AgentVariation("heuristic_d2", "heuristic", 2),
            AgentVariation("heuristic_d3", "heuristic", 3),
            AgentVariation("minimax_d4", "minimax", 4),
            AgentVariation("minimax_d5", "minimax", 5),
            AgentVariation("minimax_d6", "minimax", 6),
            AgentVariation("mcts_d7", "mcts", 7),
            AgentVariation("mcts_d8", "mcts", 8),
            AgentVariation("descent_d9", "descent", 9),
            AgentVariation("neural_d10", "neural", 10),
        ]

        for agent in builtin:
            if agent.agent_id not in self.state.agents:
                self.state.agents[agent.agent_id] = agent

    def _get_cycle_key(self, board_type: str, num_players: int) -> str:
        """Get unique key for board/player combination."""
        return f"{board_type}_{num_players}p"

    def _ensure_cycle_state(self, board_type: str, num_players: int) -> CycleState:
        """Ensure cycle state exists for configuration."""
        key = self._get_cycle_key(board_type, num_players)
        if key not in self.state.cycles:
            self.state.cycles[key] = CycleState(board_type, num_players)
        return self.state.cycles[key]

    # =========================================================================
    # Selfplay Scheduling
    # =========================================================================

    def get_next_selfplay_config(self, cluster_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Get next selfplay configuration based on data needs and diversity.

        Returns a configuration dict with:
        - board_type: Board type to use
        - num_players: Number of players
        - engine_mode: AI engine mode
        - difficulty_band: Difficulty band for mixed mode
        - asymmetric: Whether to use asymmetric opponents
        - strong_config: Config for strong player (if asymmetric)
        - weak_config: Config for weak player (if asymmetric)

        Args:
            cluster_data: Optional manifest of existing data distribution.
        """
        # Calculate weights for each board config based on deficit
        weighted_configs = []

        for config in BOARD_CONFIGS:
            key = self._get_cycle_key(config["board_type"], config["num_players"])
            cycle = self.state.cycles.get(key)
            current_games = cycle.total_games if cycle else 0

            # Higher weight for configs with larger deficit
            deficit_ratio = max(0, 1 - current_games / config["min_games"])
            weight = config["priority"] * (1 + deficit_ratio * 2)
            weighted_configs.append((config, weight))

        # Normalize weights
        total_weight = sum(w for _, w in weighted_configs)
        if total_weight > 0:
            weighted_configs = [(c, w / total_weight) for c, w in weighted_configs]

        # Select board config
        rand = random.random()
        cumulative = 0
        selected_board = weighted_configs[0][0]
        for config, weight in weighted_configs:
            cumulative += weight
            if rand <= cumulative:
                selected_board = config
                break

        # Decide on symmetric vs asymmetric game
        use_asymmetric = random.random() < 0.3  # 30% asymmetric games

        if use_asymmetric:
            # Select asymmetric configuration
            asym_weights = [c["weight"] for c in ASYMMETRIC_CONFIGS]
            total_asym = sum(asym_weights)
            asym_weights = [w / total_asym for w in asym_weights]

            rand = random.random()
            cumulative = 0
            selected_asym = ASYMMETRIC_CONFIGS[0]
            for i, weight in enumerate(asym_weights):
                cumulative += weight
                if rand <= cumulative:
                    selected_asym = ASYMMETRIC_CONFIGS[i]
                    break

            return {
                "board_type": selected_board["board_type"],
                "num_players": selected_board["num_players"],
                "engine_mode": "mixed",  # Mixed for asymmetric
                "asymmetric": True,
                "strong_config": selected_asym["strong"],
                "weak_config": selected_asym["weak"],
            }
        else:
            # Select diverse AI config
            ai_weights = [c["weight"] for c in DIVERSE_AI_CONFIGS]
            total_ai = sum(ai_weights)
            ai_weights = [w / total_ai for w in ai_weights]

            rand = random.random()
            cumulative = 0
            selected_ai = DIVERSE_AI_CONFIGS[0]
            for i, weight in enumerate(ai_weights):
                cumulative += weight
                if rand <= cumulative:
                    selected_ai = DIVERSE_AI_CONFIGS[i]
                    break

            return {
                "board_type": selected_board["board_type"],
                "num_players": selected_board["num_players"],
                "engine_mode": selected_ai["engine_mode"],
                "difficulty_band": "canonical",
                "asymmetric": False,
            }

    def get_diverse_selfplay_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Get a batch of diverse selfplay configurations.

        Ensures variety across AI types, difficulties, and board configurations.
        """
        configs = []
        used_keys = set()

        for _ in range(batch_size):
            config = self.get_next_selfplay_config()

            # Try to avoid duplicates in the batch
            key = f"{config['board_type']}_{config['num_players']}_{config['engine_mode']}"
            attempts = 0
            while key in used_keys and attempts < 5:
                config = self.get_next_selfplay_config()
                key = f"{config['board_type']}_{config['num_players']}_{config['engine_mode']}"
                attempts += 1

            used_keys.add(key)
            configs.append(config)

        return configs

    def record_games_completed(self, board_type: str, num_players: int,
                               games: int, engine_mode: str = "mixed",
                               difficulty: int = 0):
        """Record completed selfplay games."""
        cycle = self._ensure_cycle_state(board_type, num_players)
        cycle.total_games += games
        cycle.games_since_last_training += games

        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO game_counts
            (board_type, num_players, engine_mode, difficulty, game_count, last_updated)
            VALUES (?, ?, ?, ?,
                COALESCE((SELECT game_count FROM game_counts
                    WHERE board_type=? AND num_players=? AND engine_mode=? AND difficulty=?), 0) + ?,
                ?)
        """, (board_type, num_players, engine_mode, difficulty,
              board_type, num_players, engine_mode, difficulty,
              games, time.time()))
        conn.commit()
        conn.close()

        self.state.total_games_scheduled += games
        self._save_state()

    # =========================================================================
    # Training Triggers
    # =========================================================================

    def check_training_needed(self) -> List[Tuple[str, int]]:
        """Check which configurations need training.

        Returns list of (board_type, num_players) tuples ready for training.
        """
        ready = []
        now = time.time()

        for key, cycle in self.state.cycles.items():
            # Skip if pending training/evaluation
            if cycle.pending_training or cycle.pending_evaluation:
                continue

            # Skip if in cooldown
            if now - cycle.last_training_time < TRAINING_COOLDOWN_SECONDS:
                continue

            # Check if enough new games
            if cycle.games_since_last_training >= MIN_NEW_GAMES_FOR_TRAINING:
                ready.append((cycle.board_type, cycle.num_players))

        return ready

    def trigger_training(self, board_type: str, num_players: int) -> bool:
        """Mark training as triggered for configuration."""
        cycle = self._ensure_cycle_state(board_type, num_players)
        if cycle.pending_training:
            return False

        cycle.pending_training = True
        self.state.total_training_triggered += 1
        self._save_state()
        return True

    def handle_training_complete(self, board_type: str, num_players: int,
                                  model_path: str, games_used: int):
        """Handle training completion."""
        cycle = self._ensure_cycle_state(board_type, num_players)
        cycle.pending_training = False
        cycle.pending_evaluation = True
        cycle.last_training_time = time.time()
        cycle.games_since_last_training = 0
        cycle.current_iteration += 1

        # Record in history
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO training_history
            (board_type, num_players, iteration, started_at, completed_at, games_used, model_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (board_type, num_players, cycle.current_iteration,
              cycle.last_training_time, time.time(), games_used, model_path))
        conn.commit()
        conn.close()

        self._save_state()

    # =========================================================================
    # Tournament Scheduling
    # =========================================================================

    def get_tournament_matchups(self, board_type: str, num_players: int,
                                 new_model_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get diverse tournament matchups for evaluation.

        Includes:
        1. New model vs current best (if new model provided)
        2. New model vs diverse AI opponents
        3. Cross-AI-type matchups for calibration
        """
        matchups = []
        key = self._get_cycle_key(board_type, num_players)
        cycle = self.state.cycles.get(key)

        # 1. New model vs best (primary evaluation)
        if new_model_path and cycle and cycle.best_model_path:
            matchups.append({
                "agent_a": {"type": "neural", "model_path": new_model_path},
                "agent_b": {"type": "neural", "model_path": cycle.best_model_path},
                "games": TOURNAMENT_GAMES_PER_MATCHUP,
                "purpose": "primary_evaluation",
            })

        # 2. New model vs diverse AI opponents
        if new_model_path:
            diverse_opponents = [
                {"type": "random", "difficulty": 1},
                {"type": "heuristic", "difficulty": 3},
                {"type": "minimax", "difficulty": 5},
                {"type": "mcts", "difficulty": 7},
            ]
            for opponent in diverse_opponents:
                matchups.append({
                    "agent_a": {"type": "neural", "model_path": new_model_path},
                    "agent_b": opponent,
                    "games": TOURNAMENT_GAMES_PER_MATCHUP // 2,
                    "purpose": "diverse_calibration",
                })

        # 3. Cross-AI-type matchups (every 5th tournament)
        if self.state.total_tournaments_run % 5 == 0:
            cross_matchups = [
                ("mcts_d8", "minimax_d5"),
                ("descent_d9", "mcts_d7"),
                ("heuristic_d3", "random_d1"),
            ]
            for agent_a_id, agent_b_id in cross_matchups:
                agent_a = self.state.agents.get(agent_a_id)
                agent_b = self.state.agents.get(agent_b_id)
                if agent_a and agent_b:
                    matchups.append({
                        "agent_a": {"type": agent_a.ai_type, "difficulty": agent_a.difficulty},
                        "agent_b": {"type": agent_b.ai_type, "difficulty": agent_b.difficulty},
                        "games": TOURNAMENT_GAMES_PER_MATCHUP // 2,
                        "purpose": "cross_type_calibration",
                    })

        return matchups

    def record_tournament_result(self, agent_a: str, agent_b: str,
                                  board_type: str, num_players: int,
                                  wins_a: int, wins_b: int, draws: int):
        """Record tournament result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tournament_results
            (agent_a, agent_b, board_type, num_players, wins_a, wins_b, draws, played_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (agent_a, agent_b, board_type, num_players,
              wins_a, wins_b, draws, time.time()))
        conn.commit()
        conn.close()

        self.state.total_tournaments_run += 1
        self._save_state()

    def handle_evaluation_complete(self, board_type: str, num_players: int,
                                     win_rate: float, model_path: str):
        """Handle evaluation completion and decide on promotion."""
        cycle = self._ensure_cycle_state(board_type, num_players)
        cycle.pending_evaluation = False

        promoted = win_rate >= PROMOTION_THRESHOLD

        if promoted:
            # Backup previous best
            if cycle.best_model_path and Path(cycle.best_model_path).exists():
                backup_path = str(cycle.best_model_path).replace("_best.", "_prev_best.")
                try:
                    import shutil
                    shutil.copy2(cycle.best_model_path, backup_path)
                except Exception as e:
                    print(f"[ImprovementManager] Backup failed: {e}")

            # Promote new model
            cycle.best_model_path = model_path

            # Update training history
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_history SET promoted = 1
                WHERE board_type = ? AND num_players = ? AND model_path = ?
            """, (board_type, num_players, model_path))
            conn.commit()
            conn.close()

        self._save_state()
        return promoted

    def update_cycle_phase(self, cycle_id: str, phase: str,
                            training_job_id: Optional[str] = None,
                            error_message: Optional[str] = None):
        """Update cycle phase for p2p_orchestrator compatibility.

        Args:
            cycle_id: Cycle identifier (e.g., "square8_2p")
            phase: New phase ("idle", "training", "evaluating", "error")
            training_job_id: Optional job ID if in training phase
            error_message: Optional error message if in error phase
        """
        if cycle_id not in self.state.cycles:
            # Parse cycle_id to create cycle state
            parts = cycle_id.replace("_", " ").split()
            if len(parts) >= 2:
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))
                self._ensure_cycle_state(board_type, num_players)
            else:
                return

        cycle = self.state.cycles.get(cycle_id)
        if not cycle:
            return

        if phase == "idle":
            cycle.pending_training = False
            cycle.pending_evaluation = False
        elif phase == "training":
            cycle.pending_training = True
            cycle.pending_evaluation = False
        elif phase == "evaluating":
            cycle.pending_training = False
            cycle.pending_evaluation = True
        elif phase == "error":
            cycle.pending_training = False
            cycle.pending_evaluation = False

        if error_message:
            print(f"[ImprovementManager] Cycle {cycle_id} error: {error_message}")

        self._save_state()

    # =========================================================================
    # CMA-ES Integration
    # =========================================================================

    def check_cmaes_needed(self) -> List[Tuple[str, int]]:
        """Check which configurations need CMA-ES optimization."""
        ready = []
        now = time.time()

        for key, cycle in self.state.cycles.items():
            # Check time since last CMA-ES
            if now - cycle.last_cmaes_time < CMAES_TRIGGER_INTERVAL:
                continue

            # Check games since last CMA-ES
            games_since = cycle.total_games  # Simplified - track properly in production
            if games_since >= CMAES_MIN_GAMES_SINCE_LAST:
                ready.append((cycle.board_type, cycle.num_players))

        return ready

    def handle_cmaes_complete(self, board_type: str, num_players: int,
                               best_weights: Dict[str, float]):
        """Handle CMA-ES optimization completion.

        Creates new agent variation with optimized weights.
        """
        cycle = self._ensure_cycle_state(board_type, num_players)
        cycle.last_cmaes_time = time.time()

        # Store weights
        key = self._get_cycle_key(board_type, num_players)
        self.state.last_cmaes_weights[key] = best_weights

        # Create new agent variation
        agent_id = f"cmaes_{board_type}_{num_players}p_{int(time.time())}"
        agent = AgentVariation(
            agent_id=agent_id,
            ai_type="heuristic",
            difficulty=5,  # Mid-tier
            weights=best_weights,
            elo_rating=1500.0,
        )
        self.state.agents[agent_id] = agent

        self._save_state()
        return agent_id

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return {
            "last_update": self.state.last_update,
            "total_games_scheduled": self.state.total_games_scheduled,
            "total_training_triggered": self.state.total_training_triggered,
            "total_tournaments_run": self.state.total_tournaments_run,
            "cycles": {
                k: {
                    "board_type": v.board_type,
                    "num_players": v.num_players,
                    "total_games": v.total_games,
                    "games_since_training": v.games_since_last_training,
                    "iteration": v.current_iteration,
                    "best_model": v.best_model_path,
                    "pending_training": v.pending_training,
                    "pending_evaluation": v.pending_evaluation,
                }
                for k, v in self.state.cycles.items()
            },
            "agents_count": len(self.state.agents),
            "training_ready": self.check_training_needed(),
            "cmaes_ready": self.check_cmaes_needed(),
        }

    def get_data_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of training data across configurations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT board_type, num_players, engine_mode, SUM(game_count) as total
            FROM game_counts
            GROUP BY board_type, num_players, engine_mode
        """)

        distribution = {}
        for row in cursor.fetchall():
            key = f"{row[0]}_{row[1]}p"
            if key not in distribution:
                distribution[key] = {}
            distribution[key][row[2]] = row[3]

        conn.close()
        return distribution


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI for testing improvement cycle manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Improvement Cycle Manager")
    parser.add_argument("--db", default="logs/improvement_manager.db",
                        help="Path to state database")
    parser.add_argument("--status", action="store_true",
                        help="Print current status")
    parser.add_argument("--next-config", action="store_true",
                        help="Get next selfplay configuration")
    parser.add_argument("--batch", type=int, default=0,
                        help="Get batch of diverse configs")

    args = parser.parse_args()

    ai_service_path = Path(__file__).resolve().parents[1]
    ringrift_path = ai_service_path.parent

    manager = ImprovementCycleManager(
        db_path=ai_service_path / args.db,
        ringrift_path=ringrift_path,
    )

    if args.status:
        status = manager.get_status()
        print(json.dumps(status, indent=2))

    elif args.next_config:
        config = manager.get_next_selfplay_config()
        print(json.dumps(config, indent=2))

    elif args.batch > 0:
        configs = manager.get_diverse_selfplay_batch(args.batch)
        for i, config in enumerate(configs):
            print(f"{i+1}. {json.dumps(config)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
