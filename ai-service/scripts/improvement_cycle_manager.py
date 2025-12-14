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

# Training thresholds (tunable via environment variables)
MIN_NEW_GAMES_FOR_TRAINING = int(os.environ.get("RINGRIFT_MIN_GAMES_FOR_TRAINING", "1000"))
TRAINING_COOLDOWN_SECONDS = int(os.environ.get("RINGRIFT_TRAINING_COOLDOWN_SECONDS", "900"))  # 15 min between training runs
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
    prev_best_model_path: Optional[str] = None  # For rollback on consecutive failures
    pending_training: bool = False
    pending_evaluation: bool = False
    consecutive_failures: int = 0  # Track consecutive failed promotions
    max_consecutive_failures: int = 5  # Rollback after this many failures


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
        # Ensure canonical cycles exist for all tracked configs so the manager can
        # surface status immediately (and training triggers have something to scan).
        for config in BOARD_CONFIGS:
            self._ensure_cycle_state(config["board_type"], config["num_players"])
        self._save_state()

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

        # New table for incremental cycle state persistence (crash recovery)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cycle_states (
                cycle_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                total_games INTEGER DEFAULT 0,
                games_since_last_training INTEGER DEFAULT 0,
                last_training_time REAL DEFAULT 0,
                last_cmaes_time REAL DEFAULT 0,
                current_iteration INTEGER DEFAULT 0,
                best_model_path TEXT,
                best_winrate REAL DEFAULT 0,
                pending_training INTEGER DEFAULT 0,
                pending_evaluation INTEGER DEFAULT 0,
                training_job_id TEXT,
                last_update REAL DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def _load_state(self) -> ManagerState:
        """Load state from database.

        Loads from both the JSON blob (legacy) and cycle_states table.
        The cycle_states table takes precedence for cycle data if present,
        enabling crash recovery.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM state WHERE key = 'manager_state'")
        row = cursor.fetchone()
        conn.close()

        state = ManagerState()

        if row:
            try:
                data = json.loads(row[0])
                state.last_update = data.get("last_update", 0.0)
                state.total_games_scheduled = data.get("total_games_scheduled", 0)
                state.total_training_triggered = data.get("total_training_triggered", 0)
                state.total_tournaments_run = data.get("total_tournaments_run", 0)
                state.last_cmaes_weights = data.get("last_cmaes_weights", {})

                for key, cycle_data in data.get("cycles", {}).items():
                    state.cycles[key] = CycleState(**cycle_data)

                for agent_id, agent_data in data.get("agents", {}).items():
                    state.agents[agent_id] = AgentVariation(**agent_data)
            except Exception as e:
                print(f"[ImprovementManager] Error loading state from JSON blob: {e}")

        # Load from cycle_states table (takes precedence, enables crash recovery)
        try:
            db_cycles = self._load_cycle_states_from_db()
            if db_cycles:
                # Merge: DB table takes precedence for existing keys
                for key, cycle in db_cycles.items():
                    state.cycles[key] = cycle
                print(f"[ImprovementManager] Loaded {len(db_cycles)} cycles from DB table")
        except Exception as e:
            print(f"[ImprovementManager] Error loading cycles from DB table: {e}")

        return state

    def _save_state(self):
        """Save state to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        self.state.last_update = time.time()
        data = {
            "last_update": self.state.last_update,
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

        # Also persist cycle states to dedicated table for crash recovery
        for key, cycle in self.state.cycles.items():
            self._save_cycle_state_to_db(cursor, key, cycle)

        conn.commit()
        conn.close()

    def _save_cycle_state_to_db(self, cursor, cycle_id: str, cycle: "CycleState"):
        """Save a single cycle state to the dedicated table."""
        cursor.execute("""
            INSERT OR REPLACE INTO cycle_states
            (cycle_id, board_type, num_players, total_games, games_since_last_training,
             last_training_time, last_cmaes_time, current_iteration, best_model_path,
             best_winrate, pending_training, pending_evaluation, training_job_id, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cycle_id,
            cycle.board_type,
            cycle.num_players,
            cycle.total_games,
            cycle.games_since_last_training,
            cycle.last_training_time,
            getattr(cycle, 'last_cmaes_time', 0),
            cycle.current_iteration,
            cycle.best_model_path,
            getattr(cycle, 'best_winrate', 0),
            1 if cycle.pending_training else 0,
            1 if cycle.pending_evaluation else 0,
            getattr(cycle, 'training_job_id', None),
            time.time(),
        ))

    def _load_cycle_states_from_db(self) -> Dict[str, "CycleState"]:
        """Load cycle states from dedicated table for crash recovery."""
        cycles = {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT cycle_id, board_type, num_players, total_games, games_since_last_training,
                       last_training_time, last_cmaes_time, current_iteration, best_model_path,
                       best_winrate, pending_training, pending_evaluation, training_job_id
                FROM cycle_states
            """)
            for row in cursor.fetchall():
                (cycle_id, board_type, num_players, total_games, games_since,
                 last_train, last_cmaes, iteration, best_model, best_wr,
                 pending_train, pending_eval, job_id) = row

                cycle = CycleState(board_type, num_players)
                cycle.total_games = total_games or 0
                cycle.games_since_last_training = games_since or 0
                cycle.last_training_time = last_train or 0
                cycle.current_iteration = iteration or 0
                cycle.best_model_path = best_model
                cycle.pending_training = bool(pending_train)
                cycle.pending_evaluation = bool(pending_eval)
                # Optional fields that may not exist on older CycleState
                if hasattr(cycle, 'last_cmaes_time'):
                    cycle.last_cmaes_time = last_cmaes or 0
                if hasattr(cycle, 'best_winrate'):
                    cycle.best_winrate = best_wr or 0
                if hasattr(cycle, 'training_job_id'):
                    cycle.training_job_id = job_id

                cycles[cycle_id] = cycle
        except Exception as e:
            print(f"[ImprovementManager] Error loading cycle states: {e}")
        finally:
            conn.close()

        return cycles

    def save_cycle_state(self, board_type: str, num_players: int):
        """Persist a single cycle state immediately (for critical updates)."""
        key = self._get_cycle_key(board_type, num_players)
        cycle = self.state.cycles.get(key)
        if not cycle:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            self._save_cycle_state_to_db(cursor, key, cycle)
            conn.commit()
        finally:
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

    def update_from_cluster_totals(self, cluster_totals: Dict[str, Any]) -> bool:
        """Update per-cycle game totals from a cluster totals dict.

        Expects the `ClusterDataManifest.by_board_type` shape from `p2p_orchestrator.py`:
            {\"square8_2p\": {\"total_games\": 123, \"nodes\": [...]}, ...}

        Returns True if any state changed.
        """
        changed = False
        totals = cluster_totals or {}

        for config in BOARD_CONFIGS:
            board_type = config["board_type"]
            num_players = int(config["num_players"])
            key = self._get_cycle_key(board_type, num_players)

            cycle = self._ensure_cycle_state(board_type, num_players)
            entry = totals.get(key) or {}

            try:
                total_games = int(entry.get("total_games", 0) or 0)
            except Exception:
                total_games = 0
            if total_games < 0:
                total_games = 0

            # Baseline sync: initialize totals without immediately triggering
            # training from historical data.
            baseline = (
                cycle.total_games == 0
                and cycle.games_since_last_training == 0
                and cycle.current_iteration == 0
                and float(cycle.last_training_time or 0.0) <= 0.0
                and not cycle.pending_training
                and not cycle.pending_evaluation
            )
            if baseline:
                if cycle.total_games != total_games:
                    cycle.total_games = total_games
                    changed = True
                continue

            delta = total_games - cycle.total_games
            if delta > 0:
                cycle.total_games = total_games
                cycle.games_since_last_training += delta
                changed = True
            elif delta < 0:
                # Data may have been cleaned up; reflect the new total but do not
                # decrement games_since_last_training.
                cycle.total_games = total_games
                changed = True

        if changed:
            self._save_state()

        return changed

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

    def get_next_selfplay_config_for_node(
        self,
        node_gpu_power: float,
        node_memory_gb: int,
        cluster_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Get next selfplay config optimized for a specific node's hardware.

        Routes heavy workloads (hex, square19, 3p/4p) to powerful nodes and
        light workloads (square8, 2p) to lighter nodes.

        Args:
            node_gpu_power: GPU power score (0 = CPU only, higher = more powerful)
            node_memory_gb: Node memory in GB
            cluster_data: Optional manifest of existing data distribution.

        Returns:
            Configuration dict matching get_next_selfplay_config output.
        """
        # Classify node capability tier
        # Tier 3: Heavy (GH200, H100, 5090, >128GB memory) - gets hex/sq19/3p/4p preferentially
        # Tier 2: Medium (3080/3090, 64-128GB) - balanced mix
        # Tier 1: Light (<64GB, weak GPU) - gets sq8/2p preferentially
        if node_gpu_power >= 80 or node_memory_gb >= 128:
            tier = 3
        elif node_gpu_power >= 30 or node_memory_gb >= 64:
            tier = 2
        else:
            tier = 1

        # Filter BOARD_CONFIGS based on tier
        if tier == 3:
            # Heavy nodes: prefer heavy workloads
            # Boost priority for hex/sq19 and 3p/4p
            tier_configs = []
            for config in BOARD_CONFIGS:
                weight = config["priority"]
                if config["board_type"] in ("hexagonal", "square19"):
                    weight *= 3.0  # 3x boost for large boards
                if config["num_players"] >= 3:
                    weight *= 2.0  # 2x boost for multiplayer
                tier_configs.append((config, weight))
        elif tier == 2:
            # Medium nodes: balanced weights
            tier_configs = [(config, config["priority"]) for config in BOARD_CONFIGS]
        else:
            # Light nodes: prefer sq8/2p, filter out heavy workloads
            tier_configs = []
            for config in BOARD_CONFIGS:
                if config["board_type"] == "square8":
                    weight = config["priority"] * 3.0  # 3x boost for sq8
                    if config["num_players"] == 2:
                        weight *= 2.0  # Additional 2x for 2p
                    tier_configs.append((config, weight))
                elif config["num_players"] == 2 and node_memory_gb >= 32:
                    # Allow hex/sq19 2p only if node has reasonable memory
                    tier_configs.append((config, config["priority"] * 0.5))

            # Fallback if no configs matched
            if not tier_configs:
                tier_configs = [
                    ({"board_type": "square8", "num_players": 2, "priority": 1.0, "min_games": 15000}, 1.0)
                ]

        # Apply data deficit weighting (same as get_next_selfplay_config)
        weighted_configs = []
        for config, tier_weight in tier_configs:
            key = self._get_cycle_key(config["board_type"], config["num_players"])
            cycle = self.state.cycles.get(key)
            current_games = cycle.total_games if cycle else 0
            deficit_ratio = max(0, 1 - current_games / config.get("min_games", 5000))
            weight = tier_weight * (1 + deficit_ratio * 2)
            weighted_configs.append((config, weight))

        # Normalize and select
        total_weight = sum(w for _, w in weighted_configs)
        if total_weight > 0:
            weighted_configs = [(c, w / total_weight) for c, w in weighted_configs]

        rand = random.random()
        cumulative = 0
        selected_board = weighted_configs[0][0]
        for config, weight in weighted_configs:
            cumulative += weight
            if rand <= cumulative:
                selected_board = config
                break

        # Use same AI selection logic as base method
        use_asymmetric = random.random() < 0.3

        if use_asymmetric:
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
                "engine_mode": "mixed",
                "asymmetric": True,
                "strong_config": selected_asym["strong"],
                "weak_config": selected_asym["weak"],
            }
        else:
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

        # Save full state and immediately persist this cycle for crash recovery
        self._save_state()
        self.save_cycle_state(board_type, num_players)

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
            # Store previous best for rollback
            cycle.prev_best_model_path = cycle.best_model_path

            # Backup previous best to file
            if cycle.best_model_path and Path(cycle.best_model_path).exists():
                backup_path = str(cycle.best_model_path).replace("_best.", "_prev_best.")
                try:
                    import shutil
                    shutil.copy2(cycle.best_model_path, backup_path)
                except Exception as e:
                    print(f"[ImprovementManager] Backup failed: {e}")

            # Promote new model
            cycle.best_model_path = model_path
            cycle.consecutive_failures = 0  # Reset failure counter on success

            # Update training history
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_history SET promoted = 1
                WHERE board_type = ? AND num_players = ? AND model_path = ?
            """, (board_type, num_players, model_path))
            conn.commit()
            conn.close()

            print(f"[ImprovementManager] {board_type}_{num_players}p: Promoted model with {win_rate:.1%} win rate")
        else:
            # Track consecutive failures
            cycle.consecutive_failures += 1
            print(f"[ImprovementManager] {board_type}_{num_players}p: Model not promoted ({win_rate:.1%} < {PROMOTION_THRESHOLD:.1%}), "
                  f"consecutive failures: {cycle.consecutive_failures}/{cycle.max_consecutive_failures}")

            # Check if rollback is needed
            if cycle.consecutive_failures >= cycle.max_consecutive_failures:
                self._rollback_to_previous_best(cycle)

        # Save full state and immediately persist this cycle for crash recovery
        self._save_state()
        self.save_cycle_state(board_type, num_players)
        return promoted

    def _rollback_to_previous_best(self, cycle: CycleState):
        """Rollback to previous best model after consecutive failures."""
        if not cycle.prev_best_model_path:
            print(f"[ImprovementManager] {cycle.board_type}_{cycle.num_players}p: No previous model to rollback to")
            cycle.consecutive_failures = 0  # Reset to avoid infinite loop
            return

        print(f"[ImprovementManager] {cycle.board_type}_{cycle.num_players}p: ROLLBACK - "
              f"{cycle.consecutive_failures} consecutive failures, reverting to previous best model")

        # Swap current and previous
        cycle.best_model_path = cycle.prev_best_model_path
        cycle.prev_best_model_path = None
        cycle.consecutive_failures = 0

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

    # =========================================================================
    # Training Data Quality Tracking
    # =========================================================================

    def get_training_quality_metrics(self) -> Dict[str, Any]:
        """Get training data quality metrics for monitoring.

        Returns metrics about:
        - AI type diversity in training data
        - Difficulty level distribution
        - Asymmetric vs symmetric game ratio
        - Game length statistics
        - Win rate balance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metrics = {
            "ai_type_diversity": {},
            "difficulty_distribution": {},
            "asymmetric_ratio": 0.0,
            "total_games": 0,
            "games_by_config": {},
        }

        # Get AI type diversity (from engine_mode)
        cursor.execute("""
            SELECT engine_mode, SUM(game_count) as total
            FROM game_counts
            GROUP BY engine_mode
        """)
        engine_counts = {}
        total_games = 0
        for row in cursor.fetchall():
            engine_counts[row[0]] = row[1]
            total_games += row[1]

        metrics["total_games"] = total_games
        if total_games > 0:
            metrics["ai_type_diversity"] = {
                k: v / total_games for k, v in engine_counts.items()
            }

        # Get config distribution
        cursor.execute("""
            SELECT board_type, num_players, SUM(game_count) as total
            FROM game_counts
            GROUP BY board_type, num_players
        """)
        for row in cursor.fetchall():
            key = f"{row[0]}_{row[1]}p"
            metrics["games_by_config"][key] = row[2]

        conn.close()

        # Calculate diversity score (0-1, higher is more diverse)
        if metrics["ai_type_diversity"]:
            # Shannon entropy normalized to 0-1
            import math
            entropy = -sum(p * math.log(p) for p in metrics["ai_type_diversity"].values() if p > 0)
            max_entropy = math.log(len(metrics["ai_type_diversity"]))
            metrics["diversity_score"] = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            metrics["diversity_score"] = 0.0

        return metrics

    # =========================================================================
    # Automatic Rollback Detection
    # =========================================================================

    def check_rollback_needed(self, board_type: str, num_players: int,
                               max_consecutive_failures: int = 5) -> Tuple[bool, str]:
        """Check if rollback is needed due to consecutive training failures.

        Returns (should_rollback, reason).
        """
        key = self._get_cycle_key(board_type, num_players)
        cycle = self.state.cycles.get(key)

        if not cycle:
            return False, "No cycle state"

        # Check training history for consecutive failures
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT promoted, completed_at
            FROM training_history
            WHERE board_type = ? AND num_players = ?
            ORDER BY completed_at DESC
            LIMIT ?
        """, (board_type, num_players, max_consecutive_failures))

        results = cursor.fetchall()
        conn.close()

        if len(results) < max_consecutive_failures:
            return False, f"Only {len(results)} training runs, need {max_consecutive_failures}"

        # Check if all recent runs failed to promote
        failures = sum(1 for r in results if r[0] == 0)
        if failures >= max_consecutive_failures:
            return True, f"{failures} consecutive training failures without promotion"

        return False, "Training improving normally"

    def execute_rollback(self, board_type: str, num_players: int) -> bool:
        """Execute rollback to previous best model.

        Returns True if rollback succeeded.
        """
        key = self._get_cycle_key(board_type, num_players)
        cycle = self.state.cycles.get(key)

        if not cycle or not cycle.best_model_path:
            print(f"[ImprovementManager] No model to rollback for {key}")
            return False

        # Find previous best model
        prev_best_path = str(cycle.best_model_path).replace("_best.", "_prev_best.")
        prev_best = Path(prev_best_path)

        if not prev_best.exists():
            print(f"[ImprovementManager] No previous best model at {prev_best}")
            return False

        try:
            import shutil
            current_best = Path(cycle.best_model_path)

            # Backup current (failed) best
            failed_backup = str(current_best).replace("_best.", f"_failed_{int(time.time())}.")
            if current_best.exists():
                shutil.copy2(current_best, failed_backup)

            # Restore previous best
            shutil.copy2(prev_best, current_best)

            print(f"[ImprovementManager] Rollback complete: {prev_best} -> {current_best}")

            # Reset cycle state
            cycle.pending_training = False
            cycle.pending_evaluation = False
            self._save_state()

            return True
        except Exception as e:
            print(f"[ImprovementManager] Rollback failed: {e}")
            return False


# =============================================================================
# Health Monitoring & Process Management
# =============================================================================

# Health monitoring thresholds
HEALTH_CHECK_LOG_STALE_SECONDS = 300  # Alert if log not updated in 5 min
HEALTH_CHECK_PROGRESS_RATE_MIN = 0.1  # Alert if progress < 10% of expected
HEALTH_CHECK_CPU_VS_PROGRESS_THRESHOLD = 10  # High CPU but low progress ratio


@dataclass
class ProcessHealth:
    """Health status for a running process."""
    process_name: str
    pid: Optional[int]
    log_file: Optional[str]
    log_mtime: Optional[float]
    is_stale: bool
    cpu_time_seconds: float
    progress_rate: float  # e.g., epochs/hour or games/hour
    expected_rate: float
    is_slow: bool
    status: str  # "healthy", "stale", "slow", "stuck"


def check_log_staleness(log_file: str, threshold_seconds: float = HEALTH_CHECK_LOG_STALE_SECONDS) -> Tuple[bool, float]:
    """Check if a log file has become stale (no updates).

    Returns:
        Tuple of (is_stale, seconds_since_update)
    """
    if not os.path.exists(log_file):
        return True, float('inf')

    mtime = os.path.getmtime(log_file)
    age = time.time() - mtime
    return age > threshold_seconds, age


def check_training_process_health(
    log_file: str,
    expected_epochs_per_hour: float = 20.0,
) -> ProcessHealth:
    """Check health of a training process.

    Monitors:
    - Log file staleness (no updates for > 5 min)
    - Progress rate (epochs per hour vs expected)
    - CPU time vs actual progress (stuck detection)
    """
    is_stale, log_age = check_log_staleness(log_file)

    # Parse log for progress
    epochs_completed = 0
    start_time = None

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Epoch [' in line:
                        # Parse epoch number from "Epoch [N/M]"
                        import re
                        match = re.search(r'Epoch \[(\d+)/\d+\]', line)
                        if match:
                            epochs_completed = int(match.group(1))
                    if start_time is None and 'Starting training' in line:
                        start_time = os.path.getmtime(log_file)
        except Exception:
            pass

    # Calculate progress rate
    if start_time:
        hours_elapsed = (time.time() - start_time) / 3600.0
        progress_rate = epochs_completed / hours_elapsed if hours_elapsed > 0 else 0
    else:
        progress_rate = 0

    is_slow = progress_rate < expected_epochs_per_hour * HEALTH_CHECK_PROGRESS_RATE_MIN

    # Determine overall status
    if is_stale and log_age > 600:  # No log update for 10+ min
        status = "stuck"
    elif is_stale:
        status = "stale"
    elif is_slow:
        status = "slow"
    else:
        status = "healthy"

    return ProcessHealth(
        process_name="training",
        pid=None,
        log_file=log_file,
        log_mtime=os.path.getmtime(log_file) if os.path.exists(log_file) else None,
        is_stale=is_stale,
        cpu_time_seconds=0,  # Would need psutil to get this
        progress_rate=progress_rate,
        expected_rate=expected_epochs_per_hour,
        is_slow=is_slow,
        status=status,
    )


def check_cmaes_process_health(
    log_file: str,
    expected_games_per_hour: float = 100.0,
) -> ProcessHealth:
    """Check health of a CMA-ES process.

    Monitors:
    - Log file staleness
    - Games per hour vs expected
    - Generation progress
    """
    is_stale, log_age = check_log_staleness(log_file)

    # Parse log for progress
    games_played = 0
    generations_completed = 0

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # Count generation completions
                import re
                gen_matches = re.findall(r'Generation (\d+)/\d+ complete', content)
                if gen_matches:
                    generations_completed = max(int(g) for g in gen_matches)

                # Count games from progress lines
                game_matches = re.findall(r'overall: (\d+)/\d+', content)
                if game_matches:
                    games_played = max(int(g) for g in game_matches)
        except Exception:
            pass

    # Calculate progress rate
    mtime = os.path.getmtime(log_file) if os.path.exists(log_file) else time.time()
    ctime = os.path.getctime(log_file) if os.path.exists(log_file) else time.time()
    hours_elapsed = (time.time() - ctime) / 3600.0
    progress_rate = games_played / hours_elapsed if hours_elapsed > 0 else 0

    is_slow = progress_rate < expected_games_per_hour * HEALTH_CHECK_PROGRESS_RATE_MIN

    # Determine status
    if is_stale and log_age > 600:
        status = "stuck"
    elif is_stale:
        status = "stale"
    elif is_slow:
        status = "slow"
    else:
        status = "healthy"

    return ProcessHealth(
        process_name="cmaes",
        pid=None,
        log_file=log_file,
        log_mtime=mtime,
        is_stale=is_stale,
        cpu_time_seconds=0,
        progress_rate=progress_rate,
        expected_rate=expected_games_per_hour,
        is_slow=is_slow,
        status=status,
    )


def cleanup_cmaes_processes(host: Optional[str] = None) -> int:
    """Kill all CMA-ES processes on a host.

    Args:
        host: SSH host to clean up, or None for local.

    Returns:
        Number of processes killed.
    """
    import subprocess

    if host:
        cmd = f"ssh {host} 'pkill -9 -f run_cmaes; sleep 1; pgrep -f run_cmaes | wc -l'"
    else:
        cmd = "pkill -9 -f run_cmaes; sleep 1; pgrep -f run_cmaes | wc -l"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        remaining = int(result.stdout.strip()) if result.stdout.strip() else 0
        return remaining
    except Exception as e:
        print(f"[HealthMonitor] Cleanup failed: {e}")
        return -1


def restart_process_if_stuck(
    process_name: str,
    health: ProcessHealth,
    restart_callback: Optional[callable] = None,
) -> bool:
    """Auto-restart a stuck process.

    Args:
        process_name: Name for logging
        health: Health check result
        restart_callback: Function to call to restart the process

    Returns:
        True if restart was triggered
    """
    if health.status == "stuck":
        print(f"[HealthMonitor] {process_name} appears stuck:")
        print(f"  - Log stale for {(time.time() - health.log_mtime):.0f}s" if health.log_mtime else "  - No log file")
        print(f"  - Progress rate: {health.progress_rate:.2f} vs expected {health.expected_rate:.2f}")

        if restart_callback:
            print(f"[HealthMonitor] Triggering restart for {process_name}...")
            cleanup_cmaes_processes()  # Clean up first
            restart_callback()
            return True
        else:
            print(f"[HealthMonitor] No restart callback provided for {process_name}")

    return False


# GPU vs CPU expected rates
GPU_CMAES_GAMES_PER_HOUR = 4000.0  # ~1.1 games/sec * 3600
CPU_CMAES_GAMES_PER_HOUR = 15.0    # ~0.004 games/sec * 3600


def run_health_check(
    training_log: Optional[str] = None,
    cmaes_log: Optional[str] = None,
    use_gpu_rates: bool = True,
    host: Optional[str] = None,
) -> Dict[str, ProcessHealth]:
    """Run health checks on all monitored processes.

    Args:
        training_log: Path to training log file
        cmaes_log: Path to CMA-ES log file
        use_gpu_rates: Use GPU-expected rates (much higher) for CMA-ES
        host: SSH host for remote processes

    Returns:
        Dict of process name to health status
    """
    results = {}

    if training_log:
        results["training"] = check_training_process_health(training_log)

    if cmaes_log:
        expected_rate = GPU_CMAES_GAMES_PER_HOUR if use_gpu_rates else CPU_CMAES_GAMES_PER_HOUR
        results["cmaes"] = check_cmaes_process_health(cmaes_log, expected_rate)

    return results


def format_health_report(health_results: Dict[str, ProcessHealth]) -> str:
    """Format health check results as a human-readable report."""
    lines = ["=" * 60, "HEALTH CHECK REPORT", f"Time: {datetime.now().isoformat()}", "=" * 60, ""]

    for name, health in health_results.items():
        status_emoji = {
            "healthy": "✓",
            "stale": "⚠",
            "slow": "⚠",
            "stuck": "✗"
        }.get(health.status, "?")

        lines.append(f"{status_emoji} {name.upper()}: {health.status}")
        lines.append(f"   Log file: {health.log_file or 'N/A'}")
        if health.log_mtime:
            age_sec = time.time() - health.log_mtime
            lines.append(f"   Log age: {age_sec:.0f}s ({age_sec/60:.1f} min)")
        lines.append(f"   Progress rate: {health.progress_rate:.1f}/hr (expected: {health.expected_rate:.1f}/hr)")
        lines.append(f"   Rate ratio: {health.progress_rate / health.expected_rate * 100:.1f}%")
        lines.append("")

    return "\n".join(lines)


def run_health_monitor_loop(
    training_log: Optional[str] = None,
    cmaes_log: Optional[str] = None,
    check_interval: int = 120,
    use_gpu_rates: bool = True,
    host: Optional[str] = None,
    auto_restart: bool = False,
) -> None:
    """Run continuous health monitoring loop.

    Args:
        training_log: Path to training log file
        cmaes_log: Path to CMA-ES log file
        check_interval: Seconds between health checks
        use_gpu_rates: Expect GPU-level performance for CMA-ES
        host: SSH host for remote processes
        auto_restart: Whether to automatically restart stuck processes
    """
    print(f"\n{'#' * 60}")
    print("HEALTH MONITOR STARTED")
    print(f"{'#' * 60}")
    print(f"Check interval: {check_interval}s")
    print(f"Training log: {training_log or 'Not monitored'}")
    print(f"CMA-ES log: {cmaes_log or 'Not monitored'}")
    print(f"GPU rates: {use_gpu_rates}")
    print(f"Auto-restart: {auto_restart}")
    print(f"Host: {host or 'local'}")
    print()

    consecutive_failures = {"training": 0, "cmaes": 0}

    while True:
        try:
            results = run_health_check(
                training_log=training_log,
                cmaes_log=cmaes_log,
                use_gpu_rates=use_gpu_rates,
                host=host,
            )

            report = format_health_report(results)
            print(report)

            # Track consecutive failures and auto-restart
            for name, health in results.items():
                if health.status in ("stuck", "stale"):
                    consecutive_failures[name] += 1
                    print(f"[HealthMonitor] {name} unhealthy ({consecutive_failures[name]} consecutive)")

                    if auto_restart and consecutive_failures[name] >= 3:
                        print(f"[HealthMonitor] Auto-restart triggered for {name}")
                        if name == "cmaes":
                            killed = cleanup_cmaes_processes(host)
                            print(f"[HealthMonitor] Killed {killed} CMA-ES processes")
                        consecutive_failures[name] = 0
                else:
                    consecutive_failures[name] = 0

            print(f"[HealthMonitor] Next check in {check_interval}s...\n")
            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n[HealthMonitor] Interrupted, stopping...")
            break
        except Exception as e:
            print(f"[HealthMonitor] Error: {e}")
            time.sleep(check_interval)


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

    # Health monitoring arguments
    parser.add_argument("--monitor", action="store_true",
                        help="Run health monitoring loop")
    parser.add_argument("--health-check", action="store_true",
                        help="Run single health check")
    parser.add_argument("--training-log", type=str, default=None,
                        help="Path to training log file for health monitoring")
    parser.add_argument("--cmaes-log", type=str, default=None,
                        help="Path to CMA-ES log file for health monitoring")
    parser.add_argument("--check-interval", type=int, default=120,
                        help="Seconds between health checks (default: 120)")
    parser.add_argument("--cpu-mode", action="store_true",
                        help="Use CPU-expected rates for CMA-ES (slower)")
    parser.add_argument("--host", type=str, default=None,
                        help="SSH host for remote health checks")
    parser.add_argument("--auto-restart", action="store_true",
                        help="Automatically restart stuck processes")

    args = parser.parse_args()

    ai_service_path = Path(__file__).resolve().parents[1]
    ringrift_path = ai_service_path.parent

    # Health monitoring commands (don't need manager)
    if args.monitor:
        run_health_monitor_loop(
            training_log=args.training_log,
            cmaes_log=args.cmaes_log,
            check_interval=args.check_interval,
            use_gpu_rates=not args.cpu_mode,
            host=args.host,
            auto_restart=args.auto_restart,
        )
        return

    if args.health_check:
        results = run_health_check(
            training_log=args.training_log,
            cmaes_log=args.cmaes_log,
            use_gpu_rates=not args.cpu_mode,
            host=args.host,
        )
        print(format_health_report(results))
        return

    # Manager-based commands
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
