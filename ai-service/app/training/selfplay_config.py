"""Unified configuration for all selfplay scripts.

This module provides a centralized configuration system for selfplay,
eliminating duplicated argument parsing across 34+ scripts.

Usage:
    from app.training.selfplay_config import SelfplayConfig, parse_selfplay_args

    # In scripts:
    config = parse_selfplay_args()
    # Or create directly:
    config = SelfplayConfig(board_type="hex", num_players=2, engine_mode="nnue-guided")
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.models import BoardType
from app.utils.canonical_naming import normalize_board_type as _canonical_normalize
from app.utils.parallel_defaults import get_default_workers, get_parallel_games_default


class EngineMode(str, Enum):
    """Available selfplay engine modes."""
    HEURISTIC = "heuristic-only"
    NNUE_GUIDED = "nnue-guided"
    POLICY_ONLY = "policy-only"
    NN_MINIMAX = "nn-minimax"
    NN_DESCENT = "nn-descent"
    GUMBEL_MCTS = "gumbel-mcts"
    MCTS = "mcts"
    MIXED = "mixed"
    DIVERSE = "diverse"
    RANDOM = "random"
    DESCENT_ONLY = "descent-only"
    MAXN = "maxn"
    BRS = "brs"
    # Experimental AI engine modes
    GMO = "gmo"  # Gradient Move Optimization (entropy-guided gradient ascent) [DEPRECATED]
    EBMO = "ebmo"  # Energy-Based Move Optimization [DEPRECATED]
    IG_GMO = "ig-gmo"  # Information-Gain GMO [DEPRECATED]
    CAGE = "cage"  # Constraint-Aware Graph Energy-based optimization [DEPRECATED]
    # GNN-based engine modes (replaces deprecated GMO/EBMO)
    GNN = "gnn"  # Pure GNN policy network with Gumbel sampling
    HYBRID = "hybrid"  # CNN-GNN hybrid with Gumbel MCTS


ENGINE_MODE_ALIASES: dict[str, str] = {
    # Common alias forms (underscores vs hyphens)
    "gumbel_mcts": EngineMode.GUMBEL_MCTS.value,
    "gumbel": EngineMode.GUMBEL_MCTS.value,
    "policy_only": EngineMode.POLICY_ONLY.value,
    "nnue_guided": EngineMode.NNUE_GUIDED.value,
    "nn_minimax": EngineMode.NN_MINIMAX.value,
    "nn_descent": EngineMode.NN_DESCENT.value,
    # Common shorthand
    "mcts-only": EngineMode.MCTS.value,
    "mcts_only": EngineMode.MCTS.value,
    "descent": EngineMode.DESCENT_ONLY.value,
    "heuristic": EngineMode.HEURISTIC.value,
    "heuristic_only": EngineMode.HEURISTIC.value,
    "random-only": EngineMode.RANDOM.value,
    # GNN aliases
    "gnn-policy": EngineMode.GNN.value,
    "gnn_policy": EngineMode.GNN.value,
    "hybrid-gnn": EngineMode.HYBRID.value,
    "hybrid_gnn": EngineMode.HYBRID.value,
    "cnn-gnn": EngineMode.HYBRID.value,
    "cnn_gnn": EngineMode.HYBRID.value,
}


# =============================================================================
# GPU Requirement Metadata (December 2025)
# =============================================================================
#
# Defines which engine modes require GPU (CUDA/MPS) vs. CPU-compatible modes.
# Used by job dispatch to match engine modes to appropriate nodes.
# =============================================================================

# GPU-required modes (require neural network inference)
# These MUST run on nodes with CUDA or MPS GPU available
GPU_REQUIRED_ENGINE_MODES: frozenset[EngineMode] = frozenset({
    EngineMode.GUMBEL_MCTS,  # Gumbel MCTS with neural network
    EngineMode.MCTS,  # Monte Carlo Tree Search with NN evaluation
    EngineMode.NNUE_GUIDED,  # NNUE-guided search
    EngineMode.POLICY_ONLY,  # Pure policy network
    EngineMode.NN_MINIMAX,  # Neural network minimax
    EngineMode.NN_DESCENT,  # Neural network descent
    EngineMode.GNN,  # Graph neural network
    EngineMode.HYBRID,  # CNN-GNN hybrid
    EngineMode.GMO,  # Gradient Move Optimization (deprecated, but still GPU)
    EngineMode.EBMO,  # Energy-Based Move Optimization (deprecated, but still GPU)
    EngineMode.IG_GMO,  # Information-Gain GMO (deprecated, but still GPU)
    EngineMode.CAGE,  # Constraint-Aware Graph Energy (deprecated, but still GPU)
})

# CPU-compatible modes (no neural network required)
# These can run on any node, including CPU-only nodes
CPU_COMPATIBLE_ENGINE_MODES: frozenset[EngineMode] = frozenset({
    EngineMode.HEURISTIC,  # Pure heuristic evaluation
    EngineMode.RANDOM,  # Random move selection
    EngineMode.DESCENT_ONLY,  # Gradient descent without NN
    EngineMode.MAXN,  # Max-N algorithm
    EngineMode.BRS,  # Best Reply Search
})

# Mixed modes - can use GPU if available, but fall back to CPU
# Note: MIXED and DIVERSE can work with CPU-only opponents
MIXED_ENGINE_MODES: frozenset[EngineMode] = frozenset({
    EngineMode.MIXED,  # Mixed opponent pool
    EngineMode.DIVERSE,  # Diverse engine mix
})


def engine_mode_requires_gpu(mode: EngineMode | str) -> bool:
    """Check if an engine mode requires GPU (CUDA or MPS).

    This is the authoritative function for determining if an engine mode
    needs GPU capability. Used by job dispatch to match modes to nodes.

    Args:
        mode: Engine mode as EngineMode enum or string

    Returns:
        True if the mode requires GPU, False if CPU-compatible

    Example:
        >>> engine_mode_requires_gpu(EngineMode.GUMBEL_MCTS)
        True
        >>> engine_mode_requires_gpu("heuristic-only")
        False
        >>> engine_mode_requires_gpu("mixed")
        False  # Mixed can fall back to CPU opponents
    """
    if isinstance(mode, str):
        # Normalize and convert to enum
        normalized = normalize_engine_mode(mode)
        try:
            mode = EngineMode(normalized)
        except ValueError:
            # Unknown mode, assume GPU required for safety
            return True

    return mode in GPU_REQUIRED_ENGINE_MODES


def engine_mode_is_cpu_compatible(mode: EngineMode | str) -> bool:
    """Check if an engine mode can run on CPU-only nodes.

    Inverse of engine_mode_requires_gpu, but also handles mixed modes.

    Args:
        mode: Engine mode as EngineMode enum or string

    Returns:
        True if the mode can run on CPU-only nodes
    """
    if isinstance(mode, str):
        normalized = normalize_engine_mode(mode)
        try:
            mode = EngineMode(normalized)
        except ValueError:
            return False

    return mode in CPU_COMPATIBLE_ENGINE_MODES or mode in MIXED_ENGINE_MODES


def normalize_engine_mode(raw_mode: str) -> str:
    """Normalize engine mode aliases to canonical EngineMode values."""
    normalized = raw_mode.strip().lower()
    return ENGINE_MODE_ALIASES.get(normalized, normalized)


class OutputFormat(str, Enum):
    """Output format for game data."""
    JSONL = "jsonl"
    DB = "db"
    NPZ = "npz"


@dataclass
class SelfplayConfig:
    """Unified configuration for selfplay generation.

    This dataclass consolidates all configuration options across selfplay scripts,
    providing a single source of truth for:
    - Board type and player count
    - Engine mode and search parameters
    - Output format and paths
    - Resource management (GPU, workers, disk)
    - Recording options
    """

    # Core game settings
    board_type: str = "square8"
    num_players: int = 2
    num_games: int = 1000

    # Engine settings
    # Jan 12, 2026: Changed default from NNUE_GUIDED to MIXED for harness diversity
    # MIXED mode rotates between random, heuristic, mcts, minimax, policy_only
    # This ensures diverse training data and proper Elo tracking per harness type
    engine_mode: EngineMode = EngineMode.MIXED
    search_depth: int = 3
    mcts_simulations: int = 800
    simulation_budget: int = 800  # Gumbel MCTS budget (800 = quality, 64 = throughput)
    difficulty: int = 8  # AI difficulty level (1-10), affects simulation budget
    temperature: float = 1.0
    temperature_threshold: int = 40  # Move number after which to use greedy (increased for more exploration)

    # Output settings
    output_format: OutputFormat = OutputFormat.DB
    output_dir: str | None = None
    output_file: str | None = None
    record_db: str | None = None

    # Recording options
    store_history_entries: bool = True
    lean_db: bool = False
    snapshot_interval: int = 20
    cache_nnue_features: bool = True

    # Heuristic pre-computation (Jan 2026 - for fast v5-heavy exports)
    # When True, computes and stores heuristic features during selfplay
    # This provides 10-20x speedup for exports with --full-heuristics
    # Default True: new selfplay games are automatically cache-ready
    compute_heuristics_on_write: bool = True
    full_heuristics: bool = True  # True = 49 features, False = 21 fast features

    # Resource settings (parallelism is the default)
    num_workers: int = field(default_factory=get_default_workers)
    parallel_games: int = field(default_factory=get_parallel_games_default)  # Jan 12, 2026: Parallel game simulation
    batch_size: int = 256  # Default 256, use get_optimal_batch_size() for auto-tuning
    use_gpu: bool = True
    gpu_device: int = 0
    device: str | None = None  # CUDA device string (e.g., "cuda:0")

    # Game limits
    max_moves: int = 1000  # Maximum moves per game before termination
    record_samples: bool = True  # Record training samples during selfplay

    # Disk monitoring thresholds
    disk_warning_percent: int = 75
    disk_critical_percent: int = 85

    # Ramdrive settings
    use_ramdrive: bool = False
    ramdrive_path: str | None = None
    sync_interval: int = 300  # seconds

    # Reproducibility
    seed: int | None = None
    checkpoint_interval: int = 100

    # Metadata
    source: str = "selfplay"
    profile_id: str | None = None

    # Heuristic weights (for heuristic-based engines)
    weights_file: str | None = None
    weights_profile: str | None = None

    # Worker coordination (for distributed selfplay)
    worker_id: str | None = None

    # Telemetry settings
    telemetry_path: str | None = None
    telemetry_interval: int = 50  # seconds

    # NN batching (for distributed/GPU selfplay)
    # Jan 12, 2026: Enabled by default for parallel selfplay efficiency
    nn_batch_enabled: bool = True
    nn_batch_timeout_ms: int = 50
    nn_max_batch_size: int = 256

    # Neural network usage
    use_neural_net: bool = False
    prefer_nnue: bool = True
    nn_model_id: str | None = None
    # Architecture version for model selection (v2, v4, v5, v5-heavy, etc.)
    # Jan 5, 2026: Added for architecture selection feedback loop
    model_version: str = "v5"

    # Shadow validation (for quality checking)
    shadow_validation: bool = False
    shadow_sample_rate: float = 0.05
    shadow_threshold: float = 0.001

    # Game rules customization
    lps_victory_rounds: int = 3
    min_game_length: int = 0
    random_opening_moves: int = 0

    # Distributed selfplay settings
    hosts: str | None = None
    max_parallel_per_host: int = 2
    difficulty_band: str = "light"  # "light", "canonical", etc.

    # Pipeline automation (2025-12)
    emit_pipeline_events: bool = False  # Emit SELFPLAY_COMPLETE for auto-trigger pipeline

    # PFSP opponent selection (December 2025)
    # Prioritizes opponents with ~50% win rate for maximum learning signal
    use_pfsp: bool = True  # Enabled by default

    # Note: simulation_budget defined above (default 800)
    # Note: difficulty defined above (default 8)

    # Elo-adaptive budget (December 2025)
    model_elo: float | None = None  # Model Elo for adaptive budget calculation
    training_epoch: int = 0  # Training epoch for progressive budget scaling

    # Mixed opponent diversity (December 2025)
    mixed_opponents: bool = False  # Enable mixed opponent training
    opponent_mix: dict[str, float] | None = None  # Custom opponent mix distribution

    # Per-player AI configuration (December 2025)
    # Enables heterogeneous opponents where each player uses a different AI type
    # Maps player number (1-indexed) to AI configuration
    # Example: {1: {"engine": "gumbel-mcts", "budget": 200}, 2: {"engine": "heuristic"}, ...}
    player_ai_configs: dict[int, dict[str, Any]] | None = None

    # Additional engine-specific options
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Normalize board type
        self.board_type = self._normalize_board_type(self.board_type)

        # Convert string engine mode to enum
        if isinstance(self.engine_mode, str):
            engine_mode = normalize_engine_mode(self.engine_mode)
            self.engine_mode = EngineMode(engine_mode)

        # Convert string output format to enum
        if isinstance(self.output_format, str):
            self.output_format = OutputFormat(self.output_format)

        # Set default output directory based on board type
        if self.output_dir is None:
            self.output_dir = f"data/selfplay/{self.config_key}"

        # Set default database path
        if self.record_db is None and self.output_format == OutputFormat.DB:
            self.record_db = f"data/games/{self.config_key}.db"

    @staticmethod
    def _normalize_board_type(board_type: str) -> str:
        """Normalize board type to canonical form.

        Delegates to canonical_naming.normalize_board_type for consistency.
        January 2026: Centralized to avoid 17+ duplicated alias mappings.
        """
        return _canonical_normalize(board_type)

    @property
    def config_key(self) -> str:
        """Generate canonical config key for this configuration.

        Format: {board_type}_{num_players}p
        Example: square8_2p, hexagonal_4p
        """
        return f"{self.board_type}_{self.num_players}p"

    @property
    def board_type_enum(self) -> BoardType:
        """Get BoardType enum for this configuration."""
        mapping = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hexagonal": BoardType.HEXAGONAL,
        }
        return mapping.get(self.board_type, BoardType.SQUARE8)

    def get_effective_budget(self) -> int:
        """Get the effective simulation budget for MCTS search.

        Returns the simulation_budget field (default 800).
        Note: simulation_budget was changed to default 800 in Jan 2026 fix.
        Previously it could be None and would fallback through Elo/difficulty.

        Returns:
            Simulation budget for MCTS search (default: 800)
        """
        return self.simulation_budget

    def get_player_ai_config(self, player: int) -> dict[str, Any]:
        """Get the AI configuration for a specific player.

        For cross-AI games with heterogeneous opponents, each player can have
        a different AI configuration (engine type, budget, difficulty, etc.).

        Args:
            player: Player number (1-indexed)

        Returns:
            AI configuration dict with at least "engine" key.
            Falls back to the global engine_mode if no per-player config is set.

        Example:
            >>> config = SelfplayConfig(
            ...     engine_mode="gumbel-mcts",
            ...     player_ai_configs={
            ...         1: {"engine": "gumbel-mcts", "budget": 200},
            ...         2: {"engine": "heuristic", "difficulty": 5},
            ...     }
            ... )
            >>> config.get_player_ai_config(1)
            {"engine": "gumbel-mcts", "budget": 200}
            >>> config.get_player_ai_config(2)
            {"engine": "heuristic", "difficulty": 5}
        """
        # If per-player configs are set, use them
        if self.player_ai_configs and player in self.player_ai_configs:
            return self.player_ai_configs[player]

        # Fall back to global engine mode
        return {
            "engine": self.engine_mode.value if hasattr(self.engine_mode, 'value') else str(self.engine_mode),
            "difficulty": self.difficulty,
            "budget": self.get_effective_budget(),
        }

    def is_heterogeneous_game(self) -> bool:
        """Check if this config uses different AI types for different players.

        Returns:
            True if player_ai_configs is set with different engines for different players.
        """
        if not self.player_ai_configs:
            return False

        engines = set()
        for config in self.player_ai_configs.values():
            engines.add(config.get("engine", "unknown"))

        return len(engines) > 1

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "board_type": self.board_type,
            "num_players": self.num_players,
            "num_games": self.num_games,
            "engine_mode": self.engine_mode.value,
            "search_depth": self.search_depth,
            "mcts_simulations": self.mcts_simulations,
            "temperature": self.temperature,
            "output_format": self.output_format.value,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "config_key": self.config_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelfplayConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k != "config_key"})


def create_argument_parser(
    description: str = "Run selfplay game generation",
    include_ramdrive: bool = True,
    include_gpu: bool = True,
) -> argparse.ArgumentParser:
    """Create a standardized argument parser for selfplay scripts.

    This centralizes all argument parsing logic that was previously
    duplicated across 34+ scripts.
    """
    parser = argparse.ArgumentParser(description=description)

    # Core game settings
    game_group = parser.add_argument_group("Game Settings")
    game_group.add_argument(
        "--board", "--board-type",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hex", "hexagonal", "full_hex", "hex24"],
        help="Board type (default: square8)",
    )
    game_group.add_argument(
        "--num-players", "-p",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)",
    )
    game_group.add_argument(
        "--num-games", "-n",
        type=int,
        default=1000,
        help="Number of games to generate (default: 1000)",
    )

    # Engine settings
    engine_group = parser.add_argument_group("Engine Settings")
    engine_choices = [e.value for e in EngineMode]
    engine_choices.extend(ENGINE_MODE_ALIASES.keys())
    engine_choices = sorted(set(engine_choices))

    engine_group.add_argument(
        "--engine-mode", "-e",
        type=str,
        default="nnue-guided",
        choices=engine_choices,
        help="AI engine mode (default: nnue-guided)",
    )
    engine_group.add_argument(
        "--search-depth",
        type=int,
        default=3,
        help="Search depth for minimax engines (default: 3)",
    )
    engine_group.add_argument(
        "--mcts-simulations",
        type=int,
        default=800,
        help="MCTS simulations per move (default: 800)",
    )
    engine_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for move selection (default: 1.0)",
    )
    engine_group.add_argument(
        "--simulation-budget",
        type=int,
        default=None,
        help="Gumbel MCTS simulation budget (default: auto based on difficulty)",
    )
    engine_group.add_argument(
        "--difficulty",
        type=int,
        default=None,
        help="Difficulty level 1-10 for auto budget selection (default: 8)",
    )
    engine_group.add_argument(
        "--model-elo",
        type=float,
        default=None,
        help="Model Elo for Elo-adaptive budget (December 2025). "
             "If set, overrides --difficulty for budget calculation.",
    )
    engine_group.add_argument(
        "--training-epoch",
        type=int,
        default=0,
        help="Training epoch for progressive budget scaling (default: 0). "
             "Used with --model-elo for Elo-adaptive budget.",
    )

    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for game data",
    )
    output_group.add_argument(
        "--output-format",
        type=str,
        default="db",
        choices=["db", "jsonl", "npz"],
        help="Output format (default: db)",
    )
    output_group.add_argument(
        "--record-db",
        type=str,
        default=None,
        help="Path to game recording database",
    )
    output_group.add_argument(
        "--lean-db",
        action="store_true",
        help="Use lean recording (no history entries)",
    )
    output_group.add_argument(
        "--cache-nnue-features",
        action="store_true",
        default=True,
        help="Cache NNUE features after recording",
    )

    # Resource settings (parallelism is the default - Jan 12, 2026)
    resource_group = parser.add_argument_group("Resource Settings")
    resource_group.add_argument(
        "--num-workers", "-w",
        type=int,
        default=None,  # None = use auto-scaling default (cpu_count - 1)
        help="Number of parallel workers (default: auto-scaled based on CPU count)",
    )
    resource_group.add_argument(
        "--parallel-games",
        type=int,
        default=None,  # None = use auto-scaling default (16 for 8+ cores)
        help="Number of parallel games in selfplay (default: 16 for 8+ cores, 8 otherwise)",
    )
    resource_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for GPU processing (default: auto-calculated based on board/GPU)",
    )
    resource_group.add_argument(
        "--auto-batch-size",
        action="store_true",
        default=True,
        help="Auto-calculate optimal batch size based on board type and GPU memory (default: True)",
    )
    resource_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N games (default: 100)",
    )

    if include_gpu:
        gpu_group = parser.add_argument_group("GPU Settings")
        gpu_group.add_argument(
            "--no-gpu",
            action="store_true",
            help="Disable GPU acceleration",
        )
        gpu_group.add_argument(
            "--gpu-device",
            type=int,
            default=0,
            help="GPU device index (default: 0)",
        )

    if include_ramdrive:
        ramdrive_group = parser.add_argument_group("Ramdrive Settings")
        ramdrive_group.add_argument(
            "--use-ramdrive",
            action="store_true",
            help="Use ramdrive for fast I/O",
        )
        ramdrive_group.add_argument(
            "--ramdrive-path",
            type=str,
            default="/dev/shm/ringrift",
            help="Path to ramdrive mount",
        )
        ramdrive_group.add_argument(
            "--sync-interval",
            type=int,
            default=300,
            help="Sync interval in seconds (default: 300)",
        )

    # Heuristic weights
    weights_group = parser.add_argument_group("Heuristic Weights")
    weights_group.add_argument(
        "--weights-file",
        type=str,
        default=None,
        help="Path to heuristic weights file (JSON)",
    )
    weights_group.add_argument(
        "--weights-profile",
        type=str,
        default=None,
        help="Profile name within weights file",
    )

    # Worker/Telemetry settings
    worker_group = parser.add_argument_group("Worker Settings")
    worker_group.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID for distributed coordination",
    )
    worker_group.add_argument(
        "--telemetry-path",
        type=str,
        default=None,
        help="Path for telemetry output",
    )
    worker_group.add_argument(
        "--telemetry-interval",
        type=int,
        default=50,
        help="Telemetry reporting interval in seconds (default: 50)",
    )

    # NN batching (enabled by default since Jan 12, 2026)
    nn_group = parser.add_argument_group("NN Batching")
    nn_group.add_argument(
        "--no-nn-batch",
        action="store_true",
        dest="no_nn_batch",
        help="Disable NN batching (enabled by default for efficiency)",
    )
    nn_group.add_argument(
        "--nn-batch-enabled",
        action="store_true",
        dest="nn_batch_enabled_legacy",
        help="[DEPRECATED] NN batching is now enabled by default. Use --no-nn-batch to disable.",
    )
    nn_group.add_argument(
        "--nn-batch-timeout-ms",
        type=int,
        default=50,
        help="NN batch timeout in milliseconds (default: 50)",
    )
    nn_group.add_argument(
        "--nn-max-batch-size",
        type=int,
        default=256,
        help="Maximum NN batch size (default: 256)",
    )

    # Shadow validation
    validation_group = parser.add_argument_group("Validation")
    validation_group.add_argument(
        "--shadow-validation",
        action="store_true",
        help="Enable shadow validation for quality checking",
    )
    validation_group.add_argument(
        "--shadow-sample-rate",
        type=float,
        default=0.05,
        help="Shadow validation sample rate (default: 0.05)",
    )

    # Game rules customization
    rules_group = parser.add_argument_group("Game Rules")
    rules_group.add_argument(
        "--lps-victory-rounds",
        type=int,
        default=3,
        help="LPS victory rounds required (default: 3)",
    )
    rules_group.add_argument(
        "--min-game-length",
        type=int,
        default=0,
        help="Minimum game length to record (default: 0)",
    )
    rules_group.add_argument(
        "--random-opening-moves",
        type=int,
        default=0,
        help="Random opening moves for diversity (default: 0)",
    )

    # Distributed settings
    distributed_group = parser.add_argument_group("Distributed")
    distributed_group.add_argument(
        "--hosts",
        type=str,
        default=None,
        help="Comma-separated list of hosts for distributed selfplay",
    )
    distributed_group.add_argument(
        "--max-parallel-per-host",
        type=int,
        default=2,
        help="Max parallel jobs per host (default: 2)",
    )
    distributed_group.add_argument(
        "--difficulty-band",
        type=str,
        default="light",
        choices=["light", "canonical", "full"],
        help="Difficulty band for game generation (default: light)",
    )

    # Reproducibility
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    misc_group.add_argument(
        "--profile-id",
        type=str,
        default=None,
        help="AI profile ID for tracking",
    )
    misc_group.add_argument(
        "--source",
        type=str,
        default="selfplay",
        help="Source identifier for metadata",
    )

    # Pipeline automation (2025-12)
    pipeline_group = parser.add_argument_group("Pipeline Automation")
    pipeline_group.add_argument(
        "--emit-pipeline-events",
        action="store_true",
        help="Emit pipeline events (SELFPLAY_COMPLETE) for auto-trigger pipeline. "
             "Use with train_cli.py --enable-pipeline-auto-trigger for full automation.",
    )

    # PFSP opponent selection (December 2025)
    pfsp_group = parser.add_argument_group("PFSP Opponent Selection")
    pfsp_group.add_argument(
        "--disable-pfsp",
        action="store_true",
        help="Disable PFSP (Prioritized Fictitious Self-Play) opponent selection. "
             "PFSP is enabled by default and prioritizes opponents with ~50%% win rate "
             "for maximum learning signal. Not recommended to disable.",
    )

    # Mixed opponent diversity (December 2025)
    mixed_group = parser.add_argument_group("Mixed Opponent Training")
    mixed_group.add_argument(
        "--mixed-opponents",
        action="store_true",
        help="Enable mixed opponent training with random/heuristic/MCTS mix. "
             "Default mix: 30%% random, 40%% heuristic, 30%% MCTS. "
             "Automatically sets --engine-mode to 'mixed'.",
    )
    mixed_group.add_argument(
        "--opponent-mix",
        type=str,
        default=None,
        help="Custom opponent mix in format 'random:0.3,heuristic:0.4,mcts:0.3'. "
             "Only used with --mixed-opponents.",
    )

    return parser


def _get_batch_size(parsed, board_type: str) -> int:
    """Get batch size - use explicit value or auto-calculate.

    Dec 2025: Added dynamic batch sizing based on board complexity and GPU memory.

    Args:
        parsed: Parsed argparse namespace
        board_type: Board type string (e.g., "hex8", "square19")

    Returns:
        Batch size to use for selfplay
    """
    # If explicit batch_size provided, use it
    if parsed.batch_size is not None:
        return parsed.batch_size

    # Auto-calculate based on board type and GPU
    try:
        from app.ai.gpu_parallel_games import get_optimal_batch_size
        num_players = getattr(parsed, "num_players", 2)
        batch = get_optimal_batch_size(board_type=board_type, num_players=num_players)
        return batch
    except ImportError:
        # Fallback if import fails
        return 256


def parse_selfplay_args(
    args: list[str] | None = None,
    description: str = "Run selfplay game generation",
) -> SelfplayConfig:
    """Parse command-line arguments and return SelfplayConfig.

    This is the main entry point for scripts to get configuration.

    Args:
        args: Command-line arguments (default: sys.argv)
        description: Parser description

    Returns:
        Configured SelfplayConfig instance
    """
    parser = create_argument_parser(description)
    parsed = parser.parse_args(args)

    # Extract board_type before config construction (needed for _get_batch_size)
    board_type = parsed.board

    # Map parsed args to config
    config = SelfplayConfig(
        # Core game settings
        board_type=board_type,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        # Engine settings
        engine_mode=parsed.engine_mode,
        search_depth=parsed.search_depth,
        mcts_simulations=parsed.mcts_simulations,
        temperature=parsed.temperature,
        simulation_budget=getattr(parsed, "simulation_budget", None),
        difficulty=getattr(parsed, "difficulty", None),
        model_elo=getattr(parsed, "model_elo", None),
        training_epoch=getattr(parsed, "training_epoch", 0),
        # Output settings
        output_format=parsed.output_format,
        output_dir=parsed.output_dir,
        record_db=parsed.record_db,
        lean_db=getattr(parsed, "lean_db", False),
        store_history_entries=not getattr(parsed, "lean_db", False),
        cache_nnue_features=getattr(parsed, "cache_nnue_features", True),
        # Resource settings (Jan 12, 2026: use dataclass defaults if not specified)
        num_workers=parsed.num_workers if parsed.num_workers is not None else get_default_workers(),
        parallel_games=getattr(parsed, "parallel_games", None) or get_parallel_games_default(),
        batch_size=_get_batch_size(parsed, board_type),
        checkpoint_interval=parsed.checkpoint_interval,
        use_gpu=not getattr(parsed, "no_gpu", False),
        gpu_device=getattr(parsed, "gpu_device", 0),
        # Ramdrive settings
        use_ramdrive=getattr(parsed, "use_ramdrive", False),
        ramdrive_path=getattr(parsed, "ramdrive_path", None),
        sync_interval=getattr(parsed, "sync_interval", 300),
        # Reproducibility/metadata
        seed=parsed.seed,
        source=parsed.source,
        profile_id=getattr(parsed, "profile_id", None),
        # Heuristic weights
        weights_file=getattr(parsed, "weights_file", None),
        weights_profile=getattr(parsed, "weights_profile", None),
        # Worker coordination
        worker_id=getattr(parsed, "worker_id", None),
        # Telemetry settings
        telemetry_path=getattr(parsed, "telemetry_path", None),
        telemetry_interval=getattr(parsed, "telemetry_interval", 50),
        # NN batching (Jan 12, 2026: enabled by default, use --no-nn-batch to disable)
        nn_batch_enabled=not getattr(parsed, "no_nn_batch", False),
        nn_batch_timeout_ms=getattr(parsed, "nn_batch_timeout_ms", 50),
        nn_max_batch_size=getattr(parsed, "nn_max_batch_size", 256),
        # Shadow validation
        shadow_validation=getattr(parsed, "shadow_validation", False),
        shadow_sample_rate=getattr(parsed, "shadow_sample_rate", 0.05),
        # Game rules customization
        lps_victory_rounds=getattr(parsed, "lps_victory_rounds", 3),
        min_game_length=getattr(parsed, "min_game_length", 0),
        random_opening_moves=getattr(parsed, "random_opening_moves", 0),
        # Distributed settings
        hosts=getattr(parsed, "hosts", None),
        max_parallel_per_host=getattr(parsed, "max_parallel_per_host", 2),
        difficulty_band=getattr(parsed, "difficulty_band", "light"),
        # Pipeline automation
        emit_pipeline_events=getattr(parsed, "emit_pipeline_events", False),
        # PFSP opponent selection (enabled by default, use --disable-pfsp to turn off)
        use_pfsp=not getattr(parsed, "disable_pfsp", False),
    )

    # Handle --mixed-opponents flag
    if getattr(parsed, "mixed_opponents", False):
        # Override engine mode to MIXED
        config.engine_mode = EngineMode.MIXED
        config.mixed_opponents = True

        # Parse custom opponent mix if provided
        opponent_mix_str = getattr(parsed, "opponent_mix", None)
        if opponent_mix_str:
            try:
                # Parse format: "random:0.3,heuristic:0.4,mcts:0.3"
                mix_dict = {}
                for pair in opponent_mix_str.split(","):
                    key, value = pair.split(":")
                    mix_dict[key.strip()] = float(value.strip())
                config.opponent_mix = mix_dict
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Failed to parse --opponent-mix '{opponent_mix_str}': {e}. "
                    "Using default mix."
                )

    return config


# Convenience functions for common configurations
def get_default_config(board_type: str = "square8", num_players: int = 2) -> SelfplayConfig:
    """Get a default configuration for quick testing."""
    return SelfplayConfig(board_type=board_type, num_players=num_players)


def get_production_config(board_type: str, num_players: int) -> SelfplayConfig:
    """Get a production-ready configuration."""
    return SelfplayConfig(
        board_type=board_type,
        num_players=num_players,
        num_games=100000,
        engine_mode=EngineMode.DIVERSE,
        search_depth=4,
        mcts_simulations=1600,
        store_history_entries=True,
        cache_nnue_features=True,
        checkpoint_interval=500,
    )


# =============================================================================
# Curriculum Config Templates
# =============================================================================
#
# Templates for generating diverse training data across the strength spectrum.
# Use these to create curriculum learning pipelines where:
# - Early training uses high-temperature exploration data
# - Mid training uses moderate-strength games
# - Late training uses strong-play trajectories
#
# Reference: NN_STRENGTHENING_PLAN.md section A) Data quality and selfplay
# =============================================================================


@dataclass
class CurriculumStage:
    """A single stage in a curriculum learning schedule."""

    name: str
    engine_mode: EngineMode
    temperature: float
    mcts_simulations: int
    search_depth: int
    games_per_config: int
    random_opening_moves: int = 0
    description: str = ""


# Pre-defined curriculum stages from weak exploration to strong play
CURRICULUM_STAGES: dict[str, CurriculumStage] = {
    # Stage 1: High-temperature exploration (early training)
    "explore_random": CurriculumStage(
        name="explore_random",
        engine_mode=EngineMode.RANDOM,
        temperature=2.0,
        mcts_simulations=0,
        search_depth=0,
        games_per_config=100,
        random_opening_moves=4,
        description="Pure random for game diversity and opening exploration",
    ),
    "explore_weak": CurriculumStage(
        name="explore_weak",
        engine_mode=EngineMode.HEURISTIC,
        temperature=1.5,
        mcts_simulations=0,
        search_depth=1,
        games_per_config=200,
        random_opening_moves=2,
        description="Weak heuristic with exploration noise",
    ),
    # Stage 2: Moderate strength (mid training)
    "moderate_mcts": CurriculumStage(
        name="moderate_mcts",
        engine_mode=EngineMode.MCTS,
        temperature=1.0,
        mcts_simulations=400,
        search_depth=2,
        games_per_config=500,
        description="Moderate MCTS for tactical learning",
    ),
    "moderate_nnue": CurriculumStage(
        name="moderate_nnue",
        engine_mode=EngineMode.NNUE_GUIDED,
        temperature=1.0,
        mcts_simulations=400,
        search_depth=3,
        games_per_config=500,
        description="NNUE-guided play for position evaluation learning",
    ),
    # Stage 3: Strong play (late training)
    "strong_gumbel": CurriculumStage(
        name="strong_gumbel",
        engine_mode=EngineMode.GUMBEL_MCTS,
        temperature=0.5,
        mcts_simulations=800,
        search_depth=4,
        games_per_config=800,
        description="Gumbel MCTS for strong policy targets",
    ),
    "strong_full": CurriculumStage(
        name="strong_full",
        engine_mode=EngineMode.GUMBEL_MCTS,
        temperature=0.3,
        mcts_simulations=1600,
        search_depth=4,
        games_per_config=1000,
        description="Full strength for championship-quality trajectories",
    ),
    # Experimental modes (use with caution)
    "experimental_gmo": CurriculumStage(
        name="experimental_gmo",
        engine_mode=EngineMode.GMO,
        temperature=0.8,
        mcts_simulations=200,
        search_depth=3,
        games_per_config=200,
        description="Gradient Move Optimization (experimental)",
    ),
    # Mixed opponent diversity training (December 2025)
    "robust_diverse": CurriculumStage(
        name="robust_diverse",
        engine_mode=EngineMode.MIXED,
        temperature=1.0,
        mcts_simulations=400,
        search_depth=2,
        games_per_config=600,
        random_opening_moves=1,
        description="Mixed opponent training (30% random, 40% heuristic, 30% MCTS) for robust play",
    ),
}


def get_curriculum_config(
    stage: str | CurriculumStage,
    board_type: str = "square8",
    num_players: int = 2,
) -> SelfplayConfig:
    """Get a SelfplayConfig for a specific curriculum stage.

    Args:
        stage: Curriculum stage name or CurriculumStage object
        board_type: Board type (square8, square19, hexagonal)
        num_players: Number of players (2, 3, 4)

    Returns:
        Configured SelfplayConfig for the curriculum stage
    """
    if isinstance(stage, str):
        if stage not in CURRICULUM_STAGES:
            raise ValueError(
                f"Unknown curriculum stage: {stage}. "
                f"Available: {list(CURRICULUM_STAGES.keys())}"
            )
        stage = CURRICULUM_STAGES[stage]

    return SelfplayConfig(
        board_type=board_type,
        num_players=num_players,
        num_games=stage.games_per_config,
        engine_mode=stage.engine_mode,
        temperature=stage.temperature,
        mcts_simulations=stage.mcts_simulations,
        search_depth=stage.search_depth,
        random_opening_moves=stage.random_opening_moves,
        store_history_entries=True,
        cache_nnue_features=True,
        source=f"curriculum_{stage.name}",
    )


def get_full_curriculum(
    board_type: str = "square8",
    num_players: int = 2,
    stages: list[str] | None = None,
) -> list[SelfplayConfig]:
    """Get a complete curriculum of SelfplayConfigs for training.

    Args:
        board_type: Board type
        num_players: Number of players
        stages: Optional list of stage names (default: all stages in order)

    Returns:
        List of SelfplayConfigs progressing from exploration to strong play
    """
    if stages is None:
        # Default progression order
        stages = [
            "explore_random",
            "explore_weak",
            "moderate_mcts",
            "moderate_nnue",
            "strong_gumbel",
            "strong_full",
        ]

    return [
        get_curriculum_config(stage, board_type, num_players)
        for stage in stages
    ]


def get_all_configs_curriculum(
    stages: list[str] | None = None,
) -> list[SelfplayConfig]:
    """Get curriculum configs for all 12 board/player combinations.

    This produces a comprehensive training curriculum covering:
    - 3 board types: square8, square19, hexagonal
    - 4 player counts: 2, 3, 4 players

    Args:
        stages: Optional list of stage names to include

    Returns:
        List of SelfplayConfigs for all combinations and stages
    """
    configs = []
    board_types = ["square8", "square19", "hexagonal"]
    player_counts = [2, 3, 4]

    for board_type in board_types:
        for num_players in player_counts:
            configs.extend(get_full_curriculum(board_type, num_players, stages))

    return configs


def list_curriculum_stages() -> dict[str, str]:
    """Get a mapping of stage names to descriptions."""
    return {name: stage.description for name, stage in CURRICULUM_STAGES.items()}
