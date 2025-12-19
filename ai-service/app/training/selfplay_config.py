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
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.models import BoardType


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
    engine_mode: EngineMode = EngineMode.NNUE_GUIDED
    search_depth: int = 3
    mcts_simulations: int = 800
    temperature: float = 1.0
    temperature_threshold: int = 30  # Move number after which to use greedy

    # Output settings
    output_format: OutputFormat = OutputFormat.DB
    output_dir: Optional[str] = None
    output_file: Optional[str] = None
    record_db: Optional[str] = None

    # Recording options
    store_history_entries: bool = True
    lean_db: bool = False
    snapshot_interval: int = 20
    cache_nnue_features: bool = True

    # Resource settings
    num_workers: int = 1
    batch_size: int = 256
    use_gpu: bool = True
    gpu_device: int = 0

    # Disk monitoring thresholds
    disk_warning_percent: int = 75
    disk_critical_percent: int = 85

    # Ramdrive settings
    use_ramdrive: bool = False
    ramdrive_path: Optional[str] = None
    sync_interval: int = 300  # seconds

    # Reproducibility
    seed: Optional[int] = None
    checkpoint_interval: int = 100

    # Metadata
    source: str = "selfplay"
    profile_id: Optional[str] = None

    # Heuristic weights (for heuristic-based engines)
    weights_file: Optional[str] = None
    weights_profile: Optional[str] = None

    # Worker coordination (for distributed selfplay)
    worker_id: Optional[str] = None

    # Telemetry settings
    telemetry_path: Optional[str] = None
    telemetry_interval: int = 50  # seconds

    # NN batching (for distributed/GPU selfplay)
    nn_batch_enabled: bool = False
    nn_batch_timeout_ms: int = 50
    nn_max_batch_size: int = 256

    # Shadow validation (for quality checking)
    shadow_validation: bool = False
    shadow_sample_rate: float = 0.05
    shadow_threshold: float = 0.001

    # Game rules customization
    lps_victory_rounds: int = 3
    min_game_length: int = 0
    random_opening_moves: int = 0

    # Distributed selfplay settings
    hosts: Optional[str] = None
    max_parallel_per_host: int = 2
    difficulty_band: str = "light"  # "light", "canonical", etc.

    # Additional engine-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Normalize board type
        self.board_type = self._normalize_board_type(self.board_type)

        # Convert string engine mode to enum
        if isinstance(self.engine_mode, str):
            engine_mode = self.engine_mode
            if engine_mode == "random-only":
                engine_mode = "random"
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
        """Normalize board type to canonical form."""
        # Map aliases to canonical names
        aliases = {
            "hex": "hexagonal",
            "hex8": "hex8",
            "square": "square8",
            "sq8": "square8",
            "sq19": "square19",
            "square19": "square19",
        }
        return aliases.get(board_type.lower(), board_type.lower())

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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "SelfplayConfig":
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
        choices=["square8", "square19", "hex8", "hex", "hexagonal"],
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
    if "random-only" not in engine_choices:
        engine_choices.append("random-only")

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

    # Resource settings
    resource_group = parser.add_argument_group("Resource Settings")
    resource_group.add_argument(
        "--num-workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    resource_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for GPU processing (default: 256)",
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

    # NN batching
    nn_group = parser.add_argument_group("NN Batching")
    nn_group.add_argument(
        "--nn-batch-enabled",
        action="store_true",
        help="Enable NN batching for distributed evaluation",
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

    return parser


def parse_selfplay_args(
    args: Optional[List[str]] = None,
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

    # Map parsed args to config
    config = SelfplayConfig(
        # Core game settings
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        # Engine settings
        engine_mode=parsed.engine_mode,
        search_depth=parsed.search_depth,
        mcts_simulations=parsed.mcts_simulations,
        temperature=parsed.temperature,
        # Output settings
        output_format=parsed.output_format,
        output_dir=parsed.output_dir,
        record_db=parsed.record_db,
        lean_db=getattr(parsed, "lean_db", False),
        store_history_entries=not getattr(parsed, "lean_db", False),
        cache_nnue_features=getattr(parsed, "cache_nnue_features", True),
        # Resource settings
        num_workers=parsed.num_workers,
        batch_size=parsed.batch_size,
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
        # NN batching
        nn_batch_enabled=getattr(parsed, "nn_batch_enabled", False),
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
