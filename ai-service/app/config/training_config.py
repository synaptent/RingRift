"""Training configuration dataclasses for CMA-ES and Neural Network training.

This module provides typed configuration for all training-related scripts,
enabling:
- Reproducibility through serializable configs
- Environment variable overrides
- Config diffing for experiment comparison

Usage:
    from app.config.training_config import CMAESConfig, NeuralNetConfig

    # Create config from defaults
    config = CMAESConfig()

    # Create from environment variables
    config = CMAESConfig.from_env()

    # Create from CLI args
    config = CMAESConfig.from_args(args)

    # Serialize for reproducibility
    config.to_json("config.json")
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES heuristic weight optimization.

    All parameters can be overridden via environment variables prefixed
    with RINGRIFT_CMAES_. For example:
        RINGRIFT_CMAES_GENERATIONS=30
        RINGRIFT_CMAES_POPULATION_SIZE=20
    """

    # Core CMA-ES parameters
    generations: int = 20
    population_size: int = 16
    sigma: float = 0.5

    # Evaluation settings
    games_per_eval: int = 24
    max_moves: int = 200
    eval_randomness: float = 0.02

    # Board and player configuration
    board_type: str = "square8"
    eval_boards: List[str] = field(default_factory=lambda: ["square8"])
    num_players: int = 2

    # State pool for mid-game evaluation
    state_pool_id: str = "v1"
    eval_mode: str = "multi-start"  # multi-start, from-start, random

    # Opponent settings
    opponent_mode: str = "baseline-only"  # baseline-only, self-play, mixed

    # Distributed training
    distributed: bool = False
    workers: List[str] = field(default_factory=list)
    eval_timeout: float = 300.0

    # Output and reproducibility
    output_dir: str = "logs/cmaes"
    seed: int = 42
    progress_interval_sec: int = 30

    # Run metadata
    run_id: Optional[str] = None

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_CMAES_") -> CMAESConfig:
        """Create config from environment variables."""
        kwargs = {}

        env_mapping = {
            "GENERATIONS": ("generations", int),
            "POPULATION_SIZE": ("population_size", int),
            "SIGMA": ("sigma", float),
            "GAMES_PER_EVAL": ("games_per_eval", int),
            "MAX_MOVES": ("max_moves", int),
            "EVAL_RANDOMNESS": ("eval_randomness", float),
            "BOARD_TYPE": ("board_type", str),
            "EVAL_BOARDS": ("eval_boards", lambda x: x.split(",")),
            "NUM_PLAYERS": ("num_players", int),
            "STATE_POOL_ID": ("state_pool_id", str),
            "EVAL_MODE": ("eval_mode", str),
            "OPPONENT_MODE": ("opponent_mode", str),
            "DISTRIBUTED": ("distributed", lambda x: x.lower() == "true"),
            "WORKERS": ("workers", lambda x: x.split(",")),
            "EVAL_TIMEOUT": ("eval_timeout", float),
            "OUTPUT_DIR": ("output_dir", str),
            "SEED": ("seed", int),
            "PROGRESS_INTERVAL_SEC": ("progress_interval_sec", int),
        }

        for env_suffix, (field_name, converter) in env_mapping.items():
            env_var = prefix + env_suffix
            if env_var in os.environ:
                value = os.environ[env_var]
                kwargs[field_name] = converter(value)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str | Path] = None) -> str:
        """Serialize to JSON string, optionally saving to file."""
        json_str = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path:
            Path(path).write_text(json_str)
        return json_str

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> CMAESConfig:
        """Load config from JSON file or string."""
        path = Path(path_or_str)
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = json.loads(path_or_str)
        return cls(**data)

    def config_hash(self) -> str:
        """Generate a hash of the config for cache invalidation."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class NeuralNetConfig:
    """Configuration for neural network training.

    Environment variable prefix: RINGRIFT_NN_
    """

    # Model architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    input_channels: int = 17
    policy_head_channels: int = 32
    value_head_channels: int = 32

    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5

    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 1.0

    # Data configuration
    board_type: str = "square8"
    history_length: int = 3
    augment_rotations: bool = True
    augment_reflections: bool = True

    # Device and performance
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    model_id: str = "ringrift_v1"
    checkpoint_dir: str = "models"
    save_every_n_epochs: int = 10

    # Logging
    log_dir: str = "logs/tensorboard"
    log_every_n_steps: int = 100

    # Data paths
    data_dir: str = "data/training"

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_NN_") -> NeuralNetConfig:
        """Create config from environment variables."""
        kwargs = {}

        env_mapping = {
            "HIDDEN_LAYERS": ("hidden_layers", lambda x: [int(n) for n in x.split(",")]),
            "INPUT_CHANNELS": ("input_channels", int),
            "POLICY_HEAD_CHANNELS": ("policy_head_channels", int),
            "VALUE_HEAD_CHANNELS": ("value_head_channels", int),
            "BATCH_SIZE": ("batch_size", int),
            "LEARNING_RATE": ("learning_rate", float),
            "WEIGHT_DECAY": ("weight_decay", float),
            "EPOCHS": ("epochs", int),
            "WARMUP_EPOCHS": ("warmup_epochs", int),
            "POLICY_WEIGHT": ("policy_weight", float),
            "VALUE_WEIGHT": ("value_weight", float),
            "BOARD_TYPE": ("board_type", str),
            "HISTORY_LENGTH": ("history_length", int),
            "AUGMENT_ROTATIONS": ("augment_rotations", lambda x: x.lower() == "true"),
            "AUGMENT_REFLECTIONS": ("augment_reflections", lambda x: x.lower() == "true"),
            "DEVICE": ("device", str),
            "NUM_WORKERS": ("num_workers", int),
            "PIN_MEMORY": ("pin_memory", lambda x: x.lower() == "true"),
            "MODEL_ID": ("model_id", str),
            "CHECKPOINT_DIR": ("checkpoint_dir", str),
            "SAVE_EVERY_N_EPOCHS": ("save_every_n_epochs", int),
            "LOG_DIR": ("log_dir", str),
            "LOG_EVERY_N_STEPS": ("log_every_n_steps", int),
            "DATA_DIR": ("data_dir", str),
            "SEED": ("seed", int),
        }

        for env_suffix, (field_name, converter) in env_mapping.items():
            env_var = prefix + env_suffix
            if env_var in os.environ:
                value = os.environ[env_var]
                kwargs[field_name] = converter(value)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str | Path] = None) -> str:
        """Serialize to JSON string, optionally saving to file."""
        json_str = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path:
            Path(path).write_text(json_str)
        return json_str

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> NeuralNetConfig:
        """Load config from JSON file or string."""
        path = Path(path_or_str)
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = json.loads(path_or_str)
        return cls(**data)

    def get_device(self) -> str:
        """Get the actual device to use for training."""
        import torch

        if self.device != "auto":
            return self.device

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation."""

    # Generation settings
    num_games: int = 1000
    max_moves_per_game: int = 200
    parallel_games: int = 4

    # AI settings
    ai_type: str = "heuristic"  # heuristic, mcts, neural
    temperature: float = 1.0
    temperature_drop_move: int = 30
    exploration_fraction: float = 0.25

    # Board configuration
    board_type: str = "square8"
    num_players: int = 2

    # Output
    output_uri: str = "data/selfplay"
    compress: bool = True
    batch_size: int = 1000  # Samples to buffer before writing

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_SELFPLAY_") -> SelfPlayConfig:
        """Create config from environment variables."""
        kwargs = {}

        env_mapping = {
            "NUM_GAMES": ("num_games", int),
            "MAX_MOVES_PER_GAME": ("max_moves_per_game", int),
            "PARALLEL_GAMES": ("parallel_games", int),
            "AI_TYPE": ("ai_type", str),
            "TEMPERATURE": ("temperature", float),
            "TEMPERATURE_DROP_MOVE": ("temperature_drop_move", int),
            "EXPLORATION_FRACTION": ("exploration_fraction", float),
            "BOARD_TYPE": ("board_type", str),
            "NUM_PLAYERS": ("num_players", int),
            "OUTPUT_URI": ("output_uri", str),
            "COMPRESS": ("compress", lambda x: x.lower() == "true"),
            "BATCH_SIZE": ("batch_size", int),
            "SEED": ("seed", int),
        }

        for env_suffix, (field_name, converter) in env_mapping.items():
            env_var = prefix + env_suffix
            if env_var in os.environ:
                value = os.environ[env_var]
                kwargs[field_name] = converter(value)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str | Path] = None) -> str:
        """Serialize to JSON string, optionally saving to file."""
        json_str = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path:
            Path(path).write_text(json_str)
        return json_str
