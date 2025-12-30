"""CLI argument utilities for scripts.

Provides common argument patterns and helpers for command-line scripts.
Consolidates 500+ lines of duplicated argument parsing across 30+ scripts.

Usage:
    from scripts.lib.cli import (
        # Common args
        add_common_args,
        add_verbose_arg,
        add_dry_run_arg,
        add_config_arg,
        add_output_arg,
        add_limit_arg,
        add_parallel_arg,
        add_timeout_arg,
        # Board configuration
        add_board_args,
        parse_board_type,
        get_config_key,
        parse_config_key,
        # Database args (Dec 2025)
        add_db_args,
        add_elo_db_arg,
        add_game_db_arg,
        # Model args (Dec 2025)
        add_model_args,
        add_model_version_arg,
        # Training args (Dec 2025)
        add_training_args,
        add_selfplay_args,
        # Utilities
        add_node_arg,
        setup_cli_logging,
        validate_path_arg,
        confirm_action,
    )

    parser = argparse.ArgumentParser()
    add_common_args(parser)       # Adds --verbose and --dry-run
    add_board_args(parser)        # Adds --board and --num-players
    add_db_args(parser, use_discovery=True)  # Adds --db and --use-discovery
    add_model_args(parser)        # Adds --model
    args = parser.parse_args()

    # Setup logging based on --verbose flag
    logger = setup_cli_logging("my_script", args)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from scripts.lib.logging_config import setup_script_logging

# Import BoardType for type hints and mapping
try:
    from app.models import BoardType
    _HAS_BOARD_TYPE = True
except ImportError:
    _HAS_BOARD_TYPE = False
    BoardType = None  # type: ignore

logger = logging.getLogger(__name__)


# Standard board configurations
BOARD_TYPES = ["square8", "square19", "hexagonal", "hex8"]
VALID_PLAYER_COUNTS = [2, 3, 4]

# Canonical mapping from string names to BoardType enum
# Includes all common aliases used across scripts
if _HAS_BOARD_TYPE:
    BOARD_TYPE_MAP: dict[str, BoardType] = {
        # Standard names
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex8": BoardType.HEX8,
        # Aliases
        "8": BoardType.SQUARE8,
        "19": BoardType.SQUARE19,
        "hex": BoardType.HEXAGONAL,
        "s8": BoardType.SQUARE8,
        "s19": BoardType.SQUARE19,
    }
else:
    BOARD_TYPE_MAP = {}


def parse_board_type(value: str) -> BoardType:
    """Parse a board type string to BoardType enum.

    Handles common aliases and is case-insensitive.

    Args:
        value: Board type string (e.g., "square8", "hex8", "19")

    Returns:
        BoardType enum value

    Raises:
        ValueError: If board type string is not recognized

    Example:
        board_type = parse_board_type("square8")
        board_type = parse_board_type("19")  # Returns BoardType.SQUARE19
    """
    if not _HAS_BOARD_TYPE:
        raise ImportError("BoardType not available - app.models not importable")

    normalized = value.lower().strip()
    if normalized in BOARD_TYPE_MAP:
        return BOARD_TYPE_MAP[normalized]

    # Try to match by enum name directly
    try:
        return BoardType(normalized)
    except ValueError:
        pass

    # Try uppercase enum name
    try:
        return BoardType[normalized.upper()]
    except KeyError:
        pass

    valid = ", ".join(sorted(BOARD_TYPE_MAP.keys()))
    raise ValueError(f"Unknown board type: {value}. Valid options: {valid}")


def add_verbose_arg(
    parser: argparse.ArgumentParser,
    short: str = "-v",
    long: str = "--verbose",
    help_text: str = "Enable verbose output",
) -> None:
    """Add --verbose argument to parser.

    Args:
        parser: ArgumentParser to add argument to
        short: Short flag (default: -v)
        long: Long flag (default: --verbose)
        help_text: Help text for the argument
    """
    parser.add_argument(
        short,
        long,
        action="store_true",
        help=help_text,
    )


def add_dry_run_arg(
    parser: argparse.ArgumentParser,
    long: str = "--dry-run",
    help_text: str = "Show what would be done without executing",
) -> None:
    """Add --dry-run argument to parser.

    Args:
        parser: ArgumentParser to add argument to
        long: Long flag (default: --dry-run)
        help_text: Help text for the argument
    """
    parser.add_argument(
        long,
        action="store_true",
        help=help_text,
    )


def add_config_arg(
    parser: argparse.ArgumentParser,
    short: str = "-c",
    long: str = "--config",
    default: str | None = None,
    required: bool = False,
    help_text: str = "Configuration file path",
) -> None:
    """Add --config argument to parser.

    Args:
        parser: ArgumentParser to add argument to
        short: Short flag (default: -c)
        long: Long flag (default: --config)
        default: Default config path
        required: Whether argument is required
        help_text: Help text for the argument
    """
    parser.add_argument(
        short,
        long,
        type=str,
        default=default,
        required=required,
        help=help_text,
    )


def add_node_arg(
    parser: argparse.ArgumentParser,
    long: str = "--node-id",
    required: bool = True,
    default: str | None = None,
    help_text: str = "Node identifier",
) -> None:
    """Add --node-id argument to parser.

    If not required and no default, will try to use hostname.

    Args:
        parser: ArgumentParser to add argument to
        long: Long flag (default: --node-id)
        required: Whether argument is required
        default: Default node ID
        help_text: Help text for the argument
    """
    if default is None and not required:
        import socket
        default = socket.gethostname()

    parser.add_argument(
        long,
        type=str,
        required=required,
        default=default,
        help=help_text,
    )


def add_board_args(
    parser: argparse.ArgumentParser,
    board_arg: str = "--board",
    players_arg: str = "--num-players",
    default_board: str = "square8",
    default_players: int = 2,
    board_choices: list[str] | None = None,
    player_choices: list[int] | None = None,
) -> None:
    """Add board and num-players arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
        board_arg: Board argument name (default: --board)
        players_arg: Players argument name (default: --num-players)
        default_board: Default board type
        default_players: Default number of players
        board_choices: Valid board type choices
        player_choices: Valid player count choices
    """
    parser.add_argument(
        board_arg,
        type=str,
        default=default_board,
        choices=board_choices or BOARD_TYPES,
        help=f"Board type (default: {default_board})",
    )
    parser.add_argument(
        players_arg,
        type=int,
        default=default_players,
        choices=player_choices or VALID_PLAYER_COUNTS,
        help=f"Number of players (default: {default_players})",
    )


def add_output_arg(
    parser: argparse.ArgumentParser,
    short: str = "-o",
    long: str = "--output",
    default: str | None = None,
    required: bool = False,
    help_text: str = "Output file or directory path",
) -> None:
    """Add --output argument to parser.

    Args:
        parser: ArgumentParser to add argument to
        short: Short flag (default: -o)
        long: Long flag (default: --output)
        default: Default output path
        required: Whether argument is required
        help_text: Help text for the argument
    """
    parser.add_argument(
        short,
        long,
        type=str,
        default=default,
        required=required,
        help=help_text,
    )


def add_limit_arg(
    parser: argparse.ArgumentParser,
    long: str = "--limit",
    default: int | None = None,
    help_text: str = "Maximum number of items to process",
) -> None:
    """Add --limit argument to parser.

    Args:
        parser: ArgumentParser to add argument to
        long: Long flag (default: --limit)
        default: Default limit (None = no limit)
        help_text: Help text for the argument
    """
    parser.add_argument(
        long,
        type=int,
        default=default,
        help=help_text,
    )


def add_parallel_arg(
    parser: argparse.ArgumentParser,
    short: str = "-j",
    long: str = "--jobs",
    default: int | None = None,
    help_text: str = "Number of parallel jobs (default: CPU count)",
) -> None:
    """Add --jobs argument for parallelism.

    Args:
        parser: ArgumentParser to add argument to
        short: Short flag (default: -j)
        long: Long flag (default: --jobs)
        default: Default job count (None = auto-detect)
        help_text: Help text for the argument
    """
    default = default or os.cpu_count()
    parser.add_argument(
        short,
        long,
        type=int,
        default=default,
        help=help_text,
    )


def add_timeout_arg(
    parser: argparse.ArgumentParser,
    long: str = "--timeout",
    default: int = 300,
    help_text: str = "Timeout in seconds",
) -> None:
    """Add --timeout argument to parser.

    Args:
        parser: ArgumentParser to add argument to
        long: Long flag (default: --timeout)
        default: Default timeout in seconds
        help_text: Help text for the argument
    """
    parser.add_argument(
        long,
        type=int,
        default=default,
        help=help_text,
    )


def add_common_args(
    parser: argparse.ArgumentParser,
    verbose: bool = True,
    dry_run: bool = True,
    config: bool = False,
    config_default: str | None = None,
) -> None:
    """Add common arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
        verbose: Add --verbose flag
        dry_run: Add --dry-run flag
        config: Add --config argument
        config_default: Default config path if config=True
    """
    if verbose:
        add_verbose_arg(parser)
    if dry_run:
        add_dry_run_arg(parser)
    if config:
        add_config_arg(parser, default=config_default)


def setup_cli_logging(
    name: str,
    args: argparse.Namespace,
    verbose_attr: str = "verbose",
    log_dir: str = "logs",
) -> logging.Logger:
    """Setup logging based on CLI arguments.

    Args:
        name: Logger name
        args: Parsed arguments
        verbose_attr: Attribute name for verbose flag
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    verbose = getattr(args, verbose_attr, False)
    level = "DEBUG" if verbose else "INFO"

    return setup_script_logging(
        name,
        log_dir=log_dir,
        level=level,
    )


def get_config_key(board: str, num_players: int) -> str:
    """Get standard config key from board and player count.

    Args:
        board: Board type (square8, hexagonal, etc.)
        num_players: Number of players

    Returns:
        Config key like "square8_2p" or "hexagonal_3p"
    """
    return f"{board}_{num_players}p"


def parse_config_key(config_key: str) -> tuple[str, int]:
    """Parse config key into board and player count.

    Note: Delegates to canonical app.coordination.config_key module.
    Raises ValueError for CLI compatibility.

    Args:
        config_key: Config key like "square8_2p"

    Returns:
        Tuple of (board_type, num_players)

    Raises:
        ValueError: If config key format is invalid
    """
    from app.coordination.config_key import ConfigKey

    result = ConfigKey.parse(config_key)
    if result is None:
        raise ValueError(f"Invalid config key format: {config_key}")
    return result.to_tuple()


def validate_path_arg(
    path: str | Path,
    must_exist: bool = True,
    is_file: bool = False,
    is_dir: bool = False,
    create_dir: bool = False,
) -> Path:
    """Validate a path argument.

    Args:
        path: Path to validate
        must_exist: Require path to exist
        is_file: Require path to be a file
        is_dir: Require path to be a directory
        create_dir: Create directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    path = Path(path)

    if create_dir and is_dir:
        path.mkdir(parents=True, exist_ok=True)
        return path

    if must_exist and not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")

    if is_file and path.exists() and not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {path}")

    if is_dir and path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path is not a directory: {path}")

    return path


def confirm_action(
    message: str,
    default: bool = False,
    skip_confirm: bool = False,
) -> bool:
    """Ask user for confirmation.

    Args:
        message: Confirmation message
        default: Default response if user just hits enter
        skip_confirm: Skip confirmation and return True

    Returns:
        True if user confirms, False otherwise
    """
    if skip_confirm:
        return True

    suffix = " [Y/n] " if default else " [y/N] "
    try:
        response = input(message + suffix).strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def print_dry_run_notice() -> None:
    """Print a notice that this is a dry run."""
    print("\n" + "=" * 60)
    print("DRY RUN MODE - No changes will be made")
    print("=" * 60 + "\n")


def print_summary(
    title: str,
    items: dict[str, Any],
    indent: int = 2,
) -> None:
    """Print a formatted summary.

    Args:
        title: Summary title
        items: Dictionary of items to display
        indent: Indentation level
    """
    print(f"\n{title}")
    print("-" * len(title))
    prefix = " " * indent
    for key, value in items.items():
        print(f"{prefix}{key}: {value}")
    print()


def create_subparser_with_common_args(
    subparsers: Any,
    name: str,
    help_text: str,
    verbose: bool = True,
    dry_run: bool = False,
) -> argparse.ArgumentParser:
    """Create a subparser with common arguments.

    Args:
        subparsers: Subparser group from add_subparsers()
        name: Subcommand name
        help_text: Help text for subcommand
        verbose: Add --verbose flag
        dry_run: Add --dry-run flag

    Returns:
        Created subparser
    """
    parser = subparsers.add_parser(name, help=help_text)
    add_common_args(parser, verbose=verbose, dry_run=dry_run)
    return parser


# =============================================================================
# Database Arguments (December 2025 - Consolidation)
# =============================================================================


def add_db_args(
    parser: argparse.ArgumentParser,
    db_arg: str = "--db",
    db_dir_arg: str | None = None,
    db_pattern_arg: str | None = None,
    default_db: str | None = None,
    required: bool = False,
    use_discovery: bool = False,
    help_text: str = "Database path",
) -> None:
    """Add database-related arguments to parser.

    Supports single DB, directory-based discovery, and glob patterns.
    This consolidates the 25+ different database argument patterns across scripts.

    Args:
        parser: ArgumentParser to add arguments to
        db_arg: Database path argument name (default: --db)
        db_dir_arg: Database directory argument name (optional, e.g., --db-dir)
        db_pattern_arg: Glob pattern argument name (optional, e.g., --db-pattern)
        default_db: Default database path
        required: Whether --db is required (ignored if use_discovery=True)
        use_discovery: Add --use-discovery flag for GameDiscovery
        help_text: Help text for --db argument

    Example:
        add_db_args(parser, use_discovery=True)
        # Adds: --db, --use-discovery

        add_db_args(parser, db_dir_arg="--db-dir", db_pattern_arg="--db-pattern")
        # Adds: --db, --db-dir, --db-pattern
    """
    # Main database path argument
    parser.add_argument(
        db_arg,
        type=Path,
        default=Path(default_db) if default_db else None,
        required=required and not use_discovery,
        help=help_text,
    )

    # Optional directory argument
    if db_dir_arg:
        parser.add_argument(
            db_dir_arg,
            type=Path,
            default=None,
            help="Directory to search for databases",
        )

    # Optional glob pattern argument
    if db_pattern_arg:
        parser.add_argument(
            db_pattern_arg,
            type=str,
            default=None,
            help="Glob pattern for database files (e.g., 'data/games/*.db')",
        )

    # GameDiscovery integration
    if use_discovery:
        parser.add_argument(
            "--use-discovery",
            action="store_true",
            help="Use GameDiscovery to find all matching databases automatically",
        )


def add_elo_db_arg(
    parser: argparse.ArgumentParser,
    default: str = "data/unified_elo.db",
    help_text: str = "Path to ELO database",
) -> None:
    """Add ELO database argument with standard default.

    Args:
        parser: ArgumentParser to add argument to
        default: Default ELO database path
        help_text: Help text for the argument
    """
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(default),
        help=help_text,
    )


def add_game_db_arg(
    parser: argparse.ArgumentParser,
    required: bool = True,
    default: str | None = None,
    help_text: str = "Path to game database (.db file)",
) -> None:
    """Add game database argument for selfplay/replay databases.

    Args:
        parser: ArgumentParser to add argument to
        required: Whether argument is required
        default: Default path
        help_text: Help text for the argument
    """
    parser.add_argument(
        "--db",
        type=Path,
        required=required,
        default=Path(default) if default else None,
        help=help_text,
    )


# =============================================================================
# Model Arguments (December 2025 - Consolidation)
# =============================================================================


def add_model_args(
    parser: argparse.ArgumentParser,
    model_arg: str = "--model",
    models_dir_arg: str | None = None,
    weights_arg: str | None = None,
    default_model: str | None = None,
    default_models_dir: str = "models",
    required: bool = False,
    help_text: str = "Path to model weights (.pth file)",
) -> None:
    """Add model-related arguments to parser.

    Consolidates model path, models directory, and weights arguments.

    Args:
        parser: ArgumentParser to add arguments to
        model_arg: Model path argument name (default: --model)
        models_dir_arg: Models directory argument name (optional, e.g., --models-dir)
        weights_arg: Initial weights argument name (optional, e.g., --init-weights)
        default_model: Default model path
        default_models_dir: Default models directory
        required: Whether --model is required
        help_text: Help text for --model argument

    Example:
        add_model_args(parser, models_dir_arg="--models-dir", weights_arg="--init-weights")
        # Adds: --model, --models-dir, --init-weights
    """
    parser.add_argument(
        model_arg,
        type=Path,
        default=Path(default_model) if default_model else None,
        required=required,
        help=help_text,
    )

    if models_dir_arg:
        parser.add_argument(
            models_dir_arg,
            type=Path,
            default=Path(default_models_dir),
            help=f"Directory containing model files (default: {default_models_dir})",
        )

    if weights_arg:
        parser.add_argument(
            weights_arg,
            type=Path,
            default=None,
            help="Initial weights to load for transfer learning",
        )


def add_model_version_arg(
    parser: argparse.ArgumentParser,
    default: str = "v2",
    choices: list[str] | None = None,
) -> None:
    """Add model version argument.

    Args:
        parser: ArgumentParser to add argument to
        default: Default model version
        choices: Valid model version choices
    """
    parser.add_argument(
        "--model-version",
        type=str,
        default=default,
        choices=choices or ["v1", "v2", "v3", "v4", "nnue"],
        help=f"Model architecture version (default: {default})",
    )


# =============================================================================
# Training Arguments (December 2025 - Consolidation)
# =============================================================================


def add_training_args(
    parser: argparse.ArgumentParser,
    include_batch_size: bool = True,
    include_epochs: bool = True,
    include_lr: bool = True,
    include_device: bool = True,
    default_batch_size: int = 512,
    default_epochs: int = 20,
    default_lr: float = 0.001,
) -> None:
    """Add common training arguments.

    Args:
        parser: ArgumentParser to add arguments to
        include_batch_size: Add --batch-size argument
        include_epochs: Add --epochs argument
        include_lr: Add --learning-rate argument
        include_device: Add --device argument
        default_batch_size: Default batch size
        default_epochs: Default number of epochs
        default_lr: Default learning rate
    """
    if include_batch_size:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=default_batch_size,
            help=f"Training batch size (default: {default_batch_size})",
        )

    if include_epochs:
        parser.add_argument(
            "--epochs",
            type=int,
            default=default_epochs,
            help=f"Number of training epochs (default: {default_epochs})",
        )

    if include_lr:
        parser.add_argument(
            "--learning-rate", "--lr",
            type=float,
            default=default_lr,
            help=f"Learning rate (default: {default_lr})",
        )

    if include_device:
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            choices=["cpu", "cuda", "mps"],
            help="Device for training (default: auto-detect)",
        )


def add_selfplay_args(
    parser: argparse.ArgumentParser,
    include_num_games: bool = True,
    include_engine: bool = True,
    default_num_games: int = 100,
    default_engine: str = "heuristic",
) -> None:
    """Add common selfplay arguments.

    Args:
        parser: ArgumentParser to add arguments to
        include_num_games: Add --num-games argument
        include_engine: Add --engine argument
        default_num_games: Default number of games
        default_engine: Default engine mode
    """
    if include_num_games:
        parser.add_argument(
            "--num-games", "-n",
            type=int,
            default=default_num_games,
            help=f"Number of games to generate (default: {default_num_games})",
        )

    if include_engine:
        parser.add_argument(
            "--engine",
            type=str,
            default=default_engine,
            choices=[
                "heuristic", "gumbel", "mcts", "nnue-guided",
                "policy-only", "nn-descent", "mixed",
            ],
            help=f"Selfplay engine mode (default: {default_engine})",
        )
