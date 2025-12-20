"""CLI argument utilities for scripts.

Provides common argument patterns and helpers for command-line scripts.

Usage:
    from scripts.lib.cli import (
        add_common_args,
        add_verbose_arg,
        add_dry_run_arg,
        add_config_arg,
        add_board_args,
        add_node_arg,
        setup_cli_logging,
    )

    parser = argparse.ArgumentParser()
    add_common_args(parser)  # Adds --verbose and --dry-run
    add_board_args(parser)   # Adds --board and --num-players
    args = parser.parse_args()

    # Setup logging based on --verbose flag
    logger = setup_cli_logging("my_script", args)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    BOARD_TYPE_MAP: Dict[str, "BoardType"] = {
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


def parse_board_type(value: str) -> "BoardType":
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
    default: Optional[str] = None,
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
    default: Optional[str] = None,
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
    board_choices: Optional[List[str]] = None,
    player_choices: Optional[List[int]] = None,
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
    default: Optional[str] = None,
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
    default: Optional[int] = None,
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
    default: Optional[int] = None,
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
    config_default: Optional[str] = None,
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

    Args:
        config_key: Config key like "square8_2p"

    Returns:
        Tuple of (board_type, num_players)

    Raises:
        ValueError: If config key format is invalid
    """
    if "_" not in config_key or not config_key.endswith("p"):
        raise ValueError(f"Invalid config key format: {config_key}")

    parts = config_key.rsplit("_", 1)
    board = parts[0]
    players_str = parts[1].rstrip("p")

    try:
        num_players = int(players_str)
    except ValueError:
        raise ValueError(f"Invalid player count in config key: {config_key}")

    return board, num_players


def validate_path_arg(
    path: Union[str, Path],
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
    items: Dict[str, Any],
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
