"""Script Runner - Common patterns for CLI scripts.

Provides a standardized way to set up and run CLI scripts with:
- Argument parsing with common options
- Logging setup
- Signal handling
- Context management for cleanup
"""

from __future__ import annotations

import argparse
import logging
import signal
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ScriptRunner",
    "add_common_args",
    "setup_script",
]


@dataclass
class ScriptConfig:
    """Configuration for a script run."""
    name: str
    verbose: bool = False
    dry_run: bool = False
    config_path: Path | None = None
    log_level: str = "INFO"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to an argument parser.

    Adds:
    - --verbose/-v: Enable verbose output
    - --quiet/-q: Suppress non-error output
    - --dry-run: Preview without making changes
    - --config/-c: Path to config file
    - --log-level: Set logging level

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing them",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )


class ScriptRunner:
    """Runner for CLI scripts with common patterns.

    Provides:
    - Argument parsing with common options
    - Logging setup
    - Signal handling for graceful shutdown
    - Context management for cleanup

    Usage:
        runner = ScriptRunner("my_script", description="Does something useful")
        runner.add_argument("--input", required=True)
        runner.add_argument("--output", default="output.txt")

        args = runner.parse_args()
        with runner.run_context():
            do_work(args.input, args.output)
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        add_common: bool = True,
    ):
        """Initialize the script runner.

        Args:
            name: Script name (used for logging)
            description: Script description for help text
            add_common: Whether to add common arguments
        """
        self.name = name
        self.parser = argparse.ArgumentParser(
            prog=name,
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.logger: logging.Logger | None = None
        self._shutdown_requested = False
        self._cleanup_handlers: list[Callable[[], None]] = []

        if add_common:
            add_common_args(self.parser)

    def add_argument(self, *args: Any, **kwargs: Any) -> None:
        """Add an argument to the parser.

        Wrapper around ArgumentParser.add_argument().
        """
        self.parser.add_argument(*args, **kwargs)

    def add_subparsers(self, **kwargs: Any) -> argparse._SubParsersAction:
        """Add subparsers for subcommands.

        Returns:
            SubParsersAction for adding subcommands
        """
        return self.parser.add_subparsers(**kwargs)

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """Parse command line arguments and set up logging.

        Args:
            args: Arguments to parse (defaults to sys.argv)

        Returns:
            Parsed arguments namespace
        """
        parsed = self.parser.parse_args(args)

        # Set up logging
        self._setup_logging(parsed)

        return parsed

    def _setup_logging(self, args: argparse.Namespace) -> None:
        """Set up logging based on parsed arguments."""
        # Determine log level
        if hasattr(args, "verbose") and args.verbose:
            level = logging.DEBUG
        elif hasattr(args, "quiet") and args.quiet:
            level = logging.ERROR
        elif hasattr(args, "log_level"):
            level = getattr(logging, args.log_level)
        else:
            level = logging.INFO

        # Use centralized logging if available
        try:
            from app.core.logging_config import setup_logging
            self.logger = setup_logging(self.name, level=level)
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            self.logger = logging.getLogger(self.name)

    @contextmanager
    def run_context(self):
        """Context manager for script execution.

        Sets up signal handlers and runs cleanup on exit.

        Usage:
            with runner.run_context():
                do_work()
        """
        # Register signal handlers
        original_handlers = self._setup_signal_handlers()

        try:
            yield
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("Interrupted by user")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Script failed: {e}")
            raise
        finally:
            # Run cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Cleanup handler failed: {e}")

            # Restore original signal handlers
            self._restore_signal_handlers(original_handlers)

    def on_cleanup(self, handler: Callable[[], None]) -> None:
        """Register a cleanup handler to run on exit.

        Args:
            handler: Function to call during cleanup
        """
        self._cleanup_handlers.append(handler)

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_requested = True

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    def _setup_signal_handlers(self) -> dict[int, Any]:
        """Set up signal handlers for graceful shutdown."""
        original = {}

        def handler(signum: int, frame: Any) -> None:
            if self.logger:
                self.logger.info(f"Received signal {signum}, shutting down")
            self._shutdown_requested = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            with suppress(ValueError, OSError):
                original[sig] = signal.signal(sig, handler)

        return original

    def _restore_signal_handlers(self, original: dict[int, Any]) -> None:
        """Restore original signal handlers."""
        for sig, handler in original.items():
            with suppress(ValueError, OSError):
                signal.signal(sig, handler)


def setup_script(
    name: str,
    description: str | None = None,
    **extra_args: dict[str, Any],
) -> tuple[argparse.Namespace, logging.Logger]:
    """Quick setup for simple scripts.

    Creates a runner, adds any extra arguments, parses args, and returns
    the parsed args and logger.

    Args:
        name: Script name
        description: Script description
        **extra_args: Additional arguments to add (name: kwargs dict)

    Returns:
        Tuple of (parsed_args, logger)

    Example:
        args, logger = setup_script(
            "my_script",
            description="Does something",
            input={"required": True, "help": "Input file"},
            output={"default": "out.txt"},
        )
    """
    runner = ScriptRunner(name, description)

    for arg_name, arg_kwargs in extra_args.items():
        # Convert underscore to dash for CLI
        cli_name = f"--{arg_name.replace('_', '-')}"
        runner.add_argument(cli_name, **arg_kwargs)

    args = runner.parse_args()
    return args, runner.logger or logging.getLogger(name)
