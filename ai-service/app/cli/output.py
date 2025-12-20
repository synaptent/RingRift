"""CLI Output Utilities - Formatted console output.

Provides utilities for formatted console output:
- Status messages with icons
- Tables
- Progress bars
- Colored output (when terminal supports it)
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = [
    "ProgressBar",
    "print_error",
    "print_progress",
    "print_status",
    "print_success",
    "print_table",
    "print_warning",
]

# ANSI color codes (used when terminal supports it)
_COLORS = {
    "reset": "\033[0m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}

# Check if we should use colors
_USE_COLORS = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _color(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if not _USE_COLORS:
        return text
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def print_status(message: str, status: str = "info") -> None:
    """Print a status message with icon.

    Args:
        message: Message to print
        status: Status type (info, success, warning, error)
    """
    icons = {
        "info": "i",
        "success": "+",
        "warning": "!",
        "error": "x",
    }
    colors = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
    }
    icon = icons.get(status, "i")
    color = colors.get(status, "reset")
    print(f"[{_color(icon, color)}] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print_status(message, "error")


def print_success(message: str) -> None:
    """Print a success message."""
    print_status(message, "success")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print_status(message, "warning")


def print_table(
    data: list[dict[str, Any]],
    columns: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Print data as a formatted table.

    Args:
        data: List of dicts to display
        columns: Column keys to display (defaults to all keys from first row)
        headers: Optional display names for columns

    Example:
        print_table([
            {"name": "Alice", "score": 100},
            {"name": "Bob", "score": 85},
        ])
    """
    if not data:
        print("(no data)")
        return

    # Determine columns
    if columns is None:
        columns = list(data[0].keys())

    # Get headers
    headers = headers or {}
    display_headers = [headers.get(col, col) for col in columns]

    # Calculate column widths
    widths = []
    for i, col in enumerate(columns):
        values = [str(row.get(col, "")) for row in data]
        max_val = max(len(v) for v in values) if values else 0
        widths.append(max(max_val, len(display_headers[i])))

    # Print header
    header_row = " | ".join(
        h.ljust(w) for h, w in zip(display_headers, widths, strict=False)
    )
    print(_color(header_row, "bold"))
    print("-" * len(header_row))

    # Print data rows
    for row in data:
        values = [str(row.get(col, "")).ljust(widths[i]) for i, col in enumerate(columns)]
        print(" | ".join(values))


def print_progress(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    width: int = 40,
) -> None:
    """Print a progress bar.

    Args:
        current: Current progress value
        total: Total value
        prefix: Text before the bar
        suffix: Text after the bar
        width: Bar width in characters
    """
    if total <= 0:
        return

    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    percent = current * 100 / total

    line = f"\r{prefix}[{bar}] {percent:5.1f}% {suffix}"
    print(line, end="", flush=True)

    if current >= total:
        print()  # New line when complete


class ProgressBar:
    """Context manager for progress display.

    Usage:
        with ProgressBar(total=100, prefix="Processing") as bar:
            for item in items:
                process(item)
                bar.update()

        # Or with explicit values
        bar = ProgressBar(total=100)
        bar.update(50)
        bar.finish()
    """

    def __init__(
        self,
        total: int,
        prefix: str = "",
        width: int = 40,
    ):
        """Initialize progress bar.

        Args:
            total: Total number of items
            prefix: Text before the bar
            width: Bar width in characters
        """
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0

    def update(self, amount: int = 1) -> None:
        """Update progress.

        Args:
            amount: Amount to increment (default 1)
        """
        self.current = min(self.current + amount, self.total)
        print_progress(self.current, self.total, self.prefix, width=self.width)

    def set(self, value: int) -> None:
        """Set progress to a specific value.

        Args:
            value: New progress value
        """
        self.current = min(value, self.total)
        print_progress(self.current, self.total, self.prefix, width=self.width)

    def finish(self) -> None:
        """Complete the progress bar."""
        self.set(self.total)

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, *args: Any) -> None:
        if self.current < self.total:
            print()  # Ensure we end on a new line
