"""CLI Utilities for RingRift Scripts.

This package provides standardized patterns for command-line scripts:
- Argument parsing helpers
- Script runner with common setup
- Output formatting
- Progress display

Usage:
    from app.cli import (
        ScriptRunner,
        add_common_args,
        setup_script,
        print_status,
        print_table,
    )

    # Quick script setup
    def main():
        runner = ScriptRunner("my_script")
        runner.add_argument("--config", required=True)

        args = runner.parse_args()
        with runner.run_context():
            do_work(args)

    # Or use setup_script for simple cases
    args, logger = setup_script("my_script", description="Does something")
"""

from app.cli.output import (
    ProgressBar,
    print_error,
    print_progress,
    print_status,
    print_success,
    print_table,
)
from app.cli.runner import (
    ScriptRunner,
    add_common_args,
    setup_script,
)

__all__ = [
    "ProgressBar",
    # Runner
    "ScriptRunner",
    "add_common_args",
    "print_error",
    "print_progress",
    # Output
    "print_status",
    "print_success",
    "print_table",
    "setup_script",
]
