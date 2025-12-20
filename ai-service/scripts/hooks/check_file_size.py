#!/usr/bin/env python3
"""Pre-commit hook to warn about large Python files.

Encourages modular design by warning when files exceed line thresholds.
Files over 1000 lines should be considered for refactoring.
Files over 2000 lines are strongly discouraged.

Usage:
    python scripts/hooks/check_file_size.py [files...]
"""

import sys
from pathlib import Path

# Line count thresholds
WARN_THRESHOLD = 1000  # Warn at this many lines
ERROR_THRESHOLD = 3000  # Error at this many lines (prevent new monoliths)

# Known large files that are grandfathered in
# These should be refactored eventually but don't block commits
GRANDFATHERED_FILES = {
    "app/ai/neural_net.py",  # 6962 lines - needs refactoring to package
    "app/game_engine.py",  # 4437 lines - needs modularization
    "app/training/train.py",  # 4198 lines - needs splitting
    "app/ai/mcts_ai.py",  # Large MCTS implementation
    "app/ai/gpu_parallel_games.py",  # GPU game runner
    "app/coordination/__init__.py",  # Re-exports, needs cleanup
}


def count_lines(filepath: Path) -> int:
    """Count non-empty, non-comment lines in a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return len(lines)
    except Exception:
        return 0


def main() -> int:
    """Check file sizes and warn/error as appropriate."""
    if len(sys.argv) < 2:
        return 0

    exit_code = 0
    warnings = []
    errors = []

    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if not path.exists() or not path.suffix == ".py":
            continue

        line_count = count_lines(path)

        # Check if file is grandfathered
        relative_path = str(path).replace("\\", "/")
        is_grandfathered = any(
            relative_path.endswith(gf) for gf in GRANDFATHERED_FILES
        )

        if line_count >= ERROR_THRESHOLD and not is_grandfathered:
            errors.append(f"{filepath}: {line_count} lines (max {ERROR_THRESHOLD})")
            exit_code = 1
        elif line_count >= WARN_THRESHOLD and not is_grandfathered:
            warnings.append(f"{filepath}: {line_count} lines")

    if warnings:
        print("Warning: Large files detected (consider refactoring):")
        for w in warnings:
            print(f"  - {w}")
        print()

    if errors:
        print("Error: Files exceed maximum line count:")
        for e in errors:
            print(f"  - {e}")
        print("\nPlease split these files into smaller modules.")
        print("If this is intentional, add the file to GRANDFATHERED_FILES in")
        print("scripts/hooks/check_file_size.py")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
