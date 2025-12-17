#!/usr/bin/env python3
"""Pre-commit hook to detect hardcoded resource thresholds.

This hook checks for hardcoded resource thresholds that bypass the unified
resource_guard module. All resource limits should use resource_guard for
consistent 80% utilization enforcement.

Usage:
    python scripts/hooks/check_resource_thresholds.py [files...]

    # Check all Python files
    python scripts/hooks/check_resource_thresholds.py $(git ls-files '*.py')

    # As a pre-commit hook
    cp scripts/hooks/check_resource_thresholds.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Returns exit code 0 if no issues, 1 if hardcoded thresholds found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Files that are allowed to define resource limits
ALLOWED_FILES = {
    "app/utils/resource_guard.py",
    "app/coordination/resource_targets.py",
    "app/coordination/safeguards.py",
    "tests/test_resource_guard.py",
    "scripts/hooks/check_resource_thresholds.py",  # This file
}

# Patterns that indicate hardcoded thresholds EXCEEDING allowed limits
# Values at exactly 80% (or 70% for disk) are OK - only flag values above
THRESHOLD_PATTERNS = [
    # Disk thresholds > 70% (71-100)
    (r'disk.*(?:threshold|limit|max|critical).*[=:]\s*(?:0\.)?([789][1-9]|[89]\d|1\d{2})', "disk threshold >70%"),
    # Memory thresholds > 80% (81-100)
    (r'memory.*(?:threshold|limit|max|critical).*[=:]\s*(?:0\.)?([89][1-9]|9\d|1\d{2})', "memory threshold >80%"),
    # CPU thresholds > 80% (81-100)
    (r'cpu.*(?:threshold|limit|max|critical).*[=:]\s*(?:0\.)?([89][1-9]|9\d|1\d{2})', "cpu threshold >80%"),
    # GPU thresholds > 80% (81-100)
    (r'gpu.*(?:threshold|limit|max|critical).*[=:]\s*(?:0\.)?([89][1-9]|9\d|1\d{2})', "gpu threshold >80%"),
    # Load thresholds > 80% (81-100)
    (r'load.*(?:threshold|limit|max).*[=:]\s*(?:0\.)?([89][1-9]|9\d|1\d{2})', "load threshold >80%"),
    # Generic high utilization patterns > 80%
    (r'(?:UTIL|util).*MAX.*[=:]\s*(?:0\.)?([89][1-9]|9\d|1\d{2})', "utilization max >80%"),
]

# Patterns that indicate proper resource_guard usage (allow these)
GUARD_PATTERNS = [
    r'from\s+app\.utils\.resource_guard\s+import',
    r'import.*resource_guard',
    r'HAS_RESOURCE_GUARD',
    r'resource_can_proceed',
    r'check_disk_space',
    r'check_memory',
    r'RESOURCE_LIMITS',
]


def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check a file for hardcoded resource thresholds.

    Args:
        filepath: Path to the Python file to check

    Returns:
        List of (line_number, line_content, issue_description) tuples
    """
    issues = []

    # Skip allowed files
    rel_path = str(filepath)
    for allowed in ALLOWED_FILES:
        if rel_path.endswith(allowed):
            return []

    try:
        content = filepath.read_text()
        lines = content.split("\n")
    except Exception:
        return []

    # Check if file uses resource_guard
    uses_guard = any(re.search(pattern, content, re.IGNORECASE) for pattern in GUARD_PATTERNS)

    for i, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check for hardcoded thresholds
        for pattern, desc in THRESHOLD_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Don't flag if this file imports resource_guard
                # and the line is part of a fallback
                if uses_guard and "fallback" in line.lower():
                    continue
                if uses_guard and "default" in line.lower():
                    continue
                if "# OK:" in line or "# ALLOW:" in line:
                    continue

                issues.append((i, line.strip()[:80], desc))

    return issues


def main() -> int:
    """Main entry point.

    Returns:
        0 if no issues, 1 if hardcoded thresholds found
    """
    if len(sys.argv) < 2:
        # No files specified, check all Python files in ai-service
        ai_service = Path(__file__).parent.parent.parent
        files = list(ai_service.glob("**/*.py"))
    else:
        files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]

    all_issues = []

    for filepath in files:
        if not filepath.exists():
            continue

        issues = check_file(filepath)
        if issues:
            all_issues.append((filepath, issues))

    if all_issues:
        print("=" * 70)
        print("RESOURCE THRESHOLD VIOLATIONS DETECTED")
        print("=" * 70)
        print()
        print("The following files have hardcoded resource thresholds that may")
        print("bypass the unified resource_guard module (80% limit enforcement).")
        print()
        print("To fix:")
        print("1. Import from app.utils.resource_guard")
        print("2. Use check_disk_space(), check_memory(), check_cpu(), etc.")
        print("3. Or add '# OK: <reason>' comment if intentional")
        print()

        for filepath, issues in all_issues:
            print(f"\n{filepath}:")
            for line_num, line_content, desc in issues:
                print(f"  Line {line_num}: {desc}")
                print(f"    {line_content}")

        print()
        print(f"Total: {sum(len(i) for _, i in all_issues)} issues in {len(all_issues)} files")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
