#!/usr/bin/env python3
"""Audit test coverage by comparing source modules to test files.

December 2025: Identifies modules without corresponding test files.
This is a static analysis (not runtime coverage).

Usage:
    python scripts/audit_test_coverage.py
    python scripts/audit_test_coverage.py --module coordination
    python scripts/audit_test_coverage.py --json > coverage_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModuleCoverage:
    """Coverage information for a source module."""

    module_path: str
    module_name: str
    loc: int
    has_test: bool
    test_files: list[str] = field(default_factory=list)
    test_count: int = 0

    def to_dict(self) -> dict:
        return {
            "module": self.module_path,
            "name": self.module_name,
            "loc": self.loc,
            "has_test": self.has_test,
            "test_files": self.test_files,
            "test_count": self.test_count,
        }


def count_loc(file_path: Path) -> int:
    """Count lines of code in a file."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
    except (UnicodeDecodeError, OSError):
        return 0


def count_tests(file_path: Path) -> int:
    """Count test functions in a test file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        # Count def test_ and async def test_ functions
        count = content.count("def test_")
        count += content.count("async def test_")
        return count
    except (UnicodeDecodeError, OSError):
        return 0


def find_test_files(test_root: Path, module_name: str) -> list[tuple[Path, int]]:
    """Find test files that might test a given module."""
    test_files: list[tuple[Path, int]] = []

    # Common test file patterns
    patterns = [
        f"test_{module_name}.py",
        f"test_{module_name}s.py",  # plural
        f"{module_name}_test.py",
        f"tests_{module_name}.py",
    ]

    # Also check for partial matches
    for test_file in test_root.rglob("test_*.py"):
        file_name = test_file.name.lower()
        module_lower = module_name.lower()

        # Exact match
        if file_name in [p.lower() for p in patterns]:
            test_count = count_tests(test_file)
            test_files.append((test_file, test_count))
            continue

        # Partial match (module name appears in test file name)
        if module_lower in file_name.replace("test_", "").replace(".py", ""):
            test_count = count_tests(test_file)
            test_files.append((test_file, test_count))

    return test_files


def analyze_coverage(
    source_root: Path,
    test_root: Path,
    module_filter: str | None = None,
) -> list[ModuleCoverage]:
    """Analyze test coverage for source modules."""
    results: list[ModuleCoverage] = []

    for py_file in source_root.rglob("*.py"):
        # Skip __pycache__ and hidden directories
        if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
            continue

        # Skip __init__.py files
        if py_file.name == "__init__.py":
            continue

        # Apply module filter
        relative_path = str(py_file.relative_to(source_root))
        if module_filter and module_filter not in relative_path:
            continue

        module_name = py_file.stem  # filename without .py
        loc = count_loc(py_file)

        # Find corresponding test files
        test_files = find_test_files(test_root, module_name)

        has_test = len(test_files) > 0
        test_file_paths = [str(tf.relative_to(test_root)) for tf, _ in test_files]
        total_tests = sum(count for _, count in test_files)

        results.append(ModuleCoverage(
            module_path=relative_path,
            module_name=module_name,
            loc=loc,
            has_test=has_test,
            test_files=test_file_paths,
            test_count=total_tests,
        ))

    return results


def format_report(results: list[ModuleCoverage]) -> str:
    """Format coverage analysis as human-readable report."""
    # Sort by LOC (largest untested first)
    results = sorted(results, key=lambda m: (-int(not m.has_test), -m.loc))

    total_modules = len(results)
    tested_modules = sum(1 for m in results if m.has_test)
    untested_modules = total_modules - tested_modules

    total_loc = sum(m.loc for m in results)
    tested_loc = sum(m.loc for m in results if m.has_test)
    untested_loc = total_loc - tested_loc

    coverage_pct = (tested_modules / total_modules * 100) if total_modules > 0 else 0
    loc_coverage_pct = (tested_loc / total_loc * 100) if total_loc > 0 else 0

    lines = [
        "=" * 100,
        "TEST COVERAGE AUDIT REPORT",
        "=" * 100,
        "",
        "SUMMARY:",
        "-" * 50,
        f"  Total modules: {total_modules}",
        f"  Tested modules: {tested_modules} ({coverage_pct:.1f}%)",
        f"  Untested modules: {untested_modules}",
        f"",
        f"  Total LOC: {total_loc:,}",
        f"  Tested LOC: {tested_loc:,} ({loc_coverage_pct:.1f}%)",
        f"  Untested LOC: {untested_loc:,}",
        "",
        "UNTESTED MODULES (sorted by LOC):",
        "-" * 100,
        "",
        f"{'Module':<70} {'LOC':>8}",
        "-" * 100,
    ]

    # Show untested modules
    untested = [m for m in results if not m.has_test]
    for m in untested[:50]:
        short_path = m.module_path
        if len(short_path) > 68:
            short_path = "..." + short_path[-65:]
        lines.append(f"{short_path:<70} {m.loc:>8}")

    if len(untested) > 50:
        lines.append(f"... and {len(untested) - 50} more untested modules")

    # Show tested modules summary
    lines.extend([
        "",
        "TESTED MODULES:",
        "-" * 100,
        "",
        f"{'Module':<50} {'LOC':>8} {'Tests':>8} {'Test Files':>30}",
        "-" * 100,
    ])

    tested = [m for m in results if m.has_test][:30]
    for m in tested:
        short_path = m.module_name
        if len(short_path) > 48:
            short_path = short_path[-48:]
        test_file_str = ", ".join(m.test_files[:2])
        if len(m.test_files) > 2:
            test_file_str += f", +{len(m.test_files) - 2}"
        if len(test_file_str) > 28:
            test_file_str = test_file_str[:25] + "..."

        lines.append(f"{short_path:<50} {m.loc:>8} {m.test_count:>8} {test_file_str:>30}")

    # Group by subdirectory
    by_subdir: dict[str, tuple[int, int, int, int]] = defaultdict(lambda: (0, 0, 0, 0))
    for m in results:
        parts = Path(m.module_path).parts
        subdir = parts[0] if parts else "root"
        total, tested, total_loc, tested_loc = by_subdir[subdir]
        by_subdir[subdir] = (
            total + 1,
            tested + (1 if m.has_test else 0),
            total_loc + m.loc,
            tested_loc + (m.loc if m.has_test else 0),
        )

    lines.extend([
        "",
        "COVERAGE BY SUBDIRECTORY:",
        "-" * 100,
        "",
        f"{'Subdirectory':<30} {'Modules':>10} {'Tested':>10} {'Coverage':>10} {'LOC':>10} {'Tested LOC':>12}",
        "-" * 100,
    ])

    for subdir in sorted(by_subdir.keys(), key=lambda s: -by_subdir[s][0]):
        total, tested, total_loc, tested_loc = by_subdir[subdir]
        pct = (tested / total * 100) if total > 0 else 0
        lines.append(
            f"{subdir:<30} {total:>10} {tested:>10} {pct:>9.1f}% {total_loc:>10} {tested_loc:>12}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit test coverage for source modules"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("app"),
        help="Source directory (default: app)",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=Path("tests"),
        help="Test directory (default: tests)",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Filter to specific module (e.g., 'coordination')",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    results = analyze_coverage(
        args.source,
        args.tests,
        module_filter=args.module,
    )

    if args.json:
        output = {
            "total_modules": len(results),
            "tested_modules": sum(1 for m in results if m.has_test),
            "modules": [m.to_dict() for m in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(results))


if __name__ == "__main__":
    main()
