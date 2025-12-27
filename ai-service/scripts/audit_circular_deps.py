#!/usr/bin/env python3
"""Audit circular dependencies in the codebase.

Scans Python files in ai-service and detects circular import dependencies.
Useful for identifying import cycles that could cause runtime issues.

Usage:
    python scripts/audit_circular_deps.py [--path PATH] [--verbose] [--max-depth N]

Examples:
    # Audit entire ai-service
    python scripts/audit_circular_deps.py

    # Audit specific package
    python scripts/audit_circular_deps.py --path app/coordination

    # Verbose output with max cycle depth
    python scripts/audit_circular_deps.py --verbose --max-depth 5
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def extract_imports(file_path: Path, base_path: Path) -> list[str]:
    """Extract all imports from a Python file.

    Returns list of module paths that are within the project.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("app.", "scripts.")):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(("app.", "scripts.")):
                imports.append(node.module)

    return imports


def module_path_from_file(file_path: Path, base_path: Path) -> str | None:
    """Convert file path to module path."""
    try:
        rel_path = file_path.relative_to(base_path)
        parts = list(rel_path.parts)

        # Remove .py extension
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # Handle __init__.py
        if parts[-1] == "__init__":
            parts = parts[:-1]

        if not parts:
            return None

        return ".".join(parts)
    except ValueError:
        return None


def build_dependency_graph(
    base_path: Path,
    target_path: Path | None = None,
) -> dict[str, set[str]]:
    """Build a dependency graph from Python files.

    Returns dict mapping module -> set of imported modules.
    """
    graph: dict[str, set[str]] = defaultdict(set)

    search_path = target_path or base_path

    for py_file in search_path.rglob("*.py"):
        # Skip test files and archive
        if "test" in py_file.parts or "archive" in py_file.parts:
            continue
        if "__pycache__" in py_file.parts:
            continue

        module = module_path_from_file(py_file, base_path)
        if not module:
            continue

        imports = extract_imports(py_file, base_path)
        for imp in imports:
            graph[module].add(imp)

    return graph


def find_cycles(
    graph: dict[str, set[str]],
    max_depth: int = 10,
) -> list[list[str]]:
    """Find all cycles in the dependency graph using DFS."""
    cycles = []
    visited = set()
    path = []
    path_set = set()

    def dfs(node: str, depth: int = 0) -> None:
        if depth > max_depth:
            return

        if node in path_set:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)
        path_set.add(node)

        for neighbor in graph.get(node, []):
            dfs(neighbor, depth + 1)

        path.pop()
        path_set.remove(node)

    for start_node in graph:
        visited.clear()
        path.clear()
        path_set.clear()
        dfs(start_node)

    # Deduplicate cycles (same cycle can be found from different starting points)
    unique_cycles = []
    seen = set()
    for cycle in cycles:
        # Normalize cycle by rotating to start with smallest element
        min_idx = cycle[:-1].index(min(cycle[:-1]))
        normalized = tuple(cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]])
        if normalized not in seen:
            seen.add(normalized)
            unique_cycles.append(list(normalized))

    return unique_cycles


def categorize_cycles(cycles: list[list[str]]) -> dict[str, list[list[str]]]:
    """Categorize cycles by severity."""
    categories = {
        "critical": [],  # Cross-package cycles
        "high": [],      # Same package, different modules
        "low": [],       # Same subpackage
    }

    for cycle in cycles:
        packages = set()
        for module in cycle[:-1]:
            parts = module.split(".")
            if len(parts) >= 2:
                packages.add(parts[1])  # app.{package}

        if len(packages) > 1:
            categories["critical"].append(cycle)
        elif len(cycle) > 3:
            categories["high"].append(cycle)
        else:
            categories["low"].append(cycle)

    return categories


def print_report(
    cycles: list[list[str]],
    graph: dict[str, set[str]],
    verbose: bool = False,
) -> None:
    """Print audit report."""
    print("=" * 70)
    print("CIRCULAR DEPENDENCY AUDIT REPORT")
    print("=" * 70)
    print()

    total_modules = len(graph)
    total_deps = sum(len(deps) for deps in graph.values())
    print(f"Modules scanned: {total_modules}")
    print(f"Total dependencies: {total_deps}")
    print(f"Cycles found: {len(cycles)}")
    print()

    if not cycles:
        print("No circular dependencies detected.")
        return

    categories = categorize_cycles(cycles)

    for severity, cycle_list in categories.items():
        if not cycle_list:
            continue

        print(f"\n{severity.upper()} SEVERITY ({len(cycle_list)} cycles):")
        print("-" * 50)

        for i, cycle in enumerate(cycle_list, 1):
            print(f"\n  Cycle {i}:")
            for j, module in enumerate(cycle):
                arrow = " -> " if j < len(cycle) - 1 else ""
                print(f"    {module}{arrow}")

    if verbose:
        print("\n" + "=" * 70)
        print("DEPENDENCY DETAILS")
        print("=" * 70)

        for module, deps in sorted(graph.items()):
            if deps:
                print(f"\n{module}:")
                for dep in sorted(deps):
                    print(f"  -> {dep}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit circular dependencies in the codebase"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Specific path to audit (relative to ai-service)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed dependency information",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum cycle depth to search (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    # Determine base path
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent  # ai-service/

    target_path = None
    if args.path:
        target_path = base_path / args.path
        if not target_path.exists():
            print(f"Error: Path not found: {target_path}", file=sys.stderr)
            return 1

    print(f"Scanning: {target_path or base_path}")
    print()

    graph = build_dependency_graph(base_path, target_path)
    cycles = find_cycles(graph, args.max_depth)

    if args.json:
        import json
        result = {
            "modules": len(graph),
            "dependencies": sum(len(deps) for deps in graph.values()),
            "cycles": [
                {"path": cycle, "severity": _get_severity(cycle)}
                for cycle in cycles
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        print_report(cycles, graph, args.verbose)

    # Return non-zero if critical cycles found
    categories = categorize_cycles(cycles)
    return 1 if categories["critical"] else 0


def _get_severity(cycle: list[str]) -> str:
    """Get severity for a single cycle."""
    packages = set()
    for module in cycle[:-1]:
        parts = module.split(".")
        if len(parts) >= 2:
            packages.add(parts[1])

    if len(packages) > 1:
        return "critical"
    elif len(cycle) > 3:
        return "high"
    return "low"


if __name__ == "__main__":
    sys.exit(main())
