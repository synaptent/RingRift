#!/usr/bin/env python3
"""Audit Python import graph health for app/scripts/tests modules.

Supports two primary Phase 17 use cases:
1. Listing zero-inbound modules to review for dead-code cleanup.
2. Detecting circular dependencies in a filtered module subtree.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_ROOTS = ("app", "scripts", "tests")
SKIP_PARTS = {"__pycache__", "archive"}


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        yield path


def _module_from_file(root: Path, path: Path) -> str | None:
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        return None

    parts = list(rel_path.parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    prefix = root.name
    return ".".join([prefix] + parts) if parts else prefix


def _module_candidates_from_import_from(
    current_module: str,
    current_path: Path,
    node: ast.ImportFrom,
) -> list[str]:
    """Resolve a from-import to candidate module names.

    This handles both absolute imports and relative imports such as
    ``from .gpu_board_encoding import GPUBoardEncodingMixin`` and
    ``from app.coordination import _exports_core``.
    """

    candidates: list[str] = []
    module_name = node.module or ""

    if node.level:
        current_parts = current_module.split(".")
        package_parts = current_parts if current_path.name == "__init__.py" else current_parts[:-1]
        if node.level - 1 > len(package_parts):
            return []
        base_parts = package_parts[: len(package_parts) - (node.level - 1)]
        if module_name:
            base_parts = base_parts + module_name.split(".")
        base_module = ".".join(base_parts)
    else:
        base_module = module_name

    if not base_module.startswith(DEFAULT_ROOTS):
        if not any(base_module == root or base_module.startswith(f"{root}.") for root in DEFAULT_ROOTS):
            return []

    if base_module:
        candidates.append(base_module)

    for alias in node.names:
        if alias.name == "*":
            continue
        if base_module:
            candidates.append(f"{base_module}.{alias.name}")

    return candidates


def _build_module_index(base_dir: Path, roots: tuple[str, ...]) -> dict[str, Path]:
    module_to_path: dict[str, Path] = {}
    for root_name in roots:
        root = base_dir / root_name
        if not root.exists():
            continue
        for path in _iter_python_files(root):
            module = _module_from_file(root, path)
            if module:
                module_to_path[module] = path
    return module_to_path


def _normalize_to_known_module(candidate: str, module_to_path: dict[str, Path]) -> str | None:
    target = candidate
    while target:
        if target in module_to_path:
            return target
        if "." not in target:
            return None
        target = target.rsplit(".", 1)[0]
    return None


def build_import_graph(base_dir: Path, roots: tuple[str, ...]) -> tuple[dict[str, set[str]], dict[str, Path]]:
    module_to_path = _build_module_index(base_dir, roots)
    graph: dict[str, set[str]] = defaultdict(set)

    for module, path in module_to_path.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            raw_candidates: list[str] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    raw_candidates.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                raw_candidates.extend(_module_candidates_from_import_from(module, path, node))

            for candidate in raw_candidates:
                normalized = _normalize_to_known_module(candidate, module_to_path)
                if normalized is None or normalized == module:
                    continue
                graph[module].add(normalized)

        graph.setdefault(module, set())

    return graph, module_to_path


def find_cycles(graph: dict[str, set[str]], max_depth: int = 10) -> list[list[str]]:
    cycles: list[list[str]] = []
    path: list[str] = []
    path_set: set[str] = set()

    def dfs(node: str, depth: int = 0) -> None:
        if depth > max_depth:
            return
        if node in path_set:
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return

        path.append(node)
        path_set.add(node)
        for neighbor in graph.get(node, ()):
            dfs(neighbor, depth + 1)
        path.pop()
        path_set.remove(node)

    for start in sorted(graph):
        dfs(start)

    unique_cycles: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for cycle in cycles:
        headless = cycle[:-1]
        if not headless:
            continue
        min_index = min(range(len(headless)), key=headless.__getitem__)
        normalized = tuple(headless[min_index:] + headless[:min_index] + [headless[min_index]])
        if normalized not in seen:
            seen.add(normalized)
            unique_cycles.append(list(normalized))
    return unique_cycles


def zero_inbound_modules(
    graph: dict[str, set[str]],
    module_to_path: dict[str, Path],
    module_prefix: str,
) -> list[tuple[str, Path]]:
    inbound: dict[str, set[str]] = defaultdict(set)
    for module, dependencies in graph.items():
        for dependency in dependencies:
            inbound[dependency].add(module)

    candidates: list[tuple[str, Path]] = []
    for module, path in sorted(module_to_path.items()):
        if path.name == "__init__.py":
            continue
        if module_prefix and not (module == module_prefix or module.startswith(f"{module_prefix}.")):
            continue
        if inbound[module]:
            continue
        candidates.append((module, path))
    return candidates


def _serialize_cycles(cycles: list[list[str]]) -> list[list[str]]:
    return [list(cycle) for cycle in cycles]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Python import graph health")
    parser.add_argument(
        "--roots",
        default=",".join(DEFAULT_ROOTS),
        help="Comma-separated roots relative to ai-service/ (default: app,scripts,tests)",
    )
    parser.add_argument(
        "--module-prefix",
        default="app",
        help="Only report modules starting with this prefix (default: app)",
    )
    parser.add_argument(
        "--report",
        choices=("summary", "cycles", "zero-inbound", "all"),
        default="summary",
        help="Report type to print",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum DFS depth for cycle detection (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plain text",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    roots = tuple(part.strip() for part in args.roots.split(",") if part.strip())
    graph, module_to_path = build_import_graph(base_dir, roots)
    filtered_modules = sorted(
        module for module in module_to_path if module == args.module_prefix or module.startswith(f"{args.module_prefix}.")
    )
    filtered_graph = {module: {dep for dep in graph[module] if dep in filtered_modules} for module in filtered_modules}
    cycles = find_cycles(filtered_graph, max_depth=args.max_depth)
    zero_inbound = zero_inbound_modules(graph, module_to_path, args.module_prefix)

    summary = {
        "roots": list(roots),
        "module_prefix": args.module_prefix,
        "module_count": len(filtered_modules),
        "dependency_count": sum(len(deps) for deps in filtered_graph.values()),
        "cycle_count": len(cycles),
        "zero_inbound_count": len(zero_inbound),
    }

    if args.json:
        payload: dict[str, object] = {"summary": summary}
        if args.report in {"cycles", "all"}:
            payload["cycles"] = _serialize_cycles(cycles)
        if args.report in {"zero-inbound", "all"}:
            payload["zero_inbound"] = [
                {"module": module, "path": str(path.relative_to(base_dir))}
                for module, path in zero_inbound
            ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"roots={','.join(roots)}")
    print(f"module_prefix={args.module_prefix}")
    print(f"modules={summary['module_count']}")
    print(f"dependencies={summary['dependency_count']}")
    print(f"cycles={summary['cycle_count']}")
    print(f"zero_inbound={summary['zero_inbound_count']}")

    if args.report in {"cycles", "all"}:
        print("\nCycles:")
        for cycle in cycles:
            print("  " + " -> ".join(cycle))

    if args.report in {"zero-inbound", "all"}:
        print("\nZero-inbound modules:")
        for module, path in zero_inbound:
            print(f"  {module}  ({path.relative_to(base_dir)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
