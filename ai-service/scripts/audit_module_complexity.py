#!/usr/bin/env python3
"""Audit module complexity across the codebase.

December 2025: Identifies complex modules that may benefit from refactoring.
Metrics include: LOC, class count, method count, cyclomatic complexity.

Usage:
    python scripts/audit_module_complexity.py
    python scripts/audit_module_complexity.py --top 20
    python scripts/audit_module_complexity.py --module coordination
    python scripts/audit_module_complexity.py --json > complexity_audit.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ModuleMetrics:
    """Complexity metrics for a single module."""

    file_path: str
    loc: int  # Lines of code (non-blank, non-comment)
    total_lines: int  # Total lines including blanks/comments
    class_count: int
    function_count: int
    method_count: int  # Methods inside classes
    max_function_loc: int
    max_function_name: str
    avg_function_loc: float
    import_count: int
    docstring_coverage: float  # Percentage of functions with docstrings

    @property
    def complexity_score(self) -> float:
        """Calculate overall complexity score (higher = more complex)."""
        # Weighted combination of metrics
        return (
            self.loc * 0.3
            + self.method_count * 5
            + self.function_count * 3
            + self.max_function_loc * 2
            + (1 - self.docstring_coverage) * 100
        )

    def to_dict(self) -> dict:
        return {
            "file": self.file_path,
            "loc": self.loc,
            "total_lines": self.total_lines,
            "classes": self.class_count,
            "functions": self.function_count,
            "methods": self.method_count,
            "max_function_loc": self.max_function_loc,
            "max_function_name": self.max_function_name,
            "avg_function_loc": round(self.avg_function_loc, 1),
            "import_count": self.import_count,
            "docstring_coverage": round(self.docstring_coverage * 100, 1),
            "complexity_score": round(self.complexity_score, 1),
        }


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to collect complexity metrics."""

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.classes: list[tuple[str, int, int]] = []  # (name, start, end)
        self.functions: list[tuple[str, int, int, bool]] = []  # (name, start, end, has_docstring)
        self.methods: list[tuple[str, str, int, int, bool]] = []  # (class, name, start, end, has_docstring)
        self.imports: int = 0
        self.current_class: str | None = None

    def visit_Import(self, node: ast.Import) -> None:
        self.imports += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports += len(node.names) if node.names else 1
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        end_line = self._get_end_line(node)
        self.classes.append((node.name, node.lineno, end_line))

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        end_line = self._get_end_line(node)
        has_docstring = (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )

        if self.current_class:
            self.methods.append((
                self.current_class,
                node.name,
                node.lineno,
                end_line,
                has_docstring,
            ))
        else:
            self.functions.append((
                node.name,
                node.lineno,
                end_line,
                has_docstring,
            ))

        self.generic_visit(node)

    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line of a node."""
        if hasattr(node, "end_lineno") and node.end_lineno:
            return node.end_lineno
        # Fallback: estimate from body
        max_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno") and child.lineno:
                max_line = max(max_line, child.lineno)
        return max_line


def count_loc(source: str) -> tuple[int, int]:
    """Count lines of code (excluding blanks and comments)."""
    lines = source.splitlines()
    total = len(lines)
    loc = 0
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    # Single-line docstring
                    continue
                in_docstring = True
                continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            continue

        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        loc += 1

    return loc, total


def analyze_module(file_path: Path) -> ModuleMetrics | None:
    """Analyze a single Python module for complexity metrics."""
    try:
        source = file_path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(file_path))

        visitor = ComplexityVisitor(source_lines)
        visitor.visit(tree)

        loc, total_lines = count_loc(source)

        # Calculate function/method metrics
        all_funcs = [
            (name, end - start + 1, has_doc)
            for name, start, end, has_doc in visitor.functions
        ] + [
            (f"{cls}.{name}", end - start + 1, has_doc)
            for cls, name, start, end, has_doc in visitor.methods
        ]

        if all_funcs:
            max_func = max(all_funcs, key=lambda x: x[1])
            max_function_loc = max_func[1]
            max_function_name = max_func[0]
            avg_function_loc = sum(f[1] for f in all_funcs) / len(all_funcs)
            docstring_count = sum(1 for f in all_funcs if f[2])
            docstring_coverage = docstring_count / len(all_funcs)
        else:
            max_function_loc = 0
            max_function_name = ""
            avg_function_loc = 0.0
            docstring_coverage = 1.0  # No functions = 100% coverage

        return ModuleMetrics(
            file_path=str(file_path),
            loc=loc,
            total_lines=total_lines,
            class_count=len(visitor.classes),
            function_count=len(visitor.functions),
            method_count=len(visitor.methods),
            max_function_loc=max_function_loc,
            max_function_name=max_function_name,
            avg_function_loc=avg_function_loc,
            import_count=visitor.imports,
            docstring_coverage=docstring_coverage,
        )

    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return None


def scan_directory(
    root: Path,
    module_filter: str | None = None,
    exclude_tests: bool = False,
) -> list[ModuleMetrics]:
    """Scan directory for module complexity."""
    results: list[ModuleMetrics] = []

    for py_file in root.rglob("*.py"):
        # Skip __pycache__ and hidden directories
        if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
            continue

        # Apply module filter
        if module_filter:
            if module_filter not in str(py_file):
                continue

        # Optionally exclude test files
        if exclude_tests and "test" in py_file.name.lower():
            continue

        metrics = analyze_module(py_file)
        if metrics:
            results.append(metrics)

    return results


def format_report(results: list[ModuleMetrics], top_n: int | None = None) -> str:
    """Format metrics as human-readable report."""
    # Sort by complexity score
    results = sorted(results, key=lambda m: -m.complexity_score)

    if top_n:
        results = results[:top_n]

    lines = [
        "=" * 100,
        "MODULE COMPLEXITY AUDIT REPORT",
        "=" * 100,
        "",
        f"Total modules scanned: {len(results)}",
        "",
        "TOP COMPLEX MODULES (by complexity score):",
        "-" * 100,
        "",
        f"{'Module':<60} {'LOC':>6} {'Classes':>8} {'Methods':>8} {'MaxFunc':>8} {'Score':>8}",
        "-" * 100,
    ]

    for m in results:
        short_path = m.file_path
        if "app/" in short_path:
            short_path = short_path[short_path.index("app/"):]
        if len(short_path) > 58:
            short_path = "..." + short_path[-55:]

        lines.append(
            f"{short_path:<60} {m.loc:>6} {m.class_count:>8} {m.method_count:>8} "
            f"{m.max_function_loc:>8} {m.complexity_score:>8.0f}"
        )

    # Summary statistics
    if results:
        total_loc = sum(m.loc for m in results)
        total_methods = sum(m.method_count for m in results)
        avg_complexity = sum(m.complexity_score for m in results) / len(results)

        lines.extend([
            "",
            "-" * 100,
            f"{'TOTALS':<60} {total_loc:>6} {'':<8} {total_methods:>8} {'':<8} {avg_complexity:>8.0f}",
            "",
            "REFACTORING RECOMMENDATIONS:",
            "-" * 100,
        ])

        # Find modules that need refactoring
        high_complexity = [m for m in results if m.complexity_score > 500]
        high_method_count = [m for m in results if m.method_count > 30]
        low_docstring = [m for m in results if m.docstring_coverage < 0.5 and m.method_count > 10]

        if high_complexity:
            lines.append(f"\n  High complexity (score > 500): {len(high_complexity)} modules")
            for m in high_complexity[:5]:
                lines.append(f"    - {m.file_path}")

        if high_method_count:
            lines.append(f"\n  High method count (> 30 methods): {len(high_method_count)} modules")
            for m in high_method_count[:5]:
                lines.append(f"    - {m.file_path} ({m.method_count} methods)")

        if low_docstring:
            lines.append(f"\n  Low docstring coverage (< 50%): {len(low_docstring)} modules")
            for m in low_docstring[:5]:
                lines.append(f"    - {m.file_path} ({m.docstring_coverage*100:.0f}%)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit module complexity in the codebase"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("app"),
        help="Root directory to scan (default: app)",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Filter to specific module (e.g., 'coordination')",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Show top N modules (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Exclude test files",
    )

    args = parser.parse_args()

    results = scan_directory(
        args.root,
        module_filter=args.module,
        exclude_tests=args.exclude_tests,
    )

    if args.json:
        output = {
            "modules": [m.to_dict() for m in sorted(results, key=lambda m: -m.complexity_score)],
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(results, top_n=args.top))


if __name__ == "__main__":
    main()
