#!/usr/bin/env python3
"""Audit broad exception handlers across the codebase.

December 2025: Identifies `except Exception:` handlers that should be narrowed
to specific exception types for better debugging and error handling.

Usage:
    python scripts/audit_exception_handlers.py
    python scripts/audit_exception_handlers.py --top 20
    python scripts/audit_exception_handlers.py --module coordination
    python scripts/audit_exception_handlers.py --json > exception_audit.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ExceptionHandler:
    """Represents a broad exception handler found in code."""

    file_path: str
    line_number: int
    handler_type: str  # "Exception", "BaseException", or bare "except:"
    has_logging: bool
    has_reraise: bool
    context: str  # surrounding code context
    function_name: str | None
    class_name: str | None

    def to_dict(self) -> dict:
        return {
            "file": self.file_path,
            "line": self.line_number,
            "type": self.handler_type,
            "has_logging": self.has_logging,
            "has_reraise": self.has_reraise,
            "context": self.context,
            "function": self.function_name,
            "class": self.class_name,
        }


@dataclass
class AuditResult:
    """Results of exception handler audit."""

    handlers: list[ExceptionHandler] = field(default_factory=list)
    files_scanned: int = 0
    total_handlers: int = 0
    by_module: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_handler(self, handler: ExceptionHandler) -> None:
        self.handlers.append(handler)
        self.total_handlers += 1
        # Extract module from path
        parts = Path(handler.file_path).parts
        if "app" in parts:
            idx = parts.index("app")
            if idx + 1 < len(parts):
                module = parts[idx + 1]
                self.by_module[module] += 1
        self.by_type[handler.handler_type] += 1


class ExceptionHandlerVisitor(ast.NodeVisitor):
    """AST visitor to find broad exception handlers."""

    def __init__(self, file_path: str, source_lines: list[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.handlers: list[ExceptionHandler] = []
        self.current_class: str | None = None
        self.current_function: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        handler_type = self._get_handler_type(node)
        if handler_type in ("Exception", "BaseException", "bare"):
            context = self._get_context(node.lineno)
            has_logging = self._has_logging(node)
            has_reraise = self._has_reraise(node)

            self.handlers.append(
                ExceptionHandler(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    handler_type=handler_type,
                    has_logging=has_logging,
                    has_reraise=has_reraise,
                    context=context,
                    function_name=self.current_function,
                    class_name=self.current_class,
                )
            )
        self.generic_visit(node)

    def _get_handler_type(self, node: ast.ExceptHandler) -> str:
        if node.type is None:
            return "bare"
        if isinstance(node.type, ast.Name):
            return node.type.id
        if isinstance(node.type, ast.Attribute):
            return node.type.attr
        return "unknown"

    def _get_context(self, line_no: int, context_lines: int = 3) -> str:
        start = max(0, line_no - context_lines - 1)
        end = min(len(self.source_lines), line_no + context_lines)
        lines = self.source_lines[start:end]
        return "\n".join(f"{start + i + 1:4d}: {line.rstrip()}" for i, line in enumerate(lines))

    def _has_logging(self, node: ast.ExceptHandler) -> bool:
        """Check if handler contains logging call."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr in ("debug", "info", "warning", "error", "exception", "critical"):
                        return True
        return False

    def _has_reraise(self, node: ast.ExceptHandler) -> bool:
        """Check if handler re-raises the exception."""
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                return True
        return False


def scan_file(file_path: Path) -> Iterator[ExceptionHandler]:
    """Scan a single Python file for broad exception handlers."""
    try:
        source = file_path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(file_path))
        visitor = ExceptionHandlerVisitor(str(file_path), source_lines)
        visitor.visit(tree)
        yield from visitor.handlers
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)


def scan_directory(
    root: Path,
    module_filter: str | None = None,
    exclude_tests: bool = False,
) -> AuditResult:
    """Scan directory for broad exception handlers."""
    result = AuditResult()

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

        result.files_scanned += 1
        for handler in scan_file(py_file):
            result.add_handler(handler)

    return result


def format_report(result: AuditResult, top_n: int | None = None) -> str:
    """Format audit result as human-readable report."""
    lines = [
        "=" * 80,
        "EXCEPTION HANDLER AUDIT REPORT",
        "=" * 80,
        "",
        f"Files scanned: {result.files_scanned}",
        f"Total broad handlers: {result.total_handlers}",
        "",
        "BY MODULE:",
        "-" * 40,
    ]

    for module, count in sorted(result.by_module.items(), key=lambda x: -x[1]):
        lines.append(f"  {module:30s} {count:5d}")

    lines.extend([
        "",
        "BY TYPE:",
        "-" * 40,
    ])

    for handler_type, count in sorted(result.by_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {handler_type:30s} {count:5d}")

    # Sort handlers by priority (no logging = higher priority)
    handlers = sorted(
        result.handlers,
        key=lambda h: (h.has_logging, h.has_reraise, h.file_path, h.line_number),
    )

    if top_n:
        handlers = handlers[:top_n]

    lines.extend([
        "",
        f"TOP {len(handlers)} HANDLERS TO NARROW:",
        "-" * 80,
    ])

    for i, h in enumerate(handlers, 1):
        status = []
        if not h.has_logging:
            status.append("NO LOGGING")
        if not h.has_reraise:
            status.append("NO RERAISE")
        status_str = ", ".join(status) if status else "has logging"

        lines.extend([
            "",
            f"[{i}] {h.file_path}:{h.line_number}",
            f"    Type: except {h.handler_type}:",
            f"    Location: {h.class_name or ''}.{h.function_name or '<module>'}",
            f"    Status: {status_str}",
            f"    Context:",
        ])
        for ctx_line in h.context.split("\n"):
            lines.append(f"        {ctx_line}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit broad exception handlers in the codebase"
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
        default=20,
        help="Show top N handlers (default: 20)",
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

    result = scan_directory(
        args.root,
        module_filter=args.module,
        exclude_tests=args.exclude_tests,
    )

    if args.json:
        output = {
            "files_scanned": result.files_scanned,
            "total_handlers": result.total_handlers,
            "by_module": dict(result.by_module),
            "by_type": dict(result.by_type),
            "handlers": [h.to_dict() for h in result.handlers],
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(result, top_n=args.top))


if __name__ == "__main__":
    main()
