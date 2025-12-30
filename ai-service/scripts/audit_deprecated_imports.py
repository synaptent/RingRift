#!/usr/bin/env python3
"""
Deprecation Enforcement Script

Scans the codebase for:
1. Deprecated modules that lack proper deprecation warnings
2. Usage of deprecated modules in non-test code
3. Missing replacement documentation for deprecated modules

Usage:
    python scripts/audit_deprecated_imports.py           # Check all
    python scripts/audit_deprecated_imports.py --strict  # Fail on any issue
    python scripts/audit_deprecated_imports.py --fix     # Add missing deprecation warnings
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DeprecationInfo:
    """Information about a deprecated module."""

    module_path: Path
    has_warning: bool
    replacement: Optional[str] = None
    removal_date: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ImportViolation:
    """A violation of deprecated import usage."""

    file_path: Path
    line_number: int
    imported_module: str
    replacement: Optional[str] = None


@dataclass
class AuditResults:
    """Results of the deprecation audit."""

    deprecated_modules: list[DeprecationInfo] = field(default_factory=list)
    missing_warnings: list[Path] = field(default_factory=list)
    import_violations: list[ImportViolation] = field(default_factory=list)
    missing_replacements: list[Path] = field(default_factory=list)


# Known deprecated modules and their replacements
DEPRECATED_MODULES = {
    # Format: "module.path": ("replacement", "removal_date")
    "app.coordination.auto_evaluation_daemon": (
        "app.coordination.evaluation_daemon",
        "Q2 2026",
    ),
    "app.coordination.replication_monitor": (
        "app.coordination.unified_replication_daemon",
        "Q2 2026",
    ),
    "app.coordination.replication_repair_daemon": (
        "app.coordination.unified_replication_daemon",
        "Q2 2026",
    ),
    "app.coordination.cross_process_events": (
        "app.coordination.event_router",
        "Q2 2026",
    ),
    "app.coordination.system_health_monitor": (
        "app.coordination.unified_health_manager",
        "Q2 2026",
    ),
    "app.coordination.node_health_monitor": (
        "app.coordination.health_check_orchestrator",
        "Q2 2026",
    ),
    "app.coordination.queue_populator_daemon": (
        "app.coordination.unified_queue_populator",
        "Q2 2026",
    ),
    "app.coordination.model_distribution_daemon": (
        "app.coordination.unified_distribution_daemon",
        "Q2 2026",
    ),
    "app.coordination.npz_distribution_daemon": (
        "app.coordination.unified_distribution_daemon",
        "Q2 2026",
    ),
    "app.coordination.bandwidth_manager": (
        "app.coordination.resources.bandwidth",
        "Q2 2026",
    ),
    "app.sync.cluster_hosts": ("app.config.cluster_config", "Q2 2026"),
    "app.core.singleton_mixin": ("app.coordination.singleton_mixin", "Q2 2026"),
    "app.ai.gmo_ai": ("app.ai.gumbel_search_engine", "Q2 2026"),
    "app.ai.gmo_v2": ("app.ai.gumbel_search_engine", "Q2 2026"),
    "app.ai.ebmo_ai": ("app.ai.gumbel_search_engine", "Q2 2026"),
    "app.ai.neural_net.v6_large": ("app.ai.neural_net.v5_heavy_large", "Q2 2026"),
}

# Patterns that indicate a deprecation warning
DEPRECATION_PATTERNS = [
    r"warnings\.warn\([^)]*DeprecationWarning",
    r"DeprecationWarning",
    r"deprecated",
    r"DEPRECATED",
]

# Directories to exclude from violation checking (test files can use deprecated imports)
EXCLUDED_DIRS = {"tests", "archive", "deprecated", ".venv", "__pycache__"}


def find_python_files(root: Path, exclude_dirs: set[str] = None) -> list[Path]:
    """Find all Python files in a directory tree."""
    exclude_dirs = exclude_dirs or set()
    files = []
    for path in root.rglob("*.py"):
        if not any(excluded in path.parts for excluded in exclude_dirs):
            files.append(path)
    return files


def has_deprecation_warning(file_path: Path) -> bool:
    """Check if a file contains a deprecation warning."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return any(re.search(pattern, content) for pattern in DEPRECATION_PATTERNS)
    except Exception:
        return False


def extract_deprecation_info(file_path: Path) -> Optional[DeprecationInfo]:
    """Extract deprecation information from a file."""
    try:
        content = file_path.read_text(encoding="utf-8")

        has_warning = any(
            re.search(pattern, content) for pattern in DEPRECATION_PATTERNS
        )

        # Try to find replacement info
        replacement = None
        replacement_match = re.search(
            r"(?:use|Use|import from|replaced by|replacement:?)\s+[`'\"]?([a-zA-Z0-9_.]+)[`'\"]?",
            content,
        )
        if replacement_match:
            replacement = replacement_match.group(1)

        # Try to find removal date
        removal_date = None
        date_match = re.search(r"(Q[1-4] 20\d{2}|20\d{2}[-/]?\d{2})", content)
        if date_match:
            removal_date = date_match.group(1)

        return DeprecationInfo(
            module_path=file_path,
            has_warning=has_warning,
            replacement=replacement,
            removal_date=removal_date,
        )
    except Exception:
        return None


def find_deprecated_imports(file_path: Path) -> list[ImportViolation]:
    """Find imports of deprecated modules in a file."""
    violations = []
    seen = set()  # Track (line, module) to avoid duplicates

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in DEPRECATED_MODULES:
                        key = (node.lineno, alias.name)
                        if key not in seen:
                            seen.add(key)
                            replacement, _ = DEPRECATED_MODULES[alias.name]
                            violations.append(
                                ImportViolation(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    imported_module=alias.name,
                                    replacement=replacement,
                                )
                            )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Check exact match first
                    if node.module in DEPRECATED_MODULES:
                        key = (node.lineno, node.module)
                        if key not in seen:
                            seen.add(key)
                            replacement, _ = DEPRECATED_MODULES[node.module]
                            violations.append(
                                ImportViolation(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    imported_module=node.module,
                                    replacement=replacement,
                                )
                            )
                    else:
                        # Check for imports from deprecated module subpaths
                        for deprecated_module in DEPRECATED_MODULES:
                            if (
                                node.module.startswith(deprecated_module + ".")
                                and node.module != deprecated_module
                            ):
                                key = (node.lineno, node.module)
                                if key not in seen:
                                    seen.add(key)
                                    replacement, _ = DEPRECATED_MODULES[
                                        deprecated_module
                                    ]
                                    violations.append(
                                        ImportViolation(
                                            file_path=file_path,
                                            line_number=node.lineno,
                                            imported_module=node.module,
                                            replacement=replacement,
                                        )
                                    )
                                break
    except SyntaxError:
        pass
    except Exception:
        pass

    return violations


def audit_deprecations(root: Path) -> AuditResults:
    """Run a full deprecation audit."""
    results = AuditResults()

    # Check deprecated modules for proper warnings
    for module_path, (replacement, removal_date) in DEPRECATED_MODULES.items():
        # Convert module path to file path
        file_path = root / module_path.replace(".", "/")
        file_path = file_path.with_suffix(".py")

        if file_path.exists():
            info = extract_deprecation_info(file_path)
            if info:
                info.replacement = replacement
                info.removal_date = removal_date
                results.deprecated_modules.append(info)

                if not info.has_warning:
                    results.missing_warnings.append(file_path)

    # Find import violations in non-test code
    for py_file in find_python_files(root / "app", EXCLUDED_DIRS):
        violations = find_deprecated_imports(py_file)
        # Filter out self-references (deprecated module importing from itself is OK)
        for v in violations:
            module_from_path = str(py_file.relative_to(root)).replace("/", ".")[:-3]
            if not module_from_path.startswith(v.imported_module):
                results.import_violations.append(v)

    # Also check scripts
    for py_file in find_python_files(root / "scripts", EXCLUDED_DIRS):
        violations = find_deprecated_imports(py_file)
        results.import_violations.extend(violations)

    return results


def print_report(results: AuditResults) -> None:
    """Print a human-readable audit report."""
    print("=" * 70)
    print("DEPRECATION AUDIT REPORT")
    print("=" * 70)

    # Summary
    print(f"\nDeprecated modules tracked: {len(results.deprecated_modules)}")
    print(f"Modules missing warnings:   {len(results.missing_warnings)}")
    print(f"Import violations:          {len(results.import_violations)}")

    # Missing warnings
    if results.missing_warnings:
        print("\n" + "-" * 70)
        print("MODULES MISSING DEPRECATION WARNINGS")
        print("-" * 70)
        for path in results.missing_warnings:
            print(f"  - {path}")

    # Import violations
    if results.import_violations:
        print("\n" + "-" * 70)
        print("DEPRECATED IMPORT VIOLATIONS")
        print("-" * 70)
        for v in results.import_violations:
            print(f"  {v.file_path}:{v.line_number}")
            print(f"    imports: {v.imported_module}")
            if v.replacement:
                print(f"    use:     {v.replacement}")
            print()

    # Status
    print("\n" + "=" * 70)
    if results.missing_warnings or results.import_violations:
        print("STATUS: ISSUES FOUND")
        print(
            "\nRun with --fix to add missing deprecation warnings."
            if results.missing_warnings
            else ""
        )
    else:
        print("STATUS: OK - No deprecation issues found")
    print("=" * 70)


def generate_deprecation_warning(
    module_name: str, replacement: str, removal_date: str
) -> str:
    """Generate a deprecation warning code snippet."""
    return f'''import warnings

warnings.warn(
    "{module_name} is deprecated and will be removed in {removal_date}. "
    "Use {replacement} instead.",
    DeprecationWarning,
    stacklevel=2,
)
'''


def add_missing_warnings(results: AuditResults, root: Path) -> int:
    """Add deprecation warnings to modules missing them."""
    fixed = 0
    for path in results.missing_warnings:
        # Find the replacement info
        module_path = str(path.relative_to(root)).replace("/", ".")[:-3]
        if module_path in DEPRECATED_MODULES:
            replacement, removal_date = DEPRECATED_MODULES[module_path]

            try:
                content = path.read_text(encoding="utf-8")

                # Check if warnings is already imported
                if "import warnings" not in content:
                    warning_code = generate_deprecation_warning(
                        module_path, replacement, removal_date
                    )
                else:
                    warning_code = f'''
warnings.warn(
    "{module_path} is deprecated and will be removed in {removal_date}. "
    "Use {replacement} instead.",
    DeprecationWarning,
    stacklevel=2,
)
'''

                # Insert after docstring and imports
                lines = content.split("\n")
                insert_pos = 0

                # Skip shebang
                if lines and lines[0].startswith("#!"):
                    insert_pos = 1

                # Skip docstring
                in_docstring = False
                for i, line in enumerate(lines[insert_pos:], insert_pos):
                    stripped = line.strip()
                    if not in_docstring and stripped.startswith('"""'):
                        in_docstring = True
                        if stripped.endswith('"""') and len(stripped) > 3:
                            in_docstring = False
                        insert_pos = i + 1
                    elif in_docstring and stripped.endswith('"""'):
                        in_docstring = False
                        insert_pos = i + 1
                    elif not stripped or stripped.startswith("#"):
                        insert_pos = i + 1
                    else:
                        break

                # Find end of imports
                for i, line in enumerate(lines[insert_pos:], insert_pos):
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")):
                        insert_pos = i + 1
                    elif stripped and not stripped.startswith("#"):
                        break

                # Insert the warning
                lines.insert(insert_pos, warning_code)
                path.write_text("\n".join(lines), encoding="utf-8")
                print(f"Added deprecation warning to: {path}")
                fixed += 1

            except Exception as e:
                print(f"Failed to add warning to {path}: {e}")

    return fixed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit deprecated module usage in the codebase"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any issues found",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Add missing deprecation warnings to deprecated modules",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    args = parser.parse_args()

    # Find the ai-service root
    script_dir = Path(__file__).parent
    root = script_dir.parent
    if not (root / "app").exists():
        print("Error: Must run from ai-service directory", file=sys.stderr)
        return 1

    results = audit_deprecations(root)

    if args.json:
        import json

        output = {
            "deprecated_modules": len(results.deprecated_modules),
            "missing_warnings": [str(p) for p in results.missing_warnings],
            "import_violations": [
                {
                    "file": str(v.file_path),
                    "line": v.line_number,
                    "module": v.imported_module,
                    "replacement": v.replacement,
                }
                for v in results.import_violations
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(results)

    if args.fix:
        fixed = add_missing_warnings(results, root)
        print(f"\nFixed {fixed} modules with missing warnings.")

    if args.strict and (results.missing_warnings or results.import_violations):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
