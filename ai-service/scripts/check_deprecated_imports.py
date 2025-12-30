#!/usr/bin/env python3
"""Deprecation enforcement script for RingRift AI Service.

Scans the codebase for imports of deprecated modules and reports violations.
Can be used as a pre-commit hook or CI check.

Usage:
    python scripts/check_deprecated_imports.py              # Scan all files
    python scripts/check_deprecated_imports.py --strict     # Exit with error on violations
    python scripts/check_deprecated_imports.py --json       # Output JSON report
    python scripts/check_deprecated_imports.py app/         # Scan specific directory

December 30, 2025: Created as part of technical debt management.
Deprecated modules are scheduled for removal in Q2 2026.
"""

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# =============================================================================
# Deprecated Module Registry
# =============================================================================

# Modules scheduled for removal in Q2 2026
# Format: module_path -> (replacement, removal_date, description)
DEPRECATED_MODULES: dict[str, tuple[str, str, str]] = {
    # Sync modules
    "app.coordination.cluster_data_sync": (
        "app.coordination.auto_sync_daemon.AutoSyncDaemon",
        "Q2 2026",
        "Use AutoSyncDaemon(strategy='broadcast')",
    ),
    "app.coordination.ephemeral_sync": (
        "app.coordination.auto_sync_daemon.AutoSyncDaemon",
        "Q2 2026",
        "Use AutoSyncDaemon(strategy='ephemeral')",
    ),
    "app.coordination.sync_coordinator": (
        "app.coordination.auto_sync_daemon.AutoSyncDaemon",
        "Q2 2026",
        "Use AutoSyncDaemon for sync operations",
    ),
    # Health modules
    "app.coordination.system_health_monitor": (
        "app.coordination.unified_health_manager",
        "Q2 2026",
        "Use unified_health_manager.get_system_health_score()",
    ),
    "app.coordination.node_health_monitor": (
        "app.coordination.health_check_orchestrator",
        "Q2 2026",
        "Use health_check_orchestrator.HealthCheckOrchestrator",
    ),
    # Distribution modules
    "app.coordination.model_distribution_daemon": (
        "app.coordination.unified_distribution_daemon",
        "Q2 2026",
        "Use unified_distribution_daemon with DataType.MODEL",
    ),
    "app.coordination.npz_distribution_daemon": (
        "app.coordination.unified_distribution_daemon",
        "Q2 2026",
        "Use unified_distribution_daemon with DataType.NPZ",
    ),
    # Replication modules
    "app.coordination.replication_monitor": (
        "app.coordination.unified_replication_daemon",
        "Q2 2026",
        "Use unified_replication_daemon",
    ),
    "app.coordination.replication_repair_daemon": (
        "app.coordination.unified_replication_daemon",
        "Q2 2026",
        "Use unified_replication_daemon",
    ),
    # Idle shutdown modules
    "app.coordination.lambda_idle_daemon": (
        "app.coordination.unified_idle_shutdown_daemon",
        "Q2 2026",
        "Lambda GH200 nodes are dedicated training infrastructure",
    ),
    "app.coordination.vast_idle_daemon": (
        "app.coordination.unified_idle_shutdown_daemon",
        "Q2 2026",
        "Use unified_idle_shutdown_daemon.create_vast_idle_daemon()",
    ),
    # Queue populator
    "app.coordination.queue_populator_daemon": (
        "app.coordination.unified_queue_populator",
        "Q2 2026",
        "Use unified_queue_populator.UnifiedQueuePopulator",
    ),
    # Event modules (re-export wrappers)
    "app.coordination.event_emitters": (
        "app.coordination.event_router",
        "Q2 2026",
        "Use event_router.get_event_bus() and DataEvent directly",
    ),
    "app.coordination.stage_events": (
        "app.coordination.event_router",
        "Q2 2026",
        "Use event_router.get_router() instead",
    ),
    "app.coordination.data_events": (
        "app.coordination.event_router",
        "Q2 2026",
        "Use event_router.get_router() instead",
    ),
    # Core singleton (prefer coordination version)
    "app.core.singleton_mixin": (
        "app.coordination.singleton_mixin",
        "Q2 2026",
        "Use app.coordination.singleton_mixin for all singleton patterns",
    ),
}

# Specific class/function imports that are deprecated
# Format: (module_pattern, import_name) -> (replacement, removal_date, description)
# Use None for module_pattern to match any module
DEPRECATED_IMPORTS: dict[tuple[Optional[str], str], tuple[str, str, str]] = {
    # From queue_populator
    (None, "PopulatorConfig"): (
        "QueuePopulatorConfig",
        "Q2 2026",
        "Renamed to QueuePopulatorConfig",
    ),
    ("app.coordination.queue_populator", "QueuePopulator"): (
        "UnifiedQueuePopulator",
        "Q2 2026",
        "Use UnifiedQueuePopulator from unified_queue_populator",
    ),
    # From distribution
    (None, "ModelDistributionDaemon"): (
        "UnifiedDistributionDaemon",
        "Q2 2026",
        "Use UnifiedDistributionDaemon with DataType.MODEL",
    ),
    (None, "NPZDistributionDaemon"): (
        "UnifiedDistributionDaemon",
        "Q2 2026",
        "Use UnifiedDistributionDaemon with DataType.NPZ",
    ),
    # From deprecated event modules (NOT from event_router itself)
    ("app.coordination.stage_events", "get_event_bus"): (
        "get_router from event_router",
        "Q2 2026",
        "Use get_router() from event_router",
    ),
    ("app.coordination.data_events", "get_event_bus"): (
        "get_router from event_router",
        "Q2 2026",
        "Use get_router() from event_router",
    ),
    ("app.distributed.data_events", "get_event_bus"): (
        "get_router from event_router",
        "Q2 2026",
        "Use get_router() from event_router",
    ),
    (None, "get_stage_event_bus"): (
        "get_router from event_router",
        "Q2 2026",
        "Use get_router() from event_router",
    ),
    (None, "get_data_event_bus"): (
        "get_router from event_router",
        "Q2 2026",
        "Use get_router() from event_router",
    ),
}

# Files to exclude from scanning (test files, archived code, etc.)
EXCLUDE_PATTERNS: list[str] = [
    "archive/",
    "test_",
    "_test.py",
    "conftest.py",
    "__pycache__",
    ".git",
    "venv/",
    ".venv/",
]


@dataclass
class Violation:
    """A single deprecation violation."""
    file_path: str
    line_number: int
    import_name: str
    replacement: str
    removal_date: str
    description: str
    import_type: str = "module"  # "module" or "name"


@dataclass
class ScanResult:
    """Result of scanning a file or directory."""
    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)


def should_skip_file(path: Path) -> bool:
    """Check if a file should be skipped during scanning."""
    path_str = str(path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def scan_file(file_path: Path) -> ScanResult:
    """Scan a Python file for deprecated imports."""
    result = ScanResult()

    if should_skip_file(file_path):
        return result

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        result.errors.append(f"Syntax error in {file_path}: {e}")
        return result
    except UnicodeDecodeError as e:
        result.errors.append(f"Encoding error in {file_path}: {e}")
        return result

    result.files_scanned = 1

    for node in ast.walk(tree):
        # Check import statements: import foo.bar
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name in DEPRECATED_MODULES:
                    replacement, removal_date, desc = DEPRECATED_MODULES[module_name]
                    result.violations.append(Violation(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        import_name=module_name,
                        replacement=replacement,
                        removal_date=removal_date,
                        description=desc,
                        import_type="module",
                    ))

        # Check from imports: from foo.bar import baz
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Check if the module itself is deprecated
                if node.module in DEPRECATED_MODULES:
                    replacement, removal_date, desc = DEPRECATED_MODULES[node.module]
                    result.violations.append(Violation(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        import_name=node.module,
                        replacement=replacement,
                        removal_date=removal_date,
                        description=desc,
                        import_type="module",
                    ))

                # Check for deprecated specific imports
                for alias in node.names:
                    import_name = alias.name
                    # Check with specific module pattern
                    key_with_module = (node.module, import_name)
                    key_any_module = (None, import_name)

                    if key_with_module in DEPRECATED_IMPORTS:
                        replacement, removal_date, desc = DEPRECATED_IMPORTS[key_with_module]
                        result.violations.append(Violation(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            import_name=f"{node.module}.{import_name}",
                            replacement=replacement,
                            removal_date=removal_date,
                            description=desc,
                            import_type="name",
                        ))
                    elif key_any_module in DEPRECATED_IMPORTS:
                        replacement, removal_date, desc = DEPRECATED_IMPORTS[key_any_module]
                        result.violations.append(Violation(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            import_name=f"{node.module}.{import_name}",
                            replacement=replacement,
                            removal_date=removal_date,
                            description=desc,
                            import_type="name",
                        ))

    return result


def scan_directory(directory: Path, recursive: bool = True) -> ScanResult:
    """Scan a directory for deprecated imports."""
    result = ScanResult()

    if recursive:
        py_files = directory.rglob("*.py")
    else:
        py_files = directory.glob("*.py")

    for py_file in py_files:
        file_result = scan_file(py_file)
        result.violations.extend(file_result.violations)
        result.files_scanned += file_result.files_scanned
        result.errors.extend(file_result.errors)

    return result


def format_violation(v: Violation) -> str:
    """Format a violation for console output."""
    return (
        f"  {v.file_path}:{v.line_number}\n"
        f"    Import: {v.import_name}\n"
        f"    Replace with: {v.replacement}\n"
        f"    Removal: {v.removal_date}\n"
        f"    {v.description}\n"
    )


def format_json(result: ScanResult) -> str:
    """Format result as JSON."""
    return json.dumps({
        "files_scanned": result.files_scanned,
        "violation_count": len(result.violations),
        "error_count": len(result.errors),
        "violations": [
            {
                "file": v.file_path,
                "line": v.line_number,
                "import": v.import_name,
                "replacement": v.replacement,
                "removal_date": v.removal_date,
                "description": v.description,
                "type": v.import_type,
            }
            for v in result.violations
        ],
        "errors": result.errors,
    }, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Check for deprecated imports in RingRift AI Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Scan app/ and scripts/
  %(prog)s --strict            # Exit with code 1 if violations found
  %(prog)s --json              # Output JSON report
  %(prog)s app/coordination/   # Scan specific directory
  %(prog)s --list              # List all deprecated modules
        """,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["app/", "scripts/"],
        help="Paths to scan (default: app/ scripts/)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if violations found",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all deprecated modules and exit",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't scan subdirectories",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Deprecated Modules (Q2 2026 removal):")
        print("=" * 60)
        for module, (replacement, date, desc) in sorted(DEPRECATED_MODULES.items()):
            print(f"\n{module}")
            print(f"  Replace with: {replacement}")
            print(f"  {desc}")
        print("\n\nDeprecated Imports:")
        print("=" * 60)
        for (module_pattern, name), (replacement, date, desc) in sorted(
            DEPRECATED_IMPORTS.items(), key=lambda x: (x[0][1], x[0][0] or "")
        ):
            if module_pattern:
                print(f"\nfrom {module_pattern} import {name}")
            else:
                print(f"\n{name} (from any module)")
            print(f"  Replace with: {replacement}")
            print(f"  {desc}")
        return 0

    # Scan mode
    result = ScanResult()

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            file_result = scan_file(path)
        elif path.is_dir():
            file_result = scan_directory(path, recursive=not args.no_recursive)
        else:
            print(f"Warning: Path not found: {path_str}", file=sys.stderr)
            continue

        result.violations.extend(file_result.violations)
        result.files_scanned += file_result.files_scanned
        result.errors.extend(file_result.errors)

    # Output
    if args.json:
        print(format_json(result))
    else:
        print(f"Scanned {result.files_scanned} files")
        print()

        if result.violations:
            print(f"Found {len(result.violations)} deprecated import(s):")
            print()
            for v in result.violations:
                print(format_violation(v))
        else:
            print("No deprecated imports found.")

        if result.errors:
            print(f"\n{len(result.errors)} error(s):")
            for err in result.errors:
                print(f"  {err}")

    # Exit code
    if args.strict and result.violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
