#!/usr/bin/env python3
"""Verify all DaemonType references in the codebase resolve to actual enum members.

Prevents the class of bug where removing a DaemonType enum member crashes the
cluster because 23+ files still reference it (as happened Mar 23, 2026).

Usage:
    python scripts/check_daemon_reference_integrity.py
    # Exit code 0 = all references valid, 1 = broken references found

Runs in CI alongside check_layer_violations.py (<2s runtime).
"""
import ast
import re
import sys
from pathlib import Path


def get_daemon_type_members() -> set[str]:
    """Parse DaemonType enum members from source (no import needed)."""
    source = Path("app/coordination/daemon_types.py").read_text()
    tree = ast.parse(source)
    members = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DaemonType":
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            members.add(target.id)
    return members


def find_daemon_type_references(root: Path) -> list[tuple[Path, int, str]]:
    """Find all DaemonType.<NAME> references in Python files."""
    pattern = re.compile(r"DaemonType\.([A-Z][A-Z0-9_]+)")
    refs = []
    for py_file in root.rglob("*.py"):
        # Skip __pycache__, .venv, venv, archive
        parts = py_file.parts
        if any(p in ("__pycache__", ".venv", "venv", "archive", ".git") for p in parts):
            continue
        try:
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                stripped = line.lstrip()
                # Skip comments and docstring examples
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith(">>>"):
                    continue
                # Skip lines that are clearly example/docstring content
                if "MY_DAEMON" in line or "example" in line.lower():
                    continue
                for match in pattern.finditer(line):
                    refs.append((py_file, i, match.group(1)))
        except (UnicodeDecodeError, OSError):
            pass
    return refs


def main() -> int:
    root = Path(".")
    if not Path("app/coordination/daemon_types.py").exists():
        root = Path("ai-service")
        if not (root / "app/coordination/daemon_types.py").exists():
            print("ERROR: Cannot find daemon_types.py. Run from ai-service/ or repo root.")
            return 1

    members = get_daemon_type_members()
    print(f"DaemonType has {len(members)} members")

    refs = find_daemon_type_references(root)
    print(f"Found {len(refs)} DaemonType references across codebase")

    broken = []
    for path, line, name in refs:
        if name not in members:
            broken.append((path, line, name))

    if broken:
        print(f"\nERROR: {len(broken)} broken DaemonType references:\n")
        for path, line, name in sorted(broken):
            print(f"  {path}:{line}: DaemonType.{name} does not exist")
        print(f"\nThese references will cause ImportError/AttributeError at runtime.")
        print("Fix: Either add the member back to DaemonType or remove the reference.")
        return 1
    else:
        print("OK: All DaemonType references resolve to valid enum members")

    # Bonus: check DAEMON_REGISTRY keys match DaemonType members
    registry_source = (root / "app/coordination/daemon_registry.py").read_text()
    registry_refs = set(re.findall(r"DaemonType\.([A-Z_]+)", registry_source))
    orphan_registry = registry_refs - members - {"S", "P"}  # Single-letter false positives from line wraps
    if orphan_registry:
        print(f"\nWARN: daemon_registry.py references non-existent types: {orphan_registry}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
