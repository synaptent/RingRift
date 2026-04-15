"""Contracts for broad exception handling and type-safety drift."""

from __future__ import annotations

import ast
import re
from pathlib import Path


AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = AI_SERVICE_ROOT / "app"
RULES_ROOT = APP_ROOT / "rules"
TYPE_IGNORE_PATTERN = re.compile(r"# type: ignore(?:\[[^]]+\])?")
TYPE_IGNORE_BASELINE = 199
TYPE_IGNORE_UPPER_BOUND = TYPE_IGNORE_BASELINE + 5


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _bare_except_locations(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]


def _type_ignore_count(root: Path) -> int:
    total = 0
    for path in _iter_python_files(root):
        total += len(TYPE_IGNORE_PATTERN.findall(path.read_text()))
    return total


def _rules_any_signature_locations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        annotations = [
            *(arg.annotation for arg in node.args.posonlyargs),
            *(arg.annotation for arg in node.args.args),
            *(arg.annotation for arg in node.args.kwonlyargs),
            node.args.vararg.annotation if node.args.vararg else None,
            node.args.kwarg.annotation if node.args.kwarg else None,
            node.returns,
        ]
        for annotation in annotations:
            if annotation is None:
                continue
            rendered = ast.unparse(annotation)
            if "Any" in rendered:
                offenders.append(f"{path.relative_to(AI_SERVICE_ROOT)}:{node.lineno}:{node.name}:{rendered}")
                break
    return offenders


def test_app_has_no_bare_except_clauses() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(APP_ROOT):
        offenders.extend(
            f"{path.relative_to(AI_SERVICE_ROOT)}:{lineno}"
            for lineno in _bare_except_locations(path)
        )
    assert not offenders, f"Found bare except clauses in app/: {offenders}"


def test_app_type_ignore_count_does_not_drift_up() -> None:
    total = _type_ignore_count(APP_ROOT)
    assert total <= TYPE_IGNORE_UPPER_BOUND, (
        f"app/ type-ignore count drifted up to {total}; upper bound is {TYPE_IGNORE_UPPER_BOUND}"
    )


def test_rules_function_signatures_do_not_use_any() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(RULES_ROOT):
        offenders.extend(_rules_any_signature_locations(path))
    assert not offenders, f"Found Any in app/rules function signatures: {offenders}"
