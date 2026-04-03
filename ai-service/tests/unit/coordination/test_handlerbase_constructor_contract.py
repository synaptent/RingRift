from __future__ import annotations

import ast
from pathlib import Path


AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
COORDINATION_ROOT = AI_SERVICE_ROOT / "app" / "coordination"
HANDLER_BASE_NAMES = {"HandlerBase", "BaseEventHandler"}


def _handlerbase_config_assignment_offenders() -> list[str]:
    offenders: list[str] = []

    for path in sorted(COORDINATION_ROOT.rglob("*.py")):
        if "deprecated" in path.parts:
            continue

        source = path.read_text(errors="ignore")
        try:
            module = ast.parse(source)
        except SyntaxError as exc:
            offenders.append(f"{path.relative_to(AI_SERVICE_ROOT)}: syntax error: {exc}")
            continue

        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue

            base_names: set[str] = set()
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.add(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.add(base.attr)

            if not (base_names & HANDLER_BASE_NAMES):
                continue

            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name != "__init__":
                    continue

                for stmt in fn.body:
                    if not isinstance(stmt, ast.Assign):
                        continue
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and target.attr == "config"
                        ):
                            rel = path.relative_to(AI_SERVICE_ROOT)
                            offenders.append(f"{rel}:{node.name}")
                            break

    return offenders


def test_handlerbase_subclasses_do_not_assign_self_config_directly() -> None:
    offenders = _handlerbase_config_assignment_offenders()

    assert offenders == [], (
        "HandlerBase/BaseEventHandler subclasses should resolve config once and pass "
        "it through super().__init__(..., config=resolved_config) instead of assigning "
        f"self.config directly in __init__: {offenders}"
    )
