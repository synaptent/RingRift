"""Surface-area dashboard and ratchets for the AI service package tree."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = AI_SERVICE_ROOT / "app"

TOP_LEVEL_EXPORT_BUDGETS = {
    "app.ai": 16,
    "app.analysis": 10,
    "app.caching": 15,
    "app.cli": 12,
    "app.config": 65,
    "app.core": 55,
    "app.db": 10,
    "app.distributed": 145,
    "app.errors": 50,
    "app.evaluation": 25,
    "app.events": 10,
    "app.execution": 25,
    "app.game_engine": 10,
    "app.integration": 35,
    "app.interfaces": 10,
    "app.mcts": 10,
    "app.metrics": 140,
    "app.models": 45,
    "app.monitoring": 35,
    "app.notation": 12,
    "app.observability": 8,
    "app.p2p": 35,
    "app.providers": 16,
    "app.quality": 30,
    "app.routes": 8,
    "app.rules": 10,
    "app.storage": 8,
    "app.sync": 15,
    "app.testing": 20,
    "app.tournament": 60,
    "app.training": 30,
    "app.utils": 30,
    "app.validation": 25,
}

TOP_LEVEL_LINE_BUDGETS = {
    "app.coordination": 100,
}

DIRECT_UNIMPORTED_APP_FILE_BUDGET = 60


@dataclass(frozen=True)
class PackageSurface:
    name: str
    exports: int
    max_exports: int | None
    lines: int
    max_lines: int | None


def _python_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    ]


def _module_name(path: Path) -> str:
    relative = path.relative_to(AI_SERVICE_ROOT).with_suffix("")
    if relative.name == "__init__":
        relative = relative.parent
    return ".".join(relative.parts)


def _package_name(path: Path) -> str:
    module_name = _module_name(path)
    if path.name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0]


def _resolve_import_from(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    package_parts = _package_name(path).split(".")
    prefix = package_parts[: max(0, len(package_parts) - node.level + 1)]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _literal_all_count(init_file: Path) -> int:
    tree = ast.parse(init_file.read_text(encoding="utf-8"), filename=str(init_file))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            return sum(
                1
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return 0


def _top_level_surfaces() -> list[PackageSurface]:
    package_dirs = [path for path in APP_ROOT.iterdir() if path.is_dir() and (path / "__init__.py").exists()]
    surfaces: list[PackageSurface] = []
    for package_dir in sorted(package_dirs):
        name = f"app.{package_dir.name}"
        init_file = package_dir / "__init__.py"
        lines = len(init_file.read_text(encoding="utf-8").splitlines())
        surfaces.append(
            PackageSurface(
                name=name,
                exports=_literal_all_count(init_file),
                max_exports=TOP_LEVEL_EXPORT_BUDGETS.get(name),
                lines=lines,
                max_lines=TOP_LEVEL_LINE_BUDGETS.get(name),
            )
        )
    return surfaces


def _is_module_entrypoint(path: Path) -> bool:
    if path.name in {"__init__.py", "main.py"}:
        return True
    if path.name.endswith("_cli.py"):
        return True

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
            return True
    return False


def _direct_unimported_app_files() -> list[tuple[int, str, str]]:
    all_files = _python_files(AI_SERVICE_ROOT)
    app_files = _python_files(APP_ROOT)
    module_to_file = {_module_name(path): path for path in app_files}
    imported_by: dict[str, set[str]] = defaultdict(set)

    def add_reference(module_name: str, importer: str) -> None:
        if not module_name.startswith("app"):
            return
        parts = module_name.split(".")
        for index in range(1, len(parts) + 1):
            imported_by[".".join(parts[:index])].add(importer)

    for path in all_files:
        importer = _module_name(path) if path.is_relative_to(APP_ROOT) else str(path.relative_to(AI_SERVICE_ROOT))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_reference(alias.name, importer)
            elif isinstance(node, ast.ImportFrom):
                base_module = _resolve_import_from(path, node)
                if not base_module:
                    continue
                add_reference(base_module, importer)
                for alias in node.names:
                    candidate = f"{base_module}.{alias.name}"
                    if candidate in module_to_file:
                        add_reference(candidate, importer)

    unimported: list[tuple[int, str, str]] = []
    for module_name, path in module_to_file.items():
        if _is_module_entrypoint(path) or module_name == "app.main":
            continue
        importers = {importer for importer in imported_by.get(module_name, set()) if importer != module_name}
        if importers:
            continue
        unimported.append((path.stat().st_size, module_name, path.relative_to(AI_SERVICE_ROOT).as_posix()))

    return sorted(unimported, key=lambda item: (-item[0], item[2]))


def test_top_level_package_surface_dashboard_stays_under_budget() -> None:
    for surface in _top_level_surfaces():
        print(
            f"SURFACE {surface.name}: {surface.exports} exports"
            f" (max: {surface.max_exports}), {surface.lines} lines"
            f" (max lines: {surface.max_lines})"
        )
        if surface.max_exports is not None:
            assert surface.exports <= surface.max_exports
        if surface.max_lines is not None:
            assert surface.lines <= surface.max_lines


def test_direct_unimported_app_file_budget_does_not_grow() -> None:
    unimported = _direct_unimported_app_files()

    print(f"DIRECT_UNIMPORTED_APP_FILES {len(unimported)} (max: {DIRECT_UNIMPORTED_APP_FILE_BUDGET})")
    for size, module_name, relative_path in unimported[:20]:
        print(f"UNIMPORTED {size:7d} {module_name:75s} {relative_path}")

    assert len(unimported) <= DIRECT_UNIMPORTED_APP_FILE_BUDGET
