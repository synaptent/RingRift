"""Meta-tests for Python test infrastructure hygiene."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = AI_SERVICE_ROOT / "tests"
ACTIVE_TEST_EXCLUDES = {
    ("tests", "archive"),
    ("tests", "unit", "distributed", "pending"),
}
COMMON_SUFFIXES = (
    "_daemon",
    "_mixin",
    "_manager",
    "_coordinator",
    "_orchestrator",
    "_handler",
    "_controller",
    "_service",
    "_utils",
    "_types",
    "_config",
    "_base",
    "_executor",
    "_registry",
    "_monitor",
    "_watchdog",
)
TOKEN_STOPWORDS = {
    "test",
    "unit",
    "integration",
    "contracts",
    "coordination",
    "training",
    "app",
    "ai",
}
EXPLICIT_TEST_OWNERSHIP = {
    "auxiliary_tasks": ["tests/unit/training/test_training_module_smoke.py"],
    "checkpointing": ["tests/unit/training/test_training_module_smoke.py"],
    "distillation": ["tests/unit/training/test_training_module_smoke.py"],
    "lr_finder": ["tests/unit/training/test_training_module_smoke.py"],
    "opening_book": ["tests/unit/training/test_training_module_smoke.py"],
    "pbt": ["tests/unit/training/test_training_module_smoke.py"],
    "thread_integration": ["tests/unit/training/test_training_module_smoke.py"],
}


def _is_active_test_file(path: Path) -> bool:
    rel = path.relative_to(AI_SERVICE_ROOT).parts
    return not any(rel[: len(prefix)] == prefix for prefix in ACTIVE_TEST_EXCLUDES)


def _iter_active_test_files() -> list[Path]:
    return [path for path in sorted(TESTS_ROOT.rglob("test_*.py")) if _is_active_test_file(path)]


def _iter_test_functions(tree: ast.AST) -> list[ast.AST]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]


def _imported_modules(tree: ast.AST, prefix: str) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names if alias.name.startswith(prefix))
    return modules


def _module_exists(module_name: str) -> bool:
    module_path = AI_SERVICE_ROOT / (module_name.replace(".", "/") + ".py")
    package_init = AI_SERVICE_ROOT / module_name.replace(".", "/") / "__init__.py"
    return module_path.exists() or package_init.exists()


def _candidate_tokens(stem: str) -> set[str]:
    candidates = {stem}
    for suffix in COMMON_SUFFIXES:
        if stem.endswith(suffix):
            candidates.add(stem[: -len(suffix)])
    candidates.update(part for part in re.split(r"[_\-]+", stem) if part)
    return {token for token in candidates if token not in TOKEN_STOPWORDS and len(token) >= 3}


ACTIVE_TEST_FILES = _iter_active_test_files()
ACTIVE_TEST_STRINGS = [str(path.relative_to(AI_SERVICE_ROOT)) for path in ACTIVE_TEST_FILES]
TOP_LEVEL_COORDINATION_MODULES = sorted(
    path for path in (AI_SERVICE_ROOT / "app" / "coordination").glob("*.py") if path.name != "__init__.py"
)
TOP_LEVEL_TRAINING_MODULES = sorted(
    path for path in (AI_SERVICE_ROOT / "app" / "training").glob("*.py") if path.name != "__init__.py"
)


def _has_corresponding_test(module_path: Path) -> bool:
    stem = module_path.stem
    explicit = EXPLICIT_TEST_OWNERSHIP.get(stem)
    if explicit:
      return all((AI_SERVICE_ROOT / rel_path).exists() for rel_path in explicit)

    candidate_tokens = _candidate_tokens(stem)
    for test_path, test_rel in zip(ACTIVE_TEST_FILES, ACTIVE_TEST_STRINGS):
        test_stem = test_path.stem
        if stem in test_stem or stem in test_rel:
            return True
        if any(token in test_stem or token in test_rel for token in candidate_tokens):
            return True
    return False


@pytest.mark.parametrize(
    "test_file",
    ACTIVE_TEST_FILES,
    ids=lambda path: str(path.relative_to(AI_SERVICE_ROOT)),
)
def test_active_test_files_define_at_least_one_test(test_file: Path) -> None:
    tree = ast.parse(test_file.read_text())
    assert _iter_test_functions(tree), (
        f"{test_file.relative_to(AI_SERVICE_ROOT)} is named like a test file but defines no "
        "discoverable test functions or methods"
    )


@pytest.mark.parametrize(
    "test_file",
    ACTIVE_TEST_FILES,
    ids=lambda path: str(path.relative_to(AI_SERVICE_ROOT)),
)
def test_active_test_files_do_not_import_archive_directly(test_file: Path) -> None:
    tree = ast.parse(test_file.read_text())
    archive_imports = _imported_modules(tree, "archive")
    assert not archive_imports, (
        f"{test_file.relative_to(AI_SERVICE_ROOT)} imports archive modules directly: {archive_imports}"
    )


@pytest.mark.parametrize(
    "test_file",
    ACTIVE_TEST_FILES,
    ids=lambda path: str(path.relative_to(AI_SERVICE_ROOT)),
)
def test_active_test_files_only_import_existing_app_modules(test_file: Path) -> None:
    tree = ast.parse(test_file.read_text())
    broken = sorted(
        {
            module_name
            for module_name in _imported_modules(tree, "app.")
            if not _module_exists(module_name)
        }
    )
    assert not broken, (
        f"{test_file.relative_to(AI_SERVICE_ROOT)} imports missing app modules: {broken}"
    )


@pytest.mark.parametrize(
    "module_path",
    [pytest.param(path, id=path.name) for path in TOP_LEVEL_COORDINATION_MODULES],
)
def test_top_level_coordination_modules_have_corresponding_tests(module_path: Path) -> None:
    assert _has_corresponding_test(module_path), (
        f"{module_path.relative_to(AI_SERVICE_ROOT)} has no corresponding active test file"
    )


@pytest.mark.parametrize(
    "module_path",
    [pytest.param(path, id=path.name) for path in TOP_LEVEL_TRAINING_MODULES],
)
def test_top_level_training_modules_have_corresponding_tests(module_path: Path) -> None:
    assert _has_corresponding_test(module_path), (
        f"{module_path.relative_to(AI_SERVICE_ROOT)} has no corresponding active test file"
    )
