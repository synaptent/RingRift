"""Contracts for complete, machine-readable GitHub workflow policy."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = ROOT / "scripts" / "check_github_workflows.py"
REGISTRY_PATH = ROOT / "docs" / "data" / "workflow_policy_registry.json"


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("check_github_workflows", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _discovered_paths() -> set[str]:
    workflow_dir = ROOT / ".github" / "workflows"
    return {path.relative_to(ROOT).as_posix() for pattern in ("*.yml", "*.yaml") for path in workflow_dir.glob(pattern)}


def test_checked_in_workflow_policy_passes_checker() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "required=4" in result.stdout
    assert "scheduled=5" in result.stdout
    assert "informational=1" in result.stdout


def test_registry_exactly_classifies_discovered_workflows() -> None:
    errors: list[str] = []

    classifications = CHECKER._validate_policy_registry(_registry(), _discovered_paths(), errors)

    assert errors == []
    assert set(classifications) == _discovered_paths()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda data: data.update(schema_version=2), "schema_version must be 1"),
        (lambda data: data["workflows"].pop(), "workflow is unclassified"),
        (
            lambda data: data["workflows"].append(
                {
                    "path": ".github/workflows/stale.yml",
                    "classification": "required",
                    "rationale": "Stale test entry.",
                }
            ),
            "registry entry does not match a workflow file",
        ),
        (lambda data: data["workflows"].append(copy.deepcopy(data["workflows"][0])), "duplicate path"),
        (lambda data: data["workflows"][0].update(classification="optional"), "invalid workflow classification"),
        (lambda data: data["workflows"][0].update(rationale=""), "rationale must be non-empty"),
        (lambda data: data["workflows"][0].update(path="../ci.yml"), "invalid workflow path"),
    ],
)
def test_invalid_registry_is_rejected(mutation: Any, expected_error: str) -> None:
    registry = _registry()
    mutation(registry)
    errors: list[str] = []

    CHECKER._validate_policy_registry(registry, _discovered_paths(), errors)

    assert any(expected_error in error for error in errors), errors


@pytest.mark.parametrize(
    ("classification", "triggers", "expected_error"),
    [
        ("required", "workflow_dispatch: {}", "required workflow needs"),
        ("scheduled", "workflow_dispatch: {}", "scheduled workflow needs"),
        ("informational", "push:\n    branches: [main]", "informational workflow cannot"),
    ],
)
def test_trigger_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classification: str,
    triggers: str,
    expected_error: str,
) -> None:
    workflow = tmp_path / ".github" / "workflows" / "test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(f"name: Test\n\non:\n  {triggers}\n\njobs: {{}}\n", encoding="utf-8")
    monkeypatch.setattr(CHECKER, "ROOT", tmp_path)
    errors: list[str] = []

    CHECKER._check_policy_trigger_compatibility([workflow], {".github/workflows/test.yml": classification}, errors)

    assert any(expected_error in error for error in errors), errors
