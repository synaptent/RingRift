"""Fail-closed contracts for Python dependency audit exceptions."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "ai-service" / "scripts" / "check_python_dependency_audit.py"
TODAY = date(2026, 7, 17)


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("check_python_dependency_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _exception(**updates: Any) -> dict[str, Any]:
    entry = {
        "advisory_id": "PYSEC-2026-1325",
        "package": "ecdsa",
        "rationale": "No fixed upstream release; replacement or isolation is tracked.",
        "tracking_issue": "https://github.com/synaptent/RingRift/issues/113",
        "approved_on": "2026-07-17",
        "expires_on": "2026-08-31",
    }
    entry.update(updates)
    return entry


def _ledger(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "exceptions": list(entries)}


def _report(
    *,
    advisory_id: str | None = None,
    aliases: list[str] | None = None,
    fixes: list[str] | None = None,
) -> dict[str, Any]:
    vulns = []
    if advisory_id:
        vulns.append(
            {
                "id": advisory_id,
                "aliases": aliases or [],
                "fix_versions": fixes or [],
                "description": "Fixture advisory.",
            }
        )
    return {
        "dependencies": [{"name": "ecdsa", "version": "0.19.2", "vulns": vulns}],
        "fixes": [],
    }


def _validated(ledger: dict[str, Any], report: dict[str, Any]) -> tuple[list[Any], list[Any], list[str]]:
    exceptions, ledger_errors = CHECKER._parse_exception_ledger(ledger, today=TODAY)
    findings, report_errors = CHECKER._parse_audit_report(report)
    _, evaluation_errors = CHECKER._evaluate_findings(findings, exceptions)
    return exceptions, findings, ledger_errors + report_errors + evaluation_errors


def test_clean_audit_with_empty_ledger_passes() -> None:
    _, findings, errors = _validated(_ledger(), _report())

    assert findings == []
    assert errors == []


def test_unknown_finding_fails() -> None:
    _, _, errors = _validated(_ledger(), _report(advisory_id="PYSEC-2026-1325"))

    assert any("unapproved finding" in error for error in errors)


def test_valid_unfixable_exception_passes() -> None:
    _, findings, errors = _validated(_ledger(_exception()), _report(advisory_id="PYSEC-2026-1325"))

    assert len(findings) == 1
    assert errors == []


def test_exception_can_match_an_advisory_alias() -> None:
    _, _, errors = _validated(
        _ledger(_exception()),
        _report(advisory_id="GHSA-fixture", aliases=["PYSEC-2026-1325"]),
    )

    assert errors == []


def test_exactly_45_day_exception_passes() -> None:
    exceptions, errors = CHECKER._parse_exception_ledger(_ledger(_exception()), today=TODAY)

    assert len(exceptions) == 1
    assert errors == []


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda data: data.update(schema_version=2), "schema_version must be 1"),
        (lambda data: data.update(extra=True), "must contain exactly"),
        (lambda data: data["exceptions"][0].pop("rationale"), "must contain exactly"),
        (lambda data: data["exceptions"][0].update(extra=True), "must contain exactly"),
        (lambda data: data["exceptions"][0].update(rationale=""), "blank or non-string"),
        (
            lambda data: data["exceptions"][0].update(tracking_issue="https://example.com/109"),
            "RingRift GitHub issue URL",
        ),
        (lambda data: data["exceptions"][0].update(approved_on="2026-07-18"), "in the future"),
        (lambda data: data["exceptions"][0].update(expires_on="2026-07-16"), "expires before"),
        (lambda data: data["exceptions"][0].update(expires_on="2026-09-01"), "45-day maximum"),
        (
            lambda data: data["exceptions"].append(copy.deepcopy(data["exceptions"][0])),
            "duplicate exception",
        ),
    ],
)
def test_invalid_ledger_entry_fails(mutation: Any, expected_error: str) -> None:
    ledger = _ledger(_exception())
    mutation(ledger)

    _, errors = CHECKER._parse_exception_ledger(ledger, today=TODAY)

    assert any(expected_error in error for error in errors), errors


def test_expired_exception_fails() -> None:
    _, errors = CHECKER._parse_exception_ledger(
        _ledger(_exception(approved_on="2026-05-01", expires_on="2026-06-15")), today=TODAY
    )

    assert any("expired on" in error for error in errors)


def test_malformed_ledger_json_fails(tmp_path: Path) -> None:
    ledger_path = tmp_path / "exceptions.json"
    ledger_path.write_text("{not-json", encoding="utf-8")

    _, errors = CHECKER._read_json(ledger_path, "exception ledger")

    assert any("invalid JSON" in error for error in errors)


def test_stale_unused_exception_fails() -> None:
    _, _, errors = _validated(_ledger(_exception()), _report())

    assert any("stale or unused exception" in error for error in errors)


def test_fixable_advisory_cannot_be_excepted() -> None:
    _, _, errors = _validated(
        _ledger(_exception()),
        _report(advisory_id="PYSEC-2026-1325", fixes=["0.20.0"]),
    )

    assert any("fixable finding cannot be excepted" in error for error in errors)


def test_duplicate_audit_findings_are_deduplicated() -> None:
    report = _report(advisory_id="PYSEC-2026-1325", aliases=["CVE-2026-1"])
    report["dependencies"][0]["vulns"].append(
        {
            "id": "CVE-2026-1",
            "aliases": ["PYSEC-2026-1325"],
            "fix_versions": [],
        }
    )

    findings, errors = CHECKER._parse_audit_report(report)

    assert errors == []
    assert len(findings) == 1


def test_alias_equivalent_ledger_entries_are_rejected() -> None:
    _, _, errors = _validated(
        _ledger(_exception(), _exception(advisory_id="CVE-2026-1")),
        _report(
            advisory_id="GHSA-fixture",
            aliases=["PYSEC-2026-1325", "CVE-2026-1"],
        ),
    )

    assert any("alias-equivalent duplicate exceptions" in error for error in errors)


@pytest.mark.parametrize(
    ("stdout", "returncode", "expected_error"),
    [
        ("", 2, "failed with exit 2"),
        ("", 1, "returned empty JSON"),
        ("not-json", 1, "returned invalid JSON"),
    ],
)
def test_pip_audit_process_failures_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
    expected_error: str,
) -> None:
    monkeypatch.setattr(
        CHECKER.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], returncode, stdout, "fixture stderr")),
    )

    _, errors = CHECKER._run_pip_audit(Path("requirements.txt"))

    assert any(expected_error in error for error in errors)


def test_missing_python_executable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CHECKER.subprocess, "run", Mock(side_effect=FileNotFoundError("missing")))

    _, errors = CHECKER._run_pip_audit(Path("requirements.txt"))

    assert any("could not start pip-audit" in error for error in errors)


def test_pip_audit_command_requests_strict_json(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps(_report()), ""))
    monkeypatch.setattr(CHECKER.subprocess, "run", run)

    report, errors = CHECKER._run_pip_audit(Path("requirements.txt"))

    assert errors == []
    assert report == _report()
    command = run.call_args.args[0]
    assert command == [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        "requirements.txt",
        "--format",
        "json",
        "--progress-spinner",
        "off",
        "--strict",
    ]


def test_audited_runtime_pins_and_ci_entrypoint_are_aligned() -> None:
    requirements = (ROOT / "ai-service" / "requirements.txt").read_text(encoding="utf-8")
    main_dockerfile = (ROOT / "ai-service" / "Dockerfile").read_text(encoding="utf-8")
    inference_dockerfile = (ROOT / "ai-service" / "docker" / "Dockerfile.inference").read_text(encoding="utf-8")
    worker_dockerfile = (ROOT / "ai-service" / "docker" / "Dockerfile.cmaes-worker").read_text(encoding="utf-8")
    ci_workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    for pin in (
        "fastapi==0.139.2",
        "starlette==1.3.1",
        "aiohttp==3.14.1",
        "torch==2.13.0",
        "torchvision==0.28.0",
        "msgpack==1.2.1",
    ):
        assert pin in requirements
    assert "torch==2.13.0" in main_dockerfile
    assert "torch-scatter" not in main_dockerfile
    assert "torch-sparse" not in main_dockerfile
    for pin in ("fastapi==0.139.2", "starlette==1.3.1", "torch==2.13.0"):
        assert pin in inference_dockerfile
    worker_requirement_filter = next(
        line for line in worker_dockerfile.splitlines() if "grep -E" in line
    )
    assert "fastapi" not in worker_requirement_filter
    assert 'fastapi==0.122.0' in worker_dockerfile
    assert "starlette==1.3.1" not in worker_dockerfile
    assert "python scripts/check_python_dependency_audit.py" in ci_workflow
    assert "pip-audit -r requirements.txt" not in ci_workflow
