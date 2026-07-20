#!/usr/bin/env python3
"""Run pip-audit with a strict, expiring exception ledger."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS = ROOT / "ai-service" / "requirements.txt"
DEFAULT_EXCEPTIONS = ROOT / "docs" / "security" / "python_audit_exceptions.json"
LEDGER_SCHEMA_VERSION = 1
MAX_EXCEPTION_DAYS = 45
LEDGER_KEYS = {"schema_version", "exceptions"}
EXCEPTION_KEYS = {
    "advisory_id",
    "package",
    "rationale",
    "tracking_issue",
    "approved_on",
    "expires_on",
}
TRACKING_ISSUE_PATTERN = re.compile(r"^https://github\.com/synaptent/RingRift/issues/[1-9][0-9]*$")


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    advisory_ids: frozenset[str]
    fix_versions: tuple[str, ...]


@dataclass(frozen=True)
class AuditException:
    advisory_id: str
    package: str
    rationale: str
    tracking_issue: str
    approved_on: date
    expires_on: date


def _normalize_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _normalize_advisory(advisory_id: str) -> str:
    return advisory_id.strip().upper()


def _read_json(path: Path, label: str) -> tuple[Any, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except FileNotFoundError:
        return None, [f"{label} is missing: {path}"]
    except json.JSONDecodeError as exc:
        return None, [f"{label} has invalid JSON at line {exc.lineno}: {exc.msg}"]


def _parse_exception_ledger(ledger: Any, *, today: date) -> tuple[list[AuditException], list[str]]:
    errors: list[str] = []
    exceptions: list[AuditException] = []
    if not isinstance(ledger, dict):
        return exceptions, ["exception ledger must be a JSON object"]
    if set(ledger) != LEDGER_KEYS:
        errors.append("exception ledger must contain exactly schema_version and exceptions")
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        errors.append(f"exception ledger schema_version must be {LEDGER_SCHEMA_VERSION}")
    entries = ledger.get("exceptions")
    if not isinstance(entries, list):
        errors.append("exception ledger exceptions must be an array")
        return exceptions, errors

    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        prefix = f"exception entry {index}"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if set(entry) != EXCEPTION_KEYS:
            errors.append(f"{prefix} must contain exactly: {', '.join(sorted(EXCEPTION_KEYS))}")
            continue

        string_fields = {key: entry.get(key) for key in ("advisory_id", "package", "rationale", "tracking_issue")}
        invalid_strings = [
            key for key, value in string_fields.items() if not isinstance(value, str) or not value.strip()
        ]
        if invalid_strings:
            errors.append(f"{prefix} has blank or non-string fields: {', '.join(invalid_strings)}")
            continue

        advisory_id = _normalize_advisory(string_fields["advisory_id"])
        package = _normalize_package(string_fields["package"])
        if not advisory_id or not package:
            errors.append(f"{prefix} has an invalid advisory_id or package")
            continue
        tracking_issue = string_fields["tracking_issue"].strip()
        if not TRACKING_ISSUE_PATTERN.fullmatch(tracking_issue):
            errors.append(f"{prefix} tracking_issue must be a RingRift GitHub issue URL")
            continue

        try:
            approved_on = date.fromisoformat(entry["approved_on"])
            expires_on = date.fromisoformat(entry["expires_on"])
        except (TypeError, ValueError):
            errors.append(f"{prefix} approved_on and expires_on must be ISO dates")
            continue

        if approved_on > today:
            errors.append(f"{prefix} approval date is in the future: {approved_on}")
        if expires_on < approved_on:
            errors.append(f"{prefix} expires before its approval date")
        elif (expires_on - approved_on).days > MAX_EXCEPTION_DAYS:
            errors.append(f"{prefix} exceeds the {MAX_EXCEPTION_DAYS}-day maximum")
        if expires_on < today:
            errors.append(f"{prefix} expired on {expires_on}")

        key = (package, advisory_id)
        if key in seen:
            errors.append(f"duplicate exception for {package} {advisory_id}")
        seen.add(key)
        exceptions.append(
            AuditException(
                advisory_id=advisory_id,
                package=package,
                rationale=string_fields["rationale"].strip(),
                tracking_issue=tracking_issue,
                approved_on=approved_on,
                expires_on=expires_on,
            )
        )
    return exceptions, errors


def _merge_finding(findings: list[Finding], candidate: Finding) -> None:
    for index, existing in enumerate(findings):
        if existing.package == candidate.package and existing.advisory_ids.intersection(candidate.advisory_ids):
            findings[index] = Finding(
                package=existing.package,
                version=existing.version,
                advisory_ids=existing.advisory_ids.union(candidate.advisory_ids),
                fix_versions=tuple(sorted(set(existing.fix_versions).union(candidate.fix_versions))),
            )
            return
    findings.append(candidate)


def _parse_audit_report(report: Any) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    errors: list[str] = []
    if not isinstance(report, dict):
        return findings, ["pip-audit JSON must be an object"]
    if set(report) != {"dependencies", "fixes"}:
        errors.append("pip-audit JSON must contain exactly dependencies and fixes")
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list):
        errors.append("pip-audit dependencies must be an array")
        return findings, errors
    if not isinstance(report.get("fixes"), list):
        errors.append("pip-audit fixes must be an array")

    for dependency_index, dependency in enumerate(dependencies):
        prefix = f"pip-audit dependency {dependency_index}"
        if not isinstance(dependency, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = dependency.get("name")
        version = dependency.get("version")
        vulns = dependency.get("vulns")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix} needs a non-empty name")
            continue
        if not isinstance(version, str) or not version.strip():
            errors.append(f"{prefix} needs a non-empty version")
            continue
        if not isinstance(vulns, list):
            errors.append(f"{prefix} vulns must be an array")
            continue

        for vuln_index, vuln in enumerate(vulns):
            vuln_prefix = f"{prefix} vulnerability {vuln_index}"
            if not isinstance(vuln, dict):
                errors.append(f"{vuln_prefix} must be an object")
                continue
            advisory_id = vuln.get("id")
            aliases = vuln.get("aliases", [])
            fix_versions = vuln.get("fix_versions")
            if not isinstance(advisory_id, str) or not advisory_id.strip():
                errors.append(f"{vuln_prefix} needs a non-empty id")
                continue
            if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias.strip() for alias in aliases):
                errors.append(f"{vuln_prefix} aliases must be non-empty strings")
                continue
            if not isinstance(fix_versions, list) or not all(
                isinstance(fix, str) and fix.strip() for fix in fix_versions
            ):
                errors.append(f"{vuln_prefix} fix_versions must be strings")
                continue
            ids = frozenset(_normalize_advisory(value) for value in [advisory_id, *aliases] if value.strip())
            _merge_finding(
                findings,
                Finding(
                    package=_normalize_package(name),
                    version=version.strip(),
                    advisory_ids=ids,
                    fix_versions=tuple(sorted(set(fix_versions))),
                ),
            )
    return findings, errors


def _evaluate_findings(findings: list[Finding], exceptions: list[AuditException]) -> tuple[int, list[str]]:
    errors: list[str] = []
    excepted_findings: set[int] = set()
    matched_exception_by_finding: dict[int, AuditException] = {}
    for exception in exceptions:
        matches = [
            index
            for index, finding in enumerate(findings)
            if finding.package == exception.package and exception.advisory_id in finding.advisory_ids
        ]
        if not matches:
            errors.append(f"stale or unused exception: {exception.package} {exception.advisory_id}")
            continue
        for index in matches:
            finding = findings[index]
            previous = matched_exception_by_finding.get(index)
            if previous is not None:
                errors.append(
                    "alias-equivalent duplicate exceptions for "
                    f"{finding.package}: {previous.advisory_id} and {exception.advisory_id}"
                )
                continue
            matched_exception_by_finding[index] = exception
            if finding.fix_versions:
                errors.append(
                    f"fixable finding cannot be excepted: {finding.package} "
                    f"{exception.advisory_id} (fix: {', '.join(finding.fix_versions)})"
                )
                continue
            excepted_findings.add(index)

    for index, finding in enumerate(findings):
        if index not in excepted_findings:
            errors.append(
                f"unapproved finding: {finding.package} {finding.version} ({', '.join(sorted(finding.advisory_ids))})"
            )
    return len(excepted_findings), errors


def _run_pip_audit(requirements: Path) -> tuple[Any, list[str]]:
    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        str(requirements),
        "--format",
        "json",
        "--progress-spinner",
        "off",
        "--strict",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return None, [f"could not start pip-audit: {exc}"]
    if result.returncode not in {0, 1}:
        detail = result.stderr.strip() or "no diagnostic output"
        return None, [f"pip-audit failed with exit {result.returncode}: {detail}"]
    if not result.stdout.strip():
        detail = result.stderr.strip() or "no diagnostic output"
        return None, [f"pip-audit returned empty JSON: {detail}"]
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, [f"pip-audit returned invalid JSON at line {exc.lineno}: {exc.msg}"]
    return report, []


def _print_errors(errors: list[str]) -> None:
    print("Python dependency audit failed:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    args = parser.parse_args(argv)

    ledger, errors = _read_json(args.exceptions, "exception ledger")
    exceptions, ledger_errors = _parse_exception_ledger(ledger, today=date.today())
    errors.extend(ledger_errors)
    if errors:
        _print_errors(errors)
        return 1

    report, audit_errors = _run_pip_audit(args.requirements)
    findings, report_errors = _parse_audit_report(report)
    errors.extend(audit_errors)
    errors.extend(report_errors)
    if errors:
        _print_errors(errors)
        return 1

    excepted_count, evaluation_errors = _evaluate_findings(findings, exceptions)
    if evaluation_errors:
        _print_errors(evaluation_errors)
        return 1

    print(f"Python dependency audit passed: {len(findings)} finding(s), {excepted_count} temporary exception(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
