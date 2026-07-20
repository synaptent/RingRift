#!/usr/bin/env python3
"""Lightweight GitHub Actions workflow guardrails for fresh-clone CI."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = ROOT / ".github" / "workflows"
POLICY_REGISTRY_PATH = ROOT / "docs" / "data" / "workflow_policy_registry.json"
POLICY_SCHEMA_VERSION = 1
POLICY_CLASSIFICATIONS = {"required", "scheduled", "informational"}
LOCAL_USES_PATTERN = re.compile(r"^\s*uses:\s+(\./[^\s#]+)\s*$")
SECRET_IF_PATTERN = re.compile(r"^\s*if:\s*\$\{\{.*\bsecrets\.")
WORKFLOW_POLICY_PATH_PATTERN = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")
TOP_LEVEL_TRIGGER_PATTERN = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_-]*):")


def _workflow_paths() -> list[Path]:
    return sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])


def _check_workflow(path: Path, errors: list[str]) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "\t" in line:
            errors.append(f"{path.relative_to(ROOT)}:{line_number}: tab character in workflow YAML")

        if SECRET_IF_PATTERN.search(line):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: do not reference secrets.* "
                "directly in an if expression; mirror through job env first"
            )

        local_uses = LOCAL_USES_PATTERN.match(line)
        if local_uses:
            action_path = ROOT / local_uses.group(1)
            action_file = action_path / "action.yml"
            if not action_file.exists():
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number}: local action missing {action_file.relative_to(ROOT)}"
                )


def _load_policy_registry(errors: list[str]) -> Any:
    try:
        return json.loads(POLICY_REGISTRY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"{POLICY_REGISTRY_PATH.relative_to(ROOT)}: workflow policy registry is missing")
    except json.JSONDecodeError as exc:
        errors.append(f"{POLICY_REGISTRY_PATH.relative_to(ROOT)}:{exc.lineno}: invalid JSON: {exc.msg}")
    return None


def _top_level_triggers(path: Path) -> set[str]:
    """Return block-style, top-level workflow triggers without a YAML dependency."""
    triggers: set[str] = set()
    in_on_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_on_block:
            if line == "on:":
                in_on_block = True
            continue
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            break
        match = TOP_LEVEL_TRIGGER_PATTERN.match(line)
        if match:
            triggers.add(match.group(1))
    return triggers


def _validate_policy_registry(registry: Any, discovered_paths: set[str], errors: list[str]) -> dict[str, str]:
    """Validate registry structure and exact coverage; return valid path classifications."""
    classifications: dict[str, str] = {}
    if not isinstance(registry, dict):
        errors.append("workflow policy registry must be a JSON object")
        return classifications

    if registry.get("schema_version") != POLICY_SCHEMA_VERSION:
        errors.append(f"workflow policy registry schema_version must be {POLICY_SCHEMA_VERSION}")

    policy_classes = registry.get("policy_classes")
    if not isinstance(policy_classes, dict):
        errors.append("workflow policy registry policy_classes must be an object")
    else:
        actual_classes = set(policy_classes)
        if actual_classes != POLICY_CLASSIFICATIONS:
            errors.append(
                "workflow policy registry policy_classes must define exactly: "
                + ", ".join(sorted(POLICY_CLASSIFICATIONS))
            )
        for name, description in policy_classes.items():
            if not isinstance(description, str) or not description.strip():
                errors.append(f"workflow policy class {name!r} needs a non-empty description")

    entries = registry.get("workflows")
    if not isinstance(entries, list):
        errors.append("workflow policy registry workflows must be an array")
        return classifications

    seen: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"workflow policy entry {index}"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue

        path = entry.get("path")
        classification = entry.get("classification")
        rationale = entry.get("rationale")
        if not isinstance(path, str) or not WORKFLOW_POLICY_PATH_PATTERN.fullmatch(path):
            errors.append(f"{prefix} has an invalid workflow path")
            continue
        if path in seen:
            errors.append(f"workflow policy registry has duplicate path: {path}")
        seen.add(path)
        if classification not in POLICY_CLASSIFICATIONS:
            errors.append(f"{path}: invalid workflow classification {classification!r}")
            continue
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{path}: workflow policy rationale must be non-empty")
        classifications[path] = classification

    registered_paths = set(classifications)
    for path in sorted(discovered_paths - registered_paths):
        errors.append(f"{path}: workflow is unclassified")
    for path in sorted(registered_paths - discovered_paths):
        errors.append(f"{path}: registry entry does not match a workflow file")
    return classifications


def _check_policy_trigger_compatibility(
    workflows: list[Path], classifications: dict[str, str], errors: list[str]
) -> None:
    merge_triggers = {"pull_request", "pull_request_target", "push", "merge_group"}
    for path in workflows:
        relative_path = path.relative_to(ROOT).as_posix()
        classification = classifications.get(relative_path)
        if classification is None:
            continue
        triggers = _top_level_triggers(path)
        if classification == "required" and not triggers.intersection(merge_triggers):
            errors.append(f"{relative_path}: required workflow needs a pull request or push trigger")
        elif classification == "scheduled" and "schedule" not in triggers:
            errors.append(f"{relative_path}: scheduled workflow needs a schedule trigger")
        elif classification == "informational" and ("schedule" in triggers or triggers.intersection(merge_triggers)):
            errors.append(
                f"{relative_path}: informational workflow cannot use schedule, pull request, or push triggers"
            )


def main() -> int:
    errors: list[str] = []
    workflows = _workflow_paths()
    if not workflows:
        print("No GitHub workflow files found.", file=sys.stderr)
        return 1

    for path in workflows:
        _check_workflow(path, errors)

    registry = _load_policy_registry(errors)
    discovered_paths = {path.relative_to(ROOT).as_posix() for path in workflows}
    classifications = _validate_policy_registry(registry, discovered_paths, errors)
    _check_policy_trigger_compatibility(workflows, classifications, errors)

    if errors:
        print("GitHub workflow check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    counts = Counter(classifications.values())
    summary = ", ".join(
        f"{classification}={counts[classification]}" for classification in sorted(POLICY_CLASSIFICATIONS)
    )
    print(f"GitHub workflow check passed for {len(workflows)} workflow files ({summary}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
