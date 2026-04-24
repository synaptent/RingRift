#!/usr/bin/env python3
"""Lightweight GitHub Actions workflow guardrails for fresh-clone CI."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = ROOT / ".github" / "workflows"
LOCAL_USES_PATTERN = re.compile(r"^\s*uses:\s+(\./[^\s#]+)\s*$")
SECRET_IF_PATTERN = re.compile(r"^\s*if:\s*\$\{\{.*\bsecrets\.")


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


def main() -> int:
    errors: list[str] = []
    workflows = _workflow_paths()
    if not workflows:
        print("No GitHub workflow files found.", file=sys.stderr)
        return 1

    for path in workflows:
        _check_workflow(path, errors)

    if errors:
        print("GitHub workflow check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"GitHub workflow check passed for {len(workflows)} workflow files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
