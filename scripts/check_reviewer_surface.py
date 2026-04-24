#!/usr/bin/env python3
"""Validate the outside-reviewer entrypoint and supported-path map."""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "docs" / "data" / "reviewer_surface_manifest.json"
MAX_MANIFEST_AGE_DAYS = 45
REQUIRED_GUIDE_SECTIONS = (
    "## Review Thesis",
    "## Ten-Minute Review Path",
    "## Evidence Boundary",
    "## Trust Commands",
    "## What To Ignore At First",
    "## Current Reviewer Risks",
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _failures() -> list[str]:
    return []


def _repo_path(path: str) -> Path:
    return ROOT / path


def _is_external_link(target: str) -> bool:
    return (
        "://" in target
        or target.startswith("#")
        or target.startswith("mailto:")
        or target.startswith("tel:")
    )


def _resolve_markdown_link(source: Path, target: str) -> Path | None:
    if _is_external_link(target):
        return None
    clean = target.split("#", 1)[0]
    if not clean:
        return None
    if clean.startswith("/"):
        return ROOT / clean.lstrip("/")
    return (source.parent / clean).resolve()


def _check_local_links(paths: list[Path], errors: list[str]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(text):
            resolved = _resolve_markdown_link(path, target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(f"{path.relative_to(ROOT)} has broken local link: {target}")


def _check_manifest_age(manifest: dict[str, Any], errors: list[str]) -> None:
    raw = manifest.get("as_of")
    if not isinstance(raw, str):
        errors.append("reviewer manifest must include string as_of date")
        return
    try:
        as_of = date.fromisoformat(raw)
    except ValueError:
        errors.append(f"reviewer manifest as_of is not ISO date: {raw}")
        return
    age_days = (date.today() - as_of).days
    if age_days < 0:
        errors.append(f"reviewer manifest as_of is in the future: {raw}")
    elif age_days > MAX_MANIFEST_AGE_DAYS:
        errors.append(
            f"reviewer manifest is stale: {raw} is {age_days} days old "
            f"(max {MAX_MANIFEST_AGE_DAYS})"
        )


def _check_paths(manifest: dict[str, Any], errors: list[str]) -> None:
    entrypoint = manifest.get("reviewer_entrypoint")
    if entrypoint != "docs/REVIEWER_GUIDE.md":
        errors.append("reviewer_entrypoint must be docs/REVIEWER_GUIDE.md")

    for key in ("must_read", "supported_code_surfaces"):
        if key not in manifest:
            errors.append(f"reviewer manifest missing {key}")

    for item in manifest.get("must_read", []):
        if not isinstance(item, str):
            errors.append(f"must_read item is not a string: {item!r}")
            continue
        if item.startswith("docs/archive") or item.startswith("archive"):
            errors.append(f"must_read must not start in archive surface: {item}")
        if not _repo_path(item).exists():
            errors.append(f"must_read path missing: {item}")

    for item in manifest.get("supported_code_surfaces", []):
        if not isinstance(item, dict):
            errors.append(f"supported_code_surfaces item is not an object: {item!r}")
            continue
        path = item.get("path")
        role = item.get("role")
        if not isinstance(path, str) or not path:
            errors.append(f"supported_code_surfaces item missing path: {item!r}")
            continue
        if not isinstance(role, str) or not role:
            errors.append(f"supported_code_surfaces item missing role: {item!r}")
        if not _repo_path(path).exists():
            errors.append(f"supported code surface missing: {path}")


def _check_guide(errors: list[str]) -> None:
    guide = ROOT / "docs" / "REVIEWER_GUIDE.md"
    if not guide.exists():
        errors.append("docs/REVIEWER_GUIDE.md is missing")
        return

    text = guide.read_text(encoding="utf-8")
    for section in REQUIRED_GUIDE_SECTIONS:
        if section not in text:
            errors.append(f"docs/REVIEWER_GUIDE.md missing section: {section}")

    if "docs/data/results_evidence_manifest.json" not in text:
        errors.append("reviewer guide must link the result evidence manifest")
    if "scripts/check_reviewer_surface.py" not in text:
        errors.append("reviewer guide must link its validator")


def _check_discoverability(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "INDEX.md").read_text(encoding="utf-8")
    package_json = _load_json(ROOT / "package.json")
    scripts = package_json.get("scripts", {})

    if "docs/REVIEWER_GUIDE.md" not in readme:
        errors.append("README.md must link docs/REVIEWER_GUIDE.md")
    if "REVIEWER_GUIDE.md" not in docs_index:
        errors.append("docs/INDEX.md must link REVIEWER_GUIDE.md")
    if scripts.get("reviewer:check") != "python3 scripts/check_reviewer_surface.py":
        errors.append("package.json must expose reviewer:check")


def main() -> int:
    errors = _failures()
    if not MANIFEST_PATH.exists():
        print(f"Missing {MANIFEST_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 1

    manifest = _load_json(MANIFEST_PATH)
    _check_manifest_age(manifest, errors)
    _check_paths(manifest, errors)
    _check_guide(errors)
    _check_discoverability(errors)

    link_paths = [
        ROOT / "README.md",
        ROOT / "docs" / "INDEX.md",
        ROOT / "docs" / "REVIEWER_GUIDE.md",
    ]
    _check_local_links(link_paths, errors)

    if errors:
        print("Reviewer surface check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Reviewer surface check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
