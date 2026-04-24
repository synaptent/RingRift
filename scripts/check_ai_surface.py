#!/usr/bin/env python3
"""Validate the supported-vs-experimental AI surface manifest."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "docs" / "data" / "ai_surface_manifest.json"
MAX_MANIFEST_AGE_DAYS = 45
SUPPORTED_KEYS = (
    "supported_ai_surfaces",
    "experimental_or_diagnostic_surfaces",
    "historical_surfaces",
)
FORBIDDEN_SUPPORTED_PREFIXES = (
    "archive",
    "docs/archive",
    "ai-service/archive",
    "ai-service/scripts/archive",
)
REQUIRED_SUPPORTED_PATHS = {
    "ai-service/app/rules",
    "ai-service/app/game_engine",
    "ai-service/app/training",
    "ai-service/scripts/minimal_alphazero_loop.py",
    "ai-service/scripts/generate_canonical_selfplay.py",
    "ai-service/scripts/check_ts_python_replay_parity.py",
    "ai-service/scripts/check_canonical_phase_history.py",
    "ai-service/scripts/export_replay_dataset.py",
    "ai-service/scripts/jsonl_to_npz.py",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(path: str) -> Path:
    return ROOT / path


def _path_exists(path: str) -> bool:
    if "*" in path:
        return any(ROOT.glob(path))
    return _repo_path(path).exists()


def _check_manifest_age(manifest: dict[str, Any], errors: list[str]) -> None:
    raw = manifest.get("as_of")
    if not isinstance(raw, str):
        errors.append("AI surface manifest must include string as_of date")
        return
    try:
        as_of = date.fromisoformat(raw)
    except ValueError:
        errors.append(f"AI surface manifest as_of is not ISO date: {raw}")
        return
    age_days = (date.today() - as_of).days
    if age_days < 0:
        errors.append(f"AI surface manifest as_of is in the future: {raw}")
    elif age_days > MAX_MANIFEST_AGE_DAYS:
        errors.append(
            f"AI surface manifest is stale: {raw} is {age_days} days old "
            f"(max {MAX_MANIFEST_AGE_DAYS})"
        )


def _check_surface_items(manifest: dict[str, Any], errors: list[str]) -> None:
    for key in SUPPORTED_KEYS:
        items = manifest.get(key)
        if not isinstance(items, list) or not items:
            errors.append(f"AI surface manifest missing non-empty {key}")
            continue
        for item in items:
            if not isinstance(item, dict):
                errors.append(f"{key} item is not an object: {item!r}")
                continue
            path = item.get("path")
            role = item.get("role")
            if not isinstance(path, str) or not path:
                errors.append(f"{key} item missing path: {item!r}")
                continue
            if not isinstance(role, str) or not role:
                errors.append(f"{key} item missing role: {item!r}")
            if not _path_exists(path):
                errors.append(f"{key} path missing: {path}")

    supported = {
        item.get("path")
        for item in manifest.get("supported_ai_surfaces", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    missing_required = sorted(REQUIRED_SUPPORTED_PATHS - supported)
    for path in missing_required:
        errors.append(f"supported_ai_surfaces missing required path: {path}")

    for path in sorted(supported):
        for forbidden in FORBIDDEN_SUPPORTED_PREFIXES:
            if path == forbidden or path.startswith(f"{forbidden}/"):
                errors.append(f"historical path must not be supported: {path}")


def _check_discoverability(errors: list[str]) -> None:
    manifest_ref = "docs/data/ai_surface_manifest.json"
    checks = {
        "README.md": ROOT / "README.md",
        "docs/REVIEWER_GUIDE.md": ROOT / "docs" / "REVIEWER_GUIDE.md",
        "ai-service/scripts/README.md": ROOT / "ai-service" / "scripts" / "README.md",
    }
    for label, path in checks.items():
        text = path.read_text(encoding="utf-8")
        if manifest_ref not in text:
            errors.append(f"{label} must mention {manifest_ref}")

    package_json = _load_json(ROOT / "package.json")
    scripts = package_json.get("scripts", {})
    if scripts.get("ai:surface:check") != "python3 scripts/check_ai_surface.py":
        errors.append("package.json must expose ai:surface:check")


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"Missing {MANIFEST_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 1

    errors: list[str] = []
    manifest = _load_json(MANIFEST_PATH)
    _check_manifest_age(manifest, errors)
    _check_surface_items(manifest, errors)
    _check_discoverability(errors)

    if not isinstance(manifest.get("promotion_rule"), str) or not manifest["promotion_rule"]:
        errors.append("AI surface manifest must include promotion_rule")
    if not isinstance(manifest.get("publication_rule"), str) or not manifest["publication_rule"]:
        errors.append("AI surface manifest must include publication_rule")

    if errors:
        print("AI surface check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("AI surface check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
