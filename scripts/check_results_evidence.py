#!/usr/bin/env python3
"""Validate public RingRift result claims against the checked-in snapshot."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "docs" / "data" / "results_snapshot.json"
MANIFEST_PATH = ROOT / "docs" / "data" / "results_evidence_manifest.json"
MAX_SNAPSHOT_AGE_DAYS = 30
PUBLIC_DOCS = (
    "README.md",
    "docs/RESULTS.md",
    "docs/RESEARCH_SNAPSHOT.md",
    "docs/PROJECT_BRIEF.md",
    "docs/REPRODUCIBILITY.md",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _headline(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    headline = {}
    for item in snapshot.get("headline", []):
        config = item.get("config")
        if isinstance(config, str):
            headline[config] = item
    return headline


def _fmt_number(value: Any) -> str:
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:.1f}"


def _check_snapshot_age(snapshot: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
    snapshot_as_of = snapshot.get("as_of")
    manifest_as_of = manifest.get("as_of")
    if snapshot_as_of != manifest_as_of:
        errors.append(
            f"results snapshot as_of ({snapshot_as_of}) does not match evidence manifest as_of ({manifest_as_of})"
        )
    if not isinstance(snapshot_as_of, str):
        errors.append("results_snapshot.json must include string as_of")
        return
    try:
        as_of = date.fromisoformat(snapshot_as_of)
    except ValueError:
        errors.append(f"results_snapshot.json as_of is not ISO date: {snapshot_as_of}")
        return
    age_days = (date.today() - as_of).days
    if age_days < 0:
        errors.append(f"results_snapshot.json as_of is in the future: {snapshot_as_of}")
    elif age_days > MAX_SNAPSHOT_AGE_DAYS:
        errors.append(
            f"results_snapshot.json is stale: {snapshot_as_of} is {age_days} days old "
            f"(max {MAX_SNAPSHOT_AGE_DAYS})"
        )


def _check_manifest_claims(
    headline: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    claims = manifest.get("repo_verifiable_claims", [])
    fields_by_config: dict[str, dict[str, Any]] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        fields = claim.get("fields")
        if not isinstance(fields, dict):
            continue
        config = fields.get("config")
        if isinstance(config, str) and {"start_elo", "best_elo", "promotions"} <= set(fields):
            fields_by_config[config] = fields

    for config, item in headline.items():
        fields = fields_by_config.get(config)
        if fields is None:
            errors.append(f"evidence manifest missing repo-verifiable headline claim for {config}")
            continue
        for key in ("start_elo", "best_elo", "promotions"):
            if fields.get(key) != item.get(key):
                errors.append(
                    f"evidence manifest {config}.{key}={fields.get(key)!r} "
                    f"does not match snapshot {item.get(key)!r}"
                )


def _check_public_docs(headline: dict[str, dict[str, Any]], errors: list[str]) -> None:
    docs = {path: _read(path) for path in PUBLIC_DOCS}

    for config, item in headline.items():
        start = _fmt_number(item["start_elo"])
        best = _fmt_number(item["best_elo"])
        promotions = str(int(item["promotions"]))

        required_docs = ("docs/RESULTS.md", "docs/RESEARCH_SNAPSHOT.md", "docs/PROJECT_BRIEF.md")
        for doc_path in required_docs:
            text = docs[doc_path]
            if config not in text:
                errors.append(f"{doc_path} missing headline config {config}")
            if best not in text:
                errors.append(f"{doc_path} missing headline Elo {best} for {config}")
            if promotions not in text:
                errors.append(f"{doc_path} missing promotion count {promotions} for {config}")

        if config in {"hex8_2p", "square8_2p"}:
            readme = docs["README.md"]
            if f"`{config}`" not in readme or best not in readme or promotions not in readme:
                errors.append(f"README.md missing supported headline claim for {config}")

            reproducibility = docs["docs/REPRODUCIBILITY.md"]
            if config not in reproducibility or best not in reproducibility:
                errors.append(f"docs/REPRODUCIBILITY.md missing reproducibility claim for {config}")

        if start != "1500":
            errors.append(f"unexpected non-1500 start Elo for {config}: {start}")


def _check_progression(snapshot: dict[str, Any], errors: list[str]) -> None:
    progression = snapshot.get("square8_2p_progression", {})
    points = progression.get("points", []) if isinstance(progression, dict) else []
    if not points:
        errors.append("results snapshot missing square8_2p progression points")
        return
    latest = points[-1]
    latest_elo = _fmt_number(latest.get("elo"))
    results = _read("docs/RESULTS.md")
    if latest_elo not in results:
        errors.append(f"docs/RESULTS.md missing latest square8_2p progression Elo {latest_elo}")


def main() -> int:
    errors: list[str] = []
    snapshot = _load_json(SNAPSHOT_PATH)
    manifest = _load_json(MANIFEST_PATH)
    headline = _headline(snapshot)

    if not headline:
        errors.append("results snapshot has no headline claims")
    _check_snapshot_age(snapshot, manifest, errors)
    _check_manifest_claims(headline, manifest, errors)
    _check_public_docs(headline, errors)
    _check_progression(snapshot, errors)

    if errors:
        print("Results evidence check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Results evidence check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
