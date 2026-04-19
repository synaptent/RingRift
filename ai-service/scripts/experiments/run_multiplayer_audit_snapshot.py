#!/usr/bin/env python3
"""Run the standard multiplayer audit set and write a timestamped snapshot.

This is a thin, read-only wrapper around ``multiplayer_audit.py``. It uses the
checked-in fleet manifest to discover the repo's standard multiplayer trainer
work directories, runs the per-config audit for every local metrics file that
exists, and writes a timestamped JSON artifact for later comparison.

The wrapper intentionally does not modify trainer state, model files, or live
service configuration. It only reads:

- ``docs/data/training_fleet_manifest.json`` for the standard target list
- ``data/minimal_loop_*/metrics.jsonl`` files under ``ai-service/``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
AI_SERVICE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "data" / "training_fleet_manifest.json"
DEFAULT_OUTPUT_DIR = AI_SERVICE_ROOT / "data" / "multiplayer_audit_snapshots"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multiplayer_audit import build_audit_report  # noqa: E402


@dataclass(frozen=True)
class AuditTarget:
    config: str
    metrics_path: Path
    source: str
    node: str | None = None


def _is_multiplayer_config(config: str | None) -> bool:
    return isinstance(config, str) and config.endswith(("3p", "4p"))


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected manifest JSON object in {manifest_path}")
    return payload


def _iter_manifest_targets(manifest: dict[str, Any]) -> list[AuditTarget]:
    discovered: dict[str, AuditTarget] = {}

    for entry in manifest.get("nodes", []):
        if not isinstance(entry, dict):
            continue
        config = entry.get("target_config")
        work_dir = entry.get("work_dir")
        if not _is_multiplayer_config(config) or not isinstance(work_dir, str):
            continue
        discovered.setdefault(
            config,
            AuditTarget(
                config=config,
                metrics_path=Path(work_dir) / "metrics.jsonl",
                source="manifest_node",
                node=str(entry.get("name")) if entry.get("name") else None,
            ),
        )

    for entry in manifest.get("script_only_canaries", []):
        if not isinstance(entry, dict):
            continue
        config = entry.get("target_config")
        work_dir = entry.get("work_dir")
        if not _is_multiplayer_config(config) or not isinstance(work_dir, str):
            continue
        discovered.setdefault(
            config,
            AuditTarget(
                config=config,
                metrics_path=Path(work_dir) / "metrics.jsonl",
                source="script_only_canary",
                node=None,
            ),
        )

    return list(discovered.values())


def discover_targets(
    *,
    manifest_path: Path,
    ai_service_root: Path,
    config_filters: set[str] | None = None,
) -> tuple[list[AuditTarget], list[dict[str, Any]]]:
    manifest = _load_manifest(manifest_path)
    found: list[AuditTarget] = []
    skipped: list[dict[str, Any]] = []

    for target in _iter_manifest_targets(manifest):
        if config_filters and target.config not in config_filters:
            continue
        resolved_metrics_path = ai_service_root / target.metrics_path
        resolved = AuditTarget(
            config=target.config,
            metrics_path=resolved_metrics_path,
            source=target.source,
            node=target.node,
        )
        if resolved_metrics_path.exists():
            found.append(resolved)
            continue
        skipped.append(
            {
                "config": resolved.config,
                "source": resolved.source,
                "node": resolved.node,
                "metrics_path": str(resolved.metrics_path),
                "reason": "metrics_missing",
            }
        )

    return found, skipped


def _default_output_path(output_dir: Path, generated_at: datetime) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"multiplayer_audit_{timestamp}.json"


def build_snapshot_artifact(
    *,
    manifest_path: Path,
    ai_service_root: Path,
    targets: list[AuditTarget],
    skipped_targets: list[dict[str, Any]],
    initial_elo: float,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = generated_at or datetime.now(timezone.utc)
    reports: list[dict[str, Any]] = []

    for target in targets:
        report = build_audit_report(
            metrics_path=target.metrics_path,
            config=target.config,
            initial_elo=initial_elo,
        )
        report["source"] = target.source
        report["node"] = target.node
        reports.append(report)

    return {
        "generated_at": created_at.isoformat(),
        "manifest_path": str(manifest_path),
        "ai_service_root": str(ai_service_root),
        "selected_configs": [target.config for target in targets],
        "reports": reports,
        "skipped_targets": skipped_targets,
    }


def _print_summary(artifact: dict[str, Any]) -> None:
    print("\n=== multiplayer audit snapshot ===")
    print(f"  generated_at:   {artifact['generated_at']}")
    print(f"  selected:       {artifact['selected_configs']}")
    print(f"  reports:        {len(artifact['reports'])}")
    print(f"  skipped:        {len(artifact['skipped_targets'])}")
    for report in artifact["reports"]:
        print(
            "  - "
            f"{report['config']}: iter={report['iterations_analyzed']} "
            f"promos={report['promotions']} latest_elo={report['latest_recomputed_estimated_elo']}"
        )
    for skipped in artifact["skipped_targets"]:
        print(
            "  - skipped "
            f"{skipped['config']} ({skipped['reason']}): {skipped['metrics_path']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to docs/data/training_fleet_manifest.json.",
    )
    parser.add_argument(
        "--ai-service-root",
        type=Path,
        default=AI_SERVICE_ROOT,
        help="Root directory containing data/minimal_loop_*/metrics.jsonl.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Optional multiplayer config filter(s), e.g. square8_3p.",
    )
    parser.add_argument(
        "--initial-elo",
        type=float,
        default=1500.0,
        help="Initial Elo used for recomputing corrected multiplayer history.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for timestamped snapshot files when --output is not provided.",
    )
    args = parser.parse_args()

    config_filters = set(args.configs or [])
    targets, skipped_targets = discover_targets(
        manifest_path=args.manifest,
        ai_service_root=args.ai_service_root,
        config_filters=config_filters or None,
    )
    if not targets:
        parser.error(
            "No local multiplayer metrics files found for the selected manifest targets. "
            "Use --config to narrow the set or ensure metrics.jsonl exists."
        )

    generated_at = datetime.now(timezone.utc)
    artifact = build_snapshot_artifact(
        manifest_path=args.manifest,
        ai_service_root=args.ai_service_root,
        targets=targets,
        skipped_targets=skipped_targets,
        initial_elo=args.initial_elo,
        generated_at=generated_at,
    )

    output_path = args.output or _default_output_path(args.output_dir, generated_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    _print_summary(artifact)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
