#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_results_visuals import generate_visuals


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_ROOT = ROOT / "ai-service" / "data"

HEADLINE_CONFIGS = ("hex8_2p", "square8_2p", "square8_3p")
PROGRESSION_CONFIG = "square8_2p"


def _candidate_paths(metrics_root: Path, config: str) -> list[Path]:
    return [
        metrics_root / f"minimal_loop_{config}" / "metrics.jsonl",
        metrics_root / "proven_experiments" / config / "metrics.jsonl",
    ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _resolve_metrics(search_roots: list[Path], config: str) -> tuple[Path | None, list[dict[str, Any]]]:
    for root in search_roots:
        for candidate in _candidate_paths(root, config):
            if candidate.exists():
                return candidate, _load_jsonl(candidate)
    return None, []


def _parse_record_date(record: dict[str, Any]) -> str | None:
    timestamp = record.get("timestamp")
    if timestamp is None:
        return None
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date().isoformat()
    if isinstance(timestamp, str):
        try:
            normalized = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).date().isoformat()
        except ValueError:
            return None
    return None


def _fallback_promotions(item: dict[str, Any]) -> int:
    value = item.get("promotions", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _summarize_headline(config: str, records: list[dict[str, Any]], fallback_item: dict[str, Any]) -> dict[str, Any]:
    if not records:
        return fallback_item

    start_elo = float(fallback_item.get("start_elo", records[0].get("estimated_elo", 1500.0)))
    best_elo = max(float(record.get("estimated_elo", start_elo)) for record in records)
    promotions = max(
        int(record.get("total_promotions", _fallback_promotions(fallback_item)))
        for record in records
    )
    return {
        "config": config,
        "start_elo": round(start_elo, 1),
        "best_elo": round(best_elo, 1),
        "promotions": promotions,
    }


def _summarize_progression(
    config: str,
    records: list[dict[str, Any]],
    fallback_progression: dict[str, Any],
    window: int,
) -> dict[str, Any]:
    if not records:
        return fallback_progression

    tail = records[-window:]
    return {
        "config": config,
        "points": [
            {
                "iteration": int(record["iteration"]),
                "elo": round(float(record.get("estimated_elo", 1500.0)), 1),
                "promoted": bool(record.get("promoted", False)),
            }
            for record in tail
        ],
    }


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the checked-in results snapshot and SVG artifacts from local metrics files."
    )
    parser.add_argument(
        "--snapshot",
        default="docs/data/results_snapshot.json",
        help="Snapshot JSON to update in place.",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/assets/results",
        help="Output directory for generated SVGs.",
    )
    parser.add_argument(
        "--metrics-root",
        action="append",
        help=(
            "Root directory to search for metrics files. "
            "Defaults to ai-service/data. Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--progression-window",
        type=int,
        default=5,
        help="How many recent points to keep for the square8_2p progression chart.",
    )
    parser.add_argument(
        "--as-of",
        help="Override the as_of date in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the refreshed snapshot without writing files.",
    )
    args = parser.parse_args()

    snapshot_path = ROOT / args.snapshot
    out_dir = ROOT / args.out_dir
    snapshot = _load_snapshot(snapshot_path)

    existing_headline = {
        item["config"]: item for item in snapshot.get("headline", []) if "config" in item
    }
    fallback_progression = snapshot.get(
        "square8_2p_progression",
        {"config": PROGRESSION_CONFIG, "points": []},
    )

    search_roots = [Path(path).resolve() for path in (args.metrics_root or [str(DEFAULT_METRICS_ROOT)])]
    source_report: dict[str, str] = {}
    observed_dates: list[str] = []

    headline_items: list[dict[str, Any]] = []
    progression_records: list[dict[str, Any]] = []
    for config in HEADLINE_CONFIGS:
        source_path, records = _resolve_metrics(search_roots, config)
        if source_path is not None:
            source_report[config] = str(source_path)
            for record in records:
                record_date = _parse_record_date(record)
                if record_date:
                    observed_dates.append(record_date)
        else:
            source_report[config] = "missing"
        fallback_item = existing_headline.get(
            config,
            {"config": config, "start_elo": 1500.0, "best_elo": 1500.0, "promotions": 0},
        )
        headline_items.append(_summarize_headline(config, records, fallback_item))
        if config == PROGRESSION_CONFIG:
            progression_records = records

    refreshed = {
        **snapshot,
        "as_of": args.as_of or (max(observed_dates) if observed_dates else snapshot.get("as_of")),
        "headline": headline_items,
        "square8_2p_progression": _summarize_progression(
            PROGRESSION_CONFIG,
            progression_records,
            fallback_progression,
            args.progression_window,
        ),
    }

    if args.dry_run:
        print(json.dumps({"sources": source_report, "snapshot": refreshed}, indent=2))
        return

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(refreshed, indent=2) + "\n", encoding="utf-8")
    generate_visuals(snapshot_path, out_dir)

    print(f"Updated {snapshot_path}")
    print(f"Updated {out_dir / 'headline_results.svg'}")
    print(f"Updated {out_dir / 'square8_2p_progression.svg'}")
    for config, source in source_report.items():
        print(f"{config}: {source}")


if __name__ == "__main__":
    main()
