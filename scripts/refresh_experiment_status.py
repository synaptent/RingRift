#!/usr/bin/env python3
"""Build a machine-readable experiment evidence status artifact.

The public result docs should not depend on memory or chat logs. This script
combines the checked-in fleet manifest, headline result snapshot, and optional
live status snapshots into docs/data/experiment_status.json so docs can cite
specific evidence paths and fields.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLEET_MANIFEST = ROOT / "docs" / "data" / "training_fleet_manifest.json"
DEFAULT_RESULTS_SNAPSHOT = ROOT / "docs" / "data" / "results_snapshot.json"
DEFAULT_TRAINING_STATUS = ROOT / "docs" / "data" / "training_status.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "experiment_status.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_json(path)


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if isinstance(value.get("experiments"), list):
            return [item for item in value["experiments"] if isinstance(item, dict)]
        return [value]
    return []


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _source_entry(path: Path, source_type: str) -> dict[str, str]:
    return {
        "source_type": source_type,
        "path": _repo_relative(path),
    }


def _status_key(entry: dict[str, Any]) -> tuple[str | None, str | None]:
    return entry.get("node"), entry.get("config") or entry.get("target_config")


def _latest_metric(entry: dict[str, Any]) -> dict[str, Any] | None:
    metric = entry.get("latest_metrics")
    if isinstance(metric, dict):
        return metric
    tail = entry.get("metrics_tail")
    if isinstance(tail, list):
        for candidate in reversed(tail):
            if isinstance(candidate, dict):
                return candidate
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _progress_from_status(entry: dict[str, Any]) -> dict[str, Any]:
    progress = entry.get("progress")
    if isinstance(progress, dict):
        return progress
    return {}


def extract_seat_fairness(metric: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return normalized per-seat WR evidence from a metrics row if present."""
    if not metric:
        return None

    candidates: list[dict[str, Any]] = []
    quality_gate = metric.get("quality_gate")
    if isinstance(quality_gate, dict):
        details = quality_gate.get("details")
        if isinstance(details, dict) and isinstance(details.get("seat_fairness"), dict):
            candidates.append(details["seat_fairness"])

    quality = metric.get("quality")
    if isinstance(quality, dict):
        details = quality.get("details")
        if isinstance(details, dict) and isinstance(details.get("seat_fairness"), dict):
            candidates.append(details["seat_fairness"])

    evaluation = metric.get("evaluation")
    if isinstance(evaluation, dict):
        if isinstance(evaluation.get("seat_fairness"), dict):
            candidates.append(evaluation["seat_fairness"])
        if isinstance(evaluation.get("seat_wr"), dict):
            candidates.append({"seat_wr": evaluation.get("seat_wr")})

    if isinstance(metric.get("seat_fairness"), dict):
        candidates.append(metric["seat_fairness"])
    if isinstance(metric.get("seat_wr"), dict):
        candidates.append({"seat_wr": metric.get("seat_wr")})

    for candidate in candidates:
        seat_wr = candidate.get("seat_wr")
        if not isinstance(seat_wr, dict):
            continue
        normalized = {
            str(seat): round(float(wr), 3)
            for seat, wr in sorted(seat_wr.items(), key=lambda item: int(item[0]))
        }
        result: dict[str, Any] = {"seat_wr": normalized}
        if isinstance(candidate.get("seat_games"), dict):
            result["seat_games"] = {
                str(seat): int(games)
                for seat, games in sorted(candidate["seat_games"].items(), key=lambda item: int(item[0]))
            }
        if candidate.get("wr_ratio") is not None:
            result["wr_ratio"] = round(float(candidate["wr_ratio"]), 3)
        if candidate.get("skipped"):
            result["skipped"] = str(candidate["skipped"])
        return result
    return None


def _status_from_entry(entry: dict[str, Any], metric: dict[str, Any] | None) -> str:
    if entry.get("process_alive") is True or entry.get("loop_alive") is True:
        return "running"
    if entry.get("process_alive") is False or entry.get("loop_alive") is False:
        return "inactive"
    if _progress_from_status(entry):
        return "observed"
    if metric:
        return "has_metrics"
    return str(entry.get("status") or "unknown")


def _build_experiment_from_status(
    entry: dict[str, Any],
    source: dict[str, str],
) -> dict[str, Any]:
    metric = _latest_metric(entry)
    progress = _progress_from_status(entry)
    evaluation = metric.get("evaluation", {}) if isinstance(metric, dict) else {}
    if not isinstance(evaluation, dict):
        evaluation = {}

    estimated_elo = (
        _coerce_float(progress.get("estimated_elo"))
        or _coerce_float(entry.get("elo"))
        or _coerce_float(metric.get("estimated_elo") if metric else None)
    )
    promotions = (
        _coerce_int(progress.get("total_promotions"))
        if progress.get("total_promotions") is not None
        else _coerce_int(entry.get("promotions"))
    )
    if promotions is None and metric:
        promotions = _coerce_int(metric.get("total_promotions"))

    evidence = dict(source)
    remote_evidence = entry.get("remote_evidence")
    if isinstance(remote_evidence, dict):
        evidence = {
            "source_type": "live_status_snapshot",
            "remote": remote_evidence,
        }

    experiment: dict[str, Any] = {
        "config": entry.get("config") or entry.get("target_config"),
        "node": entry.get("node") or entry.get("name"),
        "role": entry.get("role"),
        "model_version": entry.get("model_version") or entry.get("MODEL_VERSION"),
        "status": _status_from_entry(entry, metric),
        "stage": progress.get("stage") or entry.get("stage"),
        "iteration": _coerce_int(progress.get("iteration")) or _coerce_int(entry.get("iteration"))
        or _coerce_int(metric.get("iteration") if metric else None),
        "estimated_elo": round(estimated_elo, 1) if estimated_elo is not None else None,
        "promotions": promotions,
        "work_dir": entry.get("work_dir"),
        "evidence": evidence,
    }

    latest_decision = progress.get("last_decision") or evaluation.get("decision")
    latest_win_rate = _coerce_float(progress.get("last_win_rate")) or _coerce_float(evaluation.get("win_rate"))
    latest_games_played = _coerce_int(evaluation.get("games_played"))
    if latest_decision is not None:
        experiment["latest_decision"] = latest_decision
    if latest_win_rate is not None:
        experiment["latest_win_rate"] = round(latest_win_rate, 4)
    if latest_games_played is not None:
        experiment["latest_games_played"] = latest_games_played

    seat_fairness = extract_seat_fairness(metric)
    if seat_fairness:
        experiment["seat_fairness"] = seat_fairness
    if entry.get("gpu_utilization_pct") is not None:
        experiment["gpu_utilization_pct"] = _coerce_float(entry.get("gpu_utilization_pct"))
    if entry.get("gpu_memory_used_mb") is not None:
        experiment["gpu_memory_used_mb"] = _coerce_float(entry.get("gpu_memory_used_mb"))

    return {key: value for key, value in experiment.items() if value is not None}


def _build_manifest_experiment(
    node: dict[str, Any],
    status_entry: dict[str, Any] | None,
    source: dict[str, str],
) -> dict[str, Any]:
    if status_entry:
        merged = {**node, **status_entry}
        return _build_experiment_from_status(merged, source)
    experiment = {
        "config": node.get("target_config"),
        "node": node.get("name"),
        "role": node.get("role"),
        "status": "configured",
        "work_dir": node.get("work_dir"),
        "evidence": source,
    }
    return {key: value for key, value in experiment.items() if value is not None}


def _headline_claims(snapshot: dict[str, Any], snapshot_path: Path) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for index, item in enumerate(snapshot.get("headline", [])):
        config = item.get("config")
        if not config:
            continue
        claims.append(
            {
                "claim_id": f"{config}_best_elo",
                "config": config,
                "field": f"headline[{index}].best_elo",
                "value": item.get("best_elo"),
                "evidence": _source_entry(snapshot_path, "results_snapshot"),
            }
        )
        claims.append(
            {
                "claim_id": f"{config}_promotions",
                "config": config,
                "field": f"headline[{index}].promotions",
                "value": item.get("promotions"),
                "evidence": _source_entry(snapshot_path, "results_snapshot"),
            }
        )
    return claims


def build_experiment_status(
    *,
    fleet_manifest_path: Path,
    results_snapshot_path: Path,
    training_status_path: Path | None = None,
    extra_status_paths: list[Path] | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    fleet_manifest = _load_json(fleet_manifest_path)
    results_snapshot = _load_json(results_snapshot_path)

    status_entries: list[tuple[dict[str, Any], dict[str, str]]] = []
    if training_status_path:
        training_status = _maybe_load_json(training_status_path)
        for entry in _as_list(training_status):
            status_entries.append((entry, _source_entry(training_status_path, "training_status")))
    for path in extra_status_paths or []:
        for entry in _as_list(_load_json(path)):
            status_entries.append((entry, _source_entry(path, "extra_status")))

    by_node: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    by_config: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for entry, source in status_entries:
        node, config = _status_key(entry)
        if node:
            by_node[node] = (entry, source)
        if config:
            by_config[config] = (entry, source)

    experiments: list[dict[str, Any]] = []
    used_status_ids: set[int] = set()
    for node in fleet_manifest.get("nodes", []):
        name = node.get("name")
        config = node.get("target_config")
        status_pair = by_node.get(name) if name else None
        if status_pair is None and not name and config:
            status_pair = by_config.get(config)
        if status_pair is None:
            experiments.append(_build_manifest_experiment(node, None, _source_entry(fleet_manifest_path, "fleet_manifest")))
            continue
        entry, source = status_pair
        used_status_ids.add(id(entry))
        experiments.append(_build_manifest_experiment(node, entry, source))

    for entry, source in status_entries:
        if id(entry) not in used_status_ids:
            experiments.append(_build_experiment_from_status(entry, source))

    for experiment in results_snapshot.get("current_experiments", []):
        if not isinstance(experiment, dict):
            continue
        experiments.append(
            {
                "config": experiment.get("config"),
                "model_version": experiment.get("model_version"),
                "status": experiment.get("status", "operator_reported"),
                "work_dir": experiment.get("work_dir"),
                "note": experiment.get("note"),
                "evidence": _source_entry(results_snapshot_path, "results_snapshot_current_experiments"),
            }
        )

    source_files = [
        _repo_relative(fleet_manifest_path),
        _repo_relative(results_snapshot_path),
    ]
    if training_status_path and training_status_path.exists():
        source_files.append(_repo_relative(training_status_path))
    external_status_inputs: list[str] = []
    for path in extra_status_paths or []:
        try:
            path.resolve().relative_to(ROOT)
        except ValueError:
            external_status_inputs.append(path.as_posix())
        else:
            source_files.append(_repo_relative(path))

    artifact = {
        "schema_version": 1,
        "generated_at": observed_at or datetime.now(timezone.utc).isoformat(),
        "source_files": source_files,
        "headline_claims": _headline_claims(results_snapshot, results_snapshot_path),
        "experiments": experiments,
    }
    if external_status_inputs:
        artifact["external_status_inputs"] = external_status_inputs
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh docs/data/experiment_status.json from checked-in and optional live status evidence."
    )
    parser.add_argument("--fleet-manifest", default=str(DEFAULT_FLEET_MANIFEST))
    parser.add_argument("--results-snapshot", default=str(DEFAULT_RESULTS_SNAPSHOT))
    parser.add_argument(
        "--training-status",
        default=str(DEFAULT_TRAINING_STATUS),
        help="Optional training status JSON. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--extra-status",
        action="append",
        default=[],
        help="Additional JSON status file. May be a single object, list, or {'experiments': [...]} object.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--observed-at", help="Override generated_at for deterministic tests.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    training_status = Path(args.training_status) if args.training_status else None
    artifact = build_experiment_status(
        fleet_manifest_path=Path(args.fleet_manifest),
        results_snapshot_path=Path(args.results_snapshot),
        training_status_path=training_status,
        extra_status_paths=[Path(path) for path in args.extra_status],
        observed_at=args.observed_at,
    )

    text = json.dumps(artifact, indent=2) + "\n"
    if args.dry_run:
        print(text, end="")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Updated {_repo_relative(output_path)}")


if __name__ == "__main__":
    main()
