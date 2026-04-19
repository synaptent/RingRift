#!/usr/bin/env python3
"""Audit multiplayer trainer metrics with corrected-Elo and seat-fairness summaries.

Background
----------
The multiplayer training lanes now have two important fixes on ``main``:

1. ``dfb3d20c1`` corrected the promotion Elo delta so ``3p/4p`` promotions are
   measured against the fair-seat baseline instead of a ``2p`` formula.
2. ``6555cdbaa`` replaced the seat-fairness max/min ratio gate with a
   chi-square test against the same iteration's selfplay seat distribution.

This script is the read-only analysis companion to those fixes. It consumes one
or more ``metrics.jsonl`` files, emits a per-iteration audit row for each
multiplayer config, and writes a JSON report that can be re-run as new
iterations land.

Output
------
Writes a JSON summary under ``data/multiplayer_audit/`` by default and prints a
concise text summary to stdout. Each iteration row includes:

- promotion / rejection outcome
- corrected Elo delta per promotion
- logged vs recomputed Elo
- seat-fairness details (`seat_wr`, selfplay null, expected wins, chi-square p)
- a verdict label:
  - ``threshold_audit_candidate`` for promotions
  - ``chi_square_fired`` when seat-fairness fired
  - ``clean_rejection`` otherwise

Safety
------
Read-only with respect to training. This script never mutates trainer state,
never touches model files, and only reads the ``metrics.jsonl`` paths it is
given.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_RE = re.compile(r"(hex8|hexagonal|square8|square19)_(\d)p")
SEAT_WARNING_TOKEN = "SEAT_WR_IMBALANCE"
DEFAULT_OUTPUT_DIR = Path("data/multiplayer_audit")
INITIAL_ELO = 1500.0


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_config_from_path(metrics_path: Path) -> str:
    candidates = [
        metrics_path.parent.name,
        *(parent.name for parent in metrics_path.parents),
        metrics_path.name,
    ]
    for candidate in candidates:
        match = CONFIG_RE.search(candidate)
        if match:
            return f"{match.group(1)}_{match.group(2)}p"
    raise ValueError(
        f"Could not infer config from {metrics_path}. Pass --config explicitly."
    )


def _num_players_from_config(config: str) -> int:
    match = CONFIG_RE.fullmatch(config)
    if not match:
        raise ValueError(f"Config must look like square8_3p or hex8_4p, got {config!r}")
    return int(match.group(2))


def _promotion_elo_delta(win_rate: float, num_players: int) -> float:
    if not 0.0 < win_rate < 1.0 or num_players < 2:
        return 0.0
    if num_players == 2:
        return 400.0 * math.log10(win_rate / (1.0 - win_rate))
    fair_win_rate = 1.0 / float(num_players)
    fair_odds = fair_win_rate / (1.0 - fair_win_rate)
    odds = win_rate / (1.0 - win_rate)
    return 400.0 * math.log10(odds / fair_odds)


def _load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            metric = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(metric, dict):
            rows.append(metric)
    return rows


def _extract_quality_gate(metric: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("quality_gate", "quality"):
        block = metric.get(key)
        if isinstance(block, dict):
            return block
    return None


def _extract_seat_fairness(metric: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    quality_gate = _extract_quality_gate(metric)
    if quality_gate:
        details = quality_gate.get("details")
        if isinstance(details, dict) and isinstance(details.get("seat_fairness"), dict):
            candidates.append(details["seat_fairness"])

    evaluation = metric.get("evaluation")
    if isinstance(evaluation, dict):
        if isinstance(evaluation.get("seat_fairness"), dict):
            candidates.append(evaluation["seat_fairness"])
        if isinstance(evaluation.get("seat_wr"), dict):
            candidates.append({"seat_wr": evaluation["seat_wr"]})

    direct = metric.get("seat_fairness")
    if isinstance(direct, dict):
        candidates.append(direct)

    return candidates[0] if candidates else None


def _extract_warnings(metric: dict[str, Any]) -> list[str]:
    quality_gate = _extract_quality_gate(metric)
    if not quality_gate:
        return []
    warnings = quality_gate.get("warnings")
    if isinstance(warnings, list):
        return [str(item) for item in warnings]
    return []


def _normalize_mapping(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        coerced = _coerce_float(value)
        if coerced is None:
            continue
        normalized[str(key)] = round(coerced, 4)
    return normalized or None


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        only = round(sorted_values[0], 4)
        return {
            "count": 1,
            "min": only,
            "p25": only,
            "median": only,
            "p75": only,
            "max": only,
        }
    q25, _, q75 = statistics.quantiles(sorted_values, n=4, method="inclusive")
    return {
        "count": len(sorted_values),
        "min": round(sorted_values[0], 4),
        "p25": round(q25, 4),
        "median": round(statistics.median(sorted_values), 4),
        "p75": round(q75, 4),
        "max": round(sorted_values[-1], 4),
    }


def build_audit_report(
    *,
    metrics_path: Path,
    config: str,
    initial_elo: float = INITIAL_ELO,
) -> dict[str, Any]:
    num_players = _num_players_from_config(config)
    if num_players < 3:
        raise ValueError(f"multiplayer_audit.py expects a 3p/4p config, got {config}")

    metrics = _load_metrics(metrics_path)
    rows: list[dict[str, Any]] = []
    verdict_counts: Counter[str] = Counter()
    corrected_running_elo = initial_elo
    chi_square_p_values: list[float] = []
    corrected_promotion_deltas: list[dict[str, Any]] = []

    for metric in metrics:
        iteration = _coerce_int(metric.get("iteration"))
        if iteration is None:
            continue

        evaluation = metric.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        logged_elo = _coerce_float(metric.get("estimated_elo"))
        win_rate = _coerce_float(evaluation.get("win_rate"))
        games_played = _coerce_int(evaluation.get("games_played"))
        promoted_raw = metric.get("promoted")
        promoted = (
            promoted_raw
            if isinstance(promoted_raw, bool)
            else evaluation.get("decision") == "promote"
        )

        corrected_delta = None
        if promoted and win_rate is not None:
            corrected_delta = _promotion_elo_delta(win_rate, num_players)
            corrected_running_elo += corrected_delta
            corrected_promotion_deltas.append(
                {
                    "iteration": iteration,
                    "win_rate": round(win_rate, 4),
                    "corrected_elo_delta": round(corrected_delta, 1),
                }
            )

        seat_fairness = _extract_seat_fairness(metric)
        warnings = _extract_warnings(metric)
        chi_square_p_value = None
        chi_square_stat = None
        chi_square_fired = False
        seat_fairness_skipped = None
        seat_fairness_note = None
        seat_wr = None
        selfplay_baseline_seat_wr = None
        expected_seat_wins = None

        if seat_fairness:
            chi_square_p_value = _coerce_float(seat_fairness.get("chi_square_p_value"))
            chi_square_stat = _coerce_float(seat_fairness.get("chi_square_stat"))
            if chi_square_p_value is not None:
                chi_square_p_values.append(chi_square_p_value)
            seat_fairness_skipped = seat_fairness.get("skipped")
            seat_fairness_note = seat_fairness.get("note")
            seat_wr = _normalize_mapping(seat_fairness.get("seat_wr"))
            selfplay_baseline_seat_wr = _normalize_mapping(
                seat_fairness.get("selfplay_baseline_seat_wr")
            )
            expected_seat_wins = _normalize_mapping(
                seat_fairness.get("expected_seat_wins")
            )
            chi_square_fired = (
                chi_square_p_value is not None and chi_square_p_value < 0.05
            )

        if not chi_square_fired and any(SEAT_WARNING_TOKEN in warning for warning in warnings):
            chi_square_fired = True

        if promoted:
            verdict = "threshold_audit_candidate"
        elif chi_square_fired:
            verdict = "chi_square_fired"
        else:
            verdict = "clean_rejection"

        verdict_counts[verdict] += 1
        rows.append(
            {
                "iteration": iteration,
                "decision": evaluation.get("decision"),
                "promoted": bool(promoted),
                "win_rate": round(win_rate, 4) if win_rate is not None else None,
                "games_played": games_played,
                "logged_estimated_elo": round(logged_elo, 1) if logged_elo is not None else None,
                "corrected_elo_delta": (
                    round(corrected_delta, 1) if corrected_delta is not None else None
                ),
                "recomputed_estimated_elo": round(corrected_running_elo, 1),
                "seat_wr": seat_wr,
                "selfplay_baseline_seat_wr": selfplay_baseline_seat_wr,
                "expected_seat_wins": expected_seat_wins,
                "chi_square_stat": round(chi_square_stat, 4) if chi_square_stat is not None else None,
                "chi_square_p_value": (
                    round(chi_square_p_value, 4) if chi_square_p_value is not None else None
                ),
                "chi_square_fired": chi_square_fired,
                "seat_fairness_skipped": seat_fairness_skipped,
                "seat_fairness_note": seat_fairness_note,
                "warnings": warnings,
                "verdict": verdict,
            }
        )

    return {
        "config": config,
        "num_players": num_players,
        "metrics_path": str(metrics_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "iterations_analyzed": len(rows),
        "promotions": sum(1 for row in rows if row["promoted"]),
        "latest_logged_estimated_elo": rows[-1]["logged_estimated_elo"] if rows else None,
        "latest_recomputed_estimated_elo": rows[-1]["recomputed_estimated_elo"] if rows else initial_elo,
        "verdict_counts": dict(verdict_counts),
        "chi_square_p_value_distribution": _quantiles(chi_square_p_values),
        "corrected_promotion_deltas": corrected_promotion_deltas,
        "rows": rows,
    }


def _default_output_path(output_dir: Path, reports: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(reports) == 1:
        return output_dir / f"{reports[0]['config']}_audit.json"
    return output_dir / "multiplayer_audit.json"


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"\n=== multiplayer audit: {report['config']} "
        f"({report['iterations_analyzed']} iterations) ==="
    )
    print(f"  metrics:            {report['metrics_path']}")
    print(f"  promotions:         {report['promotions']}")
    print(f"  logged latest elo:  {report['latest_logged_estimated_elo']}")
    print(f"  recomputed elo:     {report['latest_recomputed_estimated_elo']}")
    print(f"  verdict counts:     {report['verdict_counts']}")
    if report["chi_square_p_value_distribution"]:
        dist = report["chi_square_p_value_distribution"]
        print(
            "  chi-square p:       "
            f"n={dist['count']} min={dist['min']} median={dist['median']} max={dist['max']}"
        )
    if report["corrected_promotion_deltas"]:
        deltas = ", ".join(
            f"iter {item['iteration']} -> +{item['corrected_elo_delta']}"
            for item in report["corrected_promotion_deltas"]
        )
        print(f"  corrected deltas:   {deltas}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="+",
        help="Path(s) to trainer metrics.jsonl files",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Optional config override(s) matching the metrics paths order",
    )
    parser.add_argument(
        "--initial-elo",
        type=float,
        default=INITIAL_ELO,
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
        help="Directory for generated JSON report when --output is not provided.",
    )
    args = parser.parse_args()

    metrics_paths = [Path(path) for path in args.metrics]
    config_overrides = args.configs or []
    if config_overrides and len(config_overrides) != len(metrics_paths):
        parser.error("--config must be omitted or provided once per metrics path")

    reports: list[dict[str, Any]] = []
    for index, metrics_path in enumerate(metrics_paths):
        config = (
            config_overrides[index]
            if config_overrides
            else _infer_config_from_path(metrics_path)
        )
        reports.append(
            build_audit_report(
                metrics_path=metrics_path,
                config=config,
                initial_elo=args.initial_elo,
            )
        )

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
    }
    output_path = args.output or _default_output_path(args.output_dir, reports)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    for report in reports:
        _print_summary(report)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
