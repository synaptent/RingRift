#!/usr/bin/env python
"""Offline difficulty calibration analysis for Square-8 2-player ladder tiers.

This script ingests pre-aggregated human calibration metrics, joins them with
the current ladder configuration and tier candidate registry, optionally
enriches with recent tier evaluation / perf artefacts, and emits a
machine-readable JSON summary plus an optional Markdown report.

The calibration aggregates are expected to be exported from metrics or
warehouse pipelines into a JSON file; this script is intentionally file-
based and does not talk to Prometheus or any database directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure app/ is importable when running as a script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config.ladder_config import (
    LadderTierConfig,
    get_ladder_tier_config,
)
from app.models import AIType, BoardType
from app.training.tier_promotion_registry import (
    DEFAULT_SQUARE8_2P_REGISTRY_PATH,
    load_square8_two_player_registry,
)

ALLOWED_SQUARE8_2P_TIERS = {"D2", "D4", "D6", "D8", "D9", "D10"}


class CalibrationInputError(Exception):
    """Raised when the calibration aggregates input file is invalid."""


@dataclass
class CalibrationSegmentAggregate:
    """Aggregated calibration metrics for a single player segment."""

    segment: str
    n_games: int
    human_win_rate: float
    difficulty_mean: float
    difficulty_p10: float
    difficulty_p90: float


@dataclass
class CalibrationTierAggregate:
    """Aggregated calibration metrics for a single difficulty tier."""

    tier: str
    difficulty: int
    segments: list[CalibrationSegmentAggregate]


@dataclass
class CalibrationAggregateRoot:
    """Top-level calibration aggregate payload for a time window."""

    board: str
    num_players: int
    window: dict[str, Any]
    tiers: list[CalibrationTierAggregate]


@dataclass(frozen=True)
class TierCalibrationThresholds:
    """Simple, tweakable thresholds for classifying calibration segments.

    Values are loosely derived from the qualitative bands in
    AI_HUMAN_CALIBRATION_GUIDE.md and AI_DIFFICULTY_CALIBRATION_ANALYSIS.md.
    """

    target_low: float
    target_high: float
    too_easy_win: float
    too_hard_win: float
    easy_mean_max: float = 2.5
    hard_mean_min: float = 3.5


# Thresholds are keyed by tier name; they currently apply to all segments
# for that tier and can be refined in future if segment-specific bands are
# needed.
TIER_THRESHOLDS: dict[str, TierCalibrationThresholds] = {
    "D2": TierCalibrationThresholds(
        target_low=0.25,
        target_high=0.70,
        too_easy_win=0.80,
        too_hard_win=0.20,
    ),
    "D4": TierCalibrationThresholds(
        target_low=0.30,
        target_high=0.70,
        too_easy_win=0.80,
        too_hard_win=0.25,
    ),
    "D6": TierCalibrationThresholds(
        target_low=0.40,
        target_high=0.60,
        too_easy_win=0.75,
        too_hard_win=0.25,
    ),
    "D8": TierCalibrationThresholds(
        target_low=0.30,
        target_high=0.60,
        too_easy_win=0.65,
        too_hard_win=0.20,
    ),
}


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_calibration_aggregates(path: str) -> CalibrationAggregateRoot:
    """Load and validate the calibration aggregates JSON file."""

    file_path = Path(path)
    if not file_path.exists():
        raise CalibrationInputError("Calibration aggregates file not found: " f"{file_path}")

    try:
        payload = _load_json_file(file_path)
    except json.JSONDecodeError as exc:
        raise CalibrationInputError("Failed to parse calibration aggregates JSON: " f"{exc}") from exc

    if not isinstance(payload, dict):
        raise CalibrationInputError("Calibration aggregates root must be a JSON object.")

    board_val = payload.get("board")
    num_players_val = payload.get("num_players")
    invalid_board = board_val != "square8"
    invalid_players = not isinstance(num_players_val, int) or num_players_val != 2
    if invalid_board or invalid_players:
        raise CalibrationInputError(
            "Calibration aggregates must be for board='square8', "
            "num_players=2; "
            f"got board={board_val!r}, num_players={num_players_val!r}."
        )
    # At this point inputs are validated and we only support the canonical
    # Square-8 2-player configuration.
    board = "square8"
    num_players = 2

    window = payload.get("window") or {}
    if not isinstance(window, dict):
        raise CalibrationInputError("window must be an object with 'start'/'end' fields.")

    raw_tiers = payload.get("tiers") or []
    if not isinstance(raw_tiers, list):
        raise CalibrationInputError("tiers must be a list.")

    tiers: list[CalibrationTierAggregate] = []
    for entry in raw_tiers:
        if not isinstance(entry, dict):
            raise CalibrationInputError("Each tier entry must be an object.")
        tier_name = str(entry.get("tier", "")).upper()
        if tier_name not in ALLOWED_SQUARE8_2P_TIERS:
            allowed = sorted(ALLOWED_SQUARE8_2P_TIERS)
            raise CalibrationInputError(f"Unsupported tier {tier_name!r}; expected one of " f"{allowed}.")
        difficulty = int(entry.get("difficulty", 0))
        expected_difficulty = int(tier_name[1:])
        if difficulty != expected_difficulty:
            raise CalibrationInputError(
                f"Tier {tier_name!r} has mismatched difficulty=" f"{difficulty}; expected {expected_difficulty}."
            )

        raw_segments = entry.get("segments") or []
        if not isinstance(raw_segments, list):
            raise CalibrationInputError(f"segments for tier {tier_name} must be a list.")

        segments: list[CalibrationSegmentAggregate] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                raise CalibrationInputError("Each segment entry must be an object.")
            segment_name = str(seg.get("segment", ""))
            n_games = int(seg.get("n_games", 0))
            human_win_rate = float(seg.get("human_win_rate", 0.0))
            if n_games < 0:
                raise CalibrationInputError(
                    f"Segment {segment_name!r} in tier {tier_name} has negative "
                    f"n_games={n_games}."
                )
            if not (0.0 <= human_win_rate <= 1.0):
                raise CalibrationInputError(
                    f"Segment {segment_name!r} in tier {tier_name} has invalid "
                    f"human_win_rate={human_win_rate}."
                )

            difficulty_mean = float(seg.get("difficulty_mean", 0.0))
            difficulty_p10 = float(seg.get("difficulty_p10", 0.0))
            difficulty_p90 = float(seg.get("difficulty_p90", 0.0))

            segments.append(
                CalibrationSegmentAggregate(
                    segment=segment_name,
                    n_games=n_games,
                    human_win_rate=human_win_rate,
                    difficulty_mean=difficulty_mean,
                    difficulty_p10=difficulty_p10,
                    difficulty_p90=difficulty_p90,
                )
            )

        tiers.append(
            CalibrationTierAggregate(
                tier=tier_name,
                difficulty=difficulty,
                segments=segments,
            )
        )

    return CalibrationAggregateRoot(
        board=board,
        num_players=num_players,
        window=window,
        tiers=tiers,
    )


def _classify_segment(
    tier_name: str,
    segment: CalibrationSegmentAggregate,
    min_sample_size: int,
) -> tuple[bool, str]:
    """Return (sample_ok, status) for a calibration segment.

    status ∈ {"too_hard", "too_easy", "in_band", "inconclusive"}.
    """

    thresholds = TIER_THRESHOLDS.get(tier_name)
    if thresholds is None:
        # Fallback: treat as inconclusive if we have no thresholds configured.
        return False, "inconclusive"

    if segment.n_games < min_sample_size:
        return False, "inconclusive"

    win = segment.human_win_rate
    mean = segment.difficulty_mean

    # Too easy: human win rate well above target band and
    # low perceived difficulty.
    if win >= thresholds.too_easy_win and mean <= thresholds.easy_mean_max:
        return True, "too_easy"

    # Too hard: human win rate well below target band and
    # high perceived difficulty.
    if win <= thresholds.too_hard_win and mean >= thresholds.hard_mean_min:
        return True, "too_hard"

    in_band_win = thresholds.target_low <= win <= thresholds.target_high
    in_band_mean = 2.5 <= mean <= 3.5
    if in_band_win and in_band_mean:
        return True, "in_band"

    # Data is mixed or only weakly indicative.
    return True, "inconclusive"


def _build_ladder_view(cfg: LadderTierConfig) -> dict[str, Any]:
    if isinstance(cfg.ai_type, AIType):
        ai_type_value = cfg.ai_type.value
    else:
        ai_type_value = str(cfg.ai_type)
    return {
        "model_id": cfg.model_id,
        "ai_type": ai_type_value,
        "heuristic_profile_id": cfg.heuristic_profile_id,
    }


def _select_latest_promoted_candidate(
    tier_name: str,
    ladder_model_id: str | None,
    registry: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the most recent gated_promote candidate for the ladder model."""

    tiers = registry.get("tiers") or {}
    tier_block = tiers.get(tier_name)
    if not isinstance(tier_block, dict):
        return None

    candidates = tier_block.get("candidates") or []
    if not isinstance(candidates, list):
        return None

    # registry helpers append candidates, so iterate from the end
    # for "most recent".
    for entry in reversed(candidates):
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "gated_promote":
            continue
        if ladder_model_id and entry.get("model_id") != ladder_model_id:
            continue
        return entry

    return None


def _load_eval_and_perf_for_candidate(
    run_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Best-effort loading of tier_eval_result.json and perf reports."""

    eval_payload: dict[str, Any] | None = None
    perf_payload: dict[str, Any] | None = None

    eval_path = run_dir / "tier_eval_result.json"
    if eval_path.exists():
        try:
            eval_payload = _load_json_file(eval_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(
                "Warning: failed to load tier evaluation result from " f"{eval_path}: {exc}",
                file=sys.stderr,
            )
    else:
        print(
            f"Warning: tier_eval_result.json not found under {run_dir}",
            file=sys.stderr,
        )

    perf_path = run_dir / "tier_perf_report.json"
    if perf_path.exists():
        try:
            perf_payload = _load_json_file(perf_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(
                "Warning: failed to load tier perf report from " f"{perf_path}: {exc}",
                file=sys.stderr,
            )
    else:
        # Not all tiers have perf budgets; this is informational only.
        print(
            f"Warning: tier_perf_report.json not found under {run_dir}",
            file=sys.stderr,
        )

    # Attempt to load gate_report.json as a side-effect so missing artefacts
    # are surfaced to the caller, even though we don't currently expose
    # gate-level metrics in the calibration summary.
    gate_path = run_dir / "gate_report.json"
    if not gate_path.exists():
        print(
            f"Warning: gate_report.json not found under {run_dir}",
            file=sys.stderr,
        )

    return eval_payload, perf_payload


def _build_evaluation_view(
    eval_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not eval_payload:
        return None

    metrics = eval_payload.get("metrics") or {}
    win_vs_baseline = metrics.get("win_rate_vs_baseline")
    win_vs_prev = metrics.get("win_rate_vs_previous_tier")

    return {
        "overall_pass": eval_payload.get("overall_pass"),
        "win_rates": {
            "vs_baseline": win_vs_baseline,
            "vs_previous_tier": win_vs_prev,
        },
    }


def _build_perf_view(
    perf_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not perf_payload:
        return None

    metrics = perf_payload.get("metrics") or {}
    eval_block = perf_payload.get("evaluation") or {}

    return {
        "overall_pass": eval_block.get("overall_pass"),
        "avg_ms": metrics.get("average_ms"),
        "p95_ms": metrics.get("p95_ms"),
    }


def _summarise_tier(
    tier_agg: CalibrationTierAggregate,
    registry: dict[str, Any],
    eval_root: Path,
    min_sample_size: int,
) -> dict[str, Any]:
    """Join a single tier's calibration aggregates with ladder and registry."""

    tier_name = tier_agg.tier
    difficulty = tier_agg.difficulty

    ladder_cfg = get_ladder_tier_config(
        difficulty=difficulty,
        board_type=BoardType.SQUARE8,
        num_players=2,
    )
    ladder_view = _build_ladder_view(ladder_cfg)

    tiers = registry.get("tiers") or {}
    tier_block = tiers.get(tier_name)
    registry_current = None
    if isinstance(tier_block, dict):
        registry_current = tier_block.get("current")

    latest_candidate = _select_latest_promoted_candidate(
        tier_name=tier_name,
        ladder_model_id=ladder_cfg.model_id,
        registry=registry,
    )

    candidate_source_run_dir: str | None = None
    if latest_candidate is not None:
        candidate_source_run_dir = latest_candidate.get("source_run_dir")

    eval_view: dict[str, Any] | None = None
    perf_view: dict[str, Any] | None = None
    if candidate_source_run_dir:
        run_dir = Path(eval_root) / Path(candidate_source_run_dir)
        eval_payload, perf_payload = _load_eval_and_perf_for_candidate(run_dir)
        eval_view = _build_evaluation_view(eval_payload)
        perf_view = _build_perf_view(perf_payload)

    # Per-segment calibration status.
    segment_entries: list[dict[str, Any]] = []
    too_easy_segments: list[str] = []
    too_hard_segments: list[str] = []
    in_band_segments: list[str] = []
    inconclusive_segments: list[str] = []

    for seg in tier_agg.segments:
        sample_ok, status = _classify_segment(tier_name, seg, min_sample_size)
        entry = {
            "segment": seg.segment,
            "n_games": seg.n_games,
            "human_win_rate": seg.human_win_rate,
            "difficulty_mean": seg.difficulty_mean,
            "difficulty_p10": seg.difficulty_p10,
            "difficulty_p90": seg.difficulty_p90,
            "sample_ok": sample_ok,
            "status": status,
        }
        segment_entries.append(entry)

        if not sample_ok or status == "inconclusive":
            inconclusive_segments.append(seg.segment)
        elif status == "too_easy":
            too_easy_segments.append(seg.segment)
        elif status == "too_hard":
            too_hard_segments.append(seg.segment)
        elif status == "in_band":
            in_band_segments.append(seg.segment)

    # Overall calibration status per tier.
    overall_status: str
    if not tier_agg.segments or all(not entry["sample_ok"] for entry in segment_entries):
        overall_status = "inconclusive"
    elif too_easy_segments and not too_hard_segments:
        overall_status = "too_easy"
    elif too_hard_segments and not too_easy_segments:
        overall_status = "too_hard"
    elif too_easy_segments and too_hard_segments:
        overall_status = "mixed"
    elif in_band_segments and not (too_easy_segments or too_hard_segments):
        overall_status = "in_band"
    else:
        overall_status = "inconclusive"

    # Short human-readable notes.
    notes_parts: list[str] = []
    if too_easy_segments:
        segs = ", ".join(sorted(too_easy_segments))
        notes_parts.append(f"Segments {segs} appear too easy vs target bands.")
    if too_hard_segments:
        segs = ", ".join(sorted(too_hard_segments))
        notes_parts.append(f"Segments {segs} appear too hard vs target bands.")
    if in_band_segments and not (too_easy_segments or too_hard_segments):
        segs = ", ".join(sorted(in_band_segments))
        notes_parts.append(f"Segments {segs} sit within the target band.")
    if not notes_parts:
        if overall_status == "inconclusive":
            notes_parts.append("Insufficient or mixed data for tier " f"{tier_name}; treat as inconclusive.")
        else:
            notes_parts.append("Calibration status derived from current aggregates.")

    calibration_block = {
        "segments": segment_entries,
        "overall_status": overall_status,
        "notes": " ".join(notes_parts),
    }

    registry_view: dict[str, Any] = {}
    if registry_current is not None:
        registry_view["current"] = registry_current
    if latest_candidate is not None:
        registry_view["latest_candidate"] = {
            "candidate_id": latest_candidate.get("candidate_id") or latest_candidate.get("candidate_model_id"),
            "status": latest_candidate.get("status"),
            "source_run_dir": latest_candidate.get("source_run_dir"),
        }

    return {
        "tier": tier_name,
        "difficulty": difficulty,
        "ladder": ladder_view,
        "registry": registry_view,
        "evaluation": eval_view,
        "perf": perf_view,
        "calibration": calibration_block,
    }


def build_calibration_summary(
    aggregates: CalibrationAggregateRoot,
    registry: dict[str, Any],
    eval_root: str,
    window_label: str | None,
    min_sample_size: int,
) -> dict[str, Any]:
    """Construct the top-level summary JSON structure."""

    window = dict(aggregates.window or {})
    if window_label:
        window["label"] = window_label
    else:
        start = window.get("start")
        if isinstance(start, str) and len(start) >= 7:
            window.setdefault("label", start[:7])

    eval_root_path = Path(eval_root)

    tier_summaries: list[dict[str, Any]] = []
    for tier_agg in aggregates.tiers:
        tier_summaries.append(
            _summarise_tier(
                tier_agg=tier_agg,
                registry=registry,
                eval_root=eval_root_path,
                min_sample_size=min_sample_size,
            )
        )

    summary: dict[str, Any] = {
        "board": aggregates.board,
        "num_players": aggregates.num_players,
        "window": window,
        "tiers": tier_summaries,
    }
    return summary


def _format_bool_pass(value: bool | None) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "N/A"


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Render a concise Markdown report from the summary JSON payload."""

    board = summary.get("board", "square8")
    num_players = summary.get("num_players", 2)
    window = summary.get("window") or {}
    label = window.get("label")
    start = window.get("start")
    end = window.get("end")

    if label:
        window_str = str(label)
    elif start and end:
        window_str = f"{start} .. {end}"
    else:
        window_str = ""

    lines: list[str] = []
    header = f"## Calibration summary \u2013 {board} {num_players}-player"
    if window_str:
        header += f", window {window_str}"
    lines.append(header)
    lines.append("")

    for tier in summary.get("tiers", []):
        tier_name = tier.get("tier")
        lines.append(f"### Tier {tier_name}")
        lines.append("")

        ladder = tier.get("ladder") or {}
        model_id = ladder.get("model_id") or "N/A"
        ai_type = ladder.get("ai_type") or "N/A"
        heuristic_profile_id = ladder.get("heuristic_profile_id") or "N/A"
        lines.append("**Ladder model:** " f"{model_id} ({ai_type}, {heuristic_profile_id})  ")

        evaluation = tier.get("evaluation") or {}
        eval_overall = _format_bool_pass(evaluation.get("overall_pass"))
        win_rates = evaluation.get("win_rates") or {}
        vs_baseline = win_rates.get("vs_baseline")
        vs_prev = win_rates.get("vs_previous_tier")
        eval_parts = [f"overall_pass {eval_overall}"]
        if isinstance(vs_baseline, (int, float)):
            eval_parts.append(f"win vs baseline {vs_baseline:.2f}")
        if isinstance(vs_prev, (int, float)):
            eval_parts.append(f"win vs previous-tier {vs_prev:.2f}")
        lines.append(f"**Evaluation:** {', '.join(eval_parts)}  ")

        perf = tier.get("perf") or {}
        perf_overall = _format_bool_pass(perf.get("overall_pass"))
        avg_ms = perf.get("avg_ms")
        p95_ms = perf.get("p95_ms")
        perf_parts = [f"overall_pass {perf_overall}"]
        if isinstance(avg_ms, (int, float)):
            perf_parts.append(f"avg {avg_ms:.1f} ms")
        if isinstance(p95_ms, (int, float)):
            perf_parts.append(f"p95 {p95_ms:.1f} ms")
        lines.append(f"**Perf:** {', '.join(perf_parts)}  ")
        lines.append("")

        # Segment table.
        lines.append("| Segment | n_games | human_win_rate | diff_mean | p10 | " "p90 | sample_ok | status |")
        lines.append("|---------|---------|----------------|-----------|-----|" "-----|-----------|--------|")
        for seg in (tier.get("calibration") or {}).get("segments", []):
            segment_name = seg.get("segment")
            n_games = seg.get("n_games")
            win_rate = seg.get("human_win_rate")
            diff_mean = seg.get("difficulty_mean")
            p10 = seg.get("difficulty_p10")
            p90 = seg.get("difficulty_p90")
            sample_ok = "yes" if seg.get("sample_ok") else "no"
            status = seg.get("status") or "inconclusive"
            row = (
                f"| {segment_name} | {n_games} | {win_rate:.2f} | "
                f"{diff_mean:.2f} | {p10:.2f} | {p90:.2f} | "
                f"{sample_ok} | {status} |"
            )
            lines.append(row)

        lines.append("")
        calibration = tier.get("calibration") or {}
        overall_status = calibration.get("overall_status") or "inconclusive"
        notes = calibration.get("notes") or ""
        lines.append(f"**Overall calibration status:** {overall_status}")
        lines.append("")
        lines.append("**Notes:**")
        if notes:
            lines.append(f"- {notes}")
        else:
            lines.append("- (none)")
        lines.append("")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Square-8 2-player difficulty calibration aggregates and "
            "join them with ladder, registry, and tier eval/perf artefacts."
        ),
    )

    parser.add_argument(
        "--calibration-aggregates",
        required=True,
        help=("Path to calibration aggregates JSON file exported from " "metrics or logs."),
    )

    parser.add_argument(
        "--registry-path",
        default=DEFAULT_SQUARE8_2P_REGISTRY_PATH,
        help=("Path to the Square-8 2-player tier candidate registry JSON " "(default: %(default)s)."),
    )

    parser.add_argument(
        "--eval-root",
        default=".",
        help=(
            "Root directory under which tier eval/gate runs are stored. "
            "Candidate source_run_dir values from the registry are joined "
            "against this root (default: current directory)."
        ),
    )

    parser.add_argument(
        "--output-json",
        default="calibration_summary.square8_2p.json",
        help=("Path to write the machine-readable calibration summary JSON " "(default: %(default)s)."),
    )

    parser.add_argument(
        "--output-md",
        default="calibration_summary.square8_2p.md",
        help=("Optional path to write a human-readable Markdown summary " "(default: %(default)s)."),
    )

    parser.add_argument(
        "--window-label",
        default=None,
        help=(
            "Optional label for the calibration window (for example "
            '"2025-11"). If omitted, derived from the window.start value '
            "when available."
        ),
    )

    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=30,
        help=("Minimum n_games required for a segment to be treated as " "well-sampled (default: %(default)s)."),
    )

    return parser.parse_args(argv)


def _ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        aggregates = load_calibration_aggregates(args.calibration_aggregates)
    except CalibrationInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(
            "Unexpected error while loading calibration aggregates: " f"{exc}",
            file=sys.stderr,
        )
        return 1

    registry = load_square8_two_player_registry(path=args.registry_path)
    summary = build_calibration_summary(
        aggregates=aggregates,
        registry=registry,
        eval_root=args.eval_root,
        window_label=args.window_label,
        min_sample_size=args.min_sample_size,
    )

    # Write JSON summary.
    json_path = Path(args.output_json)
    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path
    try:
        _ensure_parent_dir(json_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Calibration summary JSON written to: {json_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"Error writing JSON summary to {json_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    # Write Markdown summary if a path is provided.
    if args.output_md:
        md_path = Path(args.output_md)
        if not md_path.is_absolute():
            md_path = Path.cwd() / md_path
        try:
            _ensure_parent_dir(md_path)
            markdown = build_markdown_report(summary)
            md_path.write_text(markdown, encoding="utf-8")
            print(f"Calibration summary Markdown written to: {md_path}")
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"Error writing Markdown summary to {md_path}: {exc}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
