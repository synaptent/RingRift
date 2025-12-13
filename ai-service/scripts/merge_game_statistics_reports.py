#!/usr/bin/env python3
"""Merge JSON reports produced by analyze_game_statistics.py.

This is useful for distributed selfplay setups where each host analyzes its own
last-24h JSONL files locally (fast, low bandwidth), then the coordinator merges
the resulting small JSON summaries into a combined view.

Example:
  python scripts/merge_game_statistics_reports.py \\
    --inputs logs/selfplay/collected_last24h/20251212_181004/reports/*.json \\
    --output logs/selfplay/collected_last24h/20251212_181004/combined.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_input_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for p in inputs:
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        else:
            paths.append(p)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    result: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def _merge_counter_dict(target: dict[str, int], source: dict[str, Any] | None) -> None:
    if not source:
        return
    for k, v in source.items():
        try:
            target[k] = target.get(k, 0) + int(v)
        except (TypeError, ValueError):
            continue


def _config_key(board_type: str, num_players: int) -> str:
    return f"{board_type}_{num_players}p"


def _derive_aggregated_victory_types(cfg: dict[str, Any]) -> dict[str, int]:
    victory_types = cfg.get("victory_types", {}) or {}
    stalemate_by_tb = cfg.get("stalemate_by_tiebreaker", {}) or {}

    territory = int(victory_types.get("territory", 0) or 0)
    elimination = int(victory_types.get("elimination", 0) or 0)
    ring_elim = int(victory_types.get("ring_elimination", 0) or 0)
    lps = int(victory_types.get("lps", 0) or 0)

    stalemate_total = int(victory_types.get("stalemate", 0) or 0)
    stalemate_territory = int(stalemate_by_tb.get("territory", 0) or 0)
    stalemate_ring_elim = int(stalemate_by_tb.get("ring_elimination", 0) or 0)

    if stalemate_total > 0 and not stalemate_by_tb:
        stalemate_territory = stalemate_total

    return {
        "territory": territory + stalemate_territory,
        "elimination": elimination + ring_elim + stalemate_ring_elim,
        "lps": lps,
    }


def merge_reports(report_paths: list[Path], *, strict: bool) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "timestamp": _now_iso(),
        "summary": {"total_games": 0, "total_moves": 0, "data_sources": 0},
        "configurations": {},
        "recovery_analysis": {},
        "inputs": [],
        "merge_warnings": [],
    }

    by_config: dict[tuple[str, int], dict[str, Any]] = {}

    for path in report_paths:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                raise ValueError("empty file")
            report = json.loads(raw)
        except Exception as e:
            msg = f"Skipping invalid report {path}: {e}"
            if strict:
                raise
            merged["merge_warnings"].append(msg)
            continue

        summary = report.get("summary", {}) or {}
        total_games = int(summary.get("total_games", 0) or 0)
        total_moves = int(summary.get("total_moves", 0) or 0)
        data_sources = int(summary.get("data_sources", 0) or 0)
        merged["summary"]["total_games"] += total_games
        merged["summary"]["total_moves"] += total_moves
        merged["summary"]["data_sources"] += data_sources
        merged["inputs"].append(
            {
                "path": str(path),
                "total_games": total_games,
                "total_moves": total_moves,
                "data_sources": data_sources,
            }
        )

        for _, cfg in (report.get("configurations", {}) or {}).items():
            board_type = str(cfg.get("board_type", "unknown"))
            num_players = int(cfg.get("num_players", 2) or 2)
            key = (board_type, num_players)

            if key not in by_config:
                by_config[key] = {
                    "board_type": board_type,
                    "num_players": num_players,
                    "total_games": 0,
                    "total_moves": 0,
                    "total_time_seconds": 0.0,
                    "wins_by_player": {},
                    "victory_types": {},
                    "stalemate_by_tiebreaker": {},
                    "draws": 0,
                    "games_with_recovery": 0,
                    "games_with_fe": 0,
                    "games_with_recovery_slide": 0,
                    "recovery_slides_by_player": {},
                    "wins_with_recovery_slide": 0,
                    "wins_without_recovery_slide": 0,
                    "games_with_late_fe_winner": 0,
                    "fe_by_player": {},
                    "move_type_counts": {},
                    "total_captures": 0,
                    "total_chain_captures": 0,
                    "ring_placement_moves": 0,
                    "territory_claims": 0,
                    "line_formations": 0,
                    "first_capture_by_player": {},
                    "first_capturer_wins": 0,
                    "first_capturer_loses": 0,
                    "starting_rings_per_player_counts": {},
                    "victory_threshold_counts": {},
                    "territory_victory_threshold_counts": {},
                    "lps_rounds_required_counts": {},
                    "recovery_slides_by_mode": {},
                    "games_with_stack_strike": 0,
                    "wins_with_stack_strike": 0,
                    "timeout_move_count_hist": {},
                }

            out = by_config[key]
            out["total_games"] += int(cfg.get("total_games", 0) or 0)
            out["total_moves"] += int(cfg.get("total_moves", 0) or 0)
            out["draws"] += int(cfg.get("draws", 0) or 0)
            out["games_with_recovery"] += int(cfg.get("games_with_recovery", 0) or 0)
            out["games_with_fe"] += int(cfg.get("games_with_fe", 0) or 0)

            out["games_with_recovery_slide"] += int(cfg.get("games_with_recovery_slide", 0) or 0)
            out["wins_with_recovery_slide"] += int(cfg.get("wins_with_recovery_slide", 0) or 0)
            out["wins_without_recovery_slide"] += int(cfg.get("wins_without_recovery_slide", 0) or 0)
            out["games_with_late_fe_winner"] += int(cfg.get("games_with_late_fe_winner", 0) or 0)
            out["games_with_stack_strike"] += int(cfg.get("games_with_stack_strike", 0) or 0)
            out["wins_with_stack_strike"] += int(cfg.get("wins_with_stack_strike", 0) or 0)

            # Float fields
            try:
                out["total_time_seconds"] += float(cfg.get("total_time_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass

            # Counter dicts
            _merge_counter_dict(out["wins_by_player"], cfg.get("wins_by_player"))
            _merge_counter_dict(out["victory_types"], cfg.get("victory_types"))
            _merge_counter_dict(out["stalemate_by_tiebreaker"], cfg.get("stalemate_by_tiebreaker"))
            _merge_counter_dict(out["recovery_slides_by_player"], cfg.get("recovery_slides_by_player"))
            _merge_counter_dict(out["fe_by_player"], cfg.get("fe_by_player"))
            _merge_counter_dict(out["move_type_counts"], cfg.get("move_type_counts"))
            _merge_counter_dict(out["first_capture_by_player"], cfg.get("first_capture_by_player"))
            _merge_counter_dict(
                out["starting_rings_per_player_counts"], cfg.get("starting_rings_per_player_counts")
            )
            _merge_counter_dict(out["victory_threshold_counts"], cfg.get("victory_threshold_counts"))
            _merge_counter_dict(
                out["territory_victory_threshold_counts"], cfg.get("territory_victory_threshold_counts")
            )
            _merge_counter_dict(
                out["lps_rounds_required_counts"], cfg.get("lps_rounds_required_counts")
            )
            _merge_counter_dict(out["recovery_slides_by_mode"], cfg.get("recovery_slides_by_mode"))
            _merge_counter_dict(out["timeout_move_count_hist"], cfg.get("timeout_move_count_hist"))

            # Scalar counters
            for scalar in [
                "total_captures",
                "total_chain_captures",
                "ring_placement_moves",
                "territory_claims",
                "line_formations",
                "first_capturer_wins",
                "first_capturer_loses",
            ]:
                try:
                    out[scalar] += int(cfg.get(scalar, 0) or 0)
                except (TypeError, ValueError):
                    continue

    # Finalize merged configurations with derived rates.
    merged_configs: dict[str, Any] = {}
    for (board_type, num_players), cfg in sorted(by_config.items()):
        total_games = int(cfg.get("total_games", 0) or 0)
        total_moves = int(cfg.get("total_moves", 0) or 0)
        total_time_seconds = float(cfg.get("total_time_seconds", 0.0) or 0.0)

        cfg["moves_per_game"] = (total_moves / total_games) if total_games else 0.0
        cfg["games_per_second"] = (total_games / total_time_seconds) if total_time_seconds else 0.0

        wins_by_player = cfg.get("wins_by_player", {}) or {}
        cfg["win_rates"] = {
            str(p): (int(wins_by_player.get(str(p), 0) or 0) / total_games if total_games else 0.0)
            for p in range(1, num_players + 1)
        }

        victory_types = cfg.get("victory_types", {}) or {}
        cfg["victory_type_rates"] = {
            k: (int(v) / total_games if total_games else 0.0)
            for k, v in victory_types.items()
            if isinstance(v, (int, float))
        }

        cfg["aggregated_victory_types"] = _derive_aggregated_victory_types(cfg)

        merged_configs[_config_key(board_type, num_players)] = cfg

    merged["configurations"] = merged_configs
    return merged


def _format_markdown_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = report.get("summary", {}) or {}
    lines.append("# RingRift Combined Selfplay Summary")
    lines.append("")
    lines.append(f"**Generated:** {report.get('timestamp', '')}")
    lines.append(f"**Total Games:** {summary.get('total_games', 0):,}")
    lines.append(f"**Total Moves:** {summary.get('total_moves', 0):,}")
    lines.append(f"**Data Sources:** {summary.get('data_sources', 0):,}")
    lines.append("")

    lines.append("## Aggregated Victory Categories")
    lines.append("")
    lines.append("| Config | Games | LPS | Elimination | Territory |")
    lines.append("|--------|-------|-----|-------------|-----------|")
    for cfg_key, cfg in (report.get("configurations", {}) or {}).items():
        total_games = int(cfg.get("total_games", 0) or 0)
        if total_games <= 0:
            continue
        agg = cfg.get("aggregated_victory_types") or _derive_aggregated_victory_types(cfg)
        lps = int(agg.get("lps", 0) or 0)
        elim = int(agg.get("elimination", 0) or 0)
        terr = int(agg.get("territory", 0) or 0)
        lines.append(
            f"| {cfg_key} | {total_games} | "
            f"{100*lps/total_games:.1f}% ({lps}) | "
            f"{100*elim/total_games:.1f}% ({elim}) | "
            f"{100*terr/total_games:.1f}% ({terr}) |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="JSON report files (or directories containing *.json).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for merged JSON (default: stdout).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid/empty input reports instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_paths = _iter_input_paths(args.inputs)
    if not report_paths:
        print("Error: No input reports found", file=sys.stderr)
        return 1

    merged = merge_reports(report_paths, strict=args.strict)

    if args.format in ("json", "both"):
        if args.output:
            out_path = args.output if args.format == "json" else args.output.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        else:
            print(json.dumps(merged, indent=2))

    if args.format in ("markdown", "both"):
        md = _format_markdown_summary(merged)
        if args.output:
            md_path = args.output if args.format == "markdown" else args.output.with_suffix(".md")
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md, encoding="utf-8")
        else:
            print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
