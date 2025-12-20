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
import contextlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure `app.*` imports work regardless of the current working directory.
AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))


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


def _merge_breakdown_dict(target: dict[str, Any], source: dict[str, Any] | None) -> None:
    """Merge analyzer "breakdown" dicts keyed by config value.

    Expected payload shape:
      { "<key>": {"games": int, "total_moves": int, "victory_types": { "<vt>": int, ... } }, ... }
    """
    if not source:
        return
    if not isinstance(source, dict):
        return

    for key, payload in source.items():
        if not isinstance(payload, dict):
            continue
        out = target.setdefault(key, {"games": 0, "total_moves": 0, "victory_types": {}})
        with contextlib.suppress(TypeError, ValueError):
            out["games"] = int(out.get("games", 0) or 0) + int(payload.get("games", 0) or 0)
        with contextlib.suppress(TypeError, ValueError):
            out["total_moves"] = int(out.get("total_moves", 0) or 0) + int(payload.get("total_moves", 0) or 0)

        vt_out = out.setdefault("victory_types", {})
        vt_in = payload.get("victory_types", {}) or {}
        if isinstance(vt_in, dict) and isinstance(vt_out, dict):
            for vt, count in vt_in.items():
                try:
                    vt_out[vt] = int(vt_out.get(vt, 0) or 0) + int(count or 0)
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

    int(victory_types.get("stalemate", 0) or 0)
    stalemate_territory = int(stalemate_by_tb.get("territory", 0) or 0)
    stalemate_ring_elim = int(stalemate_by_tb.get("ring_elimination", 0) or 0)

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
                    "starting_rings_per_player_breakdown": {},
                    "victory_threshold_breakdown": {},
                    "games_with_starting_rings_mismatch": 0,
                    "games_with_victory_threshold_mismatch": 0,
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
            out["games_with_starting_rings_mismatch"] += int(
                cfg.get("games_with_starting_rings_mismatch", 0) or 0
            )
            out["games_with_victory_threshold_mismatch"] += int(
                cfg.get("games_with_victory_threshold_mismatch", 0) or 0
            )

            # Float fields
            with contextlib.suppress(TypeError, ValueError):
                out["total_time_seconds"] += float(cfg.get("total_time_seconds", 0.0) or 0.0)

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
            _merge_breakdown_dict(
                out["starting_rings_per_player_breakdown"], cfg.get("starting_rings_per_player_breakdown")
            )
            _merge_breakdown_dict(
                out["victory_threshold_breakdown"], cfg.get("victory_threshold_breakdown")
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
    try:
        from app.models.core import BoardType
        from app.rules.core import BOARD_CONFIGS, get_victory_threshold

        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        canonical_available = True
    except Exception:
        canonical_available = False
        board_type_map = {}
        BOARD_CONFIGS = {}
        get_victory_threshold = None  # type: ignore[assignment]

    lines: list[str] = []
    summary = report.get("summary", {}) or {}
    lines.append("# RingRift Combined Selfplay Summary")
    lines.append("")
    lines.append(f"**Generated:** {report.get('timestamp', '')}")
    lines.append(f"**Total Games:** {summary.get('total_games', 0):,}")
    lines.append(f"**Total Moves:** {summary.get('total_moves', 0):,}")
    lines.append(f"**Data Sources:** {summary.get('data_sources', 0):,}")
    lines.append("")

    lines.append("## Outcomes (victory_type)")
    lines.append("")
    lines.append("| Config | Games | Avg moves | LPS | Elim | Terr | Stalemate | Timeout | Draw | Unknown |")
    lines.append("|--------|------:|----------:|-----|------|------|----------|--------:|-----:|--------:|")
    for cfg_key, cfg in (report.get("configurations", {}) or {}).items():
        total_games = int(cfg.get("total_games", 0) or 0)
        if total_games <= 0:
            continue
        total_moves = int(cfg.get("total_moves", 0) or 0)
        avg_moves = total_moves / total_games if total_games else 0.0
        vtypes = cfg.get("victory_types", {}) or {}
        lps = int(vtypes.get("lps", 0) or 0)
        terr = int(vtypes.get("territory", 0) or 0)
        elim = int(vtypes.get("elimination", 0) or 0) + int(vtypes.get("ring_elimination", 0) or 0)
        stalemate = int(vtypes.get("stalemate", 0) or 0)
        timeout = int(vtypes.get("timeout", 0) or 0)
        draw = int(vtypes.get("draw", 0) or 0)
        unknown = int(vtypes.get("unknown", 0) or 0)
        lines.append(
            f"| {cfg_key} | {total_games} | {avg_moves:.1f} | "
            f"{100*lps/total_games:.1f}% ({lps}) | "
            f"{100*elim/total_games:.1f}% ({elim}) | "
            f"{100*terr/total_games:.1f}% ({terr}) | "
            f"{100*stalemate/total_games:.1f}% ({stalemate}) | "
            f"{100*timeout/total_games:.1f}% ({timeout}) | "
            f"{draw} | {unknown} |"
        )
    lines.append("")

    lines.append("## Config Drift (ring supply)")
    lines.append("")
    lines.append("| Config | Observed ringsPerPlayer | Expected ringsPerPlayer | Observed victoryThreshold | Expected victoryThreshold |")
    lines.append("|--------|------------------------|-------------------------|--------------------------|---------------------------|")
    for cfg_key, cfg in (report.get("configurations", {}) or {}).items():
        total_games = int(cfg.get("total_games", 0) or 0)
        if total_games <= 0:
            continue

        observed_rings = cfg.get("starting_rings_per_player_counts", {}) or {}
        observed_threshold = cfg.get("victory_threshold_counts", {}) or {}

        board_type = str(cfg.get("board_type", "unknown"))
        num_players = int(cfg.get("num_players", 2) or 2)
        expected_rings = ""
        expected_threshold = ""

        if canonical_available and board_type in board_type_map:
            bt = board_type_map[board_type]
            expected_rings = str(BOARD_CONFIGS[bt].rings_per_player) if bt in BOARD_CONFIGS else ""
            expected_threshold = str(get_victory_threshold(bt, num_players)) if get_victory_threshold else ""

        lines.append(
            f"| {cfg_key} | {json.dumps(observed_rings, sort_keys=True)} | "
            f"{expected_rings or '-'} | {json.dumps(observed_threshold, sort_keys=True)} | "
            f"{expected_threshold or '-'} |"
        )
    lines.append("")

    lines.append("## Recovery (recovery_slide)")
    lines.append("")
    lines.append(
        "| Config | Games | Games w/ recovery_slide | Games w/ stack_strike | recovery_slide modes (move counts) |"
    )
    lines.append("|--------|------:|-----------------------:|----------------------:|-----------------------------------|")
    for cfg_key, cfg in (report.get("configurations", {}) or {}).items():
        total_games = int(cfg.get("total_games", 0) or 0)
        if total_games <= 0:
            continue
        games_with_recovery = int(cfg.get("games_with_recovery_slide", 0) or 0)
        games_with_stack_strike = int(cfg.get("games_with_stack_strike", 0) or 0)
        modes = cfg.get("recovery_slides_by_mode", {}) or {}
        try:
            modes_str = ", ".join(f"{k}:{int(v)}" for k, v in sorted(modes.items()))
        except Exception:
            modes_str = json.dumps(modes, sort_keys=True)

        lines.append(
            f"| {cfg_key} | {total_games} | "
            f"{games_with_recovery} ({100*games_with_recovery/total_games:.1f}%) | "
            f"{games_with_stack_strike} ({100*games_with_stack_strike/total_games:.1f}%) | "
            f"{modes_str} |"
        )
    lines.append("")

    # If configs are mixed, show outcome breakdown by observed ringsPerPlayer.
    any_mixed = any(
        len((cfg.get("starting_rings_per_player_breakdown", {}) or {}).keys()) > 1
        for cfg in (report.get("configurations", {}) or {}).values()
    )
    if any_mixed:
        lines.append("## Outcomes By ringsPerPlayer (mixed configs)")
        lines.append("")
        lines.append("| Config | ringsPerPlayer | Games | Avg moves | Stalemate | LPS | Elim | Terr | Timeout | Unknown |")
        lines.append("|--------|--------------:|------:|----------:|----------:|----:|-----:|-----:|--------:|--------:|")

        for cfg_key, cfg in (report.get("configurations", {}) or {}).items():
            breakdown = cfg.get("starting_rings_per_player_breakdown", {}) or {}
            if not isinstance(breakdown, dict) or len(breakdown.keys()) <= 1:
                continue

            rows: list[tuple[int, str, dict[str, Any]]] = []
            for rings_key, payload in breakdown.items():
                if not isinstance(payload, dict):
                    continue
                games = int(payload.get("games", 0) or 0)
                rows.append((games, str(rings_key), payload))

            for games, rings_key, payload in sorted(rows, key=lambda t: (-t[0], t[1]))[:5]:
                if games <= 0:
                    continue
                total_moves = int(payload.get("total_moves", 0) or 0)
                avg_moves = total_moves / games if games else 0.0
                vtypes = payload.get("victory_types", {}) or {}
                if not isinstance(vtypes, dict):
                    vtypes = {}

                stalemate = int(vtypes.get("stalemate", 0) or 0)
                timeout = int(vtypes.get("timeout", 0) or 0)
                unknown = int(vtypes.get("unknown", 0) or 0)
                lps = int(vtypes.get("lps", 0) or 0)
                terr = int(vtypes.get("territory", 0) or 0)
                elim = int(vtypes.get("elimination", 0) or 0) + int(vtypes.get("ring_elimination", 0) or 0)

                lines.append(
                    f"| {cfg_key} | {rings_key} | {games} | {avg_moves:.1f} | "
                    f"{100*stalemate/games:.1f}% ({stalemate}) | "
                    f"{100*lps/games:.1f}% ({lps}) | "
                    f"{100*elim/games:.1f}% ({elim}) | "
                    f"{100*terr/games:.1f}% ({terr}) | "
                    f"{100*timeout/games:.1f}% ({timeout}) | "
                    f"{100*unknown/games:.1f}% ({unknown}) |"
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
