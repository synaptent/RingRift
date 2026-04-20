#!/usr/bin/env python3
"""Characterize whether training configs are actually plateaued.

Reads ``metrics.jsonl`` files for all active trainer configs and computes:

* **eval_wr trajectory** — per-iteration candidate win rate. A true plateau
  shows near-zero slope over the last N iters.
* **eval_wr variance** — stddev within the last N iters. Low variance +
  near-threshold mean = "stuck at the decision boundary" signature.
* **promotion rate** — promotions per iter over the last N iters.
* **selfplay seat WR variance** — how lopsided the in-iteration seat win
  distribution is. If the game has natural seat asymmetry, variance is
  roughly constant; if training is distorting the value head, variance
  may drift.
* **policy entropy proxy** — uses MCTS visit-distribution data when
  available in ``iter_N.jsonl`` to detect distribution narrowing (a
  symptom of self-play collapse).

Output: JSON summary to stdout + optional markdown table. Read-only;
does not touch training state.

Usage::

    python3 scripts/experiments/plateau_characterization.py \\
        --metrics-dir /tmp/ringrift_metrics \\
        --window 10 \\
        --format markdown

Design: standalone, no dependencies beyond the Python stdlib. Safe to
run locally on any node or the coordinator without installing anything.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ConfigSummary:
    config: str
    iters_recorded: int
    last_iter: int | None
    last_elo: float | None
    last_promos: int | None
    window_iters: int
    eval_wr_mean: float | None
    eval_wr_std: float | None
    eval_wr_slope_per_iter: float | None
    promotion_rate_in_window: float | None
    rejection_streak_current: int
    rejection_streak_max: int
    seat_wr_variance_mean: float | None
    seat_wr_variance_std: float | None
    selfplay_wr_near_threshold: bool
    plateau_signature_flags: list[str]


def _linear_slope(ys: list[float]) -> float | None:
    """Compute slope of a simple best-fit line over integer x indices."""
    n = len(ys)
    if n < 2:
        return None
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def _seat_wr_variance(selfplay: dict[str, Any]) -> float | None:
    """Compute variance of per-seat WR in a selfplay block.

    If the block contains p1_wins/p2_wins/p3_wins/p4_wins style keys,
    derive per-seat WR as wins/completed and return the variance.
    """
    wins = {}
    for k, v in selfplay.items():
        if k.startswith("p") and k.endswith("_wins"):
            try:
                wins[k] = int(v)
            except (TypeError, ValueError):
                continue
    total = sum(wins.values())
    if not wins or total == 0:
        return None
    wrs = [w / total for w in wins.values()]
    if len(wrs) < 2:
        return None
    return statistics.pvariance(wrs)


def _plateau_flags(
    eval_wr_std: float | None,
    eval_wr_slope: float | None,
    eval_wr_mean: float | None,
    promotion_rate: float | None,
    rejection_streak: int,
) -> list[str]:
    flags: list[str] = []
    # Tight distribution near a threshold
    if eval_wr_std is not None and eval_wr_std < 0.02:
        flags.append("LOW_EVAL_VARIANCE")  # very tight WR distribution
    # Flat trend
    if eval_wr_slope is not None and abs(eval_wr_slope) < 0.005:
        flags.append("FLAT_TREND")  # <0.5% WR change per iter
    # No promotions
    if promotion_rate is not None and promotion_rate == 0:
        flags.append("NO_PROMOTIONS")
    # Stuck right at decision boundary (2p promote threshold is 0.55)
    if (
        eval_wr_mean is not None
        and 0.44 <= eval_wr_mean <= 0.50
        and eval_wr_std is not None
        and eval_wr_std < 0.025
    ):
        flags.append("STUCK_AT_2P_THRESHOLD")
    if rejection_streak >= 3:
        flags.append(f"REJECTION_STREAK_{rejection_streak}")
    return flags


def analyze_metrics_file(path: Path, window: int) -> ConfigSummary:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue

    rows.sort(key=lambda r: r.get("iteration", 0))

    config = path.stem.replace("_metrics", "")
    iters_recorded = len(rows)
    last = rows[-1] if rows else None

    window_rows = rows[-window:]
    eval_wrs = [
        r.get("evaluation", {}).get("win_rate")
        for r in window_rows
        if isinstance(r.get("evaluation"), dict)
        and r.get("evaluation", {}).get("win_rate") is not None
    ]
    eval_wrs = [float(w) for w in eval_wrs if isinstance(w, (int, float))]
    eval_wr_mean = statistics.mean(eval_wrs) if eval_wrs else None
    eval_wr_std = statistics.stdev(eval_wrs) if len(eval_wrs) >= 2 else None
    eval_wr_slope = _linear_slope(eval_wrs) if len(eval_wrs) >= 2 else None

    promoted_count = sum(1 for r in window_rows if r.get("promoted"))
    promotion_rate = promoted_count / len(window_rows) if window_rows else None

    # Rejection streak from the tail
    current_streak = 0
    for r in reversed(rows):
        if r.get("promoted"):
            break
        current_streak += 1
    max_streak = 0
    running = 0
    for r in rows:
        if r.get("promoted"):
            max_streak = max(max_streak, running)
            running = 0
        else:
            running += 1
    max_streak = max(max_streak, running)

    seat_variances = [
        v
        for v in (_seat_wr_variance(r.get("selfplay", {}) or {}) for r in window_rows)
        if v is not None
    ]
    seat_mean = statistics.mean(seat_variances) if seat_variances else None
    seat_std = statistics.stdev(seat_variances) if len(seat_variances) >= 2 else None

    near_threshold = (
        eval_wr_mean is not None
        and 0.44 <= eval_wr_mean <= 0.50
        and eval_wr_std is not None
        and eval_wr_std < 0.03
    )

    flags = _plateau_flags(
        eval_wr_std, eval_wr_slope, eval_wr_mean, promotion_rate, current_streak
    )

    return ConfigSummary(
        config=config,
        iters_recorded=iters_recorded,
        last_iter=last.get("iteration") if last else None,
        last_elo=last.get("estimated_elo") if last else None,
        last_promos=last.get("total_promotions") if last else None,
        window_iters=len(window_rows),
        eval_wr_mean=round(eval_wr_mean, 4) if eval_wr_mean is not None else None,
        eval_wr_std=round(eval_wr_std, 4) if eval_wr_std is not None else None,
        eval_wr_slope_per_iter=round(eval_wr_slope, 5) if eval_wr_slope is not None else None,
        promotion_rate_in_window=round(promotion_rate, 3) if promotion_rate is not None else None,
        rejection_streak_current=current_streak,
        rejection_streak_max=max_streak,
        seat_wr_variance_mean=round(seat_mean, 5) if seat_mean is not None else None,
        seat_wr_variance_std=round(seat_std, 5) if seat_std is not None else None,
        selfplay_wr_near_threshold=near_threshold,
        plateau_signature_flags=flags,
    )


def format_markdown(summaries: list[ConfigSummary]) -> str:
    header = (
        "| config | iters | last_elo | promos | last_N_mean_wr | last_N_std | "
        "slope/iter | promo_rate | rej_streak | plateau_flags |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for s in summaries:
        rows.append(
            f"| {s.config} | {s.iters_recorded} | "
            f"{s.last_elo if s.last_elo is not None else '?':>4} | "
            f"{s.last_promos} | "
            f"{s.eval_wr_mean if s.eval_wr_mean is not None else '?'} | "
            f"{s.eval_wr_std if s.eval_wr_std is not None else '?'} | "
            f"{s.eval_wr_slope_per_iter if s.eval_wr_slope_per_iter is not None else '?'} | "
            f"{s.promotion_rate_in_window if s.promotion_rate_in_window is not None else '?'} | "
            f"{s.rejection_streak_current} (max {s.rejection_streak_max}) | "
            f"{', '.join(s.plateau_signature_flags) or '—'} |"
        )
    return header + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        default="/tmp/ringrift_metrics",
        help="Directory containing per-config metrics.jsonl files (named <config>_metrics.jsonl)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Iteration window for trend/variance analysis (default: 10)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Output format",
    )
    args = parser.parse_args()

    mdir = Path(args.metrics_dir)
    if not mdir.is_dir():
        raise SystemExit(f"metrics dir not found: {mdir}")

    files = sorted(mdir.glob("*_metrics.jsonl"))
    if not files:
        raise SystemExit(f"no *_metrics.jsonl files in {mdir}")

    summaries = [analyze_metrics_file(p, args.window) for p in files]

    if args.format == "json":
        print(json.dumps([asdict(s) for s in summaries], indent=2))
    else:
        print(f"# Plateau characterization (window={args.window} iters)\n")
        print(format_markdown(summaries))
        print()
        # Summary
        plateaued = [s for s in summaries if s.selfplay_wr_near_threshold]
        if plateaued:
            print(f"## Configs showing plateau signature: {len(plateaued)}/{len(summaries)}")
            for s in plateaued:
                print(
                    f"- **{s.config}**: eval_wr mean={s.eval_wr_mean}, "
                    f"std={s.eval_wr_std}, slope={s.eval_wr_slope_per_iter}/iter, "
                    f"rejections={s.rejection_streak_current}"
                )


if __name__ == "__main__":
    main()
