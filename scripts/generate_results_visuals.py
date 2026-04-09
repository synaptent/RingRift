#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


COLORS = {
    "hex8_2p": "#1b5e20",
    "square8_2p": "#0d47a1",
    "square8_3p": "#b26a00",
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _headline_svg(snapshot: dict) -> str:
    items = snapshot["headline"]
    width = 860
    height = 360
    margin_left = 90
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_elo = max(item["best_elo"] for item in items)
    min_elo = 1400
    scale = chart_h / (max_elo - min_elo)
    bar_width = 140
    gap = 70

    bars = []
    labels = []
    y_ticks = []
    for idx, tick in enumerate(range(1400, 2101, 100)):
        y = margin_top + chart_h - (tick - min_elo) * scale
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="#d7dee7" stroke-width="1" />'
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#4a5568">{tick}</text>'
        )

    for idx, item in enumerate(items):
        x = margin_left + gap + idx * (bar_width + gap)
        bar_h = (item["best_elo"] - min_elo) * scale
        y = margin_top + chart_h - bar_h
        color = COLORS.get(item["config"], "#455a64")
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" '
            f'rx="8" fill="{color}" opacity="0.92" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2}" y="{margin_top + chart_h + 24}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#1f2933">{item["config"]}</text>'
            f'<text x="{x + bar_width / 2}" y="{margin_top + chart_h + 44}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="13" fill="#52606d">'
            f'{item["best_elo"]:.1f} Elo · {item["promotions"]} promotions</text>'
            f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="{color}">{item["best_elo"]:.1f}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="26" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#102a43">
    RingRift headline results
  </text>
  <text x="{margin_left}" y="44" font-family="Arial, sans-serif" font-size="13" fill="#52606d">
    Snapshot as of {snapshot["as_of"]}. Higher bars indicate stronger best reported Elo.
  </text>
  {''.join(y_ticks)}
  <line x1="{margin_left}" y1="{margin_top + chart_h:.1f}" x2="{width - margin_right}" y2="{margin_top + chart_h:.1f}" stroke="#7b8794" stroke-width="2" />
  {''.join(bars)}
  {''.join(labels)}
</svg>
"""


def _progression_svg(snapshot: dict) -> str:
    series = snapshot["square8_2p_progression"]
    points = series["points"]
    width = 860
    height = 360
    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 65
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    min_iter = min(point["iteration"] for point in points)
    max_iter = max(point["iteration"] for point in points)
    min_elo = min(point["elo"] for point in points) - 10
    max_elo = max(point["elo"] for point in points) + 20

    def x_at(iteration: int) -> float:
        if max_iter == min_iter:
            return margin_left + chart_w / 2
        return margin_left + (iteration - min_iter) * chart_w / (max_iter - min_iter)

    def y_at(elo: float) -> float:
        return margin_top + chart_h - (elo - min_elo) * chart_h / (max_elo - min_elo)

    tick_lines = []
    for tick in range(int(min_elo // 25 * 25), int(max_elo // 25 * 25) + 26, 25):
        if tick < min_elo or tick > max_elo:
            continue
        y = y_at(tick)
        tick_lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e4e7eb" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#52606d">{tick}</text>'
        )

    path = " ".join(
        ("M" if idx == 0 else "L") + f" {x_at(point['iteration']):.1f} {y_at(point['elo']):.1f}"
        for idx, point in enumerate(points)
    )
    markers = []
    labels = []
    for point in points:
        x = x_at(point["iteration"])
        y = y_at(point["elo"])
        promoted = point.get("promoted", False)
        fill = "#2f855a" if promoted else "#0d47a1"
        radius = 7 if promoted else 5
        markers.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" />')
        labels.append(
            f'<text x="{x:.1f}" y="{margin_top + chart_h + 24}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#1f2933">iter {point["iteration"]}</text>'
        )
        labels.append(
            f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="{fill}">{point["elo"]:.1f}</text>'
        )

    note = "Promotion" if any(point.get("promoted") for point in points) else "Checkpoint"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="26" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#102a43">
    square8_2p recent Elo progression
  </text>
  <text x="{margin_left}" y="44" font-family="Arial, sans-serif" font-size="13" fill="#52606d">
    The April 2026 clean-harness run is the clearest recent improvement story in the project.
  </text>
  {''.join(tick_lines)}
  <line x1="{margin_left}" y1="{margin_top + chart_h:.1f}" x2="{width - margin_right}" y2="{margin_top + chart_h:.1f}" stroke="#7b8794" stroke-width="2" />
  <path d="{path}" fill="none" stroke="#0d47a1" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  {''.join(markers)}
  {''.join(labels)}
  <rect x="{width - 208}" y="24" width="160" height="54" rx="10" fill="#f7fafc" stroke="#d9e2ec" />
  <circle cx="{width - 188}" cy="45" r="6" fill="#2f855a" />
  <text x="{width - 174}" y="49" font-family="Arial, sans-serif" font-size="13" fill="#1f2933">{note}</text>
  <circle cx="{width - 188}" cy="64" r="5" fill="#0d47a1" />
  <text x="{width - 174}" y="68" font-family="Arial, sans-serif" font-size="13" fill="#1f2933">Non-promotion checkpoint</text>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight SVG visuals for RingRift results docs.")
    parser.add_argument(
        "--snapshot",
        default="docs/data/results_snapshot.json",
        help="Path to the checked-in results snapshot JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/assets/results",
        help="Directory where SVGs will be written.",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    out_dir = Path(args.out_dir)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    _write(out_dir / "headline_results.svg", _headline_svg(snapshot))
    _write(out_dir / "square8_2p_progression.svg", _progression_svg(snapshot))

    print(f"Wrote {out_dir / 'headline_results.svg'}")
    print(f"Wrote {out_dir / 'square8_2p_progression.svg'}")


if __name__ == "__main__":
    main()
