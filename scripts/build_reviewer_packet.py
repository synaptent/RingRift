#!/usr/bin/env python3
"""Build a compact reviewer packet from the supported RingRift surface."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "docs" / "data" / "reviewer_surface_manifest.json"
DEFAULT_OUTPUT = ROOT / "dist" / "reviewer_packet"
EXTRA_PACKET_FILES = (
    "docs/data/results_snapshot.json",
    "docs/data/results_evidence_manifest.json",
    "docs/assets/results/headline_results.svg",
    "docs/assets/results/square8_2p_progression.svg",
    "docs/assets/readme/hex8-sandbox-live.png",
    "docs/data/ai_surface_manifest.json",
)


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _copy_file(relative_path: str, output_dir: Path) -> Path:
    source = ROOT / relative_path
    if not source.exists():
        raise FileNotFoundError(f"Reviewer packet source missing: {relative_path}")
    target = output_dir / "files" / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _packet_files(manifest: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for item in manifest.get("must_read", []):
        if isinstance(item, str) and (ROOT / item).is_file():
            files.append(item)
    for item in EXTRA_PACKET_FILES:
        if item not in files:
            files.append(item)
    return files


def _write_index(manifest: dict[str, Any], copied_files: list[str], output_dir: Path) -> None:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# RingRift Reviewer Packet",
        "",
        f"Generated at: `{generated_at}`",
        f"Reviewer manifest: `{manifest.get('as_of', 'unknown')}`",
        "",
        "This packet is a compact copy of the supported review surface. It does not include large external artifacts such as model checkpoints or training NPZ files.",
        "",
        "## Included Files",
        "",
    ]
    for relative_path in copied_files:
        lines.append(f"- [files/{relative_path}](files/{relative_path})")
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "- Public result claims are copied from `docs/data/results_snapshot.json`.",
            "- Claim provenance is copied from `docs/data/results_evidence_manifest.json`.",
            "- Supported-vs-experimental AI boundaries are copied from `docs/data/ai_surface_manifest.json`.",
            "- External artifacts remain referenced from `docs/REPRODUCIBILITY.md`.",
        ]
    )
    (output_dir / "PACKET_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Output directory for the reviewer packet.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before rebuilding.",
    )
    args = parser.parse_args()

    output_dir = Path(args.out).resolve()
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    files = _packet_files(manifest)
    copied_files: list[str] = []
    for relative_path in files:
        _copy_file(relative_path, output_dir)
        copied_files.append(relative_path)
    _write_index(manifest, copied_files, output_dir)

    print(f"Reviewer packet written to {output_dir}")
    print(f"Included files: {len(copied_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
