#!/usr/bin/env python3
"""Verify that every checked-in canonical model checkpoint matches its sidecar.

For each `ai-service/models/canonical_*.pth` the script computes a SHA-256 hash
and compares it to the `.sha256` sidecar that is tracked in git. The output is
a small verification matrix that an outsider can use to decide whether a
checkpoint is publishable / loadable as advertised.

This is the *broad sidecar-only sweep*. For a deeper release gate that also
verifies the checkpoint actually loads and that its metadata matches the
advertised board/player config for a specific public artifact, see
`ai-service/scripts/audit_public_model_artifacts.py`.

Exit codes:
  0  every checkpoint with a sidecar matched
  1  at least one checkpoint mismatched its sidecar
  2  fatal usage / IO error

This is intentionally a small, dependency-free script. It is meant to be safe
to run on any clean checkout that has the sidecars (the actual `.pth` files are
gitignored, so missing checkpoints are reported, not failed).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "ai-service" / "models"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sidecar(sidecar: Path) -> str | None:
    """Return the first non-empty whitespace-trimmed token of the sidecar."""
    try:
        for raw_line in sidecar.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                return line.split()[0]
    except OSError:
        return None
    return None


def find_canonical_pairs(models_dir: Path) -> list[tuple[Path, Path]]:
    """Return (.pth, .sha256) pairs for every canonical checkpoint sidecar."""
    pairs: list[tuple[Path, Path]] = []
    if not models_dir.is_dir():
        return pairs
    for sidecar in sorted(models_dir.glob("canonical_*.pth.sha256")):
        pth = sidecar.with_suffix("")  # drop .sha256 → leaves .pth
        pairs.append((pth, sidecar))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing canonical_*.pth and canonical_*.pth.sha256",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress matching rows; only print mismatches and missing files",
    )
    args = parser.parse_args()

    pairs = find_canonical_pairs(args.models_dir)
    if not pairs:
        print(f"No canonical_*.pth.sha256 sidecars found under {args.models_dir}", file=sys.stderr)
        return 2

    width_name = max(len(p[1].name) for p in pairs)
    header = f"{'sidecar'.ljust(width_name)}  status     expected[:12]  actual[:12]"
    if not args.quiet:
        print(header)
        print("-" * len(header))

    any_mismatch = False
    for pth, sidecar in pairs:
        expected = parse_sidecar(sidecar)
        if expected is None:
            print(f"{sidecar.name.ljust(width_name)}  BAD-SIDECAR  -             -")
            any_mismatch = True
            continue
        if not pth.is_file():
            print(f"{sidecar.name.ljust(width_name)}  MISSING-PTH  {expected[:12]}  -")
            continue
        actual = sha256_file(pth)
        if actual == expected:
            if not args.quiet:
                print(f"{sidecar.name.ljust(width_name)}  OK           {expected[:12]}  {actual[:12]}")
        else:
            print(f"{sidecar.name.ljust(width_name)}  MISMATCH     {expected[:12]}  {actual[:12]}")
            any_mismatch = True

    return 1 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
