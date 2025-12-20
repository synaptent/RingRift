#!/usr/bin/env python
"""Generate axis-aligned heuristic weight profiles for diagnostics.

This script constructs single-factor (axis-aligned) profiles for each
heuristic weight defined in BASE_V1_BALANCED_WEIGHTS /
HEURISTIC_WEIGHT_KEYS and writes them as JSON files compatible with
existing CMA-ES / GA tooling.

It is intentionally self-contained and does not modify any runtime
behaviour or training configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple
from collections.abc import Iterable


# Allow imports from app/ when run from the ai-service root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
    HeuristicWeights,
)


OUTPUT_DIR = os.path.join("logs", "axis_aligned", "profiles")


def build_axis_aligned_profile(
    factor_key: str,
    *,
    base_weights: HeuristicWeights | dict[str, float] | None = None,
    ordered_keys: Iterable[str] | None = None,
) -> tuple[HeuristicWeights, HeuristicWeights, float]:
    """Construct positive/negative axis-aligned profiles for a single factor.

    Parameters
    ----------
    factor_key:
        The heuristic weight key to activate (e.g. "WEIGHT_TERRITORY").
    base_weights:
        Baseline weights to derive magnitudes from. Defaults to
        BASE_V1_BALANCED_WEIGHTS.
    ordered_keys:
        Canonical ordering of heuristic weight keys. Defaults to
        HEURISTIC_WEIGHT_KEYS.

    Returns
    -------
    Tuple[HeuristicWeights, HeuristicWeights, float]
        (pos_profile, neg_profile, magnitude)
    """

    if base_weights is None:
        base_weights = BASE_V1_BALANCED_WEIGHTS
    if ordered_keys is None:
        ordered_keys = HEURISTIC_WEIGHT_KEYS

    if factor_key not in base_weights:
        raise KeyError(f"Unknown heuristic weight key: {factor_key!r}")

    base_val = float(base_weights[factor_key])
    mag = max(abs(base_val), 1.0)

    pos_profile: HeuristicWeights = {}
    neg_profile: HeuristicWeights = {}

    for key in ordered_keys:
        if key == factor_key:
            pos_profile[key] = mag
            neg_profile[key] = -mag
        else:
            pos_profile[key] = 0.0
            neg_profile[key] = 0.0

    # Validation: profile keys must match the baseline schema exactly.
    baseline_keys = set(BASE_V1_BALANCED_WEIGHTS.keys())
    pos_keys = set(pos_profile.keys())
    neg_keys = set(neg_profile.keys())
    assert pos_keys == baseline_keys == neg_keys, (
        f"Axis-aligned profile keys mismatch for {factor_key!r}: "
        f"baseline={sorted(baseline_keys)}, pos={sorted(pos_keys)}, "
        f"neg={sorted(neg_keys)}"
    )

    # Validation: exactly one non-zero entry per profile.
    non_zero_pos = [k for k, v in pos_profile.items() if abs(v) > 1e-9]
    non_zero_neg = [k for k, v in neg_profile.items() if abs(v) > 1e-9]

    assert non_zero_pos == [factor_key], (
        f"Positive axis-aligned profile for {factor_key!r} has " f"non-zero keys {non_zero_pos}"
    )
    assert non_zero_neg == [factor_key], (
        f"Negative axis-aligned profile for {factor_key!r} has " f"non-zero keys {non_zero_neg}"
    )

    return pos_profile, neg_profile, mag


def write_profile_json(
    path: str,
    weights: HeuristicWeights,
    *,
    factor: str,
    sign: str,
    base_magnitude: float,
    dry_run: bool = False,
) -> None:
    """Write a single axis-aligned profile JSON file.

    In dry-run mode, print the payload instead of writing to disk.
    """

    payload = {
        "weights": weights,
        "meta": {
            "factor": factor,
            "sign": sign,
            "base_magnitude": base_magnitude,
        },
    }

    if dry_run:
        print(f"[DRY-RUN] Would write {path}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def generate_axis_aligned_profiles(*, dry_run: bool = False) -> int:
    """Generate pos/neg axis-aligned profiles for all heuristic factors.

    Returns an exit code (0 on success, non-zero on failure).
    """

    total_written = 0

    try:
        baseline_keys = set(BASE_V1_BALANCED_WEIGHTS.keys())
        canonical = list(HEURISTIC_WEIGHT_KEYS)
        canonical_set = set(canonical)

        # Sanity check: canonical key list matches the baseline schema.
        assert canonical_set == baseline_keys, (
            "HEURISTIC_WEIGHT_KEYS must match BASE_V1_BALANCED_WEIGHTS keys "
            "for axis-aligned profile generation; got "
            f"missing={sorted(baseline_keys - canonical_set)}, "
            f"extra={sorted(canonical_set - baseline_keys)}"
        )

        for key in canonical:
            pos_profile, neg_profile, mag = build_axis_aligned_profile(key)

            pos_path = os.path.join(OUTPUT_DIR, f"{key}_pos.json")
            neg_path = os.path.join(OUTPUT_DIR, f"{key}_neg.json")

            write_profile_json(
                pos_path,
                pos_profile,
                factor=key,
                sign="pos",
                base_magnitude=mag,
                dry_run=dry_run,
            )
            write_profile_json(
                neg_path,
                neg_profile,
                factor=key,
                sign="neg",
                base_magnitude=mag,
                dry_run=dry_run,
            )

            total_written += 2

    except AssertionError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        msg = "ERROR: Unexpected failure during axis-aligned generation: " f"{exc}"
        print(msg)
        return 1

    if dry_run:
        print(f"[DRY-RUN] Would generate {total_written} axis-aligned profiles " f"(pos/neg) in {OUTPUT_DIR}")
    else:
        print(f"Generated {total_written} axis-aligned profiles (pos/neg) " f"in {OUTPUT_DIR}")

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate axis-aligned HeuristicAI weight profiles for " "diagnostics.")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Show what would be written without touching the filesystem; " "profiles are printed to stdout."),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    exit_code = generate_axis_aligned_profiles(dry_run=args.dry_run)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
