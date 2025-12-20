#!/usr/bin/env python
"""Parity-focused healthcheck harness for the Python rules engine.

This script runs a small, bounded subset of TS↔Python rules parity checks
using existing fixtures and test helpers, then aggregates any mismatches into
a compact, machine-readable JSON summary. It is intended for CI or nightly
jobs as a fast "PARITY-* health snapshot" complementary to the longer
contract-vector and orchestrator-parity suites.

The current implementation focuses on:

- contract_vectors_v2: v2 contract vectors executed via the Python
  contract test runner.
- plateau_snapshots: TS-generated plateau ComparableSnapshot JSON fixtures
  for selected seeds (PARITY-TS-PY-SEED-PLATEAU).

Additional suites (e.g. line+territory scenarios, active-no-moves
regressions) can be wired in by extending SUITE_RUNNERS.

NOTE: This harness intentionally reuses existing test helper modules under
``ai-service/tests`` rather than duplicating parity logic. It should remain
light-weight and non-invasive; it does not change rules semantics.

TODO(P17.B7): once validated, wire this summary into the TS-side
``ringrift_rules_parity_mismatches_total{mismatch_type,suite}`` metric and
PARITY-* catalogue described in docs/INVARIANTS_AND_PARITY_FRAMEWORK.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Iterable, Sequence

# Ensure `app.*` and `tests.*` imports resolve when run from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Contract vectors runner and helpers
import tests.contracts.test_contract_vectors as contract_vectors  # type: ignore  # noqa: E402,E501

# Plateau snapshot builders (seed plateau parity)
from tests.parity import (  # type: ignore  # noqa: E402
    test_ts_seed_plateau_snapshot_parity as plateau_mod,
)

from app.utils.progress_reporter import ProgressReporter  # type: ignore  # noqa: E402


@dataclass
class ParityCaseResult:
    """Result for a single parity case (fixture, vector, or scenario)."""

    suite: str
    case_id: str
    mismatch_type: str | None
    details: str | None = None


SUPPORTED_SUITES = (
    "contract_vectors_v2",
    "plateau_snapshots",
)

PARITY_ID_BY_SUITE: dict[str, str] = {
    # Contract vectors parity (full TS↔Python turn/phase semantics).
    "contract_vectors_v2": "PARITY-TS-PY-CONTRACT-VECTORS",
    # Seed / plateau parity based on TS ComparableSnapshot exports.
    "plateau_snapshots": "PARITY-TS-PY-SEED-PLATEAU",
}

MAX_SAMPLE_MISMATCHES = 10


def _classify_contract_vector_mismatch(failures: Sequence[str]) -> str:
    """Best-effort classification of contract-vector mismatches.

    The classification is derived from assertion messages emitted by
    tests/contracts/test_contract_vectors.py and is intentionally
    coarse-grained – the goal is to bucket mismatches by high-level
    category (validation / status / hash / s_invariant) for metrics and
    dashboards, not to encode every possible failure shape.
    """
    kind_flags: dict[str, bool] = {
        "validation": False,
        "status": False,
        "hash": False,
        "s_invariant": False,
    }

    for msg in failures:
        if msg.startswith("Exception during apply_move:"):
            kind_flags["validation"] = True
        elif (
            msg.startswith("gameStatus:")
            or msg.startswith(
                "currentPhase:",
            )
            or msg.startswith("currentPlayer:")
        ):
            kind_flags["status"] = True
        elif msg.startswith("sInvariantDelta:"):
            kind_flags["s_invariant"] = True
        elif (
            msg.startswith("stackCount:")
            or msg.startswith(
                "markerCount:",
            )
            or msg.startswith("collapsedCount:")
        ):
            kind_flags["hash"] = True

    if kind_flags["validation"]:
        return "validation"
    if kind_flags["status"]:
        return "status"
    if kind_flags["s_invariant"]:
        return "s_invariant"
    if kind_flags["hash"]:
        return "hash"
    return "unknown"


def run_contract_vectors_suite() -> list[ParityCaseResult]:
    """Run the v2 contract vectors against the Python engine.

    This reuses the existing contract-vector runner module, but instead of
    asserting via pytest it records pass/fail outcomes per vector.

    Vectors with explicit 'skip' field or 'multi_phase' tag are excluded
    because they require orchestrator-level execution, not single-move testing.
    """
    results: list[ParityCaseResult] = []
    all_vectors = contract_vectors.load_all_vectors()

    # Filter out vectors that require orchestrator execution or have known
    # fixture issues (matching the logic in test_contract_vector pytest test)
    # Orchestrator-tagged vectors may have expected values that assume a different
    # phase of turn processing (e.g., before victory check runs) and aren't
    # suitable for single-move apply_move testing.
    vectors = [
        v for v in all_vectors
        if not v.skip
        and "multi_phase" not in v.tags
        and "orchestrator" not in v.tags
        and v.id not in contract_vectors.KNOWN_FAILING_VECTORS
    ]

    total_vectors = len(vectors)
    reporter = ProgressReporter(
        total_units=total_vectors,
        unit_name="vector",
        report_interval_sec=10.0,
        context_label="contract_vectors_v2",
    )

    mismatches_so_far = 0

    for idx, vec in enumerate(vectors, start=1):
        validation = contract_vectors.execute_vector(vec)
        if validation.passed:
            results.append(
                ParityCaseResult(
                    suite="contract_vectors_v2",
                    case_id=vec.id,
                    mismatch_type=None,
                ),
            )
        else:
            mismatch_type = _classify_contract_vector_mismatch(
                validation.failures,
            )
            details = "; ".join(validation.failures)
            results.append(
                ParityCaseResult(
                    suite="contract_vectors_v2",
                    case_id=vec.id,
                    mismatch_type=mismatch_type,
                    details=details or None,
                ),
            )
            mismatches_so_far += 1

        # Throttled progress reporting (~10s by default) so long-running
        # contract-vector suites remain chatty in CI and diagnostics.
        reporter.update(
            completed=idx,
            extra_metrics={"mismatches": mismatches_so_far},
        )

    reporter.finish(
        extra_metrics={"total_mismatches": mismatches_so_far},
    )

    return results


def _iter_plateau_snapshots() -> Iterable[tuple[str, Any]]:
    """Yield (case_id, snapshot_dict) for available plateau fixtures."""
    import json

    for path in (plateau_mod.SEED1_SNAPSHOT, plateau_mod.SEED18_SNAPSHOT):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            snapshot: dict[str, Any] = json.load(f)
        label = snapshot.get("label") or path.stem
        yield label, snapshot


def run_plateau_snapshots_suite() -> list[ParityCaseResult]:
    """Run plateau snapshot parity checks (seed plateau parity).

    For each TS ComparableSnapshot JSON fixture we:

    1. Hydrate a Python GameState via the test builder.
    2. Reconstruct a Python-side ComparableSnapshot dict.
    3. Normalise and compare JSON payloads.

    Any structural difference is reported as a ``hash`` mismatch; exceptions
    during hydration are reported as ``validation`` mismatches.
    """
    results: list[ParityCaseResult] = []

    for case_id, snapshot in _iter_plateau_snapshots():
        try:
            state = plateau_mod._build_game_state_from_snapshot(snapshot)
            py_snapshot = plateau_mod._python_comparable_snapshot(
                snapshot.get("label") or "python",
                state,
            )
            ts_norm = plateau_mod._normalise_for_comparison(snapshot)
            py_norm = plateau_mod._normalise_for_comparison(py_snapshot)

            if py_norm == ts_norm:
                results.append(
                    ParityCaseResult(
                        suite="plateau_snapshots",
                        case_id=case_id,
                        mismatch_type=None,
                    ),
                )
            else:
                details = "Python ComparableSnapshot diverged from TS plateau " f"fixture for case {case_id!r}."
                results.append(
                    ParityCaseResult(
                        suite="plateau_snapshots",
                        case_id=case_id,
                        mismatch_type="hash",
                        details=details,
                    ),
                )
        except AssertionError as exc:
            results.append(
                ParityCaseResult(
                    suite="plateau_snapshots",
                    case_id=case_id,
                    mismatch_type="hash",
                    details=("AssertionError during plateau reconstruction: " f"{exc}"),
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                ParityCaseResult(
                    suite="plateau_snapshots",
                    case_id=case_id,
                    mismatch_type="validation",
                    details=f"{type(exc).__name__}: {exc}",
                ),
            )

    return results


def _summarise_parity_results(
    results: Sequence[ParityCaseResult],
    *,
    profile: str = "parity-healthcheck",
) -> dict[str, Any]:
    """Aggregate per-case parity results into a compact summary dict."""
    total_cases = len(results)
    mismatches_total = sum(1 for r in results if r.mismatch_type)

    mismatches_by_type: dict[str, int] = {}
    mismatches_by_suite: dict[str, int] = {}
    suites: list[str] = []

    for r in results:
        if r.suite not in suites:
            suites.append(r.suite)
        if r.mismatch_type:
            mismatches_by_type[r.mismatch_type] = mismatches_by_type.get(r.mismatch_type, 0) + 1
            mismatches_by_suite[r.suite] = mismatches_by_suite.get(r.suite, 0) + 1

    samples: list[dict[str, Any]] = []
    for r in results:
        if not r.mismatch_type:
            continue
        samples.append(
            {
                "suite": r.suite,
                "case_id": r.case_id,
                "mismatch_type": r.mismatch_type,
                "details": r.details,
            },
        )
        if len(samples) >= MAX_SAMPLE_MISMATCHES:
            break

    summary: dict[str, Any] = {
        "profile": profile or "parity-healthcheck",
        "suites": sorted(set(suites)),
        "total_cases": total_cases,
        "mismatches_total": mismatches_total,
        "mismatches_by_type": mismatches_by_type,
        "mismatches_by_suite": mismatches_by_suite,
        "samples": samples,
        "parity_ids_by_suite": {suite: PARITY_ID_BY_SUITE.get(suite, "") for suite in sorted(set(suites))},
    }

    # Leave a structured hook for future TS-side rules_parity_mismatches_total
    # wiring. Downstream metrics collectors can map (suite, mismatch_type)
    # and PARITY-* IDs to Prometheus label sets.
    return summary


SUITE_RUNNERS = {
    "contract_vectors_v2": run_contract_vectors_suite,
    "plateau_snapshots": run_plateau_snapshots_suite,
}


def run_parity_healthcheck(
    args: argparse.Namespace,
) -> tuple[list[ParityCaseResult], dict[str, Any]]:
    """Execute the selected suites and return (results, summary)."""
    if getattr(args, "suite", None):
        selected = list(dict.fromkeys(args.suite))
    else:
        selected = list(SUITE_RUNNERS.keys())

    invalid = [s for s in selected if s not in SUITE_RUNNERS]
    if invalid:
        raise SystemExit(
            f"Unknown suite(s): {', '.join(sorted(invalid))}. "
            f"Supported suites: {', '.join(sorted(SUITE_RUNNERS))}.",
        )

    all_results: list[ParityCaseResult] = []
    for suite in selected:
        runner = SUITE_RUNNERS[suite]
        suite_results = runner()
        all_results.extend(suite_results)

    summary = _summarise_parity_results(
        all_results,
        profile=getattr(args, "profile", "parity-healthcheck"),
    )
    return all_results, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded TS↔Python rules parity healthcheck over selected "
            "suites (contract vectors, plateau snapshots)."
        ),
    )
    parser.add_argument(
        "--profile",
        default="parity-healthcheck",
        help=("Logical profile name to embed in the JSON summary " "(default: parity-healthcheck)."),
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUPPORTED_SUITES,
        help=("Optional suite to run. May be provided multiple times. " "Defaults to running all supported suites."),
    )
    parser.add_argument(
        "--summary-json",
        help=("Optional path to write the parity summary JSON. Directories " "are created if needed."),
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help=("If set, exit with non-zero status when any mismatches are " "observed. Intended for CI/nightly gates."),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    _results, summary = run_parity_healthcheck(args)

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.summary_json:
        directory = os.path.dirname(args.summary_json) or "."
        os.makedirs(directory, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    if args.fail_on_mismatch and summary.get("mismatches_total", 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
