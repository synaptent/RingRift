import json
import os
import sys
from typing import Dict, List, Tuple

import pytest

# Ensure app and scripts packages are importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import scripts.run_parity_healthcheck as parity  # type: ignore
from scripts.run_parity_healthcheck import (  # type: ignore
    ParityCaseResult,
)


def test_parity_healthcheck_summary_aggregates_mismatches() -> None:
    """_summarise_parity_results should aggregate basic mismatch stats."""
    results: list[ParityCaseResult] = [
        ParityCaseResult(
            suite="contract_vectors_v2",
            case_id="vec-001",
            mismatch_type=None,
        ),
        ParityCaseResult(
            suite="contract_vectors_v2",
            case_id="vec-002",
            mismatch_type="hash",
            details="hash mismatch",
        ),
        ParityCaseResult(
            suite="plateau_snapshots",
            case_id="square8-2p-seed1",
            mismatch_type="status",
            details="status mismatch",
        ),
    ]

    summary = parity._summarise_parity_results(  # type: ignore[attr-defined]
        results,
    )

    assert summary["profile"] == "parity-healthcheck"
    assert summary["total_cases"] == 3
    assert summary["mismatches_total"] == 2

    by_type = summary["mismatches_by_type"]
    assert by_type["hash"] == 1
    assert by_type["status"] == 1

    by_suite = summary["mismatches_by_suite"]
    assert by_suite["contract_vectors_v2"] == 1
    assert by_suite["plateau_snapshots"] == 1

    suites = set(summary["suites"])
    assert "contract_vectors_v2" in suites
    assert "plateau_snapshots" in suites

    # Samples should include both mismatches with expected shape.
    samples = summary["samples"]
    assert len(samples) == 2
    for sample in samples:
        assert "suite" in sample
        assert "case_id" in sample
        assert "mismatch_type" in sample

    # Parity IDs should be surfaced for known suites to align with
    # docs/INVARIANTS_AND_PARITY_FRAMEWORK.md PARITY-* catalogue.
    parity_ids = summary["parity_ids_by_suite"]
    assert (
        parity_ids["contract_vectors_v2"]
        == "PARITY-TS-PY-CONTRACT-VECTORS"
    )
    assert parity_ids["plateau_snapshots"] == "PARITY-TS-PY-SEED-PLATEAU"


def test_parity_healthcheck_cli_writes_summary_json_without_failing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """CLI should write summary JSON and exit cleanly when no mismatches."""
    captured_args: dict[str, object] = {}

    def fake_run(
        args: object,
    ) -> tuple[list[ParityCaseResult], dict[str, object]]:
        captured_args["args"] = args
        # No mismatches in this synthetic run.
        results: list[ParityCaseResult] = [
            ParityCaseResult(
                suite="contract_vectors_v2",
                case_id="vec-001",
                mismatch_type=None,
            ),
        ]
        summary: dict[str, object] = {
            "profile": getattr(args, "profile", "parity-healthcheck"),
            "suites": ["contract_vectors_v2"],
            "total_cases": 1,
            "mismatches_total": 0,
            "mismatches_by_type": {},
            "mismatches_by_suite": {},
            "samples": [],
            "parity_ids_by_suite": {
                "contract_vectors_v2": "PARITY-TS-PY-CONTRACT-VECTORS",
            },
        }
        return results, summary

    monkeypatch.setattr(parity, "run_parity_healthcheck", fake_run)

    summary_path = tmp_path / "parity-healthcheck.summary.json"

    # Invoke CLI entrypoint with explicit argv so tests do not depend on
    # global sys.argv mutation.
    parity.main(
        [
            "--summary-json",
            str(summary_path),
            "--profile",
            "parity-healthcheck",
        ],
    )

    # Ensure the helper saw the profile and summary file was written.
    args = captured_args.get("args")
    assert args is not None
    assert args.profile == "parity-healthcheck"

    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["profile"] == "parity-healthcheck"
    assert payload["mismatches_total"] == 0
    assert payload["suites"] == ["contract_vectors_v2"]
    assert "mismatches_by_type" in payload
    assert "samples" in payload


def test_parity_healthcheck_cli_respects_fail_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """CLI should exit with code 1 when mismatches exist and flag is set."""

    def fake_run(
        args: object,
    ) -> tuple[list[ParityCaseResult], dict[str, object]]:
        results: list[ParityCaseResult] = [
            ParityCaseResult(
                suite="contract_vectors_v2",
                case_id="vec-001",
                mismatch_type="hash",
                details="synthetic mismatch",
            ),
        ]
        summary: dict[str, object] = {
            "profile": getattr(args, "profile", "parity-healthcheck"),
            "suites": ["contract_vectors_v2"],
            "total_cases": 1,
            "mismatches_total": 1,
            "mismatches_by_type": {"hash": 1},
            "mismatches_by_suite": {"contract_vectors_v2": 1},
            "samples": [
                {
                    "suite": "contract_vectors_v2",
                    "case_id": "vec-001",
                    "mismatch_type": "hash",
                    "details": "synthetic mismatch",
                },
            ],
            "parity_ids_by_suite": {
                "contract_vectors_v2": "PARITY-TS-PY-CONTRACT-VECTORS",
            },
        }
        return results, summary

    monkeypatch.setattr(parity, "run_parity_healthcheck", fake_run)

    summary_path = tmp_path / "parity-healthcheck.mismatches.json"

    with pytest.raises(SystemExit) as excinfo:
        parity.main(
            [
                "--summary-json",
                str(summary_path),
                "--profile",
                "parity-healthcheck",
                "--fail-on-mismatch",
            ],
        )

    assert excinfo.value.code == 1
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["profile"] == "parity-healthcheck"
    assert payload["mismatches_total"] == 1
    assert payload["mismatches_by_type"]["hash"] == 1
    assert payload["samples"][0]["case_id"] == "vec-001"
