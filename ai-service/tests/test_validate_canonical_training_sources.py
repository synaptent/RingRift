import json
import os
import sys
from pathlib import Path

import pytest


# Ensure we can import from ai-service/ when pytest is invoked from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.validate_canonical_training_sources import validate_canonical_sources  # noqa: E402


def _write_registry(
    path: Path,
    *,
    db_name: str,
    status: str,
    gate_summary: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Training Data Registry",
                "",
                "## Game Replay Databases",
                "",
                "| Database | Board Type | Players | Status | Gate Summary | Notes |",
                "| -------- | ---------- | ------- | ------ | ------------ | ----- |",
                f"| `{db_name}` | square8 | 2 | **{status}** | {gate_summary} | test |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_validate_canonical_sources_default_rejects_pending_gate(tmp_path: Path) -> None:
    registry_path = tmp_path / "TRAINING_DATA_REGISTRY.md"
    db_path = tmp_path / "data" / "games" / "canonical_square8.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")  # existence is not required but keeps tests explicit

    gate_summary_name = "db_health.canonical_square8.json"
    gate_summary_path = tmp_path / gate_summary_name
    gate_summary_path.write_text(
        json.dumps(
            {
                "canonical_ok": True,
                "parity_gate": {"passed_canonical_parity_gate": True},
            }
        ),
        encoding="utf-8",
    )

    _write_registry(
        registry_path,
        db_name=db_path.name,
        status="pending_gate",
        gate_summary=gate_summary_name,
    )

    result = validate_canonical_sources(registry_path, [db_path])
    assert result["ok"] is False
    problems = "\n".join(result.get("problems", []))
    assert "pending_gate" in problems


def test_validate_canonical_sources_allows_pending_gate_when_configured(tmp_path: Path) -> None:
    registry_path = tmp_path / "TRAINING_DATA_REGISTRY.md"
    db_path = tmp_path / "data" / "games" / "canonical_square8.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")

    gate_summary_name = "db_health.canonical_square8.json"
    gate_summary_path = tmp_path / gate_summary_name
    gate_summary_path.write_text(
        json.dumps(
            {
                "canonical_ok": True,
                "parity_gate": {"passed_canonical_parity_gate": True},
            }
        ),
        encoding="utf-8",
    )

    _write_registry(
        registry_path,
        db_name=db_path.name,
        status="pending_gate",
        gate_summary=gate_summary_name,
    )

    result = validate_canonical_sources(
        registry_path,
        [db_path],
        allowed_statuses=["canonical", "pending_gate"],
    )
    assert result["ok"] is True

