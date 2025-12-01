"""
Tests for territory-schema validation integration in train_heuristic_weights.

These tests exercise the new `validate_territory_schema` path that allows
training runs to fail fast on malformed territory / combined-margin datasets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure app package is importable when running tests directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.training.train_heuristic_weights import (  # noqa: E402
    train_from_jsonl,
)


def test_train_from_jsonl_raises_on_invalid_territory_dataset(tmp_path: Path) -> None:
    """train_from_jsonl should fail fast when schema validation finds errors."""

    dataset_path = tmp_path / "invalid_territory_dataset.jsonl"
    output_path = tmp_path / "out.json"

    # Write a deliberately malformed record that is missing required fields
    # such as `game_state` and AI metadata. This should be caught by
    # territory_dataset_validation before any training is attempted.
    dataset_path.write_text(
        '{"player_number": 1, "target": 0.0, "time_weight": 1.0, '
        '"engine_mode": "mixed", "num_players": 2}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        train_from_jsonl(
            dataset_path=str(dataset_path),
            output_path=str(output_path),
            validate_territory_schema=True,
            max_validation_errors=5,
        )

    message = str(excinfo.value)
    assert "Territory dataset validation failed" in message
    assert str(dataset_path) in message

