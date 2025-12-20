"""Tests for the Square-8 2-player difficulty calibration analysis script."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config.ladder_config import (  # type: ignore[import]  # noqa: E402
    LadderTierConfig,
)
from app.models import AIType, BoardType  # type: ignore[import]  # noqa: E402
import scripts.analyze_difficulty_calibration as adc  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _make_calibration_payload() -> dict[str, Any]:
    """Return a small but representative calibration aggregates payload."""
    return {
        "board": "square8",
        "num_players": 2,
        "window": {
            "start": "2025-11-01T00:00:00Z",
            "end": "2025-11-29T23:59:59Z",
        },
        "tiers": [
            {
                "tier": "D4",
                "difficulty": 4,
                "segments": [
                    {
                        "segment": "new",
                        "n_games": 120,
                        "human_win_rate": 0.42,
                        "difficulty_mean": 3.6,
                        "difficulty_p10": 2.0,
                        "difficulty_p90": 4.8,
                    },
                    {
                        "segment": "intermediate",
                        "n_games": 85,
                        "human_win_rate": 0.51,
                        "difficulty_mean": 3.1,
                        "difficulty_p10": 2.4,
                        "difficulty_p90": 4.1,
                    },
                ],
            },
            {
                "tier": "D6",
                "difficulty": 6,
                "segments": [
                    {
                        "segment": "strong",
                        "n_games": 40,
                        "human_win_rate": 0.45,
                        "difficulty_mean": 3.2,
                        "difficulty_p10": 2.6,
                        "difficulty_p90": 3.9,
                    },
                ],
            },
        ],
    }


def _make_registry_payload(run_rel_path: str) -> dict[str, Any]:
    """Return a minimal candidate registry payload for D4."""
    current = {
        "tier": "D4",
        "difficulty": 4,
        "board": "square8",
        "board_type": "square8",
        "num_players": 2,
        "model_id": "current_d4",
        "heuristic_profile_id": "heuristic_v1_2p",
        "ai_type": "MINIMAX",
        "ladder_source": "test",
    }
    candidate = {
        "candidate_id": "sq8_2p_d4_demo",
        "candidate_model_id": "sq8_2p_d4_demo",
        "tier": "D4",
        "board": "square8",
        "num_players": 2,
        "source_run_dir": run_rel_path,
        "training_report": "training_report.json",
        "gate_report": "gate_report.json",
        "promotion_plan": "promotion_plan.json",
        "model_id": "current_d4",
        "heuristic_profile_id": "heuristic_v1_2p",
        "status": "gated_promote",
    }
    return {
        "board": "square8",
        "num_players": 2,
        "tiers": {
            "D4": {
                "current": current,
                "candidates": [candidate],
            }
        },
    }


def _make_ladder_config(difficulty: int) -> LadderTierConfig:
    """Return a simple LadderTierConfig stub for Square-8 2p."""
    model_id = "current_d4" if difficulty == 4 else f"model_d{difficulty}"
    return LadderTierConfig(
        difficulty=difficulty,
        board_type=BoardType.SQUARE8,
        num_players=2,
        ai_type=AIType.MINIMAX,
        model_id=model_id,
        heuristic_profile_id="heuristic_v1_2p",
        randomness=0.1,
        think_time_ms=1000,
        notes="test-stub",
    )


def test_analyze_difficulty_calibration_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end happy path for the calibration analysis script."""
    # Synthetic calibration aggregates.
    calib_path = tmp_path / "calibration_aggregates.json"
    _write_json(calib_path, _make_calibration_payload())

    # Synthetic registry with a single promoted D4 candidate.
    run_rel_path = os.fspath(Path("runs") / "sq8_2p" / "D4" / "demo")
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, _make_registry_payload(run_rel_path))

    # Eval/perf artefacts for the registry candidate.
    eval_root = tmp_path / "eval_root"
    run_dir = eval_root / run_rel_path
    eval_payload: dict[str, Any] = {
        "tier": "D4",
        "board_type": "square8",
        "num_players": 2,
        "overall_pass": True,
        "metrics": {
            "win_rate_vs_baseline": 0.82,
            "win_rate_vs_previous_tier": 0.58,
        },
    }
    perf_payload: dict[str, Any] = {
        "tier_name": "D4_SQ8_2P",
        "difficulty": 4,
        "board_type": "square8",
        "num_players": 2,
        "metrics": {
            "average_ms": 95.0,
            "p95_ms": 170.0,
        },
        "budget": {
            "max_avg_move_ms": 120.0,
            "max_p95_move_ms": 220.0,
        },
        "evaluation": {
            "overall_pass": True,
        },
    }
    _write_json(run_dir / "tier_eval_result.json", eval_payload)
    _write_json(run_dir / "tier_perf_report.json", perf_payload)

    # Monkeypatch ladder and registry helpers used by the script.
    def _fake_get_ladder_tier_config(
        difficulty: int,
        board_type: BoardType,
        num_players: int,
    ) -> LadderTierConfig:
        assert board_type is BoardType.SQUARE8
        assert num_players == 2
        return _make_ladder_config(difficulty)

    def _fake_load_registry(path: str | None = None) -> dict[str, Any]:
        assert path is None or os.path.samefile(path, registry_path)
        return json.loads(registry_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        adc,
        "get_ladder_tier_config",
        _fake_get_ladder_tier_config,
    )
    monkeypatch.setattr(
        adc,
        "load_square8_two_player_registry",
        _fake_load_registry,
    )

    output_json = eval_root / "calibration_summary.square8_2p.json"
    output_md = eval_root / "calibration_summary.square8_2p.md"

    argv: list[str] = [
        "--calibration-aggregates",
        os.fspath(calib_path),
        "--registry-path",
        os.fspath(registry_path),
        "--eval-root",
        os.fspath(eval_root),
        "--output-json",
        os.fspath(output_json),
        "--output-md",
        os.fspath(output_md),
        "--window-label",
        "2025-11",
    ]

    rc = adc.main(argv)
    assert rc == 0
    assert output_json.exists()
    assert output_md.exists()

    # Load and inspect summary JSON.
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["board"] == "square8"
    assert summary["num_players"] == 2
    assert summary["window"]["label"] == "2025-11"

    tiers = summary.get("tiers") or []
    d4_rows = [t for t in tiers if t.get("tier") == "D4"]
    assert len(d4_rows) == 1
    d4 = d4_rows[0]

    # Ladder view should reflect the stubbed current_d4 model.
    ladder = d4.get("ladder") or {}
    assert ladder.get("model_id") == "current_d4"

    # Calibration segments should be present with sample_ok flags.
    calib = d4.get("calibration") or {}
    segments = calib.get("segments") or []
    assert segments
    seg_by_name = {seg["segment"]: seg for seg in segments}
    assert seg_by_name["new"]["sample_ok"] is True
    assert seg_by_name["intermediate"]["sample_ok"] is True

    overall_status = calib.get("overall_status")
    assert overall_status in {
        "too_hard",
        "too_easy",
        "mixed",
        "in_band",
        "inconclusive",
    }

    # Evaluation and perf views should be wired from artefacts.
    evaluation = d4.get("evaluation") or {}
    assert evaluation.get("overall_pass") is True
    win_rates = evaluation.get("win_rates") or {}
    assert win_rates.get("vs_baseline") == pytest.approx(0.82)

    perf = d4.get("perf") or {}
    assert perf.get("overall_pass") is True
    assert perf.get("avg_ms") == pytest.approx(95.0)
    assert perf.get("p95_ms") == pytest.approx(170.0)

    # Markdown should contain tier header, ladder model id, and a segment row.
    md_text = output_md.read_text(encoding="utf-8")
    assert "Tier D4" in md_text
    assert "current_d4" in md_text
    assert "| new |" in md_text or "| new  |" in md_text


def test_analyze_difficulty_calibration_validation_error(
    tmp_path: Path,
) -> None:
    """Invalid board/num_players should cause a non-zero exit
    and no outputs."""
    bad_path = tmp_path / "bad_calibration_aggregates.json"
    bad_payload = {
        "board": "hex",
        "num_players": 2,
        "window": {},
        "tiers": [],
    }
    _write_json(bad_path, bad_payload)

    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    argv: list[str] = [
        "--calibration-aggregates",
        os.fspath(bad_path),
        "--output-json",
        os.fspath(output_json),
        "--output-md",
        os.fspath(output_md),
    ]

    rc = adc.main(argv)
    assert rc != 0
    assert not output_json.exists()
    assert not output_md.exists()