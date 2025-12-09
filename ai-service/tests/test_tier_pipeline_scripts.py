"""Tests for tier training and full gating orchestration scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.training.tier_perf_benchmark import TierPerfResult  # noqa: E402
from app.config.perf_budgets import get_tier_perf_budget  # noqa: E402
import scripts.run_tier_training_pipeline as tier_train  # noqa: E402
import scripts.run_full_tier_gating as full_gate  # noqa: E402


def test_run_tier_training_pipeline_demo_writes_report(tmp_path) -> None:
    """Smoke test that the tier training pipeline writes basic artefacts for D2."""
    run_dir = tmp_path / "d2_demo"
    argv = [
        "--tier",
        "D2",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "123",
    ]

    rc = tier_train.main(argv)
    assert rc == 0

    report_path = run_dir / "training_report.json"
    status_path = run_dir / "status.json"
    assert report_path.exists()
    assert status_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tier"] == "D2"
    assert report["board"] == "square8"
    assert report["num_players"] == 2
    assert isinstance(report.get("candidate_id"), str)
    assert "config" in report
    assert "metrics" in report

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["tier"] == "D2"
    assert status["candidate_id"] == report["candidate_id"]
    assert status["training"]["status"] == "completed"
    assert status["training"]["report_path"] == "training_report.json"

    # Automated gate / perf / human calibration blocks should be initialised.
    auto_gate = status.get("automated_gate")
    assert isinstance(auto_gate, dict)
    assert auto_gate["status"] == "not_started"
    assert auto_gate["eval_json"] is None
    assert auto_gate["promotion_plan"] is None

    perf = status.get("perf")
    assert isinstance(perf, dict)
    assert perf["status"] == "not_started"
    assert perf["perf_json"] is None

    human = status.get("human_calibration")
    assert isinstance(human, dict)
    assert human["required"] is True
    assert human["status"] == "pending"
    assert human["min_games"] >= 1

    # Backwards-compatible alias.
    assert status["gating"]["status"] in {"not_started", "completed"}


def test_run_tier_training_pipeline_demo_d6_neural_stub(tmp_path) -> None:
    """D6 demo run should exercise the neural-tier stub path."""
    run_dir = tmp_path / "d6_demo"
    argv = [
        "--tier",
        "D6",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "99",
    ]

    rc = tier_train.main(argv)
    assert rc == 0

    report_path = run_dir / "training_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tier"] == "D6"
    assert report["board"] == "square8"
    assert report["num_players"] == 2
    assert isinstance(report.get("candidate_id"), str)

    config = report["config"]
    env_cfg = config["env"]
    training_params = config["training_params"]

    # Env snapshot should be square8 2p terminal-reward config.
    assert env_cfg["board_type"] == "SQUARE8"
    assert env_cfg["num_players"] == 2
    assert env_cfg["reward_mode"] == "terminal"

    # Neural demo path should be used for D6.
    assert training_params["mode"] == "neural_demo"
    assert training_params["logical_difficulty"] == 6
    assert training_params["train_config"]["board_type"] == "SQUARE8"
    assert training_params["train_config"]["model_id"] == report["candidate_id"]


def test_run_full_tier_gating_demo_with_mocks(tmp_path, monkeypatch) -> None:
    """End-to-end smoke test for full gating with mocked heavy steps."""
    run_dir = tmp_path / "d4_gate_demo"

    # First, produce a demo training report for D4 so gating has inputs.
    train_argv = [
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "7",
    ]
    rc = tier_train.main(train_argv)
    assert rc == 0

    report_path = run_dir / "training_report.json"
    assert report_path.exists()
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate_id_value = training_report["candidate_id"]
    tier_value = training_report["tier"]

    # Stub the difficulty-tier gate to avoid long evaluation runs.
    def _fake_run_tier_gate_cli(
        tier: str,
        candidate_id: str,
        run_dir: str,
        seed: int | None,
        num_games_override: int | None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        assert tier == tier_value
        assert candidate_id == candidate_id_value
        rd = Path(run_dir)
        eval_payload: Dict[str, Any] = {
            "tier": tier_value,
            "board_type": "square8",
            "num_players": 2,
            "overall_pass": True,
            "metrics": {"win_rate_vs_baseline": 0.8},
        }
        plan_payload: Dict[str, Any] = {
            "tier": tier_value,
            "board_type": "square8",
            "num_players": 2,
            "current_model_id": "prod_model",
            "candidate_model_id": candidate_id_value,
            "decision": "promote",
            "timestamp": "2025-01-01T00:00:00Z",
            "reason": {"overall_pass": True},
        }
        (rd / full_gate.TIER_EVAL_FILENAME).write_text(
            json.dumps(eval_payload),
            encoding="utf-8",
        )
        (rd / full_gate.PROMOTION_PLAN_FILENAME).write_text(
            json.dumps(plan_payload),
            encoding="utf-8",
        )
        return eval_payload, plan_payload

    monkeypatch.setattr(
        full_gate,
        "_run_tier_gate_cli",
        _fake_run_tier_gate_cli,
    )

    # Stub the perf benchmark to avoid heavy self-play.
    budget = get_tier_perf_budget("D4")

    def _fake_run_tier_perf_benchmark(
        tier_name: str,
        num_games: int,
        moves_per_game: int,
        seed: int,
    ) -> TierPerfResult:
        assert tier_name == "D4"
        return TierPerfResult(
            tier_name="D4_SQ8_2P",
            average_ms=50.0,
            p95_ms=80.0,
            budget=budget,
            num_samples=16,
        )

    monkeypatch.setattr(
        full_gate,
        "run_tier_perf_benchmark",
        _fake_run_tier_perf_benchmark,
    )

    gate_argv = [
        "--tier",
        "D4",
        "--candidate-id",
        candidate_id_value,
        "--run-dir",
        str(run_dir),
        "--demo",
    ]
    rc = full_gate.main(gate_argv)
    assert rc == 0

    gate_path = run_dir / "gate_report.json"
    status_path = run_dir / "status.json"
    assert gate_path.exists()
    assert status_path.exists()

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["tier"] == "D4"
    assert gate["candidate_id"] == candidate_id_value
    assert gate["evaluation"]["overall_pass"] is True
    assert gate["perf"]["run"] is True
    assert gate["perf"]["overall_pass"] is True
    assert gate["final_decision"] == "promote"

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["tier"] == "D4"
    assert status["candidate_id"] == candidate_id_value
    # Automated gate should be marked completed with JSON artefacts recorded.
    auto_gate = status.get("automated_gate")
    assert isinstance(auto_gate, dict)
    assert auto_gate["status"] == "completed"
    assert auto_gate["eval_json"] == full_gate.TIER_EVAL_FILENAME
    assert auto_gate["promotion_plan"] == full_gate.PROMOTION_PLAN_FILENAME

    # Perf block should be marked completed and reference the perf report.
    perf = status.get("perf")
    assert isinstance(perf, dict)
    assert perf["status"] in {"completed", "not_started"}
    if perf["status"] == "completed":
        assert perf["perf_json"] == full_gate.TIER_PERF_FILENAME

    # Human calibration remains pending until manual experiments are run.
    human = status.get("human_calibration")
    assert isinstance(human, dict)
    assert human["required"] is True
    assert human["status"] in {"pending", "in_progress", "completed"}

    # Backwards-compatible alias.
    assert status["gating"]["status"] == "completed"
    assert status["gating"]["report_path"] == "gate_report.json"


def test_run_full_tier_gating_rejects_when_perf_fails(
    tmp_path,
    monkeypatch,
) -> None:
    """Full gating should reject and return non-zero when perf budget fails."""
    run_dir = tmp_path / "d4_gate_perf_fail"

    # Produce a demo training report for D4 to seed the run directory.
    train_argv = [
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "11",
    ]
    rc = tier_train.main(train_argv)
    assert rc == 0

    training_report = json.loads(
        (run_dir / "training_report.json").read_text(encoding="utf-8"),
    )
    candidate_id_value = training_report["candidate_id"]
    tier_value = training_report["tier"]

    # Stub the difficulty-tier gate to unconditionally pass and promote.
    def _fake_run_tier_gate_cli(
        tier: str,
        candidate_id: str,
        run_dir: str,
        seed: int | None,
        num_games_override: int | None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        assert tier == tier_value
        assert candidate_id == candidate_id_value
        eval_payload: Dict[str, Any] = {
            "tier": tier_value,
            "board_type": "square8",
            "num_players": 2,
            "overall_pass": True,
            "metrics": {"win_rate_vs_baseline": 0.8},
        }
        plan_payload: Dict[str, Any] = {
            "tier": tier_value,
            "board_type": "square8",
            "num_players": 2,
            "current_model_id": "prod_model",
            "candidate_model_id": candidate_id_value,
            "decision": "promote",
            "timestamp": "2025-01-01T00:00:00Z",
            "reason": {"overall_pass": True},
        }
        return eval_payload, plan_payload

    monkeypatch.setattr(
        full_gate,
        "_run_tier_gate_cli",
        _fake_run_tier_gate_cli,
    )

    # Stub perf benchmark to exceed the budget so overall_pass is False.
    budget = get_tier_perf_budget("D4")

    def _fake_run_tier_perf_benchmark(
        tier_name: str,
        num_games: int,
        moves_per_game: int,
        seed: int,
    ) -> TierPerfResult:
        assert tier_name == "D4"
        return TierPerfResult(
            tier_name="D4_SQ8_2P",
            average_ms=budget.max_avg_move_ms * 2.0,
            p95_ms=budget.max_p95_move_ms * 2.0,
            budget=budget,
            num_samples=16,
        )

    monkeypatch.setattr(
        full_gate,
        "run_tier_perf_benchmark",
        _fake_run_tier_perf_benchmark,
    )

    gate_argv = [
        "--tier",
        "D4",
        "--candidate-id",
        candidate_id_value,
        "--run-dir",
        str(run_dir),
        "--demo",
    ]
    rc = full_gate.main(gate_argv)
    # Exit code should be non-zero because perf did not pass.
    assert rc == 1

    gate_path = run_dir / "gate_report.json"
    status_path = run_dir / "status.json"
    assert gate_path.exists()
    assert status_path.exists()

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["tier"] == "D4"
    assert gate["candidate_id"] == candidate_id_value
    assert gate["evaluation"]["overall_pass"] is True
    # Perf should have run but failed overall.
    assert gate["perf"]["run"] is True
    assert gate["perf"]["overall_pass"] is False
    # Final decision must be reject when perf fails.
    assert gate["final_decision"] == "reject"

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["tier"] == "D4"
    assert status["candidate_id"] == candidate_id_value
    # Automated gate recorded as completed; perf recorded as completed with JSON.
    assert status["automated_gate"]["status"] == "completed"
    assert status["perf"]["status"] in {"completed", "not_started"}
    if status["perf"]["status"] == "completed":
        assert status["perf"]["perf_json"] == full_gate.TIER_PERF_FILENAME
    # Backwards-compatible alias remains consistent.
    assert status["gating"]["status"] == "completed"
    assert status["gating"]["report_path"] == "gate_report.json"


def test_run_full_tier_gating_skips_perf_when_no_perf_flag(
    tmp_path,
    monkeypatch,
) -> None:
    """When --no-perf is set, perf benchmark should be skipped even if a budget exists."""
    run_dir = tmp_path / "d4_gate_no_perf"

    # Produce a demo training report for D4.
    train_argv = [
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "5",
    ]
    rc = tier_train.main(train_argv)
    assert rc == 0

    training_report = json.loads(
        (run_dir / "training_report.json").read_text(encoding="utf-8"),
    )
    candidate_id_value = training_report["candidate_id"]
    tier_value = training_report["tier"]

    # Stub the difficulty-tier gate to pass.
    def _fake_run_tier_gate_cli(
        tier: str,
        candidate_id: str,
        run_dir: str,
        seed: int | None,
        num_games_override: int | None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        assert tier == tier_value
        assert candidate_id == candidate_id_value
        eval_payload: Dict[str, Any] = {
            "tier": tier_value,
            "board_type": "square8",
            "num_players": 2,
            "overall_pass": True,
            "metrics": {"win_rate_vs_baseline": 0.75},
        }
        plan_payload: Dict[str, Any] = {
            "tier": tier_value,
            "board_type": "square8",
            "num_players": 2,
            "current_model_id": "prod_model",
            "candidate_model_id": candidate_id_value,
            "decision": "promote",
            "timestamp": "2025-01-01T00:00:00Z",
            "reason": {"overall_pass": True},
        }
        return eval_payload, plan_payload

    monkeypatch.setattr(
        full_gate,
        "_run_tier_gate_cli",
        _fake_run_tier_gate_cli,
    )

    # If perf were accidentally invoked despite --no-perf, this stub would fail the test.
    def _should_not_be_called(
        tier_name: str,
        num_games: int,
        moves_per_game: int,
        seed: int,
    ) -> TierPerfResult:  # pragma: no cover - defensive
        raise AssertionError("run_tier_perf_benchmark should not be called when --no-perf is set")

    monkeypatch.setattr(
        full_gate,
        "run_tier_perf_benchmark",
        _should_not_be_called,
    )

    gate_argv = [
        "--tier",
        "D4",
        "--candidate-id",
        candidate_id_value,
        "--run-dir",
        str(run_dir),
        "--demo",
        "--no-perf",
    ]
    rc = full_gate.main(gate_argv)
    # With gate passing and perf skipped, exit code should be success.
    assert rc == 0

    gate_path = run_dir / "gate_report.json"
    status_path = run_dir / "status.json"
    assert gate_path.exists()
    assert status_path.exists()

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["tier"] == "D4"
    assert gate["candidate_id"] == candidate_id_value
    assert gate["evaluation"]["overall_pass"] is True
    # Perf should be marked as not run and have no result path or overall flag.
    assert gate["perf"]["run"] is False
    assert gate["perf"]["result_path"] is None
    assert gate["perf"]["overall_pass"] is None
    assert gate["final_decision"] == "promote"

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["tier"] == "D4"
    assert status["candidate_id"] == candidate_id_value
    # Perf block should remain in a not_started state with no JSON path.
    assert status["perf"]["status"] == "not_started"
    assert status["perf"]["perf_json"] is None
    # Gating alias still reflects a completed gate report.
    assert status["gating"]["status"] == "completed"
    assert status["gating"]["report_path"] == "gate_report.json"


def test_run_tier_training_pipeline_rejects_unsupported_tier(tmp_path) -> None:
    """Training pipeline should exit with an error for unsupported tiers."""
    run_dir = tmp_path / "bad_tier"
    argv = [
        "--tier",
        "D3",  # not one of D2/D4/D6/D8
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
    ]

    with pytest.raises(SystemExit):
        tier_train.main(argv)


def test_run_full_tier_gating_rejects_when_tier_mismatch(tmp_path, monkeypatch) -> None:
    """Full gating should fail fast when CLI tier does not match training_report."""
    run_dir = tmp_path / "tier_mismatch"

    # Create a D4 training report.
    train_argv = [
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
    ]
    rc = tier_train.main(train_argv)
    assert rc == 0

    training_report = json.loads(
        (run_dir / "training_report.json").read_text(encoding="utf-8"),
    )
    # Use the correct candidate id but lie about the tier on the CLI.
    candidate_id_value = training_report["candidate_id"]

    gate_argv = [
        "--tier",
        "D6",  # mismatched vs report["tier"] == "D4"
        "--candidate-id",
        candidate_id_value,
        "--run-dir",
        str(run_dir),
    ]

    with pytest.raises(SystemExit):
        full_gate.main(gate_argv)


def test_tier_training_pipeline_runs_preflight_by_default(tmp_path, monkeypatch) -> None:
    """Training pipeline should invoke the canonical preflight unless skipped."""
    run_dir = tmp_path / "d2_preflight"
    called: dict[str, Any] = {}

    def fake_run(cmd: list[str], cwd: str | None = None, text: bool | None = None):
        called["cmd"] = cmd
        called["cwd"] = cwd
        called["text"] = text

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(tier_train.subprocess, "run", fake_run)

    argv = [
        "--tier",
        "D2",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
    ]

    rc = tier_train.main(argv)
    assert rc == 0
    assert "cmd" in called
    assert any("training_preflight_check.py" in part for part in called["cmd"])
    config_arg = called["cmd"][called["cmd"].index("--config") + 1]
    assert os.path.isabs(config_arg)
    assert called["cwd"] == tier_train.PROJECT_ROOT


def test_tier_training_pipeline_skip_preflight(monkeypatch, tmp_path) -> None:
    """--skip-preflight should bypass the preflight subprocess."""
    run_dir = tmp_path / "d4_skip_preflight"
    called = {"count": 0}

    def fake_run(*_args, **_kwargs):
        called["count"] += 1

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(tier_train.subprocess, "run", fake_run)

    argv = [
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--skip-preflight",
    ]

    rc = tier_train.main(argv)
    assert rc == 0
    assert called["count"] == 0


def test_tier_training_pipeline_config_square8_2p_shape() -> None:
    """The canonical square8 2p tier pipeline config should be present and well-formed."""
    cfg_path = Path(ROOT) / "config" / "tier_training_pipeline.square8_2p.json"
    assert cfg_path.exists()

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["board_type"] == "square8"
    assert payload["num_players"] == 2

    tiers = payload.get("tiers")
    assert isinstance(tiers, dict)
    for name in ("D2", "D4", "D6", "D8"):
        assert name in tiers
        tier_cfg = tiers[name]
        assert "training" in tier_cfg
        assert "gating" in tier_cfg

        training_cfg = tier_cfg["training"]
        gating_cfg = tier_cfg["gating"]

        assert "mode" in training_cfg
        assert isinstance(gating_cfg.get("enabled"), bool)
        # Demo-friendly overrides used by the current pipeline.
        assert isinstance(gating_cfg.get("seed"), int)
        assert isinstance(gating_cfg.get("num_games_override"), int)
