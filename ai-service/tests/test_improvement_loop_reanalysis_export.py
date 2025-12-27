from __future__ import annotations

import sys
from pathlib import Path

from scripts import run_improvement_loop as loop


def test_export_training_data_uses_played_export(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(loop, "AI_SERVICE_ROOT", tmp_path)

    captured = {}

    def fake_run_command(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return 0, "", ""

    monkeypatch.setattr(loop, "run_command", fake_run_command)

    config = {
        "board": "square8",
        "players": 2,
        "replay_db": str(tmp_path / "data" / "games" / "canonical_square8.db"),
        "dataset_policy_target": "played",
        "dataset_max_games": 42,
    }

    ok, _out_path = loop.export_training_data(config, iteration=0, dry_run=False)
    assert ok
    cmd = captured["cmd"]
    # Use the running interpreter for reproducibility across envs.
    assert cmd[0] == sys.executable
    assert "scripts/export_replay_dataset.py" in cmd
    assert "--max-games" in cmd
    assert cmd[cmd.index("--max-games") + 1] == "42"


def test_export_training_data_uses_reanalysis_when_requested(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(loop, "AI_SERVICE_ROOT", tmp_path)

    # Create a best-model file so nn_model_id can be inferred.
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "square8_2p_best.pth").write_bytes(b"ok")

    captured = {}

    def fake_run_command(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return 0, "", ""

    monkeypatch.setattr(loop, "run_command", fake_run_command)

    config = {
        "board": "square8",
        "players": 2,
        "replay_db": str(tmp_path / "data" / "games" / "canonical_square8.db"),
        "dataset_policy_target": "mcts_visits",
        "policy_search_think_time_ms": 25,
        "policy_temperature": 0.75,
    }

    ok, _out_path = loop.export_training_data(config, iteration=0, dry_run=False)
    assert ok
    cmd = captured["cmd"]
    # Use the running interpreter for reproducibility across envs.
    assert cmd[0] == sys.executable
    assert "scripts/reanalyze_replay_dataset.py" in cmd
    assert cmd[cmd.index("--policy-target") + 1] == "mcts_visits"
    assert cmd[cmd.index("--policy-search-think-time-ms") + 1] == "25"
    assert cmd[cmd.index("--policy-temperature") + 1] == "0.75"
    assert cmd[cmd.index("--nn-model-id") + 1] == "square8_2p_best"

