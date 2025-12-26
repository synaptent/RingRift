from __future__ import annotations

from scripts import run_improvement_loop as loop


def test_run_selfplay_passes_difficulty_band(monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return 0, "", ""

    monkeypatch.setattr(loop, "run_command", fake_run_command)

    config = {
        "board": "square8",
        "players": 2,
        "games_per_iter": 3,
        "max_moves": 200,
        "replay_db": "data/games/canonical_square8.db",
        "canonical_mode": True,
        "gate_summary": "db_health.canonical_square8.json",
        "registry_path": "TRAINING_DATA_REGISTRY.md",
        "allow_pending_gate": True,
        "selfplay_difficulty_band": "canonical",
    }

    ok, games, _staging_db_path = loop.run_selfplay(config, iteration=0, dry_run=True)
    assert ok
    assert games == 3

    cmd = captured["cmd"]
    assert "scripts/generate_canonical_selfplay.py" in cmd
    assert cmd[cmd.index("--difficulty-band") + 1] == "canonical"

