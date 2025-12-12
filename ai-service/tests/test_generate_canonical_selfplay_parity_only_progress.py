from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.generate_canonical_selfplay as gen


class _FakeConn:
    def execute(self, _sql: str):
        cur = MagicMock()
        cur.fetchone.return_value = {"n": 6001}
        return cur


class _FakeConnCM:
    def __enter__(self):
        return _FakeConn()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDB:
    def __init__(self, _path: str):
        pass

    def _get_conn(self):
        return _FakeConnCM()


def test_parity_only_mode_passes_progress_every_and_warns_on_large_db(
    tmp_path, capsys
) -> None:
    db_path = tmp_path / "canonical_square8.db"
    db_path.write_bytes(b"not-a-real-sqlite-db")
    summary_path = db_path.with_suffix(db_path.suffix + ".parity_gate.json")
    summary_path.write_text("{}", encoding="utf-8")

    captured_cmd: list[str] = []

    def fake_run_cmd(cmd, **_kwargs):
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return proc

    with patch.object(gen, "GameReplayDB", _FakeDB):
        with patch.object(gen, "_run_cmd", side_effect=fake_run_cmd):
            out = gen.run_selfplay_and_parity(
                board_type="square8",
                num_games=0,
                db_path=db_path,
                num_players=2,
                parity_limit_games_per_db=0,
            )

    assert out["returncode"] == 0
    assert "--progress-every" in captured_cmd
    assert "25" in captured_cmd

    stderr = capsys.readouterr().err
    assert "parity-only mode will check ALL games" in stderr

