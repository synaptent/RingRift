from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.models import BoardType
from app.training.env import TrainingEnvConfig, make_env
from scripts.ingest_policy_selfplay import ingest_policy_selfplay_files


def _write_policy_jsonl(path: Path) -> None:
    env = make_env(TrainingEnvConfig(board_type=BoardType.HEX8, num_players=2, max_moves=120))
    state = env.reset(seed=321)
    moves: list[dict[str, object]] = []

    for move_idx in range(20):
        legal_moves = env.legal_moves()
        assert legal_moves
        move = legal_moves[0]
        move_payload = move.model_dump(by_alias=True, exclude_none=True, mode="json")
        move_payload["mcts_policy"] = {"0": 0.7, "1": 0.3}
        move_payload["policy_target"] = move_idx % 4 != 0
        moves.append(move_payload)
        state, _reward, done, _info = env.step(move)
        if done:
            break

    payload = {
        "game_id": "policy-ingest-1",
        "board_type": "hex8",
        "num_players": 2,
        "winner": state.winner or 1,
        "status": getattr(state.game_status, "value", state.game_status),
        "num_moves": len(moves),
        "moves": moves,
        "timestamp": "2026-04-12T00:00:00Z",
        "provenance": {
            "source": "gumbel_selfplay",
            "engine_mode": "gumbel-mcts",
            "model_sha": "abc123",
            "node_id": "gh200-11",
            "opponent_type": "selfplay",
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_ingest_policy_selfplay_creates_supplemental_npz(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")
    monkeypatch.setenv("RINGRIFT_MIN_MOVES", "1")

    jsonl_path = tmp_path / "input.jsonl"
    _write_policy_jsonl(jsonl_path)

    output_dir = tmp_path / "supplemental"
    state_dir = tmp_path / "state"

    summary = ingest_policy_selfplay_files(
        input_paths=[jsonl_path],
        output_dir=output_dir,
        state_dir=state_dir,
        board_type="hex8",
        num_players=2,
        completion_rate_threshold=0.0,
    )

    output_npz = Path(summary.output_npz)
    assert output_npz.exists()
    assert summary.games_kept == 1
    assert summary.policy_target_moves > 0

    with np.load(output_npz, allow_pickle=True) as data:
        assert "supplemental_manifest_json" in data.files
        manifest = json.loads(str(data["supplemental_manifest_json"]))
        assert manifest["games_kept"] == 1
        assert manifest["provenance"]["node_ids"] == ["gh200-11"]

    with pytest.raises(ValueError, match="No policy-bearing selfplay records"):
        ingest_policy_selfplay_files(
            input_paths=[jsonl_path],
            output_dir=output_dir,
            state_dir=state_dir,
            board_type="hex8",
            num_players=2,
            completion_rate_threshold=0.0,
        )
