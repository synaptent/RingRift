"""Contracts for the supported minimal training loop and legacy comparison."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from app.config.thresholds import (
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_STANDARD,
    get_gauntlet_simulations,
)
from app.models import BoardType
from app.training.board_encoding_contract import get_expected_channels
from app.training.env import TrainingEnvConfig, make_env

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AI_SERVICE_ROOT.parent

BOARD_CONFIGS = [
    ("hex8", 2),
    ("hex8", 3),
    ("hex8", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
]


def _minimal_loop_source() -> str:
    return (AI_SERVICE_ROOT / "scripts" / "minimal_alphazero_loop.py").read_text()


def test_canonical_model_files_exist_for_all_configs() -> None:
    """Every supported board/player config has a canonical checkpoint."""
    for board_type, num_players in BOARD_CONFIGS:
        model_path = AI_SERVICE_ROOT / "models" / f"canonical_{board_type}_{num_players}p.pth"
        assert model_path.exists(), f"Missing canonical model: {model_path}"
        assert model_path.stat().st_size > 0, f"Empty canonical model: {model_path}"


def test_npz_export_produces_expected_schema(tmp_path: Path, monkeypatch) -> None:
    """A tiny legal JSONL self-play record exports to the training NPZ schema."""
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")
    monkeypatch.setenv("RINGRIFT_MIN_MOVES", "1")

    from scripts.jsonl_to_npz import convert_jsonl_to_npz

    env = make_env(TrainingEnvConfig(board_type=BoardType.HEX8, num_players=2, max_moves=120))
    state = env.reset(seed=123)
    moves = []

    for _ in range(20):
        legal_moves = env.legal_moves()
        assert legal_moves, "Generated contract game ran out of legal moves"
        move = legal_moves[0]
        move_payload = move.model_dump(by_alias=True, exclude_none=True, mode="json")
        move_payload.setdefault("mcts_policy", {"0": 1.0})
        moves.append(move_payload)
        state, _reward, done, _info = env.step(move)
        if done:
            break

    jsonl_path = tmp_path / "contract_game.jsonl"
    npz_path = tmp_path / "contract_game.npz"
    game_status = getattr(state.game_status, "value", state.game_status)
    jsonl_path.write_text(
        json.dumps(
            {
                "game_id": "contract_hex8_2p",
                "board_type": "hex8",
                "num_players": 2,
                "winner": state.winner or 1,
                "status": game_status,
                "num_moves": len(moves),
                "moves": moves,
                "timestamp": "2026-04-10T00:00:00Z",
            }
        )
        + "\n"
    )

    stats = convert_jsonl_to_npz(
        [jsonl_path],
        npz_path,
        "hex8",
        players_filter=2,
        max_games=1,
        gpu_selfplay_mode=True,
    )

    assert stats.games_processed == 1
    assert stats.positions_extracted > 0
    assert npz_path.exists()

    expected_keys = {
        "features",
        "globals",
        "values",
        "policy_indices",
        "policy_values",
        "move_numbers",
        "total_game_moves",
        "phases",
        "values_mp",
        "num_players",
        "history_length",
        "feature_version",
        "policy_encoding",
        "encoder_type",
        "base_channels",
        "in_channels",
        "board_type",
        "spatial_size",
        "policy_size",
        "data_schema_version",
    }
    with np.load(npz_path, allow_pickle=True) as data:
        assert expected_keys.issubset(set(data.files))
        assert data["features"].dtype == np.float32
        assert data["features"].shape[0] == stats.positions_extracted
        assert data["features"].shape[1] == get_expected_channels(BoardType.HEX8, "v2")
        assert data["globals"].shape[0] == stats.positions_extracted
        assert data["values"].shape[0] == stats.positions_extracted


def test_gauntlet_eval_budget_matches_threshold_source_of_truth() -> None:
    """Gauntlet budgets are centralized through thresholds.py for all configs."""
    assert GUMBEL_BUDGET_QUALITY == GUMBEL_BUDGET_STANDARD
    for board_type, num_players in BOARD_CONFIGS:
        assert get_gauntlet_simulations(num_players, board_type) == GUMBEL_BUDGET_QUALITY


def test_minimal_loop_staged_promotion_contract_is_explicit() -> None:
    """The minimal loop's staged seat-fair promotion thresholds are stable."""
    tree = ast.parse(_minimal_loop_source())
    stages_2p = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_EVAL_STAGES_2P":
                    stages_2p = ast.literal_eval(node.value)

    assert stages_2p == [
        (50, 0.60, 0.42),
        (100, 0.56, 0.46),
        (200, 0.53, 0.48),
        (400, 0.501, 0.0),
    ]

    source = _minimal_loop_source()
    assert "eval_stages = _get_eval_stages()" in source
    assert "--promote-threshold" in source
    assert 'promoted = ev.get("decision") == "promote"' in source


def test_legacy_training_trigger_deviations_are_documented() -> None:
    """Legacy training dispatch is not silently treated as minimal-loop equivalent."""
    strategy_doc = (REPO_ROOT / "docs" / "architecture" / "TRAINING_INFRASTRUCTURE_STRATEGY.md").read_text()
    assert "Legacy vs Minimal: Contract Comparison" in strategy_doc
    assert "staged seat-fair" in strategy_doc
    assert "scripts/p2p/mixins/training_pipeline_mixin.py" in strategy_doc

    training_actions = (AI_SERVICE_ROOT / "app" / "coordination" / "training_executor_actions.py").read_text()
    assert "candidate_" in training_actions
    assert "PROMOTION_THRESHOLD" not in training_actions
