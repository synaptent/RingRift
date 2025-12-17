from __future__ import annotations

from pathlib import Path

import pytest

pytest.skip(
    "CurriculumTrainer not implemented - test for planned feature",
    allow_module_level=True
)


def _make_trainer(tmp_path: Path) -> CurriculumTrainer:
    cfg = CurriculumConfig(
        board_type=BoardType.SQUARE8,
        generations=1,
        games_per_generation=0,
        training_epochs=0,
        eval_games=10,
        promotion_threshold=0.7,
        promotion_confidence=0.95,
        output_dir=str(tmp_path),
    )
    trainer = CurriculumTrainer(cfg, base_model_path=None)
    # Keep global models sync inside tmpdir.
    trainer._models_root_dir = lambda: tmp_path / "models"  # type: ignore[method-assign]
    return trainer


def test_curriculum_promotion_requires_wilson_lower_bound(monkeypatch, tmp_path):
    trainer = _make_trainer(tmp_path)

    # Stub out heavy steps.
    monkeypatch.setattr(trainer, "_generate_self_play_data", lambda *_: (tmp_path / "d.npz", 0))
    monkeypatch.setattr(trainer, "_combine_with_history", lambda *_: tmp_path / "c.npz")

    candidate_path = tmp_path / "candidate.pth"
    candidate_path.write_bytes(b"x")
    monkeypatch.setattr(trainer, "_train_candidate", lambda *_: (candidate_path, {"total": 0.0, "policy": 0.0, "value": 0.0}))

    # Small-sample 8/10 should fail Wilson lower bound gate at 95%.
    monkeypatch.setattr(
        trainer,
        "_evaluate_candidate",
        lambda *_: {
            "win_rate": 0.8,
            "draw_rate": 0.0,
            "loss_rate": 0.2,
            "games_played": 10,
            "avg_game_length": 1.0,
            "wins": 8,
            "losses": 2,
            "draws": 0,
        },
    )

    result = trainer.run_generation(0)
    assert result.promoted is False

