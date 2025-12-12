from __future__ import annotations

import json

from scripts import validate_and_promote_weights as v  # type: ignore[import]


def test_validate_weights_all_draws_is_inconclusive(monkeypatch, tmp_path) -> None:
    """All-draw outcomes should not be treated as a statistical rejection."""
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps({"weights": {}}), encoding="utf-8")

    def _fake_play_games(*_args, **_kwargs):
        return 0, 10, 0

    monkeypatch.setattr(v, "play_validation_games", _fake_play_games)

    result = v.validate_weights(
        str(candidate_path),
        num_games=10,
        num_players=2,
        min_win_rate=0.52,
        confidence=0.95,
    )

    assert result.games_played == 10
    assert result.candidate_wins == 0
    assert result.default_wins == 0
    assert result.draws == 10
    assert result.is_significant is False
    assert result.recommendation == "INCONCLUSIVE"

