from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from app.models import BoardType
from app.training import game_gauntlet


def test_evaluation_progress_payload_includes_config_key_and_board_type(monkeypatch):
    mock_router = MagicMock()

    import app.coordination.event_router as event_router

    monkeypatch.setattr(event_router, "get_router", lambda: mock_router)

    monkeypatch.setattr(game_gauntlet, "create_neural_ai", lambda *args, **kwargs: object())
    monkeypatch.setattr(game_gauntlet, "create_baseline_ai", lambda *args, **kwargs: object())

    def fake_play_single_game(**kwargs):
        return SimpleNamespace(
            candidate_won=True,
            winner=1,
            victory_reason="test",
            move_count=1,
        )

    monkeypatch.setattr(game_gauntlet, "play_single_game", fake_play_single_game)

    game_gauntlet._evaluate_single_opponent(
        baseline=game_gauntlet.BaselineOpponent.RANDOM,
        model_path="dummy",
        board_type=BoardType.SQUARE19,
        games_per_opponent=1,
        num_players=2,
        verbose=False,
        model_getter=None,
        model_type="cnn",
        early_stopping=False,
        early_stopping_confidence=0.95,
        early_stopping_min_games=1,
        parallel_games=1,
    )

    mock_router.publish_sync.assert_called()
    matching_calls = [
        call
        for call in mock_router.publish_sync.call_args_list
        if len(call.args) >= 2 and call.args[1].get("config_key") == "square19_2p"
    ]
    assert matching_calls, "Expected evaluation progress events to be published"
    payload = matching_calls[-1].args[1]
    assert payload["config_key"] == "square19_2p"
    assert payload["board_type"] == "square19"
