from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from app.models import BoardType
from app.training import game_gauntlet


def test_evaluation_progress_payload_includes_config_key_and_board_type(monkeypatch):
    captured_events = []

    class _InterceptBus:
        """Wraps real bus but captures publish_sync calls."""
        def __init__(self, real_bus):
            self._real = real_bus
        def publish_sync(self, event):
            captured_events.append(event)
            return event
        def __getattr__(self, name):
            return getattr(self._real, name)

    # Wrap whatever bus get_event_bus returns (real or mock) with our interceptor
    import app.coordination.event_router as event_router
    original_get = event_router.get_event_bus
    intercept_bus = None

    def intercepting_get():
        nonlocal intercept_bus
        real = original_get()
        if real is None:
            real = MagicMock()
        if intercept_bus is None:
            intercept_bus = _InterceptBus(real)
        return intercept_bus

    monkeypatch.setattr(event_router, "get_event_bus", intercepting_get)

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

    assert captured_events, "Expected evaluation progress events to be published"
    progress_event = next(
        event
        for event in captured_events
        if getattr(event, "payload", {}).get("config_key") == "square19_2p"
    )
    payload = progress_event.payload
    assert payload["config_key"] == "square19_2p"
    assert payload["board_type"] == "square19"
