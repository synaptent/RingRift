import pytest


@pytest.mark.parametrize(
    "board_type,num_players",
    [
        ("square8", 2),
        ("square8", 3),
        ("square8", 4),
        ("square19", 2),
        ("square19", 3),
        ("square19", 4),
        ("hexagonal", 2),
        ("hexagonal", 3),
        ("hexagonal", 4),
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        {"name": "heuristic", "difficulty": 2, "ai_type": "heuristic", "use_neural_net": False},
        {"name": "minimax", "difficulty": 3, "ai_type": "minimax", "use_neural_net": False},
        {"name": "minimax_nnue", "difficulty": 4, "ai_type": "minimax", "use_neural_net": True},
        {"name": "mcts", "difficulty": 5, "ai_type": "mcts", "use_neural_net": False},
        {"name": "mcts_neural", "difficulty": 6, "ai_type": "mcts", "use_neural_net": True},
        {"name": "descent", "difficulty": 9, "ai_type": "descent", "use_neural_net": False},
        {"name": "descent_neural", "difficulty": 9, "ai_type": "descent", "use_neural_net": True},
    ],
    ids=lambda c: c["name"],
)
def test_ai_smoke_select_move_all_boards(
    board_type: str,
    num_players: int,
    case: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-test that canonical AI tiers can select a move on each board.

    This is intentionally lightweight: we set randomness=1.0 so search-based
    engines can pick a legal move quickly, while still ensuring that neural
    tiers can initialize their neural backends on demand.
    """
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")
    monkeypatch.delenv("RINGRIFT_REQUIRE_NEURAL_NET", raising=False)

    from app.main import _create_ai_instance
    from app.models import AIConfig, AIType, BoardType
    from app.training.generate_data import create_initial_state

    board_enum = BoardType(board_type)
    state = create_initial_state(board_type=board_enum, num_players=num_players)

    # Use canonical best-model aliases when applicable; allow fresh weights so
    # local test runs do not require shipping all checkpoints.
    nn_model_id = None
    if case["ai_type"] in {"mcts", "descent"}:
        if board_type == "square8":
            nn_model_id = f"ringrift_best_sq8_{num_players}p"
        elif board_type == "square19":
            nn_model_id = f"ringrift_best_sq19_{num_players}p"
        else:
            nn_model_id = f"ringrift_best_hex_{num_players}p"

    config = AIConfig(
        difficulty=int(case["difficulty"]),
        randomness=1.0,
        think_time=50,
        rngSeed=1337,
        use_neural_net=bool(case["use_neural_net"]),
        nn_model_id=nn_model_id,
        allow_fresh_weights=True,
        heuristic_eval_mode="light",
        training_move_sample_limit=64,
    )

    ai = _create_ai_instance(AIType(case["ai_type"]), 1, config)
    move = ai.select_move(state)
    assert move is not None

    # Force neural initialization for neural tiers.
    if case["ai_type"] == "mcts" and case["use_neural_net"]:
        assert getattr(ai, "neural_net", None) is not None
        ai.neural_net.evaluate_batch([state])
    elif case["ai_type"] == "descent" and case["use_neural_net"]:
        assert getattr(ai, "neural_net", None) is not None
        ai.evaluate_position(state)
