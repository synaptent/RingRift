#!/usr/bin/env python3
"""Smoke-test hex8 D10 AI inference.

By default this calls a running Python AI service over HTTP. Use --in-process
when checking local model loadability without a server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType, Move
from app.training.initial_state import create_initial_state


def _move_signature(move: Move) -> str:
    payload = move.model_dump(by_alias=True, mode="json", exclude_none=True)
    payload.pop("id", None)
    payload.pop("timestamp", None)
    return json.dumps(payload, sort_keys=True)


def _assert_legal_move(state: Any, move_payload: dict[str, Any]) -> Move:
    move = Move.model_validate(move_payload)
    legal = GameEngine.get_valid_moves(state, state.current_player)
    legal_signatures = {_move_signature(candidate) for candidate in legal}
    if _move_signature(move) not in legal_signatures:
        raise AssertionError(
            f"AI returned non-legal move type={move.type.value} player={move.player}; "
            f"legal_count={len(legal)}"
        )
    return move


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body[:1000]}") from exc


def _build_request(difficulty: int) -> tuple[Any, dict[str, Any]]:
    state = create_initial_state(BoardType.HEX8, num_players=2)
    payload = {
        "game_state": state.model_dump(by_alias=True, mode="json"),
        "player_number": state.current_player,
        "difficulty": difficulty,
        "ai_type": "gumbel_mcts",
        "seed": 20260410,
    }
    return state, payload


def _run_http(args: argparse.Namespace) -> dict[str, Any]:
    state, payload = _build_request(args.difficulty)
    url = args.base_url.rstrip("/") + "/ai/move"
    start = time.time()
    response = _post_json(url, payload, args.timeout)
    latency_ms = int((time.time() - start) * 1000)

    if response.get("ai_type") != "gumbel_mcts":
        raise AssertionError(f"Expected ai_type=gumbel_mcts, got {response.get('ai_type')!r}")
    if response.get("use_neural_net") is not True:
        raise AssertionError(f"Expected use_neural_net=True, got {response.get('use_neural_net')!r}")
    if not response.get("move"):
        raise AssertionError("AI service returned no move")

    move = _assert_legal_move(state, response["move"])
    return {
        "ok": True,
        "mode": "http",
        "base_url": args.base_url,
        "latency_ms": latency_ms,
        "ai_type": response.get("ai_type"),
        "use_neural_net": response.get("use_neural_net"),
        "nn_model_id": response.get("nn_model_id"),
        "nn_checkpoint": response.get("nn_checkpoint"),
        "move_type": move.type.value,
        "difficulty": args.difficulty,
    }


def _run_in_process(args: argparse.Namespace) -> dict[str, Any]:
    from app.main import _create_ai_instance

    state, _payload = _build_request(args.difficulty)
    config = AIConfig(
        difficulty=args.difficulty,
        randomness=0.0,
        think_time=30000,
        rngSeed=20260410,
        nn_model_id="canonical_hex8_2p",
        use_neural_net=True,
        use_gpu_tree=args.use_gpu_tree,
        gpu_tree_eval_mode="nn",
        gumbel_simulation_budget=args.simulation_budget,
    )
    start = time.time()
    ai = _create_ai_instance(AIType.GUMBEL_MCTS, state.current_player, config, board_type=BoardType.HEX8)
    neural_net = getattr(ai, "neural_net", None)
    checkpoint = getattr(neural_net, "loaded_checkpoint_path", None) if neural_net is not None else None
    if neural_net is None or not checkpoint:
        raise AssertionError("Gumbel MCTS did not load the canonical hex8 neural checkpoint")
    if args.model_check_only:
        return {
            "ok": True,
            "mode": "in_process",
            "model_loaded": True,
            "checkpoint": str(checkpoint),
            "latency_ms": int((time.time() - start) * 1000),
        }

    move = ai.select_move(state)
    _assert_legal_move(state, move.model_dump(by_alias=True, mode="json", exclude_none=True))
    return {
        "ok": True,
        "mode": "in_process",
        "model_loaded": True,
        "checkpoint": str(checkpoint),
        "latency_ms": int((time.time() - start) * 1000),
        "ai_type": "gumbel_mcts",
        "use_neural_net": True,
        "move_type": move.type.value,
        "difficulty": args.difficulty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test direct AI inference")
    parser.add_argument("--base-url", default="http://localhost:8001", help="AI service base URL")
    parser.add_argument("--difficulty", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--in-process", action="store_true", help="Load and query the AI in-process")
    parser.add_argument("--model-check-only", action="store_true", help="Only verify model loadability")
    parser.add_argument("--simulation-budget", type=int, default=32, help="In-process Gumbel simulation budget")
    parser.add_argument("--use-gpu-tree", action="store_true", help="Use GPU-tree mode in --in-process smoke")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = _run_in_process(args) if args.in_process else _run_http(args)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"AI inference smoke failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "AI inference smoke passed: "
            f"mode={result['mode']} ai_type={result.get('ai_type', 'n/a')} "
            f"neural={result.get('use_neural_net', result.get('model_loaded'))} "
            f"latency_ms={result['latency_ms']}"
        )


if __name__ == "__main__":
    main()
