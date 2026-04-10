#!/usr/bin/env bash
# Product smoke gate for the supported RingRift path.
#
# Defaults to production. Override with:
#   BASE_URL=http://localhost:3000 AI_BASE_URL=http://localhost:8001 bash scripts/product_smoke_test.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$ROOT_DIR/ai-service"
BASE_URL="${BASE_URL:-https://ringrift.ai}"
AI_BASE_URL="${AI_BASE_URL:-}"
CURL_TIMEOUT="${CURL_TIMEOUT:-120}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

usage() {
  cat <<EOF
Usage: BASE_URL=https://ringrift.ai [AI_BASE_URL=http://localhost:8001] $0

Options:
  --base-url URL     Web app base URL (default: $BASE_URL)
  --ai-base-url URL  Direct AI service base URL; if omitted, /api/replay/stats is used as the AI proxy check
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --ai-base-url)
      AI_BASE_URL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

curl_json() {
  local method="$1"
  local url="$2"
  local body="${3:-}"
  if [[ -n "$body" ]]; then
    curl -fsS --max-time "$CURL_TIMEOUT" -X "$method" \
      -H "Content-Type: application/json" \
      --data @"$body" \
      "$url"
  else
    curl -fsS --max-time "$CURL_TIMEOUT" -X "$method" "$url"
  fi
}

echo "==> Generating smoke payloads"
cd "$AI_DIR"
PYTHONPATH=. python3 - "$TMP_DIR" <<'PY'
import json
import sys
import uuid
from pathlib import Path

from app.game_engine import GameEngine
from app.models import BoardType, GameStatus
from app.training.initial_state import create_initial_state

out_dir = Path(sys.argv[1])

ai_state = create_initial_state(BoardType.HEX8, num_players=2).model_copy(
    update={"id": "product-smoke-ai"}
)
(out_dir / "ai_move_payload.json").write_text(json.dumps({
    "state": ai_state.model_dump(by_alias=True, mode="json"),
    "difficulty": 10,
    "playerNumber": 1,
}))

game_id = f"product-smoke-{uuid.uuid4()}"
state = create_initial_state(BoardType.HEX8, num_players=2).model_copy(update={"id": game_id})
initial_state = state.model_dump(by_alias=True, mode="json")
moves = []
for _ in range(5):
    legal = GameEngine.get_valid_moves(state, state.current_player)
    if legal:
        move = legal[0]
    else:
        requirement = GameEngine.get_phase_requirement(state, state.current_player)
        if requirement is None:
            raise RuntimeError("Could not synthesize enough smoke moves")
        move = GameEngine.synthesize_bookkeeping_move(requirement, state)
    moves.append(move.model_dump(by_alias=True, mode="json", exclude_none=True))
    state = GameEngine.apply_move(state, move, trace_mode=True)

final_state = state.model_copy(update={"game_status": GameStatus.COMPLETED, "winner": 1})
(out_dir / "replay_payload.json").write_text(json.dumps({
    "gameId": game_id,
    "initialState": initial_state,
    "finalState": final_state.model_dump(by_alias=True, mode="json"),
    "moves": moves,
    "metadata": {
        "source": "sandbox_smoke_quarantine",
        "playerTypes": ["human", "ai"],
        "parity_status": "smoke_test_excluded",
        "excluded_from_training": True,
        "termination_reason": "ring_elimination",
        "smoke": True,
    },
}))
PY

echo "==> Server health: $BASE_URL/health"
curl_json GET "$BASE_URL/health" >"$TMP_DIR/server_health.json"

if [[ -n "$AI_BASE_URL" ]]; then
  echo "==> AI health: $AI_BASE_URL/health"
  curl_json GET "$AI_BASE_URL/health" >"$TMP_DIR/ai_health.json"
else
  echo "==> AI proxy health: $BASE_URL/api/replay/stats"
  curl_json GET "$BASE_URL/api/replay/stats" >"$TMP_DIR/ai_health.json"
fi

echo "==> Sandbox AI move"
curl_json POST "$BASE_URL/api/games/sandbox/ai/move" "$TMP_DIR/ai_move_payload.json" >"$TMP_DIR/ai_move_response.json"
python3 - "$TMP_DIR/ai_move_response.json" <<'PY'
import json
import sys

response = json.loads(open(sys.argv[1]).read())
ai_type = response.get("aiType") or response.get("ai_type")
if not ai_type:
    raise SystemExit(f"AI response missing ai_type: {response}")
print(f"  ai_type={ai_type} neural={response.get('useNeuralNet')}")
PY

echo "==> Replay store smoke"
curl_json POST "$BASE_URL/api/replay/games" "$TMP_DIR/replay_payload.json" >"$TMP_DIR/replay_response.json"
python3 - "$TMP_DIR/replay_response.json" <<'PY'
import json
import sys

response = json.loads(open(sys.argv[1]).read())
if response.get("success") is not True:
    raise SystemExit(f"Replay response did not report success: {response}")
print(
    "  gameId={gameId} totalMoves={totalMoves} parity={parityStatus} acceptedForTraining={acceptedForTraining}".format(
        **response
    )
)
PY

echo "==> Local hex8_2p model loadability"
cd "$AI_DIR"
PYTHONPATH=. python3 scripts/ai_inference_smoke.py --in-process --model-check-only --json >"$TMP_DIR/model_check.json"
python3 - "$TMP_DIR/model_check.json" <<'PY'
import json
import sys

lines = [line.strip() for line in open(sys.argv[1]) if line.strip()]
response = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
if not response.get("ok") or not response.get("model_loaded"):
    raise SystemExit(f"Model loadability check failed: {response}")
print(f"  checkpoint={response.get('checkpoint')}")
PY

echo "Product smoke passed."
