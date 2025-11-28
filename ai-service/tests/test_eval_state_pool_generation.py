import os
import sys
import tempfile
import unittest

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import GameState, BoardType  # noqa: E402
from app.training.env import RingRiftEnv  # noqa: E402
from scripts.run_self_play_soak import (  # noqa: E402
    _append_state_to_jsonl,
)


class TestSquare8StatePoolExport(unittest.TestCase):
    def test_append_state_to_jsonl_roundtrip(self) -> None:
        """Round-trip a single Square8 GameState via JSONL helper."""
        env = RingRiftEnv(BoardType.SQUARE8)
        state = env.reset()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pool.jsonl")

            _append_state_to_jsonl(path, state)

            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            self.assertEqual(len(lines), 1)

            # Must be parseable back into a GameState instance.
            loaded = GameState.model_validate_json(
                lines[0],
            )  # type: ignore[attr-defined]

            # Sanity-check basic invariants of the exported state.
            self.assertEqual(loaded.board.type, BoardType.SQUARE8)
            # Enum value is lower-case "active"; compare using the enum itself
            # to avoid coupling to serialisation details.
            from app.models import GameStatus  # type: ignore  # noqa: E402

            self.assertEqual(loaded.game_status, GameStatus.ACTIVE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()