import json
import os
import sys

import pytest

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai import heuristic_weights as hw  # noqa: E402


def test_load_trained_profiles_propagates_legacy_player_count_aliases(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = dict(hw.HEURISTIC_WEIGHT_PROFILES)

    try:
        # Precondition: board-specific ids that are strict aliases share the same object.
        assert hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_sq8_2p"] is hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_2p"]
        assert hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_hex_2p"] is hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_2p"]
        assert hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_sq19_2p"] is not hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_2p"]

        trained_path = tmp_path / "trained_profiles.json"
        trained_path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "heuristic_v1_2p": {"WEIGHT_STACK_CONTROL": 12345.0},
                    }
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv(hw.TRAINED_PROFILES_ENV, str(trained_path))
        loaded = hw.load_trained_profiles_if_available()

        assert "heuristic_v1_2p" in loaded
        # Propagated into alias board-specific ids (but not ones with distinct baselines).
        assert "heuristic_v1_sq8_2p" in loaded
        assert "heuristic_v1_hex_2p" in loaded
        assert "heuristic_v1_sq19_2p" not in loaded

        assert hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_2p"]["WEIGHT_STACK_CONTROL"] == 12345.0
        assert hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_sq8_2p"]["WEIGHT_STACK_CONTROL"] == 12345.0
        assert hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_hex_2p"]["WEIGHT_STACK_CONTROL"] == 12345.0
        assert hw.HEURISTIC_WEIGHT_PROFILES["heuristic_v1_sq19_2p"].get("WEIGHT_STACK_CONTROL") != 12345.0
    finally:
        hw.HEURISTIC_WEIGHT_PROFILES.clear()
        hw.HEURISTIC_WEIGHT_PROFILES.update(original)
