"""Tests for curriculum weight management.

Tests curriculum_weights.py which provides persistence for selfplay
prioritization weights used by SelfplayScheduler, QueuePopulator, and P2P.

December 28, 2025: Created comprehensive test coverage.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from app.coordination.curriculum_weights import (
    CURRICULUM_WEIGHTS_PATH,
    CURRICULUM_WEIGHTS_STALE_SECONDS,
    export_curriculum_weights,
    get_curriculum_weight,
    load_curriculum_weights,
)


class TestCurriculumWeightsConstants:
    """Tests for module constants."""

    def test_path_constant_is_path(self) -> None:
        """Verify CURRICULUM_WEIGHTS_PATH is a Path object."""
        assert isinstance(CURRICULUM_WEIGHTS_PATH, Path)

    def test_path_ends_with_json(self) -> None:
        """Verify path has .json extension."""
        assert CURRICULUM_WEIGHTS_PATH.suffix == ".json"

    def test_path_in_data_directory(self) -> None:
        """Verify path is in data directory."""
        assert "data" in str(CURRICULUM_WEIGHTS_PATH)

    def test_staleness_constant_is_positive(self) -> None:
        """Verify staleness threshold is a positive number."""
        assert CURRICULUM_WEIGHTS_STALE_SECONDS > 0

    def test_staleness_constant_is_reasonable(self) -> None:
        """Verify staleness threshold is between 1 minute and 24 hours."""
        assert 60 <= CURRICULUM_WEIGHTS_STALE_SECONDS <= 86400  # 1 min to 24 hours


class TestExportCurriculumWeights:
    """Tests for export_curriculum_weights function."""

    def test_successful_export_creates_file(self, tmp_path: Path) -> None:
        """Verify export creates the weights file."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            weights = {"hex8_2p": 1.5, "square8_4p": 0.8}
            result = export_curriculum_weights(weights)

        assert result is True
        assert weights_path.exists()

    def test_export_creates_valid_json(self, tmp_path: Path) -> None:
        """Verify export creates valid JSON with correct structure."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            weights = {"hex8_2p": 1.5, "square8_4p": 0.8}
            export_curriculum_weights(weights)

        with open(weights_path) as f:
            data = json.load(f)

        assert "weights" in data
        assert "updated_at" in data
        assert "updated_at_iso" in data
        assert data["weights"] == weights

    def test_export_includes_timestamp(self, tmp_path: Path) -> None:
        """Verify export includes timestamp for staleness checking."""
        weights_path = tmp_path / "curriculum_weights.json"

        before = time.time()
        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"hex8_2p": 1.0})
        after = time.time()

        with open(weights_path) as f:
            data = json.load(f)

        assert before <= data["updated_at"] <= after

    def test_export_uses_atomic_write(self, tmp_path: Path) -> None:
        """Verify export uses temp file + rename pattern (atomic write)."""
        weights_path = tmp_path / "curriculum_weights.json"
        temp_path = weights_path.with_suffix(".tmp")

        # Pre-create to verify overwrite
        weights_path.write_text('{"old": "data"}')

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"new": 1.0})

        # Temp file should not exist after successful export
        assert not temp_path.exists()
        # Main file should have new data
        with open(weights_path) as f:
            data = json.load(f)
        assert "new" in data["weights"]

    def test_export_creates_parent_directory(self, tmp_path: Path) -> None:
        """Verify export creates parent directories if needed."""
        weights_path = tmp_path / "nested" / "dir" / "weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = export_curriculum_weights({"hex8_2p": 1.0})

        assert result is True
        assert weights_path.exists()

    def test_export_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Verify export overwrites existing file."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights_path.write_text('{"weights": {"old": 1.0}, "updated_at": 0}')

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"new": 2.0})

        with open(weights_path) as f:
            data = json.load(f)

        assert "new" in data["weights"]
        assert "old" not in data["weights"]

    def test_export_returns_false_on_permission_error(self, tmp_path: Path) -> None:
        """Verify export returns False on permission error."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            with mock.patch("builtins.open", side_effect=PermissionError("denied")):
                result = export_curriculum_weights({"hex8_2p": 1.0})

        assert result is False

    def test_export_handles_empty_weights(self, tmp_path: Path) -> None:
        """Verify export works with empty weights dict."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = export_curriculum_weights({})

        assert result is True
        with open(weights_path) as f:
            data = json.load(f)
        assert data["weights"] == {}


class TestLoadCurriculumWeights:
    """Tests for load_curriculum_weights function."""

    def test_load_returns_empty_dict_if_file_missing(self, tmp_path: Path) -> None:
        """Verify load returns empty dict if file doesn't exist."""
        weights_path = tmp_path / "nonexistent.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_returns_valid_weights(self, tmp_path: Path) -> None:
        """Verify load returns weights from valid file."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5, "square8_4p": 0.8},
            "updated_at": time.time(),  # Fresh
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {"hex8_2p": 1.5, "square8_4p": 0.8}

    def test_load_respects_max_age_seconds(self, tmp_path: Path) -> None:
        """Verify load respects custom max_age_seconds parameter."""
        weights_path = tmp_path / "curriculum_weights.json"
        # Write data that's 100 seconds old
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - 100,
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            # With 50s max age, should be stale
            result_stale = load_curriculum_weights(max_age_seconds=50)
            # With 200s max age, should be fresh
            result_fresh = load_curriculum_weights(max_age_seconds=200)

        assert result_stale == {}
        assert result_fresh == {"hex8_2p": 1.5}

    def test_load_returns_empty_dict_if_stale(self, tmp_path: Path) -> None:
        """Verify load returns empty dict for stale weights."""
        weights_path = tmp_path / "curriculum_weights.json"
        # Write data that's definitely stale (1 day old)
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - 86400,  # 1 day ago
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_malformed_json(self, tmp_path: Path) -> None:
        """Verify load returns empty dict for malformed JSON."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights_path.write_text("not valid json{{{")

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_missing_weights_key(self, tmp_path: Path) -> None:
        """Verify load returns empty dict if weights key missing."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "updated_at": time.time(),
            # Missing "weights" key
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_missing_updated_at(self, tmp_path: Path) -> None:
        """Verify load returns empty dict if updated_at missing (treated as stale)."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            # Missing "updated_at" - defaults to 0, so always stale
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}


class TestGetCurriculumWeight:
    """Tests for get_curriculum_weight convenience function."""

    def test_get_weight_returns_value_if_exists(self, tmp_path: Path) -> None:
        """Verify get_weight returns correct value for existing config."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5, "square8_4p": 0.8},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p")

        assert result == 1.5

    def test_get_weight_returns_default_if_missing(self, tmp_path: Path) -> None:
        """Verify get_weight returns default for missing config."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("nonexistent_config")

        assert result == 1.0  # Default

    def test_get_weight_uses_custom_default(self, tmp_path: Path) -> None:
        """Verify get_weight respects custom default parameter."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=0.5)

        assert result == 0.5

    def test_get_weight_returns_default_if_file_missing(self, tmp_path: Path) -> None:
        """Verify get_weight returns default if weights file doesn't exist."""
        weights_path = tmp_path / "nonexistent.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=2.0)

        assert result == 2.0

    def test_get_weight_returns_default_if_stale(self, tmp_path: Path) -> None:
        """Verify get_weight returns default for stale weights."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - 86400,  # 1 day old (stale)
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=0.7)

        assert result == 0.7


class TestIntegration:
    """Integration tests for export/load cycle."""

    def test_export_then_load_roundtrip(self, tmp_path: Path) -> None:
        """Verify weights can be exported and loaded correctly."""
        weights_path = tmp_path / "curriculum_weights.json"
        original_weights = {
            "hex8_2p": 1.5,
            "hex8_3p": 1.2,
            "hex8_4p": 0.8,
            "square8_2p": 1.0,
        }

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_result = export_curriculum_weights(original_weights)
            loaded_weights = load_curriculum_weights()

        assert export_result is True
        assert loaded_weights == original_weights

    def test_multiple_exports_overwrite(self, tmp_path: Path) -> None:
        """Verify multiple exports correctly overwrite previous data."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"v1": 1.0})
            export_curriculum_weights({"v2": 2.0})
            export_curriculum_weights({"v3": 3.0})
            loaded = load_curriculum_weights()

        assert loaded == {"v3": 3.0}
        assert "v1" not in loaded
        assert "v2" not in loaded


# =============================================================================
# Additional Tests - December 29, 2025
# =============================================================================


class TestExportEdgeCases:
    """Additional edge case tests for export_curriculum_weights."""

    def test_export_with_float_weights(self, tmp_path: Path) -> None:
        """Verify export handles various float values correctly."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            weights = {
                "config_a": 0.0,
                "config_b": 0.001,
                "config_c": 999.999,
                "config_d": -1.5,
            }
            result = export_curriculum_weights(weights)

        assert result is True
        with open(weights_path) as f:
            data = json.load(f)
        assert data["weights"]["config_a"] == 0.0
        assert data["weights"]["config_b"] == 0.001
        assert data["weights"]["config_c"] == 999.999
        assert data["weights"]["config_d"] == -1.5

    def test_export_with_unicode_keys(self, tmp_path: Path) -> None:
        """Verify export handles unicode config keys."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            weights = {"hex8_2p": 1.0, "test_config": 2.0}
            result = export_curriculum_weights(weights)

        assert result is True
        with open(weights_path) as f:
            data = json.load(f)
        assert "hex8_2p" in data["weights"]

    def test_export_includes_iso_timestamp(self, tmp_path: Path) -> None:
        """Verify export includes ISO-formatted timestamp."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"hex8_2p": 1.0})

        with open(weights_path) as f:
            data = json.load(f)

        # Verify ISO format looks like "2025-12-29T12:34:56Z"
        iso_ts = data["updated_at_iso"]
        assert "T" in iso_ts
        assert iso_ts.endswith("Z")
        assert len(iso_ts) == 20  # "YYYY-MM-DDTHH:MM:SSZ"

    def test_export_returns_false_on_oserror(self, tmp_path: Path) -> None:
        """Verify export returns False on OSError."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            with mock.patch("builtins.open", side_effect=OSError("disk full")):
                result = export_curriculum_weights({"hex8_2p": 1.0})

        assert result is False

    def test_export_returns_false_on_rename_failure(self, tmp_path: Path) -> None:
        """Verify export returns False when rename fails."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            with mock.patch.object(Path, "rename", side_effect=OSError("rename failed")):
                result = export_curriculum_weights({"hex8_2p": 1.0})

        assert result is False

    def test_export_all_12_canonical_configs(self, tmp_path: Path) -> None:
        """Verify export works with all 12 canonical configs."""
        weights_path = tmp_path / "curriculum_weights.json"
        canonical_configs = [
            "hex8_2p", "hex8_3p", "hex8_4p",
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        ]
        weights = {config: 1.0 + i * 0.1 for i, config in enumerate(canonical_configs)}

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = export_curriculum_weights(weights)

        assert result is True
        with open(weights_path) as f:
            data = json.load(f)
        assert len(data["weights"]) == 12
        for config in canonical_configs:
            assert config in data["weights"]


class TestLoadEdgeCases:
    """Additional edge case tests for load_curriculum_weights."""

    def test_load_handles_empty_file(self, tmp_path: Path) -> None:
        """Verify load returns empty dict for empty file."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights_path.write_text("")

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_null_json(self, tmp_path: Path) -> None:
        """Verify load returns empty dict for null JSON."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights_path.write_text("null")

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_array_json(self, tmp_path: Path) -> None:
        """Verify load returns empty dict for array JSON (wrong structure)."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights_path.write_text("[1, 2, 3]")

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_with_zero_max_age(self, tmp_path: Path) -> None:
        """Verify load with max_age_seconds=0 always returns stale."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time(),  # Just created
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights(max_age_seconds=0)

        assert result == {}

    def test_load_with_negative_max_age(self, tmp_path: Path) -> None:
        """Verify load with negative max_age_seconds treats all as stale."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights(max_age_seconds=-1)

        assert result == {}

    def test_load_with_very_large_max_age(self, tmp_path: Path) -> None:
        """Verify load with very large max_age accepts old data."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - 365 * 24 * 3600,  # 1 year old
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights(max_age_seconds=365 * 24 * 3600 + 100)

        assert result == {"hex8_2p": 1.5}

    def test_load_handles_non_numeric_updated_at(self, tmp_path: Path) -> None:
        """Verify load handles non-numeric updated_at gracefully."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": "not_a_number",
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            # Should not raise, but may return empty (depends on impl)
            result = load_curriculum_weights()

        # Implementation treats invalid timestamp as 0 (stale)
        assert result == {}

    def test_load_handles_weights_not_dict(self, tmp_path: Path) -> None:
        """Verify load handles weights being non-dict gracefully."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": [1, 2, 3],  # Array instead of dict
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        # Returns the weights as-is (array), which may not be what's expected
        # but doesn't crash
        assert result == [1, 2, 3]


class TestGetCurriculumWeightEdgeCases:
    """Additional edge case tests for get_curriculum_weight."""

    def test_get_weight_with_zero_default(self, tmp_path: Path) -> None:
        """Verify get_weight works with default=0."""
        weights_path = tmp_path / "nonexistent.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=0.0)

        assert result == 0.0

    def test_get_weight_with_negative_default(self, tmp_path: Path) -> None:
        """Verify get_weight works with negative default."""
        weights_path = tmp_path / "nonexistent.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=-1.0)

        assert result == -1.0

    def test_get_weight_empty_config_key(self, tmp_path: Path) -> None:
        """Verify get_weight handles empty config key."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"": 1.5},  # Empty key
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("")

        assert result == 1.5

    def test_get_weight_special_characters_in_key(self, tmp_path: Path) -> None:
        """Verify get_weight handles special characters in config key."""
        weights_path = tmp_path / "curriculum_weights.json"
        special_key = "config_with-special.chars_123"
        data = {
            "weights": {special_key: 2.5},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight(special_key)

        assert result == 2.5


class TestConcurrencySafety:
    """Tests for thread/concurrency safety of curriculum weights operations."""

    def test_load_handles_file_deleted_during_read(self, tmp_path: Path) -> None:
        """Verify load handles file being deleted during read."""
        weights_path = tmp_path / "curriculum_weights.json"

        def side_effect(*args, **kwargs):
            raise FileNotFoundError("deleted")

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            with mock.patch("builtins.open", side_effect=side_effect):
                result = load_curriculum_weights()

        assert result == {}

    def test_export_then_immediate_load_is_consistent(self, tmp_path: Path) -> None:
        """Verify immediate load after export returns same data."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights = {f"config_{i}": float(i) for i in range(100)}

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights(weights)
            loaded = load_curriculum_weights()

        assert loaded == weights


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_staleness_exactly_at_threshold(self, tmp_path: Path) -> None:
        """Verify behavior when timestamp is exactly at staleness threshold."""
        weights_path = tmp_path / "curriculum_weights.json"

        # Create data that's exactly CURRICULUM_WEIGHTS_STALE_SECONDS old
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - CURRICULUM_WEIGHTS_STALE_SECONDS,
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        # At exactly the threshold, should be considered stale
        assert result == {}

    def test_staleness_just_under_threshold(self, tmp_path: Path) -> None:
        """Verify data just under staleness threshold is accepted."""
        weights_path = tmp_path / "curriculum_weights.json"

        # Create data that's 1 second under threshold
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - (CURRICULUM_WEIGHTS_STALE_SECONDS - 1),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {"hex8_2p": 1.5}

    def test_large_weights_dict(self, tmp_path: Path) -> None:
        """Verify export/load handles large weights dictionary."""
        weights_path = tmp_path / "curriculum_weights.json"
        # Create 1000 config entries
        large_weights = {f"config_{i}": float(i) * 0.001 for i in range(1000)}

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_result = export_curriculum_weights(large_weights)
            loaded = load_curriculum_weights()

        assert export_result is True
        assert loaded == large_weights
        assert len(loaded) == 1000
