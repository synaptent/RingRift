"""Unit tests for training_data_manifest.py.

Tests the TrainingDataManifest system for discovering and tracking
training data across local, OWC, and S3 sources.

December 28, 2025.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.coordination.training_data_manifest import (
    DataSource,
    TrainingDataEntry,
    TrainingDataManifest,
    extract_config_key,
    get_training_data_manifest_sync,
    MANIFEST_CACHE_PATH,
)


# ==============================================================================
# DataSource Enum Tests
# ==============================================================================

class TestDataSource:
    """Tests for DataSource enum."""

    def test_local_value(self):
        assert DataSource.LOCAL == "local"
        assert DataSource.LOCAL.value == "local"

    def test_owc_value(self):
        assert DataSource.OWC == "owc"
        assert DataSource.OWC.value == "owc"

    def test_s3_value(self):
        assert DataSource.S3 == "s3"
        assert DataSource.S3.value == "s3"

    def test_from_string(self):
        assert DataSource("local") == DataSource.LOCAL
        assert DataSource("owc") == DataSource.OWC
        assert DataSource("s3") == DataSource.S3


# ==============================================================================
# TrainingDataEntry Dataclass Tests
# ==============================================================================

class TestTrainingDataEntry:
    """Tests for TrainingDataEntry dataclass."""

    def test_basic_creation(self):
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/training/hex8_2p.npz",
            size_bytes=1024 * 1024 * 100,  # 100 MB
        )
        assert entry.config_key == "hex8_2p"
        assert entry.source == DataSource.LOCAL
        assert entry.path == "/data/training/hex8_2p.npz"
        assert entry.size_bytes == 104857600

    def test_size_mb_property(self):
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=10 * 1024 * 1024,  # 10 MB
        )
        assert entry.size_mb == 10.0

    def test_size_gb_property(self):
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
        )
        assert entry.size_gb == 2.0

    def test_age_hours_with_modified_time(self):
        now = datetime.now(tz=timezone.utc)
        two_hours_ago = now - timedelta(hours=2)
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=1024,
            modified_time=two_hours_ago,
        )
        # Allow for slight timing differences
        assert 1.9 < entry.age_hours < 2.1

    def test_age_hours_with_newest_game_time(self):
        now = datetime.now(tz=timezone.utc)
        three_hours_ago = now - timedelta(hours=3)
        one_hour_ago = now - timedelta(hours=1)
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=1024,
            modified_time=three_hours_ago,  # File modified 3h ago
            newest_game_time=one_hour_ago,  # But data is only 1h old
        )
        # Should use newest_game_time, not modified_time
        assert 0.9 < entry.age_hours < 1.1

    def test_age_hours_none_when_no_time(self):
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=1024,
        )
        assert entry.age_hours is None

    def test_to_dict(self):
        now = datetime.now(tz=timezone.utc)
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/hex8_2p.npz",
            size_bytes=1024000,
            modified_time=now,
            sample_count=10000,
            version="v2",
            quality_score=0.85,
            newest_game_time=now,
        )
        result = entry.to_dict()
        assert result["config_key"] == "hex8_2p"
        assert result["source"] == "s3"
        assert result["path"] == "s3://bucket/hex8_2p.npz"
        assert result["size_bytes"] == 1024000
        assert result["sample_count"] == 10000
        assert result["version"] == "v2"
        assert result["quality_score"] == 0.85

    def test_from_dict(self):
        now = datetime.now(tz=timezone.utc)
        data = {
            "config_key": "square8_3p",
            "source": "owc",
            "path": "/Volumes/RingRift-Data/square8_3p.npz",
            "size_bytes": 5000000,
            "modified_time": now.isoformat(),
            "sample_count": 25000,
            "version": "v3",
            "quality_score": 0.92,
            "newest_game_time": now.isoformat(),
        }
        entry = TrainingDataEntry.from_dict(data)
        assert entry.config_key == "square8_3p"
        assert entry.source == DataSource.OWC
        assert entry.size_bytes == 5000000
        assert entry.sample_count == 25000
        assert entry.version == "v3"
        assert entry.quality_score == 0.92

    def test_from_dict_minimal(self):
        data = {
            "config_key": "hex8_4p",
            "source": "local",
            "path": "/data/hex8_4p.npz",
            "size_bytes": 1000,
        }
        entry = TrainingDataEntry.from_dict(data)
        assert entry.config_key == "hex8_4p"
        assert entry.modified_time is None
        assert entry.sample_count is None
        assert entry.version is None

    def test_roundtrip_serialization(self):
        now = datetime.now(tz=timezone.utc)
        original = TrainingDataEntry(
            config_key="hexagonal_2p",
            source=DataSource.S3,
            path="s3://bucket/hexagonal_2p.npz",
            size_bytes=50000000,
            modified_time=now,
            sample_count=100000,
            version="v4",
            quality_score=0.95,
            newest_game_time=now,
        )
        # Roundtrip through dict
        data = original.to_dict()
        restored = TrainingDataEntry.from_dict(data)

        assert restored.config_key == original.config_key
        assert restored.source == original.source
        assert restored.path == original.path
        assert restored.size_bytes == original.size_bytes
        assert restored.sample_count == original.sample_count
        assert restored.version == original.version
        assert restored.quality_score == original.quality_score


# ==============================================================================
# extract_config_key Function Tests
# ==============================================================================

class TestExtractConfigKey:
    """Tests for extract_config_key function."""

    def test_standard_patterns(self):
        assert extract_config_key("hex8_2p.npz") == "hex8_2p"
        assert extract_config_key("hex8_3p.npz") == "hex8_3p"
        assert extract_config_key("hex8_4p.npz") == "hex8_4p"

    def test_square_boards(self):
        assert extract_config_key("square8_2p.npz") == "square8_2p"
        assert extract_config_key("square19_3p.npz") == "square19_3p"

    def test_hexagonal_board(self):
        assert extract_config_key("hexagonal_2p_v4.npz") == "hexagonal_2p"
        assert extract_config_key("hexagonal_4p.npz") == "hexagonal_4p"

    def test_abbreviation_normalization(self):
        # sq8 -> square8
        assert extract_config_key("sq8_2p.npz") == "square8_2p"
        # sq19 -> square19
        assert extract_config_key("sq19_3p.npz") == "square19_3p"

    def test_case_insensitivity(self):
        assert extract_config_key("HEX8_2P.npz") == "hex8_2p"
        assert extract_config_key("SQUARE8_3P.NPZ") == "square8_3p"

    def test_with_prefix(self):
        assert extract_config_key("canonical_hex8_2p.npz") == "hex8_2p"
        assert extract_config_key("selfplay_square8_4p_v2.npz") == "square8_4p"

    def test_with_suffix(self):
        assert extract_config_key("hex8_2p_v5heavy.npz") == "hex8_2p"
        assert extract_config_key("square19_2p_20251227.npz") == "square19_2p"

    def test_in_path(self):
        assert extract_config_key("/data/training/hex8_2p.npz") == "hex8_2p"
        assert extract_config_key("s3://bucket/hexagonal_3p.npz") == "hexagonal_3p"

    def test_no_match(self):
        assert extract_config_key("random_file.npz") is None
        assert extract_config_key("model_v2.pth") is None
        assert extract_config_key("hex_2p.npz") is None  # Missing '8'

    def test_invalid_player_count(self):
        # Only 2, 3, 4 players supported
        assert extract_config_key("hex8_5p.npz") is None
        assert extract_config_key("hex8_1p.npz") is None


# ==============================================================================
# TrainingDataManifest Class Tests
# ==============================================================================

class TestTrainingDataManifest:
    """Tests for TrainingDataManifest class."""

    def test_empty_manifest(self):
        manifest = TrainingDataManifest()
        assert manifest.entries == {}
        assert manifest.last_refresh is None
        assert manifest.get_configs() == []

    def test_add_entry(self):
        manifest = TrainingDataManifest()
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=1024,
        )
        manifest.add_entry(entry)

        assert "hex8_2p" in manifest.entries
        assert len(manifest.entries["hex8_2p"]) == 1
        assert manifest.entries["hex8_2p"][0] == entry

    def test_add_multiple_entries_same_config(self):
        manifest = TrainingDataManifest()
        entry1 = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/local/hex8_2p.npz",
            size_bytes=1024,
        )
        entry2 = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/hex8_2p.npz",
            size_bytes=2048,
        )
        manifest.add_entry(entry1)
        manifest.add_entry(entry2)

        assert len(manifest.entries["hex8_2p"]) == 2

    def test_add_entry_updates_duplicate(self):
        manifest = TrainingDataManifest()
        entry1 = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=1024,
        )
        entry2 = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",  # Same path
            size_bytes=2048,  # Updated size
        )
        manifest.add_entry(entry1)
        manifest.add_entry(entry2)

        # Should update, not duplicate
        assert len(manifest.entries["hex8_2p"]) == 1
        assert manifest.entries["hex8_2p"][0].size_bytes == 2048

    def test_get_all_data_sorted_by_size(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p_small.npz",
            size_bytes=1000,
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/hex8_2p_large.npz",
            size_bytes=5000,
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.OWC,
            path="/Volumes/hex8_2p_medium.npz",
            size_bytes=3000,
        ))

        all_data = manifest.get_all_data("hex8_2p")
        assert len(all_data) == 3
        # Largest first
        assert all_data[0].size_bytes == 5000
        assert all_data[1].size_bytes == 3000
        assert all_data[2].size_bytes == 1000

    def test_get_all_data_empty_config(self):
        manifest = TrainingDataManifest()
        assert manifest.get_all_data("nonexistent") == []

    def test_get_best_data_by_size(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/small.npz",
            size_bytes=1000,
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/large.npz",
            size_bytes=5000,
        ))

        # With freshness_weight=0, pure size-based selection
        best = manifest.get_best_data("hex8_2p", freshness_weight=0.0)
        assert best.size_bytes == 5000

    def test_get_best_data_with_min_size(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/small.npz",
            size_bytes=1024 * 1024,  # 1 MB
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/large.npz",
            size_bytes=100 * 1024 * 1024,  # 100 MB
        ))

        # Require minimum 10 MB
        best = manifest.get_best_data("hex8_2p", min_size_mb=10.0)
        assert best.size_bytes == 100 * 1024 * 1024

        # Require minimum 200 MB (nothing qualifies)
        best = manifest.get_best_data("hex8_2p", min_size_mb=200.0)
        assert best is None

    def test_get_best_data_prefer_source(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=5000,  # Larger
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.OWC,
            path="/Volumes/hex8_2p.npz",
            size_bytes=3000,  # Smaller
        ))

        # Prefer OWC even though it's smaller
        best = manifest.get_best_data("hex8_2p", prefer_source=DataSource.OWC)
        assert best.source == DataSource.OWC

    def test_get_best_data_max_age(self):
        now = datetime.now(tz=timezone.utc)
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/fresh.npz",
            size_bytes=1000,
            modified_time=now - timedelta(hours=1),  # 1 hour old
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/stale.npz",
            size_bytes=5000,  # Larger but stale
            modified_time=now - timedelta(hours=10),  # 10 hours old
        ))

        # Max 2 hours - only fresh qualifies
        best = manifest.get_best_data("hex8_2p", max_age_hours=2.0)
        assert best.path == "/data/fresh.npz"

        # Max 24 hours - both qualify, larger wins
        best = manifest.get_best_data("hex8_2p", max_age_hours=24.0, freshness_weight=0.0)
        assert best.path == "s3://bucket/stale.npz"

    def test_get_best_data_combined_scoring(self):
        now = datetime.now(tz=timezone.utc)
        manifest = TrainingDataManifest()
        # Large but stale
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/large_stale.npz",
            size_bytes=100000,
            modified_time=now - timedelta(hours=48),  # 2 days old
        ))
        # Small but fresh
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/small_fresh.npz",
            size_bytes=50000,
            modified_time=now - timedelta(minutes=30),  # 30 min old
        ))

        # High freshness weight should prefer fresh data
        best = manifest.get_best_data("hex8_2p", freshness_weight=0.8)
        assert best.path == "/data/small_fresh.npz"

        # Low freshness weight should prefer larger data
        best = manifest.get_best_data("hex8_2p", freshness_weight=0.1)
        assert best.path == "/data/large_stale.npz"

    def test_get_best_data_single_entry(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/only.npz",
            size_bytes=1000,
        ))

        # Single entry - should return it regardless of scoring
        best = manifest.get_best_data("hex8_2p")
        assert best.path == "/data/only.npz"

    def test_get_best_data_no_data(self):
        manifest = TrainingDataManifest()
        assert manifest.get_best_data("nonexistent") is None

    def test_get_configs(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=1000,
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="square8_3p",
            source=DataSource.LOCAL,
            path="/data/square8_3p.npz",
            size_bytes=2000,
        ))

        configs = manifest.get_configs()
        assert "hex8_2p" in configs
        assert "square8_3p" in configs
        assert len(configs) == 2

    def test_get_summary(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/hex8_2p.npz",
            size_bytes=10 * 1024 * 1024,  # 10 MB
        ))
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/hex8_2p.npz",
            size_bytes=50 * 1024 * 1024,  # 50 MB (larger)
        ))

        summary = manifest.get_summary()
        assert "hex8_2p" in summary
        assert summary["hex8_2p"]["count"] == 2
        assert summary["hex8_2p"]["best_size_mb"] == 50.0
        assert summary["hex8_2p"]["best_source"] == "s3"
        assert set(summary["hex8_2p"]["sources"]) == {"local", "s3"}


# ==============================================================================
# Manifest Cache Tests
# ==============================================================================

class TestManifestCache:
    """Tests for manifest cache save/load functionality."""

    @pytest.mark.asyncio
    async def test_save_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = Path(f.name)

        try:
            with patch("app.coordination.training_data_manifest.MANIFEST_CACHE_PATH", cache_path):
                manifest = TrainingDataManifest()
                manifest.last_refresh = datetime.now(tz=timezone.utc)
                manifest.add_entry(TrainingDataEntry(
                    config_key="hex8_2p",
                    source=DataSource.LOCAL,
                    path="/data/hex8_2p.npz",
                    size_bytes=1024,
                ))

                await manifest.save_cache()

                assert cache_path.exists()
                with open(cache_path) as f:
                    data = json.load(f)
                assert "last_refresh" in data
                assert "entries" in data
                assert "hex8_2p" in data["entries"]
        finally:
            cache_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_load_cache(self):
        now = datetime.now(tz=timezone.utc)
        cache_data = {
            "last_refresh": now.isoformat(),
            "entries": {
                "hex8_2p": [
                    {
                        "config_key": "hex8_2p",
                        "source": "local",
                        "path": "/data/hex8_2p.npz",
                        "size_bytes": 5000,
                    }
                ]
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(cache_data, f)
            cache_path = Path(f.name)

        try:
            with patch("app.coordination.training_data_manifest.MANIFEST_CACHE_PATH", cache_path):
                manifest = TrainingDataManifest()
                result = await manifest.load_cache()

                assert result is True
                assert manifest.last_refresh is not None
                assert "hex8_2p" in manifest.entries
                assert manifest.entries["hex8_2p"][0].size_bytes == 5000
        finally:
            cache_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_load_cache_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nonexistent.json"

            with patch("app.coordination.training_data_manifest.MANIFEST_CACHE_PATH", cache_path):
                manifest = TrainingDataManifest()
                result = await manifest.load_cache()

                assert result is False

    @pytest.mark.asyncio
    async def test_load_cache_corrupted_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json{{{")
            cache_path = Path(f.name)

        try:
            with patch("app.coordination.training_data_manifest.MANIFEST_CACHE_PATH", cache_path):
                manifest = TrainingDataManifest()
                result = await manifest.load_cache()

                assert result is False
        finally:
            cache_path.unlink(missing_ok=True)


# ==============================================================================
# Refresh Local Tests
# ==============================================================================

class TestRefreshLocal:
    """Tests for refresh_local functionality."""

    @pytest.mark.asyncio
    async def test_refresh_local_finds_npz_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create fake NPZ files
            (data_dir / "hex8_2p.npz").write_bytes(b"x" * 1024)
            (data_dir / "square8_3p.npz").write_bytes(b"x" * 2048)
            (data_dir / "not_a_config.npz").write_bytes(b"x" * 100)  # No match

            manifest = TrainingDataManifest()
            # Mock _extract_npz_freshness to avoid parsing invalid NPZ files
            with patch.object(manifest, '_extract_npz_freshness', return_value=None):
                count = await manifest.refresh_local(data_dir)

            assert count == 2  # Only 2 match config patterns
            assert "hex8_2p" in manifest.entries
            assert "square8_3p" in manifest.entries
            assert manifest.entries["hex8_2p"][0].size_bytes == 1024
            assert manifest.entries["square8_3p"][0].size_bytes == 2048

    @pytest.mark.asyncio
    async def test_refresh_local_nonexistent_dir(self):
        manifest = TrainingDataManifest()
        count = await manifest.refresh_local(Path("/nonexistent/path"))
        assert count == 0

    @pytest.mark.asyncio
    async def test_refresh_local_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = TrainingDataManifest()
            count = await manifest.refresh_local(Path(tmpdir))
            assert count == 0


# ==============================================================================
# Singleton Tests
# ==============================================================================

class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_training_data_manifest_sync_creates_instance(self):
        import app.coordination.training_data_manifest as module

        # Reset singleton
        original = module._manifest_instance
        module._manifest_instance = None

        try:
            manifest = get_training_data_manifest_sync()
            assert manifest is not None
            assert isinstance(manifest, TrainingDataManifest)

            # Second call returns same instance
            manifest2 = get_training_data_manifest_sync()
            assert manifest is manifest2
        finally:
            module._manifest_instance = original

    def test_get_training_data_manifest_sync_loads_cache_if_exists(self):
        import app.coordination.training_data_manifest as module

        # Create a cache file
        now = datetime.now(tz=timezone.utc)
        cache_data = {
            "last_refresh": now.isoformat(),
            "entries": {
                "hex8_4p": [
                    {
                        "config_key": "hex8_4p",
                        "source": "s3",
                        "path": "s3://bucket/hex8_4p.npz",
                        "size_bytes": 9999,
                    }
                ]
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(cache_data, f)
            cache_path = Path(f.name)

        # Reset singleton
        original_instance = module._manifest_instance
        original_cache_path = module.MANIFEST_CACHE_PATH
        module._manifest_instance = None

        try:
            with patch("app.coordination.training_data_manifest.MANIFEST_CACHE_PATH", cache_path):
                manifest = get_training_data_manifest_sync()

                # Should have loaded from cache
                assert "hex8_4p" in manifest.entries
                assert manifest.entries["hex8_4p"][0].size_bytes == 9999
        finally:
            module._manifest_instance = original_instance
            cache_path.unlink(missing_ok=True)


# ==============================================================================
# NPZ Freshness Extraction Tests
# ==============================================================================

class TestNPZFreshnessExtraction:
    """Tests for extracting freshness metadata from NPZ files."""

    def test_extract_npz_freshness_no_numpy(self):
        manifest = TrainingDataManifest()

        with patch("app.coordination.training_data_manifest.NUMPY_AVAILABLE", False):
            result = manifest._extract_npz_freshness(Path("/fake/file.npz"))
            assert result is None

    def test_extract_npz_freshness_invalid_file(self):
        manifest = TrainingDataManifest()
        result = manifest._extract_npz_freshness(Path("/nonexistent/file.npz"))
        assert result is None


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_entry_with_zero_size(self):
        entry = TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/empty.npz",
            size_bytes=0,
        )
        assert entry.size_mb == 0.0
        assert entry.size_gb == 0.0

    def test_best_data_with_all_entries_too_small(self):
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/tiny.npz",
            size_bytes=100,  # Very small
        ))

        # Require 1 MB minimum
        best = manifest.get_best_data("hex8_2p", min_size_mb=1.0)
        assert best is None

    def test_best_data_with_all_entries_too_old(self):
        now = datetime.now(tz=timezone.utc)
        manifest = TrainingDataManifest()
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/ancient.npz",
            size_bytes=10000,
            modified_time=now - timedelta(days=30),  # 30 days old
        ))

        # Require max 1 hour old
        best = manifest.get_best_data("hex8_2p", max_age_hours=1.0)
        assert best is None

    def test_freshness_scoring_unknown_age(self):
        """Test that unknown age is treated as moderately stale."""
        manifest = TrainingDataManifest()
        now = datetime.now(tz=timezone.utc)

        # Entry with known age (fresh)
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.LOCAL,
            path="/data/known_fresh.npz",
            size_bytes=1000,
            modified_time=now - timedelta(minutes=30),  # 30 min old
        ))

        # Entry with unknown age but larger size
        manifest.add_entry(TrainingDataEntry(
            config_key="hex8_2p",
            source=DataSource.S3,
            path="s3://bucket/unknown_age.npz",
            size_bytes=2000,  # Larger
            # No modified_time - unknown age
        ))

        # With high freshness weight, known fresh should win
        best = manifest.get_best_data("hex8_2p", freshness_weight=0.9)
        assert best.path == "/data/known_fresh.npz"
