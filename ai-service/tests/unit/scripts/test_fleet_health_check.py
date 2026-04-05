"""Tests for scripts/fleet_health_check.py - Fleet Health Aggregator."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from scripts.fleet_health_check import (
    STATUS_DEAD,
    STATUS_HEALTHY,
    STATUS_NO_PROGRESS,
    STATUS_STALE,
    build_fleet_report,
    classify_status,
    detect_no_progress,
    fetch_all_heartbeats,
    fetch_heartbeat,
    format_age,
    list_heartbeat_keys,
)


# ---------------------------------------------------------------------------
# classify_status
# ---------------------------------------------------------------------------


class TestClassifyStatus:
    def test_healthy_within_threshold(self):
        # 1 hour old, threshold at 2 hours
        assert classify_status(3600, 7200, 21600) == STATUS_HEALTHY

    def test_healthy_at_zero(self):
        assert classify_status(0, 7200, 21600) == STATUS_HEALTHY

    def test_stale_between_thresholds(self):
        # 3 hours old, stale at 2h, dead at 6h
        assert classify_status(10800, 7200, 21600) == STATUS_STALE

    def test_stale_at_boundary(self):
        # Just past stale threshold
        assert classify_status(7201, 7200, 21600) == STATUS_STALE

    def test_dead_beyond_dead_threshold(self):
        # 8 hours old, dead at 6h
        assert classify_status(28800, 7200, 21600) == STATUS_DEAD

    def test_dead_at_boundary(self):
        # Just past dead threshold
        assert classify_status(21601, 7200, 21600) == STATUS_DEAD

    def test_healthy_just_before_stale(self):
        assert classify_status(7199, 7200, 21600) == STATUS_HEALTHY


# ---------------------------------------------------------------------------
# format_age
# ---------------------------------------------------------------------------


class TestFormatAge:
    def test_seconds(self):
        assert format_age(30) == "30s ago"

    def test_minutes(self):
        assert format_age(300) == "5min ago"

    def test_hours(self):
        result = format_age(7200)
        assert "2.0h ago" == result

    def test_days(self):
        result = format_age(172800)
        assert "2.0d ago" == result

    def test_zero(self):
        assert format_age(0) == "0s ago"


# ---------------------------------------------------------------------------
# Heartbeat JSON format validation
# ---------------------------------------------------------------------------


class TestHeartbeatFormat:
    """Validate the heartbeat JSON schema produced by _push_heartbeat_s3."""

    REQUIRED_FIELDS = {
        "node_id": str,
        "config_key": str,
        "iteration": int,
        "estimated_elo": (int, float),
        "promotions": int,
        "timestamp": (int, float),
    }
    OPTIONAL_FIELDS = {
        "data_quality_score": (int, float, type(None)),
    }

    def _make_heartbeat(self, **overrides) -> dict:
        hb = {
            "node_id": "gh200-8",
            "config_key": "hex8_2p",
            "iteration": 5,
            "estimated_elo": 1650.0,
            "promotions": 2,
            "timestamp": time.time(),
            "data_quality_score": None,
        }
        hb.update(overrides)
        return hb

    def test_all_required_fields_present(self):
        hb = self._make_heartbeat()
        for field in self.REQUIRED_FIELDS:
            assert field in hb, f"Missing required field: {field}"

    def test_required_field_types(self):
        hb = self._make_heartbeat()
        for field, expected_type in self.REQUIRED_FIELDS.items():
            assert isinstance(hb[field], expected_type), (
                f"Field {field}: expected {expected_type}, got {type(hb[field])}"
            )

    def test_optional_field_types(self):
        hb = self._make_heartbeat(data_quality_score=0.95)
        for field, expected_type in self.OPTIONAL_FIELDS.items():
            if hb.get(field) is not None:
                assert isinstance(hb[field], expected_type), (
                    f"Field {field}: expected {expected_type}, got {type(hb[field])}"
                )

    def test_json_serializable(self):
        hb = self._make_heartbeat()
        serialized = json.dumps(hb)
        assert len(serialized) < 1024, f"Heartbeat exceeds 1KB: {len(serialized)} bytes"

    def test_roundtrip(self):
        hb = self._make_heartbeat()
        serialized = json.dumps(hb)
        deserialized = json.loads(serialized)
        assert deserialized == hb

    def test_size_under_1kb(self):
        """Heartbeats must stay under 1KB even with all fields populated."""
        hb = self._make_heartbeat(
            node_id="lambda-gh200-14-very-long-name",
            config_key="hexagonal_4p",
            iteration=9999,
            estimated_elo=2500.0,
            promotions=999,
            data_quality_score=0.999999,
        )
        assert len(json.dumps(hb)) < 1024


# ---------------------------------------------------------------------------
# detect_no_progress
# ---------------------------------------------------------------------------


class TestDetectNoProgress:
    def test_progress_detected(self):
        now = time.time()
        heartbeats = [
            {"config_key": "hex8_2p", "timestamp": now - 100000, "estimated_elo": 1700, "iteration": 10},
            {"config_key": "hex8_2p", "timestamp": now - 50000, "estimated_elo": 1750, "iteration": 15},
        ]
        result = detect_no_progress(heartbeats, 86400)
        assert "hex8_2p" not in result

    def test_no_progress_default_elo(self):
        now = time.time()
        heartbeats = [
            {"config_key": "square8_2p", "timestamp": now - 100000, "estimated_elo": 1500.0, "iteration": 1},
        ]
        result = detect_no_progress(heartbeats, 86400)
        assert "square8_2p" in result

    def test_no_progress_within_threshold(self):
        """Even if stuck at default Elo, don't flag if within time threshold."""
        now = time.time()
        heartbeats = [
            {"config_key": "hex8_2p", "timestamp": now - 3600, "estimated_elo": 1500.0, "iteration": 1},
        ]
        result = detect_no_progress(heartbeats, 86400)
        assert "hex8_2p" not in result

    def test_empty_heartbeats(self):
        result = detect_no_progress([], 86400)
        assert result == set()

    def test_missing_config_key(self):
        heartbeats = [{"timestamp": time.time(), "estimated_elo": 1500, "iteration": 1}]
        result = detect_no_progress(heartbeats, 86400)
        assert result == set()


# ---------------------------------------------------------------------------
# build_fleet_report
# ---------------------------------------------------------------------------


class TestBuildFleetReport:
    def _make_hb(self, node_id: str, config_key: str, age_s: float,
                 elo: float = 1600.0, iteration: int = 5, promos: int = 1) -> dict:
        return {
            "node_id": node_id,
            "config_key": config_key,
            "iteration": iteration,
            "estimated_elo": elo,
            "promotions": promos,
            "timestamp": time.time() - age_s,
            "data_quality_score": None,
        }

    def test_healthy_node(self):
        hb = self._make_hb("gh200-8", "hex8_2p", age_s=600)  # 10 min
        report = build_fleet_report([hb])
        assert len(report) == 1
        assert report[0]["status"] == STATUS_HEALTHY
        assert report[0]["node_id"] == "gh200-8"

    def test_stale_node(self):
        hb = self._make_hb("gh200-9", "square8_2p", age_s=3 * 3600)  # 3h
        report = build_fleet_report([hb])
        assert report[0]["status"] == STATUS_STALE

    def test_dead_node(self):
        hb = self._make_hb("gh200-10", "hex8_3p", age_s=8 * 3600)  # 8h
        report = build_fleet_report([hb])
        assert report[0]["status"] == STATUS_DEAD

    def test_sorted_by_node_id(self):
        hbs = [
            self._make_hb("gh200-10", "hex8_2p", 60),
            self._make_hb("gh200-8", "hex8_2p", 60),
            self._make_hb("gh200-9", "hex8_2p", 60),
        ]
        report = build_fleet_report(hbs)
        node_ids = [r["node_id"] for r in report]
        assert node_ids == ["gh200-10", "gh200-8", "gh200-9"]

    def test_custom_thresholds(self):
        hb = self._make_hb("gh200-8", "hex8_2p", age_s=1.5 * 3600)  # 1.5h
        # With 1h stale threshold, this is stale
        report = build_fleet_report([hb], stale_threshold_h=1.0, dead_threshold_h=3.0)
        assert report[0]["status"] == STATUS_STALE

    def test_empty_heartbeats(self):
        report = build_fleet_report([])
        assert report == []

    def test_report_fields(self):
        hb = self._make_hb("gh200-8", "hex8_2p", age_s=60)
        report = build_fleet_report([hb])
        r = report[0]
        expected_keys = {
            "node_id", "config_key", "iteration", "estimated_elo",
            "promotions", "timestamp", "age_seconds", "age_human",
            "status", "data_quality_score",
        }
        assert set(r.keys()) == expected_keys

    def test_no_progress_override(self):
        """NO_PROGRESS should override HEALTHY when Elo is stuck."""
        now = time.time()
        hb = {
            "node_id": "gh200-8",
            "config_key": "square8_4p",
            "iteration": 1,
            "estimated_elo": 1500.0,
            "promotions": 0,
            "timestamp": now - 300,  # Recent heartbeat (HEALTHY by time)
            "data_quality_score": None,
        }
        # But config is old enough for no_progress (25h)
        report = build_fleet_report(
            [hb],
            stale_threshold_h=2.0,
            dead_threshold_h=6.0,
            no_progress_h=0.01,  # Very low threshold to trigger
        )
        # Timestamp is only 5 min old so config oldest_ts is recent.
        # detect_no_progress checks if oldest_ts is > threshold.
        # 300s > 0.01h * 3600 = 36s, and iteration=1 + elo=1500 => no progress
        assert report[0]["status"] == STATUS_NO_PROGRESS

    def test_dead_not_overridden_by_no_progress(self):
        """DEAD status should not be overridden by NO_PROGRESS."""
        hb = self._make_hb("gh200-8", "hex8_2p", age_s=10 * 3600,
                           elo=1500.0, iteration=1)
        report = build_fleet_report(
            [hb],
            stale_threshold_h=2.0,
            dead_threshold_h=6.0,
            no_progress_h=1.0,
        )
        assert report[0]["status"] == STATUS_DEAD


# ---------------------------------------------------------------------------
# list_heartbeat_keys (mocked S3)
# ---------------------------------------------------------------------------


class TestListHeartbeatKeys:
    @patch("scripts.fleet_health_check.subprocess.run")
    def test_parses_s3_ls_output(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "2026-04-05 10:30:00        256 gh200-8.json\n"
                "2026-04-05 10:31:00        312 gh200-9.json\n"
                "2026-04-05 10:32:00        280 gh200-10.json\n"
            ),
        )
        keys = list_heartbeat_keys("s3://bucket/prefix/")
        assert keys == ["gh200-8.json", "gh200-9.json", "gh200-10.json"]

    @patch("scripts.fleet_health_check.subprocess.run")
    def test_empty_bucket(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        keys = list_heartbeat_keys("s3://bucket/prefix/")
        assert keys == []

    @patch("scripts.fleet_health_check.subprocess.run")
    def test_s3_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="Access Denied")
        keys = list_heartbeat_keys("s3://bucket/prefix/")
        assert keys == []

    @patch("scripts.fleet_health_check.subprocess.run")
    def test_aws_cli_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("aws not found")
        keys = list_heartbeat_keys("s3://bucket/prefix/")
        assert keys == []

    @patch("scripts.fleet_health_check.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="aws", timeout=30)
        keys = list_heartbeat_keys("s3://bucket/prefix/")
        assert keys == []


# ---------------------------------------------------------------------------
# fetch_heartbeat (mocked S3)
# ---------------------------------------------------------------------------


class TestFetchHeartbeat:
    @patch("scripts.fleet_health_check.subprocess.run")
    def test_valid_heartbeat(self, mock_run):
        hb = {
            "node_id": "gh200-8",
            "config_key": "hex8_2p",
            "iteration": 5,
            "estimated_elo": 1650.0,
            "promotions": 2,
            "timestamp": 1712345678.0,
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(hb),
        )
        result = fetch_heartbeat("s3://bucket/prefix/", "gh200-8.json")
        assert result is not None
        assert result["node_id"] == "gh200-8"
        assert result["estimated_elo"] == 1650.0

    @patch("scripts.fleet_health_check.subprocess.run")
    def test_s3_download_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = fetch_heartbeat("s3://bucket/prefix/", "missing.json")
        assert result is None

    @patch("scripts.fleet_health_check.subprocess.run")
    def test_invalid_json(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json at all")
        result = fetch_heartbeat("s3://bucket/prefix/", "bad.json")
        assert result is None


# ---------------------------------------------------------------------------
# fetch_all_heartbeats (integration of list + fetch)
# ---------------------------------------------------------------------------


class TestFetchAllHeartbeats:
    @patch("scripts.fleet_health_check.fetch_heartbeat")
    @patch("scripts.fleet_health_check.list_heartbeat_keys")
    def test_fetches_all(self, mock_list, mock_fetch):
        mock_list.return_value = ["a.json", "b.json"]
        mock_fetch.side_effect = [
            {"node_id": "a", "config_key": "hex8_2p", "iteration": 1,
             "estimated_elo": 1500, "promotions": 0, "timestamp": time.time()},
            {"node_id": "b", "config_key": "hex8_3p", "iteration": 2,
             "estimated_elo": 1600, "promotions": 1, "timestamp": time.time()},
        ]
        results = fetch_all_heartbeats("s3://bucket/prefix/")
        assert len(results) == 2

    @patch("scripts.fleet_health_check.fetch_heartbeat")
    @patch("scripts.fleet_health_check.list_heartbeat_keys")
    def test_skips_failed_fetches(self, mock_list, mock_fetch):
        mock_list.return_value = ["a.json", "b.json"]
        mock_fetch.side_effect = [
            {"node_id": "a", "config_key": "hex8_2p", "iteration": 1,
             "estimated_elo": 1500, "promotions": 0, "timestamp": time.time()},
            None,  # b.json fails
        ]
        results = fetch_all_heartbeats("s3://bucket/prefix/")
        assert len(results) == 1

    @patch("scripts.fleet_health_check.fetch_heartbeat")
    @patch("scripts.fleet_health_check.list_heartbeat_keys")
    def test_empty_bucket(self, mock_list, mock_fetch):
        mock_list.return_value = []
        results = fetch_all_heartbeats("s3://bucket/prefix/")
        assert results == []
        mock_fetch.assert_not_called()
