"""Tests for datetime utilities."""

from datetime import datetime, timedelta, timezone

import pytest

from app.utils.datetime_utils import (
    date_str,
    format_age,
    format_duration,
    iso_now,
    iso_now_ms,
    parse_iso,
    time_ago,
    timestamp_str,
    to_iso,
    utc_now,
    utc_timestamp,
)


class TestUtcNow:
    """Tests for utc_now()."""

    def test_returns_datetime(self):
        result = utc_now()
        assert isinstance(result, datetime)

    def test_has_utc_timezone(self):
        result = utc_now()
        assert result.tzinfo == timezone.utc

    def test_is_recent(self):
        before = datetime.now(timezone.utc)
        result = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


class TestUtcTimestamp:
    """Tests for utc_timestamp()."""

    def test_returns_float(self):
        result = utc_timestamp()
        assert isinstance(result, float)

    def test_is_positive(self):
        result = utc_timestamp()
        assert result > 0

    def test_matches_utc_now(self):
        ts = utc_timestamp()
        now = utc_now()
        # Should be within 1 second
        assert abs(ts - now.timestamp()) < 1.0


class TestIsoNow:
    """Tests for iso_now()."""

    def test_returns_string(self):
        result = iso_now()
        assert isinstance(result, str)

    def test_ends_with_z(self):
        result = iso_now()
        assert result.endswith("Z")

    def test_is_parseable(self):
        result = iso_now()
        # Should be parseable back to datetime
        dt = parse_iso(result)
        assert isinstance(dt, datetime)

    def test_format(self):
        result = iso_now()
        # Format: YYYY-MM-DDTHH:MM:SSZ
        assert len(result) == 20
        assert result[4] == "-"
        assert result[7] == "-"
        assert result[10] == "T"
        assert result[13] == ":"
        assert result[16] == ":"


class TestIsoNowMs:
    """Tests for iso_now_ms()."""

    def test_returns_string(self):
        result = iso_now_ms()
        assert isinstance(result, str)

    def test_ends_with_z(self):
        result = iso_now_ms()
        assert result.endswith("Z")

    def test_has_milliseconds(self):
        result = iso_now_ms()
        # Should have a decimal point for milliseconds
        assert "." in result

    def test_format(self):
        result = iso_now_ms()
        # Format: YYYY-MM-DDTHH:MM:SS.mmmZ
        assert len(result) == 24
        assert result[19] == "."


class TestToIso:
    """Tests for to_iso()."""

    def test_utc_datetime(self):
        dt = datetime(2025, 12, 19, 10, 30, 0, tzinfo=timezone.utc)
        result = to_iso(dt)
        assert result == "2025-12-19T10:30:00Z"

    def test_naive_datetime_assumed_utc(self):
        dt = datetime(2025, 12, 19, 10, 30, 0)
        result = to_iso(dt)
        assert result == "2025-12-19T10:30:00Z"

    def test_non_utc_converted(self):
        # Create a datetime in a different timezone
        tz_plus_5 = timezone(timedelta(hours=5))
        dt = datetime(2025, 12, 19, 15, 30, 0, tzinfo=tz_plus_5)
        result = to_iso(dt)
        # 15:30 +5 = 10:30 UTC
        assert result == "2025-12-19T10:30:00Z"


class TestParseIso:
    """Tests for parse_iso()."""

    def test_z_suffix(self):
        result = parse_iso("2025-12-19T10:30:00Z")
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 19
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 0
        assert result.tzinfo == timezone.utc

    def test_plus_offset(self):
        result = parse_iso("2025-12-19T10:30:00+00:00")
        assert result.tzinfo is not None

    def test_with_microseconds(self):
        result = parse_iso("2025-12-19T10:30:00.123456Z")
        assert result.microsecond == 123456

    def test_roundtrip(self):
        original = utc_now()
        iso_str = to_iso(original)
        parsed = parse_iso(iso_str)
        # Should be within 1 second (to_iso loses microseconds)
        assert abs((parsed - original).total_seconds()) < 1.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_iso("not-a-timestamp")


class TestTimeAgo:
    """Tests for time_ago()."""

    def test_days_ago(self):
        result = time_ago(days=1)
        now = utc_now()
        delta = now - result
        assert 23 * 3600 < delta.total_seconds() < 25 * 3600

    def test_hours_ago(self):
        result = time_ago(hours=2)
        now = utc_now()
        delta = now - result
        assert 7100 < delta.total_seconds() < 7300

    def test_minutes_ago(self):
        result = time_ago(minutes=30)
        now = utc_now()
        delta = now - result
        assert 1790 < delta.total_seconds() < 1810

    def test_combined(self):
        result = time_ago(days=1, hours=2, minutes=30)
        now = utc_now()
        expected_seconds = 86400 + 7200 + 1800
        delta = now - result
        assert abs(delta.total_seconds() - expected_seconds) < 10

    def test_has_utc_timezone(self):
        result = time_ago(hours=1)
        assert result.tzinfo == timezone.utc


class TestFormatDuration:
    """Tests for format_duration()."""

    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration(90) == "1m 30s"

    def test_hours_minutes_seconds(self):
        assert format_duration(3661) == "1h 1m 1s"

    def test_days(self):
        assert format_duration(90000) == "1d 1h"

    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_negative(self):
        assert format_duration(-10) == "0s"

    def test_float(self):
        assert format_duration(90.5) == "1m 30s"

    def test_large_value(self):
        # 2 days, 3 hours, 4 minutes, 5 seconds
        seconds = 2 * 86400 + 3 * 3600 + 4 * 60 + 5
        assert format_duration(seconds) == "2d 3h 4m 5s"


class TestFormatAge:
    """Tests for format_age()."""

    def test_recent(self):
        dt = utc_now() - timedelta(seconds=30)
        result = format_age(dt)
        assert "30s" in result

    def test_hours_ago(self):
        dt = utc_now() - timedelta(hours=2, minutes=30)
        result = format_age(dt)
        assert "2h" in result

    def test_naive_datetime(self):
        # Naive datetime should be treated as UTC
        dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        result = format_age(dt)
        assert "1h" in result or "59m" in result

    def test_future_datetime(self):
        dt = utc_now() + timedelta(hours=1)
        result = format_age(dt)
        assert result == "in the future"


class TestDateStr:
    """Tests for date_str()."""

    def test_default_format(self):
        result = date_str()
        assert len(result) == 8
        # All digits
        assert result.isdigit()

    def test_custom_format(self):
        dt = datetime(2025, 12, 19, tzinfo=timezone.utc)
        result = date_str(dt, format="%Y-%m-%d")
        assert result == "2025-12-19"

    def test_specific_date(self):
        dt = datetime(2025, 1, 5, tzinfo=timezone.utc)
        result = date_str(dt)
        assert result == "20250105"


class TestTimestampStr:
    """Tests for timestamp_str()."""

    def test_format(self):
        result = timestamp_str()
        # Format: YYYYMMDD_HHMMSS
        assert len(result) == 15
        assert result[8] == "_"

    def test_specific_datetime(self):
        dt = datetime(2025, 12, 19, 10, 30, 45, tzinfo=timezone.utc)
        result = timestamp_str(dt)
        assert result == "20251219_103045"

    def test_default_uses_now(self):
        before = datetime.now(timezone.utc).strftime("%Y%m%d")
        result = timestamp_str()
        after = datetime.now(timezone.utc).strftime("%Y%m%d")
        # Date part should be today
        assert result.startswith(before) or result.startswith(after)
