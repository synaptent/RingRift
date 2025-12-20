"""Time-related constants for consistent usage across the codebase.

These constants replace magic numbers like 86400, 3600, etc. for better
readability and maintainability.

Usage:
    from app.utils.time_constants import SECONDS_PER_DAY, SECONDS_PER_HOUR

    cutoff = time.time() - (max_age_days * SECONDS_PER_DAY)
    timeout_hours = duration_seconds / SECONDS_PER_HOUR
"""

__all__ = [
    "DAYS_PER_WEEK",
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_MAX",
    # Common thresholds
    "DEFAULT_CACHE_TTL",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_SYNC_INTERVAL",
    "DEFAULT_TIMEOUT",
    "FAST_HEARTBEAT_INTERVAL",
    "FIFTEEN_MINUTES",
    "FIVE_MINUTES",
    "FIVE_MINUTES_MS",
    "FIVE_SECONDS_MS",
    "FREQUENT_SYNC_INTERVAL",
    # Hours/days-based
    "HOURS_PER_DAY",
    "HOURS_PER_WEEK",
    "INFREQUENT_SYNC_INTERVAL",
    "LONG_CACHE_TTL",
    "LONG_TIMEOUT",
    "MS_PER_DAY",
    "MS_PER_HOUR",
    "MS_PER_MINUTE",
    # Milliseconds-based
    "MS_PER_SECOND",
    "OLD_DATA_THRESHOLD",
    "ONE_DAY",
    "ONE_HOUR",
    "ONE_MINUTE",
    "ONE_MINUTE_MS",
    "ONE_SECOND_MS",
    "ONE_WEEK",
    "SECONDS_PER_DAY",
    "SECONDS_PER_HOUR",
    # Seconds-based
    "SECONDS_PER_MINUTE",
    "SECONDS_PER_WEEK",
    "SHORT_CACHE_TTL",
    "SHORT_TIMEOUT",
    "SIX_HOURS",
    "STALE_DATA_THRESHOLD",
    "TEN_MINUTES",
    "TEN_SECONDS_MS",
    "THIRTY_MINUTES",
    "THIRTY_SECONDS_MS",
    "TWELVE_HOURS",
    "TWO_HOURS",
]

# =============================================================================
# Seconds-based constants
# =============================================================================

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
SECONDS_PER_WEEK = 604800

# Common timeout/interval values
ONE_MINUTE = 60
FIVE_MINUTES = 300
TEN_MINUTES = 600
FIFTEEN_MINUTES = 900
THIRTY_MINUTES = 1800
ONE_HOUR = 3600
TWO_HOURS = 7200
SIX_HOURS = 21600
TWELVE_HOURS = 43200
ONE_DAY = 86400
ONE_WEEK = 604800

# =============================================================================
# Milliseconds-based constants
# =============================================================================

MS_PER_SECOND = 1000
MS_PER_MINUTE = 60_000
MS_PER_HOUR = 3_600_000
MS_PER_DAY = 86_400_000

# Common timeout values in milliseconds
ONE_SECOND_MS = 1000
FIVE_SECONDS_MS = 5000
TEN_SECONDS_MS = 10_000
THIRTY_SECONDS_MS = 30_000
ONE_MINUTE_MS = 60_000
FIVE_MINUTES_MS = 300_000

# =============================================================================
# Hours/days-based constants
# =============================================================================

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168
DAYS_PER_WEEK = 7

# =============================================================================
# Common thresholds
# =============================================================================

# Default cache TTLs
DEFAULT_CACHE_TTL = ONE_HOUR
LONG_CACHE_TTL = ONE_DAY
SHORT_CACHE_TTL = FIVE_MINUTES

# Default sync intervals
DEFAULT_SYNC_INTERVAL = FIVE_MINUTES
FREQUENT_SYNC_INTERVAL = ONE_MINUTE
INFREQUENT_SYNC_INTERVAL = ONE_HOUR

# Default timeouts
DEFAULT_TIMEOUT = THIRTY_MINUTES
SHORT_TIMEOUT = FIVE_MINUTES
LONG_TIMEOUT = ONE_HOUR

# Heartbeat intervals
DEFAULT_HEARTBEAT_INTERVAL = THIRTY_SECONDS_MS
FAST_HEARTBEAT_INTERVAL = TEN_SECONDS_MS

# Circuit breaker backoff
DEFAULT_BACKOFF_MAX = ONE_HOUR
DEFAULT_BACKOFF_BASE = ONE_MINUTE

# Age thresholds
STALE_DATA_THRESHOLD = ONE_DAY
OLD_DATA_THRESHOLD = ONE_WEEK
