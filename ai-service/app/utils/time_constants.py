"""Time-related constants for consistent usage across the codebase.

These constants replace magic numbers like 86400, 3600, etc. for better
readability and maintainability.

Usage:
    from app.utils.time_constants import SECONDS_PER_DAY, SECONDS_PER_HOUR

    cutoff = time.time() - (max_age_days * SECONDS_PER_DAY)
    timeout_hours = duration_seconds / SECONDS_PER_HOUR
"""

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
