# Event Handler Development Patterns

**Last Updated**: January 3, 2026

This guide documents best practices for writing event handlers in the RingRift coordination layer. Following these patterns ensures consistency and reduces bugs.

## Quick Start

```python
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.event_utils import normalize_event_payload, extract_config_key

class MyDaemon(HandlerBase):
    """Example daemon implementing standard patterns."""

    def __init__(self):
        super().__init__(name="my_daemon", cycle_interval=60.0)
        self._processed_configs: set[str] = set()

    async def _run_cycle(self) -> None:
        """Main work loop - called every cycle_interval seconds."""
        pass  # Your periodic work here

    def _get_event_subscriptions(self) -> dict:
        """Define event handlers."""
        return {
            "TRAINING_COMPLETED": self._on_training_completed,
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
        }

    async def _on_training_completed(self, event: dict) -> None:
        """Handle training completion event."""
        # 1. Normalize payload (handles different event formats)
        payload = normalize_event_payload(event)

        # 2. Extract config key using utility
        config_key = extract_config_key(payload)
        if not config_key:
            return  # Skip if no config

        # 3. Deduplicate (optional but recommended)
        if self._is_duplicate_event(event):
            return

        # 4. Your handler logic here
        self.logger.info(f"Training completed for {config_key}")
```

## Core Patterns

### 1. Inherit from HandlerBase

Always inherit from `HandlerBase` for daemons that handle events:

```python
from app.coordination.handler_base import HandlerBase, HealthCheckResult

class MyDaemon(HandlerBase):
    def __init__(self, config: Optional[MyConfig] = None):
        super().__init__(
            name="my_daemon",           # Unique daemon name
            cycle_interval=60.0,        # Seconds between _run_cycle calls
            config=config,              # Optional configuration
        )
```

**What HandlerBase provides:**

- Singleton management with thread-safe access
- Event subscription infrastructure
- Safe event emission (`_safe_emit_event()`, `_safe_emit_event_async()`)
- Event deduplication (hash-based with TTL)
- Standardized health check format
- Error tracking with bounded log
- Async lifecycle management (start/stop)

### 2. Use Event Utilities

**Always use `normalize_event_payload()` to extract payload:**

```python
from app.coordination.event_utils import normalize_event_payload

async def _on_my_event(self, event: dict) -> None:
    # This handles: event.payload, event.metadata, or dict event
    payload = normalize_event_payload(event)
```

**Use specialized extractors for common patterns:**

```python
from app.coordination.event_utils import (
    normalize_event_payload,
    extract_config_key,         # Get config_key from payload
    parse_config_key,           # Parse "hex8_2p" -> (board_type, num_players)
    extract_evaluation_data,    # Full extraction for EVALUATION_COMPLETED
    extract_training_data,      # Full extraction for TRAINING_COMPLETED
)

async def _on_evaluation_completed(self, event: dict) -> None:
    payload = normalize_event_payload(event)

    # Simple extraction
    config_key = extract_config_key(payload)

    # Or full extraction with parsed fields
    data = extract_evaluation_data(payload)
    if data.is_valid:
        print(f"Board: {data.board_type}, Players: {data.num_players}")
        print(f"Elo: {data.elo}, Win Rate: {data.win_rate}")
```

### 3. Deduplicate Events

Use `_is_duplicate_event()` to prevent processing the same event twice:

```python
async def _on_training_completed(self, event: dict) -> None:
    # Check for duplicate BEFORE processing
    if self._is_duplicate_event(event):
        self.stats.events_deduplicated += 1
        return

    # Process event
    payload = normalize_event_payload(event)
    # ...
```

**How deduplication works:**

- Creates SHA256 hash of event content
- Tracks hashes with 5-minute TTL (configurable)
- Returns True if hash was seen within TTL

### 4. Emit Events Safely

Use `_safe_emit_event_async()` for robust event emission:

```python
class MyDaemon(HandlerBase):
    _event_source = "MyDaemon"  # Identifies source in emitted events

    async def _run_cycle(self) -> None:
        # Safe emission with automatic error handling
        await self._safe_emit_event_async(
            "MY_EVENT_COMPLETED",
            {
                "config_key": "hex8_2p",
                "result": "success",
                "timestamp": time.time(),
            }
        )
```

**What `_safe_emit_event_async()` does:**

- Catches and logs emission errors (doesn't crash daemon)
- Adds source metadata to event
- Tracks emission stats

### 5. Implement Health Check

Override `health_check()` for daemon manager integration:

```python
def health_check(self) -> HealthCheckResult:
    """Return health status for daemon manager."""
    is_healthy = self.stats.errors_count < 10

    return HealthCheckResult(
        healthy=is_healthy,
        status="healthy" if is_healthy else "degraded",
        message=f"Processed {self.stats.events_processed} events",
        details={
            "events_processed": self.stats.events_processed,
            "errors_count": self.stats.errors_count,
            "last_activity": self.stats.last_activity,
            "success_rate": self.stats.success_rate,
        },
    )
```

## Event Types Reference

### Critical Events (Must Handle)

| Event                  | When Emitted                 | Required Payload Fields                           |
| ---------------------- | ---------------------------- | ------------------------------------------------- |
| `TRAINING_COMPLETED`   | Training finishes            | `config_key`, `model_path`, `loss`                |
| `EVALUATION_COMPLETED` | Model evaluation done        | `config_key`, `model_path`, `elo`, `games_played` |
| `MODEL_PROMOTED`       | Model promoted to production | `config_key`, `model_path`, `elo`                 |
| `DATA_SYNC_COMPLETED`  | Data sync finishes           | `source`, `target`, `files_synced`                |
| `REGRESSION_DETECTED`  | Elo regression found         | `config_key`, `model_path`, `elo_drop`            |

### Informational Events (Optional)

| Event                     | When Emitted          | Purpose           |
| ------------------------- | --------------------- | ----------------- |
| `SELFPLAY_COMPLETE`       | Selfplay batch done   | Progress tracking |
| `NEW_GAMES_AVAILABLE`     | New games in database | Pipeline trigger  |
| `PROGRESS_STALL_DETECTED` | No Elo improvement    | Recovery trigger  |

## Common Mistakes

### ❌ Wrong: Inline Payload Extraction

```python
# Don't do this - scattered and fragile
async def _on_event(self, event):
    config_key = event.get("config_key") or event.get("payload", {}).get("config_key")
    board_type = config_key.split("_")[0] if config_key else None
```

### ✓ Correct: Use Utilities

```python
from app.coordination.event_utils import normalize_event_payload, extract_config_key

async def _on_event(self, event):
    payload = normalize_event_payload(event)
    config_key = extract_config_key(payload)
```

### ❌ Wrong: Emit Without Error Handling

```python
# Don't do this - can crash daemon if emit fails
from app.coordination.event_router import emit_event
emit_event("MY_EVENT", {"data": "value"})
```

### ✓ Correct: Use Safe Emission

```python
# Safe emission with automatic error handling
await self._safe_emit_event_async("MY_EVENT", {"data": "value"})
```

### ❌ Wrong: No Deduplication

```python
# Don't do this - may process same event multiple times
async def _on_training_completed(self, event):
    payload = normalize_event_payload(event)
    # ... expensive processing ...
```

### ✓ Correct: Deduplicate First

```python
async def _on_training_completed(self, event):
    if self._is_duplicate_event(event):
        return
    payload = normalize_event_payload(event)
    # ... expensive processing ...
```

## Full Example: Quality Monitor Daemon

```python
"""Quality Monitor Daemon - monitors selfplay data quality."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.event_utils import (
    normalize_event_payload,
    extract_config_key,
    parse_config_key,
)

logger = logging.getLogger(__name__)


class QualityMonitorDaemon(HandlerBase):
    """Monitors selfplay data quality and emits alerts."""

    _instance: QualityMonitorDaemon | None = None
    _event_source = "QualityMonitorDaemon"

    def __init__(self):
        super().__init__(
            name="quality_monitor",
            cycle_interval=300.0,  # Check every 5 minutes
        )
        self._quality_scores: dict[str, float] = {}
        self._low_quality_configs: set[str] = set()

    @classmethod
    def get_instance(cls) -> QualityMonitorDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to relevant events."""
        return {
            "SELFPLAY_COMPLETE": self._on_selfplay_complete,
            "NEW_GAMES_AVAILABLE": self._on_new_games,
        }

    async def _run_cycle(self) -> None:
        """Periodic quality check cycle."""
        for config_key, score in self._quality_scores.items():
            if score < 0.5:
                self._low_quality_configs.add(config_key)
                await self._safe_emit_event_async(
                    "QUALITY_ALERT",
                    {
                        "config_key": config_key,
                        "quality_score": score,
                        "timestamp": time.time(),
                    },
                )

    async def _on_selfplay_complete(self, event: dict) -> None:
        """Handle selfplay completion."""
        if self._is_duplicate_event(event):
            return

        payload = normalize_event_payload(event)
        config_key = extract_config_key(payload)
        if not config_key:
            return

        # Update quality score
        quality = payload.get("quality_score", 1.0)
        self._quality_scores[config_key] = quality

        self.stats.events_processed += 1
        self.stats.success_count += 1

    async def _on_new_games(self, event: dict) -> None:
        """Handle new games notification."""
        if self._is_duplicate_event(event):
            return

        payload = normalize_event_payload(event)
        config_key = extract_config_key(payload)
        if not config_key:
            return

        # Trigger quality analysis
        parsed = parse_config_key(config_key)
        if parsed:
            logger.info(
                f"New games for {parsed.board_type} {parsed.num_players}p"
            )

        self.stats.events_processed += 1
        self.stats.success_count += 1

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        is_healthy = len(self._low_quality_configs) < 3

        return HealthCheckResult(
            healthy=is_healthy,
            status="healthy" if is_healthy else "degraded",
            message=f"Monitoring {len(self._quality_scores)} configs",
            details={
                "configs_monitored": len(self._quality_scores),
                "low_quality_configs": list(self._low_quality_configs),
                "events_processed": self.stats.events_processed,
                "errors_count": self.stats.errors_count,
            },
        )


def get_quality_monitor() -> QualityMonitorDaemon:
    """Get singleton QualityMonitorDaemon."""
    return QualityMonitorDaemon.get_instance()
```

## See Also

- `app/coordination/handler_base.py` - Base class implementation
- `app/coordination/event_utils.py` - Event extraction utilities
- `app/coordination/event_router.py` - Event bus implementation
- `app/coordination/data_events.py` - Event type definitions
- `docs/architecture/EVENT_SUBSCRIPTION_MATRIX.md` - Event wiring reference
