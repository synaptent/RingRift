# Core Module

Shared infrastructure utilities for the RingRift AI service.

## Overview

This module provides standardized infrastructure used across all scripts:

- Logging configuration
- Error handling and retry logic
- Graceful shutdown coordination
- Singleton patterns
- Task management
- Serialization utilities

## Key Components

### `logging_config.py` - Unified Logging

```python
from app.core import setup_logging, get_logger

# Setup at application start
setup_logging(level="INFO", log_file="logs/app.log")

# Get logger for module
logger = get_logger(__name__)
logger.info("Processing started", extra={"batch_size": 128})
```

### `error_handler.py` - Error Handling

```python
from app.core import retry, retry_async, FatalError, RetryableError

# Sync retry decorator
@retry(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_data():
    response = requests.get(url)
    if response.status_code == 429:
        raise RetryableError("Rate limited")
    return response.json()

# Async retry
@retry_async(max_attempts=5, delay=0.5)
async def async_fetch():
    ...

# Fatal errors stop retries immediately
def process():
    if critical_failure:
        raise FatalError("Unrecoverable")
```

### `shutdown.py` - Graceful Shutdown

```python
from app.core import (
    get_shutdown_manager,
    request_shutdown,
    is_shutting_down,
    on_shutdown,
    shutdown_scope,
)

# Register shutdown handler
@on_shutdown
def cleanup():
    db.close()

# Check shutdown state
if is_shutting_down():
    return early

# Context manager for cleanup
with shutdown_scope():
    run_long_task()

# Request shutdown
request_shutdown(reason="User interrupt")
```

### `singleton_mixin.py` - Singleton Patterns

```python
from app.core import SingletonMixin, ThreadSafeSingletonMixin, singleton

# Mixin approach
class DatabasePool(ThreadSafeSingletonMixin):
    def __init__(self):
        self.connections = []

pool = DatabasePool.instance()

# Decorator approach
@singleton
class ConfigManager:
    def __init__(self):
        self.config = {}
```

### `tasks.py` - Background Task Management

```python
from app.core import TaskManager, background_task, get_task_manager

# Decorator for background tasks
@background_task(name="data_sync")
async def sync_data():
    ...

# Manual task management
manager = get_task_manager()
task_id = manager.start_task("process_batch", coro)
status = manager.get_status(task_id)
manager.cancel_task(task_id)
```

### `marshalling.py` - Serialization

```python
from app.core import serialize, deserialize, to_json, from_json, Serializable

# Dataclass serialization
@dataclass
class GameResult(Serializable):
    game_id: str
    winner: int
    moves: list[dict]

result = GameResult(game_id="abc", winner=1, moves=[])
json_str = to_json(result)
restored = from_json(json_str, GameResult)

# Custom codecs
from app.core import register_codec, Codec

class DateCodec(Codec[datetime]):
    def encode(self, obj: datetime) -> str:
        return obj.isoformat()
    def decode(self, data: str) -> datetime:
        return datetime.fromisoformat(data)

register_codec(datetime, DateCodec())
```

### Additional Utilities

| File               | Purpose                        |
| ------------------ | ------------------------------ |
| `async_context.py` | Async context management       |
| `event_bus.py`     | In-process event pub/sub       |
| `health.py`        | Health check utilities         |
| `initializable.py` | Lazy initialization pattern    |
| `lifecycle.py`     | Component lifecycle management |
| `locking.py`       | Distributed locking utilities  |

## Usage Patterns

### Standard Script Setup

```python
from app.core import setup_logging, get_logger, on_shutdown

# Initialize
setup_logging(level="INFO")
logger = get_logger(__name__)

@on_shutdown
def cleanup():
    logger.info("Shutting down...")

def main():
    logger.info("Starting...")
    # ... work ...

if __name__ == "__main__":
    main()
```

### Error Recovery

```python
from app.core import retry, FatalError

@retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
def connect_to_service():
    try:
        return client.connect()
    except AuthError:
        raise FatalError("Invalid credentials")  # No retry
```

## Thread Safety

- `ThreadSafeSingletonMixin`: Thread-safe singleton with locking
- `TaskManager`: Thread-safe task tracking
- `ShutdownManager`: Atomic shutdown coordination
