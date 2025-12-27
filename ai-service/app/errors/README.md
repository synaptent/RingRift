# RingRift Error Handling Module

Centralized exception hierarchy for RingRift AI Service.

## Overview

This module provides typed exceptions that replace bare `except Exception:` handlers
throughout the codebase. All exceptions include:

- **Error codes** for structured logging and alerting
- **Retry hints** indicating if the error is transient
- **Context details** as key-value pairs

## Quick Start

```python
from app.errors import (
    RingRiftError,
    NetworkError,
    SyncError,
    TrainingError,
)

try:
    await sync_data(host)
except NetworkError as e:
    # Retryable - network issues
    await retry_with_backoff()
except SyncError as e:
    # Check if retryable
    if e.retryable:
        await retry_with_backoff()
    else:
        logger.error(f"Non-recoverable sync error: {e}")
        raise
```

## Exception Hierarchy

```
RingRiftError (base)
├── ResourceError          # GPU, memory, disk
│   ├── GPUError
│   │   └── GPUOutOfMemoryError
│   ├── DiskError
│   └── MemoryExhaustedError
├── NetworkError           # Connections, SSH, HTTP, P2P
│   ├── ConnectionError
│   ├── ConnectionTimeoutError
│   ├── SSHError
│   │   └── SSHAuthError
│   ├── HTTPError
│   └── P2PError
│       └── P2PLeaderUnavailableError
├── SyncError              # Data synchronization
│   ├── SyncTimeoutError
│   ├── SyncConflictError
│   └── SyncIntegrityError
├── TrainingError          # Model training
│   ├── DataQualityError
│   ├── ModelLoadError
│   │   └── CheckpointCorruptError
│   ├── ConvergenceError
│   └── ModelVersioningError
├── DaemonError            # Daemon lifecycle
│   ├── DaemonStartupError
│   ├── DaemonCrashError
│   ├── DaemonConfigError
│   └── DaemonDependencyError
├── ValidationError        # Input/output validation
│   ├── SchemaError
│   └── ParityError
├── ConfigurationError     # Configuration problems
│   ├── ConfigMissingError
│   └── ConfigTypeError
└── System errors
    ├── EmergencyHaltError
    ├── RetryableError
    └── NonRetryableError
```

## Error Codes

Each exception has a numeric error code for structured handling:

| Range | Category      | Examples                                         |
| ----- | ------------- | ------------------------------------------------ |
| 1xx   | Resource      | GPU_OOM (102), DISK_FULL (103)                   |
| 2xx   | Network       | CONNECTION_TIMEOUT (201), SSH_AUTH_FAILED (202)  |
| 3xx   | Sync          | SYNC_TIMEOUT (300), SYNC_INTEGRITY_FAILED (302)  |
| 4xx   | Training      | DATA_QUALITY_LOW (400), CHECKPOINT_CORRUPT (403) |
| 5xx   | Daemon        | DAEMON_START_FAILED (500), DAEMON_CRASHED (501)  |
| 6xx   | Validation    | VALIDATION_FAILED (600), PARITY_FAILED (602)     |
| 7xx   | Configuration | CONFIG_MISSING (700), CONFIG_INVALID (701)       |
| 8xx   | System        | EMERGENCY_HALT (800)                             |

## Retry Guidelines

| Exception Type     | Retryable | Notes                      |
| ------------------ | --------- | -------------------------- |
| NetworkError       | Yes       | Use exponential backoff    |
| ResourceError      | Yes       | Wait for resources to free |
| SyncTimeoutError   | Yes       | Network may recover        |
| SyncIntegrityError | No        | Data is corrupted          |
| TrainingError      | No        | Needs intervention         |
| ValidationError    | No        | Fix the data               |
| ConfigurationError | No        | Fix the config             |
| DaemonError        | Depends   | Check `.retryable`         |

## Usage Patterns

### Replace Bare Exceptions

```python
# Before (bad)
try:
    await ssh_command(host, cmd)
except Exception:
    pass

# After (good)
from app.errors import SSHError, ConnectionTimeoutError

try:
    await ssh_command(host, cmd)
except ConnectionTimeoutError:
    await retry_with_backoff()
except SSHError as e:
    logger.error(f"SSH failed: {e}")
    raise
```

### Add Context

```python
raise NetworkError(
    "Failed to connect to training node",
    details={
        "host": host,
        "port": port,
        "attempt": attempt,
    }
)
```

### JSON Serialization

```python
try:
    do_something()
except RingRiftError as e:
    error_json = e.to_dict()
    # {
    #   "error": "NetworkError",
    #   "message": "...",
    #   "code": 200,
    #   "code_name": "CONNECTION_FAILED",
    #   "retryable": true,
    #   "details": {...}
    # }
```

## December 2025 Update

This module was created to consolidate error handling across the codebase.
All new code should use these exceptions instead of bare `except Exception:`.

**Related PRs:**

- Phase 3: Reduced bare exceptions from 134 to 1 (99.3%)
- Typed handlers in 70+ files
