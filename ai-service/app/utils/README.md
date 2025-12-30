# Utils Module

Reusable utility modules for the RingRift AI service. This package provides foundational tools for configuration, resource management, game discovery, and common operations.

## Table of Contents

1. [Overview](#overview)
2. [Core Modules](#core-modules)
   - [GameDiscovery](#gamediscovery)
   - [ResourceGuard](#resourceguard)
   - [CanonicalNaming](#canonicalnaming)
   - [EnvConfig](#envconfig)
   - [TorchUtils](#torchutils)
3. [File Operations](#file-operations)
   - [FileUtils](#fileutils)
   - [Paths](#paths)
   - [YAMLUtils](#yamlutils)
   - [JSONUtils](#jsonutils)
4. [Debugging & Logging](#debugging--logging)
   - [DebugUtils](#debugutils)
   - [LoggingUtils](#loggingutils)
5. [Rate Limiting](#rate-limiting)
   - [RateLimit](#ratelimit)
   - [LoadThrottle](#loadthrottle)
6. [Other Utilities](#other-utilities)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)

---

## Overview

The utils package provides cross-cutting utilities used throughout the AI service:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Utils Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │ GameDiscovery│  │ResourceGuard │  │ CanonicalName│             │
│   │              │  │              │  │              │             │
│   │ Find all DBs │  │ Check limits │  │ Normalize    │             │
│   └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │   EnvConfig  │  │  TorchUtils  │  │  DebugUtils  │             │
│   │              │  │              │  │              │             │
│   │ Typed env    │  │ Device detect│  │ Parity debug │             │
│   └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │  RateLimit   │  │ LoadThrottle │  │ FileUtils    │             │
│   │              │  │              │  │              │             │
│   │ API limits   │  │ CPU throttle │  │ Safe I/O     │             │
│   └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Imports

```python
from app.utils import (
    # Device management
    get_device,
    get_device_info,
    # Environment configuration
    env,
    get_str,
    get_int,
    get_bool,
    # Game discovery
    GameDiscovery,
    find_all_game_databases,
    count_games_for_config,
)
```

---

## Core Modules

### GameDiscovery

Unified discovery of game databases across all storage patterns. This is the canonical way to find game data regardless of where it's stored.

**File**: `game_discovery.py` (~800 lines)

```python
from app.utils.game_discovery import (
    GameDiscovery,
    DatabaseInfo,
    GameCounts,
    find_all_game_databases,
    count_games_for_config,
    get_game_counts_summary,
)
```

#### Storage Patterns Discovered

| Pattern    | Example                                 | Description             |
| ---------- | --------------------------------------- | ----------------------- |
| Central    | `data/games/selfplay.db`                | Multi-config databases  |
| Per-config | `data/games/hex8_2p.db`                 | Single board/player DBs |
| Tournament | `data/games/tournament_sq8_4p.db`       | Tournament games        |
| Canonical  | `data/selfplay/canonical_hex8_2p.db`    | Canonical selfplay      |
| Unified    | `data/selfplay/unified_*/games.db`      | Unified loop DBs        |
| P2P        | `data/selfplay/p2p/{config}/*/games.db` | P2P cluster DBs         |
| Harvested  | `data/training/*/harvested_games.db`    | Training exports        |

#### Usage

```python
from app.utils.game_discovery import GameDiscovery

# Create discovery instance
discovery = GameDiscovery()

# Find all databases
all_dbs = discovery.find_all_databases()
for db_info in all_dbs:
    print(f"{db_info.path}: {db_info.game_count} games")
    print(f"  Board: {db_info.board_type}, Players: {db_info.num_players}")

# Find databases for specific configuration
hex8_2p_dbs = discovery.find_databases_for_config("hex8", 2)

# Get total games per config
counts = discovery.count_games_by_config()
# {'square8_2p': 50000, 'hex8_2p': 34000, ...}

# Get comprehensive counts
game_counts = discovery.get_game_counts()
print(f"Total games: {game_counts.total}")
print(f"Databases found: {game_counts.databases_found}")
print(f"By board type: {game_counts.by_board_type}")
```

#### Remote Discovery

```python
from app.utils.game_discovery import RemoteGameDiscovery

# Discover databases across cluster
remote = RemoteGameDiscovery()
cluster_dbs = remote.find_all_databases(
    hosts=["gpu-node-1", "gpu-node-2"],
    ssh_key="~/.ssh/id_cluster",
)
```

---

### ResourceGuard

Pre-operation resource checking with consistent limits across the codebase. **Do not use shutil.disk_usage(), psutil.virtual_memory(), or psutil.cpu_percent() directly** - use this module instead.

**File**: `resource_guard.py` (~1500 lines)

```python
from app.utils.resource_guard import (
    # Simple checks
    check_disk_space,
    check_memory,
    check_cpu,
    check_gpu_memory,
    can_proceed,
    # Waiting
    wait_for_resources,
    # Limits
    LIMITS,
    # Pressure monitors
    DiskPressureMonitor,
    MemoryPressureMonitor,
    DiskPressureLevel,
    MemoryPressureLevel,
)
```

#### Default Limits

| Resource   | Max Usage | Constant                    |
| ---------- | --------- | --------------------------- |
| CPU        | 80%       | `LIMITS.CPU_MAX_PERCENT`    |
| Memory     | 90%       | `LIMITS.MEMORY_MAX_PERCENT` |
| Disk       | 95%       | `LIMITS.DISK_MAX_PERCENT`   |
| GPU Memory | 85%       | `LIMITS.GPU_MAX_PERCENT`    |

#### Usage

```python
from app.utils.resource_guard import (
    check_disk_space,
    check_memory,
    check_gpu_memory,
    can_proceed,
    wait_for_resources,
    LIMITS,
)

# Before writing files
if not check_disk_space(required_gb=2.0):
    logger.warning("Insufficient disk space")
    return

# Check if we can proceed with operation
if not can_proceed():
    # Wait up to 5 minutes for resources
    if not wait_for_resources(timeout=300):
        raise RuntimeError("Resources not available")

# In long-running loops
for i in range(num_games):
    if i % 50 == 0 and not check_memory():
        logger.warning("Memory pressure, stopping early")
        break
    play_game()

# Check GPU before training
if not check_gpu_memory(required_gb=4.0):
    logger.error("Insufficient GPU memory")
    return
```

#### Pressure Monitors

```python
from app.utils.resource_guard import (
    DiskPressureMonitor,
    MemoryPressureMonitor,
    DiskPressureLevel,
)

# Create disk monitor
disk_monitor = DiskPressureMonitor(path="/data")

# Check pressure level
level = disk_monitor.get_pressure_level()
if level >= DiskPressureLevel.HIGH:
    # Trigger cleanup
    cleanup_old_files()

# Memory monitoring
mem_monitor = MemoryPressureMonitor()
if mem_monitor.is_critical():
    # Emergency measures
    clear_caches()
```

---

### CanonicalNaming

Single source of truth for board type naming and configuration keys.

**File**: `canonical_naming.py` (~250 lines)

```python
from app.utils.canonical_naming import (
    # Board type constants
    BOARD_SQUARE8,
    BOARD_SQUARE19,
    BOARD_HEX8,
    BOARD_HEXAGONAL,
    ALL_BOARD_TYPES,
    # Config key utilities
    make_config_key,
    parse_config_key,
    normalize_board_type,
    is_valid_board_type,
    get_board_type_enum,
    normalize_database_filename,
    CANONICAL_CONFIG_KEYS,
)
```

#### Canonical Values

| Board Type  | Constant          | Size      | Cells |
| ----------- | ----------------- | --------- | ----- |
| `square8`   | `BOARD_SQUARE8`   | 8x8       | 64    |
| `square19`  | `BOARD_SQUARE19`  | 19x19     | 361   |
| `hex8`      | `BOARD_HEX8`      | radius 4  | 61    |
| `hexagonal` | `BOARD_HEXAGONAL` | radius 12 | 469   |

#### Usage

```python
from app.utils.canonical_naming import (
    normalize_board_type,
    make_config_key,
    parse_config_key,
    BOARD_HEX8,
)

# Normalize aliases to canonical form
board = normalize_board_type("sq8")  # Returns "square8"
board = normalize_board_type("8x8")  # Returns "square8"
board = normalize_board_type("smallhex")  # Returns "hex8"

# Create config keys
key = make_config_key("hex8", 2)  # Returns "hex8_2p"
key = make_config_key(BOARD_HEX8, 4)  # Returns "hex8_4p"

# Parse config keys
board, players = parse_config_key("square8_2p")
# board = "square8", players = 2

# Normalize database filenames
filename = normalize_database_filename("selfplay_sq8_2player.db")
# Returns "selfplay_square8_2p.db"
```

---

### EnvConfig

Typed environment variable access with defaults and validation.

**File**: `env_config.py` (~300 lines)

```python
from app.utils.env_config import (
    EnvConfig,
    env,  # Singleton instance
    get_str,
    get_int,
    get_float,
    get_bool,
    get_list,
)
```

#### Usage

```python
from app.utils.env_config import env, get_int, get_bool

# Using singleton
debug_mode = env.get_bool("DEBUG", default=False)
batch_size = env.get_int("BATCH_SIZE", default=512)
api_key = env.get_str("API_KEY", required=True)

# Using convenience functions
port = get_int("PORT", 8080)
hosts = get_list("CLUSTER_HOSTS", separator=",")
enabled = get_bool("FEATURE_ENABLED", False)

# With validation
env.get_int("WORKERS", default=4, min_value=1, max_value=32)
env.get_float("LEARNING_RATE", default=0.001, min_value=0.0001)
```

#### Environment Variables

| Variable                         | Type | Default | Description                     |
| -------------------------------- | ---- | ------- | ------------------------------- |
| `RINGRIFT_DEBUG`                 | bool | False   | Enable debug mode               |
| `RINGRIFT_LOG_LEVEL`             | str  | INFO    | Logging level                   |
| `RINGRIFT_DATA_DIR`              | str  | data/   | Data directory                  |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | bool | True    | Skip shadow contract validation |

---

### TorchUtils

Safe PyTorch operations including device detection and memory management.

**File**: `torch_utils.py` (~500 lines)

```python
from app.utils.torch_utils import (
    get_device,
    get_device_info,
    safe_to_device,
    clear_gpu_memory,
    get_gpu_memory_usage,
    DeviceInfo,
)
```

#### Usage

```python
from app.utils.torch_utils import get_device, get_device_info

# Auto-detect best device
device = get_device()  # Returns "cuda", "mps", or "cpu"

# Get detailed info
info = get_device_info()
print(f"Device: {info.device_type}")
print(f"Name: {info.device_name}")
print(f"Memory: {info.total_memory_gb:.1f}GB")
print(f"CUDA available: {info.cuda_available}")

# Safe tensor transfer
from app.utils.torch_utils import safe_to_device
tensor = safe_to_device(my_tensor, device="cuda")

# Memory management
from app.utils.torch_utils import clear_gpu_memory, get_gpu_memory_usage
usage = get_gpu_memory_usage()  # Returns GB used
clear_gpu_memory()  # Free cached memory
```

---

## File Operations

### FileUtils

Safe file I/O operations with atomic writes and backup support.

**File**: `file_utils.py` (~200 lines)

```python
from app.utils.file_utils import (
    atomic_write,
    safe_read,
    ensure_directory,
    backup_file,
    get_file_hash,
)
```

#### Usage

```python
from app.utils.file_utils import atomic_write, safe_read

# Atomic write (prevents partial writes)
atomic_write("config.json", json.dumps(config))

# Safe read with fallback
content = safe_read("config.json", default="{}")

# Ensure directory exists
ensure_directory("data/output/models")

# Backup before modification
backup_file("important.db")  # Creates important.db.bak
```

### Paths

Path resolution and standard directory locations.

**File**: `paths.py` (~200 lines)

```python
from app.utils.paths import (
    get_project_root,
    get_data_dir,
    get_models_dir,
    get_logs_dir,
    resolve_path,
)
```

#### Usage

```python
from app.utils.paths import get_project_root, get_data_dir

root = get_project_root()  # /path/to/ai-service
data = get_data_dir()  # /path/to/ai-service/data
models = get_models_dir()  # /path/to/ai-service/models

# Resolve relative path
full_path = resolve_path("data/games/selfplay.db")
```

### YAMLUtils

Safe YAML loading and saving.

**File**: `yaml_utils.py` (~300 lines)

```python
from app.utils.yaml_utils import (
    load_yaml,
    save_yaml,
    merge_yaml,
)
```

### JSONUtils

JSON utilities with custom encoders for game types.

**File**: `json_utils.py` (~200 lines)

```python
from app.utils.json_utils import (
    dump_json,
    load_json,
    GameJSONEncoder,
)
```

---

## Debugging & Logging

### DebugUtils

State comparison and parity debugging utilities harvested from debug scripts.

**File**: `debug_utils.py` (~600 lines)

```python
from app.utils.debug_utils import (
    compare_game_states,
    format_state_diff,
    dump_state_to_file,
    ParityDebugger,
)
```

#### Usage

```python
from app.utils.debug_utils import compare_game_states, ParityDebugger

# Compare two game states
diff = compare_game_states(python_state, ts_state)
if diff:
    print(f"Divergence found: {diff}")

# Debug parity issues
debugger = ParityDebugger(game_id="abc123")
debugger.add_python_state(state, move_number=15)
debugger.add_ts_state(ts_state, move_number=15)
debugger.report_divergence()
```

### LoggingUtils

Consistent logging setup across scripts.

**File**: `logging_utils.py` (~200 lines)

```python
from app.utils.logging_utils import (
    setup_logging,
    get_logger,
    LogLevel,
)
```

#### Usage

```python
from app.utils.logging_utils import setup_logging, get_logger

# Configure logging for script
setup_logging(level="INFO", log_file="training.log")

# Get logger for module
logger = get_logger(__name__)
logger.info("Starting training")
```

---

## Rate Limiting

### RateLimit

API rate limiting with token bucket algorithm.

**File**: `rate_limit.py` (~300 lines)

```python
from app.utils.rate_limit import (
    RateLimiter,
    rate_limited,
)
```

#### Usage

```python
from app.utils.rate_limit import RateLimiter, rate_limited

# Create limiter (10 requests per second)
limiter = RateLimiter(requests_per_second=10)

# Use in code
async def fetch_data(url: str):
    await limiter.acquire()
    return await http_get(url)

# As decorator
@rate_limited(requests_per_second=5)
async def api_call(endpoint: str):
    ...
```

### LoadThrottle

CPU load-based throttling for batch operations.

**File**: `load_throttle.py` (~250 lines)

```python
from app.utils.load_throttle import (
    LoadThrottle,
    throttle_if_busy,
)
```

#### Usage

```python
from app.utils.load_throttle import LoadThrottle

# Create throttle (max 80% CPU)
throttle = LoadThrottle(max_cpu_percent=80)

# In processing loop
for item in items:
    throttle.wait_if_busy()  # Sleeps if CPU too high
    process(item)
```

---

## Other Utilities

### Assertions

Runtime assertions with descriptive messages.

**File**: `assertions.py` (~250 lines)

```python
from app.utils.assertions import (
    assert_type,
    assert_range,
    assert_not_none,
    assert_file_exists,
)
```

### AsyncUtils

Async helpers including timeouts and gathering.

**File**: `async_utils.py` (~100 lines)

```python
from app.utils.async_utils import (
    run_with_timeout,
    gather_with_errors,
)
```

### ChecksumUtils

File integrity checking.

**File**: `checksum_utils.py` (~200 lines)

```python
from app.utils.checksum_utils import (
    compute_file_hash,
    verify_checksum,
    create_manifest,
)
```

### DatetimeUtils

Date/time formatting and parsing.

**File**: `datetime_utils.py` (~200 lines)

```python
from app.utils.datetime_utils import (
    format_duration,
    parse_timestamp,
    get_timestamp,
)
```

### Result

Rust-style Result type for error handling.

**File**: `result.py` (~250 lines)

```python
from app.utils.result import Result, Ok, Err

def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)

result = divide(10, 2)
if result.is_ok():
    print(result.unwrap())
```

### TimeConstants

Standardized time constants.

**File**: `time_constants.py` (~100 lines)

```python
from app.utils.time_constants import (
    SECONDS_PER_MINUTE,
    SECONDS_PER_HOUR,
    SECONDS_PER_DAY,
    MILLISECONDS_PER_SECOND,
)
```

### MemoryConfig

Memory allocation settings.

**File**: `memory_config.py` (~50 lines)

### Ramdrive

RAM disk utilities for fast temporary storage.

**File**: `ramdrive.py` (~600 lines)

```python
from app.utils.ramdrive import (
    create_ramdrive,
    cleanup_ramdrive,
    RamdriveManager,
)
```

### Secrets

Secure secrets handling.

**File**: `secrets.py` (~250 lines)

```python
from app.utils.secrets import (
    get_secret,
    mask_secret,
)
```

### VictoryType

Victory condition definitions.

**File**: `victory_type.py` (~200 lines)

### OptionalImports

Graceful handling of optional dependencies.

**File**: `optional_imports.py` (~300 lines)

```python
from app.utils.optional_imports import (
    PROMETHEUS_AVAILABLE,
    Counter,
    Gauge,
    TORCH_AVAILABLE,
)
```

### ErrorUtils

Error formatting helpers.

**File**: `error_utils.py` (~30 lines)

### ProgressReporter

Progress reporting for long operations.

**File**: `progress_reporter.py` (~500 lines)

```python
from app.utils.progress_reporter import (
    ProgressReporter,
    progress_bar,
)
```

---

## Configuration

### Environment Variables

| Variable                       | Module         | Description          |
| ------------------------------ | -------------- | -------------------- |
| `RINGRIFT_DEBUG`               | env_config     | Enable debug mode    |
| `RINGRIFT_DATA_DIR`            | paths          | Data directory path  |
| `RINGRIFT_LOG_LEVEL`           | logging_utils  | Log level            |
| `RINGRIFT_SKIP_RESOURCE_CHECK` | resource_guard | Skip resource checks |

---

## Usage Examples

### Complete Selfplay Setup

```python
from app.utils import (
    GameDiscovery,
    get_device,
    env,
)
from app.utils.resource_guard import (
    check_disk_space,
    check_gpu_memory,
    can_proceed,
)
from app.utils.canonical_naming import make_config_key

# Configuration
board_type = "hex8"
num_players = 2
config_key = make_config_key(board_type, num_players)

# Check resources
if not check_disk_space(required_gb=5.0):
    raise RuntimeError("Insufficient disk space")
if not check_gpu_memory(required_gb=8.0):
    raise RuntimeError("Insufficient GPU memory")

# Get device
device = get_device()
print(f"Using device: {device}")

# Discover existing games
discovery = GameDiscovery()
existing = discovery.get_total_games(board_type, num_players)
print(f"Found {existing} existing games for {config_key}")

# Run selfplay
num_games = env.get_int("NUM_GAMES", default=1000)
for i in range(num_games):
    if i % 100 == 0 and not can_proceed():
        print("Resource pressure, pausing...")
        break
    play_game()
```

### Training Data Export

```python
from app.utils.game_discovery import GameDiscovery
from app.utils.resource_guard import check_disk_space
from app.utils.canonical_naming import normalize_board_type
from app.utils.logging_utils import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)

# Normalize input
board_type = normalize_board_type("sq8")  # "square8"

# Check disk space for output
if not check_disk_space(required_gb=10.0):
    logger.error("Insufficient disk space for export")
    exit(1)

# Find all databases for config
discovery = GameDiscovery()
dbs = discovery.find_databases_for_config(board_type, num_players=2)

logger.info(f"Found {len(dbs)} databases")
total_games = sum(db.game_count for db in dbs)
logger.info(f"Total games: {total_games}")

# Export from each database
for db_info in dbs:
    logger.info(f"Exporting from {db_info.path}")
    export_games(db_info.path)
```

---

## Module Reference

| Module                 | Lines | Description                  |
| ---------------------- | ----- | ---------------------------- |
| `resource_guard.py`    | 1500  | Resource checking and limits |
| `game_discovery.py`    | 800   | Database discovery           |
| `debug_utils.py`       | 600   | Parity debugging             |
| `ramdrive.py`          | 600   | RAM disk management          |
| `torch_utils.py`       | 500   | PyTorch utilities            |
| `progress_reporter.py` | 500   | Progress reporting           |
| `env_config.py`        | 300   | Environment configuration    |
| `rate_limit.py`        | 300   | Rate limiting                |
| `optional_imports.py`  | 300   | Optional dependencies        |
| `canonical_naming.py`  | 250   | Board type naming            |
| `load_throttle.py`     | 250   | CPU throttling               |
| `assertions.py`        | 250   | Runtime assertions           |
| `result.py`            | 250   | Result type                  |
| `secrets.py`           | 250   | Secrets handling             |
| `checksum_utils.py`    | 200   | File checksums               |
| `datetime_utils.py`    | 200   | Date/time utilities          |
| `file_utils.py`        | 200   | File I/O                     |
| `paths.py`             | 200   | Path resolution              |
| `json_utils.py`        | 200   | JSON utilities               |
| `logging_utils.py`     | 200   | Logging setup                |
| `victory_type.py`      | 200   | Victory conditions           |

---

## See Also

- `app/caching/README.md` - Caching utilities
- `app/core/README.md` - Core infrastructure
- `app/distributed/README.md` - Cluster utilities
- `app/db/README.md` - Database utilities

---

_Last updated: December 2025_
