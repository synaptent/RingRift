# Developer Guide

This guide covers the key patterns and canonical sources used throughout the RingRift AI service codebase. Following these patterns ensures consistency and makes the codebase easier to maintain.

## Canonical Sources

The codebase uses a "single source of truth" pattern for configuration, logging, and resource management. Always use these canonical modules instead of inline implementations.

### Configuration

**Canonical Source:** `app/config/unified_config.py`

```python
from app.config.unified_config import (
    get_config,           # Get the singleton config
    get_training_threshold,  # Training game threshold
    get_elo_db_path,      # Elo database path
    get_min_elo_improvement,  # Promotion threshold
)

# Access config values
config = get_config()
threshold = config.training.trigger_threshold_games

# Or use convenience functions
threshold = get_training_threshold()
```

**Do NOT:**

- Hardcode values like `MIN_GAMES_FOR_TRAINING = 1000`
- Create duplicate config classes
- Load config from environment variables directly (use the config module)

### Coordination Defaults

**Canonical Source:** `app/config/coordination_defaults.py`

```python
from app.config.coordination_defaults import (
    LockDefaults,
    SyncDefaults,
    HeartbeatDefaults,
)

lock_timeout = LockDefaults.LOCK_TIMEOUT
sync_interval = SyncDefaults.DATA_SYNC_INTERVAL
heartbeat_interval = HeartbeatDefaults.INTERVAL
```

**Do NOT:**

- Hardcode lock timeouts or sync intervals in coordination modules
- Define duplicate coordination defaults in scripts or services

### Logging

**Canonical Sources:**

- Scripts: `scripts/lib/logging_config.py`
- App modules: `app/core/logging_config.py`

```python
# In script entry points
from scripts.lib.logging_config import setup_script_logging
logger = setup_script_logging("my_script")

# In app modules
from app.core.logging_config import get_logger
logger = get_logger(__name__)
```

**Do NOT:**

- Use `logging.basicConfig()` directly
- Create custom log formatters (use `format_style` parameter)

### Resource Checking

**Canonical Source:** `app/utils/resource_guard.py`

```python
from app.utils.resource_guard import (
    check_disk_space,
    check_memory,
    can_proceed,
    wait_for_resources,
    LIMITS,
)

# Before disk operations
if not check_disk_space(required_gb=2.0):
    return

# Check all resources
if not can_proceed():
    wait_for_resources(timeout=300)
```

**Do NOT:**

- Use `shutil.disk_usage()` directly
- Use `psutil.virtual_memory()` directly
- Use `psutil.cpu_percent()` directly
- Hardcode utilization thresholds

### Host Configuration

**Canonical Source:** `config/distributed_hosts.yaml`

All cluster host information should be loaded from this file, not hardcoded:

```python
import yaml
from pathlib import Path

def _load_hosts():
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        return []  # Graceful fallback

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return list(config.get("hosts", {}).keys())
```

**Do NOT:**

- Hardcode IP addresses
- Hardcode hostnames
- Commit cluster-specific configuration

### Board Type Naming

**Canonical Source:** `app/utils/canonical_naming.py`

All board types must use canonical naming throughout the codebase:

| Canonical Value | Common Aliases         | Description                            |
| --------------- | ---------------------- | -------------------------------------- |
| `square8`       | sq8, square_8, 8x8     | 8×8 square board (64 cells)            |
| `square19`      | sq19, square_19, 19x19 | 19×19 square board (361 cells)         |
| `hex8`          | hex_8, smallhex        | Radius-4, diameter-8 hex (61 cells)    |
| `hexagonal`     | hex, hex24, largehex   | Radius-12, diameter-24 hex (469 cells) |

Note: Hex board numeric aliases use "diameter" (2×radius), so `hex8` = radius 4, `hex24` = radius 12.

```python
from app.utils.canonical_naming import (
    normalize_board_type,
    make_config_key,
    parse_config_key,
    normalize_database_filename,
)

# Normalize any board type alias to canonical value
board = normalize_board_type("sq8")  # Returns "square8"

# Create config keys (board_type + num_players)
key = make_config_key("sq8", 2)  # Returns "square8_2p"

# Parse config keys
board, players = parse_config_key("hexagonal_4p")  # Returns ("hexagonal", 4)

# Generate database filenames
filename = normalize_database_filename("hex", 4)  # Returns "selfplay_hexagonal_4p.db"
```

**Do NOT:**

- Use non-canonical names like "sq8" in database records
- Mix naming conventions (e.g., "hex" vs "hexagonal" in same file)
- Hardcode board type strings without normalization

### Game Recording

**Canonical Source:** `app/db/unified_recording.py`

All self-play scripts should use the unified recording interface:

```python
from app.db import UnifiedGameRecorder, RecordingConfig, RecordSource

config = RecordingConfig(
    board_type="sq8",  # Automatically normalized to "square8"
    num_players=2,
    source=RecordSource.SELF_PLAY,
    difficulty=5,
)

with UnifiedGameRecorder(config, initial_state) as recorder:
    for move in game_loop():
        recorder.add_move(move, state_after)
    recorder.finalize(final_state)
```

For one-shot recording of completed games:

```python
from app.db import record_game_unified, RecordingConfig, RecordSource

config = RecordingConfig(
    board_type="hexagonal",
    num_players=4,
    source=RecordSource.TOURNAMENT,
)

record_game_unified(config, initial_state, final_state, moves)
```

**Supported Sources:**

| Source                    | Description                       |
| ------------------------- | --------------------------------- |
| `RecordSource.SELF_PLAY`  | General self-play data collection |
| `RecordSource.SOAK_TEST`  | Long-running soak tests           |
| `RecordSource.CMAES`      | CMA-ES optimization runs          |
| `RecordSource.GAUNTLET`   | Gauntlet evaluation               |
| `RecordSource.TOURNAMENT` | Tournament games                  |
| `RecordSource.TRAINING`   | Training data generation          |

### Migrating Existing Scripts

To migrate a script from direct database recording to unified recording:

**Before (deprecated):**

```python
from app.db.game_replay import GameReplayDB
from app.db.recording import record_completed_game

db = GameReplayDB("data/games/selfplay_sq8_2p.db")  # Inconsistent naming
record_completed_game(db, initial, final, moves, metadata={"board_type": "sq8"})
```

**After (recommended):**

```python
from app.db import record_game_unified, RecordingConfig, RecordSource

config = RecordingConfig(
    board_type="sq8",  # Auto-normalized to "square8"
    num_players=2,
    source=RecordSource.SELF_PLAY,
)

record_game_unified(config, initial, final, moves)
```

**Migration checklist:**

1. Replace `GameReplayDB` imports with `from app.db import record_game_unified, RecordingConfig, RecordSource`
2. Create a `RecordingConfig` with board_type (will be auto-normalized)
3. Replace `record_completed_game()` calls with `record_game_unified()`
4. Remove manual database path construction (auto-generated from config)
5. Remove manual board type normalization (handled by RecordingConfig)

## Directory Structure

```
ai-service/
├── app/                    # Core application code
│   ├── ai/                 # AI implementations (random, heuristic, minimax, mcts, descent)
│   ├── config/             # Configuration modules (unified_config.py)
│   ├── core/               # Core utilities (logging_config.py)
│   ├── distributed/        # P2P and cluster coordination
│   ├── training/           # Training pipeline
│   └── utils/              # Shared utilities (resource_guard.py)
├── scripts/                # CLI tools and daemons
│   ├── unified_ai_loop.py  # Main training orchestrator
│   ├── unified_loop/       # Unified loop submodules
│   └── ...
├── config/                 # Configuration files (gitignored if sensitive)
│   ├── distributed_hosts.yaml       # Cluster hosts (gitignored)
│   ├── distributed_hosts.yaml.example  # Template
│   └── unified_loop.yaml            # Loop configuration
├── models/                 # Neural network checkpoints (gitignored)
├── data/                   # Training data and databases (gitignored)
└── docs/                   # Documentation
```

## Key Patterns

### Graceful Fallback

All config loading should work even if config files are missing:

```python
def _load_config():
    config_path = Path("config/distributed_hosts.yaml")
    if not config_path.exists():
        logger.warning("Config not found, using defaults")
        return {}  # Return empty, don't crash

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Config load failed: {e}")
        return {}
```

### Import Guards

For optional dependencies, use import guards:

```python
try:
    from scripts.lib.logging_config import setup_script_logging
    HAS_LOGGING_CONFIG = True
except ImportError:
    HAS_LOGGING_CONFIG = False
    setup_script_logging = None

# Later in code
if HAS_LOGGING_CONFIG:
    logger = setup_script_logging("my_script")
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
```

### Resource Limits

The codebase enforces 80% maximum utilization to prevent system overload:

| Resource | Warning | Critical |
| -------- | ------- | -------- |
| CPU      | 70%     | 80%      |
| GPU      | 70%     | 80%      |
| Memory   | 80%     | 90%      |
| Disk     | 75%     | 80%      |

These are defined in `app/utils/resource_guard.py`.

## AI Factory Pattern

**Canonical Source:** `app/ai/factory.py`

New code should use the AI factory for consistent difficulty profiles across the codebase:

```python
from app.ai.factory import AIFactory, CANONICAL_DIFFICULTY_PROFILES

# Create AI by difficulty (1-10 scale)
ai = AIFactory.create_from_difficulty(
    difficulty=5,
    board_type="square8",
    num_players=2
)

# Get difficulty profile for inspection
profile = CANONICAL_DIFFICULTY_PROFILES[5]
print(f"AI type: {profile['ai_type']}, simulations: {profile.get('simulations', 'N/A')}")
```

**Difficulty Scale:**

| Level | AI Type     | Description                            |
| ----- | ----------- | -------------------------------------- |
| 1-2   | Random      | Random move selection                  |
| 3-4   | Heuristic   | Simple evaluation functions            |
| 5-6   | Minimax     | Tree search with limited depth         |
| 7-8   | MCTS        | Monte Carlo Tree Search                |
| 9-10  | Neural MCTS | Neural network guided MCTS             |
| 11    | NNUE        | Efficiently updatable neural network   |
| 12    | EBMO        | Energy-based move optimization         |
| 13    | GMO         | Gradient move optimization             |
| 14    | IG-GMO      | Information-gain GMO (experimental)    |
| 15    | GPU Minimax | GPU-accelerated minimax (experimental) |

> **Note:** Tiers 12-15 are experimental. See `docs/EXPERIMENTAL_AI.md` for details.

**Current Status:**

Many legacy scripts (42+) still create AI instances directly. This is acceptable for existing code but new scripts should use the factory pattern for:

- Consistent difficulty calibration
- Centralized parameter management
- Easier difficulty adjustments across the codebase

**Do NOT in new code:**

- Instantiate AI classes directly without good reason
- Hardcode AI parameters like simulation counts
- Define custom difficulty scales

## Training Pipeline

The training system uses a unified loop with event-driven components:

1. **Data Collection** — Streams games from cluster nodes
2. **Training Trigger** — Fires when `trigger_threshold_games` is reached
3. **Model Training** — Neural network training with checkpointing
4. **Evaluation** — Shadow tournaments to measure Elo
5. **Promotion** — Auto-promote models that improve by `elo_threshold`

All thresholds come from `unified_config.py`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_resource_guard.py -v
```

## Common Tasks

### Adding a New Script

1. Use `setup_logging()` for logging
2. Load hosts from `distributed_hosts.yaml`
3. Use `get_training_threshold()` for training thresholds
4. Use `resource_guard` for resource checks
5. Add graceful fallbacks for missing config

### Modifying Configuration

1. Add new fields to `app/config/unified_config.py`
2. Update `config/unified_loop.yaml.example` if needed
3. Add convenience getter if commonly used
4. Document the new field

### Adding Training Features

1. Add config options to `TrainingConfig` in `unified_config.py`
2. Implement in `app/training/`
3. Integrate with `unified_ai_loop.py`
4. Add tests

## Sync System Architecture

The codebase has canonical sync scripts for different data domains. Always use these instead of deprecated alternatives.

### Canonical Sync Scripts

| Domain       | Canonical Script              | Description                                                                                                       |
| ------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Game Data    | `unified_data_sync.py`        | Rsync-based game database sync with P2P fallback                                                                  |
| Models       | `sync_models.py`              | Hash-based model distribution with deduplication; use `--use-sync-coordinator` for aria2/SSH/P2P + NFS-aware sync |
| ELO Ratings  | `elo_db_sync.py`              | ELO database synchronization                                                                                      |
| Coordination | `cluster_sync_coordinator.py` | Meta-orchestrator for all sync operations                                                                         |
| Vast.ai P2P  | `vast_p2p_sync.py`            | Specialized P2P sync for Vast.ai nodes                                                                            |

### Deprecated Sync Scripts

The following scripts are deprecated and will be removed in a future release:

| Deprecated            | Replacement            | Migration                                                     |
| --------------------- | ---------------------- | ------------------------------------------------------------- |
| `simple_game_sync.py` | `unified_data_sync.py` | `python scripts/unified_data_sync.py --once`                  |
| `model_sync_aria2.py` | `sync_models.py`       | `python scripts/sync_models.py --sync --use-sync-coordinator` |

These deprecated scripts emit `DeprecationWarning` on import.

## Evaluation System Architecture

The codebase has two main evaluation patterns: **Gauntlet** (O(n) baseline testing) and **Tournament** (round-robin or tier-based).

### Canonical Evaluation Scripts

| Type          | Canonical Script         | Description                             |
| ------------- | ------------------------ | --------------------------------------- |
| Gauntlet      | `run_gauntlet.py`        | O(n) testing against fixed baselines    |
| Tournament    | `run_tournament.py`      | Unified tournament dispatcher (8 modes) |
| Monitoring    | `auto_elo_tournament.py` | Continuous Elo monitoring daemon        |
| Data Pipeline | `gauntlet_to_elo.py`     | Convert gauntlet results to Elo ratings |

### Tournament Modes (via run_tournament.py)

```bash
python scripts/run_tournament.py --mode <mode>
```

| Mode          | Description                         |
| ------------- | ----------------------------------- |
| `basic`       | Basic AI vs AI tournament           |
| `models`      | Neural network model Elo tournament |
| `distributed` | Multi-tier difficulty ladder        |
| `ssh`         | SSH-based multi-host distribution   |
| `eval`        | Evaluation pool tournaments         |
| `diverse`     | All board/player configurations     |
| `weights`     | Heuristic weight profile testing    |
| `crossboard`  | Cross-board difficulty analysis     |

### Deprecated Evaluation Scripts

| Deprecated             | Replacement       | Migration                                       |
| ---------------------- | ----------------- | ----------------------------------------------- |
| `baseline_gauntlet.py` | `run_gauntlet.py` | `python scripts/run_gauntlet.py --local`        |
| `run_vast_gauntlet.py` | `run_gauntlet.py` | `python scripts/run_gauntlet.py --parallel 128` |

These deprecated scripts emit `DeprecationWarning` on import.

## Migration Status

### Unified Logging (scripts use scripts.lib)

All scripts now use the unified logging pattern from `scripts.lib.logging_config`. App modules continue to use `app.core.logging_config`. The migration was completed in December 2025.

**Pattern used:**

```python
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("script_name")
```

**Scripts with intentional basicConfig:**

Only 3 scripts intentionally use `logging.basicConfig()` in their `main()` function for `--verbose` flag support. These configure logging at runtime based on user arguments, which is the correct pattern for CLI tools:

- `vast_operations.py`
- `vast_instance_manager.py`
- `health_alerting.py`

### Config Integration

The `scripts/unified_loop/config.py` module now includes integration functions:

```python
from scripts.unified_loop.config import (
    sync_with_unified_config,      # Sync defaults from app.config.unified_config
    get_canonical_training_threshold,  # Get threshold from canonical source
)

# Ensure config uses canonical values
config = UnifiedLoopConfig.from_yaml(config_path)
config = sync_with_unified_config(config)
```

## Performance Optimization

### Profiling Tools

| Script                            | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `profile_selfplay_bottlenecks.py` | Detailed timing breakdown for self-play        |
| `quick_benchmark.py`              | Quick benchmark with all optimizations enabled |
| `benchmark_make_unmake.py`        | State mutation performance                     |
| `benchmark_gpu_cpu.py`            | GPU vs CPU comparison                          |
| `benchmark_ai_memory.py`          | Memory usage analysis                          |

### Running Profilers

```bash
# Detailed self-play profiling
python scripts/profile_selfplay_bottlenecks.py --board square8 --games 5

# With cProfile function-level analysis
python scripts/profile_selfplay_bottlenecks.py --board square8 --games 3 --cprofile

# Quick benchmark with optimizations
python scripts/quick_benchmark.py
```

### Performance Environment Variables

| Variable                         | Effect                                     | Default |
| -------------------------------- | ------------------------------------------ | ------- |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | Skip validation deep-copies (2-3x speedup) | `false` |
| `RINGRIFT_USE_MAKE_UNMAKE`       | Use incremental state updates              | `false` |
| `RINGRIFT_USE_BATCH_EVAL`        | Batch position evaluation                  | `false` |
| `RINGRIFT_USE_FAST_TERRITORY`    | Fast territory calculation                 | `false` |
| `RINGRIFT_USE_MOVE_CACHE`        | Cache legal moves                          | `false` |

**Recommended for training/benchmarking:**

```bash
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
```

This provides 2-3x speedup with no accuracy impact.

### Current Performance Baseline

On Apple M2 Max (square8, 2 players, HeuristicAI difficulty 5):

| Metric          | Without Skip | With Skip |
| --------------- | ------------ | --------- |
| select_move avg | ~510ms       | ~210ms    |
| apply_move avg  | ~11ms        | ~8ms      |
| Moves/second    | ~1.9         | ~4.5      |

### Optimization Recommendations

1. **Shadow Contract Skip** (Implemented) - 2-3x speedup
2. **Batch Evaluation** - Vectorized NumPy operations
3. **Numba JIT** - Compile hot loops to machine code
4. **Move Caching** - LRU cache for legal moves
5. **State Pooling** - Reuse GameState objects
