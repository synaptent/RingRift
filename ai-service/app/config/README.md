# Configuration Module

Unified configuration management for the RingRift AI service.

## Overview

This module provides a single source of truth for all configuration:

- Loading from YAML/JSON files
- Environment variable overrides
- Dataclass conversion with validation
- Threshold constants for training/evaluation

## Key Components

### `unified_config.py` - Main Configuration

The central configuration dataclass loaded from `config/unified_loop.yaml`:

```python
from app.config.unified_config import get_config

config = get_config()

# Access training settings
threshold = config.training.trigger_threshold_games

# Access all board configs
for board in config.get_all_board_configs():
    print(f"{board.board_type}_{board.num_players}p")
```

### `loader.py` - Configuration Loader

Flexible config loading with type conversion:

```python
from app.config.loader import load_config
from dataclasses import dataclass

@dataclass
class MyConfig:
    host: str = "localhost"
    port: int = 8080

# Load with auto-format detection
config = load_config("config/app.yaml", target=MyConfig)

# Environment overrides (MYAPP_HOST, MYAPP_PORT)
config = load_config("config/app.yaml", env_prefix="MYAPP_")
```

### `config_validator.py` - Validation

Validate configuration files at startup:

```python
from app.config.config_validator import validate_all_configs

result = validate_all_configs()
if not result.valid:
    for error in result.errors:
        print(f"ERROR: {error}")
```

### `thresholds.py` - Constants

Threshold values for training, evaluation, and promotion:

```python
from app.config.thresholds import (
    TRAINING_TRIGGER_GAMES,      # Min games to trigger training
    ELO_DROP_ROLLBACK,           # Elo drop to trigger rollback
    ELO_IMPROVEMENT_PROMOTE,     # Elo gain to promote model
)
```

### `training_config.py` - Training Config

Dataclasses for training configuration:

```python
from app.config.training_config import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
)
```

### `ladder_config.py` - AI Ladder

Configuration for AI difficulty ladder:

```python
from app.config.ladder_config import get_ladder_config

ladder = get_ladder_config("square8", num_players=2)
for level in ladder.levels:
    print(f"Level {level.difficulty}: {level.name}")
```

## Configuration Files

| File                            | Purpose                         |
| ------------------------------- | ------------------------------- |
| `config/unified_loop.yaml`      | Main training loop config       |
| `config/distributed_hosts.yaml` | Cluster node definitions (SSoT) |
| `config/hyperparameters.json`   | Neural network hyperparameters  |
| `config/resource_limits.yaml`   | Resource constraints            |

## Environment Variables

Key overrides:

- `RINGRIFT_CONFIG_PATH`: Override config file path
- `RINGRIFT_TRAINING_THRESHOLD`: Override training trigger
- `RINGRIFT_ELO_DB`: Override Elo database path
- `RINGRIFT_P2P_*`: P2P configuration overrides

## Validation

The `config_validator` validates:

- Required sections exist
- Values are in valid ranges
- Cross-field consistency

Warnings vs Errors:

- **Errors**: Missing required fields, invalid values
- **Warnings**: Suboptimal values (e.g., low poll intervals)
