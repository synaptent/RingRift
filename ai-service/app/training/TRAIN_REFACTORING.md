# train.py Refactoring Strategy

## Current State (December 2025)

- **File size**: 5,033 lines
- **`train_model()` function**: Lines 705-4955 (4,250 lines)
- **Parameters**: 96 function parameters
- **Cyclomatic Complexity**: 631 (target: <50)

## Recommended Decomposition

### Phase 1: Extract Configuration Objects

Replace 96 individual parameters with structured configs:

```python
@dataclass
class TrainingDataConfig:
    data_path: str | list[str]
    validate_data: bool = True
    fail_on_invalid_data: bool = False
    skip_freshness_check: bool = False
    max_data_age_hours: float = 1.0
    ...

@dataclass
class DistributedConfig:
    distributed: bool = False
    local_rank: int = -1
    scale_lr: bool = False
    lr_scale_mode: str = 'linear'
    ...

@dataclass
class CheckpointConfig:
    save_path: str
    checkpoint_dir: str = 'checkpoints'
    checkpoint_interval: int = 5
    resume_path: str | None = None
    ...
```

### Phase 2: Extract Logical Functions

| Function                    | Lines      | Purpose                      |
| --------------------------- | ---------- | ---------------------------- |
| `_setup_distributed()`      | ~870-890   | DDP initialization           |
| `_validate_training_data()` | ~890-1020  | Freshness + validation       |
| `_setup_enhancements()`     | ~1100-1250 | Hot buffer, curriculum       |
| `_create_model()`           | ~1845-2100 | Model instantiation          |
| `_setup_training()`         | ~2161-2500 | Optimizer, scheduler         |
| `_setup_fault_tolerance()`  | ~3100-3220 | Circuit breaker, checkpoints |
| `_run_training_loop()`      | ~3340-4700 | Main epoch loop              |
| `_save_final_results()`     | ~4700-4955 | Cleanup, best model          |

### Phase 3: Simplify Main Function

```python
def train_model(
    config: TrainConfig,
    data_config: TrainingDataConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    enhancement_config: EnhancementConfig | None = None,
) -> TrainingResult:
    """Train the RingRift neural network model."""

    # Setup
    _setup_distributed(distributed_config)
    _validate_training_data(data_config)
    data_loader = _create_data_loader(data_config)
    model = _create_model(config)
    optimizer, scheduler = _setup_training(model, config)

    # Train
    result = _run_training_loop(
        model, data_loader, optimizer, scheduler,
        checkpoint_config, enhancement_config,
    )

    # Finalize
    _save_final_results(model, result, checkpoint_config)
    return result
```

## Priority

This refactoring is **P2 (Medium)** priority:

- The code works correctly
- It's well-documented with section headers
- Complexity makes debugging harder but doesn't affect correctness

**Estimated effort**: 3-4 sessions
**Risk**: Medium (touching core training logic)

## Backwards Compatibility

Maintain `train_model()` signature for CLI/external callers by creating
a wrapper that converts old-style params to new config objects.

## Completed Work (December 2025)

### Phase 1 - COMPLETE

Configuration objects extracted to `train_config.py`:

- `TrainingDataConfig` - data loading and validation
- `DistributedConfig` - DDP settings
- `CheckpointConfig` - model checkpointing
- `LearningRateConfig` - LR scheduling
- `EnhancementConfig` - curriculum, hot buffer, etc.
- `FaultToleranceConfig` - circuit breaker, anomaly detection
- `ModelArchConfig` - architecture settings
- `EarlyStoppingConfig` - early stopping criteria
- `MixedPrecisionConfig` - AMP settings
- `AugmentationConfig` - data augmentation
- `HeartbeatConfig` - monitoring
- `FullTrainingConfig` - combines all above
- `config_from_legacy_params()` - backwards compatibility helper

Validation utilities extracted to `train_validation.py`:

- `validate_training_data_freshness()` - data age checking
- `validate_training_data_files()` - file integrity validation
- `validate_data_checksums()` - checksum verification
- `FreshnessResult`, `ValidationResult` - result dataclasses

All modules integrated into train.py imports and exported from `__init__.py`.

### Phase 2 - TODO (Q1 2026)

Function extraction requires careful incremental migration.

### Other Refactoring (Q1 2026)

**NeuralNetAI Migration** (`app/ai/_neural_net_legacy.py`):

- 3,000+ lines in legacy file
- Complex with multiple AI classes
- Requires careful testing
- Scheduled for Q1 2026
