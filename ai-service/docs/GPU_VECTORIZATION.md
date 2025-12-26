# GPU Vectorization Architecture

## Overview

The `app/ai/gpu_*.py` modules provide GPU-accelerated parallel game simulation for:

- CMA-ES fitness evaluation (10-100+ games per candidate in parallel)
- Selfplay data generation (10x faster than CPU)
- Tournament evaluation

## Module Structure

| Module                    | Responsibility                       |
| ------------------------- | ------------------------------------ |
| `gpu_parallel_games.py`   | Main entry point, ParallelGameRunner |
| `gpu_batch_state.py`      | BatchGameState tensor management     |
| `gpu_move_generation.py`  | Vectorized move generation           |
| `gpu_move_application.py` | Vectorized move application          |
| `gpu_line_detection.py`   | Line/five detection                  |
| `gpu_territory.py`        | Territory computation                |
| `gpu_heuristic.py`        | Position evaluation                  |
| `gpu_game_types.py`       | Enums and type utilities             |
| `gpu_selection.py`        | Move selection strategies            |

## Performance Characteristics

### CUDA (Recommended)

- 10-100 games/sec on RTX 3090
- 50-500 games/sec on A100

### CPU (Fallback)

- Uses vectorized numpy operations
- ~1-5 games/sec depending on complexity

### MPS (Apple Silicon) - NOT RECOMMENDED

MPS is currently ~100x SLOWER than CPU due to excessive CPU-GPU synchronization
from `.item()` calls in the game loop. For Apple Silicon, use `device="cpu"`.

Optimization would require eliminating ~80 `.item()` calls and fully vectorizing
all conditional logic - significant refactoring effort.

## Known Limitations

### 1. Recovery Move Gate (Incomplete)

**Location**: `gpu_move_generation.py:1391`

The GPU implementation does not enforce the full fallback-class gate for recovery moves:

- CPU: "fallback recovery only if NO line-forming recovery exists anywhere on board"
- GPU: Surfaces both empty-cell and stack-strike options whenever player is recovery-eligible

**Impact**: May generate slightly more recovery moves than strictly legal.
**Workaround**: Use shadow validation for correctness-critical applications.

### 2. Chain Capture Handling

Chain captures (multiple consecutive captures in one turn) are handled sequentially
rather than in a fully vectorized manner. Each chain step requires:

1. Apply capture
2. Check for further captures
3. Repeat

This introduces some CPU-GPU synchronization overhead.

### 3. Phase Transitions

Phase transitions (RING_PLACEMENT -> MOVEMENT, etc.) are checked per-game rather
than in a fully batched manner, as different games may be in different phases.

### 4. from_game_states() Conversion

The `BatchGameState.from_game_states()` method for converting CPU GameState objects
to GPU batch format has been updated (2025-12) to use the dictionary-based board model.
Previous implementations using `CellContent` are deprecated.

## Validation Modes

### Shadow Validation

For correctness-critical applications, enable shadow validation to compare GPU moves
against the CPU rules engine:

```python
from app.ai.shadow_validation import create_shadow_validator

validator = create_shadow_validator(enabled=True)
runner = ParallelGameRunner(
    batch_size=64,
    device="cuda",
    shadow_validator=validator,
)
```

### Parity Testing

Tests in `tests/gpu/` verify GPU-CPU parity for:

- Line detection (`test_gpu_line_detection.py`)
- Move generation (`test_gpu_move_generation.py`)
- Heuristic evaluation (`test_gpu_heuristic.py`)

**Large-scale parity validation (2025-12-23):**

| Test          | Seeds  | Passed | Rate | Status           |
| ------------- | ------ | ------ | ---- | ---------------- |
| 10K seed test | 10,000 | 10,000 | 100% | Production-ready |

GPU self-play is confirmed safe for NN training data generation with full parity.

## Configuration

Environment variables:

- `RINGRIFT_PARITY_VALIDATION=off|warn|strict` - Enable parity checking
- `RINGRIFT_PARITY_DUMP_DIR=<path>` - Directory for parity failure dumps

## Future Improvements

1. **Eliminate .item() calls**: Required for MPS performance
2. **Full recovery gate**: Implement complete fallback-class logic
3. **Batched chain captures**: Vectorize multi-step capture sequences
4. **Memory optimization**: Reduce tensor allocations in hot paths
