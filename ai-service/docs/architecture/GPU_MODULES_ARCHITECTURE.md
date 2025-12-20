# GPU Modules Architecture

This document describes the GPU-accelerated game simulation modules in `app/ai/`. These modules enable parallel game simulation for training data generation and AI search.

## Module Overview

The GPU modules were refactored in December 2025 (R5-R22) to improve maintainability and testability. The monolithic `gpu_parallel_games.py` was split into focused, single-responsibility modules:

```
app/ai/
├── gpu_parallel_games.py     (2,265 lines) - Main orchestration & ParallelGameRunner
├── gpu_move_generation.py    (1,640 lines) - Move generation for all game phases
├── gpu_move_application.py   (1,260 lines) - Move application & state updates
├── gpu_batch_state.py          (787 lines) - BatchGameState data structure
├── gpu_heuristic.py            (538 lines) - Position evaluation
├── gpu_territory.py            (496 lines) - Territory detection algorithms
├── gpu_line_detection.py       (416 lines) - Line-of-5+ detection
├── gpu_selection.py            (355 lines) - Move selection utilities
└── gpu_game_types.py           (141 lines) - Shared enums & types
```

## Module Dependency Graph

```
                    ┌─────────────────────────┐
                    │   gpu_parallel_games    │
                    │  (ParallelGameRunner)   │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ gpu_selection │     │ gpu_batch_state │     │gpu_move_generation│
│ (selection)   │     │ (BatchGameState)│     │   (BatchMoves)   │
└───────┬───────┘     └────────┬────────┘     └────────┬─────────┘
        │                      │                       │
        └──────────────────────┼───────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│gpu_move_apply │    │  gpu_heuristic  │    │gpu_line_detection│
│(apply moves)  │    │  (evaluation)   │    │  (line of 5+)   │
└───────────────┘    └────────┬────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  gpu_territory  │
                    │  (territory)    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ gpu_game_types  │
                    │ (enums/types)   │
                    └─────────────────┘
```

## Module Descriptions

### `gpu_game_types.py` (Foundation)

Shared type definitions used across all GPU modules:

- `GamePhase` enum: RING_PLACEMENT, MOVEMENT, LINE_PROCESSING, TERRITORY_PROCESSING, END_TURN
- `GameStatus` enum: ACTIVE, COMPLETED, DRAW, MAX_MOVES
- `MoveType` enum: PLACEMENT, MOVEMENT, CAPTURE, LINE_FORMATION, TERRITORY_CLAIM, SKIP, NO_ACTION, RECOVERY_SLIDE
- `DetectedLine` dataclass: Line detection results
- Utility functions: `get_int_dtype()`, `get_required_line_length()`

### `gpu_batch_state.py` (State Management)

The `BatchGameState` dataclass holds all game state tensors:

**Board Tensors** (batch_size, board_size, board_size):

- `stack_owner` - Ring stack ownership (0=empty, 1-4=player)
- `stack_height` - Total rings in stack (0-5)
- `cap_height` - Consecutive top rings of owner's color
- `marker_owner` - Marker placement
- `territory_owner` - Territory control
- `is_collapsed` - Collapsed cells

**Player Tensors** (batch_size, num_players+1):

- `rings_in_hand` - Available rings per player
- `territory_count` - Territory cells controlled
- `is_eliminated` - Player elimination status
- `eliminated_rings` - Rings lost by this player
- `buried_rings` - Rings buried in stacks (captured but not removed)
- `rings_caused_eliminated` - Rings eliminated by this player

**Factory Methods**:

- `create_batch()` - Create fresh game states
- `from_single_game()` - Convert CPU GameState to GPU
- `from_game_states()` - Batch convert multiple GameStates
- `to_game_state()` - Convert back to CPU GameState

### `gpu_move_generation.py` (Move Generation)

Generates legal moves for all game phases:

- `generate_all_moves_vectorized()` - Main entry point, phase-aware
- `generate_placement_moves_optimized()` - Ring placement moves
- `generate_movement_moves_vectorized()` - Stack movement moves
- `generate_capture_moves_vectorized()` - Capture moves
- `generate_recovery_moves()` - Recovery phase moves

Returns `BatchMoves` dataclass with flattened move tensors:

- `game_idx` - Which game each move belongs to
- `move_type` - Type of move
- `from_y`, `from_x` - Source position
- `to_y`, `to_x` - Destination position
- `moves_per_game` - Move count per game
- `move_offsets` - Index offset per game

### `gpu_move_application.py` (Move Execution)

Applies moves to update game state:

- `apply_no_action_moves_batch()` - Handle NO_ACTION/PASS
- `apply_placement_moves_batch()` - Place rings on board
- `apply_movement_moves_batch()` - Move stacks
- `apply_capture_moves_batch()` - Capture opponent stacks
- `apply_recovery_moves_vectorized()` - Recovery phase moves

Each function updates the relevant state tensors in-place.

### `gpu_selection.py` (Move Selection)

Stochastic move selection for training:

- `select_moves_vectorized()` - Fast random sampling with center bias
- `select_moves_heuristic()` - Feature-based scoring for better quality

Both use segment-wise softmax sampling with temperature control.

### `gpu_heuristic.py` (Position Evaluation)

Evaluates positions for heuristic-based AI:

- `evaluate_positions_batch()` - Main evaluation function
- Configurable feature weights (45+ parameters)
- Features: territory, center control, lines, threats, recovery

### `gpu_line_detection.py` (Line Detection)

Detects lines of 5+ consecutive stacks:

- `detect_lines_vectorized()` - Find horizontal, vertical, diagonal lines
- Used for victory detection and territory scoring

### `gpu_territory.py` (Territory Detection)

Detects and claims territory from collapsed cells:

- `detect_territory_batch()` - Find enclosed regions
- `claim_territory_batch()` - Assign territory to players
- Flood-fill algorithm on GPU

### `gpu_parallel_games.py` (Orchestration)

Main entry point containing `ParallelGameRunner`:

- `play_batch()` - Run games to completion
- `step_batch()` - Single step across all games
- Phase transition logic
- Victory detection
- Move history tracking

## Test Coverage

Each module has dedicated unit tests:

| Module               | Test File                    | Test Count |
| -------------------- | ---------------------------- | ---------- |
| gpu_move_application | test_gpu_move_application.py | 35         |
| gpu_heuristic        | test_gpu_heuristic.py        | 35         |
| gpu_selection        | test_gpu_selection.py        | 27         |
| gpu_batch_state      | test_gpu_batch_state.py      | 38         |
| gpu_line_detection   | test_gpu_line_detection.py   | 20+        |
| gpu_territory        | test_gpu_territory.py        | 15+        |
| gpu_move_generation  | test_gpu_move_generation.py  | 30+        |

## Usage Example

```python
from app.ai.gpu_batch_state import BatchGameState
from app.ai.gpu_parallel_games import ParallelGameRunner

# Create batch of games
state = BatchGameState.create_batch(
    batch_size=1000,
    board_size=8,
    num_players=2,
)

# Run games to completion
runner = ParallelGameRunner(
    batch_size=1000,
    board_size=8,
    num_players=2,
)
results = runner.play_batch(max_moves=500)

# Extract training data
for g in range(1000):
    moves = state.extract_move_history(g)
    victory_type, _ = state.derive_victory_type(g, max_moves=500)
```

## Performance Characteristics

- **Vectorized Operations**: All computations use torch tensor ops
- **No CPU-GPU Sync**: Avoids `.item()` calls in hot paths
- **Batch Processing**: Processes all games in parallel
- **Memory Efficient**: Uses int8/int16 for game state tensors

Typical throughput on M1 Mac with MPS: ~50,000 moves/second
Typical throughput on A100 GPU: ~500,000 moves/second

## Related Documentation

- [Architecture Overview](ARCHITECTURE_OVERVIEW.md) - Overall system design
- [MPS Architecture](MPS_ARCHITECTURE.md) - Apple Silicon GPU details
- [Training Pipeline](../training/TRAINING_PIPELINE.md) - How GPU games feed training
