# GPU/CPU Game Engine Phase Differences

## Overview

The GPU parallel game engine (`app/ai/gpu_parallel_games.py`) and the CPU game engine
(`app/game_engine.py`) have different phase handling semantics that make direct move-by-move
replay incompatible. This document describes the differences and proposed solutions.

## Key Differences

### 1. Recovery Timing

**GPU Engine:**

- After a capture that buries opponent's rings, the opponent immediately performs recovery
- Recovery is recorded as the opponent's move BEFORE their normal turn
- Example sequence: P1 capture → P2 recovery_slide → P2 place_ring → P2 move_stack

**CPU Engine:**

- Recovery is handled as part of the player's normal MOVEMENT phase
- The player goes through RING_PLACEMENT first, then can do recovery in MOVEMENT
- Example sequence: P1 capture → P2 (skip_placement) → P2 recovery_slide

### 2. Implicit From Positions

**GPU Engine:**

- Always records explicit `from` positions for movements/captures
- After placement, records movement with `from = placed_position`

**CPU Engine:**

- After placement, movement moves have implicit `from` (set via `must_move_from_stack_key`)
- The `from_pos` field on Move objects is `None` for movements after placement

### 3. Chain Capture Representation

**GPU Engine:**

- Records all chain capture segments as `overtaking_capture`
- Does not use `continue_capture_segment` move type

**CPU Engine:**

- First capture in chain is `overtaking_capture`
- Subsequent captures in chain are `continue_capture_segment`

## Fixes Applied

### Pydantic v1/v2 Compatibility (app/models/core.py)

Added compatibility layer to support both Pydantic v1 (installed) and v2 (used in code):

- `model_copy()` → `copy()`
- `model_dump()` → `dict()`
- `model_validate()` → `parse_obj()`
- `model_construct()` → `construct()`
- Added `allow_population_by_field_name = True` for alias handling

### Import Script Improvements (scripts/import_gpu_selfplay_to_db.py)

1. **Fixed from_pos matching**: When CPU candidate moves have `from_pos=None` (implicit),
   the matching logic now accepts GPU moves with explicit from positions if they match
   the `must_move_from_stack_key`.

2. **Copy from_pos to matched moves**: When applying matched moves, copy the GPU move's
   `from_pos` to the matched move if the matched move has `None`.

## Remaining Issues

1. **Recovery phase timing**: GPU records recovery before the player's turn, but CPU
   expects it during the player's MOVEMENT phase. The import script tries to advance
   through phases but the timing differs.

2. **Player tracking desync**: Due to recovery timing differences, player tracking can
   become desynced between GPU recording and CPU replay.

## Proposed Solutions

### Option 1: Fix GPU Recording (Recommended)

Modify GPU selfplay to record moves in CPU-compatible order:

1. Don't record recovery as a separate turn
2. Record recovery during the victim's normal MOVEMENT phase
3. Use `continue_capture_segment` for chain captures after first

### Option 2: Direct GPU → Training Path

Create a training path that doesn't require CPU validation:

1. GPU reconstructs positions during selfplay
2. Positions are saved directly to training-ready format (NPZ)
3. Bypass the JSONL → SQLite → NNUE pipeline

### Option 3: Relaxed Validation

Accept that GPU and CPU have different phase semantics:

1. Use GPU JSONL for training without full CPU validation
2. Only validate final outcomes match
3. Use shadow validation at sampling rate for parity checking

## Current Status

- GPU selfplay is generating games successfully on all GH200 hosts
- NNUE training is running on H100 using existing validated data
- New GPU data cannot be imported to canonical SQLite format yet
- The GPU data can still be used for JSONL-based training pipelines
