# GPU Architecture Simplification Plan

> **Document Status:** Active Planning Document
> **Last Updated:** 2025-12-11
> **Author:** AI Assistant

## Executive Summary

This document analyzes the current GPU codebase and provides a roadmap for architectural simplification. The analysis reveals significant opportunities to reduce complexity while improving maintainability and performance.

**Key Findings:**

- 35% of GPU code is dead/unused (cuda_rules.py: 3,613 lines)
- GPU evaluation only provides speedup for batch sizes ≥ 500
- Significant code duplication exists across GPU files
- CPU heuristic evaluation is highly optimized (21,000+ evals/sec)

---

## Table of Contents

1. [Benchmark Results](#1-benchmark-results)
2. [Architecture Analysis](#2-architecture-analysis)
3. [Dead Code Inventory](#3-dead-code-inventory)
4. [Simplification Roadmap](#4-simplification-roadmap)
5. [Decision Matrix](#5-decision-matrix)
6. [Implementation Plan](#6-implementation-plan)

---

## 1. Benchmark Results

### 1.1 CPU vs Hybrid GPU Evaluation (NVIDIA A10 CUDA)

Benchmark run on 2025-12-11 with PyTorch on Lambda Labs NVIDIA A10 GPU (23GB VRAM).

| Batch Size | CPU (ms) | Hybrid CUDA (ms) | Speedup   | Winner |
| ---------- | -------- | ---------------- | --------- | ------ |
| 10         | 1.26     | 0.81             | 1.55x     | GPU    |
| 50         | 6.40     | 1.50             | **4.28x** | GPU    |
| 100        | 12.45    | 2.49             | **4.99x** | GPU    |
| 200        | 29.36    | 4.70             | **6.24x** | GPU    |
| 500        | 68.78    | 10.49            | **6.56x** | GPU    |
| 1000       | 135.73   | 31.48            | 4.31x     | GPU    |

**Phase 1 Gate Status: ✅ PASSED** (>= 3x speedup achieved at batch 50+)

### 1.2 CPU vs Hybrid GPU Evaluation (Apple Silicon MPS)

Benchmark run on 2025-12-11 with PyTorch 2.6.0 on Apple Silicon (MPS).

| Batch Size | CPU (ms) | Hybrid MPS (ms) | Speedup | Winner |
| ---------- | -------- | --------------- | ------- | ------ |
| 10         | 0.50     | 4.38            | 0.11x   | CPU    |
| 50         | 2.27     | 9.62            | 0.24x   | CPU    |
| 100        | 4.60     | 9.13            | 0.50x   | CPU    |
| 200        | 9.25     | 10.04           | 0.92x   | ~Equal |
| 500        | 22.98    | 13.11           | 1.75x   | GPU    |

**Note:** MPS has higher overhead than CUDA, only beneficial for batch >= 500.

### 1.3 Key Insights

1. **CUDA break-even**: ~10 positions (GPU faster even for small batches)
2. **MPS break-even**: ~200-250 positions (significant overhead)
3. **Peak speedup**: 6.56x on CUDA at batch 500
4. **Sweet spot**: Batch 200-500 for best speedup-to-overhead ratio
5. **CPU throughput**: ~21,000 evals/sec on M-series, slower on Lambda (x86_64)

### 1.4 Implications for Use Cases

| Use Case                    | Typical Batch Size | CUDA       | MPS        | CPU |
| --------------------------- | ------------------ | ---------- | ---------- | --- |
| Single-move evaluation      | 1-50               | GPU (1.5x) | CPU        | CPU |
| Move ranking (AI selection) | 50-200             | GPU (4-6x) | CPU        | CPU |
| Batch selfplay (CMA-ES)     | 500-2000           | GPU (6x+)  | GPU (1.7x) | -   |
| Neural network training     | 1000+              | GPU        | GPU        | -   |

---

## 2. Architecture Analysis

### 2.1 Current File Inventory

| File                    | Lines      | Status    | Dependencies           | Purpose                |
| ----------------------- | ---------- | --------- | ---------------------- | ---------------------- |
| `gpu_parallel_games.py` | 2,821      | ACTIVE    | Core batch game engine | CMA-ES, batch selfplay |
| `gpu_batch.py`          | 917        | ACTIVE    | Heuristic evaluation   | GPU scoring            |
| `hybrid_gpu.py`         | 822        | ACTIVE    | CPU rules + GPU eval   | Safe hybrid path       |
| `numba_rules.py`        | 973        | ACTIVE    | CPU JIT rules          | Fast CPU fallback      |
| `numba_eval.py`         | 408        | ACTIVE    | CPU JIT eval           | Fast CPU eval          |
| `gpu_kernels.py`        | 812        | TEST-ONLY | Parity tests           | Move gen kernels       |
| `cuda_rules.py`         | 3,613      | **DEAD**  | None                   | Unused CUDA code       |
| **Total**               | **10,366** |           |                        |                        |

### 2.2 Evaluation Paths

```
Path 1: Pure CPU (Production Default)
┌──────────────────────────────────────────────────┐
│ HeuristicAI.evaluate_position(state)             │
│ ├─ CPU rules engine                              │
│ ├─ Numba JIT optimizations                       │
│ └─ 21,000+ evals/sec                             │
└──────────────────────────────────────────────────┘

Path 2: Hybrid GPU (Large Batches)
┌──────────────────────────────────────────────────┐
│ HybridGPUEvaluator.evaluate_positions(states)    │
│ ├─ CPU: GameState → GPU tensors                  │
│ ├─ GPU: batch_evaluate() on device               │
│ ├─ CPU: scores → numpy                           │
│ └─ Best for batch_size ≥ 500                     │
└──────────────────────────────────────────────────┘

Path 3: Full GPU Batch (CMA-ES Training)
┌──────────────────────────────────────────────────┐
│ ParallelGameRunner (gpu_parallel_games.py)       │
│ ├─ BatchGameState tensor representation          │
│ ├─ GPU move gen + apply + evaluate               │
│ ├─ Rule simplifications (chains, territories)    │
│ └─ Used for CMA-ES fitness evaluation            │
└──────────────────────────────────────────────────┘
```

### 2.3 Code Duplication Analysis

| Function               | Locations                                             | Status                                |
| ---------------------- | ----------------------------------------------------- | ------------------------------------- |
| `get_device()`         | gpu_kernels.py, gpu_batch.py                          | Consolidated (gpu_batch.py canonical) |
| `detect_lines_*`       | gpu_kernels.py, gpu_parallel_games.py, numba_rules.py | 3 copies                              |
| `evaluate_positions_*` | gpu_kernels.py, gpu_parallel_games.py, gpu_batch.py   | 3 copies                              |
| Victory checking       | gpu_kernels.py, gpu_parallel_games.py, numba_rules.py | 3 copies                              |

### 2.4 GPU vs CPU Evaluation Discrepancy Analysis (2025-12-11)

**CRITICAL FINDING:** The GPU and CPU evaluators are **intentionally divergent** by design:

| Implementation                        | Approach                 | Features                                  | Speed     |
| ------------------------------------- | ------------------------ | ----------------------------------------- | --------- |
| CPU `HeuristicAI.evaluate_position()` | Full 45-weight heuristic | All Tier 0/1/2 features, visibility-based | Baseline  |
| GPU `evaluate_positions_batch()`      | Simplified vectorized    | ~8 effective weights, 4-adjacency only    | 6x faster |

**Root Causes of Score Divergence (63%-200% observed):**

1. **Adjacency Model Mismatch**
   - CPU: 8-directional line-of-sight (`_get_visible_stacks()`)
   - GPU: 4-adjacent neighbors only (up/down/left/right)
   - Impact: Misses diagonal threats, distant stack interactions

2. **Constant Approximations**

   ```python
   # GPU approximates mobility with constants:
   mobility = stack_count * 4.0      # CPU: actual move enumeration
   stack_mobility = stack_count * 3.0  # CPU: per-stack neighbor checking
   ```

3. **Missing Feature Categories**
   - No cap_height distinction (capture power vs total height)
   - No territory closure (marker clustering analysis)
   - No territory safety (opponent proximity)
   - No LPS action advantage (multiplayer turn analysis)
   - No recovery potential evaluation

4. **Divergence Scales with Board Complexity**
   - Initial states: ~63% difference (few interactions)
   - Mid-game: >100% difference (stacks multiplying)
   - Complex states: ~200% difference (visibility model fails)

**This is NOT a bug** - it's an **architectural trade-off** documented in Phase 1.
Phase 2 goal: Reduce to 0.05 (numerical precision) by implementing full CPU features on GPU.

---

## 3. Dead Code Inventory

### 3.1 cuda_rules.py (3,613 lines) - DEAD

**Evidence:**

- Zero imports found in production code
- Zero imports found in CI/CD scripts
- Only referenced by experimental verification scripts
- CUDA (Numba CUDA) not available on target machines

**Contents:**

- `GPURuleChecker` class (~2,800 lines)
- `GPUHeuristicEvaluator` (duplicate of gpu_batch.py version)
- CUDA kernels for:
  - Territory flood-fill
  - Line detection
  - Move generation
  - Victory checking

**Recommendation:** DELETE ENTIRELY

**Risk Assessment:**

- Dependencies: None in production
- Test coverage: None
- Reversion difficulty: Git history preserves code
- Risk level: **VERY LOW**

### 3.2 gpu_kernels.py (812 lines) - PARTIAL

**Status:** Only used by test files (`tests/gpu/test_gpu_cpu_parity.py`)

**Functions tested but not in production:**

- `generate_placement_moves_vectorized()`
- `generate_normal_moves_vectorized()`
- `generate_capture_moves_vectorized()`
- `evaluate_positions_kernel()`
- `detect_lines_kernel()`

**Recommendation:** Keep for parity testing, document as test-only

---

## 4. Simplification Roadmap

### 4.1 Phase A: Dead Code Removal (Low Risk, High Impact)

**Goal:** Remove 3,613 lines of dead code

| Task                      | Lines Removed | Risk     | Time   |
| ------------------------- | ------------- | -------- | ------ |
| Delete `cuda_rules.py`    | 3,613         | Very Low | 1h     |
| Update any broken imports | 0             | None     | 1h     |
| **Total**                 | **3,613**     |          | **2h** |

**Result:** 35% reduction in GPU codebase

### 4.2 Phase B: Consolidation (Medium Risk, Medium Impact)

**Goal:** Eliminate code duplication

| Task                                          | Impact              | Risk | Time   |
| --------------------------------------------- | ------------------- | ---- | ------ |
| Consolidate `get_device()` to single location | Clean imports       | Low  | 2h     |
| Create `device.py` utility module             | Better organization | Low  | 2h     |
| Document `gpu_kernels.py` as test-only        | Clarity             | None | 1h     |
| **Total**                                     |                     |      | **5h** |

### 4.3 Phase C: Architecture Optimization (Higher Risk, Higher Value)

**Goal:** Optimize for actual use cases

| Task                                        | Impact                  | Risk   | Time |
| ------------------------------------------- | ----------------------- | ------ | ---- |
| Add batch size threshold for auto-selection | Smart routing           | Medium | 4h   |
| Profile and optimize transfer overhead      | Better small-batch perf | Medium | 8h   |
| Consider removing hybrid path if not used   | Simplification          | High   | 4h   |

---

## 5. Decision Matrix

### 5.1 Should We Keep the Hybrid GPU Path?

| Factor      | Pro                 | Con                        |
| ----------- | ------------------- | -------------------------- |
| Batch ≥500  | 1.75x faster        | Complexity                 |
| Batch <200  | -                   | 2-10x slower than CPU      |
| Maintenance | -                   | Two code paths to maintain |
| Training    | Required for CMA-ES | -                          |
| Production  | Not needed          | Adds complexity            |

**Recommendation:** Keep for training/CMA-ES, but don't use in production AI

### 5.2 Should We Delete cuda_rules.py?

| Factor | For Deletion            | Against          |
| ------ | ----------------------- | ---------------- |
| Usage  | Zero production imports | -                |
| Lines  | 3,613 (35% of GPU code) | -                |
| Future | CUDA not on roadmap     | Might want later |
| Risk   | Very low                | -                |

**Recommendation:** DELETE. Code preserved in git history if ever needed.

### 5.3 Evaluation Path Selection

```
if batch_size >= 500:
    use HybridGPUEvaluator  # 1.75x+ speedup
elif batch_size >= 200:
    use either  # Roughly equivalent
else:
    use HeuristicAI (CPU)  # Up to 10x faster
```

---

## 6. Implementation Plan

### 6.1 Immediate Actions (This Session)

- [x] Create benchmark script (`scripts/benchmark_gpu_cpu.py`)
- [x] Run benchmark and document results
- [x] Create this architecture document
- [x] Delete `cuda_rules.py` (3,613 lines removed)
- [x] Update `GPU_PIPELINE_ROADMAP.md` with benchmark results

### 6.2 Short-term Actions (Completed 2025-12-11)

- [x] Document `gpu_kernels.py` as test-only in docstring
- [x] Add integration tests for full game replay parity (`tests/gpu/test_gpu_cpu_replay_parity.py`)
- [x] Fix MarkerInfo handling in BatchGameState.from_single_game()
- [ ] Add batch size threshold to HybridGPUEvaluator (deferred to Phase 2)

### 6.3 Medium-term Actions (Future)

- [ ] Profile transfer overhead in detail
- [ ] Consider fusing multiple small batches for GPU efficiency
- [ ] Evaluate removing hybrid path from production code

---

## Appendix A: File Dependency Graph

```
Production Dependencies:
├── heuristic_ai.py
│   ├── numba_rules.py (CPU JIT)
│   └── numba_eval.py (CPU JIT)
│
├── hybrid_gpu.py
│   ├── gpu_batch.py (GPUHeuristicEvaluator)
│   └── gpu_parallel_games.py (batch_game_states_to_gpu)
│
└── run_gpu_selfplay.py / CMA-ES
    └── gpu_parallel_games.py (ParallelGameRunner)
        └── gpu_batch.py

Test Dependencies:
└── tests/gpu/test_gpu_cpu_parity.py
    └── gpu_kernels.py
    └── gpu_parallel_games.py

Test-only (documented):
└── gpu_kernels.py  # Used only by tests/gpu/test_gpu_cpu_parity.py

Deleted (2025-12-11):
└── cuda_rules.py  # 3,613 lines removed - preserved in git history
└── tests/test_cuda_cpu_parity.py  # Dependent on cuda_rules.py
└── scripts/verify_cuda_parity.py  # Dependent on cuda_rules.py
└── scripts/verify_full_fidelity.py  # Dependent on cuda_rules.py
```

## Appendix B: Benchmark Script Location

```
scripts/benchmark_gpu_cpu.py
```

Usage:

```bash
# Quick benchmark
python scripts/benchmark_gpu_cpu.py --board square8 --iterations 10

# Full benchmark with JSON output
python scripts/benchmark_gpu_cpu.py --json --output results.json

# Specific board type
python scripts/benchmark_gpu_cpu.py --board hexagonal
```

---

## Document History

| Date       | Version | Changes                                                                          |
| ---------- | ------- | -------------------------------------------------------------------------------- |
| 2025-12-11 | 1.0     | Initial architecture analysis and benchmark results                              |
| 2025-12-11 | 1.1     | Completed Phase A: cuda_rules.py deletion (3,613 lines), integration tests added |
