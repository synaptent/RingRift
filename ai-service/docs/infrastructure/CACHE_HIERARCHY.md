# Cache Hierarchy Documentation: RingRift AI-Service

_Generated: December 2025_

## Overview

The RingRift ai-service implements a **multi-layered cache hierarchy** with 5 distinct cache systems, operating at different abstraction levels and addressing different performance concerns.

```
┌─────────────────────────────────────────────────────────┐
│     Cache Coordination Orchestrator (December 2025)      │
│     - Centralized tracking across cluster                │
│     - Model promotion events trigger invalidation        │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────────┬──────────────────┐
        │                         │                  │
   ┌────▼─────┐          ┌────────▼───────┐   ┌─────▼──────┐
   │  Generic  │          │  Domain-Specific   │  Training  │
   │ Caching   │          │  Game Caches       │  Caches    │
   │ System    │          │                    │            │
   └────┬─────┘          └────────┬───────┘   └─────┬──────┘
        │                         │                  │
   [Decorators]            ┌──────┴──────┬──────┐   │
   [Memory Impl]           │             │      │   │
   [Base Classes]      [Moves]    [Territory] [Models] [Eval/Export]
```

---

## Tier 1: Generic Caching System

**Location:** `app/caching/`

### Purpose

Provides infrastructure-level caching abstractions used as foundation for specialized caches.

### Components

| Component       | File          | Description                       |
| --------------- | ------------- | --------------------------------- |
| `Cache[K, V]`   | base.py       | Abstract base class               |
| `CacheEntry`    | base.py       | Value wrapper with metadata       |
| `MemoryCache`   | memory.py     | Thread-safe LRU with optional TTL |
| `@cached`       | decorators.py | Synchronous function memoization  |
| `@async_cached` | decorators.py | Async function memoization        |

### Configuration

- **Max Size:** 1000 entries (default)
- **TTL:** None (configurable)
- **Eviction:** LRU when capacity exceeded

---

## Tier 2: Domain-Specific Caches

### Move Cache (`app/ai/move_cache.py`)

**Purpose:** Cache valid move lists for game states.

**Key Components:**

- Board metadata + phase
- Players digest (rings, eliminated status, territory)
- Stack/marker/collapsed positions
- History context (swap eligibility, last move)
- Rules options

**Configuration:**

- **Max Size:** `RINGRIFT_MOVE_CACHE_SIZE` env var, default 1000
- **TTL:** None (per-game)
- **Eviction:** LRU

**Invalidation:**

- Phase-based bypass (line_processing, territory_processing)
- Chain capture state bypass
- Manual via `clear_move_cache()`

---

### Territory Cache (`app/ai/territory_cache.py`)

**Purpose:** Pre-compute board geometry and cache region detection.

**Components:**

- `BoardGeometryCache`: Singleton per board type with neighbor arrays
- `RegionCache`: Hash-validated region results

**Key Strategy:**

- Geometry: `f"{board_type}_{size}"`
- Validity: Hash of (markers, collapsed, stacks, active_players)

**Configuration:**

- **Geometry:** Unbounded singleton (permanent)
- **Regions:** Ephemeral, rebuilt when invalid

---

### Model Cache (`app/ai/model_cache.py`)

**Purpose:** Singleton LRU cache for neural network instances.

**Key Components:**

```python
Key: (architecture_type, device_str, model_path, checkpoint_signature, board_type)
```

**Configuration:**

- **Max Size:** 10 models (`MODEL_CACHE_MAX_SIZE`)
- **TTL:** 1 hour (3600s)
- **Eviction:** TTL first, then LRU
- **GPU Cleanup:** Auto-clears CUDA/MPS on eviction

---

## Tier 3: Coordination Layer

### Cache Coordination Orchestrator (`app/coordination/cache_coordination_orchestrator.py`)

**Purpose:** Centralized cache management across distributed cluster.

**Features:**

- Tracks cache entries on all nodes
- Coordinates invalidation on model promotion
- Provides unified visibility into cache efficiency
- Multi-index tracking (by node, model, type)

**Key Strategy:**

```python
cache_id = f"{node_id}:{cache_type.value}:{model_id}"
```

**Configuration:**

- **Default TTL:** 1 hour
- **Stale threshold:** 30 minutes
- **Max entries per node:** 1000

---

### Cache Invalidation (`app/ai/cache_invalidation.py`)

**Purpose:** Bridges cache systems for coordinated invalidation.

**Registered Invalidators:**

1. `model_cache` - NN weight instances
2. `move_cache` - Legal move lists
3. `territory_cache` - Region detection
4. `eval_cache` - Training evaluation
5. `export_cache` - Model export data

**Triggers:**
| Event | Invalidation Scope |
|-------|-------------------|
| `MODEL_PROMOTED` | Full |
| `TRAINING_COMPLETED` | Selective (eval, export) |
| `REGRESSION_DETECTED` | Full |
| `NAS_COMPLETED` | Selective (model, eval, export) |

**Cooldown:** 5-second minimum between full invalidations

---

## Event Flow

```
MODEL_PROMOTED event
  ↓
CacheCoordinationOrchestrator._on_model_promoted()
  ↓
invalidate_model() marks entries as INVALIDATED
  ↓
Emit CACHE_INVALIDATED event
  ↓
ModelPromotionCacheInvalidator._on_model_promoted()
  ↓
Calls registered invalidators for Tier 2 caches
  ↓
Each clears its state (model_cache.clear(), move_cache.clear(), etc.)
```

---

## Summary Table

| Layer | System          | Key Type                  | TTL  | Max Size  | Eviction     | Invalidation          |
| ----- | --------------- | ------------------------- | ---- | --------- | ------------ | --------------------- |
| 1     | Generic Cache   | Hashed args               | None | 1000      | LRU          | Manual                |
| 2     | Move Cache      | MD5(board+hist+rules)     | None | 1000      | LRU          | Events + Phase bypass |
| 2     | Territory Cache | Hash(keys)                | N/A  | N/A       | N/A          | Hash validity         |
| 2     | Model Cache     | (arch, device, path, sig) | 1h   | 10        | LRU+TTL      | Events                |
| 3     | Orchestrator    | node:type:model           | 1h   | 1000/node | Event-driven | MODEL_PROMOTED        |

---

## Known Issues and Recommendations

### Issue 1: Dual-Listener Pattern

Both `CacheCoordinationOrchestrator` and `ModelPromotionCacheInvalidator` listen to `MODEL_PROMOTED`.

**Recommendation:** Single orchestrator pattern to prevent duplicate work.

### Issue 2: Territory Cache Reconstruction Incomplete

`_build_territories_from_cache()` returns empty list.

**Impact:** Cache validity checked but results not reused.

### Issue 3: Move Cache History Hashing

O(history_length) scan for swap_sides eligibility in key generation.

**Recommendation:** Track swap eligibility incrementally in GameState.

---

## File Locations

| Purpose         | File                                                  |
| --------------- | ----------------------------------------------------- |
| Generic caching | `app/caching/`                                        |
| Move cache      | `app/ai/move_cache.py`                                |
| Territory cache | `app/ai/territory_cache.py`                           |
| Model cache     | `app/ai/model_cache.py`                               |
| Orchestrator    | `app/coordination/cache_coordination_orchestrator.py` |
| Invalidation    | `app/ai/cache_invalidation.py`                        |
