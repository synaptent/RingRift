# Plan: Raft Async Compatibility & Cluster Stability

## Executive Summary

The P2P cluster experiences 100% CPU usage when Raft consensus is enabled due to PySyncObj's
`autoTick=True` configuration spawning busy-wait threads that don't integrate with asyncio.
This plan outlines a phased approach to fix the issue and eventually re-enable Raft for
sub-second failover while maintaining the stable Bully algorithm as fallback.

## Root Cause Analysis

### Problem 1: PySyncObj autoTick Causes 100% CPU

**Location**: `ai-service/app/p2p/raft_state.py` (lines 443-454)

```python
conf = SyncObjConf(
    autoTick=True,  # <-- ROOT CAUSE
    ...
)
```

**Why it causes 100% CPU**:

1. PySyncObj spawns internal threads for Raft operations
2. These threads run tight polling loops with minimal sleep (1ms)
3. On GPU nodes already at high CPU, this creates thermal hotspots
4. No integration with asyncio - threads compete with event loop

### Problem 2: \_stats Attribute Shadowing (FIXED)

**Status**: Already fixed in commits de0b9bf8e and 53a8df213

- GossipPeerPromotionLoop renamed `_stats` to `_promotion_stats`
- BaseLoop.get_status() now has defensive `hasattr()` check

## Implementation Plan

### Phase 1: Immediate Stability (P0) - DONE

**Status**: ✅ Complete

- [x] Keep Raft disabled via `RINGRIFT_RAFT_ENABLED=false`
- [x] Fix \_stats shadowing bug (commit de0b9bf8e)
- [x] Make get_status() defensive (commit 53a8df213)

### Phase 2: PySyncObj Manual Ticking (P1) - 2-3 hours

**Goal**: Enable Raft with manual ticking instead of autoTick threads

#### Step 2.1: Create AsyncSyncObj Wrapper

Create `ai-service/app/p2p/async_syncobj_wrapper.py`:

```python
"""Async wrapper for PySyncObj with manual ticking.

This wrapper replaces autoTick=True with manual ticking driven by
asyncio, preventing the busy-wait CPU spinning issue.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pysyncobj import SyncObj, SyncObjConf
from typing import Any, Optional

class AsyncSyncObjWrapper:
    """Wraps PySyncObj for asyncio compatibility with manual ticking."""

    def __init__(
        self,
        sync_obj: SyncObj,
        tick_interval: float = 0.05,  # 50ms between ticks
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self._sync_obj = sync_obj
        self._tick_interval = tick_interval
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="raft_tick")
        self._running = False
        self._tick_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the manual tick loop."""
        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        """Stop the manual tick loop."""
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass

    async def _tick_loop(self) -> None:
        """Drive PySyncObj from asyncio with controlled timing."""
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                # Run tick in thread pool to avoid blocking event loop
                await loop.run_in_executor(
                    self._executor,
                    self._sync_obj.doTick,
                    0.0  # Non-blocking tick
                )
            except Exception as e:
                # Log but don't crash the tick loop
                import logging
                logging.getLogger(__name__).warning(f"Raft tick error: {e}")

            # Yield control to event loop
            await asyncio.sleep(self._tick_interval)

    async def call_replicated(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a replicated method asynchronously."""
        loop = asyncio.get_event_loop()
        method = getattr(self._sync_obj, method_name)
        return await loop.run_in_executor(self._executor, method, *args)

    @property
    def sync_obj(self) -> SyncObj:
        """Access underlying SyncObj for read operations."""
        return self._sync_obj
```

#### Step 2.2: Modify ReplicatedWorkQueue Configuration

Update `ai-service/app/p2p/raft_state.py`:

```python
# Change from:
conf = SyncObjConf(
    autoTick=True,  # OLD
    ...
)

# To:
conf = SyncObjConf(
    autoTick=False,  # NEW - Manual ticking via AsyncSyncObjWrapper
    ...
)
```

#### Step 2.3: Integrate Wrapper in P2P Orchestrator

Update `ai-service/scripts/p2p_orchestrator.py`:

```python
from app.p2p.async_syncobj_wrapper import AsyncSyncObjWrapper

# In _init_raft_consensus():
self._raft_work_queue = ReplicatedWorkQueue(...)
self._raft_wrapper = AsyncSyncObjWrapper(self._raft_work_queue)

# In run():
await self._raft_wrapper.start()

# In shutdown():
await self._raft_wrapper.stop()
```

#### Step 2.4: Add Health Check for Raft Tick Loop

```python
async def _raft_health_check(self) -> dict:
    """Check Raft tick loop health."""
    return {
        "running": self._raft_wrapper._running if hasattr(self, "_raft_wrapper") else False,
        "tick_interval": self._raft_wrapper._tick_interval if hasattr(self, "_raft_wrapper") else None,
        "is_ready": self._raft_work_queue._is_ready if hasattr(self, "_raft_work_queue") else False,
    }
```

### Phase 3: Enhanced BaseLoop Type Safety (P2)

**Goal**: Prevent future \_stats shadowing issues

#### Step 3.1: Add \_stats Type Validation

Update `ai-service/scripts/p2p/loops/base.py`:

```python
def __init__(self, ...):
    # Statistics - protected attribute
    self._stats = LoopStats(name=name)

def __setattr__(self, name: str, value: Any) -> None:
    """Validate _stats is not shadowed with incompatible type."""
    if name == "_stats" and hasattr(self, "_stats"):
        from .loop_stats import LoopStats
        if not isinstance(value, LoopStats) and not hasattr(value, "to_dict"):
            import warnings
            warnings.warn(
                f"{self.__class__.__name__}: _stats should be LoopStats or have to_dict() method. "
                f"Use a different attribute name for custom stats (e.g., _custom_stats).",
                stacklevel=2
            )
    super().__setattr__(name, value)
```

#### Step 3.2: Document \_stats Contract

Add to BaseLoop docstring:

```python
"""
Protected Attributes:
    _stats (LoopStats): Run statistics. Subclasses MUST NOT override this
        with incompatible types. Use separate attributes for custom stats
        (e.g., _my_custom_stats).
"""
```

### Phase 4: Cluster Node Updates (P3)

**Goal**: Deploy fixes to all cluster nodes

#### Step 4.1: Update All Nodes

```bash
cd ai-service
python scripts/update_all_nodes.py --restart-p2p
```

#### Step 4.2: Verify Deployment

```bash
# Check each node has the fix
for node in $(grep -E "^\s+\w+:" config/distributed_hosts.yaml | cut -d: -f1); do
    echo "Checking $node..."
    ssh $node "cd RingRift && git log -1 --oneline"
done
```

### Phase 5: Long-term Raft Migration (P4) - Future

**Goal**: Evaluate native asyncio Raft alternatives

#### Option A: flowerhack/raft (Recommended)

- Native asyncio implementation
- Similar API to PySyncObj
- Drop-in replacement potential

#### Option B: etcd with aetcd3 client

- External consensus service
- Battle-tested (Kubernetes uses it)
- More operational overhead

#### Option C: rraft-py (Rust bindings)

- Highest performance
- More complex integration
- Best for very large clusters

## Testing Plan

### Unit Tests

1. Test AsyncSyncObjWrapper tick loop starts/stops correctly
2. Test call_replicated() works with timeout
3. Test \_stats type validation warning

### Integration Tests

1. Enable Raft with manual ticking, verify no 100% CPU
2. Test leader election with Raft enabled
3. Test work queue operations under load
4. Verify graceful shutdown

### Cluster Tests

1. Deploy to staging cluster (3 nodes)
2. Monitor CPU usage for 4+ hours
3. Trigger leader failover, measure recovery time
4. Compare Raft vs Bully failover times

## Success Criteria

| Metric               | Target            | Current               |
| -------------------- | ----------------- | --------------------- |
| CPU usage with Raft  | < 5% overhead     | 100% (broken)         |
| Leader failover time | < 1 second (Raft) | 70-100s (Bully)       |
| Cluster stability    | 4+ hours          | 30-60 min (with Raft) |
| Node count           | 20+ stable        | 5-10 (variable)       |

## Rollback Plan

If Phase 2 causes issues:

1. Set `RINGRIFT_RAFT_ENABLED=false`
2. Restart P2P orchestrator
3. Cluster falls back to Bully algorithm (proven stable)

## Timeline

| Phase                        | Effort    | Priority |
| ---------------------------- | --------- | -------- |
| Phase 1: Immediate Stability | Done      | P0       |
| Phase 2: Manual Ticking      | 2-3 hours | P1       |
| Phase 3: Type Safety         | 1 hour    | P2       |
| Phase 4: Cluster Updates     | 30 min    | P3       |
| Phase 5: Long-term Migration | TBD       | P4       |

## Files to Modify

1. `ai-service/app/p2p/async_syncobj_wrapper.py` (NEW)
2. `ai-service/app/p2p/raft_state.py` (modify SyncObjConf)
3. `ai-service/scripts/p2p_orchestrator.py` (integrate wrapper)
4. `ai-service/scripts/p2p/loops/base.py` (type validation)
5. `ai-service/tests/unit/p2p/test_async_syncobj_wrapper.py` (NEW)

## References

- PySyncObj docs: https://pysyncobj.readthedocs.io/
- flowerhack/raft: https://github.com/flowerhack/raft
- aetcd3: https://github.com/space-gurtam/aetcd3
- Raft paper: https://raft.github.io/raft.pdf
