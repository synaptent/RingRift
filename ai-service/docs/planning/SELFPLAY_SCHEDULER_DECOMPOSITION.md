# SelfplayScheduler Decomposition Plan

## Current State

**File**: `app/coordination/selfplay_scheduler.py`
**Size**: 3,875 LOC
**Status**: Mega-class identified for decomposition (December 2025)

## Problem Statement

The `SelfplayScheduler` class handles too many responsibilities:

1. Event handling for 20+ event types
2. Priority computation with complex weighting
3. Batch allocation across nodes
4. Node capability tracking
5. Data fetching from external sources
6. Status/metrics reporting

This makes the class difficult to:

- Test in isolation
- Understand at a glance
- Modify without risk of regression
- Extend with new features

## Proposed Decomposition

### Phase 1: Extract Event Handlers (~1,200 LOC)

Create `app/coordination/selfplay_event_handlers.py`:

```python
class SelfplayEventHandlerMixin:
    """Event handlers for SelfplayScheduler.

    Extracted from SelfplayScheduler to improve modularity.
    Uses HandlerBase patterns for consistent event subscription.
    """

    def _on_selfplay_complete(self, event) -> None: ...
    def _on_training_complete(self, event) -> None: ...
    def _on_promotion_complete(self, event) -> None: ...
    # ... 17 more handlers
```

**Benefits**:

- Clear separation of event-driven logic
- Easier to test handlers in isolation
- Follows existing mixin patterns (see `scripts/p2p/p2p_mixin_base.py`)

**Migration**:

1. Create mixin class with all `_on_*` methods
2. Have `SelfplayScheduler` inherit from mixin
3. Verify all event subscriptions still work
4. Add unit tests for individual handlers

### Phase 2: Extract Priority Calculator (~400 LOC)

Create `app/coordination/priority_calculator.py`:

```python
@dataclass
class PriorityInputs:
    """Inputs needed for priority computation."""
    config_key: str
    current_elo: float
    elo_velocity: float
    data_freshness: float
    curriculum_weight: float
    quality_score: float
    games_count: int

class PriorityCalculator:
    """Computes priority scores for selfplay configurations.

    Pure computation class - no side effects, no event handling.
    """

    def compute_dynamic_weights(self, cluster_state: ClusterState) -> DynamicWeights: ...
    def compute_priority_score(self, inputs: PriorityInputs, weights: DynamicWeights) -> float: ...
    def get_adaptive_budget(self, elo: float) -> int: ...
    def compute_target_games(self, config: str, current_elo: float) -> int: ...
```

**Benefits**:

- Pure functions are easy to test
- No external dependencies (receives data, returns results)
- Weights and computation logic isolated

### Phase 3: Extract Allocation Manager (~500 LOC)

Create `app/coordination/allocation_manager.py`:

```python
class AllocationManager:
    """Manages selfplay batch allocation across nodes.

    Handles:
    - Batch size computation
    - Node capacity matching
    - Starvation floor enforcement
    - 4-player allocation minimums
    """

    def allocate_batch(self, priorities: list[ConfigPriority], nodes: list[NodeCapability]) -> BatchAllocation: ...
    def enforce_starvation_floor(self, allocation: dict) -> dict: ...
    def enforce_4p_minimums(self, allocation: dict) -> dict: ...
    def allocate_to_nodes(self, games: int, nodes: list) -> list[NodeAllocation]: ...
```

**Benefits**:

- Allocation logic is complex and deserves isolation
- Easier to test allocation edge cases
- Clear interface for batch allocation

### Phase 4: Extract Data Accessor (~500 LOC)

Create `app/coordination/selfplay_data_accessor.py`:

```python
class SelfplayDataAccessor:
    """Fetches external data needed by SelfplayScheduler.

    Consolidates all `_get_*` methods that fetch from:
    - Elo databases
    - Game databases
    - Feedback controllers
    - Curriculum integration
    - Quality monitors
    """

    async def get_data_freshness(self) -> dict[str, float]: ...
    async def get_elo_velocities(self) -> dict[str, float]: ...
    async def get_current_elos(self) -> dict[str, float]: ...
    async def get_feedback_signals(self) -> dict[str, dict]: ...
    async def get_curriculum_weights(self) -> dict[str, float]: ...
    async def get_game_counts(self) -> dict[str, int]: ...
```

**Benefits**:

- Data access patterns consolidated
- Can add caching layer easily
- Mock-friendly for testing

### Resulting Structure

After decomposition:

```
app/coordination/
├── selfplay_scheduler.py              # Main class (~1,200 LOC, down from 3,875)
├── selfplay_event_handlers.py         # Event handler mixin (~1,200 LOC)
├── priority_calculator.py             # Pure computation (~400 LOC)
├── allocation_manager.py              # Batch allocation (~500 LOC)
└── selfplay_data_accessor.py          # Data fetching (~500 LOC)
```

Main `SelfplayScheduler` becomes:

```python
class SelfplayScheduler(SelfplayEventHandlerMixin):
    """Priority-based selfplay configuration scheduling.

    Coordinates:
    - PriorityCalculator for scoring
    - AllocationManager for batch allocation
    - SelfplayDataAccessor for external data

    December 2025: Decomposed from 3,875 LOC to ~1,200 LOC.
    """

    def __init__(self):
        self.priority_calculator = PriorityCalculator()
        self.allocation_manager = AllocationManager()
        self.data_accessor = SelfplayDataAccessor()
        self.subscribe_to_events()

    async def get_priority_configs(self, top_n: int = 12) -> list[tuple[str, float]]: ...
    async def allocate_selfplay_batch(self, target_games: int) -> BatchAllocation: ...
    def get_metrics(self) -> dict[str, Any]: ...
    def get_status(self) -> dict[str, Any]: ...
    def health_check(self) -> HealthCheckResult: ...
```

## Implementation Order

1. **Phase 1 (Event Handlers)**: Lowest risk - mixins are additive
2. **Phase 2 (Priority Calculator)**: Pure functions - easy to test
3. **Phase 3 (Allocation Manager)**: More complex - needs careful testing
4. **Phase 4 (Data Accessor)**: Optional - mostly organizational

## Testing Strategy

For each phase:

1. Create unit tests for the extracted component
2. Create integration tests verifying the main class still works
3. Run existing tests to ensure no regression
4. Update any direct imports from other modules

## Backward Compatibility

- Keep `SelfplayScheduler` as the main entry point
- Internal decomposition should be invisible to callers
- All public methods remain unchanged
- Factory function `get_selfplay_scheduler()` unchanged

## Timeline

- Phase 1: 1-2 hours
- Phase 2: 2-3 hours
- Phase 3: 3-4 hours
- Phase 4: 1-2 hours
- Testing/Integration: 2-3 hours

**Total: ~12 hours of focused work**

## Related Modules

Note: There is also `scripts/p2p/managers/selfplay_scheduler.py` (80KB) which handles P2P job targeting. This is a different concern and should remain separate. The P2P version focuses on:

- Job targeting per node
- Diversity tracking
- CPU-only job allocation

The coordination layer version (`app/coordination/`) focuses on:

- Config priority scoring
- Batch allocation planning
- Event-driven updates

Both should exist but with clear responsibilities.
