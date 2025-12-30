# Event Wiring Verification Runbook

This runbook provides procedures for verifying that all events in the RingRift AI training infrastructure are properly wired (emitters have subscribers and vice versa).

**Created**: December 27, 2025
**Version**: 1.0

## Overview

The coordination infrastructure uses 211 event types defined in `DataEventType`. This runbook helps verify:

1. All emitted events have at least one subscriber
2. Event subscriptions match the correct event type values
3. New events are properly wired before deployment

---

## Quick Verification Script

Run this script to check for orphan events:

```python
#!/usr/bin/env python3
"""Check for orphan event emitters (events emitted but no subscribers)."""

import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

AI_SERVICE = Path(__file__).parent.parent.parent
COORD_DIR = AI_SERVICE / "app" / "coordination"
P2P_DIR = AI_SERVICE / "scripts" / "p2p"

def find_emitters():
    """Find all emit calls."""
    emitters = defaultdict(list)
    # Look for emit patterns
    patterns = [
        r'emit\(["\'](\w+)["\']',
        r'emit_event\(["\'](\w+)["\']',
        r'DataEventType\.(\w+)\.value.*emit',
        r'_emit_.*_event.*["\'](\w+)["\']',
    ]
    for p in [COORD_DIR, P2P_DIR]:
        if not p.exists():
            continue
        for f in p.rglob("*.py"):
            content = f.read_text()
            for pattern in patterns:
                for m in re.finditer(pattern, content):
                    emitters[m.group(1)].append(f.name)
    return emitters

def find_subscribers():
    """Find all subscribe calls."""
    subscribers = defaultdict(list)
    patterns = [
        r'subscribe\(["\'](\w+)["\']',
        r'DataEventType\.(\w+)\.value.*subscribe',
    ]
    for p in [COORD_DIR, P2P_DIR]:
        if not p.exists():
            continue
        for f in p.rglob("*.py"):
            content = f.read_text()
            for pattern in patterns:
                for m in re.finditer(pattern, content):
                    subscribers[m.group(1)].append(f.name)
    return subscribers

def main():
    emitters = find_emitters()
    subscribers = find_subscribers()

    orphan_emitters = [e for e in emitters if e not in subscribers]
    orphan_subscribers = [s for s in subscribers if s not in emitters]

    print(f"Events emitted but no subscribers ({len(orphan_emitters)}):")
    for e in sorted(orphan_emitters):
        print(f"  {e}: {emitters[e]}")

    print(f"\nEvents subscribed but not emitted ({len(orphan_subscribers)}):")
    for s in sorted(orphan_subscribers)[:20]:  # Limit output
        print(f"  {s}: {subscribers[s]}")
    if len(orphan_subscribers) > 20:
        print(f"  ... and {len(orphan_subscribers) - 20} more")

if __name__ == "__main__":
    main()
```

**Expected Results**:

- `Events emitted but no subscribers: 0` - All emitted events should have subscribers
- `Events subscribed but not emitted` - This list is expected to be non-empty; these are subscriptions waiting for events from external systems

---

## Manual Verification Procedures

### Procedure 1: Verify a New Event Is Wired

When adding a new event type:

```bash
# 1. Check if the event is defined
grep -r "YOUR_NEW_EVENT" ai-service/app/distributed/data_events.py

# 2. Check for emitters
grep -rn "emit.*YOUR_NEW_EVENT" ai-service/app/coordination/

# 3. Check for subscribers
grep -rn "subscribe.*YOUR_NEW_EVENT" ai-service/app/coordination/

# 4. Verify emission uses correct value
# Should use: DataEventType.YOUR_NEW_EVENT.value
# Not: "YOUR_NEW_EVENT" (string literal)
```

### Procedure 2: Verify Critical Pipeline Events

Run these checks for the critical pipeline events:

```bash
# Training pipeline events
for event in TRAINING_STARTED TRAINING_COMPLETED EVALUATION_COMPLETED MODEL_PROMOTED; do
  echo "=== $event ==="
  echo "Emitters:"
  grep -rn "emit.*$event" ai-service/app/coordination/ | head -3
  echo "Subscribers:"
  grep -rn "subscribe.*$event" ai-service/app/coordination/ | head -3
  echo ""
done
```

### Procedure 3: Verify Event Type Values Match

Common bug: Using `"DATA_SYNC_COMPLETED"` instead of `DataEventType.DATA_SYNC_COMPLETED.value` ("data_sync_completed").

```bash
# Find potential string literal mismatches
grep -rn 'emit.*"[A-Z_]*"' ai-service/app/coordination/ | grep -v "DataEventType"
```

---

## Critical Event Wiring Matrix

| Event                    | Primary Emitter              | Required Subscribers                                     |
| ------------------------ | ---------------------------- | -------------------------------------------------------- |
| `TRAINING_STARTED`       | TrainingCoordinator          | SyncRouter, IdleShutdown, DataPipeline                   |
| `TRAINING_COMPLETED`     | TrainingCoordinator          | FeedbackLoop, DataPipeline, ModelDistribution            |
| `EVALUATION_COMPLETED`   | GameGauntlet                 | FeedbackLoop, CurriculumIntegration, AutoPromotionDaemon |
| `MODEL_PROMOTED`         | AutoPromotionDaemon          | ModelDistribution, FeedbackLoop                          |
| `DATA_SYNC_COMPLETED`    | SyncPlanner, P2POrchestrator | DataPipelineOrchestrator                                 |
| `NEW_GAMES_AVAILABLE`    | DataPipelineOrchestrator     | SelfplayScheduler, ExportScheduler                       |
| `REGRESSION_DETECTED`    | ModelPerformanceWatchdog     | ModelLifecycleCoordinator, DataPipeline                  |
| `ORPHAN_GAMES_DETECTED`  | OrphanDetectionDaemon        | DataPipelineOrchestrator                                 |
| `BACKPRESSURE_ACTIVATED` | BackpressureMonitor          | SyncRouter, DataPipeline                                 |

---

## Automated CI Check

Add to CI pipeline:

```yaml
# .github/workflows/event-wiring.yml
name: Event Wiring Check
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check for orphan events
        run: |
          cd ai-service
          python -c "
          import re
          from pathlib import Path
          from collections import defaultdict

          coord = Path('app/coordination')
          emitters = set()
          subscribers = set()

          for f in coord.rglob('*.py'):
              content = f.read_text()
              emitters.update(re.findall(r'emit\([\"\'](\w+)[\"\']', content))
              subscribers.update(re.findall(r'subscribe\([\"\'](\w+)[\"\']', content))

          orphans = emitters - subscribers
          if orphans:
              print(f'FAIL: Orphan emitters found: {orphans}')
              exit(1)
          print('PASS: All emitted events have subscribers')
          "
```

---

## Troubleshooting

### Event Not Being Received

1. **Check subscription timing**: Subscriber must start before emitter

   ```python
   # In master_loop.py startup order
   # WRONG: AUTO_SYNC before DATA_PIPELINE
   # RIGHT: DATA_PIPELINE before AUTO_SYNC
   ```

2. **Check event type value**:

   ```python
   # WRONG
   router.subscribe("DATA_SYNC_COMPLETED", handler)

   # RIGHT
   router.subscribe(DataEventType.DATA_SYNC_COMPLETED.value, handler)
   ```

3. **Check router instance**: Both must use same router
   ```python
   # Verify using singleton
   from app.coordination.event_router import get_event_bus
   router = get_event_bus()  # Always use this
   ```

### Event Emitted Multiple Times

1. **Check for duplicate subscriptions**:

   ```bash
   grep -rn "subscribe.*YOUR_EVENT" ai-service/app/coordination/ | wc -l
   ```

2. **Check deduplication is enabled**:
   ```python
   # EventRouter has content-based dedup via SHA256
   # If seeing duplicates, check if different data payloads
   ```

---

## Adding New Events Checklist

- [ ] Add to `DataEventType` enum in `data_events.py`
- [ ] Add emitter in appropriate module
- [ ] Add subscriber(s) in appropriate module(s)
- [ ] Verify subscriber starts before emitter (check startup order)
- [ ] Add to EVENT_SYSTEM_REFERENCE.md documentation
- [ ] Add to this runbook's Critical Event Matrix if pipeline-critical
- [ ] Run verification script to confirm wiring

---

## Related Documentation

- [EVENT_SYSTEM_REFERENCE.md](../EVENT_SYSTEM_REFERENCE.md) - Full event type reference
- [DAEMON_FAILURE_RECOVERY.md](DAEMON_FAILURE_RECOVERY.md) - Daemon restart procedures
- [CLUSTER_INTEGRATION_GUIDE.md](../CLUSTER_INTEGRATION_GUIDE.md) - Architecture overview
