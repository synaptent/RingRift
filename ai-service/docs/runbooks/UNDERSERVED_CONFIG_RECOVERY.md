# Underserved Config Recovery Runbook

**Last Updated**: December 30, 2025

This runbook documents the procedure for recovering underserved board configurations
that have insufficient training data or failing gauntlet evaluations.

## Symptoms

### Data Starvation

- Config has <1,000 games in canonical database
- Neural network underperforms vs heuristic baseline
- Gauntlet evaluation shows <60% win rate vs heuristic

### Elo Stall

- No Elo improvement for >6 hours
- PROGRESS_STALL_DETECTED event emitted
- Training iterations complete but Elo stays flat

## Diagnosis

### 1. Check Game Counts

```bash
# Quick check via P2P status
curl -s http://localhost:8770/manifest | python3 -c '
import sys, json
m = json.load(sys.stdin)
games = m.get("games", {})
for config, count in sorted(games.items(), key=lambda x: x[1]):
    status = "CRITICAL" if count < 600 else "LOW" if count < 2000 else "OK"
    print(f"{config}: {count} games [{status}]")
'

# Or check canonical databases directly
for db in data/games/canonical_*.db; do
  config=$(basename $db .db | sed 's/canonical_//')
  count=$(sqlite3 $db "SELECT COUNT(*) FROM games WHERE is_complete=1")
  echo "$config: $count games"
done
```

### 2. Check Gauntlet Results

```bash
# Recent gauntlet results from Elo database
sqlite3 data/elo/ratings.db "
SELECT config_key, model_id, elo_rating, win_rate, last_updated
FROM model_ratings
WHERE last_updated > datetime('now', '-7 days')
ORDER BY config_key, elo_rating DESC
"
```

### 3. Check Selfplay Allocation

```bash
# Current allocation from P2P
curl -s http://localhost:8770/selfplay/allocation | python3 -m json.tool

# Verify priority weights
grep -A 20 "board_priority_overrides" config/unified_loop.yaml
```

## Recovery Steps

### Step 1: Verify Cluster Availability

```bash
# Check P2P cluster status
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")} nodes")
print(f"Queue: {len(d.get(\"work_queue\", []))} items")
'

# If <50% nodes alive, run node recovery first
python scripts/recover_tailscale_nodes.py --dry-run
```

### Step 2: Update Priority Configuration

If the config isn't in underserved_configs or has wrong priority:

1. Edit `config/distributed_hosts.yaml`:

   ```yaml
   underserved_configs:
     - YOUR_CONFIG # Add at top for CRITICAL priority
   ```

2. Edit `config/unified_loop.yaml`:

   ```yaml
   board_priority_overrides:
     YOUR_CONFIG: 0 # 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW
   ```

3. Restart affected daemons or wait for config reload (~5 min)

### Step 3: Dispatch Targeted Selfplay

```bash
# Dispatch high-priority selfplay jobs via P2P
# Adjust num_games based on current deficit

# For CRITICAL configs (<600 games)
curl -X POST http://localhost:8770/selfplay/dispatch \
  -H "Content-Type: application/json" \
  -d '{"config_key": "hex8_3p", "num_games": 2000, "priority": "critical"}'

curl -X POST http://localhost:8770/selfplay/dispatch \
  -d '{"config_key": "square19_4p", "num_games": 2000, "priority": "critical"}'

# For HIGH configs (<1500 games)
curl -X POST http://localhost:8770/selfplay/dispatch \
  -d '{"config_key": "square19_3p", "num_games": 1000, "priority": "high"}'
```

### Step 4: Force Data Sync

```bash
# Trigger immediate sync to consolidate games from cluster
curl -X POST http://localhost:8770/sync/trigger \
  -d '{"type": "games", "force": true}'

# Or run manual sync from coordinator
python scripts/unified_data_sync.py --force-sync --config YOUR_CONFIG
```

### Step 5: Verify Progress

```bash
# Watch game count increase
watch -n 60 'sqlite3 data/games/canonical_hex8_3p.db "SELECT COUNT(*) FROM games"'

# Check work queue status
curl -s http://localhost:8770/status | jq '.work_queue | length'

# Verify selfplay jobs running
curl -s http://localhost:8770/jobs | jq '[.[] | select(.type=="selfplay")] | length'
```

## Prevention

### 1. Multiplayer Allocation Guarantees

The `multiplayer_allocation` config in `unified_loop.yaml` ensures minimum allocation:

- 3-player configs get at least 20% of allocation
- 4-player configs get at least 25% of allocation
- 4-player configs get 2.5x priority multiplier

### 2. Starvation Tier System

The selfplay scheduler uses starvation tiers (defined in `coordination_defaults.py`):

- **ULTRA** (<100 games): 25x priority boost
- **EMERGENCY** (<500 games): 10x priority boost
- **CRITICAL** (<1000 games): 5x priority boost

### 3. Progress Watchdog

The `progress_watchdog` daemon monitors for stalls:

- Detects <5 Elo improvement in 6 hours
- Automatically boosts selfplay priority for stalled configs
- Emits PROGRESS_STALL_DETECTED events for alerting

## Related Files

| File                                           | Purpose                                          |
| ---------------------------------------------- | ------------------------------------------------ |
| `config/distributed_hosts.yaml`                | underserved_configs list                         |
| `config/unified_loop.yaml`                     | board_priority_overrides, multiplayer_allocation |
| `app/config/coordination_defaults.py`          | Starvation tier thresholds                       |
| `app/coordination/selfplay_scheduler.py`       | Priority calculation logic                       |
| `app/coordination/progress_watchdog_daemon.py` | Stall detection                                  |

## Thresholds Reference

| Tier      | Game Count | Priority Boost | Action              |
| --------- | ---------- | -------------- | ------------------- |
| ULTRA     | <100       | 25x            | Immediate dispatch  |
| EMERGENCY | <500       | 10x            | High priority queue |
| CRITICAL  | <1000      | 5x             | Elevated priority   |
| LOW       | <2000      | 2x             | Normal queue        |
| STANDARD  | >=2000     | 1x             | Normal operation    |

## Environment Variables

Override thresholds via environment variables:

```bash
# Raise starvation thresholds for aggressive recovery
export RINGRIFT_DATA_STARVATION_ULTRA_THRESHOLD=100
export RINGRIFT_DATA_STARVATION_EMERGENCY_THRESHOLD=600
export RINGRIFT_DATA_STARVATION_CRITICAL_THRESHOLD=2000
```
