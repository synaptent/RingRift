# Data Freshness Check - Quick Reference

**Status:** Mandatory by default (December 26, 2025)
**Goal:** Prevent 95% of stale data training incidents

## TL;DR

Training now **fails by default** if data is older than 1 hour.

```bash
# ✅ Normal training (freshness check enabled)
python -m app.training.train --board-type hex8 --num-players 2 ...

# ⚠️  Skip check (NOT RECOMMENDED)
python -m app.training.train --skip-freshness-check ...

# ⚠️  Allow stale data (warns instead of fails)
python -m app.training.train --allow-stale-data ...
```

## Default Values

| Parameter              | Default | Meaning                      |
| ---------------------- | ------- | ---------------------------- |
| `skip_freshness_check` | `False` | Check **IS enabled**         |
| `max_data_age_hours`   | `1.0`   | Data must be **<1 hour old** |
| `allow_stale_data`     | `False` | **FAIL** on stale (not warn) |

## When Training is Blocked

You'll see:

```
TRAINING BLOCKED: Training data is STALE for hex8_2p:
  - Data age: 3.2 hours
  - Threshold: 1.0 hours
  - Games available: 5000
```

### Option 1: Get Fresh Data (Recommended)

```bash
python scripts/unified_data_sync.py --once
```

### Option 2: Allow Stale Data (Not Recommended)

```bash
python -m app.training.train \
  --allow-stale-data \
  --board-type hex8 --num-players 2 ...
```

**Warning:** May degrade model quality.

### Option 3: Skip Check (Dangerous)

```bash
python -m app.training.train \
  --skip-freshness-check \
  --board-type hex8 --num-players 2 ...
```

**Danger:** Only for debugging. Bypasses all safety checks.

### Option 4: Adjust Threshold

```bash
python -m app.training.train \
  --max-data-age-hours 2.5 \
  --board-type hex8 --num-players 2 ...
```

## Examples

### Standard Training (Check Enabled)

```bash
python -m app.training.train \
  --board-type hex8 \
  --num-players 2 \
  --data-path data/training/hex8_2p.npz
```

This automatically:

- ✅ Checks data age
- ✅ Fails if data is older than 1 hour
- ✅ Shows options to get fresh data

### Curriculum Training (Check Enabled)

```bash
python -m app.training.train \
  --curriculum \
  --board-type square8 \
  --num-players 4
```

Freshness check applies to curriculum training too.

### Background Training with Stale Data Allowed

```bash
nohup python -m app.training.train \
  --board-type hex8 \
  --num-players 2 \
  --allow-stale-data \
  > logs/train.log 2>&1 &
```

Use sparingly - only when fresh data sync is problematic.

## Implementation Details

### What Gets Checked

- **Game databases:** `data/games/*.db` files
- **NPZ files:** `data/training/*.npz` files
- **Age calculation:** Based on file modification time

### When Check Runs

- **Before training starts:** Pre-validation phase
- **Only on main process:** In distributed training
- **Skipped if module missing:** Falls back to warning

### Event Emission

When blocked, emits `TRAINING_BLOCKED_BY_QUALITY` event:

```python
{
    "config_key": "hex8_2p",
    "reason": "stale_data",
    "data_age_hours": 3.2,
    "threshold_hours": 1.0,
    "games_available": 5000,
}
```

This triggers selfplay acceleration to generate fresh data.

## Migration Notes

### Before (Dec 2025)

```bash
# Had to explicitly enable check
python -m app.training.train --check-data-freshness ...
```

### After (Dec 26, 2025)

```bash
# Check is always enabled
python -m app.training.train ...
```

## Testing

Run tests to verify behavior:

```bash
pytest tests/unit/training/test_freshness_check_mandatory.py -v
```

## See Also

- `docs/DATA_FRESHNESS_CHECK.md` - Full documentation
- `app/coordination/training_freshness.py` - Implementation
- `app/training/train.py` - Integration point
