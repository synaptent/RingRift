# Game Data Corruption Recovery Runbook

This runbook covers detection, diagnosis, and recovery procedures for corrupted game data in the RingRift AI training infrastructure.

**Created**: December 28, 2025
**Version**: 1.0
**Severity**: Critical

---

## Overview

Game data corruption can occur due to:

- Phase tracking bugs (e.g., hex8_4p corruption Dec 2025)
- Interrupted database writes
- Network transfer failures
- Move serialization errors

Corrupted games have moves that can't be replayed, making them useless for training.

---

## Detection Methods

### Method 1: Parity Validation Script

```bash
# Check a specific database
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8_4p.db \
  --verbose

# Check all canonical databases
for db in data/games/canonical_*.db; do
  echo "=== Checking $db ==="
  python scripts/check_ts_python_replay_parity.py --db "$db" --limit 100
done
```

### Method 2: Data Quality Checker

```bash
# Check database structure and content
python -m app.training.data_quality --db data/games/my_games.db

# Check for games with zero moves
python -c "
import sqlite3
conn = sqlite3.connect('data/games/canonical_hex8_4p.db')
result = conn.execute('''
    SELECT game_id, total_moves FROM games
    WHERE game_status = 'completed' AND total_moves = 0
''').fetchall()
print(f'Games with 0 moves: {len(result)}')
"
```

### Method 3: Export Failure Detection

```bash
# Export will fail with corrupted games
python scripts/export_replay_dataset.py \
  --db data/games/canonical_hex8_4p.db \
  --board-type hex8 --num-players 4 \
  --output /tmp/test_export.npz

# Check for specific error patterns:
# - "Move to=None" - Phase tracking bug
# - "Invalid move" - Serialization error
# - "State mismatch" - Rule engine parity issue
```

---

## Diagnosis

### Identify Corruption Pattern

```python
import sqlite3
import json

conn = sqlite3.connect("data/games/canonical_hex8_4p.db")

# Find games with replay failures
games = conn.execute("""
    SELECT game_id, moves_json
    FROM games
    WHERE game_status = 'completed'
    LIMIT 10
""").fetchall()

for game_id, moves_json in games:
    moves = json.loads(moves_json) if moves_json else []
    for i, move in enumerate(moves):
        if move.get("type") == "PLACE_RING" and move.get("to") is None:
            print(f"CORRUPT: {game_id} move {i}: to=None")
            break
        if move.get("type") == "PLACE" and move.get("position") is None:
            print(f"CORRUPT: {game_id} move {i}: position=None")
            break
```

### Common Corruption Patterns

| Pattern               | Symptom                                | Root Cause                           |
| --------------------- | -------------------------------------- | ------------------------------------ |
| `to=None`             | PLACE_RING moves have null destination | Phase extracted from post-move state |
| `position=None`       | PLACE moves missing position           | Terminal game phase serialization    |
| `Invalid chain`       | Chain capture fails                    | FSM state mismatch                   |
| `Board hash mismatch` | Hash differs from expected             | State not properly cloned            |

---

## Recovery Procedures

### Option 1: Quarantine Corrupted Games

Best for small numbers of corrupted games.

```bash
# Create quarantine table
sqlite3 data/games/canonical_hex8_4p.db <<EOF
CREATE TABLE IF NOT EXISTS orphaned_games (
    game_id TEXT PRIMARY KEY,
    quarantine_reason TEXT,
    quarantine_time TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Move corrupted games to quarantine
INSERT INTO orphaned_games (game_id, quarantine_reason)
SELECT game_id, 'Corrupted move data: to=None'
FROM games
WHERE moves_json LIKE '%"to": null%';

-- Delete from main table
DELETE FROM games
WHERE game_id IN (SELECT game_id FROM orphaned_games);
EOF
```

### Option 2: Regenerate Database

Best for heavily corrupted databases (>10% failure rate).

```bash
# 1. Backup the corrupted database
mv data/games/canonical_hex8_4p.db data/games/canonical_hex8_4p.db.corrupt

# 2. Create fresh database
python scripts/selfplay.py \
  --board hex8 --num-players 4 \
  --num-games 1000 --engine heuristic \
  --output-dir data/games/hex8_4p_fresh

# 3. Dispatch cluster-wide regeneration
curl -X POST http://leader-ip:8770/dispatch_selfplay \
  -H "Content-Type: application/json" \
  -d '{"config": "hex8_4p", "games": 500, "engine": "heuristic"}'
```

### Option 3: Partial Recovery

Extract valid games from corrupted database.

```python
import sqlite3
import json

source = sqlite3.connect("data/games/canonical_hex8_4p.db.corrupt")
dest = sqlite3.connect("data/games/canonical_hex8_4p.db")

# Copy schema
dest.executescript("""
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        -- ... copy full schema
    );
""")

# Copy only valid games
valid_count = 0
for row in source.execute("SELECT * FROM games"):
    game_id = row[0]
    moves_json = row[4]  # Adjust index

    # Validate moves
    moves = json.loads(moves_json) if moves_json else []
    is_valid = all(
        move.get("to") is not None or move.get("type") != "PLACE_RING"
        for move in moves
    )

    if is_valid:
        dest.execute("INSERT INTO games VALUES (?, ?, ...)", row)
        valid_count += 1

dest.commit()
print(f"Recovered {valid_count} valid games")
```

---

## Prevention

### 1. Pre-Commit Validation

```python
# In GameWriter.finalize()
def finalize(self):
    # Validate all moves before committing
    for move in self.moves:
        if move.type == "PLACE_RING" and move.to is None:
            raise InvalidGameError(f"Move has null destination: {move}")

    self._commit()
```

### 2. Parity Gate

Run parity validation before training:

```bash
# Require parity pass before export
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8_4p.db \
  --fail-on-mismatch
```

### 3. IntegrityCheckDaemon

Enable automatic integrity scanning:

```bash
export RINGRIFT_INTEGRITY_ENABLED=true
export RINGRIFT_INTEGRITY_CHECK_INTERVAL=3600  # 1 hour
```

---

## Post-Recovery Validation

After recovery, verify data quality:

```bash
# 1. Check game count
sqlite3 data/games/canonical_hex8_4p.db "SELECT COUNT(*) FROM games"

# 2. Run parity check
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8_4p.db

# 3. Test export
python scripts/export_replay_dataset.py \
  --db data/games/canonical_hex8_4p.db \
  --board-type hex8 --num-players 4 \
  --output /tmp/test_hex8_4p.npz

# 4. Validate training data
python -m app.training.data_quality --npz /tmp/test_hex8_4p.npz
```

---

## Related Documentation

- [PARITY_MISMATCH_DEBUG.md](PARITY_MISMATCH_DEBUG.md) - TS/Python parity issues
- [CLUSTER_SYNCHRONIZATION.md](CLUSTER_SYNCHRONIZATION.md) - Data sync procedures
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General debugging
