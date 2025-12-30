# RingRift AI Service Onboarding Roadmap

A structured guide for developers new to the RingRift AI training infrastructure.

**Created**: December 28, 2025
**Last Updated**: December 28, 2025

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.10+ installed
- [ ] Git access to the RingRift repository
- [ ] Basic understanding of neural networks (helpful, not required)
- [ ] Familiarity with command-line tools
- [ ] ~10GB disk space for models and training data

**Optional but recommended:**

- [ ] CUDA-capable GPU (training runs faster)
- [ ] SSH key configured for cluster access (for intermediate/advanced paths)
- [ ] Tailscale installed (for P2P cluster operations)

---

## Learning Paths

Choose a path based on your goals and time:

| Path                                       | Time    | Outcome                              |
| ------------------------------------------ | ------- | ------------------------------------ |
| [Beginner](#beginner-path-30-minutes)      | 30 min  | Run selfplay, understand game rules  |
| [Intermediate](#intermediate-path-2-hours) | 2 hours | Debug parity issues, add unit tests  |
| [Advanced](#advanced-path-1-day)           | 1 day   | Distributed training, add new daemon |

---

## Beginner Path (30 minutes)

**Goal**: Run local selfplay and understand the basic game engine.

### Step 1: Environment Setup (5 min)

```bash
cd ai-service

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from app.rules.ring_rift import RingRiftGame; print('OK')"
```

### Step 2: Understand the Game (10 min)

RingRift is a territory control game:

- Players take turns placing pieces on a board
- Goal: Control the most territory when the game ends
- Multiple board types: `square8`, `hex8`, `square19`, `hexagonal`
- 2-4 player support

**Play a quick game visually:**

```bash
# Interactive CLI game (2 players, small board)
python -c "
from app.rules.ring_rift import RingRiftGame

# Create game
game = RingRiftGame('square8', num_players=2)

# Print initial board
game.print_board()
print(f'Current player: {game.current_player}')
print(f'Valid moves: {len(game.get_valid_moves())}')
"
```

### Step 3: Run Selfplay (10 min)

Generate training data by having the AI play against itself:

```bash
# Quick heuristic selfplay (fast, uses rule-based AI)
python scripts/selfplay.py \
  --board square8 \
  --num-players 2 \
  --engine heuristic \
  --num-games 10

# Check the output
ls -la data/games/
```

**Output**: A SQLite database with completed games.

### Step 4: Inspect Game Data (5 min)

```bash
# Check game count
python -c "
from app.utils.game_discovery import GameDiscovery
d = GameDiscovery()
for db in d.find_all_databases():
    print(f'{db.board_type}_{db.num_players}p: {db.game_count} games')
"
```

**Checkpoint**: You should now understand:

- The game mechanics (place pieces, control territory)
- How selfplay generates training data
- Where game databases are stored (`data/games/`)

---

## Intermediate Path (2 hours)

**Goal**: Debug a parity issue and add unit tests.

**Prerequisites**: Complete Beginner path or equivalent familiarity.

### Step 1: Understand Parity (20 min)

The Python game engine must exactly match the TypeScript engine:

```
TypeScript (src/shared/engine/) = Source of Truth
Python (ai-service/app/rules/)  = Must Mirror TS
```

**Why parity matters:**

- Games played in browser use TS engine
- Training data comes from Python engine
- If they disagree, models learn wrong patterns

**Check parity status:**

```bash
# Run parity test (requires Node.js for TS execution)
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_square8_2p.db \
  --limit 10
```

### Step 2: Read the Parity Debug Guide (15 min)

Read these docs to understand parity debugging:

1. `docs/runbooks/PARITY_MISMATCH_DEBUG.md` - Step-by-step debugging
2. `docs/runbooks/HEXAGONAL_PARITY_BUG.md` - Example of a real parity bug fix

### Step 3: Add a Unit Test (45 min)

Find an untested module and add tests:

```bash
# Find modules without tests
find app/coordination -name "*.py" -exec basename {} \; | \
  sort | uniq | \
  while read f; do
    if [ ! -f "tests/unit/coordination/test_$f" ]; then
      echo "Missing: test_$f"
    fi
  done
```

**Example: Adding tests for a utility function**

```python
# tests/unit/coordination/test_example_module.py
import pytest
from app.coordination.example_module import some_function

class TestSomeFunction:
    def test_basic_case(self):
        result = some_function("input")
        assert result == "expected"

    def test_edge_case(self):
        with pytest.raises(ValueError):
            some_function(None)

    def test_empty_input(self):
        result = some_function("")
        assert result == ""
```

**Run your tests:**

```bash
pytest tests/unit/coordination/test_example_module.py -v
```

### Step 4: Debug a Failing Test (30 min)

Run the full test suite and fix any failures:

```bash
# Run all coordination tests
pytest tests/unit/coordination/ -v --tb=short

# Run with coverage
pytest tests/unit/coordination/ --cov=app/coordination --cov-report=html
```

### Step 5: Understand the Event System (10 min)

The coordination layer uses events for communication:

```python
from app.coordination.event_router import get_event_bus
from app.distributed.data_events import DataEventType

# Subscribe to events
bus = get_event_bus()
bus.subscribe(DataEventType.TRAINING_COMPLETED, my_handler)

# Emit events
from app.coordination.event_emitters import emit_training_complete
emit_training_complete(config_key="hex8_2p", model_path="...")
```

**Checkpoint**: You should now understand:

- How parity testing works
- The test file naming convention
- Basic event system usage

---

## Advanced Path (1 day)

**Goal**: Set up distributed training and add a new daemon.

**Prerequisites**: Complete Intermediate path, cluster access.

### Step 1: Cluster Setup (1 hour)

**Get cluster access:**

```bash
# Check your SSH key works
ssh -i ~/.ssh/id_cluster ubuntu@nebius-backbone-1 "hostname"

# Verify P2P status (if leader is running)
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive peers: {d.get(\"alive_peers\")}")
'
```

**Understand cluster configuration:**

```bash
# Read the cluster config
cat config/distributed_hosts.yaml

# Key sections:
# - nodes: All cluster nodes with connection info
# - p2p_voters: Nodes that participate in leader election
# - sync: Data synchronization configuration
```

### Step 2: Run Distributed Selfplay (1 hour)

**Start the master loop** (main automation entry point):

```bash
# Dry run first to see what would happen
python scripts/master_loop.py --dry-run

# Watch mode to monitor without running
python scripts/master_loop.py --watch

# Full automation (runs indefinitely)
python scripts/master_loop.py
```

**Dispatch individual jobs via P2P:**

```bash
# Check P2P leader
curl -s http://localhost:8770/status | jq .leader_id

# If this node is leader, dispatch selfplay
curl -X POST http://localhost:8770/dispatch_selfplay \
  -H "Content-Type: application/json" \
  -d '{"board_type": "hex8", "num_players": 2, "num_games": 100}'
```

### Step 3: Understand the Daemon System (2 hours)

**Read key architecture docs:**

1. `CLAUDE.md` - Full AI service context
2. `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Daemon operations
3. `app/coordination/daemon_registry.py` - All 85 daemon types

**Key daemon concepts:**

```python
from app.coordination.daemon_types import DaemonType
from app.coordination.daemon_manager import get_daemon_manager

# Get singleton daemon manager
dm = get_daemon_manager()

# Start a specific daemon
await dm.start(DaemonType.AUTO_SYNC)

# Check daemon health
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
print(f"Status: {health}")

# List all daemons
for dtype in DaemonType:
    print(f"{dtype.name}")
```

### Step 4: Add a New Daemon (3 hours)

**Step 4.1: Define the daemon type**

```python
# app/coordination/daemon_types.py
class DaemonType(str, Enum):
    # ... existing types ...
    MY_NEW_DAEMON = "my_new_daemon"
```

**Step 4.2: Create the daemon runner**

```python
# app/coordination/daemon_runners.py

async def create_my_new_daemon() -> None:
    """Run the MyNewDaemon until shutdown."""
    from app.coordination.my_new_daemon import MyNewDaemon

    daemon = MyNewDaemon()
    await daemon.start()

    # Wait for shutdown signal
    while daemon.is_running:
        await asyncio.sleep(1)
```

**Step 4.3: Register in daemon_registry.py**

```python
# app/coordination/daemon_registry.py

DAEMON_REGISTRY: Dict[DaemonType, DaemonSpec] = {
    # ... existing entries ...
    DaemonType.MY_NEW_DAEMON: DaemonSpec(
        runner_name="create_my_new_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
        auto_restart=True,
        health_check_interval=30.0,
    ),
}
```

**Step 4.4: Implement the daemon**

```python
# app/coordination/my_new_daemon.py
from app.coordination.handler_base import HandlerBase, HealthCheckResult

class MyNewDaemon(HandlerBase):
    """Example daemon that does periodic work."""

    def __init__(self):
        super().__init__(name="my_new_daemon", cycle_interval=60.0)
        self._processed_count = 0

    async def _run_cycle(self) -> None:
        """Main work loop - called every cycle_interval seconds."""
        # Do your daemon's work here
        self._processed_count += 1
        self._log_info(f"Processed cycle {self._processed_count}")

    def _get_event_subscriptions(self) -> dict:
        """Subscribe to events that trigger work."""
        return {
            "SOME_EVENT": self._on_some_event,
        }

    async def _on_some_event(self, event: dict) -> None:
        """Handle incoming events."""
        self._log_info(f"Received event: {event}")

    def health_check(self) -> HealthCheckResult:
        """Report daemon health for monitoring."""
        return HealthCheckResult(
            healthy=self._is_running,
            message="OK" if self._is_running else "Not running",
            details={
                "processed_count": self._processed_count,
                "cycle_interval": self._cycle_interval,
            },
        )

# Module-level singleton accessor
_instance: Optional[MyNewDaemon] = None

def get_my_new_daemon() -> MyNewDaemon:
    global _instance
    if _instance is None:
        _instance = MyNewDaemon()
    return _instance
```

**Step 4.5: Add tests**

```python
# tests/unit/coordination/test_my_new_daemon.py
import pytest
from app.coordination.my_new_daemon import MyNewDaemon, get_my_new_daemon

class TestMyNewDaemon:
    def test_initialization(self):
        daemon = MyNewDaemon()
        assert daemon.name == "my_new_daemon"
        assert daemon._cycle_interval == 60.0

    def test_health_check_not_running(self):
        daemon = MyNewDaemon()
        health = daemon.health_check()
        assert not health.healthy

    @pytest.mark.asyncio
    async def test_run_cycle(self):
        daemon = MyNewDaemon()
        await daemon._run_cycle()
        assert daemon._processed_count == 1

    def test_singleton(self):
        d1 = get_my_new_daemon()
        d2 = get_my_new_daemon()
        assert d1 is d2
```

### Step 5: Training Pipeline Deep Dive (1 hour)

**Understand the full pipeline:**

```
Selfplay → Game DB → Export → NPZ → Train → Evaluate → Promote
```

**Key coordinators:**

| Stage      | Coordinator                | Event                  |
| ---------- | -------------------------- | ---------------------- |
| Selfplay   | `SelfplayScheduler`        | `SELFPLAY_COMPLETE`    |
| Export     | `DataPipelineOrchestrator` | `NEW_GAMES_AVAILABLE`  |
| Training   | `TrainingCoordinator`      | `TRAINING_COMPLETED`   |
| Evaluation | `EvaluationDaemon`         | `EVALUATION_COMPLETED` |
| Promotion  | `AutoPromotionDaemon`      | `MODEL_PROMOTED`       |

**Watch events flow:**

```bash
# Enable debug logging
export RINGRIFT_LOG_LEVEL=DEBUG

# Run master loop and watch events
python scripts/master_loop.py --watch
```

**Checkpoint**: You should now understand:

- Cluster architecture and P2P communication
- The full daemon lifecycle
- How to add new coordinators to the system

---

## Quick Reference

### Essential Commands

```bash
# Selfplay
python scripts/selfplay.py --board hex8 --num-players 2 --engine gumbel --num-games 100

# Export training data
python scripts/export_replay_dataset.py --use-discovery --board-type hex8 --num-players 2 --output data/training/hex8_2p.npz

# Train
python -m app.training.train --board-type hex8 --num-players 2 --data-path data/training/hex8_2p.npz

# Check parity
python scripts/check_ts_python_replay_parity.py --db data/games/my_games.db

# Run tests
pytest tests/unit/coordination/ -v

# Master loop (cluster automation)
python scripts/master_loop.py --watch
```

### Essential Files

| File                                 | Purpose                         |
| ------------------------------------ | ------------------------------- |
| `CLAUDE.md`                          | AI service context and commands |
| `app/coordination/daemon_manager.py` | Daemon lifecycle                |
| `app/coordination/event_router.py`   | Event system                    |
| `app/rules/ring_rift.py`             | Game engine                     |
| `config/distributed_hosts.yaml`      | Cluster config                  |

### Environment Variables

| Variable                         | Default | Purpose                         |
| -------------------------------- | ------- | ------------------------------- |
| `RINGRIFT_LOG_LEVEL`             | INFO    | Logging verbosity               |
| `RINGRIFT_DATA_DIR`              | data    | Data directory                  |
| `RINGRIFT_ALLOW_PENDING_GATE`    | false   | Allow training without parity   |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | true    | Skip shadow contract validation |

See `ENV_REFERENCE.md` and `ENV_REFERENCE_COMPREHENSIVE.md` for complete reference.

---

## Getting Help

1. **Documentation**: Start with `CLAUDE.md` for full context
2. **Runbooks**: See `docs/runbooks/MASTER_RUNBOOK_INDEX.md` for operational guides
3. **Tests**: Look at existing tests for usage examples
4. **Code Comments**: Most modules have detailed docstrings

---

## Next Steps After Onboarding

Depending on your role:

| Role           | Focus Area                                                    |
| -------------- | ------------------------------------------------------------- |
| ML Engineer    | Model architectures in `app/ai/`, training in `app/training/` |
| Infrastructure | Daemons in `app/coordination/`, P2P in `scripts/p2p/`         |
| Game Developer | Rules in `app/rules/`, parity with TypeScript                 |
| DevOps         | Cluster config, runbooks, monitoring                          |

---

## Appendix: File Structure

```
ai-service/
├── app/
│   ├── ai/              # Neural networks, MCTS, search
│   ├── config/          # Centralized configuration
│   ├── coordination/    # Training pipeline orchestration
│   ├── core/            # Core utilities (SSH, logging)
│   ├── db/              # Database utilities
│   ├── distributed/     # Cluster tools
│   ├── rules/           # Game engine (mirrors TS)
│   ├── training/        # Training pipeline
│   └── utils/           # Utilities
├── config/              # YAML configuration files
├── data/
│   ├── games/           # Game databases
│   └── training/        # NPZ training files
├── docs/                # Documentation
│   └── runbooks/        # Operational procedures
├── models/              # Trained model checkpoints
├── scripts/             # CLI tools
│   └── p2p/             # P2P orchestrator components
└── tests/               # Test suite
    └── unit/
        └── coordination/  # Coordination tests
```
