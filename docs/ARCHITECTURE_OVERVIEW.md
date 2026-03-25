# RingRift Architecture Overview

**Recreating AlphaZero in 6-8 weeks: One non-programmer, a laptop, and Claude Code**

A from-scratch recreation of DeepMind's AlphaZero training pipeline, built to train AI for a new abstract strategy game designed as a long-term testbed for AI research.

**The question:** Can one non-technical person, 8 years after AlphaZero, replicate what required hundreds of DeepMind engineers?

**The answer:** Yes, with AI-assisted development. The AI can also train itself to play worse than a simple heuristic. Be careful what you wish for.

---

## System at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RINGRIFT SYSTEM                                 │
│                    (Enterprise-grade infrastructure for 0 users)             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐   │
│  │   WEB APP   │     │  GAME API   │     │      AI SERVICE (Python)    │   │
│  │   (React)   │────▶│  (Node.js)  │────▶│                             │   │
│  │             │     │             │     │  ┌─────────────────────────┐│   │
│  └─────────────┘     └─────────────┘     │  │   Neural Network AI    ││   │
│        │                   │             │  │   (currently losing)   ││   │
│        │              WebSocket          │  └─────────────────────────┘│   │
│        ▼                   │             │              │              │   │
│  ┌─────────────┐          │             │              ▼              │   │
│  │   Browser   │◀─────────┘             │  ┌─────────────────────────┐│   │
│  │   Client    │                        │  │   Gumbel MCTS Search   ││   │
│  │  (0 users)  │                        │  │   (GPU Accelerated)    ││   │
│  └─────────────┘                        │  └─────────────────────────┘│   │
│                                         └─────────────────────────────┘   │
│                                                       │                    │
│                                                       ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    P2P TRAINING CLUSTER                              │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │
│  │  │ Lambda  │ │ Vast.ai │ │ RunPod  │ │ Nebius  │ │ Hetzner │       │  │
│  │  │ GH200   │ │ RTX5090 │ │  H100   │ │  H100   │ │  (CPU)  │       │  │
│  │  │ x11     │ │  x14    │ │  x6     │ │  x3     │ │  x3     │       │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │  │
│  │       │           │           │           │           │             │  │
│  │       └───────────┴───────────┴───────────┴───────────┘             │  │
│  │                           │                                          │  │
│  │                    P2P Mesh Network                                  │  │
│  │       (Leader Election, Gossip, SWIM, Things I Don't Understand)     │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Metrics (A Study in Misplaced Priorities)

| Category           | Metric               | Value            | Commentary                                        |
| ------------------ | -------------------- | ---------------- | ------------------------------------------------- |
| **Codebase**       | Python modules       | 341              | For 0 users                                       |
|                    | Test files           | 984              | Each one a monument                               |
|                    | Test coverage        | 99.5%            | Tests pass; AI still regresses                    |
|                    | Lines of code        | ~250,000         | ~250,000 ways to fail                             |
| **Infrastructure** | Daemon types         | 132 (116 active) | Plus 16 deprecated but never deleted              |
|                    | Recovery daemons     | 11               | To recover from the other 121                     |
|                    | Event types          | 292              | I don't know what half do                         |
|                    | P2P loops            | 22               | Running 24/7                                      |
| **Cluster**        | Total nodes          | ~41              | Overkill for 0 users                              |
|                    | GPU memory           | ~1.5 TB          | Burning $$$                                       |
|                    | Providers            | 7                | Lambda, Vast, RunPod, Nebius, Vultr, Hetzner, AWS |
|                    | Autonomous runtime   | 48+ hours        | Can regress autonomously too                      |
| **Models**         | Trained configs      | 12               | All board/player combos                           |
|                    | Best model Elo       | 1141             | Lower than heuristic (1200)                       |
|                    | GPU selfplay speedup | 57x              | Can generate bad data faster                      |

---

## The Game: RingRift

### The Design Philosophy

Before training AlphaZero, you need something worth training it on. Chess and Go are "solved" from a research perspective—we know the approach works on them. RingRift was designed from scratch as a long-term AI research testbed.

**The vision:**

> _"A cosmos to cleft. Structure birthing emergency."_
>
> _"A game about the tension between freedom and boundaries — between containment and creativity."_
>
> _"Tradeoffs and sacrifices intrinsic — strength and weakness interlaced, implying one another all the way through."_

The goal: a game where complexity emerges from simple rules, where seeming victory can collapse into defeat, where short-term sacrifice is sometimes correct, and where the board geometry constantly reshapes incentives.

### Why a New Game? (The Human-Competitive Thesis)

Unlike most perfect-information strategy games where engines quickly outclass humans, RingRift is explicitly designed so that **strong human players can compete with and sometimes outplay strong AIs**:

| Challenge for AI             | Why It Helps Humans                                                                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Multi-player dynamics**    | 3-4 player games create coalition dynamics (temporary alliances, leader-punishing) that reward reading opponents—something humans do naturally |
| **Extreme branching factor** | Millions of legal choices per turn (vs Chess ~35, Go ~250). Makes exhaustive search prohibitively expensive                                    |
| **Long tactical chains**     | Captures cascade through territory disconnections. Requires genuine lookahead, not just pattern matching                                       |
| **Multiple victory paths**   | Ring elimination, territory control, and last-standing interact. Not reducible to single value function                                        |

### The Research Experiment

The Human-Competitive Thesis above is not just a design goal — it's a testable scientific hypothesis. RingRift is simultaneously a game AND a research experiment with five interconnected questions:

1. **Can a non-developer design a genuinely ML-resistant game?** The properties above (multi-player dynamics, extreme branching, long tactical chains, multiple victory paths) are _predictions_ about what makes games hard for neural networks. If 1.9M training games and hundreds of GPU-hours fail to produce superhuman play, these predictions are validated — an interesting result for game design and AI safety.

2. **Can frontier AI models help a non-developer build AlphaZero?** This entire infrastructure — the P2P mesh, 132 daemons, Gumbel MCTS, distributed training — was built by a solo non-developer using Claude and GPT as development partners. The system's existence (and its bugs) is data for this hypothesis.

3. **Does the self-play loop produce improvement?** As of March 2026, 171 training iterations produced regression rather than improvement due to silent pipeline bugs (encoding mismatches, disabled validation gates, model corruption). After fixes, the pipeline is running correctly for the first time. The answer to this question determines whether Hypothesis 1 or 2 is the more interesting finding.

4. **Can the results be communicated compellingly?** Regardless of AI strength outcome, the project generates narratives: solo-developer-builds-AlphaZero, ML-resistant-game-design, or debugging-132-daemons-for-$20K-of-GPU.

5. **Can the experiment become a product?** The game, web client, and infrastructure are deployed at ringrift.ai. The research narrative serves as the content strategy for attracting initial players.

See [PROJECT_GOALS.md](../PROJECT_GOALS.md) Section 2.2 for the authoritative framing of these hypotheses.

### Uniqueness Analysis: ~65-70% Novel

RingRift is a "chimeric design" synthesizing mechanics from multiple game families:

| Influence                              | What RingRift Borrowed                     |
| -------------------------------------- | ------------------------------------------ |
| **GIPF Project** (YINSH, DVONN, TZAAR) | Stacking, line formation, marker mechanics |
| **Go**                                 | Territory enclosure, pie rule              |
| **Checkers/Fanorona**                  | Mandatory chain capture continuation       |
| **Amazons**                            | Movement leaves permanent markers          |

But the **combination creates genuinely novel decision spaces**:

| Novel Mechanic                     | Why It's Unique                                                                                            |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Cap height vs stack height**     | No other game distinguishes "consecutive top rings of your color" from "total rings" for capture           |
| **Marker dual behavior**           | YINSH only flips; Amazons only burns. RingRift: opponent markers flip, own markers collapse to territory   |
| **Line → territory + elimination** | YINSH removes line but no territory cost. RingRift creates permanent territory AND forces self-elimination |
| **Three integrated victory paths** | Most abstracts have single/binary conditions. Three paths that interact                                    |
| **Native 3-4 player support**      | Rare in serious abstract strategy. Creates coalition dynamics                                              |

### AI Research Testbed

Beyond the game, RingRift is designed as a testbed for future AI algorithms:

| Algorithm Family          | Why RingRift Is Interesting                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| **Graph Neural Networks** | Board topology (especially hex) maps naturally to GNNs. Stacks = nodes, adjacency = edges      |
| **Transformers**          | Multi-player dynamics could model coalition formation via attention mechanisms                 |
| **Coalition modeling**    | 3-4 player variants require modeling temporary alliances—current MCTS assumes adversarial play |
| **Long-horizon planning** | Territory cascades ripple across board. Requires genuine lookahead                             |
| **Transfer learning**     | Four board geometries test generalization. Does hex8 transfer to hexagonal?                    |

### Board Configurations

| Board Type | Grid      | Cells | Players | Rings/Player |
| ---------- | --------- | ----- | ------- | ------------ |
| square8    | 8×8       | 64    | 2, 3, 4 | 18           |
| square19   | 19×19     | 361   | 2, 3, 4 | 72           |
| hex8       | radius 4  | 61    | 2, 3, 4 | 18           |
| hexagonal  | radius 12 | 469   | 2, 3, 4 | 96           |

**Key constraint:** TypeScript is the source of truth for game rules. Python must maintain 100% parity for training data to be valid.

**Current status:** The game is actually quite fun to play. Allegedly. We have no users to confirm this. The AI has also successfully trained itself to play worse than the heuristic fallback.

---

## Core Architecture Components

### 1. Dual-Language Game Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAME ENGINE PARITY                            │
│              (The one thing that actually works)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TypeScript (src/shared/engine/)                                │
│  ════════════════════════════════                               │
│  • Source of truth for rules                                    │
│  • Runs in browser and Node.js                                  │
│  • Used by game server for validation                           │
│                                                                 │
│           ▲                                                     │
│           │ Parity Tests (10K seeds)                            │
│           │ (Verified working)                                  │
│           ▼                                                     │
│                                                                 │
│  Python (ai-service/app/rules/)                                 │
│  ══════════════════════════════                                 │
│  • Mirror implementation for training                           │
│  • GPU-accelerated for selfplay                                 │
│  • Must produce identical game states                           │
│  • Does produce identical game states                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. P2P Orchestrator (The Heart — I Don't Fully Understand It)

**File:** `scripts/p2p_orchestrator.py` (~26,000 LOC)

```
┌─────────────────────────────────────────────────────────────────┐
│                   P2P ORCHESTRATOR                               │
│           (I built this and I'm not sure how it works)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   State     │  │    Job      │  │  Training   │             │
│  │  Manager    │  │  Manager    │  │ Coordinator │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐             │
│  │   Sync      │  │  Selfplay   │  │    Node     │             │
│  │  Planner    │  │ Scheduler   │  │  Selector   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                   ┌──────┴──────┐                               │
│                   │    Loop     │                               │
│                   │   Manager   │                               │
│                   └─────────────┘                               │
│                          │                                      │
│              ┌───────────┼───────────┐                          │
│              ▼           ▼           ▼                          │
│         22 Background Loops                                     │
│         (Elo sync, job reaping, health, etc.)                   │
│         (Running 24/7 whether or not AI is improving)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**7 Extracted Managers:**

- StateManager: Cluster state, peer tracking
- JobManager: Job dispatch, completion handling
- TrainingCoordinator: Training job lifecycle
- SyncPlanner: Data synchronization planning (sometimes doesn't sync)
- SelfplayScheduler: Priority-based selfplay allocation
- NodeSelector: GPU node selection for jobs
- LoopManager: Background loop lifecycle

### 3. Daemon Architecture

**132 daemon types** managed by a unified lifecycle system:

```
┌─────────────────────────────────────────────────────────────────┐
│                   DAEMON ARCHITECTURE                            │
│      (132 daemons to recover from 0 users, 11 to recover        │
│       from the other 121 daemons' failures)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  daemon_registry.py                                             │
│  ══════════════════                                             │
│  • Declarative daemon specifications                            │
│  • DaemonSpec dataclass                                         │
│  • Categories: sync, pipeline, health, resources                │
│                                                                 │
│           │                                                     │
│           ▼                                                     │
│                                                                 │
│  daemon_manager.py                                              │
│  ═════════════════                                              │
│  • Lifecycle: start, stop, restart                              │
│  • Health monitoring loop                                       │
│  • Auto-restart on failure                                      │
│  • Dependency ordering                                          │
│                                                                 │
│           │                                                     │
│           ▼                                                     │
│                                                                 │
│  daemon_runners.py                                              │
│  ═════════════════                                              │
│  • 124 async runner functions                                   │
│  • Factory pattern for instantiation                            │
│  • Integration with HandlerBase                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Daemon Categories:**

| Category  | Examples                                | Purpose             | Status                       |
| --------- | --------------------------------------- | ------------------- | ---------------------------- |
| Sync      | AUTO_SYNC, ELO_SYNC, MODEL_DISTRIBUTION | Data movement       | Sometimes works              |
| Pipeline  | DATA_PIPELINE, TRAINING_TRIGGER         | Training workflow   | Broken for 2 weeks           |
| Health    | NODE_HEALTH, QUALITY_MONITOR            | Monitoring          | Monitors but doesn't prevent |
| Recovery  | P2P_RECOVERY, PROGRESS_WATCHDOG         | Self-healing        | 11 daemons, still regressed  |
| Resources | IDLE_RESOURCE, MEMORY_MONITOR           | Resource management | Works                        |

### 4. Event-Driven Coordination

**292 event types** flowing through a unified event system:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT SYSTEM                                  │
│            (292 event types, mostly working)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  event_router.py                                                │
│  ═══════════════                                                │
│  • SHA256 deduplication                                         │
│  • Dead letter queue for failed events                          │
│  • Timeout protection                                           │
│  • Degradation mode                                             │
│                                                                 │
│  Key Event Flows:                                               │
│  ────────────────                                               │
│                                                                 │
│  Selfplay ──▶ NEW_GAMES_AVAILABLE                               │
│                      │                                          │
│                      ▼                                          │
│  DataPipeline ──▶ TRAINING_THRESHOLD_REACHED                    │
│                      │                                          │
│                      ▼                                          │
│  Training ──▶ TRAINING_COMPLETED                                │
│                      │        (but was it good training?)       │
│                      ▼                                          │
│  Evaluation ──▶ EVALUATION_COMPLETED                            │
│                      │        (Elo went down)                   │
│                      ▼                                          │
│  Promotion ──▶ MODEL_PROMOTED                                   │
│                      │        (promoted a worse model)          │
│                      ▼                                          │
│  Distribution ──▶ (sync to all nodes)                           │
│                               (distributed the worse model)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Training Pipeline

**7-stage pipeline** with 5 feedback loops (some of which work):

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                               │
│           (Works great, AI still got worse)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: SELFPLAY                                              │
│  ─────────────────                                              │
│  • GPU-accelerated game generation                              │
│  • Gumbel MCTS for move selection                               │
│  • Priority based on staleness, Elo velocity, quality           │
│  • 57x faster than CPU (can generate bad data 57x faster)       │
│                                                                 │
│           │                                                     │
│           ▼                                                     │
│                                                                 │
│  Stage 2: EXPORT                                                │
│  ────────────────                                               │
│  • Convert games to training samples                            │
│  • Feature extraction (board state, legal moves)                │
│  • NPZ file generation                                          │
│  • NPZ files sometimes don't get synced to GPU nodes            │
│                                                                 │
│           │                                                     │
│           ▼                                                     │
│                                                                 │
│  Stage 3: TRAINING                                              │
│  ─────────────────                                              │
│  • PyTorch training loop                                        │
│  • Early stopping with adaptive patience                        │
│  • Checkpoint averaging                                         │
│  • Anomaly detector that flags normal data as anomalous         │
│                                                                 │
│           │                                                     │
│           ▼                                                     │
│                                                                 │
│  Stage 4: EVALUATION                                            │
│  ────────────────────                                           │
│  • Gauntlet vs baselines (Random, Heuristic)                    │
│  • Win rate thresholds for promotion                            │
│  • Elo calculation                                              │
│  • Correctly identifies that models got worse                   │
│                                                                 │
│           │                                                     │
│           ▼                                                     │
│                                                                 │
│  Stage 5: PROMOTION                                             │
│  ─────────────────                                              │
│  • Model becomes new canonical                                  │
│  • Triggers distribution to cluster                             │
│  • Updates curriculum weights                                   │
│  • Will promote models that are worse than heuristic            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

FEEDBACK LOOPS (Theoretical):
══════════════════════════════
1. Quality → Training: Low quality blocks training (didn't block)
2. Elo Velocity → Selfplay: Fast-improving configs get more games (velocity was negative)
3. Regression → Curriculum: Detected regressions adjust priorities (detected, didn't fix)
4. Loss Anomaly → Exploration: Unusual losses trigger exploration (triggered on normal data)
5. Promotion → Distribution: New models sync across cluster (synced the worse models)
```

### 6. Resilience Architecture

**4-layer resilience** for 48-hour autonomous operation (and autonomous regression):

```
┌─────────────────────────────────────────────────────────────────┐
│                 RESILIENCE LAYERS                                │
│      (The AI regressed for 2 weeks despite all this)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: OS-Level Supervision                                  │
│  ═════════════════════════════                                  │
│  • launchd (macOS) / systemd (Linux)                            │
│  • Sentinel daemon, Watchdog                                    │
│  • Process restart on crash                                     │
│  • Works great                                                  │
│                                                                 │
│  Layer 2: Memory Pressure Control                               │
│  ═══════════════════════════════                                │
│  • 4 tiers: 60% → 70% → 80% → 90%                               │
│  • Progressive response: warn → pause → kill → shutdown         │
│  • GPU VRAM monitoring                                          │
│  • Also works great                                             │
│                                                                 │
│  Layer 3: Coordinator Failover                                  │
│  ═════════════════════════════                                  │
│  • Primary/standby coordinator pattern                          │
│  • LeaderProbeLoop: 10s probes, 60s failover                    │
│  • Quorum health levels: HEALTHY/DEGRADED/MINIMUM/LOST          │
│  • Failover tested and working                                  │
│                                                                 │
│  Layer 4: Cluster Health Aggregation                            │
│  ════════════════════════════════════                           │
│  • 257 health check mechanisms                                  │
│  • Circuit breakers with 4-tier escalation                      │
│  • Multi-transport failover                                     │
│  • All the infrastructure, none of the insight                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Circuit Breaker Types (9):**

- Operation circuit breakers
- Transport circuit breakers
- Node circuit breakers
- Per-host breakers
- Per-operation breakers
- Cascade prevention
- TTL decay (prevents permanent exclusion)

**What they protect against:** Node failures, network issues, memory pressure

**What they don't protect against:** Training on stale data for 2 weeks

---

## GPU Selfplay Optimization

The journey from slow to 57x speedup (can now generate bad data much faster):

```
┌─────────────────────────────────────────────────────────────────┐
│              GPU SELFPLAY OPTIMIZATION                           │
│       (This part actually works as intended)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BEFORE: 80 .item() calls                                       │
│  ═══════════════════════                                        │
│  • Each .item() = GPU→CPU round trip                            │
│  • ~100ms per game                                              │
│  • CPU bottlenecked                                             │
│                                                                 │
│  AFTER: 1 .item() call (statistics only)                        │
│  ═══════════════════════════════════════                        │
│  • Fully vectorized move generation                             │
│  • Batch processing on GPU                                      │
│  • ~1.7ms per game on RTX 5090                                  │
│                                                                 │
│  RESULT: 57x speedup                                            │
│                                                                 │
│  Key techniques:                                                │
│  • torch.where() instead of Python conditionals                 │
│  • Batch processing multiple games simultaneously               │
│  • Keep tensors on GPU throughout game                          │
│  • Vectorized legal move masking                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Anomaly Detector (A Case Study)

```
┌─────────────────────────────────────────────────────────────────┐
│                 THE ANOMALY DETECTOR                             │
│            (Detects the wrong anomalies)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Purpose: Halt training when something is wrong                 │
│                                                                 │
│  Configuration:                                                 │
│  • Threshold: 3 standard deviations from mean                   │
│  • Trigger: 10 consecutive "anomalies"                          │
│                                                                 │
│  What it caught:                                                │
│  • Loss values of 65-67 during early epochs                     │
│  • (These are normal. Mean was ~60.9)                           │
│  • Result: Halted training                                      │
│                                                                 │
│  What it missed:                                                │
│  • Training data was 2 weeks old                                │
│  • NPZ sync daemon wasn't syncing                               │
│  • Models were regressing                                       │
│  • Result: AI got worse for 2 weeks                             │
│                                                                 │
│  Current status: Disabled                                       │
│                                                                 │
│  Lesson: It's easier to detect unusual numbers than             │
│          unusual situations                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
ai-service/
├── app/
│   ├── ai/                    # Neural networks, MCTS, heuristics
│   │   ├── gpu_parallel_games.py    # Vectorized GPU selfplay (works great)
│   │   ├── gumbel_search_engine.py  # Unified MCTS entry point
│   │   └── neural_net_*.py          # Model architectures
│   ├── config/                # Environment, cluster, thresholds
│   ├── coordination/          # 208 modules for orchestration
│   │   ├── daemon_manager.py        # Daemon lifecycle
│   │   ├── daemon_registry.py       # Daemon specifications (132 of them)
│   │   ├── daemon_runners.py        # 124 runner functions
│   │   ├── event_router.py          # Event bus (292 event types)
│   │   ├── handler_base.py          # Base class for handlers
│   │   └── ...
│   ├── rules/                 # Python game engine (mirrors TS, 100% parity)
│   └── training/              # Training pipeline
├── scripts/
│   ├── p2p_orchestrator.py    # Main P2P server (~26K LOC, I don't understand it)
│   ├── p2p/
│   │   ├── managers/          # 7 extracted manager classes
│   │   └── loops/             # 22 background loops
│   └── master_loop.py         # Main automation entry point
├── config/
│   └── distributed_hosts.yaml # Cluster node configuration (41 nodes)
├── models/                    # Trained model checkpoints (worse than heuristic)
└── tests/                     # 984 test files (all passing, AI still regressed)
```

---

## How It Was Built

**Timeline:** 6-8 weeks
**Developer:** Non-programmer, non-technical background
**Tool:** Claude Code (AI-assisted development)
**Starting knowledge:** Couldn't write a for loop, didn't know what a daemon was
**Ending knowledge:** Have 132 daemons, still don't fully understand what they do

**The process:**

1. Describe problems in plain English
   - "The cluster keeps losing nodes after 6 hours"
   - "I need selfplay to prioritize configs that are falling behind"
   - "Wait, why are the models getting dumber?"
2. Claude investigates codebase (reads 20-50 files per session)
3. Claude proposes solutions that fit existing patterns
4. Iterate: "That's close, but also consider X"
5. Claude writes tests automatically (this is why coverage is 99.5%)
6. Tests pass
7. Something else breaks 2 weeks later

**What made it work:**

- CLAUDE.md files provide persistent context across sessions
- High test coverage lets Claude verify changes work
- Modular architecture allows incremental development
- Clear separation of concerns
- Willingness to say "I don't understand, explain it differently"

**What didn't work:**

- Building infrastructure before users (41 nodes, 0 users)
- Not tracking costs from day 1 ($585 AWS bill, $140/month ghost EC2 instance)
- Assuming "more nodes = better" without questioning why
- Not noticing when models regressed for 2 weeks
- Trusting tests to catch semantic problems (they catch syntax problems)

---

## Lessons Learned

1. **AI-assisted development is collaborative** — You still need to understand the problem deeply (or at least notice when models get worse)

2. **Tests aren't optional** — They're how Claude knows it didn't break something

3. **Tests aren't sufficient** — They verify code runs, not that it's correct

4. **Documentation compounds** — Claude reads it too, making future sessions more effective

5. **Start simple** — Let complexity emerge from real problems

6. **The last 20% takes 80% of the time** — Getting to "production-ready" is hard

7. **Monitor outcomes, not activities** — I knew every daemon was running. I didn't know the AI was getting worse.

8. **You can build things you don't understand** — This is not always good

---

## What's Next

- Right-sizing infrastructure costs (41 nodes for 0 users is a choice)
- Fixing the training loop so models improve instead of regress
- Potentially open-sourcing the training infrastructure
- Writing detailed blog posts about specific components
- Maybe getting some actual users for the game
- Probably discovering more things that were broken all along

---

## The Real Metrics

| What I Thought Mattered  | What Actually Mattered |
| ------------------------ | ---------------------- |
| 99.5% test coverage      | 0 users                |
| 132 daemon types         | 0 users                |
| 41 GPU nodes             | 0 users                |
| 292 event types          | 0 users                |
| 48h autonomous operation | 0 users                |
| 57x GPU speedup          | 0 users                |

---

_Built with Claude Code, January 2026_

_The AI that trained itself to lose_
