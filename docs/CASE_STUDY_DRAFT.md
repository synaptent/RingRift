# DRAFT: Case Study — Building a Distributed Training System for RingRift

> **Status: DRAFT** — This documents the training infrastructure as built. The game has not launched publicly and the neural network models have not been validated by real players yet.

## Training Neural Network Opponents Across a Heterogeneous GPU Fleet

---

## Introduction

RingRift is a multiplayer territory control board game where players place and move stacks of rings to claim territory, form lines, and outmaneuver opponents. The game supports four board geometries -- 8x8 square (64 cells), 19x19 square (361 cells), small hexagonal (61 cells), and large hexagonal (469 cells) -- with 2, 3, or 4 players on each. That produces 12 distinct game configurations, each with its own strategic landscape.

We wanted neural network AI opponents that could play all 12 configurations competently. The approach follows AlphaZero-style self-play: generate games where the AI plays against itself, train a neural network on those games, evaluate whether the new model is stronger, and repeat. The fundamental loop is simple. Making it work reliably across dozens of GPU nodes from different cloud providers, running unattended for days at a time, turned out to be the real engineering challenge.

A single training iteration for one configuration -- generate 1,000 games, export, train, evaluate -- takes roughly 4-8 hours on a single GPU depending on the board size. Multiply that by 12 configurations, each needing dozens of iterations to reach competitive strength, and single-machine training becomes impractical. We needed to distribute the work.

This case study describes the system we built: a P2P mesh network coordinating ~41 nodes with ~1.5TB of aggregate GPU memory, running a 7-stage training pipeline with 132 daemon types, 5 feedback loops, and enough resilience machinery to operate autonomously for 48+ hours.

## Architecture Overview

### The Training Pipeline

The pipeline has seven stages that form a continuous loop:

1. **Selfplay** -- GPU workers generate games using Gumbel MCTS (a quality-focused tree search variant) or heuristic engines. Games are stored in per-configuration SQLite databases.
2. **Export** -- Game databases are converted to NumPy NPZ arrays suitable for training. The exporter handles board geometry encoding, frame stacking, and data augmentation.
3. **Train** -- A PyTorch training job updates the neural network using the exported data, with early stopping and loss monitoring.
4. **Evaluate** -- The candidate model runs a gauntlet against baselines (random play, heuristic engine, previous best model). It must clear win-rate thresholds to proceed.
5. **Promote** -- Models that pass evaluation are promoted to canonical status and become the new baseline.
6. **Distribute** -- The promoted model is synced to all cluster nodes so the next round of selfplay uses the stronger model.
7. **Monitor** -- Continuous health checks, Elo tracking, and regression detection ensure the system stays on track.

Each stage is implemented as a combination of scripts and daemon processes, connected through an event bus with 292 event types.

### P2P Mesh Network

Rather than a traditional client-server architecture with a fixed coordinator, the cluster uses a peer-to-peer mesh network. Every node runs an instance of the P2P orchestrator (a ~28K LOC Python HTTP server) that handles leader election, job dispatching, and state synchronization.

Leader election uses the Bully algorithm: nodes are assigned priorities, and the highest-priority available node becomes leader. The leader is responsible for dispatching selfplay jobs, coordinating training, and maintaining the global work queue. If the leader fails, a new election completes within ~60 seconds thanks to a dedicated LeaderProbeLoop that sends 10-second health probes and triggers forced election after 6 consecutive failures.

Seven CPU-only nodes (Hetzner servers) participate as voters in the quorum without running GPU workloads, providing election stability even when GPU workers are busy or restarting. The quorum system operates across four health levels -- HEALTHY, DEGRADED, MINIMUM, and LOST -- allowing the cluster to continue useful work even when some nodes are unavailable rather than halting entirely.

### Daemon-Based Automation

The system runs 132 daemon types (116 active, 16 deprecated) organized in a three-layer architecture:

- **DaemonRegistry** -- A declarative registry mapping each daemon type to a specification (category, priority, health check interval, restart policy).
- **DaemonManager** -- A lifecycle coordinator that starts, stops, monitors, and auto-restarts daemons. It enforces startup ordering so event subscribers initialize before emitters.
- **DaemonRunners** -- 124 async runner functions that instantiate and execute the actual daemon logic.

Daemons communicate through an in-memory event bus with SHA-256 deduplication and a dead-letter queue for failed deliveries. This loose coupling means adding a new daemon that reacts to training completions, for example, requires only subscribing to the `TRAINING_COMPLETED` event -- no changes to existing code.

## Key Engineering Challenges

### 1. Heterogeneous GPU Fleet Management

The cluster includes hardware from six cloud providers spanning three GPU generations:

| Provider | Hardware                  | VRAM     | Role                        |
| -------- | ------------------------- | -------- | --------------------------- |
| Lambda   | GH200 (x11)               | 96 GB    | Primary selfplay + training |
| Vast.ai  | RTX 5090, 4090, 3090, A40 | 16-24 GB | Selfplay                    |
| RunPod   | H100, A100 (x4), L40S     | 40-80 GB | Training + selfplay         |
| Nebius   | H100 (x2), L40S           | 48-80 GB | Training backbone           |
| Vultr    | A100 vGPU                 | 20 GB    | Selfplay                    |
| Hetzner  | CPU only (x3)             | --       | P2P voters                  |

This diversity creates real problems. A selfplay job that runs comfortably on a GH200 with 96 GB of VRAM will OOM on a Vultr A100 vGPU with 20 GB. Training a v5-heavy-large architecture (25-35M parameters) requires nodes with at least 40 GB VRAM, which rules out the consumer GPUs entirely.

The NodeSelector manager handles this by tracking each node's GPU capabilities, current memory usage, and load. When dispatching jobs, it matches work to nodes: large-model training goes to H100s and GH200s, while smaller selfplay tasks with v2 architectures (2-4M parameters) can run on consumer hardware. A tiered Gumbel budget system adjusts the MCTS simulation count based on available resources -- 64 simulations for fast bootstrapping on weaker GPUs, up to 800+ for quality data generation on powerful ones.

The consumer GPUs actually turned out to be surprisingly useful for one thing: running many small selfplay games in parallel with lower simulation budgets. The aggregate throughput of fourteen Vast.ai nodes running lightweight selfplay often exceeded the output of fewer, more powerful machines.

### 2. SQLite Under Distributed Load

Each selfplay worker stores games in a local SQLite database. This is convenient for single-node operation but creates problems at scale. On the coordinator node, which aggregates data from the entire cluster, we encountered file descriptor exhaustion: dozens of databases open simultaneously for reads during export, plus incoming sync connections writing new data.

The solution had three parts. First, all blocking SQLite calls were wrapped in `asyncio.to_thread()` to prevent the event loop from stalling -- 275 usages across 76 files. Second, connection pooling with explicit limits prevents the fd count from growing unboundedly. Third, periodic `VACUUM` operations reclaim space from deleted records, but only when database size exceeds a threshold (large databases can take minutes to vacuum, and an ill-timed vacuum during an export window would block the pipeline).

We also hit a subtler issue: SQLite's `PRAGMA journal_mode=WAL` improves concurrent read performance but increases memory usage proportional to the write volume. On nodes running heavy selfplay, this caused gradual memory growth. The MemoryPressureController daemon monitors RAM usage across four tiers (CAUTION at 60%, WARNING at 70%, CRITICAL at 80%, EMERGENCY at 90%) and takes progressive action -- from pausing selfplay to forcing garbage collection to triggering graceful shutdown.

### 3. Transfer Learning Across Player Counts

Training a 4-player model from scratch requires substantially more data than a 2-player model because the game tree is larger and the value predictions are more complex (predicting win probabilities for 4 players instead of 2). We use transfer learning to bootstrap 4-player training from converged 2-player models.

The challenge is that the neural network's value head has a different output dimension for different player counts. A 2-player model outputs 2 win probabilities; a 4-player model needs 4. The policy head (which predicts moves) and the shared residual backbone are identical regardless of player count.

The `transfer_2p_to_4p.py` script handles this by loading the 2-player checkpoint, copying all shared weights (backbone and policy head), and initializing a new value head with the correct output dimension. The new value head's weights are initialized by duplicating and scaling the 2-player weights rather than random initialization, which preserves the model's learned sense of positional advantage.

In practice, transferred 4-player models reach competitive Elo ratings in roughly half the iterations required by training from scratch, representing a significant savings in GPU hours across the 4 board geometries.

### 4. Autonomous Operation and Recovery

The cluster needs to run unattended. Cloud GPU instances get preempted, network connections drop, and nodes occasionally crash under memory pressure. Eleven dedicated recovery daemons handle different failure modes:

- **PROGRESS_WATCHDOG** monitors Elo ratings and detects stalls. If a configuration's Elo has not improved after a configurable number of iterations, it triggers investigation -- increasing selfplay diversity, adjusting simulation budgets, or flagging the configuration for manual review.
- **P2P_RECOVERY** monitors the P2P orchestrator process itself and restarts it if health checks fail.
- **STALE_FALLBACK** handles sync failures gracefully. If a node cannot fetch the latest model after 5 attempts or 45 minutes, it falls back to using the most recent locally-available model rather than sitting idle.
- **MEMORY_MONITOR** tracks GPU VRAM and system RAM, proactively killing non-essential processes before OOM kills arrive.

The 4-layer resilience architecture stacks OS-level process supervision (launchd on macOS, systemd on Linux), memory pressure management, coordinator failover (primary/standby), and a cluster-wide health aggregator. The system has been verified to run for 48+ hours without intervention, with a measured mean time to recovery under 2.5 minutes for most failure types and ~70 seconds for leader failover.

### 5. Multi-Transport Data Synchronization

Syncing game databases and model checkpoints across 41 nodes on different networks is unreliable if you depend on a single transport. The system implements a multi-transport failover chain:

1. **Tailscale** (primary) -- All Lambda GH200 nodes and the local coordinator are on a Tailscale mesh VPN. Direct peer-to-peer transfers over Tailscale are fast and reliable.
2. **SSH/rsync** (fallback) -- For nodes not on Tailscale, standard rsync over SSH handles bulk data transfer.
3. **Base64 over SSH** -- When rsync fails due to connection resets (common on Vast.ai nodes with unstable networking), a fallback encodes files as base64 and pipes them through an SSH command. Slower but more reliable for small-to-medium files.
4. **HTTP** (last resort) -- The P2P orchestrator exposes endpoints for model downloads, allowing nodes to pull checkpoints via HTTP when all SSH-based methods fail.

The SyncPlanner manager decides which transport to use based on the target node's connectivity profile and recent transfer history. Nodes that have had repeated SSH failures automatically get downgraded to HTTP for subsequent transfers until their connectivity improves.

## Results

### Models Trained

All 12 canonical configurations have trained models:

| Board                 | 2-Player | 3-Player | 4-Player |
| --------------------- | -------- | -------- | -------- |
| hex8 (61 cells)       | 38 MB    | 38 MB    | 38 MB    |
| square8 (64 cells)    | 32 MB    | 15 MB    | 366 MB   |
| square19 (361 cells)  | 102 MB   | 103 MB   | 103 MB   |
| hexagonal (469 cells) | 166 MB   | 166 MB   | 166 MB   |

The model sizes reflect the architecture choices: smaller boards use lighter architectures (v2, 2-4M parameters) while larger boards use v5-heavy or v5-heavy-large (8-35M parameters) to handle the increased state complexity.

> **Note:** These models have not yet been validated by human players. Elo ratings are measured against heuristic baselines and previous model iterations via self-play evaluation. Actual playing strength against humans is unknown.

### Cluster Scale

The operational cluster comprises approximately 41 configured nodes with roughly 1.5 TB of aggregate GPU memory. The 11 Lambda GH200 nodes form the backbone for both selfplay generation and training, while the remaining 30 nodes across Vast.ai, RunPod, Nebius, and Vultr contribute selfplay throughput and occasional training capacity.

### System Metrics

- **Daemon types**: 132 total (116 active)
- **Event types**: 292 in the event bus
- **Health checks**: 257+ across P2P and coordination layers
- **Circuit breaker types**: 9 with 4-tier escalation
- **Test coverage**: 984 tests, 99.5% coverage
- **Codebase**: 341 Python modules in the AI service

### GPU Selfplay Performance

The vectorized GPU selfplay engine achieves 6-57x speedup over CPU selfplay depending on hardware:

| GPU      | Speedup | Notes                      |
| -------- | ------- | -------------------------- |
| A10      | 6.5x    | Entry-level datacenter     |
| RTX 4090 | ~20x    | Consumer high-end          |
| RTX 5090 | 57x     | Batch size 200             |
| GH200    | ~30x    | 96 GB allows large batches |

The engine runs game simulation entirely on CUDA with only 14 remaining `.item()` calls (down from 80+ after optimization), keeping data on the GPU throughout the selfplay loop.

## Lessons Learned

**Start with the simplest thing that works, then add complexity as needed.** The first version of the distributed system was a shell script that SSH'd into each node and ran selfplay. The P2P mesh, daemon system, and event bus were added incrementally as pain points emerged. If we had tried to design the full system upfront, we would have gotten the abstractions wrong.

**SQLite is surprisingly capable in distributed systems -- until it is not.** For our use case (append-heavy game databases, occasional bulk reads for export), SQLite outperforms PostgreSQL in simplicity and operational overhead. The key is wrapping all IO in async threads, managing connection counts carefully, and accepting that some operations (like cross-node queries) require a different approach. The cluster-wide data catalog sits on top of per-node SQLite databases rather than replacing them.

**Heterogeneous hardware is an asset, not just a liability.** The consumer GPUs on Vast.ai are cheap and plentiful. They cannot run large training jobs, but they can run thousands of small selfplay games. The NodeSelector's ability to match work to hardware capability turned a logistical headache into a cost advantage.

**Quorum-based degradation beats hard failure thresholds.** Early versions of the system would halt all operations if too many nodes went offline. The four-tier quorum health system (HEALTHY/DEGRADED/MINIMUM/LOST) was a significant improvement -- the cluster continues doing useful work even when operating below full capacity, and cleanly refuses dangerous operations (like model promotion) when it does not have enough consensus.

**Recovery daemons should be as simple as possible.** The most reliable recovery daemons are the ones that do one thing: check a condition, take an action. The ProgressWatchdog checks if Elo has stalled, then adjusts selfplay parameters. The P2PRecovery daemon checks if the orchestrator is healthy, then restarts it if not. Attempts to build "smart" recovery logic that anticipated multiple failure modes simultaneously were harder to test and less reliable than simple, composable recovery daemons.

**Cross-language parity testing is non-negotiable.** The game rules are defined in TypeScript and mirrored in Python. Without automated parity testing (replaying games through both engines and comparing results), subtle rule divergences would silently corrupt training data. The parity gate -- which blocks selfplay databases from being used for training until they pass TypeScript replay verification -- has caught bugs that would have otherwise wasted days of GPU time on training against incorrect game data.

---

_RingRift is an open-source project. The full source code, including the distributed training infrastructure described in this case study, is available on [GitHub](https://github.com/an0mium/RingRift)._
