# AI Service Scripts

This directory contains scripts for the RingRift AI training and improvement infrastructure.

## Canonical Entry Points

### Primary Orchestrator

**`unified_ai_loop.py`** - The canonical self-improvement orchestrator. This is the main entry point for the AI improvement loop.

```bash
# Start the unified loop
python scripts/unified_ai_loop.py --start

# Run in foreground with verbose output
python scripts/unified_ai_loop.py --foreground --verbose

# Check status
python scripts/unified_ai_loop.py --status
```

Features:

- Streaming data collection from distributed nodes (60s sync)
- Shadow tournament evaluation (15min lightweight)
- Training scheduler with data quality gates
- Model promotion with Elo thresholds
- Adaptive curriculum weighting
- Value calibration analysis
- Temperature scheduling for exploration control

## Key Categories

### P2P Cluster Orchestration

- `p2p_orchestrator.py` - **Primary cluster orchestrator** (1.1MB, self-healing P2P coordination)
  - Leader election and peer discovery
  - Auto-starts selfplay/training jobs
  - Vast.ai and Lambda Labs integration
  - Supports all board types: square8, hex8, square19, hexagonal
  - Start: `python scripts/p2p_orchestrator.py --node-id <name> --peers <ip>:8770`
- `vast_p2p_sync.py` - **Vast.ai instance synchronization** (22KB)
  - Synchronizes Vast.ai instances with P2P network
  - Instance status tracking, node unretiring, P2P orchestrator startup
  - Auto-updates `distributed_hosts.yaml` with new instances
  - Start: `python scripts/vast_p2p_sync.py --full` (check + sync + start P2P)
  - Check only: `python scripts/vast_p2p_sync.py --check`
- `vast_keepalive.py` - **Keepalive manager** for Vast.ai instances (8KB)
  - Prevents idle instance termination via periodic heartbeats
  - Auto-restart stopped instances via Vast.ai CLI
  - SSH health checks and worker restart on unhealthy instances
  - Status: `python scripts/vast_keepalive.py --status`
  - Full cycle: `python scripts/vast_keepalive.py --auto`
  - Install cron: `python scripts/vast_keepalive.py --install-cron`
- `cluster_monitoring.sh` - **Comprehensive cluster health monitoring** (14KB)
  - Tailscale mesh status, Vast.ai CLI monitoring, Lambda node health
  - P2P orchestrator status, selfplay/gauntlet monitoring
  - Continuous mode: `./scripts/cluster_monitoring.sh --continuous --interval 60`
- `node_resilience.py` - **Node resilience daemon** (53KB)
  - P2P orchestrator monitoring with local selfplay fallback
  - Auto-reconnect on network changes, IP change detection
  - Start: `python scripts/node_resilience.py --daemon`
- `cluster_manager.py` - Cluster node management
- `cluster_worker.py` - Worker node implementation
- `cluster_health_check.py` - Health monitoring

### Training

- `multi_config_training_loop.py` - **Multi-board training coordinator** (52KB)
  - Adaptive curriculum based on Elo ratings
  - Balance mode for underrepresented configs
  - Supports hex8, square8, square19, hexagonal configurations
  - **Dynamic JSONL directory discovery** (2025-12-17) - auto-discovers selfplay data sources
  - Victory-type balanced sampling (`--sampling-weights victory_type`)
  - Policy label smoothing, hex augmentation
  - Start: `python scripts/multi_config_training_loop.py --configs all`
- `run_nn_training_baseline.py` - **Primary NN training script** (21KB)
  - Board-specific hyperparameters from `config/hyperparameters.json`
  - Optimized settings: batch_size=256, warmup_epochs=5, cosine scheduler
  - Victory-type balanced sampling (`--sampling-weights victory_type`)
  - Start: `python scripts/run_nn_training_baseline.py --board square8 --num-players 2`
- `run_optimized_training.py` - **Optimized training wrapper**
  - Auto-selects best hyperparameters per board/player config
  - Optional Elo calibration after training (`--run-elo`)
  - Start: `python scripts/run_optimized_training.py --board hexagonal --players 4 --victory-balanced`
- `train_nnue_policy.py` - **NNUE policy head training** (41KB)
  - AMP (mixed precision) training
  - KL divergence loss for MCTS distillation (`--auto-kl-loss`, `--use-kl-loss`)
  - Direct JSONL loading (`--jsonl data/selfplay/*.jsonl`)
  - Temperature annealing, label smoothing
  - Start: `python scripts/train_nnue_policy.py --jsonl data/selfplay/mcts_*.jsonl --auto-kl-loss`
- `train_nnue.py` - **Standard NNUE training** (112KB)
  - Value and policy head training from SQLite databases
  - Start: `python scripts/train_nnue.py --db data/games/training.db --board-type square8`
- `run_improvement_loop.py` - **Alternative improvement loop** (95KB)
  - Complete self-improvement cycle with selfplay, training, and evaluation
  - Start: `python scripts/run_improvement_loop.py --board square8 --iterations 10`
- `training_completion_watcher.py` - **Auto-Elo trigger daemon**
  - Monitors training logs and triggers Elo tournaments on completion
  - Start: `python scripts/training_completion_watcher.py --daemon`
- `auto_training_pipeline.py` - **Automated training pipeline** (40KB)
  - Complete workflow: data collection → value training → policy training → A/B test → selfplay → sync
  - Curriculum-based policy training with staged progression (endgame → opening)
  - A/B testing to validate policy model improvements before deployment
  - Policy-guided selfplay for high-quality training data generation
  - Start: `python scripts/auto_training_pipeline.py --board-type square8 --num-players 2`
  - Dry run: `python scripts/auto_training_pipeline.py --dry-run --board-type square8`
  - Policy only: `python scripts/auto_training_pipeline.py --skip-collect --skip-backfill --skip-train`
- `train_nnue_policy_curriculum.py` - **Staged curriculum policy training** (17KB)
  - Progressive training: endgame → late-mid → midgame → opening → full
  - Transfer learning between stages
  - Board-specific max_moves_per_position auto-selection
  - Start: `python scripts/train_nnue_policy_curriculum.py --db data/games/*.db --board-type square8`
- `ab_test_policy_models.py` - **A/B testing for policy models** (12KB)
  - Compare policy model against baseline
  - Multi-think-time testing (50ms, 100ms, 200ms, 500ms)
  - Start: `python scripts/ab_test_policy_models.py --model-a models/nnue/policy.pt --board-type square8`
- `reanalyze_mcts_policy.py` - **MCTS policy reanalysis**
  - Adds MCTS visit distributions to existing games for KL loss training
  - Start: `python scripts/reanalyze_mcts_policy.py --input games.jsonl --output mcts_games.jsonl`
- `curriculum_training.py` - Generation-based curriculum training
- `run_self_play_soak.py` - Self-play data generation (158KB)
- `run_hybrid_selfplay.py` - Hybrid self-play modes (67KB)
- `hex8_training_pipeline.py` - **Hex8-specific pipeline** (19KB)
  - Optimized for hex8 board training

### Selfplay Data Generation

- `run_distributed_selfplay.py` - **Distributed selfplay for cloud workers** (60KB)
  - Designed for CPU-based cloud VMs (AWS, GCP, Azure)
  - Auto-calculates max_moves from board/player configuration
  - Supports diverse AI matchups (neural, heuristic, MCTS)
  - **Neural batching for 2-5x throughput** (2025-12-17):
    ```bash
    python scripts/run_distributed_selfplay.py \
      --board-type hex8 --num-players 2 --num-games 1000 \
      --enable-nn-batching --nn-batch-timeout-ms 50 \
      --output file://data/selfplay/hex8_2p/games.jsonl
    ```
- `run_gpu_selfplay.py` - **GPU-accelerated selfplay** (55KB)
  - 10-100x speedup via parallel game simulation
  - Supports all board types: `square8`, `hex8`, `square19`, `hexagonal`
  - Start: `python scripts/run_gpu_selfplay.py --board hex8 --num-games 1000 --output-dir data/selfplay/gpu_hex8`
- `run_hybrid_selfplay.py` - **Hybrid CPU/GPU selfplay** (67KB)
  - Mixed MCTS and neural network modes
  - Start: `python scripts/run_hybrid_selfplay.py --board-type hex8 --engine-mode gumbel-mcts`
- `cluster_control.py` - **Cluster selfplay orchestration**
  - Start GPU selfplay on all cluster nodes: `python scripts/cluster_control.py selfplay start --board hex8`
  - Check cluster status: `python scripts/cluster_control.py status`

### Evaluation

- `auto_elo_tournament.py` - **Automated Elo tournament daemon**
  - Periodic tournaments with Slack alerts
  - Regression detection
  - Start daemon: `python scripts/auto_elo_tournament.py --daemon --interval 14400`
- `run_model_elo_tournament.py` - Model Elo tournaments
- `run_diverse_tournaments.py` - Multi-configuration tournaments
- `elo_promotion_gate.py` - Elo-based model promotion with Wilson confidence intervals

### Data Management

- `cluster_sync_coordinator.py` - **Cluster-wide sync orchestrator** (coordinates all sync utilities)
  - Full sync: `python scripts/cluster_sync_coordinator.py --mode full`
  - Models only: `python scripts/cluster_sync_coordinator.py --mode models`
  - Games only: `python scripts/cluster_sync_coordinator.py --mode games`
  - ELO only: `python scripts/cluster_sync_coordinator.py --mode elo`
  - Status: `python scripts/cluster_sync_coordinator.py --status`
  - Uses aria2/tailscale/cloudflare for hard-to-reach nodes
- `unified_data_sync.py` - **Unified data sync service** (replaces deprecated scripts below)
  - Run as daemon: `python scripts/unified_data_sync.py`
  - With watchdog: `python scripts/unified_data_sync.py --watchdog`
  - One-shot sync: `python scripts/unified_data_sync.py --once`
- `streaming_data_collector.py` - _(DEPRECATED)_ Incremental game data sync - use `unified_data_sync.py`
- `collector_watchdog.py` - _(DEPRECATED)_ Collector health monitoring - use `unified_data_sync.py --watchdog`
- `sync_all_data.py` - _(DEPRECATED)_ Batch data sync - use `unified_data_sync.py --once`
- `build_canonical_training_pool_db.py` - **Canonical training pool aggregation** (32KB)
  - Per-game canonical history and parity gates
  - Holdout exclusion, quarantine of failing games
  - Single ingestion point for all training data
  - Start: `python scripts/build_canonical_training_pool_db.py --output data/games/canonical.db`
- `aggregate_jsonl_to_db.py` - JSONL to SQLite conversion
- `elo_db_sync.py` - **Distributed Elo database sync** (32KB)
  - Multi-transport support (Tailscale, aria2, HTTP)
  - Worker/coordinator modes, automatic discovery
  - Mac Studio as authoritative coordinator
  - Start: `python scripts/elo_db_sync.py --mode coordinator`
- `auto_export_training_data.py` - **Automated training data export**
  - Exports data for underrepresented board/player configs
  - Start: `python scripts/auto_export_training_data.py --dry-run`
- `export_replay_dataset.py` - **Export training samples from replays** (41KB)
  - Rank-aware value encoding for multiplayer
  - Quality filtering (completed games, move count ranges)
  - Incremental export with caching
  - Start: `python scripts/export_replay_dataset.py --db data/games/training.db --output data/npz/`
- `jsonl_to_npz.py` - **JSONL to NPZ conversion** (50KB)
  - Game replaying with proper feature extraction
  - Checkpointing every N games, encoder selection
  - Start: `python scripts/jsonl_to_npz.py --input data/selfplay/*.jsonl --output data/npz/`
- `distributed_export.py` - **Distributed parallel export** (40KB)
  - Game chunking for parallel processing, HTTP serving
  - aria2 multi-source downloads, NPZ chunk merging
  - Start: `python scripts/distributed_export.py --coordinator --chunks 10`
- `filter_training_data.py` - Filter and clean training data

### Model Management

- `sync_models.py` - Model synchronization across cluster
- `prune_models.py` - Old model cleanup
- `model_promotion_manager.py` - Automated model promotion
- `validate_models.py` - Validate model files for corruption

### Cluster Management

- `update_cluster_code.py` - **Cluster code synchronization**
  - Push code updates to all cluster nodes
  - Auto-stash local changes: `python scripts/update_cluster_code.py --auto-stash`
  - Force reset: `python scripts/update_cluster_code.py --force-reset`
  - Status: `python scripts/update_cluster_code.py --status`
- `update_distributed_hosts.py` - Update distributed hosts configuration
- `vast_autoscaler.py` - Vast.ai instance autoscaling
- `vast_lifecycle.py` - Vast.ai instance lifecycle management
- `vast_p2p_manager.py` - Vast.ai P2P network management
- `cluster_auto_recovery.py` - Auto-recover failed cluster nodes
- `cluster_automation.py` - Cluster automation utilities
- `cluster_control.py` - Cluster control commands

### Analysis

- `analyze_game_statistics.py` - **Comprehensive game statistics** (107KB)
  - Victory type distribution, win rates, game length stats, recovery usage
  - AI type breakdown, data quality metrics, metadata fixing
  - Supports JSONL recursive scanning, quarantine mode
  - Start: `python scripts/analyze_game_statistics.py --db data/games/all.db`
  - JSONL: `python scripts/analyze_game_statistics.py --jsonl data/selfplay/ --recursive`
- `check_ts_python_replay_parity.py` - **TS/Python parity validation** (77KB)
  - Post-move/post-bridge view semantics verification
  - Canonical/legacy parity modes, structural issue detection
  - Critical quality gate for training data
  - Start: `python scripts/check_ts_python_replay_parity.py --canonical --verbose`
- `track_elo_improvement.py` - Elo trend tracking
- `aggregate_elo_results.py` - Aggregate Elo results from multiple sources
- `baseline_gauntlet.py` - **Run baseline model gauntlet** (20KB)
  - Evaluate models against baseline opponents
  - Start: `python scripts/baseline_gauntlet.py --model models/best.pth`
- `two_stage_gauntlet.py` - **Two-stage model evaluation** (31KB)
  - Stage 1 screening (10 games, 40% threshold)
  - Stage 2 deep evaluation (50 games, Wilson score intervals)
  - Auto-promotion of top performers with game recording
  - Start: `python scripts/two_stage_gauntlet.py --model models/candidate.pth`
- `shadow_tournament_service.py` - **Continuous model evaluation service** (35KB)
  - 15-minute shadow tournaments (10-20 games)
  - Hourly full tournaments, regression detection
  - Provides early feedback without waiting for full Elo calibration
  - Start: `python scripts/shadow_tournament_service.py --daemon`

### Dashboard & Monitoring

- `dashboard_server.py` - **RingRift Dashboard Server** (24KB)
  - Unified web interface for training monitoring and replay
  - Start: `python scripts/dashboard_server.py --port 8080`
  - **Dashboard URLs** (default port 8080):
    - `/` - Main Dashboard (Elo leaderboard, cluster status)
    - `/training` - Training Metrics (loss curves, throughput, LR schedule)
    - `/replay` - Game Replay Viewer (move-by-move with AI annotations)
    - `/compare` - Model Comparison (side-by-side performance)
    - `/tensorboard` - TensorBoard (auto-starts if not running)
  - **API Endpoints**:
    - `/api/leaderboard` - Elo rankings
    - `/api/cluster/status` - Cluster health
    - `/api/training/loss-curves` - Training loss data
    - `/api/elo/progression` - Elo over time
    - `/api/replay/games` - Game list for replay
    - `/api/tensorboard/status` - TensorBoard status
- `dashboard_assets/` - Dashboard frontend assets
  - `model_dashboard.html` - Main dashboard page
  - `training_dashboard.html` - Training metrics page
  - `replay_viewer.html` - Game replay interface
  - `model_comparison.html` - Model comparison page
  - `board_renderer.js` - Board rendering (square/hex)
  - `replay_viewer.js` - Replay navigation logic

### Data Validation

- `training_preflight_check.py` - **Pre-training validation** (12KB)
  - Database integrity, data volume, feature consistency checks
  - Resource availability verification (GPU, disk, memory)
  - Model checkpoint validity
  - Start: `python scripts/training_preflight_check.py`
- `holdout_validation.py` - **Overfitting detection** (18KB)
  - Holdout set management for unseen data evaluation
  - Train vs holdout loss comparison (warning: >0.10 gap)
  - Value head calibration metrics
  - Start: `python scripts/holdout_validation.py --evaluate --model models/square8_2p.pt`
  - Stats: `python scripts/holdout_validation.py --stats`
- `generate_canonical_selfplay.py` - **End-to-end canonical validation** (15KB)
  - TS/Python parity + canonical history validation
  - FE/territory fixture tests
  - Produces health summary JSON
  - Start: `python scripts/generate_canonical_selfplay.py --board-type square8 --num-games 50`

### Hyperparameter Optimization

- `run_gpu_cmaes.py` - **GPU-accelerated CMA-ES** (25KB)
  - 10-100x faster fitness evaluation on GPU
  - Single GPU: `python scripts/run_gpu_cmaes.py --board square8 --generations 50`
  - Multi-GPU: `python scripts/run_gpu_cmaes.py --board square8 --multi-gpu`
- `run_distributed_gpu_cmaes.py` - **Distributed CMA-ES**
  - Cluster-wide heuristic weight optimization
  - Start coordinator: `python scripts/run_distributed_gpu_cmaes.py --mode coordinator`
- `run_iterative_cmaes.py` - **Iterative CMA-ES refinement**
  - Warm start from previous best weights
  - Population adaptation, sigma annealing
  - Start: `python scripts/run_iterative_cmaes.py --board square8 --iterations 5`
- `run_cmaes_optimization.py` - **Basic CMA-ES** (CPU-based)
  - Simpler CMA-ES without GPU acceleration
- `cmaes_cloud_worker.py` - **Cloud CMA-ES worker**
  - Remote worker for distributed CMA-ES optimization

### Benchmarking

- `benchmark_engine.py` - Engine performance benchmarking
- `benchmark_gpu_cpu.py` - GPU vs CPU performance comparison
- `benchmark_policy.py` - Policy head performance benchmarking
- `benchmark_ai_memory.py` - AI memory usage benchmarking

## Module Dependencies

### Canonical Service Interfaces

| Module                                | Purpose                | Usage                                        |
| ------------------------------------- | ---------------------- | -------------------------------------------- |
| `app.training.elo_service`            | Elo rating operations  | `get_elo_service()` singleton                |
| `app.training.curriculum`             | Curriculum training    | `CurriculumTrainer`, `CurriculumConfig`      |
| `app.training.value_calibration`      | Value head calibration | `ValueCalibrator`, `CalibrationTracker`      |
| `app.training.temperature_scheduling` | Exploration control    | `TemperatureScheduler`, `create_scheduler()` |

### Supporting Modules

| Module                                | Purpose                     |
| ------------------------------------- | --------------------------- |
| `app.tournament.elo`                  | Elo calculation utilities   |
| `app.training.elo_reconciliation`     | Distributed Elo consistency |
| `app.distributed.cluster_coordinator` | Cluster coordination        |
| `app.integration.pipeline_feedback`   | Training feedback loops     |

## Archived Scripts

The `archive/` subdirectory contains deprecated scripts that have been superseded:

| Script                                   | Superseded By                              |
| ---------------------------------------- | ------------------------------------------ |
| `master_self_improvement.py`             | `unified_ai_loop.py`                       |
| `unified_improvement_controller.py`      | `unified_ai_loop.py`                       |
| `integrated_self_improvement.py`         | `unified_ai_loop.py`                       |
| `export_replay_dataset.py`               | Direct DB queries                          |
| `validate_canonical_training_sources.py` | Data quality gates in `unified_ai_loop.py` |

## Resource Management

All scripts enforce **80% maximum resource utilization** to prevent overloading:

### Resource Limits (enforced 2025-12-16)

| Resource | Warning | Critical  | Notes                              |
| -------- | ------- | --------- | ---------------------------------- |
| Disk     | 65%     | 70%       | Tighter limit - cleanup takes time |
| Memory   | 70%     | 80%       | Hard stop when exceeded            |
| CPU      | 70%     | 80%       | Hard stop when exceeded            |
| GPU      | 70%     | 80%       | CUDA memory safety                 |
| Load Avg | -       | 1.5x CPUs | System overload detection          |

### Using Resource Guard

```python
from app.utils.resource_guard import (
    check_disk_space, check_memory, check_gpu_memory,
    can_proceed, wait_for_resources, ResourceGuard
)

# Pre-flight check before heavy operations
if not can_proceed(disk_required_gb=5.0, mem_required_gb=2.0):
    logger.error("Resource limits exceeded")
    sys.exit(1)

# Context manager for resource-safe operations
with ResourceGuard(disk_required_gb=5.0, mem_required_gb=2.0) as guard:
    if not guard.ok:
        return  # Resources not available
    # ... do work ...

# Periodic check in long-running loops
for i in range(num_games):
    if i % 50 == 0 and not check_memory():
        logger.warning("Memory pressure, stopping early")
        break
```

### Key Files

- `app/utils/resource_guard.py` - Unified resource checking utilities
- `app/coordination/safeguards.py` - Circuit breakers and backpressure
- `app/coordination/resource_targets.py` - Utilization targets for scaling
- `app/coordination/resource_optimizer.py` - PID-based workload adjustment
- `scripts/disk_monitor.py` - Disk cleanup automation

## Environment Variables

| Variable                          | Description                                 | Default |
| --------------------------------- | ------------------------------------------- | ------- |
| `RINGRIFT_DISABLE_LOCAL_TASKS`    | Skip local training/eval (coordinator mode) | `false` |
| `RINGRIFT_TRACE_DEBUG`            | Enable detailed tracing                     | `false` |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS`  | Skip shadow contract validation             | `false` |
| `RINGRIFT_ENABLE_POLICY_TRAINING` | Enable NNUE policy training                 | `1`     |
| `RINGRIFT_POLICY_AUTO_KL_LOSS`    | Auto-detect and enable KL loss              | `1`     |
| `RINGRIFT_POLICY_KL_MIN_COVERAGE` | Min MCTS coverage for auto-KL               | `0.3`   |
| `RINGRIFT_POLICY_KL_MIN_SAMPLES`  | Min samples for auto-KL                     | `50`    |
| `RINGRIFT_ENABLE_AUTO_HP_TUNING`  | Enable hyperparameter auto-tuning           | `0`     |
| `RINGRIFT_SOCKS_PROXY`            | SOCKS5 proxy URL for P2P                    | (none)  |
| `RINGRIFT_P2P_VERBOSE`            | Enable verbose P2P logging                  | `false` |

## Cluster Node Requirements

### Hardware Requirements

| Component | Minimum               | Recommended     | Notes                     |
| --------- | --------------------- | --------------- | ------------------------- |
| GPU       | NVIDIA GTX 1080 (8GB) | RTX 3090 / A100 | CUDA 11.7+ required       |
| RAM       | 16GB                  | 32GB+           | For large batch training  |
| Storage   | 50GB SSD              | 200GB+ NVMe     | Fast I/O for data loading |
| Network   | 100Mbps               | 1Gbps           | P2P sync bandwidth        |

### Software Requirements

- **Python 3.10+** with PyTorch 2.0+
- **CUDA 11.7+** (for GPU training)
- **Tailscale** (P2P mesh networking)
- **rsync** (data synchronization)

### Node Roles

| Role                | Description                        | Resources              |
| ------------------- | ---------------------------------- | ---------------------- |
| **Coordinator**     | Leader election, task distribution | Low GPU, high network  |
| **Trainer**         | Neural network training            | High GPU, high RAM     |
| **Selfplay Worker** | Game generation                    | Medium GPU, medium RAM |
| **Evaluator**       | Model evaluation, gauntlet         | Medium GPU             |

### Network Ports

| Port | Service               | Protocol |
| ---- | --------------------- | -------- |
| 8770 | P2P Orchestrator      | TCP      |
| 8080 | Dashboard Server      | HTTP     |
| 6006 | TensorBoard           | HTTP     |
| 5432 | PostgreSQL (optional) | TCP      |

### Quick Setup

```bash
# 1. Install Tailscale for P2P mesh
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up

# 2. Clone and setup
git clone <repo> && cd ai-service
pip install -r requirements.txt

# 3. Start P2P orchestrator
python scripts/p2p_orchestrator.py --node-id $(hostname) --peers <coordinator>:8770

# 4. Verify connectivity
python scripts/p2p_orchestrator.py --status
```

### Vast.ai Specific

For Vast.ai GPU instances:

- Use `vast_p2p_sync.py --full` to auto-configure instances
- Keepalive: `vast_keepalive.py --auto` (prevents idle termination)
- Minimum: RTX 3090 / A5000 instances recommended

## Configuration

The unified loop reads configuration from `config/unified_loop.yaml`. Key settings:

- Data sync intervals
- Training thresholds
- Evaluation frequencies
- Cluster coordination options
