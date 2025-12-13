#!/usr/bin/env python
"""Master pipeline orchestrator for distributed AI training.

This script coordinates the complete AI training pipeline across all available
compute resources. It manages:
- Distributed selfplay across AWS instances and local machines
- Data synchronization and aggregation
- Neural network training (on Mac Studio with MPS)
- CMA-ES heuristic optimization
- NNUE training
- Model evaluation tournaments

Usage:
    # Run a single iteration of the pipeline
    python scripts/pipeline_orchestrator.py --iterations 1

    # Run continuous improvement loop
    python scripts/pipeline_orchestrator.py --iterations 10 --continuous

    # Run specific phase only
    python scripts/pipeline_orchestrator.py --phase selfplay
    python scripts/pipeline_orchestrator.py --phase sync
    python scripts/pipeline_orchestrator.py --phase training
    python scripts/pipeline_orchestrator.py --phase evaluation

    # Dry run (show what would be executed)
    python scripts/pipeline_orchestrator.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# SSH retry configuration
SSH_MAX_RETRIES = 3
SSH_BASE_DELAY = 2.0  # seconds
SSH_MAX_DELAY = 30.0  # seconds
SSH_BACKOFF_FACTOR = 2.0

# Smart polling configuration
POLL_INTERVAL_SECONDS = 60  # Check every minute
MAX_PHASE_WAIT_MINUTES = 120  # Maximum wait for any phase
SELFPLAY_MIN_GAMES_THRESHOLD = 50  # Min games before proceeding
CMAES_COMPLETION_CHECK_CMD = "pgrep -f 'run_iterative_cmaes' >/dev/null && echo running || echo done"


@dataclass
class WorkerConfig:
    """Configuration for a compute worker."""

    name: str
    host: str  # user@hostname format
    role: str  # "selfplay", "training", "cmaes", "mixed"
    capabilities: List[str]
    ssh_key: Optional[str] = None
    ssh_port: int = 22  # Non-standard port for Vast.ai etc
    remote_path: str = "~/ringrift/ai-service"
    max_parallel_jobs: int = 1


@dataclass
class SelfplayJob:
    """Configuration for a selfplay job."""

    board_type: str
    num_players: int
    num_games: int
    engine_mode: str = "mixed"  # Use mixed for diverse training data
    max_moves: int = 500
    seed: int = 0
    use_trained_profiles: bool = True  # Load CMA-ES optimized heuristics
    use_neural_net: bool = False  # Enable NN for descent/mcts


@dataclass
class PipelineState:
    """Current state of the pipeline with full checkpointing support."""

    iteration: int = 0
    phase: str = "idle"
    phase_completed: Dict[str, bool] = field(default_factory=dict)  # Tracks which phases completed
    games_generated: Dict[str, int] = field(default_factory=dict)
    models_trained: List[str] = field(default_factory=list)
    last_sync: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    # Elo tracking
    elo_ratings: Dict[str, float] = field(default_factory=dict)  # model_id -> Elo rating
    elo_history: List[Dict[str, Any]] = field(default_factory=list)  # List of Elo updates
    # Model registry
    model_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # model_id -> metadata
    # Game deduplication
    seen_game_hashes: Set[str] = field(default_factory=set)
    # Tier gating
    tier_promotions: Dict[str, str] = field(default_factory=dict)  # config -> current tier


# Worker configurations loaded from gitignored config file
# See config/distributed_hosts.yaml for actual host configuration
def load_workers_from_config() -> List[WorkerConfig]:
    """Load worker configurations from distributed_hosts.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print(f"Warning: {config_path} not found. Using empty worker list.")
        print("Copy config/distributed_hosts.example.yaml to config/distributed_hosts.yaml")
        return []

    with open(config_path) as f:
        config = yaml.safe_load(f)

    workers = []
    for name, host_config in config.get("hosts", {}).items():
        if host_config.get("status") != "ready":
            continue
        workers.append(WorkerConfig(
            name=name,
            host=f"{host_config.get('ssh_user', 'ubuntu')}@{host_config.get('ssh_host', '')}",
            role=host_config.get("role", "selfplay"),
            capabilities=host_config.get("capabilities", ["square8"]),
            ssh_key=host_config.get("ssh_key"),
            ssh_port=host_config.get("ssh_port", 22),
            remote_path=host_config.get("ringrift_path", "~/ringrift/ai-service"),
            max_parallel_jobs=host_config.get("max_parallel_jobs", 2),
        ))
    return workers

WORKERS = load_workers_from_config()

# Default selfplay configuration per iteration
# Each board×player gets multiple engine modes for diverse training data:
# - mixed: Samples from canonical ladder (random, heuristic, minimax, mcts, descent)
# - heuristic-only: Pure heuristic games (benefits from CMA-ES trained weights)
# - descent-only: Pure descent games (benefits from NN when available)
# - nn-only: Neural-enabled descent/mcts (when NN checkpoint exists)
#
# The mix ensures:
# 1. Heuristics are exercised (so CMA-ES improvements get tested)
# 2. Search algorithms get trained heuristics for leaf evaluation
# 3. NN gets diverse training data from all skill levels
# 4. Self-play covers the full strength spectrum

DEFAULT_SELFPLAY_CONFIG = {
    # Square8 2p - Primary training config, most games
    "square8_2p_mixed": SelfplayJob("square8", 2, 40, "mixed", 500),
    "square8_2p_heuristic": SelfplayJob("square8", 2, 20, "heuristic-only", 500),
    "square8_2p_minimax": SelfplayJob("square8", 2, 15, "minimax-only", 500),
    "square8_2p_mcts": SelfplayJob("square8", 2, 15, "mcts-only", 500),
    "square8_2p_descent": SelfplayJob("square8", 2, 20, "descent-only", 500),
    "square8_2p_nn": SelfplayJob("square8", 2, 10, "nn-only", 500, use_neural_net=True),

    # Square8 3p
    "square8_3p_mixed": SelfplayJob("square8", 3, 25, "mixed", 600),
    "square8_3p_heuristic": SelfplayJob("square8", 3, 15, "heuristic-only", 600),
    "square8_3p_descent": SelfplayJob("square8", 3, 15, "descent-only", 600),

    # Square8 4p
    "square8_4p_mixed": SelfplayJob("square8", 4, 20, "mixed", 700),
    "square8_4p_heuristic": SelfplayJob("square8", 4, 10, "heuristic-only", 700),
    "square8_4p_descent": SelfplayJob("square8", 4, 10, "descent-only", 700),

    # Square19 2p - Larger board, fewer games
    "square19_2p_mixed": SelfplayJob("square19", 2, 20, "mixed", 800),
    "square19_2p_heuristic": SelfplayJob("square19", 2, 15, "heuristic-only", 800),
    "square19_2p_descent": SelfplayJob("square19", 2, 15, "descent-only", 800),

    # Square19 3p/4p
    "square19_3p_mixed": SelfplayJob("square19", 3, 15, "mixed", 1000),
    "square19_4p_mixed": SelfplayJob("square19", 4, 10, "mixed", 1200),

    # Hexagonal - Similar to square19
    "hex_2p_mixed": SelfplayJob("hexagonal", 2, 20, "mixed", 800),
    "hex_2p_heuristic": SelfplayJob("hexagonal", 2, 15, "heuristic-only", 800),
    "hex_2p_descent": SelfplayJob("hexagonal", 2, 15, "descent-only", 800),

    # Hex 3p/4p
    "hex_3p_mixed": SelfplayJob("hexagonal", 3, 15, "mixed", 1000),
    "hex_4p_mixed": SelfplayJob("hexagonal", 4, 10, "mixed", 1200),
}


class PipelineOrchestrator:
    """Master orchestrator for the AI training pipeline."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        state_path: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.config = self._load_config(config_path) if config_path else {}
        self.state_path = state_path or "logs/pipeline/state.json"
        self.state = self._load_state()
        self.dry_run = dry_run
        self.log_dir = Path("logs/pipeline")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration from JSON."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r") as f:
            return json.load(f)

    def _load_state(self) -> PipelineState:
        """Load pipeline state from disk."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                return PipelineState(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        return PipelineState()

    def _save_state(self) -> None:
        """Save pipeline state to disk."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state.__dict__, f, indent=2, default=str)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[ERROR]", "OK": "[OK]"}
        print(f"{timestamp} {prefix.get(level, '[???]')} {message}")

    async def run_remote_command(
        self,
        worker: WorkerConfig,
        command: str,
        background: bool = False,
    ) -> Tuple[int, str, str]:
        """Run a command on a remote worker via SSH."""
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=10"]
        if worker.ssh_key:
            ssh_cmd.extend(["-i", worker.ssh_key])
        if worker.ssh_port != 22:
            ssh_cmd.extend(["-p", str(worker.ssh_port)])
        ssh_cmd.append(worker.host)

        if background:
            # Wrap command with nohup for background execution
            full_cmd = f"cd {worker.remote_path} && nohup bash -c '{command}' > /dev/null 2>&1 &"
        else:
            full_cmd = f"cd {worker.remote_path} && {command}"

        ssh_cmd.append(full_cmd)

        if self.dry_run:
            self.log(f"[DRY RUN] {worker.name}: {full_cmd[:100]}...")
            return 0, "", ""

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode or 0, stdout.decode(), stderr.decode()
        except Exception as e:
            return 1, "", str(e)

    async def check_worker_health(self, worker: WorkerConfig) -> bool:
        """Check if a worker is reachable and healthy."""
        code, stdout, stderr = await self.run_remote_command(worker, "echo 'healthy'")
        return code == 0 and "healthy" in stdout

    async def get_worker_game_count(self, worker: WorkerConfig) -> int:
        """Get the total game count from a worker's selfplay.db."""
        cmd = "source venv/bin/activate && sqlite3 data/games/selfplay.db 'SELECT COUNT(*) FROM games WHERE status=\"completed\"' 2>/dev/null || echo 0"
        code, stdout, _ = await self.run_remote_command(worker, cmd)
        if code == 0:
            try:
                return int(stdout.strip())
            except ValueError:
                pass
        return 0

    async def dispatch_selfplay(
        self,
        worker: WorkerConfig,
        job: SelfplayJob,
        iteration: int,
        job_key: str,
    ) -> bool:
        """Dispatch a selfplay job to a worker.

        The job will use trained heuristic profiles if available on the worker,
        and neural networks if the job requests them and checkpoints exist.
        """
        seed = iteration * 10000 + job.seed
        log_file = f"logs/selfplay/iter{iteration}_{job_key}.jsonl"

        # Build optional flags
        optional_flags = []

        # Use trained heuristic profiles if available
        if job.use_trained_profiles:
            # Workers should have trained_heuristic_profiles.json synced from profile-sync phase
            optional_flags.append(
                f"--heuristic-weights-file {worker.remote_path}/data/trained_heuristic_profiles.json"
            )
            # Use board-specific profile key
            board_abbrev = {"square8": "sq8", "square19": "sq19", "hexagonal": "hex"}.get(
                job.board_type, job.board_type[:3]
            )
            profile_key = f"heuristic_v1_{board_abbrev}_{job.num_players}p"
            optional_flags.append(f"--heuristic-profile {profile_key}")

        # Enable neural network for nn-only or when explicitly requested
        env_extras = ""
        if job.use_neural_net or job.engine_mode == "nn-only":
            env_extras = "export RINGRIFT_USE_NEURAL_NET=1\n"

        flags_str = " \\\n    ".join(optional_flags) if optional_flags else ""

        cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
export RINGRIFT_TRAINED_HEURISTIC_PROFILES={worker.remote_path}/data/trained_heuristic_profiles.json
{env_extras}python scripts/run_self_play_soak.py \\
    --num-games {job.num_games} \\
    --board-type {job.board_type} \\
    --engine-mode {job.engine_mode} \\
    --num-players {job.num_players} \\
    --max-moves {job.max_moves} \\
    --seed {seed} \\
    --log-jsonl {log_file} \\
    {flags_str}
"""
        self.log(f"Dispatching {job_key} ({job.num_games} {job.engine_mode} games) to {worker.name}")

        code, _, stderr = await self.run_remote_command(worker, cmd, background=True)
        if code != 0:
            self.log(f"Failed to dispatch to {worker.name}: {stderr}", "ERROR")
            return False
        return True

    async def run_selfplay_phase(self, iteration: int) -> Dict[str, int]:
        """Run the selfplay phase across all workers."""
        self.state.phase = "selfplay"
        self._save_state()
        self.log(f"=== Starting Selfplay Phase (Iteration {iteration}) ===")

        # Check worker health
        healthy_workers = []
        for worker in WORKERS:
            if worker.role in ["selfplay", "mixed"]:
                if await self.check_worker_health(worker):
                    healthy_workers.append(worker)
                    self.log(f"Worker {worker.name}: healthy", "OK")
                else:
                    self.log(f"Worker {worker.name}: unreachable", "WARN")

        if not healthy_workers:
            self.log("No healthy workers available!", "ERROR")
            return {}

        # Distribute jobs across workers with round-robin assignment
        # This ensures all job types get dispatched, not just the first few
        job_items = list(DEFAULT_SELFPLAY_CONFIG.items())
        tasks = []
        worker_idx = 0

        for job_key, job in job_items:
            if not healthy_workers:
                break
            # Round-robin worker assignment
            worker = healthy_workers[worker_idx % len(healthy_workers)]
            tasks.append(self.dispatch_selfplay(worker, job, iteration, job_key))
            worker_idx += 1

        # Wait for all dispatches to complete
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)
        self.log(f"Dispatched {success_count}/{len(tasks)} selfplay jobs across {len(healthy_workers)} workers")

        return {"dispatched": success_count, "total": len(tasks)}

    async def run_sync_phase(self) -> bool:
        """Sync all selfplay AND tournament data from remote workers.

        This phase pulls:
        1. Selfplay game databases from workers (via sync_selfplay_data.sh)
        2. Tournament JSONL files from workers (for training data)

        Both sources are merged into the training data pool.
        """
        self.state.phase = "sync"
        self._save_state()
        self.log("=== Starting Data Sync Phase ===")

        success = True

        # Step 1: Sync selfplay databases using existing script
        sync_script = Path(__file__).parent / "sync_selfplay_data.sh"
        if sync_script.exists():
            if self.dry_run:
                self.log(f"[DRY RUN] Would run: {sync_script} --merge --to-mac-studio")
            else:
                try:
                    result = subprocess.run(
                        [str(sync_script), "--merge", "--to-mac-studio"],
                        capture_output=True,
                        text=True,
                        timeout=1800,  # 30 minute timeout
                    )
                    if result.returncode == 0:
                        self.log("Selfplay DB sync completed", "OK")
                    else:
                        self.log(f"Selfplay sync failed: {result.stderr[:200]}", "ERROR")
                        success = False
                except subprocess.TimeoutExpired:
                    self.log("Selfplay sync timed out", "ERROR")
                    success = False
                except Exception as e:
                    self.log(f"Selfplay sync error: {e}", "ERROR")
                    success = False
        else:
            self.log(f"Sync script not found: {sync_script}", "WARN")

        # Step 2: Sync tournament JSONL files from workers
        await self.sync_tournament_games()

        self.state.last_sync = datetime.now().isoformat()
        self._save_state()
        return success

    async def sync_tournament_games(self) -> int:
        """Pull tournament game JSONL files from all workers and merge locally.

        Tournament games are high-quality training data from strong AI matchups.
        They are saved as JSONL files in logs/tournaments/ on each worker.
        """
        self.log("Syncing tournament games from workers...")

        local_tournament_dir = Path("logs/tournaments/merged")
        local_tournament_dir.mkdir(parents=True, exist_ok=True)

        merged_count = 0

        for worker in WORKERS:
            if not await self.check_worker_health(worker):
                continue

            # Find tournament JSONL files on worker
            find_cmd = f"find {worker.remote_path}/logs/tournaments -name 'games.jsonl' 2>/dev/null || true"
            code, stdout, _ = await self.run_remote_command(worker, find_cmd)

            if code != 0 or not stdout.strip():
                continue

            jsonl_files = stdout.strip().split("\n")
            self.log(f"  {worker.name}: Found {len(jsonl_files)} tournament files")

            for remote_path in jsonl_files:
                if not remote_path.strip():
                    continue

                # Read remote file content
                cat_cmd = f"cat {remote_path}"
                code, content, _ = await self.run_remote_command(worker, cat_cmd)

                if code != 0 or not content.strip():
                    continue

                # Append to local merged file
                merged_file = local_tournament_dir / "all_tournaments.jsonl"
                with open(merged_file, "a") as f:
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")

                lines = len(content.strip().split("\n"))
                merged_count += lines

        if merged_count > 0:
            self.log(f"Merged {merged_count} tournament games to {local_tournament_dir}", "OK")

        return merged_count

    async def sync_heuristic_profiles(self) -> bool:
        """Sync trained heuristic profiles from all workers.

        After CMA-ES completes on remote workers, this method pulls the
        trained_heuristic_profiles.json from each worker and merges them
        into a unified local copy. The merged profiles are then pushed
        back to all workers so they use the latest heuristics for selfplay.
        """
        self.log("=== Syncing Trained Heuristic Profiles ===")

        local_profiles_path = Path("data/trained_heuristic_profiles.json")
        merged_profiles: Dict[str, Any] = {
            "version": "1.3.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "updated": datetime.now().strftime("%Y-%m-%d"),
            "description": "CMA-ES optimized heuristic profiles (merged from distributed workers)",
            "profiles": {},
            "training_metadata": {},
        }

        # Load existing local profiles
        if local_profiles_path.exists():
            try:
                with open(local_profiles_path) as f:
                    existing = json.load(f)
                    merged_profiles["profiles"].update(existing.get("profiles", {}))
                    merged_profiles["training_metadata"].update(existing.get("training_metadata", {}))
            except (json.JSONDecodeError, OSError) as e:
                self.log(f"Warning: Could not load existing profiles: {e}", "WARN")

        # Pull profiles from each worker
        pull_count = 0
        for worker in WORKERS:
            if not await self.check_worker_health(worker):
                continue

            # Fetch remote profiles
            cmd = f"cat {worker.remote_path}/data/trained_heuristic_profiles.json 2>/dev/null || echo '{{}}'"
            code, stdout, _ = await self.run_remote_command(worker, cmd)

            if code == 0 and stdout.strip() and stdout.strip() != "{}":
                try:
                    remote_profiles = json.loads(stdout)
                    remote_data = remote_profiles.get("profiles", {})
                    remote_meta = remote_profiles.get("training_metadata", {})

                    # Merge profiles, preferring higher fitness
                    for key, weights in remote_data.items():
                        existing_meta = merged_profiles["training_metadata"].get(key, {})
                        new_meta = remote_meta.get(key, {})

                        existing_fitness = existing_meta.get("fitness", 0)
                        new_fitness = new_meta.get("fitness", 0)

                        if new_fitness > existing_fitness or key not in merged_profiles["profiles"]:
                            merged_profiles["profiles"][key] = weights
                            merged_profiles["training_metadata"][key] = new_meta
                            self.log(f"  Merged {key} from {worker.name} (fitness: {new_fitness:.3f})")
                            pull_count += 1
                except json.JSONDecodeError as e:
                    self.log(f"Warning: Invalid JSON from {worker.name}: {e}", "WARN")

        if pull_count > 0:
            # Save merged profiles locally
            local_profiles_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_profiles_path, "w") as f:
                json.dump(merged_profiles, f, indent=2)
            self.log(f"Saved merged profiles to {local_profiles_path}", "OK")

            # Push merged profiles back to all workers
            profiles_json = json.dumps(merged_profiles)
            push_count = 0
            for worker in WORKERS:
                if not await self.check_worker_health(worker):
                    continue

                push_cmd = f"mkdir -p {worker.remote_path}/data && cat > {worker.remote_path}/data/trained_heuristic_profiles.json << 'EOFPROFILES'\n{profiles_json}\nEOFPROFILES"
                code, _, _ = await self.run_remote_command(worker, push_cmd)
                if code == 0:
                    push_count += 1

            self.log(f"Pushed merged profiles to {push_count} workers", "OK")

        return pull_count > 0

    async def run_training_phase(self, iteration: int) -> Dict[str, bool]:
        """Run NN training on Mac Studio for all board×player configurations.

        Trains neural networks for each configuration that has sufficient
        training data. Prioritizes square8 2p (most data) then expands to
        other configs based on available games.
        """
        self.state.phase = "training"
        self._save_state()
        self.log(f"=== Starting Training Phase (Iteration {iteration}) ===")

        results = {}

        # Find Mac Studio worker (or any training-capable worker)
        training_worker = next(
            (w for w in WORKERS if w.role in ["training", "mixed"] and w.name == "mac-studio"),
            next((w for w in WORKERS if w.role in ["training", "mixed"]), None)
        )
        if not training_worker:
            self.log("No training worker configured", "ERROR")
            return results

        if not await self.check_worker_health(training_worker):
            self.log(f"{training_worker.name} not reachable", "ERROR")
            return results

        # Training configurations: (board, players, epochs, min_games_required)
        # More epochs for primary configs, fewer for secondary
        TRAINING_CONFIGS = [
            # Primary configs - full training
            ("square8", 2, 50, 100),
            ("square8", 3, 40, 50),
            ("square8", 4, 40, 40),
            # Secondary configs - lighter training
            ("square19", 2, 30, 50),
            ("hexagonal", 2, 30, 50),
            # Tertiary configs - minimal training if data available
            ("square19", 3, 20, 30),
            ("square19", 4, 20, 20),
            ("hexagonal", 3, 20, 30),
            ("hexagonal", 4, 20, 20),
        ]

        for board, players, epochs, min_games in TRAINING_CONFIGS:
            config_key = f"{board}_{players}p"

            # Check if sufficient training data exists
            check_cmd = f"sqlite3 {training_worker.remote_path}/data/games/merged_latest.db \"SELECT COUNT(*) FROM games WHERE board_type='{board}' AND num_players={players} AND status='completed'\" 2>/dev/null || echo 0"
            code, stdout, _ = await self.run_remote_command(training_worker, check_cmd)
            game_count = int(stdout.strip()) if code == 0 and stdout.strip().isdigit() else 0

            if game_count < min_games:
                self.log(f"{config_key}: Skipping (only {game_count}/{min_games} games)", "WARN")
                results[config_key] = False
                continue

            # Train this configuration
            train_cmd = f"""
source venv/bin/activate
export PYTHONPATH={training_worker.remote_path}
export RINGRIFT_TRAINED_HEURISTIC_PROFILES={training_worker.remote_path}/data/trained_heuristic_profiles.json
python -m app.training.train \\
    --data-path data/games/merged_latest.db \\
    --board-type {board} \\
    --num-players {players} \\
    --epochs {epochs} \\
    --batch-size 256 \\
    --device mps \\
    --save-path models/{config_key}_iter{iteration}.pth \\
    --save-best models/{config_key}_best.pth
"""
            self.log(f"Training {config_key} ({game_count} games, {epochs} epochs)...")
            code, stdout, stderr = await self.run_remote_command(training_worker, train_cmd, background=False)
            results[config_key] = code == 0

            if code == 0:
                self.log(f"{config_key} training completed", "OK")
                self.state.models_trained.append(f"{config_key}_iter{iteration}")
            else:
                self.log(f"{config_key} training failed: {stderr[:200]}", "ERROR")

        self._save_state()
        return results

    async def run_cmaes_phase(self, iteration: int) -> Dict[str, bool]:
        """Run CMA-ES optimization across all board×player configurations.

        This phase runs iterative CMA-ES heuristic tuning for each of the 9
        board×player combinations. Jobs are distributed across available
        workers based on their capabilities (some workers may not support
        larger boards due to memory constraints).

        The trained heuristic profiles are saved to data/trained_heuristic_profiles.json
        and automatically loaded by subsequent selfplay phases.
        """
        self.state.phase = "cmaes"
        self._save_state()
        self.log(f"=== Starting CMA-ES Phase (Iteration {iteration}) ===")

        results = {}

        # Define all 9 board×player CMA-ES configurations
        # Format: (board, num_players, generations_per_iter, max_iterations, capabilities_required)
        CMAES_CONFIGS = [
            # Square8 - can run on any worker (low memory)
            ("square8", 2, 15, 5, ["square8"]),
            ("square8", 3, 12, 4, ["square8"]),
            ("square8", 4, 10, 4, ["square8"]),
            # Square19 - requires more memory, LAN workers preferred
            ("square19", 2, 12, 4, ["square19"]),
            ("square19", 3, 10, 3, ["square19"]),
            ("square19", 4, 8, 3, ["square19"]),
            # Hexagonal - similar to square19
            ("hexagonal", 2, 12, 4, ["hex"]),
            ("hexagonal", 3, 10, 3, ["hex"]),
            ("hexagonal", 4, 8, 3, ["hex"]),
        ]

        # Find workers capable of CMA-ES
        cmaes_workers = []
        for worker in WORKERS:
            if worker.role in ["cmaes", "mixed", "training"]:
                if await self.check_worker_health(worker):
                    cmaes_workers.append(worker)
                    self.log(f"CMA-ES worker {worker.name}: healthy (caps: {worker.capabilities})", "OK")
                else:
                    self.log(f"CMA-ES worker {worker.name}: unreachable", "WARN")

        if not cmaes_workers:
            self.log("No healthy CMA-ES workers available!", "ERROR")
            return results

        # Match configs to capable workers
        async def dispatch_cmaes_job(
            worker: WorkerConfig,
            board: str,
            num_players: int,
            gens_per_iter: int,
            max_iters: int,
        ) -> Tuple[str, bool]:
            """Dispatch a single CMA-ES job to a worker."""
            config_key = f"{board}_{num_players}p"
            output_dir = f"logs/cmaes/iter{iteration}/{config_key}"

            # Determine worker URLs for distributed mode
            # Use all workers that support this board type
            compatible_workers = [
                w for w in cmaes_workers
                if any(cap in w.capabilities for cap in [board[:3], board, "all"])
            ]
            worker_urls = ",".join([
                f"http://{w.host.split('@')[-1]}:8765"
                for w in compatible_workers
                if w != worker  # Exclude self for distributed workers
            ]) if len(compatible_workers) > 1 else ""

            distributed_flag = "--distributed" if worker_urls else ""
            workers_arg = f"--workers {worker_urls}" if worker_urls else ""

            cmaes_cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
python scripts/run_iterative_cmaes.py \\
    --board {board} \\
    --num-players {num_players} \\
    --generations-per-iter {gens_per_iter} \\
    --max-iterations {max_iters} \\
    --population-size 14 \\
    --games-per-eval 8 \\
    --sigma 0.5 \\
    --output-dir {output_dir} \\
    {distributed_flag} {workers_arg}
"""
            self.log(f"Dispatching CMA-ES {config_key} to {worker.name}...")
            code, _, stderr = await self.run_remote_command(worker, cmaes_cmd, background=True)

            if code == 0:
                self.log(f"CMA-ES {config_key} dispatched to {worker.name}", "OK")
                return config_key, True
            else:
                self.log(f"CMA-ES {config_key} dispatch failed: {stderr[:200]}", "ERROR")
                return config_key, False

        # Distribute jobs across workers, matching capabilities
        tasks = []
        worker_idx = 0

        for board, num_players, gens, max_iters, required_caps in CMAES_CONFIGS:
            # Find a worker with required capabilities
            capable_worker = None
            for i in range(len(cmaes_workers)):
                worker = cmaes_workers[(worker_idx + i) % len(cmaes_workers)]
                # Check if worker has required capability (or "all")
                if "all" in worker.capabilities or any(
                    cap in worker.capabilities for cap in required_caps
                ):
                    capable_worker = worker
                    worker_idx = (worker_idx + i + 1) % len(cmaes_workers)
                    break

            if capable_worker:
                tasks.append(dispatch_cmaes_job(
                    capable_worker, board, num_players, gens, max_iters
                ))
            else:
                config_key = f"{board}_{num_players}p"
                self.log(f"No worker capable of {config_key} (needs {required_caps})", "WARN")
                results[config_key] = False

        # Execute all dispatches concurrently
        if tasks:
            dispatch_results = await asyncio.gather(*tasks)
            for config_key, success in dispatch_results:
                results[config_key] = success

        # Summary
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        self.log(f"CMA-ES phase: dispatched {success_count}/{total_count} jobs")

        return results

    async def run_evaluation_phase(self, iteration: int) -> Dict[str, float]:
        """Run evaluation tournaments and SAVE games for training.

        This phase serves two purposes:
        1. Evaluate model/heuristic strength via head-to-head tournaments
        2. Generate high-quality training data from strong AI matchups

        Tournament games are saved to logs/tournaments/iter{N}/ and should be
        merged into training data during the sync phase.
        """
        self.state.phase = "evaluation"
        self._save_state()
        self.log(f"=== Starting Evaluation Phase (Iteration {iteration}) ===")

        results = {}

        # Find workers for evaluation (distribute tournaments)
        eval_workers = [w for w in WORKERS if await self.check_worker_health(w)]
        if not eval_workers:
            self.log("No healthy workers for evaluation", "ERROR")
            return results

        # Comprehensive tournament matchups for diverse training data
        # Format: (p1_type, p1_diff, p2_type, p2_diff, board, games)
        # Higher difficulty = stronger AI = higher quality games
        TOURNAMENT_MATCHUPS = [
            # Cross-AI-type matchups (generates diverse training data)
            ("Heuristic", 5, "MCTS", 6, "Square8", 10),
            ("Heuristic", 5, "Minimax", 5, "Square8", 10),
            ("MCTS", 6, "Minimax", 5, "Square8", 10),

            # Same-type tier progression (measures improvement)
            ("Heuristic", 3, "Heuristic", 5, "Square8", 8),
            ("MCTS", 5, "MCTS", 7, "Square8", 8),

            # Multi-board coverage
            ("Heuristic", 5, "MCTS", 5, "Square19", 6),
            ("Heuristic", 5, "MCTS", 5, "Hex", 6),

            # Strong vs strong (highest quality games)
            ("MCTS", 7, "MCTS", 8, "Square8", 6),
        ]

        # Distribute matchups across workers
        tasks = []
        for idx, (p1, p1d, p2, p2d, board, games) in enumerate(TOURNAMENT_MATCHUPS):
            worker = eval_workers[idx % len(eval_workers)]
            matchup_key = f"{p1}{p1d}_vs_{p2}{p2d}_{board}"
            output_dir = f"logs/tournaments/iter{iteration}/{matchup_key}"

            eval_cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
export RINGRIFT_TRAINED_HEURISTIC_PROFILES={worker.remote_path}/data/trained_heuristic_profiles.json
mkdir -p {output_dir}
python scripts/run_ai_tournament.py \\
    --p1 {p1} --p1-diff {p1d} \\
    --p2 {p2} --p2-diff {p2d} \\
    --board {board} \\
    --games {games} \\
    --output-dir {output_dir}
"""
            self.log(f"Dispatching tournament {matchup_key} to {worker.name}")
            tasks.append(self._run_tournament(worker, eval_cmd, matchup_key))

        # Run all tournaments concurrently
        tournament_results = await asyncio.gather(*tasks)
        for matchup_key, success, win_rate in tournament_results:
            results[matchup_key] = win_rate if success else 0.0

        # Summary
        success_count = sum(1 for _, s, _ in tournament_results if s)
        self.log(f"Evaluation: {success_count}/{len(TOURNAMENT_MATCHUPS)} tournaments completed")

        return results

    async def _run_tournament(
        self,
        worker: WorkerConfig,
        cmd: str,
        matchup_key: str,
    ) -> Tuple[str, bool, float]:
        """Run a single tournament and parse results."""
        code, stdout, stderr = await self.run_remote_command(worker, cmd)
        if code != 0:
            self.log(f"Tournament {matchup_key} failed: {stderr[:100]}", "ERROR")
            return matchup_key, False, 0.0

        # Parse win rate from output (e.g., "P1 wins: 6/10 (60.0%)")
        import re
        match = re.search(r"(\d+\.?\d*)%", stdout)
        win_rate = float(match.group(1)) / 100 if match else 0.5

        self.log(f"Tournament {matchup_key}: {win_rate:.1%} P1 win rate", "OK")
        return matchup_key, True, win_rate

    async def run_full_iteration(self, iteration: int) -> bool:
        """Run a complete pipeline iteration.

        Pipeline flow:
        1. Selfplay: Generate games with current models/heuristics
        2. Sync: Pull selfplay data from workers
        3. CMA-ES: Optimize heuristics for all 9 board×player configs (background)
        4. NN Training: Train neural network on new data (parallel with CMA-ES wait)
        5. Profile Sync: Pull trained heuristics, merge, push to all workers
        6. Evaluation: Tournament to compare new vs old models
        """
        self.log(f"\n{'='*60}")
        self.log(f"=== Pipeline Iteration {iteration} ===")
        self.log(f"{'='*60}\n")

        self.state.iteration = iteration
        self._save_state()

        # Phase 1: Selfplay - Generate games across all board×player configs
        selfplay_result = await self.run_selfplay_phase(iteration)
        if not selfplay_result.get("dispatched", 0):
            self.log("Selfplay phase failed", "ERROR")
            return False

        # Wait for selfplay jobs to accumulate data
        if not self.dry_run:
            self.log("Waiting for selfplay jobs to generate data...")
            await asyncio.sleep(300)  # 5 minutes initial wait

        # Phase 2: Sync selfplay data from workers
        if not await self.run_sync_phase():
            self.log("Sync phase failed, continuing...", "WARN")

        # Phase 3: CMA-ES heuristic optimization (dispatches background jobs)
        # This runs all 9 board×player configurations across available workers
        cmaes_results = await self.run_cmaes_phase(iteration)

        # Phase 4: NN Training (runs while CMA-ES jobs process in background)
        training_results = await self.run_training_phase(iteration)

        # Phase 5: Wait for CMA-ES completion and sync heuristic profiles
        if any(cmaes_results.values()):
            if not self.dry_run:
                # Wait for CMA-ES jobs to complete (they run in background)
                # CMA-ES typically takes 30-60 minutes per config
                cmaes_wait_minutes = 45
                self.log(f"Waiting {cmaes_wait_minutes} min for CMA-ES jobs to complete...")
                await asyncio.sleep(cmaes_wait_minutes * 60)

            # Pull trained heuristic profiles from all workers and merge
            await self.sync_heuristic_profiles()

        # Phase 6: Evaluation - Compare models/heuristics
        eval_results = await self.run_evaluation_phase(iteration)

        self.state.phase = "complete"
        self._save_state()

        self.log(f"\n{'='*60}")
        self.log(f"=== Iteration {iteration} Complete ===")
        self.log(f"{'='*60}")
        self.log(f"Selfplay:  {selfplay_result}")
        self.log(f"CMA-ES:    {cmaes_results}")
        self.log(f"Training:  {training_results}")
        self.log(f"Eval:      {eval_results}")
        self.log(f"{'='*60}\n")

        return True

    async def run(
        self,
        iterations: int = 1,
        start_iteration: int = 0,
        phase: Optional[str] = None,
    ) -> None:
        """Run the pipeline."""
        self.log("RingRift AI Training Pipeline")
        self.log(f"Iterations: {iterations}, Start: {start_iteration}")
        if self.dry_run:
            self.log("*** DRY RUN MODE - No commands will be executed ***")

        if phase:
            # Run single phase
            self.log(f"Running single phase: {phase}")
            iteration = start_iteration or self.state.iteration
            if phase == "selfplay":
                await self.run_selfplay_phase(iteration)
            elif phase == "sync":
                await self.run_sync_phase()
            elif phase == "training":
                await self.run_training_phase(iteration)
            elif phase == "cmaes":
                await self.run_cmaes_phase(iteration)
            elif phase == "profile-sync":
                await self.sync_heuristic_profiles()
            elif phase == "evaluation":
                await self.run_evaluation_phase(iteration)
            else:
                self.log(f"Unknown phase: {phase}", "ERROR")
            return

        # Run full iterations
        for i in range(start_iteration, start_iteration + iterations):
            success = await self.run_full_iteration(i)
            if not success:
                self.log(f"Iteration {i} failed, stopping pipeline", "ERROR")
                break

        self.log("Pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="RingRift AI Training Pipeline Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline configuration JSON",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of pipeline iterations to run",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="Starting iteration number",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["selfplay", "sync", "training", "cmaes", "profile-sync", "evaluation"],
        help="Run only a specific phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default="logs/pipeline/state.json",
        help="Path to state file",
    )
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        config_path=args.config,
        state_path=args.state_path,
        dry_run=args.dry_run,
    )

    asyncio.run(
        orchestrator.run(
            iterations=args.iterations,
            start_iteration=args.start_iteration,
            phase=args.phase,
        )
    )


if __name__ == "__main__":
    main()
