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
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    engine_mode: str = "descent-only"
    max_moves: int = 500
    seed: int = 0


@dataclass
class PipelineState:
    """Current state of the pipeline."""

    iteration: int = 0
    phase: str = "idle"
    games_generated: Dict[str, int] = field(default_factory=dict)
    models_trained: List[str] = field(default_factory=list)
    last_sync: Optional[str] = None
    errors: List[str] = field(default_factory=list)


# Worker configurations - based on PLAN.md cluster
# Local machines use Tailscale IPs (100.x.x.x)
# Cloud machines use public IPs
WORKERS = [
    # === Local Mac Machines (Tailscale) ===
    WorkerConfig(
        name="mac-studio",
        host="armand@100.107.168.125",
        role="mixed",  # training + selfplay
        capabilities=["square8", "square19", "hexagonal", "nn", "nnue", "mps"],
        ssh_key="~/.ssh/id_cluster",
        remote_path="~/Development/RingRift/ai-service",
        max_parallel_jobs=4,
    ),
    WorkerConfig(
        name="mbp-16gb",
        host="armand@100.66.142.46",
        role="selfplay",
        capabilities=["square8", "square19", "hexagonal"],
        remote_path="~/Development/RingRift/ai-service",
        max_parallel_jobs=2,
    ),
    WorkerConfig(
        name="mbp-64gb",
        host="armand@100.92.222.49",
        role="selfplay",
        capabilities=["square8", "square19", "hexagonal"],
        remote_path="~/Development/RingRift/ai-service",
        max_parallel_jobs=4,
    ),
    # === AWS Instances ===
    WorkerConfig(
        name="aws-staging",
        host="ubuntu@54.198.219.106",
        role="selfplay",
        capabilities=["square8", "square19", "hexagonal"],
        ssh_key="~/.ssh/ringrift-staging-key.pem",
        remote_path="~/ringrift/ai-service",
        max_parallel_jobs=4,
    ),
    WorkerConfig(
        name="aws-extra",
        host="ubuntu@3.208.88.21",
        role="selfplay",
        capabilities=["square8", "square19", "hexagonal"],
        ssh_key="~/.ssh/ringrift-staging-key.pem",
        remote_path="~/ringrift/ai-service",
        max_parallel_jobs=2,
    ),
    # === Lambda Labs GPU Instances ===
    WorkerConfig(
        name="lambda-h100",
        host="ubuntu@209.20.157.81",
        role="mixed",  # GPU selfplay + training
        capabilities=["square8", "square19", "hexagonal", "gpu", "nn"],
        remote_path="~/ringrift/ai-service",
        max_parallel_jobs=8,
    ),
    WorkerConfig(
        name="lambda-a10",
        host="ubuntu@150.136.65.197",
        role="selfplay",
        capabilities=["square8", "square19", "hexagonal", "gpu"],
        remote_path="~/ringrift/ai-service",
        max_parallel_jobs=4,
    ),
    # === Vast.ai GPU Instance ===
    WorkerConfig(
        name="vast-3090",
        host="root@79.116.93.241",
        role="selfplay",
        capabilities=["square8", "square19", "hexagonal", "gpu"],
        ssh_port=47070,  # Non-standard Vast.ai port
        remote_path="~/ringrift/ai-service",
        max_parallel_jobs=4,
    ),
]

# Default selfplay configuration per iteration
DEFAULT_SELFPLAY_CONFIG = {
    "square8_2p": SelfplayJob("square8", 2, 100, max_moves=500),
    "square8_3p": SelfplayJob("square8", 3, 60, max_moves=600),
    "square8_4p": SelfplayJob("square8", 4, 40, max_moves=700),
    "square19_2p": SelfplayJob("square19", 2, 50, max_moves=800),
    "square19_3p": SelfplayJob("square19", 3, 30, max_moves=1000),
    "square19_4p": SelfplayJob("square19", 4, 20, max_moves=1200),
    "hex_2p": SelfplayJob("hexagonal", 2, 50, max_moves=800),
    "hex_3p": SelfplayJob("hexagonal", 3, 30, max_moves=1000),
    "hex_4p": SelfplayJob("hexagonal", 4, 20, max_moves=1200),
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
    ) -> bool:
        """Dispatch a selfplay job to a worker."""
        seed = iteration * 10000 + job.seed
        log_file = f"logs/selfplay/iter{iteration}_{job.board_type}_{job.num_players}p.jsonl"

        cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
python scripts/run_self_play_soak.py \
    --num-games {job.num_games} \
    --board-type {job.board_type} \
    --engine-mode {job.engine_mode} \
    --num-players {job.num_players} \
    --max-moves {job.max_moves} \
    --seed {seed} \
    --log-jsonl {log_file}
"""
        self.log(f"Dispatching {job.board_type} {job.num_players}p ({job.num_games} games) to {worker.name}")

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

        # Distribute jobs across workers
        jobs = list(DEFAULT_SELFPLAY_CONFIG.values())
        job_idx = 0
        tasks = []

        for worker in healthy_workers:
            for _ in range(worker.max_parallel_jobs):
                if job_idx >= len(jobs):
                    break
                job = jobs[job_idx]
                tasks.append(self.dispatch_selfplay(worker, job, iteration))
                job_idx += 1

        # Wait for all dispatches to complete
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)
        self.log(f"Dispatched {success_count}/{len(tasks)} selfplay jobs")

        return {"dispatched": success_count, "total": len(tasks)}

    async def run_sync_phase(self) -> bool:
        """Sync all selfplay data from remote workers to local."""
        self.state.phase = "sync"
        self._save_state()
        self.log("=== Starting Data Sync Phase ===")

        sync_script = Path(__file__).parent / "sync_selfplay_data.sh"
        if not sync_script.exists():
            self.log(f"Sync script not found: {sync_script}", "ERROR")
            return False

        if self.dry_run:
            self.log(f"[DRY RUN] Would run: {sync_script} --merge --to-mac-studio")
            return True

        try:
            result = subprocess.run(
                [str(sync_script), "--merge", "--to-mac-studio"],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )
            if result.returncode == 0:
                self.log("Data sync completed successfully", "OK")
                self.state.last_sync = datetime.now().isoformat()
                self._save_state()
                return True
            else:
                self.log(f"Sync failed: {result.stderr}", "ERROR")
                return False
        except subprocess.TimeoutExpired:
            self.log("Sync timed out after 30 minutes", "ERROR")
            return False
        except Exception as e:
            self.log(f"Sync error: {e}", "ERROR")
            return False

    async def run_training_phase(self, iteration: int) -> Dict[str, bool]:
        """Run NN training on Mac Studio."""
        self.state.phase = "training"
        self._save_state()
        self.log(f"=== Starting Training Phase (Iteration {iteration}) ===")

        results = {}

        # Find Mac Studio worker
        mac_studio = next((w for w in WORKERS if w.name == "mac-studio"), None)
        if not mac_studio:
            self.log("Mac Studio worker not configured", "ERROR")
            return results

        if not await self.check_worker_health(mac_studio):
            self.log("Mac Studio not reachable", "ERROR")
            return results

        # Train square8 2p model (primary)
        train_cmd = f"""
source venv/bin/activate
export PYTHONPATH={mac_studio.remote_path}
python app/training/train.py \
    --data-path data/games/merged_latest.db \
    --board-type square8 \
    --num-players 2 \
    --epochs 50 \
    --batch-size 256 \
    --device mps \
    --save-path models/square8_2p_iter{iteration}.pth
"""
        self.log("Starting NN training for square8 2p...")
        code, stdout, stderr = await self.run_remote_command(mac_studio, train_cmd, background=False)
        results["square8_2p"] = code == 0
        if code == 0:
            self.log("square8 2p training completed", "OK")
            self.state.models_trained.append(f"square8_2p_iter{iteration}")
        else:
            self.log(f"square8 2p training failed: {stderr[:200]}", "ERROR")

        self._save_state()
        return results

    async def run_cmaes_phase(self, iteration: int) -> Dict[str, bool]:
        """Run CMA-ES optimization on staging."""
        self.state.phase = "cmaes"
        self._save_state()
        self.log(f"=== Starting CMA-ES Phase (Iteration {iteration}) ===")

        results = {}

        # Find staging worker
        staging = next((w for w in WORKERS if w.name == "staging"), None)
        if not staging:
            self.log("Staging worker not configured", "ERROR")
            return results

        if not await self.check_worker_health(staging):
            self.log("Staging not reachable", "ERROR")
            return results

        # Run CMA-ES for square8 2p
        cmaes_cmd = f"""
source venv/bin/activate
export PYTHONPATH={staging.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
python scripts/run_iterative_cmaes.py \
    --board square8 \
    --num-players 2 \
    --generations-per-iter 10 \
    --max-iterations 3 \
    --output-dir logs/cmaes/iter{iteration}/square8_2p
"""
        self.log("Starting CMA-ES optimization for square8 2p...")
        code, _, stderr = await self.run_remote_command(staging, cmaes_cmd, background=True)
        results["square8_2p"] = code == 0
        if code == 0:
            self.log("CMA-ES square8 2p dispatched", "OK")
        else:
            self.log(f"CMA-ES dispatch failed: {stderr[:200]}", "ERROR")

        return results

    async def run_evaluation_phase(self, iteration: int) -> Dict[str, float]:
        """Run evaluation tournaments to compare models."""
        self.state.phase = "evaluation"
        self._save_state()
        self.log(f"=== Starting Evaluation Phase (Iteration {iteration}) ===")

        results = {}

        # Find a worker for evaluation
        worker = next((w for w in WORKERS if await self.check_worker_health(w)), None)
        if not worker:
            self.log("No healthy workers for evaluation", "ERROR")
            return results

        # Run tournament between iterations
        eval_cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
python scripts/run_ai_tournament.py \
    --p1 Heuristic --p1-diff 5 \
    --p2 MCTS --p2-diff 5 \
    --board Square8 \
    --games 10 \
    --output logs/tournaments/eval_iter{iteration}.json
"""
        self.log("Running evaluation tournament...")
        code, stdout, _ = await self.run_remote_command(worker, eval_cmd)
        if code == 0:
            self.log("Evaluation completed", "OK")
            # Parse results (simplified)
            results["tournament_complete"] = 1.0
        else:
            self.log("Evaluation failed", "ERROR")

        return results

    async def run_full_iteration(self, iteration: int) -> bool:
        """Run a complete pipeline iteration."""
        self.log(f"\n{'='*60}")
        self.log(f"=== Pipeline Iteration {iteration} ===")
        self.log(f"{'='*60}\n")

        self.state.iteration = iteration
        self._save_state()

        # Phase 1: Selfplay
        selfplay_result = await self.run_selfplay_phase(iteration)
        if not selfplay_result.get("dispatched", 0):
            self.log("Selfplay phase failed", "ERROR")
            return False

        # Wait for selfplay jobs to accumulate data
        if not self.dry_run:
            self.log("Waiting for selfplay jobs to generate data...")
            await asyncio.sleep(300)  # 5 minutes initial wait

        # Phase 2: Sync
        if not await self.run_sync_phase():
            self.log("Sync phase failed, continuing...", "WARN")

        # Phase 3: Training (parallel with CMA-ES)
        training_task = asyncio.create_task(self.run_training_phase(iteration))
        cmaes_task = asyncio.create_task(self.run_cmaes_phase(iteration))

        training_results = await training_task
        cmaes_results = await cmaes_task

        # Phase 4: Evaluation
        eval_results = await self.run_evaluation_phase(iteration)

        self.state.phase = "complete"
        self._save_state()

        self.log(f"\n{'='*60}")
        self.log(f"=== Iteration {iteration} Complete ===")
        self.log(f"Selfplay: {selfplay_result}")
        self.log(f"Training: {training_results}")
        self.log(f"CMA-ES: {cmaes_results}")
        self.log(f"Evaluation: {eval_results}")
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
        choices=["selfplay", "sync", "training", "cmaes", "evaluation"],
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
