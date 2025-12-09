# Distributed Training Infrastructure Plan

**Created:** 2025-12-02
**Status:** Partial Implementation
**Purpose:** Preparatory work for distributed CMA-ES and neural network training (local cluster + cloud)

## Implementation Status

| Component                 | Status      | Notes                                           |
| ------------------------- | ----------- | ----------------------------------------------- |
| Local Mac Cluster         | Implemented | See `DISTRIBUTED_SELFPLAY.md`                   |
| `--mode` argument         | Implemented | Supports `local`, `lan`, `aws`, `hybrid`        |
| Host configuration        | Implemented | `config/distributed_hosts.yaml`                 |
| Memory-based filtering    | Implemented | Auto-filters hosts by board memory requirements |
| CMA-ES distributed        | Implemented | `run_cmaes_optimization.py --distributed`       |
| Iterative CMA-ES          | Implemented | `run_iterative_cmaes.py --distributed --mode`   |
| Self-play distributed     | Implemented | `run_distributed_selfplay_soak.py --mode`       |
| NNUE training distributed | Implemented | `run_distributed_nnue_training.py --mode`       |
| Cloud auto-scaling        | Planned     | Terraform/CDK for dynamic scaling               |
| GPU training              | Planned     | See Part 3 below                                |

---

## Executive Summary

This document outlines the infrastructure needed to scale RingRift AI training across:

1. **Local Mac Cluster** - Underutilized MacBook Pros on the same WiFi network (zero cost, immediate availability)
2. **Cloud Resources** - AWS/GCP for burst capacity and GPU training

The goal is to reduce training time from hours/days to minutes/hours while maintaining reproducibility and cost efficiency.

**Current State:**

- CMA-ES: ~20 generations × 16 population × 24 games = 7,680 games per run (hours locally)
- Neural network: CPU/MPS training on local machine
- Self-play: Sequential game generation

**Target State (Phase 1 - Local Cluster):**

- CMA-ES: Distributed across 3+ Macs, 3-4x speedup, complete runs in ~1 hour
- Neural network: MPS-accelerated training on Apple Silicon
- Self-play: Parallel game generation across machines

**Target State (Phase 2 - Cloud):**

- CMA-ES: Distributed evaluation across 100+ workers, complete runs in <30 minutes
- Neural network: GPU-accelerated training with distributed data loading
- Self-play: Massively parallel game generation

---

## Part 0: Local Mac Cluster (Priority: HIGHEST)

This section covers setting up a local training cluster using underutilized MacBook Pros on the same network. This is the fastest path to distributed training with zero cloud costs.

### 0.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Local Mac Cluster                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  Coordinator    │  - Your primary Mac                        │
│  │  (Main Mac)     │  - Runs CMA-ES algorithm                   │
│  │                 │  - Distributes work via HTTP API           │
│  │  192.168.1.10   │  - Aggregates results                      │
│  └────────┬────────┘                                            │
│           │ HTTP (port 8765)                                    │
│    ┌──────┴──────┬───────────────┐                              │
│    ▼             ▼               ▼                              │
│  ┌───────┐   ┌───────┐       ┌───────┐                          │
│  │Worker │   │Worker │  ...  │Worker │  - Other MacBooks        │
│  │Mac #1 │   │Mac #2 │       │Mac #N │  - Run evaluation tasks  │
│  │.1.11  │   │.1.12  │       │.1.1N  │  - Report back via HTTP  │
│  └───────┘   └───────┘       └───────┘                          │
│                                                                  │
│  Shared Storage: SMB share on coordinator OR git-synced repo    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Advantages:**

- Zero cloud costs
- Low latency (local network)
- Apple Silicon MPS for neural network training
- No Docker required
- Simple setup

### 0.2 Prerequisites

#### On Each Mac:

1. **Python environment** (same version across all machines)

   ```bash
   # Recommended: pyenv for consistent Python versions
   pyenv install 3.11.6
   pyenv global 3.11.6
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/RingRift.git
   cd RingRift/ai-service
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Enable Remote Login (SSH)**
   - System Settings → General → Sharing → Remote Login: ON
   - Note the IP address shown

4. **Static IP or hostname** (optional but recommended)
   - Either assign static IPs in router settings
   - Or use `.local` hostnames (e.g., `armands-macbook.local`)

### 0.3 Implementation: HTTP-based Worker System

**Priority: HIGH** - Simplest approach, no additional infrastructure

#### Worker API Server

```python
# ai-service/scripts/cluster_worker.py
"""
Lightweight HTTP worker for local Mac cluster.
Run on each worker Mac to accept evaluation tasks.
"""
import argparse
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from typing import Dict, Any, Optional

# Import game engine and evaluation code
from app.game_engine import GameEngine
from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import HeuristicWeights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state pool cache
STATE_POOL_CACHE: Dict[str, Any] = {}


class WorkerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "healthy",
                "worker_id": WORKER_ID,
                "tasks_completed": TASKS_COMPLETED
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle evaluation task."""
        if self.path == "/evaluate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            task = json.loads(body)

            try:
                result = evaluate_candidate(task)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                logger.exception(f"Task failed: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")


def evaluate_candidate(task: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single CMA-ES candidate."""
    global TASKS_COMPLETED

    weights_dict = task["weights"]
    board_type = task["board_type"]
    num_players = task["num_players"]
    games_to_play = task["games_per_eval"]
    state_pool_key = task.get("state_pool_key")

    # Create AI with candidate weights
    weights = HeuristicWeights(**weights_dict)
    ai = HeuristicAI(weights)

    # Play games and compute fitness
    wins = 0
    total_moves = 0

    for game_idx in range(games_to_play):
        # Play game (simplified - actual implementation would use full game loop)
        result = play_evaluation_game(ai, board_type, num_players, state_pool_key)
        if result["winner"] == 1:  # AI is player 1
            wins += 1
        total_moves += result["moves"]

    fitness = wins / games_to_play
    TASKS_COMPLETED += 1

    return {
        "task_id": task["task_id"],
        "candidate_id": task["candidate_id"],
        "fitness": fitness,
        "wins": wins,
        "games_played": games_to_play,
        "avg_moves": total_moves / games_to_play
    }


def play_evaluation_game(ai, board_type, num_players, state_pool_key):
    """Play a single evaluation game."""
    # This would import and use the actual game playing logic
    # from run_cmaes_optimization.py
    from scripts.run_cmaes_optimization import play_single_game_from_state
    # ... implementation
    pass


WORKER_ID = "unknown"
TASKS_COMPLETED = 0


def main():
    global WORKER_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--worker-id", type=str, default=None)
    args = parser.parse_args()

    import socket
    WORKER_ID = args.worker_id or socket.gethostname()

    server = HTTPServer(("0.0.0.0", args.port), WorkerHandler)
    logger.info(f"Worker {WORKER_ID} listening on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

#### Coordinator Distributed Mode

```python
# Add to ai-service/scripts/run_cmaes_optimization.py

def evaluate_population_distributed(
    population: List[Dict[str, float]],
    workers: List[str],  # ["192.168.1.11:8765", "192.168.1.12:8765", ...]
    config: CMAESConfig,
    timeout: float = 300
) -> List[float]:
    """Distribute population evaluation across worker Macs."""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    tasks = []

    # Create tasks for each candidate
    for i, candidate in enumerate(population):
        task = {
            "task_id": str(uuid.uuid4()),
            "candidate_id": i,
            "weights": candidate,
            "board_type": config.board_type,
            "num_players": config.num_players,
            "games_per_eval": config.games_per_eval,
            "state_pool_key": config.state_pool_id
        }
        tasks.append(task)

    def evaluate_on_worker(task, worker_url):
        """Send task to worker and get result."""
        try:
            response = requests.post(
                f"http://{worker_url}/evaluate",
                json=task,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Worker {worker_url} failed: {e}")
            return None

    # Round-robin distribute tasks across workers
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            worker = workers[i % len(workers)]
            future = executor.submit(evaluate_on_worker, task, worker)
            futures[future] = task["candidate_id"]

        for future in as_completed(futures, timeout=timeout):
            candidate_id = futures[future]
            try:
                result = future.result()
                if result:
                    results[result["candidate_id"]] = result["fitness"]
            except Exception as e:
                logger.warning(f"Task for candidate {candidate_id} failed: {e}")

    # Fill in missing results with fallback
    fitness_scores = []
    for i in range(len(population)):
        if i in results:
            fitness_scores.append(results[i])
        else:
            logger.warning(f"Candidate {i} missing, using fallback fitness 0.0")
            fitness_scores.append(0.0)

    return fitness_scores
```

### 0.4 Worker Discovery and Management

#### Automatic Discovery via mDNS/Bonjour

macOS has built-in Bonjour support for zero-config networking.

```python
# ai-service/app/distributed/discovery.py
"""
Discover worker Macs on the local network using Bonjour/mDNS.
"""
import socket
from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
from typing import List, Set
import threading
import time


class WorkerDiscovery(ServiceListener):
    """Discover RingRift training workers on local network."""

    SERVICE_TYPE = "_ringrift-worker._tcp.local."

    def __init__(self):
        self.workers: Set[str] = set()
        self.lock = threading.Lock()
        self.zeroconf = Zeroconf()
        self.browser = ServiceBrowser(self.zeroconf, self.SERVICE_TYPE, self)

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            addresses = info.parsed_addresses()
            port = info.port
            if addresses:
                worker_url = f"{addresses[0]}:{port}"
                with self.lock:
                    self.workers.add(worker_url)
                print(f"Discovered worker: {worker_url}")

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        # Worker went offline - would need to track by name
        pass

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    def get_workers(self) -> List[str]:
        with self.lock:
            return list(self.workers)

    def close(self):
        self.zeroconf.close()


# Usage in coordinator
discovery = WorkerDiscovery()
time.sleep(2)  # Wait for discovery
workers = discovery.get_workers()
print(f"Found {len(workers)} workers: {workers}")
```

#### Worker Registration (Alternative)

```python
# ai-service/scripts/register_worker.py
"""
Register this Mac as a worker with the coordinator.
"""
from zeroconf import ServiceInfo, Zeroconf
import socket


def register_worker(port: int = 8765):
    """Register this worker via Bonjour."""
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname + ".local")

    service_info = ServiceInfo(
        "_ringrift-worker._tcp.local.",
        f"ringrift-worker-{hostname}._ringrift-worker._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=port,
        properties={"hostname": hostname},
    )

    zeroconf = Zeroconf()
    zeroconf.register_service(service_info)
    print(f"Registered worker at {local_ip}:{port}")

    return zeroconf, service_info
```

### 0.5 Shared State Pool Distribution

#### Option A: Git-based Sync (Simplest)

```bash
# On coordinator: commit state pools to a branch
git add data/eval_pools/
git commit -m "Update state pools"
git push origin state-pools

# On workers: pull before starting
git pull origin state-pools
```

#### Option B: HTTP Download from Coordinator

```python
# Coordinator serves state pools via HTTP
# Workers download on startup

# In cluster_worker.py
def download_state_pool(coordinator_url: str, pool_id: str):
    response = requests.get(f"http://{coordinator_url}/state-pool/{pool_id}")
    # Cache locally
```

#### Option C: SMB/AFP Share (Best for large pools)

```bash
# On coordinator Mac:
# System Settings → General → Sharing → File Sharing: ON
# Add ai-service/data/eval_pools to shared folders

# On worker Macs:
# Mount the share and symlink
ln -s /Volumes/Shared/eval_pools data/eval_pools
```

### 0.6 Quick Start Guide

**1. On Coordinator Mac:**

```bash
cd RingRift/ai-service
source .venv/bin/activate

# Start CMA-ES in distributed mode
python scripts/run_cmaes_optimization.py \
    --distributed \
    --workers "192.168.1.11:8765,192.168.1.12:8765" \
    --generations 20 \
    --population-size 16 \
    --games-per-eval 24 \
    --board square8 \
    --output logs/cmaes/distributed_run_001
```

**2. On Each Worker Mac:**

```bash
cd RingRift/ai-service
source .venv/bin/activate

# Start worker
python scripts/cluster_worker.py --port 8765
```

**3. Monitor Progress:**

```bash
# Check worker health
curl http://192.168.1.11:8765/health
curl http://192.168.1.12:8765/health

# Watch coordinator logs
tail -f logs/cmaes/distributed_run_001/run.log
```

### 0.7 Neural Network Training on Apple Silicon

Apple Silicon Macs can use MPS (Metal Performance Shaders) for GPU acceleration.

```python
# ai-service/app/training/train_neural_net.py

import torch

def get_device():
    """Get best available device for training."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# In training loop
device = get_device()
model = model.to(device)

# Note: Some operations may need CPU fallback
# Use torch.mps.empty_cache() to manage memory
```

**Performance Tips for MPS:**

- Use batch sizes of 256-512 (sweet spot for M1/M2/M3)
- `torch.mps.synchronize()` for accurate timing
- Some custom ops may need CPU fallback

### 0.8 Local Cluster Capacity Planning

| Configuration    | Estimated Throughput | CMA-ES Run Time |
| ---------------- | -------------------- | --------------- |
| 1 Mac (baseline) | ~100 games/min       | ~75 min         |
| 2 Macs           | ~190 games/min       | ~40 min         |
| 3 Macs           | ~280 games/min       | ~27 min         |
| 4 Macs           | ~360 games/min       | ~21 min         |

_Assumes M1/M2 MacBook Pro, square8 board, 2 players_

### 0.9 Robustness Features

1. **Worker health monitoring**
   - Ping workers before distributing tasks
   - Skip unresponsive workers

2. **Task retry on failure**
   - If a worker times out, reassign task to another worker
   - Maximum 2 retries before using fallback fitness

3. **Graceful degradation**
   - If all workers fail, fall back to local evaluation
   - Coordinator can also act as a worker

4. **Checkpoint compatibility**
   - Same checkpoint format as local runs
   - Can resume distributed run locally and vice versa

---

## Part 1: Infrastructure Foundation

### 1.1 Container Strategy

**Priority: HIGH** - Foundation for all cloud deployment

#### Tasks:

1. **Production-ready AI service Dockerfile**
   - Current `Dockerfile` exists but needs optimization
   - Multi-stage build for smaller images
   - GPU support via NVIDIA base images
   - Separate images for: inference, training, CMA-ES worker

   ```dockerfile
   # Example structure needed
   FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS base
   # ... dependencies

   FROM base AS inference
   # Lightweight for serving

   FROM base AS training
   # Full PyTorch with GPU support

   FROM base AS cmaes-worker
   # Game engine + evaluation only
   ```

2. **Container registry setup**
   - AWS ECR / GCP Artifact Registry / Docker Hub
   - Automated builds from CI
   - Version tagging strategy (git SHA, semantic versions)

3. **Local testing parity**
   - `docker-compose.gpu.yml` for local GPU testing
   - Ensure container behavior matches local development

### 1.2 Cloud Storage Architecture

**Priority: HIGH** - Required for data persistence and sharing

#### Storage Types Needed:

| Data Type                  | Volume             | Access Pattern        | Recommended Storage    |
| -------------------------- | ------------------ | --------------------- | ---------------------- |
| Training datasets (`.npz`) | 10-100 GB          | Write once, read many | S3/GCS Standard        |
| State pools (`.jsonl`)     | 1-10 GB            | Read-heavy            | S3/GCS with caching    |
| Model checkpoints          | 100 MB - 1 GB each | Frequent writes       | S3/GCS Standard        |
| CMA-ES checkpoints         | 10-50 KB each      | Every generation      | S3/GCS Standard        |
| Game replay DBs            | 10-100 GB          | Append-heavy          | S3/GCS or managed DB   |
| Logs and metrics           | Unbounded          | Append-only           | CloudWatch/Stackdriver |

#### Tasks:

1. **Create storage abstraction layer**

   ```python
   # ai-service/app/storage/cloud_storage.py
   class StorageBackend(Protocol):
       def upload(self, local_path: Path, remote_key: str) -> None: ...
       def download(self, remote_key: str, local_path: Path) -> None: ...
       def list(self, prefix: str) -> List[str]: ...

   class S3Storage(StorageBackend): ...
   class GCSStorage(StorageBackend): ...
   class LocalStorage(StorageBackend): ...  # For development
   ```

2. **Environment-based configuration**

   ```python
   STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")  # local, s3, gcs
   STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "")
   STORAGE_PREFIX = os.getenv("STORAGE_PREFIX", "ringrift-ai")
   ```

3. **Integrate with existing scripts**
   - `run_cmaes_optimization.py`: Upload checkpoints to cloud
   - `generate_data.py`: Upload datasets to cloud
   - `train_neural_net.py`: Download data, upload models

### 1.3 Configuration Management

**Priority: MEDIUM** - Enables reproducibility

#### Tasks:

1. **Centralized config schema**

   ```python
   # ai-service/app/config/training_config.py
   @dataclass
   class CMAESConfig:
       generations: int = 20
       population_size: int = 16
       games_per_eval: int = 24
       board_type: str = "square8"
       num_players: int = 2
       sigma: float = 1.0
       state_pool_id: str = "v1"
       # ... all CLI args as typed fields

   @dataclass
   class NeuralNetConfig:
       batch_size: int = 256
       learning_rate: float = 0.001
       epochs: int = 100
       hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
       # ...
   ```

2. **Config serialization for reproducibility**
   - Save full config with each run
   - Config hashing for cache invalidation
   - Config diffing for experiment comparison

3. **Environment variable overrides**
   - All config values overridable via env vars
   - Prefix: `RINGRIFT_TRAIN_*`

---

## Part 2: CMA-ES Distributed Training

### 2.1 Architecture Options

#### Option A: Coordinator-Worker (Recommended for initial implementation)

```
┌─────────────────┐
│   Coordinator   │  - Runs CMA-ES algorithm
│   (1 instance)  │  - Distributes candidates
│                 │  - Aggregates fitness scores
└────────┬────────┘
         │ Redis/SQS Queue
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│Worker │ │Worker │ │Worker │ │Worker │  - Evaluate candidates
│  1    │ │  2    │ │  3    │ │  N    │  - Play games
└───────┘ └───────┘ └───────┘ └───────┘  - Report fitness
```

**Pros:** Simple, fault-tolerant, works with spot instances
**Cons:** Coordinator is SPOF, queue latency

#### Option B: Ray Cluster

```
┌─────────────────────────────────────┐
│         Ray Head Node               │
│   ┌─────────────────────────────┐   │
│   │  CMA-ES Driver (ray.remote) │   │
│   └─────────────────────────────┘   │
└─────────────────────────────────────┘
              │ Ray Object Store
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ Ray   │ │ Ray   │ │ Ray   │ │ Ray   │
│Worker │ │Worker │ │Worker │ │Worker │
└───────┘ └───────┘ └───────┘ └───────┘
```

**Pros:** Built-in fault tolerance, auto-scaling, efficient data sharing
**Cons:** More complex setup, Ray learning curve

### 2.2 Implementation Plan (Option A - Coordinator-Worker)

#### Phase 1: Queue-based Communication

**Tasks:**

1. **Create message queue abstraction**

   ```python
   # ai-service/app/distributed/queue.py
   class TaskQueue(Protocol):
       def publish(self, task: EvalTask) -> str: ...
       def consume(self, timeout: float) -> Optional[EvalTask]: ...
       def ack(self, task_id: str) -> None: ...

   @dataclass
   class EvalTask:
       task_id: str
       generation: int
       candidate_id: int
       weights: Dict[str, float]
       board_type: str
       num_players: int
       games_to_play: int
       state_pool_key: str

   @dataclass
   class EvalResult:
       task_id: str
       candidate_id: int
       fitness: float
       games_played: int
       wins: int
       avg_moves: float
       eval_time_seconds: float
   ```

2. **Redis implementation (for local/small-scale)**

   ```python
   class RedisQueue(TaskQueue):
       def __init__(self, redis_url: str, queue_name: str): ...
   ```

3. **SQS implementation (for AWS scale)**
   ```python
   class SQSQueue(TaskQueue):
       def __init__(self, queue_url: str): ...
   ```

#### Phase 2: Worker Process

**Tasks:**

1. **Standalone worker script**

   ```python
   # ai-service/scripts/cmaes_worker.py
   def main():
       queue = get_queue_backend()
       storage = get_storage_backend()

       # Download state pool once
       state_pool = download_state_pool(storage, config.state_pool_key)

       while True:
           task = queue.consume(timeout=30)
           if task is None:
               continue

           try:
               fitness = evaluate_candidate(task, state_pool)
               queue.publish_result(EvalResult(...))
               queue.ack(task.task_id)
           except Exception as e:
               logger.error(f"Task {task.task_id} failed: {e}")
               # Task will be requeued after visibility timeout
   ```

2. **Graceful shutdown handling**
   - SIGTERM handler for spot instance termination
   - Checkpoint current game state if possible
   - Don't ack incomplete tasks

3. **Health check endpoint**
   - Simple HTTP endpoint for load balancer health checks
   - Report: tasks processed, current task, memory usage

#### Phase 3: Coordinator Updates

**Tasks:**

1. **Modify `run_cmaes_optimization.py` for distributed mode**

   ```python
   # New flag: --distributed
   if args.distributed:
       fitness_scores = evaluate_population_distributed(
           population, queue, timeout=args.eval_timeout
       )
   else:
       fitness_scores = evaluate_population_local(population)
   ```

2. **Distributed evaluation function**

   ```python
   def evaluate_population_distributed(
       population: List[Dict[str, float]],
       queue: TaskQueue,
       timeout: float
   ) -> List[float]:
       # Publish all tasks
       task_ids = []
       for i, candidate in enumerate(population):
           task = EvalTask(
               task_id=str(uuid4()),
               candidate_id=i,
               weights=candidate,
               ...
           )
           queue.publish(task)
           task_ids.append(task.task_id)

       # Collect results with timeout
       results = {}
       deadline = time.time() + timeout
       while len(results) < len(population) and time.time() < deadline:
           result = queue.consume_result(timeout=1)
           if result:
               results[result.candidate_id] = result.fitness

       # Handle missing results (worker failures)
       for i in range(len(population)):
           if i not in results:
               logger.warning(f"Candidate {i} timed out, using fallback fitness")
               results[i] = 0.0  # Or re-queue

       return [results[i] for i in range(len(population))]
   ```

3. **Progress tracking across workers**
   - Central progress store (Redis hash)
   - Real-time dashboard support

### 2.3 Cloud Deployment Configuration

#### AWS Deployment

**Terraform/CloudFormation resources needed:**

1. **ECS Task Definitions**
   - Coordinator task (1 instance, on-demand)
   - Worker task (auto-scaled, spot instances)

2. **Auto Scaling**

   ```hcl
   resource "aws_appautoscaling_target" "workers" {
     service_namespace  = "ecs"
     scalable_dimension = "ecs:service:DesiredCount"
     min_capacity       = 0
     max_capacity       = 100
   }

   resource "aws_appautoscaling_policy" "scale_by_queue" {
     # Scale based on SQS queue depth
     target_tracking_scaling_policy_configuration {
       target_value = 10  # Tasks per worker
       customized_metric_specification {
         metric_name = "ApproximateNumberOfMessagesVisible"
         namespace   = "AWS/SQS"
         statistic   = "Average"
       }
     }
   }
   ```

3. **Spot Instance Configuration**
   - Instance types: c6i.xlarge, c6i.2xlarge (CPU-optimized)
   - Spot price: up to 70% savings
   - Interruption handling: 2-minute warning

4. **Cost Estimation**
   | Component | Hourly Cost | Per Run (30 min) |
   |-----------|-------------|------------------|
   | Coordinator (c6i.large) | $0.085 | $0.04 |
   | 50 Workers (c6i.xlarge spot) | $0.051 × 50 = $2.55 | $1.28 |
   | SQS | ~$0.01 | ~$0.01 |
   | S3 transfers | ~$0.05 | ~$0.05 |
   | **Total** | | **~$1.40/run** |

### 2.4 Checkpointing and Recovery

**Tasks:**

1. **Generation-level checkpointing**
   - Already exists locally, extend to cloud storage
   - Upload checkpoint after each generation completes

2. **Resume from cloud checkpoint**

   ```bash
   python scripts/run_cmaes_optimization.py \
     --distributed \
     --resume s3://ringrift-training/cmaes/run_20251202_123456/checkpoint_gen015.json
   ```

3. **Coordinator failure recovery**
   - Store CMA-ES internal state (mean, covariance matrix)
   - Allow coordinator restart mid-run

---

## Part 2.5: Massive Cloud Scaling for Full Training Matrix

**Goal:** Complete iterative CMA-ES across ALL board types and player counts in <1 hour

### 2.5.1 The Training Matrix

| Board Type | Players       | Iterations | Gens/Iter | Pop | Games/Eval | Total Games         |
| ---------- | ------------- | ---------- | --------- | --- | ---------- | ------------------- |
| square8    | 2             | 10         | 30        | 30  | 40         | 360,000             |
| square8    | 3             | 10         | 30        | 30  | 40         | 360,000             |
| square8    | 4             | 10         | 30        | 30  | 40         | 360,000             |
| square19   | 2             | 10         | 30        | 30  | 40         | 360,000             |
| square19   | 3             | 10         | 30        | 30  | 40         | 360,000             |
| square19   | 4             | 10         | 30        | 30  | 40         | 360,000             |
| hex        | 2             | 10         | 30        | 30  | 40         | 360,000             |
| hex        | 3             | 10         | 30        | 30  | 40         | 360,000             |
| hex        | 4             | 10         | 30        | 30  | 40         | 360,000             |
| **Total**  | **9 configs** |            |           |     |            | **3,240,000 games** |

### 2.5.2 Parallelism Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FULL TRAINING ORCHESTRATOR                        │
│  (AWS Step Functions / Airflow / Custom Script)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Level 1: Configuration Parallelism (9 parallel runs)               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ... ┌─────────┐                │
│  │square8  │ │square8  │ │square8  │     │hex      │                │
│  │2-player │ │3-player │ │4-player │     │4-player │                │
│  └────┬────┘ └────┬────┘ └────┬────┘     └────┬────┘                │
│       │           │           │               │                      │
│  Level 2: Per-Config Worker Pool (~50 workers each)                 │
│       │           │           │               │                      │
│  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐     ┌────┴────┐                │
│  │Workers  │ │Workers  │ │Workers  │     │Workers  │                │
│  │1-50     │ │51-100   │ │101-150  │     │401-450  │                │
│  └─────────┘ └─────────┘ └─────────┘     └─────────┘                │
│                                                                      │
│  Total Workers: ~450 spot instances                                  │
│  Instance Type: c6i.xlarge (4 vCPU, 8 GB RAM)                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.5.3 Time Calculation

**Per-game timing (observed):**

- square8: ~0.5-1.0 sec/game
- square19: ~2-3 sec/game
- hex: ~1.5-2.5 sec/game

**With 50 workers per configuration:**

- Games per generation: 30 pop × 40 games = 1,200 games
- Worker throughput: ~50 games/min/worker × 50 workers = 2,500 games/min
- Time per generation: 1,200 / 2,500 = 0.48 min = ~30 seconds
- Time per iteration: 30 gens × 30 sec = 15 min
- Time for full run: 10 iterations × 15 min = **150 min (2.5 hours)**

**To achieve <1 hour, need 3x more workers (150 per config):**

- Workers per config: 150
- Total workers: 9 × 150 = 1,350 workers
- Time per generation: 1,200 / 7,500 = ~10 seconds
- Time per iteration: 30 gens × 10 sec = 5 min
- Time for full run: 10 iterations × 5 min = **50 min**

### 2.5.4 Architecture for Full Matrix Training

```python
# ai-service/scripts/run_full_training_matrix.py
"""
Launch full CMA-ES training across all configurations in parallel.
"""
import asyncio
import boto3
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TrainingConfig:
    board_type: str
    num_players: int
    iterations: int = 10
    generations_per_iter: int = 30
    population_size: int = 30
    games_per_eval: int = 40
    workers_per_config: int = 150  # For <1 hour completion


FULL_TRAINING_MATRIX = [
    TrainingConfig("square8", 2),
    TrainingConfig("square8", 3),
    TrainingConfig("square8", 4),
    TrainingConfig("square19", 2),
    TrainingConfig("square19", 3),
    TrainingConfig("square19", 4),
    TrainingConfig("hex", 2),
    TrainingConfig("hex", 3),
    TrainingConfig("hex", 4),
]


async def run_full_matrix():
    """Launch all 9 configurations in parallel."""

    # 1. Spin up worker fleets for each configuration
    ecs_client = boto3.client("ecs")

    worker_tasks = []
    for config in FULL_TRAINING_MATRIX:
        task = launch_worker_fleet(
            ecs_client,
            config=config,
            worker_count=config.workers_per_config,
            cluster_name=f"ringrift-cmaes-{config.board_type}-{config.num_players}p"
        )
        worker_tasks.append(task)

    # Wait for all workers to be running
    await asyncio.gather(*worker_tasks)

    # 2. Launch coordinators for each configuration
    coordinator_tasks = []
    for config in FULL_TRAINING_MATRIX:
        task = launch_coordinator(
            ecs_client,
            config=config,
            run_id=f"full_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        coordinator_tasks.append(task)

    # Wait for all training runs to complete
    results = await asyncio.gather(*coordinator_tasks)

    # 3. Merge all results
    merge_training_profiles(results)

    # 4. Scale down workers
    for config in FULL_TRAINING_MATRIX:
        await scale_down_workers(ecs_client, config)

    return results


async def launch_worker_fleet(ecs_client, config: TrainingConfig, worker_count: int, cluster_name: str):
    """Launch a fleet of spot instance workers for one configuration."""

    # Use ECS with spot capacity provider
    response = ecs_client.update_service(
        cluster=cluster_name,
        service=f"cmaes-worker-{config.board_type}-{config.num_players}p",
        desiredCount=worker_count,
        capacityProviderStrategy=[{
            "capacityProvider": "FARGATE_SPOT",
            "weight": 1,
            "base": 0
        }]
    )

    # Wait for workers to reach RUNNING state
    waiter = ecs_client.get_waiter("services_stable")
    await asyncio.to_thread(
        waiter.wait,
        cluster=cluster_name,
        services=[f"cmaes-worker-{config.board_type}-{config.num_players}p"]
    )

    return response


async def launch_coordinator(ecs_client, config: TrainingConfig, run_id: str):
    """Launch the CMA-ES coordinator for one configuration."""

    task_definition = f"cmaes-coordinator-{config.board_type}"

    response = ecs_client.run_task(
        cluster="ringrift-cmaes-coordinators",
        taskDefinition=task_definition,
        launchType="FARGATE",
        networkConfiguration={...},
        overrides={
            "containerOverrides": [{
                "name": "coordinator",
                "environment": [
                    {"name": "BOARD_TYPE", "value": config.board_type},
                    {"name": "NUM_PLAYERS", "value": str(config.num_players)},
                    {"name": "ITERATIONS", "value": str(config.iterations)},
                    {"name": "GENERATIONS_PER_ITER", "value": str(config.generations_per_iter)},
                    {"name": "POPULATION_SIZE", "value": str(config.population_size)},
                    {"name": "GAMES_PER_EVAL", "value": str(config.games_per_eval)},
                    {"name": "RUN_ID", "value": run_id},
                    {"name": "WORKERS_QUEUE", "value": f"ringrift-cmaes-{config.board_type}-{config.num_players}p"},
                ]
            }]
        }
    )

    # Wait for task to complete (could be hours)
    task_arn = response["tasks"][0]["taskArn"]
    await wait_for_task_completion(ecs_client, task_arn)

    # Download results from S3
    results = download_results(run_id, config)
    return results
```

### 2.5.5 Cost Estimation for Full Matrix Training

| Resource           | Count | Unit Cost (spot) | Duration | Total        |
| ------------------ | ----- | ---------------- | -------- | ------------ |
| c6i.xlarge workers | 1,350 | $0.051/hr        | 1 hr     | $68.85       |
| Coordinators (9)   | 9     | $0.085/hr        | 1 hr     | $0.77        |
| SQS messages       | ~4M   | $0.40/M          | -        | $1.60        |
| S3 storage         | 1 GB  | $0.023/GB        | -        | $0.02        |
| Data transfer      | 10 GB | $0.09/GB         | -        | $0.90        |
| **Total**          |       |                  |          | **~$72/run** |

**Cost per full training cycle: ~$72** (all 9 board/player configs, 10 iterations each)

### 2.5.6 Alternative: Staggered Execution

If spot capacity is limited or cost is a concern, run configurations sequentially with shared workers:

```
Timeline (3 hours total):
├── Hour 1: square8 (2p, 3p, 4p) with 150 workers → 3 configs done
├── Hour 2: square19 (2p, 3p, 4p) with 150 workers → 6 configs done
└── Hour 3: hex (2p, 3p, 4p) with 150 workers → 9 configs done

Cost: ~$25/run (150 workers × 3 hours × $0.051 = $23 + overhead)
```

### 2.5.7 Terraform Configuration for Auto-Scaling Fleet

```hcl
# infrastructure/terraform/cmaes_fleet.tf

resource "aws_ecs_cluster" "cmaes" {
  name = "ringrift-cmaes"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  capacity_providers = ["FARGATE_SPOT", "FARGATE"]
}

resource "aws_ecs_service" "workers" {
  for_each = toset([
    "square8-2p", "square8-3p", "square8-4p",
    "square19-2p", "square19-3p", "square19-4p",
    "hex-2p", "hex-3p", "hex-4p"
  ])

  name            = "cmaes-worker-${each.value}"
  cluster         = aws_ecs_cluster.cmaes.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = 0  # Scaled up dynamically

  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight           = 100
  }

  network_configuration {
    subnets         = var.private_subnets
    security_groups = [aws_security_group.workers.id]
  }
}

resource "aws_appautoscaling_target" "workers" {
  for_each = aws_ecs_service.workers

  max_capacity       = 200
  min_capacity       = 0
  resource_id        = "service/${aws_ecs_cluster.cmaes.name}/${each.value.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}
```

### 2.5.8 Monitoring Dashboard for Full Training

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "title": "Games Completed Per Configuration",
        "metrics": [
          ["RingRift/CMAES", "GamesCompleted", "Config", "square8-2p"],
          ["...", "square8-3p"],
          ["...", "square8-4p"],
          ["...", "square19-2p"],
          ["...", "square19-3p"],
          ["...", "square19-4p"],
          ["...", "hex-2p"],
          ["...", "hex-3p"],
          ["...", "hex-4p"]
        ],
        "period": 60
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Best Fitness Per Configuration",
        "metrics": [
          ["RingRift/CMAES", "BestFitness", "Config", "square8-2p"]
          // ... all configs
        ]
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Active Workers",
        "metrics": [["AWS/ECS", "RunningTaskCount", "ServiceName", "cmaes-worker-*"]]
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Cost Tracking",
        "metrics": [["AWS/Billing", "EstimatedCharges", "ServiceName", "AmazonECS"]]
      }
    }
  ]
}
```

### 2.5.9 Quick Launch Command

```bash
# Full matrix training in <1 hour
python scripts/run_full_training_matrix.py \
    --workers-per-config 150 \
    --iterations 10 \
    --generations-per-iter 30 \
    --population-size 30 \
    --games-per-eval 40 \
    --max-duration 3600  # 1 hour timeout

# Or with cost limits
python scripts/run_full_training_matrix.py \
    --budget-limit 100  # Stop if projected cost exceeds $100
    --stagger-configs   # Run sequentially to reduce spot pressure
```

---

## Part 3: Neural Network Training Infrastructure

### 3.1 Training Data Pipeline

#### Current State:

- `generate_data.py` creates `.npz` files locally
- Data stored as (state_features, move_label, value_label) tuples
- Sequential generation

#### Target State:

- Distributed self-play game generation
- Streaming upload to cloud storage
- Efficient data loading during training

**Tasks:**

1. **Parallel game generation**

   ```python
   # ai-service/scripts/generate_data_distributed.py
   @ray.remote
   def play_game_worker(game_id: int, config: GameConfig) -> GameRecord:
       engine = GameEngine()
       ai = HeuristicAI(config.weights)
       game = play_full_game(engine, ai)
       return GameRecord(states=..., moves=..., outcome=...)

   def generate_dataset(num_games: int, parallelism: int):
       refs = [play_game_worker.remote(i, config) for i in range(num_games)]
       for batch in ray.get(refs, batch_size=100):
           upload_batch_to_storage(batch)
   ```

2. **Streaming data format**
   - Consider TFRecord or WebDataset for efficient streaming
   - Sharded files for parallel reading
   - Metadata sidecar for dataset statistics

3. **Data versioning**
   ```
   s3://ringrift-training/datasets/
   ├── v1/
   │   ├── metadata.json  # {board_type, num_games, ai_version, ...}
   │   ├── train/
   │   │   ├── shard_0000.npz
   │   │   ├── shard_0001.npz
   │   │   └── ...
   │   └── val/
   │       └── ...
   └── v2/
       └── ...
   ```

### 3.2 GPU Training Setup

#### Instance Selection

| Use Case          | Instance Type | GPUs    | Cost/hr (on-demand) | Cost/hr (spot) |
| ----------------- | ------------- | ------- | ------------------- | -------------- |
| Development       | g4dn.xlarge   | 1× T4   | $0.526              | ~$0.16         |
| Standard training | g5.xlarge     | 1× A10G | $1.006              | ~$0.30         |
| Fast training     | g5.2xlarge    | 1× A10G | $1.212              | ~$0.36         |
| Large models      | p4d.24xlarge  | 8× A100 | $32.77              | ~$10.00        |

**Recommendation:** Start with g5.xlarge for most training runs

#### Tasks:

1. **GPU-optimized Docker image**

   ```dockerfile
   FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

   # Install Python and PyTorch with CUDA
   RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install training dependencies
   COPY requirements-training.txt .
   RUN pip install -r requirements-training.txt

   COPY . /app
   WORKDIR /app
   ```

2. **Training script enhancements**

   ```python
   # ai-service/app/training/train_neural_net.py

   # Auto-detect device
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Mixed precision training
   scaler = torch.amp.GradScaler('cuda')

   # DataLoader with multiple workers
   train_loader = DataLoader(
       dataset,
       batch_size=config.batch_size,
       num_workers=4,
       pin_memory=True,
       prefetch_factor=2
   )
   ```

3. **Checkpoint management**
   - Save model, optimizer, scheduler, epoch, best metrics
   - Upload to cloud storage after each epoch
   - Resume from cloud checkpoint

### 3.3 Experiment Tracking

**Options:**

- Weights & Biases (recommended for small teams)
- MLflow (self-hosted option)
- TensorBoard (basic, already integrated with PyTorch)

**Tasks:**

1. **Integrate W&B or MLflow**

   ```python
   import wandb

   wandb.init(
       project="ringrift-neural",
       config=config.__dict__,
       tags=[config.board_type, f"players_{config.num_players}"]
   )

   for epoch in range(config.epochs):
       train_loss = train_epoch(...)
       val_loss = validate(...)

       wandb.log({
           "epoch": epoch,
           "train_loss": train_loss,
           "val_loss": val_loss,
           "learning_rate": scheduler.get_last_lr()[0]
       })

       # Log model artifact
       if val_loss < best_val_loss:
           wandb.save("model_best.pt")
   ```

2. **Hyperparameter sweeps**
   ```yaml
   # wandb_sweep.yaml
   method: bayes
   metric:
     name: val_loss
     goal: minimize
   parameters:
     learning_rate:
       min: 0.0001
       max: 0.01
       distribution: log_uniform
     batch_size:
       values: [128, 256, 512]
     hidden_layers:
       values: [[256, 128], [512, 256, 128], [256, 128, 64, 32]]
   ```

### 3.4 Model Serving Pipeline

**Tasks:**

1. **Model export format**
   - TorchScript for production inference
   - ONNX for cross-platform compatibility

2. **Model registry**

   ```
   s3://ringrift-models/
   ├── neural/
   │   ├── square8_2p/
   │   │   ├── v1/
   │   │   │   ├── model.pt
   │   │   │   ├── model.onnx
   │   │   │   ├── config.json
   │   │   │   └── metrics.json
   │   │   └── v2/
   │   │       └── ...
   │   └── hex_2p/
   │       └── ...
   └── heuristic/
       └── trained_profiles.json
   ```

3. **Automated deployment**
   - CI/CD pipeline: train → validate → promote → deploy
   - Canary deployment for new models
   - Rollback capability

---

## Part 4: Unified Training Orchestration

### 4.1 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Self-Play   │───▶│  CMA-ES      │───▶│  Heuristic   │       │
│  │  Generation  │    │  Optimization│    │  Profiles    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│         │                                        │               │
│         ▼                                        ▼               │
│  ┌──────────────┐                        ┌──────────────┐       │
│  │  Training    │                        │  Profile     │       │
│  │  Dataset     │                        │  Validation  │       │
│  └──────┬───────┘                        └──────────────┘       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Neural Net  │───▶│  Model       │───▶│  Production  │       │
│  │  Training    │    │  Validation  │    │  Deployment  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Pipeline Orchestration Options

#### Option A: Airflow/Prefect

**Pros:** Production-ready, visual DAG, scheduling, retries
**Cons:** Infrastructure overhead

#### Option B: GitHub Actions (for simpler cases)

**Pros:** No additional infrastructure, integrated with repo
**Cons:** Limited for long-running jobs, cost for heavy usage

#### Option C: Custom Scripts with Cloud Scheduler

**Pros:** Simple, low overhead
**Cons:** Manual retry logic, less visibility

**Recommendation:** Start with Option C, move to Option A as complexity grows

### 4.3 Implementation Tasks

1. **Pipeline definition script**

   ```python
   # ai-service/scripts/run_training_pipeline.py

   @dataclass
   class PipelineConfig:
       # Self-play
       selfplay_games: int = 5000
       selfplay_parallelism: int = 50

       # CMA-ES
       cmaes_generations: int = 30
       cmaes_population: int = 20

       # Neural net
       nn_epochs: int = 100
       nn_batch_size: int = 256

   def run_pipeline(config: PipelineConfig):
       # Step 1: Generate training data
       dataset_path = generate_selfplay_data(
           num_games=config.selfplay_games,
           parallelism=config.selfplay_parallelism
       )

       # Step 2: Run CMA-ES (can run in parallel with step 1)
       heuristic_profile = run_cmaes(
           generations=config.cmaes_generations,
           population=config.cmaes_population
       )

       # Step 3: Train neural network
       model = train_neural_net(
           dataset=dataset_path,
           epochs=config.nn_epochs
       )

       # Step 4: Validate and deploy
       if validate_model(model, heuristic_profile):
           deploy_model(model)
           deploy_profile(heuristic_profile)
   ```

2. **Pipeline monitoring dashboard**
   - Stage progress
   - Resource utilization
   - Cost tracking
   - Quality metrics

---

## Part 5: Cost Management

### 5.1 Cost Optimization Strategies

1. **Spot instances for workers**
   - 60-90% cost savings
   - Acceptable for stateless evaluation workers

2. **Auto-scaling to zero**
   - Scale down when no training jobs
   - Use scheduled scaling for predictable workloads

3. **Reserved capacity for predictable workloads**
   - 1-year reserved instances for baseline capacity
   - Spot for burst capacity

4. **Right-sizing**
   - Monitor CPU/GPU utilization
   - Adjust instance types based on actual usage

### 5.2 Budget Alerts

```hcl
resource "aws_budgets_budget" "training" {
  budget_type  = "COST"
  limit_amount = "500"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator = "GREATER_THAN"
    threshold           = 80
    threshold_type      = "PERCENTAGE"
    notification_type   = "ACTUAL"
    subscriber_email    = ["team@ringrift.com"]
  }
}
```

### 5.3 Cost Estimation Summary

| Scenario                                    | Monthly Estimate |
| ------------------------------------------- | ---------------- |
| Light (2 CMA-ES runs/week, 1 NN train/week) | ~$50-100         |
| Medium (daily CMA-ES, 3 NN trains/week)     | ~$200-400        |
| Heavy (continuous optimization)             | ~$500-1000       |

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Storage abstraction layer with S3/GCS support
- [ ] GPU-enabled Docker image
- [ ] Basic cloud upload/download in training scripts
- [ ] Environment variable configuration

### Phase 2: CMA-ES Distribution (Week 3-4)

- [ ] Queue abstraction (Redis + SQS)
- [ ] Worker process implementation
- [ ] Coordinator distributed mode
- [ ] Local testing with Docker Compose
- [ ] AWS deployment scripts (Terraform/CDK)

### Phase 3: Neural Network Training (Week 5-6)

- [ ] GPU training optimization
- [ ] Data pipeline (sharded datasets)
- [ ] Experiment tracking integration (W&B)
- [ ] Model registry structure

### Phase 4: Integration & Polish (Week 7-8)

- [ ] End-to-end pipeline script
- [ ] Monitoring dashboard
- [ ] Cost tracking
- [ ] Documentation and runbooks
- [ ] Team training

---

## Appendix A: Technology Choices Summary

| Component               | Recommended         | Alternatives        |
| ----------------------- | ------------------- | ------------------- |
| Cloud Provider          | AWS                 | GCP, Azure          |
| Container Orchestration | ECS                 | EKS, GKE            |
| Queue                   | SQS + Redis         | RabbitMQ, Kafka     |
| Storage                 | S3                  | GCS, Azure Blob     |
| Experiment Tracking     | Weights & Biases    | MLflow, TensorBoard |
| Pipeline Orchestration  | Custom + CloudWatch | Airflow, Prefect    |
| IaC                     | Terraform           | CDK, CloudFormation |

## Appendix B: Required AWS Resources

1. **IAM Roles**
   - `ringrift-cmaes-coordinator` - SQS, S3, ECS
   - `ringrift-cmaes-worker` - SQS, S3
   - `ringrift-training` - S3, EC2/ECS

2. **VPC Configuration**
   - Private subnets for workers
   - NAT gateway for outbound
   - VPC endpoints for S3, SQS (cost optimization)

3. **Security Groups**
   - Internal communication between coordinator and workers
   - Outbound HTTPS for package downloads

## Appendix C: Files to Create/Modify

### New Files

```
ai-service/
├── app/
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backends.py       # Storage abstraction
│   │   └── cloud.py          # S3/GCS implementations
│   ├── distributed/
│   │   ├── __init__.py
│   │   ├── queue.py          # Queue abstraction
│   │   ├── redis_queue.py
│   │   └── sqs_queue.py
│   └── config/
│       └── training_config.py
├── scripts/
│   ├── cmaes_worker.py       # Standalone worker
│   ├── generate_data_distributed.py
│   └── run_training_pipeline.py
├── docker/
│   ├── Dockerfile.training   # GPU-enabled
│   ├── Dockerfile.worker     # CMA-ES worker
│   └── docker-compose.cloud.yml
└── infrastructure/
    ├── terraform/
    │   ├── main.tf
    │   ├── ecs.tf
    │   ├── sqs.tf
    │   └── s3.tf
    └── scripts/
        ├── deploy_coordinator.sh
        └── deploy_workers.sh
```

### Modified Files

- `run_cmaes_optimization.py` - Add `--distributed` mode
- `generate_data.py` - Add cloud storage upload
- `train_neural_net.py` - GPU optimization, cloud checkpoints
- `requirements.txt` - Add boto3, google-cloud-storage, wandb
