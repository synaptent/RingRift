# Distributed Self-Play Setup

> **Doc Status (2025-12-14): Active**

This guide explains how to set up distributed self-play across multiple local Macs, AWS instances, Lambda Labs, and Vast.ai workers.

## Deployment Modes

All distributed scripts support a `--mode` argument for selecting worker hosts:

| Mode     | Description                                      | Use Case                   |
| -------- | ------------------------------------------------ | -------------------------- |
| `local`  | Run on local machine only (no remote workers)    | Development, quick tests   |
| `lan`    | Use local Mac cluster workers from hosts config  | Zero-cost distributed runs |
| `aws`    | Use AWS staging workers (square8 only, 16GB RAM) | Cloud burst capacity       |
| `hybrid` | Use both LAN and AWS workers                     | Maximum parallelism        |

### Cloud Configuration

Configure cloud workers in `config/distributed_hosts.yaml` (copy from template):

```bash
cp config/distributed_hosts.template.yaml config/distributed_hosts.yaml
```

Example cloud host configuration:

```yaml
hosts:
  aws-staging:
    ssh_host: <your-ec2-ip>
    ssh_user: ubuntu
    ssh_key: ~/.ssh/your-key.pem
    work_dir: /home/ubuntu/ringrift
    memory_gb: 16 # Determines eligible board types
    worker_port: 8766
```

**Memory requirements by board type:**

- **square8**: 8GB minimum
- **hex8**: 8GB minimum (radius-4, 61 cells)
- **square19**: 48GB minimum
- **hexagonal**: 48GB minimum (radius-12, 469 cells)

## Quick Start

```bash
# From ai-service/
./scripts/cluster_setup.sh discover     # Find Macs on network
./scripts/cluster_setup.sh test         # Verify SSH connectivity
./scripts/cluster_setup.sh setup-all    # Install dependencies on workers
./scripts/cluster_setup.sh status       # Check worker status
./scripts/run_distributed_selfplay_matrix.sh  # Run distributed jobs
```

## Architecture

```
┌─────────────────┐     SSH     ┌─────────────────┐
│  Main Machine   │────────────>│  Worker Mac 1   │
│  (Coordinator)  │             │  (192.168.1.10) │
│                 │             └─────────────────┘
│  Runs:          │     SSH     ┌─────────────────┐
│  - Job dist.    │────────────>│  Worker Mac 2   │
│  - Result merge │             │  (192.168.1.20) │
└─────────────────┘             └─────────────────┘
```

Jobs are distributed round-robin to workers and run in parallel.

## Setup Steps

### 1. Enable SSH on Worker Macs

On each Mac you want to use as a worker:

1. Open **System Preferences** (or System Settings on macOS 13+)
2. Go to **Sharing**
3. Enable **Remote Login**
4. Note the IP address shown

### 2. Configure SSH Key Authentication

From your main machine, copy your SSH key to each worker:

```bash
# Generate key if you don't have one
ssh-keygen -t ed25519

# Copy to each worker
ssh-copy-id 192.168.1.10
ssh-copy-id 192.168.1.20
```

### 3. Add Workers to Configuration

Copy the template and edit `scripts/cluster_workers.txt`:

```bash
cp scripts/cluster_workers.txt.example scripts/cluster_workers.txt
```

Then add your worker IPs:

```
# List of worker hosts
192.168.1.10
192.168.1.20
```

### 4. Set Up Workers

Run the automated setup for all workers:

```bash
./scripts/cluster_setup.sh setup-all
```

This will:

- Clone or update the RingRift repository
- Create a Python virtual environment
- Install all dependencies

### 5. Verify Setup

```bash
./scripts/cluster_setup.sh test    # Test SSH connectivity
./scripts/cluster_setup.sh status  # Check worker health
```

## Running Distributed Self-Play

```bash
# Run the full matrix across all workers
./scripts/run_distributed_selfplay_matrix.sh

# Run locally only (skip remote workers)
LOCAL_ONLY=1 ./scripts/run_distributed_selfplay_matrix.sh
```

### Environment Variables

| Variable                             | Default                     | Description                     |
| ------------------------------------ | --------------------------- | ------------------------------- |
| `LOCAL_ONLY`                         | 0                           | Set to 1 to skip remote workers |
| `CLUSTER_WORKERS_FILE`               | scripts/cluster_workers.txt | Path to workers file            |
| `REMOTE_PROJECT_DIR`                 | ~/Development/RingRift      | Project dir on workers          |
| `GAMES_2P` / `GAMES_3P` / `GAMES_4P` | 5/3/2                       | Games per player count          |
| `SQUARE8_MAX_MOVES_*P`               | 150/200/250                 | Max moves for square8           |
| `SQUARE19_MAX_MOVES_*P`              | 350/450/550                 | Max moves for square19          |

## Cluster Setup Commands

```bash
./scripts/cluster_setup.sh discover     # Scan network for Macs
./scripts/cluster_setup.sh test         # Test SSH connectivity
./scripts/cluster_setup.sh setup HOST   # Set up single worker
./scripts/cluster_setup.sh setup-all    # Set up all workers
./scripts/cluster_setup.sh start HOST   # Start worker service
./scripts/cluster_setup.sh status       # Check all worker status
```

## HTTP Worker Service (Optional)

For more sophisticated job distribution, workers can run an HTTP service:

```bash
# Start worker service on a remote host
./scripts/cluster_setup.sh start <worker-ip>

# Check health
curl http://<worker-ip>:8765/health
```

The HTTP service supports:

- Health checks
- Task submission via REST API
- Bonjour/mDNS discovery (when network allows)

## Troubleshooting

### SSH Connection Refused

1. Verify Remote Login is enabled in Sharing preferences
2. Check the Mac's firewall settings
3. Test with: `nc -zv <host> 22`

### SSH Auth Failed

1. Copy your SSH key: `ssh-copy-id <host>`
2. Test: `ssh -o BatchMode=yes <host> echo ok`

### Worker Not Starting

Check the worker log:

```bash
ssh <host> "cat /tmp/cluster_worker.log"
```

### Python Version Too Old

The codebase requires Python 3.10+. If the system Python is older:

```bash
# Install via Homebrew
brew install python@3.11

# The setup script will auto-detect newer Python versions
```

## File Locations

- **Worker config**: `scripts/cluster_workers.txt`
- **Setup script**: `scripts/cluster_setup.sh`
- **Distributed runner**: `scripts/run_distributed_selfplay_matrix.sh`
- **Worker service**: `scripts/cluster_worker.py`
- **Results**: `logs/selfplay_matrix/`, `data/games/`

## Advanced: Pipeline Orchestrator

For production training pipelines with P2P cluster coordination, see [PIPELINE_ORCHESTRATOR.md](PIPELINE_ORCHESTRATOR.md).

The pipeline orchestrator provides:

- **P2P Backend**: REST API-based job dispatch without SSH
- **Unified Pipeline**: Self-play → Parity validation → NPZ export
- **State Management**: Resumable pipeline execution
- **Cluster Operations**: Code sync, health checks, data transfer
