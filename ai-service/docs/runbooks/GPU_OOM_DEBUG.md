# GPU Out-of-Memory Debugging Runbook

**Alert:** GPUMemoryExhausted
**Severity:** High
**Component:** Training / Selfplay
**Team:** AI Service

---

## 1. Description

GPU Out-of-Memory (OOM) errors occur when CUDA operations request more VRAM than available. These can crash training jobs, selfplay workers, or evaluation processes.

Common error messages:

- `CUDA out of memory. Tried to allocate X MiB`
- `RuntimeError: CUDA error: out of memory`
- `torch.cuda.OutOfMemoryError`

---

## 2. Impact

- **Training jobs crash** - Loss of partial epoch progress
- **Selfplay workers die** - Reduced data generation throughput
- **Gauntlet evaluation fails** - Model promotion blocked
- **Memory fragmentation** - VRAM not fully reclaimed after OOM

---

## 3. Diagnosis

### 3.1 Check Current GPU Memory State

```bash
# On affected node
nvidia-smi

# Detailed memory breakdown
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# Watch memory over time
watch -n 1 nvidia-smi
```

### 3.2 Check for Zombie Processes

```bash
# Find processes still holding GPU memory
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Kill specific zombie process
kill -9 <pid>

# Kill all Python processes holding GPU (USE WITH CAUTION)
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

### 3.3 Check Memory Usage in Code

```python
import torch

# Current allocation
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Memory snapshot for debugging
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
```

### 3.4 Identify Memory-Heavy Operations

Common culprits:

- Large batch sizes
- Model size vs available VRAM
- Gradient accumulation without clearing
- Large tensor caching
- Multi-GPU memory imbalance

Check batch size against GPU memory:

| GPU         | VRAM | Recommended Batch Size |
| ----------- | ---- | ---------------------- |
| RTX 3060    | 12GB | 64-128                 |
| RTX 4060 Ti | 16GB | 128-256                |
| RTX 4090    | 24GB | 256-512                |
| A10         | 24GB | 256-512                |
| L40S        | 48GB | 512-1024               |
| A100 40GB   | 40GB | 512-1024               |
| A100 80GB   | 80GB | 1024-2048              |
| H100        | 80GB | 1024-2048              |
| GH200       | 96GB | 2048+                  |

---

## 4. Resolution

### 4.1 Clear GPU Memory Cache

```python
import torch
import gc

# Clear cache
torch.cuda.empty_cache()
gc.collect()

# If using CUDA graphs
torch.cuda.reset_peak_memory_stats()
```

### 4.2 Reduce Batch Size

```bash
# Training
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --batch-size 256  # Reduce from 512

# Selfplay
python scripts/selfplay.py \
  --board hex8 --num-players 2 \
  --batch-size 32  # Reduce parallel games
```

### 4.3 Enable Gradient Checkpointing

In training config, enable memory-saving mode:

```python
# In app/training/train.py
model.enable_gradient_checkpointing()

# Or via CLI
python -m app.training.train --gradient-checkpointing
```

### 4.4 Use Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4.5 Adjust Gumbel MCTS Budget

For selfplay OOM, reduce simulation budget:

```python
from app.config.thresholds import GUMBEL_BUDGET_THROUGHPUT

# Use lower budget tier
budget = GUMBEL_BUDGET_THROUGHPUT  # 64 instead of 150
```

Or via environment:

```bash
export RINGRIFT_GUMBEL_BUDGET=64
python scripts/selfplay.py ...
```

### 4.6 Kill and Restart Clean

```bash
# SSH to affected node
ssh ubuntu@<node-ip>

# Kill all AI service processes
pkill -f "python.*app.training"
pkill -f "python.*selfplay"

# Wait for GPU memory release
sleep 5

# Verify memory freed
nvidia-smi

# Restart with reduced memory footprint
python -m app.training.train --batch-size 256 ...
```

---

## 5. Prevention

### 5.1 Pre-flight Memory Check

Add to job startup:

```python
import torch

def check_gpu_memory(required_gb: float = 8.0) -> bool:
    """Check if enough GPU memory is available."""
    if not torch.cuda.is_available():
        return False
    free_memory = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated()) / 1e9
    return free_memory >= required_gb
```

### 5.2 Configure Memory Limits

In training scripts:

```python
# Limit PyTorch memory allocation
torch.cuda.set_per_process_memory_fraction(0.9)

# Or via environment
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 5.3 Use Appropriate Batch Sizes

Configure batch sizes by GPU type in `app/config/thresholds.py`:

```python
GPU_BATCH_SIZE_MAP = {
    "RTX 3060": 64,
    "RTX 4090": 512,
    "A100": 1024,
    "H100": 1024,
    "GH200": 2048,
}
```

### 5.4 Monitor Memory During Training

```python
# Add memory logging to training loop
import torch
import logging

logger = logging.getLogger(__name__)

def log_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

# Call periodically in training loop
if batch_idx % 100 == 0:
    log_memory_stats()
```

---

## 6. Escalation

### 6.1 When to Escalate

- OOM persists after reducing batch size to minimum viable
- Memory leak suspected (usage grows over time)
- Multiple nodes affected simultaneously
- OOM occurs with small models that should fit

### 6.2 Information to Gather

1. Full `nvidia-smi` output at time of failure
2. Training/selfplay command line and config
3. Model architecture (board type, num_players)
4. Batch size and simulation budget
5. Stack trace from OOM error
6. Memory usage timeline if available

### 6.3 Escalation Path

1. Check if node needs restart
2. Try on different GPU type
3. Report to infrastructure team if hardware issue suspected

---

## 7. Related Alerts

- `GPUUtilizationLow` - May indicate process died after OOM
- `TrainingStuck` - OOM can cause training to hang
- `SelfplayThroughputLow` - OOM-killed workers reduce throughput

---

## 8. Quick Reference

```bash
# Clear GPU and restart job
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
sleep 3
python -m app.training.train --batch-size 256 ...

# Check memory before starting
python -c "import torch; print(f'Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f}GB')"

# Debug memory usage
python -c "
import torch
print('Allocated:', torch.cuda.memory_allocated() / 1e9, 'GB')
print('Cached:', torch.cuda.memory_reserved() / 1e9, 'GB')
print('Max:', torch.cuda.max_memory_allocated() / 1e9, 'GB')
"
```

---

**Last Updated:** December 2025
