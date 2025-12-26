# RingRift Cluster Full Utilization Guide

## Work Types and Resource Usage

| Work Type           | GPU Usage       | CPU Usage   | Memory | Scripts                       |
| ------------------- | --------------- | ----------- | ------ | ----------------------------- |
| NNUE Training       | HIGH (98%)      | Medium      | High   | `train_nnue.py`               |
| GPU Selfplay        | HIGH (90%+)     | Low         | Medium | `run_gpu_selfplay.py`         |
| ELO Tournaments     | Medium (50-80%) | Medium      | High   | `run_model_elo_tournament.py` |
| Baseline Gauntlet   | Medium (30-60%) | Medium      | Medium | `baseline_gauntlet.py`        |
| Hybrid Selfplay     | LOW (0-10%)     | HIGH        | Medium | P2P orchestrator              |
| CMA-ES Optimization | LOW (0%)        | HIGH (100%) | Low    | `run_cmaes_optimization.py`   |
| Data Aggregation    | LOW (0%)        | Medium      | Low    | `merge_game_dbs.py`           |

## Why Nodes Appear Idle

1. **Hybrid Selfplay is CPU-bound**: 45 selfplay jobs = 0% GPU because MCTS runs on CPU
2. **CMA-ES runs outside P2P**: Not tracked as jobs, but uses 100% CPU
3. **Tournaments use GPU memory**: May show 0% compute but 98% memory

## How to Keep Nodes Fully Utilized

### Automated (Recommended)

1. **Enable resource-aware work router** on each node:

   ```bash
   # Add to cron (runs every 2 minutes)
   */2 * * * * cd ~/ringrift/ai-service && python3 scripts/resource_aware_router.py --rebalance >> /tmp/work_rebalance.log 2>&1
   ```

   Legacy note: `unified_work_orchestrator.py` has been removed; use `resource_aware_router.py` directly.

2. **Enable auto-training trigger** (already deployed):
   ```bash
   # Runs every 5 minutes on leader-eligible nodes
   */5 * * * * cd ~/ringrift/ai-service && python3 scripts/auto_training_trigger.py >> /tmp/auto_training.log 2>&1
   ```

### Manual Commands

**Trigger training on idle GPU nodes:**

```bash
curl -X POST "http://localhost:8770/training/start" \
  -H "Content-Type: application/json" \
  -d '{"board_type":"square8", "num_players":2}'
```

**Start ELO tournament:**

```bash
python scripts/run_model_elo_tournament.py --board square8 --players 2 --games 20 --quick
```

**Start CMA-ES optimization:**

```bash
python scripts/run_cmaes_optimization.py --board square8 --num-players 2
```

**Start baseline gauntlet:**

```bash
python scripts/baseline_gauntlet.py --board square8 --players 2
```

## Optimal Node Assignment

| Node Type     | Best Work Types                                    |
| ------------- | -------------------------------------------------- |
| GH200 (480GB) | Training, Large tournaments, Multi-config training |
| H100          | Training, GPU selfplay                             |
| A10           | Training (smaller batch), Tournaments              |
| Mac (MPS)     | Hybrid selfplay, Small training runs               |
| CPU-only      | CMA-ES, Hybrid selfplay, Data aggregation          |

## Monitoring

Check cluster GPU utilization:

```bash
curl -s http://localhost:8770/status | python3 -c "
import sys, json
d = json.load(sys.stdin)
for n, p in d.get('peers', {}).items():
    if p.get('has_gpu'):
        print(f\"{n}: GPU={p.get('gpu_percent',0)}%, training={p.get('training_jobs',0)}, selfplay={p.get('selfplay_jobs',0)}\")
"
```

## Data Backlog

Check training data availability:

```bash
curl -s "http://localhost:8770/cluster_data_manifest" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for bt, data in (d.get('manifest', {}).get('by_board_type') or {}).items():
    print(f\"{bt}: {data.get('total_games',0):,} games\")
"
```

## Troubleshooting

**GPU at 0% but selfplay running?**

- Normal - hybrid selfplay uses CPU for MCTS
- Start training jobs to use GPU

**Training not auto-triggering?**

- Check if node is leader: `curl http://localhost:8770/health | jq .role`
- Check data manifest: `curl http://localhost:8770/cluster_data_manifest`
- Check cooldown hasn't expired

**Node appears idle but isn't?**

- Check for CMA-ES: `pgrep -fa "HeuristicAI.*json"`
- Check for tournaments: `pgrep -fa "run_model_elo_tournament"`
