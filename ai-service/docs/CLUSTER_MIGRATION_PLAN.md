# RingRift Cluster Migration Plan: Vast.ai → GH200

## Executive Summary

Consolidate from 9 scattered Vast.ai instances to 4-8 Lambda GH200 instances for:

- **10-20x throughput improvement** per instance
- **Shared 15PB NFS** eliminating sync overhead
- **Simpler architecture** with fewer nodes to manage
- **Cost parity or savings** per game generated

---

## Current State

### Vast.ai Instances (to wind down)

| Instance       | GPU         | Games           | Data Size | Status |
| -------------- | ----------- | --------------- | --------- | ------ |
| vast-5090-quad | 4x RTX 5090 | 54,721          | ~4GB      | Active |
| vast-5090-a    | RTX 5090    | 51,310          | ~4GB      | Active |
| vast-5090-b    | RTX 5090    | 19,446          | ~2GB      | Active |
| vast-3090-a    | RTX 3090    | 34,221          | ~3GB      | Active |
| vast-3080-dual | 2x RTX 3080 | 47,097          | ~4GB      | Active |
| vast-3070-a    | RTX 3070    | 28,763          | ~3GB      | Active |
| vast-3060ti    | RTX 3060 Ti | 47,913          | ~4GB      | Active |
| vast-3070-b    | RTX 3070    | 21,903          | ~2GB      | Active |
| vast-3090-b    | RTX 3090    | 33,824          | ~3GB      | Active |
| **TOTAL**      |             | **~339K games** | **~36GB** |        |

### Lambda GH200 Instances (scaling up)

| Instance       | GPU        | Memory | NFS         | Status  |
| -------------- | ---------- | ------ | ----------- | ------- |
| lambda-gh200-a | GH200 96GB | 480GB  | 15PB shared | Running |
| lambda-gh200-b | GH200 96GB | 480GB  | 15PB shared | Running |
| lambda-gh200-c | GH200 96GB | 480GB  | 15PB shared | Running |
| lambda-gh200-d | GH200 96GB | 480GB  | 15PB shared | Running |

---

## Migration Steps

### Phase 1: Data Transfer (In Progress)

```bash
# Running in background - check progress:
tail -f /tmp/vast_migration.log

# Data destination:
ssh ubuntu@192.222.51.29 "ls -la /lambda/nfs/RingRift/vast_migration/"
```

**ETA**: ~30-60 minutes for 36GB transfer

### Phase 2: Verify Data Integrity

```bash
# After migration completes, verify game counts:
ssh ubuntu@192.222.51.29 "
  cd /lambda/nfs/RingRift/vast_migration
  for dir in */; do
    echo -n \"\$dir: \"
    find \"\$dir\" -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l
  done
"
```

### Phase 3: Wind Down Vast Instances

**Order**: Wind down lowest-value instances first

1. **Immediate** (low throughput, data transferred):
   - vast-3060ti
   - vast-3070-a
   - vast-3070-b

2. **After 24h verification**:
   - vast-3080-dual
   - vast-3090-a
   - vast-3090-b

3. **Last** (highest throughput):
   - vast-5090-a
   - vast-5090-b
   - vast-5090-quad

**Wind-down command**:

```bash
# From Vast.ai dashboard or CLI:
vastai destroy instance <instance_id>
```

### Phase 4: GH200 Scale-Up

**Recommended final configuration**:

| Role             | Instances        | Board Types     | Throughput |
| ---------------- | ---------------- | --------------- | ---------- |
| Selfplay (sq8)   | 2x GH200         | sq8 2p/4p       | ~70 g/s    |
| Selfplay (large) | 2x GH200         | sq19, hex 3p/4p | ~30 g/s    |
| Training         | 1x H100          | All             | N/A        |
| **TOTAL**        | 4 GH200 + 1 H100 |                 | ~100 g/s   |

**Optional scale-up**: Add 2-4 more GH200s for faster iteration:

- 8 GH200s = ~200 g/s = 17M games/day

---

## Cost Comparison (Estimated)

### Before (Vast.ai)

| Instance Type | Count | $/hr each | Total $/hr    |
| ------------- | ----- | --------- | ------------- |
| RTX 5090      | 3     | ~$1.50    | $4.50         |
| RTX 3090      | 2     | ~$0.50    | $1.00         |
| RTX 3080      | 1     | ~$0.40    | $0.40         |
| RTX 3070      | 2     | ~$0.30    | $0.60         |
| RTX 3060 Ti   | 1     | ~$0.25    | $0.25         |
| **TOTAL**     | 9     |           | **~$6.75/hr** |

Throughput: ~15 g/s combined → **$0.45/1000 games**

### After (Lambda GH200)

| Instance Type | Count | $/hr each | Total $/hr     |
| ------------- | ----- | --------- | -------------- |
| GH200         | 4     | ~$2.50    | $10.00         |
| H100          | 1     | ~$2.00    | $2.00          |
| **TOTAL**     | 5     |           | **~$12.00/hr** |

Throughput: ~100 g/s combined → **$0.12/1000 games**

**Result: 3.75x more cost-efficient per game**

---

## Commands Reference

### Check GH200 Selfplay Status

```bash
for host in 192.222.51.29 192.222.51.167 192.222.51.162 192.222.58.122; do
  echo "$host:"
  ssh ubuntu@$host "tail -1 /tmp/gpu_selfplay.log"
done
```

### Check Shared NFS Data

```bash
ssh ubuntu@192.222.51.29 "du -sh /lambda/nfs/RingRift/*"
```

### Sync to H100 for Training

```bash
ssh ubuntu@192.222.51.29 "rsync -avz /lambda/nfs/RingRift/selfplay_data/ ubuntu@209.20.157.81:~/ringrift/ai-service/data/gh200_data/"
```

---

## Timeline

| Day           | Action                                            |
| ------------- | ------------------------------------------------- |
| Day 0 (Today) | Start data migration, verify GH200 selfplay       |
| Day 1         | Verify migration complete, wind down 3060ti/3070s |
| Day 2         | Wind down 3080/3090s                              |
| Day 3         | Wind down 5090s, full GH200 operation             |
| Day 4+        | Monitor, consider adding more GH200s              |

---

## Rollback Plan

If GH200 issues arise:

1. Keep Vast instances running until GH200 proven stable (24-48h)
2. Data is preserved on shared NFS - no data loss risk
3. Can re-provision Vast instances from saved templates if needed
