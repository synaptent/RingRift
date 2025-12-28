# Transport Escalation Failures Runbook

This runbook covers diagnosis and resolution of transport escalation failures - when the sync system exhausts all transport options for a node.

**Created**: December 28, 2025
**Version**: 1.0
**Severity**: High

---

## Overview

The RingRift sync infrastructure uses a transport escalation chain:

```
P2P HTTP → SSH/Rsync → Base64 Transfer → Disable Node
```

Transport escalation failures occur when all transports fail for a node, resulting in the node being disabled for sync operations.

---

## Transport Escalation Order

| Priority | Transport | Use Case                           | Typical Failure            |
| -------- | --------- | ---------------------------------- | -------------------------- |
| 1        | P2P HTTP  | Fast, low overhead                 | Port blocked, node offline |
| 2        | SSH/Rsync | Reliable, resumable                | SSH key issues, firewall   |
| 3        | Base64    | Last resort, works through proxies | Timeout on large files     |

---

## Detection Methods

### Method 1: Check Sync Router State

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()

# Check disabled nodes
disabled = router.get_disabled_nodes()
for node, info in disabled.items():
    print(f"{node}: disabled at {info['disabled_at']}, reason: {info['reason']}")
    print(f"  Failed transports: {info['failed_transports']}")

# Check transport failures per node
for node, failures in router.get_transport_failures().items():
    print(f"{node}: {failures}")
```

### Method 2: Event Monitoring

```bash
# Check for SYNC_STALLED events
grep "SYNC_STALLED\|SYNC_RETRY" logs/coordination.log | tail -20

# Check for transport escalation
grep "transport escalat\|fallback" logs/sync.log | tail -20
```

### Method 3: Manual Transport Tests

```bash
NODE_IP="100.123.45.67"

# Test P2P HTTP (port 8770)
curl -s --connect-timeout 5 "http://${NODE_IP}:8770/health" && echo "P2P: OK" || echo "P2P: FAILED"

# Test SSH
ssh -o ConnectTimeout=5 -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "echo SSH: OK" 2>/dev/null || echo "SSH: FAILED"

# Test rsync
rsync --dry-run -e "ssh -i ~/.ssh/id_cluster" /tmp/test.txt ubuntu@${NODE_IP}:/tmp/ && echo "Rsync: OK" || echo "Rsync: FAILED"
```

---

## Diagnosis

### Identify Failed Transport Chain

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()
node = "vast-12345"

# Get full transport history
history = router.get_transport_history(node)
for entry in history[-10:]:
    print(f"{entry['time']}: {entry['transport']} -> {entry['result']}")
    if entry.get('error'):
        print(f"  Error: {entry['error']}")
```

### Check Node Connectivity

```bash
# Full connectivity check
python -c "
from app.coordination.sync_router import SyncRouter

router = SyncRouter.get_instance()
node = 'vast-12345'

# Test each transport
for transport in ['p2p', 'ssh', 'rsync', 'base64']:
    result = router._test_transport(node, transport)
    print(f'{transport}: {\"OK\" if result.success else \"FAILED\"} - {result.message}')
"
```

### Check for Network Issues

```bash
# Check Tailscale status
tailscale status | grep ${NODE_NAME}

# Check direct connectivity
ping -c 3 ${NODE_IP}

# Check port availability
nc -zv ${NODE_IP} 8770 2>&1 | grep -o "succeeded\|failed"
nc -zv ${NODE_IP} 22 2>&1 | grep -o "succeeded\|failed"
```

---

## Recovery Procedures

### Option 1: Re-enable Node with Preferred Transport

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()

# Re-enable node and reset transport failures
router.reenable_node("vast-12345")
router.reset_transport_failures("vast-12345")

# Set preferred transport if one works
router.set_preferred_transport("vast-12345", "rsync")
```

### Option 2: Force Specific Transport

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()

# Force rsync for all operations to this node
await router.sync_to_node(
    node="vast-12345",
    data_path="data/games/canonical_hex8.db",
    force_transport="rsync"
)
```

### Option 3: Manual Sync with Base64

When all automated transports fail:

```bash
# Base64 transfer (works through most firewalls/proxies)
cat data/games/canonical_hex8.db | base64 | \
  ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} \
  'base64 -d > ~/ringrift/ai-service/data/games/canonical_hex8.db'

# Verify transfer
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} \
  "ls -la ~/ringrift/ai-service/data/games/canonical_hex8.db"
```

### Option 4: Fix Underlying Network Issue

```bash
# For Tailscale issues
sudo tailscale up --reset
tailscale ping ${NODE_NAME}

# For SSH key issues
ssh-copy-id -i ~/.ssh/id_cluster ubuntu@${NODE_IP}

# For firewall issues (on node)
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "
  sudo ufw allow 8770/tcp
  sudo ufw allow 22/tcp
  sudo ufw reload
"
```

---

## Transport-Specific Troubleshooting

### P2P HTTP Failures

**Common Causes**:

- P2P orchestrator not running
- Port 8770 blocked by firewall
- NAT issues (Lambda Labs nodes)

**Fix**:

```bash
# On the node, check P2P status
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "
  curl -s http://localhost:8770/status || echo 'P2P not running'

  # Restart if needed
  cd ~/ringrift/ai-service
  pkill -f p2p_orchestrator
  nohup python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &
"
```

### SSH/Rsync Failures

**Common Causes**:

- SSH key not authorized
- Host key changed
- Connection timeout

**Fix**:

```bash
# Clear old host key
ssh-keygen -R ${NODE_IP}

# Test with verbose
ssh -vv -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "echo OK"

# Check authorized_keys on node
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "cat ~/.ssh/authorized_keys"
```

### Base64 Transfer Failures

**Common Causes**:

- File too large (>500MB slow)
- Connection reset during transfer
- Disk space on target

**Fix**:

```bash
# Check disk space on target
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "df -h ~/ringrift"

# Use chunked transfer for large files
split -b 100M data/games/large.db large_chunk_
for chunk in large_chunk_*; do
  cat "$chunk" | base64 | ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} \
    "base64 -d >> ~/ringrift/ai-service/data/games/large.db"
done
```

---

## SYNC_STALLED Event Handler

The `SyncRouter._on_sync_stalled()` handler (December 2025) implements automatic escalation:

```python
TRANSPORT_ESCALATION_ORDER = ["p2p", "http", "rsync", "base64"]

async def _on_sync_stalled(self, event: dict) -> None:
    node = event.get("node")
    failed_transport = event.get("transport")

    # Track failure
    self._failed_transports.setdefault(node, set()).add(failed_transport)

    # Try next transport
    current_idx = TRANSPORT_ESCALATION_ORDER.index(failed_transport)
    if current_idx < len(TRANSPORT_ESCALATION_ORDER) - 1:
        next_transport = TRANSPORT_ESCALATION_ORDER[current_idx + 1]
        await self.emit_event("SYNC_RETRY_REQUESTED", {
            "node": node,
            "preferred_transport": next_transport
        })
    else:
        # All transports exhausted
        self._disable_node_sync(node, "all_transports_exhausted")
```

---

## Prevention

### 1. Multi-Transport Configuration

```yaml
# In distributed_hosts.yaml
vast-12345:
  transports:
    - type: p2p
      priority: 1
    - type: ssh
      priority: 2
      fallback: true
    - type: base64
      priority: 3
      last_resort: true
```

### 2. Transport Health Monitoring

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()

# Periodic transport health check
for node in router.get_sync_nodes():
    for transport in ["p2p", "ssh", "rsync"]:
        result = await router.test_transport(node, transport)
        if not result.success:
            # Log warning, don't disable yet
            logger.warning(f"Transport {transport} degraded for {node}")
```

### 3. Proactive Failover

```bash
# Set up backup transport before primary fails
export RINGRIFT_SYNC_BACKUP_TRANSPORT=rsync
export RINGRIFT_SYNC_FAILOVER_THRESHOLD=2  # Fail over after 2 failures
```

---

## Monitoring

### Key Metrics

| Metric             | Alert Threshold   | Source            |
| ------------------ | ----------------- | ----------------- |
| transport_failures | > 5 per node/hour | SyncRouter        |
| disabled_nodes     | > 20% of cluster  | SyncRouter        |
| escalation_count   | > 10 per hour     | SyncRouter events |

### Dashboard Query

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()
stats = router.get_stats()

print(f"Active transports: {stats['active_transports']}")
print(f"Disabled nodes: {stats['disabled_node_count']}")
print(f"Escalations today: {stats['escalations_today']}")
```

---

## Related Documentation

- [CLUSTER_CONNECTIVITY.md](CLUSTER_CONNECTIVITY.md) - Network issues
- [CLUSTER_SYNCHRONIZATION.md](CLUSTER_SYNCHRONIZATION.md) - Sync procedures
- [SYNC_HOST_CRITICAL.md](SYNC_HOST_CRITICAL.md) - Critical sync hosts
