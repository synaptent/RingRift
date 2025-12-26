# Cluster Connectivity Debugging Runbook

**Alert:** ClusterConnectivityDegraded
**Severity:** High
**Component:** Distributed Infrastructure
**Team:** AI Service

---

## 1. Description

Cluster connectivity issues affect the distributed training pipeline's ability to coordinate nodes, sync data, and distribute work. The RingRift AI cluster uses a P2P mesh network with Tailscale for secure connectivity.

Common symptoms:

- Nodes unreachable via SSH or HTTP
- P2P mesh partitioned
- Data sync failures
- Job distribution stalled

---

## 2. Impact

- **Training pipeline stalled** - Jobs can't be distributed
- **Data sync broken** - Training data not propagating
- **Model distribution failed** - New models not deployed
- **Cluster utilization drops** - Idle GPUs due to coordination failure

---

## 3. Diagnosis

### 3.1 Check P2P Cluster Status

```bash
# Quick cluster health (from any node with P2P running)
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\", \"NONE\")}")
print(f"Alive peers: {d.get(\"alive_peers\", 0)}")
print(f"Total peers: {d.get(\"total_peers\", 0)}")
print(f"Health: {d.get(\"cluster_health\", \"unknown\")}")
'

# Full cluster monitor
python -m app.distributed.cluster_monitor --watch
```

### 3.2 Test Node Connectivity

```bash
# Test SSH connectivity to specific node
ssh -i ~/.ssh/id_cluster -o ConnectTimeout=5 ubuntu@<node-ip> "echo OK"

# Test HTTP health endpoint
curl -s --connect-timeout 5 http://<node-ip>:8765/health

# Test P2P port
curl -s --connect-timeout 5 http://<node-ip>:8770/status

# Batch test all nodes
for host in $(cat config/distributed_hosts.yaml | grep 'host:' | awk '{print $2}'); do
  echo -n "$host: "
  curl -s --connect-timeout 3 http://$host:8770/status > /dev/null && echo "OK" || echo "FAIL"
done
```

### 3.3 Check Tailscale Status

```bash
# On local machine
tailscale status

# Check if nodes are connected
tailscale status | grep -E "(online|offline)"

# Test Tailscale connectivity
tailscale ping <node-tailscale-ip>
```

### 3.4 Check Network Routing

```bash
# Trace route to node
traceroute <node-ip>

# Check if port is reachable
nc -vz <node-ip> 8770

# Check DNS resolution
dig <node-hostname>
```

### 3.5 Check Node Health

```bash
# SSH to node and check services
ssh ubuntu@<node-ip> << 'EOF'
  echo "=== System Status ==="
  uptime

  echo "=== Network Interfaces ==="
  ip addr show | grep -E "inet |tailscale"

  echo "=== P2P Process ==="
  ps aux | grep p2p

  echo "=== Recent Logs ==="
  tail -20 ~/ringrift/ai-service/logs/p2p.log 2>/dev/null || echo "No log file"

  echo "=== Port Status ==="
  ss -tulpn | grep -E "8765|8770"
EOF
```

---

## 4. Resolution

### 4.1 Restart P2P Daemon on Node

```bash
ssh ubuntu@<node-ip> << 'EOF'
  cd ~/ringrift/ai-service
  pkill -f "p2p_daemon" || true
  sleep 2
  nohup python -m app.distributed.p2p_daemon > logs/p2p.log 2>&1 &
  sleep 3
  curl -s localhost:8770/status | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status", "unknown"))'
EOF
```

### 4.2 Fix Tailscale Connectivity

```bash
# On affected node
ssh ubuntu@<node-ip> << 'EOF'
  # Restart Tailscale
  sudo tailscale down
  sleep 2
  sudo tailscale up

  # Check status
  tailscale status
EOF
```

### 4.3 Force Leader Re-election

If the leader node is down:

```bash
# From any healthy node
curl -X POST http://localhost:8770/admin/trigger-election
```

Or restart P2P on voter nodes to trigger election:

```bash
# Restart P2P on all voter nodes (defined in distributed_hosts.yaml)
for voter in nebius-backbone-1 runpod-h100 runpod-a100-1 runpod-a100-2 vultr-a100-20gb; do
  echo "Restarting P2P on $voter..."
  ssh ubuntu@$voter "pkill -f p2p_daemon; cd ~/ringrift/ai-service && nohup python -m app.distributed.p2p_daemon &" &
done
wait
```

### 4.4 Repair Data Sync

```bash
# Check sync status
curl -H "X-Admin-Key: $KEY" http://localhost:8001/admin/sync/status

# Manually trigger sync
curl -X POST -H "X-Admin-Key: $KEY" \
  "http://localhost:8001/admin/sync/trigger?categories=games&categories=models"

# Force full resync
python -c "
from app.coordination.auto_sync_daemon import AutoSyncDaemon
import asyncio

daemon = AutoSyncDaemon()
asyncio.run(daemon.force_full_sync())
"
```

### 4.5 Recover Partitioned Node

If a node is partitioned from the cluster:

```bash
# On partitioned node
ssh ubuntu@<node-ip> << 'EOF'
  # Check if node can reach other nodes
  curl -s --connect-timeout 5 http://nebius-backbone-1:8770/status

  # Restart network stack if needed
  sudo systemctl restart systemd-networkd

  # Restart Tailscale
  sudo tailscale down && sudo tailscale up

  # Restart P2P daemon
  cd ~/ringrift/ai-service
  pkill -f p2p_daemon
  nohup python -m app.distributed.p2p_daemon > logs/p2p.log 2>&1 &
EOF
```

### 4.6 Update Hosts Configuration

If nodes have changed IPs:

```bash
# Edit distributed_hosts.yaml
vim config/distributed_hosts.yaml

# Validate configuration
python -c "
from app.distributed.config_loader import load_distributed_config
config = load_distributed_config()
print(f'Loaded {len(config.hosts)} hosts')
for h in config.hosts[:5]:
    print(f'  {h.name}: {h.host}')
"
```

---

## 5. Prevention

### 5.1 Health Check Monitoring

Ensure health checks are configured:

```yaml
# In prometheus/prometheus.yml
- job_name: 'ringrift-cluster'
  static_configs:
    - targets:
        - 'node1:8770'
        - 'node2:8770'
      # ... all nodes
  metrics_path: /metrics
  scrape_interval: 30s
```

### 5.2 Automatic Recovery

The P2P mesh has built-in recovery:

- **Heartbeat interval**: 5 seconds
- **Failure threshold**: 3 missed heartbeats
- **Auto-recovery**: Nodes rejoin when connectivity restored

Configure in `app/distributed/p2p_backend.py`:

```python
HEARTBEAT_INTERVAL = 5
FAILURE_THRESHOLD = 3
RECOVERY_BACKOFF = 10
```

### 5.3 Node Health Monitor

Enable the node health monitor daemon:

```python
from app.coordination.node_health_monitor import NodeHealthMonitor

monitor = NodeHealthMonitor()
monitor.start()  # Runs periodic health checks
```

### 5.4 Redundant Connectivity

Configure fallback IPs in hosts config:

```yaml
hosts:
  - name: nebius-h100
    host: 89.169.111.139 # Primary (Tailscale)
    fallback_host: 10.0.0.15 # Fallback (internal)
    port: 22
```

---

## 6. Escalation

### 6.1 When to Escalate

- More than 50% of nodes unreachable
- Leader election stuck (no leader for >5 minutes)
- Data sync completely stalled
- Provider-wide outage suspected

### 6.2 Information to Gather

1. `tailscale status` output from multiple nodes
2. P2P cluster status from healthy nodes
3. Network trace/ping results
4. Recent changes to network configuration
5. Provider status page (Lambda, Vast.ai, etc.)

### 6.3 Escalation Path

1. Check provider status pages
2. Test connectivity from different network locations
3. Contact provider support if infrastructure issue
4. Consider failover to backup cluster

---

## 7. Related Alerts

- `ClusterHealthCritical` - Overall cluster health below threshold
- `NodeEvicted` - Specific node removed from cluster
- `SyncHostCritical` - Data sync failures
- `LeaderElectionStalled` - No leader elected

---

## 8. Quick Reference

```bash
# Check cluster health
curl -s http://localhost:8770/status | jq .

# Test all nodes quickly
for ip in $(tailscale status --json | jq -r '.Peer[].TailscaleIPs[0]'); do
  curl -s --connect-timeout 2 http://$ip:8770/status > /dev/null && echo "$ip: OK" || echo "$ip: FAIL"
done

# Restart P2P on single node
ssh ubuntu@<node> "pkill -f p2p_daemon; cd ~/ringrift/ai-service && nohup python -m app.distributed.p2p_daemon &"

# Force sync
curl -X POST http://localhost:8001/admin/sync/trigger

# Check node health
python -m app.distributed.cluster_monitor --watch --interval 5
```

---

## 9. Network Topology Reference

### Provider IP Ranges

| Provider    | IP Range                       | Notes      |
| ----------- | ------------------------------ | ---------- |
| Lambda Labs | 100.x.x.x                      | Tailscale  |
| RunPod      | 38.x.x.x, 104.x.x.x, 193.x.x.x | Public IPs |
| Vast.ai     | Variable                       | Ephemeral  |
| Nebius      | 89.169.x.x                     | Public IPs |
| Vultr       | 208.167.x.x                    | Public IPs |

### Critical Ports

| Port | Service   | Protocol |
| ---- | --------- | -------- |
| 22   | SSH       | TCP      |
| 8765 | HTTP API  | TCP      |
| 8770 | P2P Mesh  | TCP      |
| 8001 | Admin API | TCP      |

### Voter Nodes (Election Quorum)

Defined in `config/distributed_hosts.yaml`:

```yaml
voter_nodes:
  - nebius-backbone-1
  - runpod-h100
  - runpod-a100-1
  - runpod-a100-2
  - vultr-a100-20gb
```

---

**Last Updated:** December 2025
