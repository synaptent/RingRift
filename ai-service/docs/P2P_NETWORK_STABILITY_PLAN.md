# P2P Network Stability Implementation Plan

## Executive Summary

This plan addresses P2P network instability affecting Lambda, Vast, and Hetzner nodes. The goal is to get all 40-50 nodes stably connected to the P2P network with reliable leader election.

### Current Issues

1. **Stale voter configuration** - gh200-a, e, f are listed as voters but retired/failing
2. **Aggressive node_resilience behavior** - Kills P2P when /status endpoint times out
3. **Intermittent Tailscale connectivity** - Causes false-positive disconnection detection
4. **Vast nodes using userspace Tailscale** - Cannot route TCP, need SSH tunnels
5. **Leader election failures** - No stable quorum due to voter disagreement
6. **Missing aiohttp** - Some Lambda nodes lack aiohttp for async HTTP server

### Infrastructure

| Type         | Nodes                          | Status           |
| ------------ | ------------------------------ | ---------------- |
| Lambda GH200 | ~14 nodes (g-s, b-new, d)      | Most working     |
| Lambda Other | h100, 2xh100, a10              | Working          |
| Vast.ai      | vast-5090, vast-4x5090, others | Need SSH tunnels |
| Hetzner      | cpu1, cpu2, cpu3               | Working          |
| AWS          | aws-staging, aws-proxy         | Working          |
| Mac          | mac-studio, mbp-64gb, mbp-16gb | Working          |

---

## Phase 1: Update Voter Configuration (Immediate)

### Problem

Current voter list in `/etc/ringrift/node.conf` includes retired/failing nodes:

- lambda-gh200-a (RETIRED)
- lambda-gh200-e (RETIRED)
- lambda-gh200-f (47 failures)

### Solution

Update voter list to only include stable nodes:

**New Voter List (10 nodes):**

```
aws-staging
lambda-h100
lambda-2xh100
lambda-a10
lambda-gh200-c
lambda-gh200-d
lambda-gh200-g
lambda-gh200-h
lambda-gh200-i
lambda-gh200-o
```

### Steps

#### 1.1 Update node.conf on all Lambda nodes

```bash
VOTERS="aws-staging,lambda-h100,lambda-2xh100,lambda-a10,lambda-gh200-c,lambda-gh200-d,lambda-gh200-g,lambda-gh200-h,lambda-gh200-i,lambda-gh200-o"

# For each Lambda node:
ssh -i ~/.ssh/id_cluster ubuntu@<NODE_IP> "
  sudo sed -i 's/RINGRIFT_P2P_VOTERS=.*/RINGRIFT_P2P_VOTERS=$VOTERS/' /etc/ringrift/node.conf
  sudo systemctl restart ringrift-p2p.service
"
```

#### 1.2 Update distributed_hosts.yaml

Remove `p2p_voter: true` from:

- lambda-gh200-a
- lambda-gh200-e
- lambda-gh200-f

Add `p2p_voter: true` to:

- lambda-gh200-g
- lambda-gh200-h
- lambda-gh200-i
- lambda-gh200-o

---

## Phase 2: Fix node_resilience.py Health Check Logic

### Problem

The `node_resilience.py` daemon kills P2P when `/status` endpoint times out:

1. `/status` timeout is only 5 seconds (line 449)
2. Under load, `/status` can take 30+ seconds
3. Daemon sees "connection lost" and kills P2P process
4. This creates a kill loop

### Solution

Modify `node_resilience.py` to be less aggressive:

#### 2.1 Increase /status timeout

```python
# Line 449: Change timeout from 5 to 15 seconds
with urllib.request.urlopen(url, timeout=15) as response:
```

#### 2.2 Add retry logic for health checks

```python
def check_p2p_health(self) -> bool:
    """Check if P2P orchestrator is running and connected."""
    for attempt in range(3):
        try:
            url = f"http://localhost:{self.config.p2p_port}/health"
            with urllib.request.urlopen(url, timeout=5 + attempt * 3) as response:
                data = json.loads(response.read().decode())
                return data.get("status") == "ok" or data.get("healthy", False)
        except Exception:
            if attempt < 2:
                time.sleep(2)  # Brief pause before retry
    return False
```

#### 2.3 Add grace period before killing P2P

```python
# Track unhealthy duration
self.p2p_unhealthy_since: float | None = None

def run_once(self):
    p2p_healthy = self.check_p2p_health()

    if p2p_healthy:
        self.p2p_unhealthy_since = None
        return

    # Track when P2P became unhealthy
    if self.p2p_unhealthy_since is None:
        self.p2p_unhealthy_since = time.time()

    unhealthy_duration = time.time() - self.p2p_unhealthy_since

    # 5 minute grace period before intervention
    if unhealthy_duration < 300:
        logger.warning(f"P2P unhealthy for {unhealthy_duration:.0f}s, waiting...")
        return

    # Only now restart P2P
    logger.error(f"P2P unhealthy for {unhealthy_duration:.0f}s, restarting")
    self.start_p2p_orchestrator()
```

#### 2.4 Separate liveness from cluster connectivity

- Only restart P2P if `/health` fails (process dead)
- Don't restart if `/status` fails (temporary network issue)
- P2P will self-heal cluster connectivity

---

## Phase 3: Install aiohttp on All Lambda Nodes

### Problem

P2P orchestrator requires `aiohttp` but some nodes are missing it.

### Solution

#### 3.1 Install on all nodes

```bash
for ip in $(cat lambda_ips.txt); do
    ssh -i ~/.ssh/id_cluster ubuntu@$ip "
        # Install in venv
        ~/ringrift/ai-service/venv/bin/pip install -q 'aiohttp>=3.9.0' 2>/dev/null || true

        # Install system-wide as fallback
        pip3 install -q 'aiohttp>=3.9.0' 2>/dev/null || true
        sudo pip3 install -q 'aiohttp>=3.9.0' 2>/dev/null || true
    "
done
```

#### 3.2 Verify requirements.txt includes aiohttp

```bash
grep -q 'aiohttp' ai-service/requirements.txt || echo 'aiohttp>=3.9.0' >> ai-service/requirements.txt
```

---

## Phase 4: Establish SSH Tunnels for Vast Nodes

### Problem

Vast nodes use "userspace" Tailscale which cannot route TCP traffic. They appear connected via Tailscale but can't actually communicate over 100.x.x.x IPs.

### Solution

Use SSH reverse tunnels through a relay node.

#### 4.1 Designate relay nodes

- Primary: `lambda-h100` (209.20.157.81 / 100.78.101.123)
- Secondary: `lambda-2xh100` (192.222.53.22 / 100.97.104.89)

#### 4.2 Deploy SSH tunnel on each Vast node

```bash
# On each Vast node:
cd /root/ringrift/ai-service

# Kill any existing tunnels
pkill -f 'ssh.*-R.*8770' || true

# Copy Lambda SSH key
echo "$LAMBDA_SSH_KEY" > /tmp/lambda_key
chmod 600 /tmp/lambda_key

# Start reverse tunnel to relay
nohup ssh -o StrictHostKeyChecking=no -i /tmp/lambda_key \
    -N -R 0:localhost:8770 \
    ubuntu@192.222.53.22 \
    > /tmp/tunnel.log 2>&1 &

# Start P2P with relay peers
PYTHONPATH=. python3 scripts/p2p_orchestrator.py \
    --node-id $NODE_ID \
    --port 8770 \
    --peers localhost:8771 \
    --relay-peers localhost:8771
```

#### 4.3 Configure autossh for persistence

```bash
# Install autossh if not present
apt-get install -y autossh

# Use autossh instead of ssh for automatic reconnection
autossh -M 0 -f \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -o "StrictHostKeyChecking=no" \
    -i /tmp/lambda_key \
    -N -R 0:localhost:8770 \
    ubuntu@192.222.53.22
```

---

## Phase 5: Improve Tailscale Connectivity Handling

### Problem

Tailscale IPs (100.x.x.x) can be intermittently unreachable, causing false disconnection detection.

### Solution

#### 5.1 Add public IP fallbacks in p2p_hosts.yaml

```yaml
known_hosts:
  - host: 100.78.101.123 # Tailscale IP (primary)
    port: 8770
    name: lambda-h100
    fallback_host: 209.20.157.81 # Public IP (fallback)
```

#### 5.2 Increase heartbeat timeout

The P2P already uses 180s timeout (PEER_TIMEOUT), which is appropriate.

#### 5.3 Prefer public IPs for voter communication

For critical voter operations (leader election), prefer public IPs which are more reliable than Tailscale mesh.

---

## Phase 6: Deployment Procedure

### Rolling Restart (preserve cluster stability)

```bash
#!/bin/bash
set -e

# 1. Deploy updated configs
echo "=== Deploying configs ==="
./scripts/deploy_voter_config.sh

# 2. Restart voters one at a time (30s wait between)
echo "=== Restarting voters ==="
VOTERS=(
    209.20.157.81     # lambda-h100
    192.222.53.22     # lambda-2xh100
)

for voter in "${VOTERS[@]}"; do
    echo "Restarting voter: $voter"
    ssh -o ConnectTimeout=10 -i ~/.ssh/id_cluster ubuntu@$voter \
        "sudo systemctl restart ringrift-p2p"
    sleep 30
done

# 3. Wait for leader election
echo "=== Waiting for leader election ==="
sleep 60

# 4. Check leader
LEADER=$(curl -s http://209.20.157.81:8770/status | python3 -c "import sys,json; print(json.load(sys.stdin).get('leader_id','NONE'))")
echo "Leader: $LEADER"

if [ "$LEADER" == "NONE" ]; then
    echo "ERROR: No leader elected"
    exit 1
fi

# 5. Restart non-voters in parallel
echo "=== Restarting non-voters ==="
# ... restart remaining nodes

# 6. Set up Vast tunnels
echo "=== Setting up Vast tunnels ==="
# ... deploy tunnels to Vast nodes

# 7. Final verification
echo "=== Verification ==="
curl -s http://209.20.157.81:8770/status | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'Leader: {d.get(\"leader_id\")}')
print(f'Voters alive: {d.get(\"voters_alive\")}/10')
print(f'Alive peers: {d.get(\"alive_peers\")}')
"
```

---

## Verification Checklist

After deployment, verify:

- [ ] Leader elected (not None)
- [ ] At least 6/10 voters alive
- [ ] 20+ alive peers
- [ ] All Lambda GH200 nodes visible
- [ ] Vast nodes visible (vast-5090, vast-4x5090)
- [ ] Hetzner CPU nodes visible
- [ ] AWS nodes visible
- [ ] Mac nodes visible
- [ ] No "P2P connection lost" in node_resilience logs
- [ ] No SIGKILL in P2P service journal

---

## Files to Modify

| File                                       | Change                     |
| ------------------------------------------ | -------------------------- |
| `ai-service/config/distributed_hosts.yaml` | Update p2p_voter flags     |
| `ai-service/scripts/node_resilience.py`    | Fix health check logic     |
| `/etc/ringrift/node.conf` (each node)      | Update RINGRIFT_P2P_VOTERS |
| `ai-service/requirements.txt`              | Ensure aiohttp listed      |

---

## Estimated Timeline

| Phase   | Tasks               | Priority    |
| ------- | ------------------- | ----------- |
| Phase 1 | Update voter config | Immediate   |
| Phase 2 | Fix node_resilience | High        |
| Phase 3 | Install aiohttp     | High        |
| Phase 4 | Vast SSH tunnels    | Medium      |
| Phase 5 | Tailscale fallbacks | Low         |
| Phase 6 | Deploy & verify     | After above |

---

## Rollback Plan

If issues occur:

1. Restore original `/etc/ringrift/node.conf` from backup
2. Restart all P2P services
3. Kill any SSH tunnels on Vast nodes
4. Revert node_resilience.py changes
