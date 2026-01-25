# P2P Cluster Monitoring Log

## Goal

Get and keep 20+ nodes stably connected to P2P network for 4+ hours with:

- Autonomous master loop operation
- Successful data sync to S3 and between nodes
- High-quality selfplay, gauntlets, tournaments
- Iterative NN training generations

---

## Monitoring Session 1 - Jan 24, 2026

### Check 1: 03:06 CST

**Cluster Status:**

- Node: mac-studio (leader)
- Alive Peers: 6
- Voters: 2/8 (CRITICAL - low quorum)

**Issues Detected:**

1. **Network partition**: 14/18 peers unreachable (78%)
2. **SSH failures**:
   - nebius-h100-3: "tailscale: failed to look up local user 'armand'"
   - vultr-a100-20gb: "Permission denied, please try again"
3. **Connection failures** to multiple Lambda GH200 nodes
4. **Selfplay jobs failing** with exit code -9 (SIGKILL)
5. **Missing neural network checkpoints**: GumbelMCTSAI requires NN but failed to load
6. **Job management error**: 'TrainingJob' object has no attribute 'target_node'

**Root Causes Identified:**

1. **SSH key/user misconfiguration**: Nodes expect different users (root vs ubuntu vs armand)
2. **P2P not running on remote nodes**: Need to restart P2P on cluster nodes
3. **Missing models on nodes**: Canonical models not synced to all nodes
4. **Code bug**: TrainingJob missing target_node attribute

### Check 2: 03:11 CST

**Cluster Status:**

- Node: mac-studio (leader)
- Active Peers: 9
- Total Peers: 18
- Voters: 6/8 (quorum OK)
- Uptime: 25 min
- Healthy: true

**Improvement:**

- Peers increased from 6 to 9 (50% improvement)
- Cluster is converging through gossip
- Many Lambda GH200 nodes now visible

**Nodes with P2P Running (17 confirmed):**

- Lambda GH200: 1, 2, 3, 4, 5, 8, 9, 10, 11, training (10 nodes)
- Nebius: h100-1, h100-3 (2 nodes)
- Hetzner: cpu1, cpu2, cpu3 (3 nodes)
- Local: mac-studio, local-mac (2 nodes)

**Nodes Unreachable (11 nodes):**

- macbook-pro-3, macbook-pro-4, macbook-pro-5: SSH issues
- vultr-a100-20gb: SSH permission denied
- nebius-backbone-1: SSH timeout
- vast-\*: 6 nodes with SSH permission denied or timeout

---

=== MONITORING SESSION - Sat Jan 24 03:33:12 CST 2026 ===

Check 1 - 03:33:12
Node ID: mac-studio
Leader: None
Active Peers: 4
Voters: 4/7 (quorum OK)

Observations:

- P2P started successfully
- GossipPeerPromotionLoop code added
- Quorum recovered
- Need to continue monitoring to verify 20+ nodes

Check 2 - 03:38:10
Node ID: mac-studio
Leader: vultr-a100-20gb
Active Peers: 6
Voters: 6/7 (quorum OK)

Improvement: Went from 4 to 6 active peers
Goal: 20+ nodes for 4+ hours

=== AUTOMATED MONITORING SESSION - Sat Jan 24 03:38:23 CST 2026 ===
Will perform 6 checks at 600s intervals

Check 1/6 - 03:38:25
Node: mac-studio, Leader: vultr-a100-20gb
Active Peers: 6, Voters: 6/7 (quorum OK)

Check 2/6 - 03:48:27
Node: mac-studio, Leader: vultr-a100-20gb
Active Peers: 8, Voters: 6/7 (quorum OK)

Check 3/6 - 03:58:29
Node: mac-studio, Leader: None
Active Peers: 3, Voters: 3/8 (quorum OK)

Check 4/6 - 04:08:37
Node: mac-studio, Leader: None
Active Peers: 6, Voters: 5/8 (quorum OK)

Check 5/6 - 04:18:41
Node: mac-studio, Leader: None
Active Peers: 8, Voters: 5/8 (quorum OK)

Check 6/6 - 14:00:11: P2P NOT RESPONDING
=== MONITORING SESSION COMPLETE - Sat Jan 24 14:00:11 CST 2026 ===

=== ANALYSIS OF MONITORING PASS 1 ===
Time: Sat Jan 24 14:03:58 CST 2026

Findings:

1. Cluster peaked at 8 active peers (target: 20+)
2. Leader election unstable - leader went to None multiple times
3. P2P crashed due to AttributeError bug (now fixed)

Current Known Peers (11):

- hetzner-cpu1, hetzner-cpu2, hetzner-cpu3 (3 voters)
- lambda-gh200-1 (1 of 11 Lambda nodes)
- mac-studio, local-mac (2 coordinators)
- nebius-h100-1, nebius-h100-3 (2 of 3 nebius)
- vast-29118471, vast-29126088 (2 Vast nodes)
- vultr-a100-20gb (1 voter)
- test (invalid/test node)

Missing Nodes:

- 10 Lambda GH200 nodes (2, 3, 4, 5, 8, 9, 10, 11, training)
- 3 MacBook nodes (macbook-pro-3, 4, 5)
- nebius-backbone-1
- Many Vast.ai nodes

Root Cause Hypothesis:

1. Lambda GH200 nodes are NAT-blocked and require relay
2. MacBook nodes may be offline (not always-on)
3. Many Vast.ai nodes may have terminated or changed IPs

Next Steps:

1. Start second monitoring pass with fixed code
2. Check if Lambda nodes are actually running P2P
3. Verify relay configuration for NAT-blocked nodes

=== MONITORING SESSION 2 - Sat Jan 24 14:05:38 CST 2026 ===
Starting 6 checks at 10-minute intervals (60 min total)

=== AUTOMATED MONITORING SESSION - Sat Jan 24 14:05:40 CST 2026 ===
Will perform 6 checks at 600s intervals

Check 1/6 - 14:05:45
Node: local-mac, Leader: None
Active Peers: 7, Voters: 5/7 (quorum OK)

Check 2/6 - 14:15:49
Node: local-mac, Leader: None
Active Peers: 7, Voters: 5/7 (quorum OK)

### Interim Check - 20:30 CST

**Current Status:**

- Node: local-mac (follower)
- Leader: None (no leader elected)
- Active Peers: 8
- Voters: 6/7 (quorum OK)

**Root Cause Identified: P2P Event Loop Blocking on Lambda Nodes**

Lambda GH200 nodes are running P2P and listening on port 8770, but the HTTP endpoints
are unresponsive because the event loop is blocked during startup. Symptoms:

1. TCP connection succeeds (Tailscale connectivity works)
2. HTTP request times out (no response from blocked event loop)
3. This affects all Lambda GH200 nodes

**Why This Matters:**

- Lambda nodes have 10 of 11 potential peers (96GB GPU each)
- Without Lambda nodes, cluster is limited to ~8 peers
- Cannot reach 20+ node target without fixing Lambda

**Immediate Action Needed:**

- Investigate blocking operation in P2P startup
- Check for synchronous SQLite or I/O operations in startup path
- Consider adding event loop health monitor to detect and recover from blocks

Check 3/6 - 14:25:50
Node: local-mac, Leader: None
Active Peers: 6, Voters: 5/7 (quorum OK)

---

## Final Analysis - Jan 24, 2026

### Monitoring Summary

**Pass 1 Results:**

- Started: 03:38 CST
- Peak peers: 8
- Issues: P2P crashed (AttributeError bug - now fixed), leader election unstable

**Pass 2 Results (Current):**

- Time: 14:26 CST
- Active peers: 8 (including 1 Lambda node: lambda-gh200-2)
- Voters: 5/7 (quorum OK)
- Leader: None (no leader elected despite quorum)
- Progress: Lambda connectivity partially restored

### Root Causes Identified

#### 1. P2P Event Loop Blocking on Lambda Nodes (CRITICAL)

**Evidence:**

- Lambda P2P starts and listens on port 8770
- TCP connections succeed (Tailscale mesh works)
- HTTP requests time out (no response)
- Connection backlog grows (26-33 pending connections)
- Event loop stuck after startup completes

**Impact:** Lambda GH200 nodes (11 nodes, 96GB GPU each) cannot participate in cluster.

**Fix Required:** Investigate remaining blocking operations in startup path:

- Check `_seed_selfplay_scheduler_game_counts_sync()`
- Check state loading operations
- Add event loop health monitor to auto-restart when blocked

#### 2. Asymmetric Tailscale Connectivity

**Evidence:**

- local-mac can reach Lambda via Tailscale
- Lambda can reach Hetzner/Vultr via Tailscale
- Lambda cannot reliably reach Mac nodes (timeouts)

**Impact:** Limits cluster connectivity patterns.

**Fix Required:**

- Verify all nodes have consistent Tailscale configuration
- Consider adding relay fallback for problematic routes

#### 3. Leader Election Not Triggering Despite Quorum

**Evidence:**

- 5/7 voters connected (quorum OK)
- LeaderProbeLoop running (no-leader detection enabled)
- No leader elected for 2+ hours

**Potential Causes:**

- LeaderProbeLoop startup grace period (20s)
- All voters think they're followers
- Election cooldown period active

**Fix Required:**

- Check `_no_leader_since` tracking in LeaderProbeLoop
- Verify election triggers are working
- May need to manually trigger election or restart P2P

### Cluster Connectivity Map

```
Connected (8 peers):
- hetzner-cpu1 (voter)
- hetzner-cpu2 (voter)
- lambda-gh200-2 (GPU)
- nebius-h100-1 (GPU)
- nebius-h100-3 (voter)
- vast-29118471 (GPU)
- vast-29126088 (GPU)
- vultr-a100-20gb (voter)

Missing/Unreachable:
- lambda-gh200-1, 3-11, training (10 Lambda nodes - event loop blocked)
- hetzner-cpu3 (voter - unreachable)
- mac-studio (coordinator - not in peer list)
- macbook nodes (offline)
- Some Vast.ai nodes (terminated/changed)
```

### Recommendations

#### Immediate Actions (P0)

1. **Fix Lambda P2P event loop blocking**
   - Profile startup path for blocking calls
   - Wrap remaining sync operations in `asyncio.to_thread()`
   - Add auto-restart when event loop latency > 5s

2. **Manually trigger leader election**

   ```bash
   curl -X POST http://localhost:8770/force_election
   ```

3. **Restart P2P on all Lambda nodes with updated code**
   - Use `update_all_nodes.py --restart-p2p`
   - After fixing the blocking issue

#### Short-term (P1)

4. **Add P2P startup health gate**
   - Don't accept HTTP connections until event loop verified responsive
   - Prevents "accepting connections but not processing" state

5. **Improve voter health detection**
   - If quorum OK but no leader for > 60s, force election
   - This should already be in LeaderProbeLoop - verify it's working

#### Medium-term (P2)

6. **Tailscale connectivity monitoring**
   - Add dashboard showing node-to-node reachability
   - Alert on asymmetric connectivity

7. **Auto-recovery for stuck P2P processes**
   - Watchdog that detects unresponsive P2P
   - Auto-restarts when HTTP health times out

---

## Next Steps

1. Investigate the blocking operation in Lambda P2P startup
2. Deploy fix and restart all Lambda nodes
3. Once Lambda nodes are connected, verify 20+ nodes and leader election
4. Continue monitoring for 4+ hours stability

Check 4/6 - 14:35:52
Node: local-mac, Leader: None
Active Peers: 5, Voters: 4/7 (quorum OK)

Check 5/6 - 14:45:53
Node: local-mac, Leader: None
Active Peers: 4, Voters: 4/7 (quorum OK)

Check 6/6 - 14:55:55
Node: local-mac, Leader: None
Active Peers: 6, Voters: 5/7 (quorum OK)

=== MONITORING SESSION COMPLETE - Sat Jan 24 14:55:55 CST 2026 ===
[2026-01-24 22:14:49] Status Check #1
Leader: local-mac
Active peers: 6/17
Voters: 5/8 alive (quorum OK)
Offline voters: mac-studio, macbook-pro-3, macbook-pro-4
Issues:

- Lambda GH200 GPU driver issues (training, 3)
- Selfplay jobs failing with exit -9 (OOM)
- vast-29118471 runaway processes (983)
  Memory: 72%, Disk: 66%

[2026-01-24 22:16:28] Root Cause Analysis #1

CONFIGURED VOTERS (7 total, quorum=4):

1. hetzner-cpu1 (100.94.174.19) - Active
2. hetzner-cpu2 (100.67.131.72) - Active
3. hetzner-cpu3 (100.126.21.102) - Active
4. vultr-a100-20gb (100.94.201.92) - Active
5. nebius-h100-1 (100.106.19.6) - Pingable, P2P NOT running
6. nebius-h100-3 (100.109.195.71) - NOT reachable via Tailscale
7. mac-studio (100.107.168.125) - Pingable, P2P NOT running

ACTUAL VOTER STATUS:

- Only 4/7 configured voters have P2P running
- P2P dynamically learned voter set with 8 nodes including laptops
- Voter mismatch between config and runtime

ROOT CAUSES:

1. Nebius nodes marked 'offline for cost reasons' - P2P not started
2. mac-studio P2P not running (coordinator role only?)
3. P2P gossip protocol learning voters not in YAML config

RECOMMENDATIONS:

1. Start P2P on nebius-h100-1 (reachable, has GPU)
2. Verify mac-studio role - should it run P2P?
3. Consider updating voter set to reflect actual available nodes

[2026-01-24 22:17:35] Status Check #2
Leader: local-mac
Active peers: 8
Uptime: 14.1 min
Voters: 5/8 alive (quorum OK)
Status: STABLE

Notes:

- nebius-h100-1 P2P stuck at 98% CPU (Raft bug, running as root - can't restart)
- nebius-h100-3 unreachable via Tailscale
- mac-studio no P2P running

[2026-01-24 22:18:26] Status Check #3
Leader: local-mac
Active peers: 8
Voters: 5/8 (quorum OK)

Node Activity:

- hetzner-cpu1: cpu=42%, gpu=0%
- hetzner-cpu2: cpu=2%, gpu=0%
- hetzner-cpu3: cpu=10%, gpu=0%
- lambda-gh200-1: cpu=1%, gpu=0%
- lambda-gh200-3: cpu=1%, gpu=0%
- lambda-gh200-4: cpu=4%, gpu=98% (ACTIVE)
- lambda-gh200-training: cpu=1%, gpu=0%
- vultr-a100-20gb: cpu=100%, gpu=12%

Status: STABLE - cluster actively processing work

[2026-01-24 22:20:57] Status Check #4
Leader: local-mac
Active peers: 6-8 (fluctuating)
Uptime: 17.4 min
Voters: 5/8 (quorum OK)

Notes:

- Connection errors to 100.92.222.49 (macbook-pro-5) - unreachable
- Peer count fluctuating between health endpoint and cluster health logs

[2026-01-24 22:22:19] Status Check #5
Leader: local-mac
Active peers: 9 (growing!)
Uptime: 18.9 min
Voters: 5/8 (quorum OK)

Nodes came online:

- vast-29126088
- lambda-gh200-training

Status: HEALTHY - cluster self-healing, nodes reconnecting

[2026-01-24 22:27:56] Status Check #6
Leader: local-mac
Active peers: 6 (local view)

ISSUE DETECTED: View inconsistency

- local-mac reports hetzner-cpu1 as offline
- hetzner-cpu1 reports healthy with 10 peers and sees local-mac as leader

Possible causes:

1. Network asymmetry
2. Heartbeat timing
3. SWIM protocol divergence

hetzner-cpu1 view:

- 10 active peers
- local-mac is leader
- Uptime: 405 seconds

Status: INVESTIGATING view inconsistency

[2026-01-24 22:28:19] CRITICAL: CLUSTER SPLIT-BRAIN DETECTED

Node views:

- local-mac: 6 peers, leader=local-mac
- hetzner-cpu1: 10 peers, leader=local-mac
- hetzner-cpu2: 5 peers, leader=nebius-h100-3 (!)
- hetzner-cpu3: 3 peers, leader=None (!)

ROOT CAUSE: Network partition causing cluster split-brain

- Multiple nodes claiming leadership
- Divergent peer views
- SWIM protocol not converging

IMMEDIATE ACTION NEEDED:

1. Investigate network connectivity between nodes
2. Force leader election or restart nodes
3. Check for firewall/routing issues

Status: CRITICAL - cluster split into partitions

[2026-01-24 22:28:52] ROOT CAUSE IDENTIFIED

The split-brain was caused by:

1. nebius-h100-1 is running P2P with WRONG node-id 'nebius-h100-3'
2. This stuck P2P (98% CPU) announced itself as leader
3. local-mac accepted the leadership announcement
4. Then arbiter became unreachable, causing local-mac to step down
5. No new leader election succeeded

Current state:

- leader_id: None (no leader)
- Cluster is leaderless and split

FIX REQUIRED:

1. Kill the stuck P2P on nebius-h100-1 (running as root)
2. Or wait for the misconfigured node to be excluded from voting

This is the fundamental instability root cause.

[2026-01-24 22:29:02] Status Check #7
Leader: hetzner-cpu3 (NEW LEADER!)
Active peers: 9
local-mac Role: follower

Cluster recovered with new leader election.
Previous split-brain resolved (for now).

================================================================================
MONITORING SUMMARY - 2026-01-24 22:32:14
================================================================================

HEALTH CHECK RESULTS (3 checks over 20 minutes):

- 22:11: 6 peers, leader=local-mac
- 22:21: 8 peers, leader=local-mac
- 22:31: 8 peers, leader=local-mac

CURRENT STATUS:

- Leader: hetzner-cpu3
- Active peers: 4
- Uptime: ~28 minutes
- Cluster state: DEGRADED (below target of 20 peers)

ROOT CAUSES IDENTIFIED:

1. nebius-h100-1 running P2P with WRONG node-id 'nebius-h100-3'
   - Stuck at 98% CPU (Raft bug)
   - Keeps announcing itself as leader causing split-brain
   - Running as root, cannot kill via SSH as ubuntu user

2. Many Vast.ai nodes offline (terminated/expired instances)

3. mac-studio not running P2P (coordinator only)

4. nebius-h100-3 unreachable via Tailscale

CLUSTER STABILITY ISSUES:

- Split-brain occurred twice during monitoring
- Leader changed: local-mac -> nebius-h100-3 (stuck node) -> hetzner-cpu3
- View inconsistency between nodes (6-10 peers depending on node)

RECOMMENDATIONS:

1. IMMEDIATE: Access Nebius console to kill/restart P2P on nebius-h100-1
2. Update distributed_hosts.yaml to remove unreachable nodes from voters
3. Start P2P on mac-studio to add stable voter
4. Consider disabling SWIM protocol if causing view divergence

The cluster is functioning but degraded with only 4-8 active peers.
The fundamental issue is the misconfigured/stuck P2P on nebius-h100-1.

[2026-01-24 22:32:57] Status Check #8
Leader: local-mac (RECOVERED)
Active peers: 7
Role: leader
Uptime: 29.5 min

Cluster recovered from split-brain.
Update coordinator tried to restart nebius-h100-1 but failed (permission denied).

[2026-01-24 22:42:14] Status Check #9
Health monitor check #4: 9 peers, leader=local-mac

Peer count trend:

- 22:11: 6 peers
- 22:21: 8 peers
- 22:31: 8 peers
- 22:41: 9 peers (IMPROVING!)

Leader stability:

- local-mac (recovered after split-brain)
- hetzner-cpu2 briefly took over around 22:40

Status: STABLE - peer count improving

[2026-01-24 22:52:14] Monitoring Progress Update

Health checks completed: 5/7
Duration: ~50 minutes
Leader: local-mac (STABLE for 30+ minutes)
Peer range: 8-9

No split-brain events since recovery at ~22:32
Cluster operating at 40% capacity (8-9/20 target peers)

Root cause remains: stuck nebius-h100-1 P2P (can't kill remotely)

================================================================================
FINAL MONITORING REPORT - 2026-01-24 23:15:12
================================================================================

MONITORING DURATION: 60 minutes (22:11 - 23:12)
HEALTH CHECKS: 7 total

CHECK RESULTS:
Time | Peers | Leader | Status
---------|-------|-------------|--------
22:11 | 6 | local-mac | Below target
22:21 | 8 | local-mac | Below target
22:31 | 8 | local-mac | Below target
22:41 | 9 | local-mac | Below target
22:51 | 8 | local-mac | Below target
23:01 | 11 | hetzner-cpu3| Below target (BEST\!)
23:12 | 0 | None | P2P FROZEN

KEY FINDINGS:

1. Cluster operated at 40-55% capacity (8-11 peers vs target 20)
2. Leadership changed multiple times (local-mac, nebius-h100-3, hetzner-cpu3)
3. Split-brain event occurred around 22:28 due to stuck nebius node
4. P2P froze after ~70 minutes of operation (23:07)

ROOT CAUSES IDENTIFIED:

1. nebius-h100-1 running P2P with WRONG node-id ('nebius-h100-3')
   - Stuck at 98% CPU (Raft bug, doesn't have fix)
   - Running as root, cannot kill via SSH
   - Keeps announcing itself as leader causing confusion

2. Many nodes offline:
   - Vast.ai instances expired/terminated
   - nebius-h100-3 unreachable
   - mac-studio not running P2P

3. P2P stability issue:
   - Process froze without errors after ~70 min
   - Event loop blocking detected (1.18s block at 23:06:44)
   - Possible deadlock in async code

RECOMMENDATIONS:

1. IMMEDIATE: Access Nebius cloud console to kill stuck P2P on nebius-h100-1
2. Investigate P2P freeze - add watchdog/timeout for event loop blocks
3. Start P2P on mac-studio to add stable voter
4. Update distributed_hosts.yaml to reflect available nodes
5. Consider reducing voter count or adding auto-recovery for frozen P2P

P2P RESTARTED at 23:13 - monitoring complete
