#!/bin/bash
# P2P Stable Configuration for 20+ Node Clusters
#
# January 2026 - P2P Stability Plan Phase 5
#
# This script configures environment variables tuned for stable operation
# of 20+ nodes across multiple cloud providers (Lambda, Vast, RunPod, Nebius,
# Hetzner, Vultr) for several hours at a time.
#
# Target Metrics:
# - Job dispatch latency: <30s
# - Leader failover: <2 minutes
# - Peer visibility: >90%
#
# Usage:
#   ./scripts/start_p2p_stable.sh [additional_args]
#
# Example:
#   ./scripts/start_p2p_stable.sh --verbose
#   ./scripts/start_p2p_stable.sh --node-id my-node

set -euo pipefail

# =============================================================================
# Timeout Configuration (conservative for multi-cloud)
# =============================================================================

# Heartbeat interval (seconds) - how often nodes send heartbeats
# Default: 15s - balanced between responsiveness and network load
export RINGRIFT_P2P_HEARTBEAT_INTERVAL="${RINGRIFT_P2P_HEARTBEAT_INTERVAL:-15}"

# Peer timeout (seconds) - how long before a peer is considered dead
# Default: 120s - allows for transient network issues
export RINGRIFT_P2P_PEER_TIMEOUT="${RINGRIFT_P2P_PEER_TIMEOUT:-120}"

# NAT-blocked peer timeout (seconds) - longer timeout for relay nodes
# Default: 180s - Lambda GH200 nodes use relay, need more tolerance
export RINGRIFT_P2P_PEER_TIMEOUT_NAT_BLOCKED="${RINGRIFT_P2P_PEER_TIMEOUT_NAT_BLOCKED:-180}"

# Suspect timeout (seconds) - how long a peer stays in "suspect" state
# Default: 15s - quick suspect-to-dead transition
export RINGRIFT_P2P_SUSPECT_TIMEOUT="${RINGRIFT_P2P_SUSPECT_TIMEOUT:-15}"

# =============================================================================
# Gossip Configuration (higher fanout for visibility)
# =============================================================================

# Gossip fanout - number of peers to forward gossip to
# Default: 12 - increased for 90%+ peer visibility in 20-40 node clusters
export RINGRIFT_P2P_GOSSIP_FANOUT="${RINGRIFT_P2P_GOSSIP_FANOUT:-12}"

# Leader gossip fanout - leaders use higher fanout for faster propagation
export RINGRIFT_P2P_GOSSIP_FANOUT_LEADER="${RINGRIFT_P2P_GOSSIP_FANOUT_LEADER:-14}"

# Follower gossip fanout
export RINGRIFT_P2P_GOSSIP_FANOUT_FOLLOWER="${RINGRIFT_P2P_GOSSIP_FANOUT_FOLLOWER:-12}"

# Gossip interval (seconds) - how often to run gossip rounds
# Default: 12s - slightly faster than heartbeat for good state convergence
export RINGRIFT_P2P_GOSSIP_INTERVAL="${RINGRIFT_P2P_GOSSIP_INTERVAL:-12}"

# =============================================================================
# Recovery Configuration (aggressive for fast cluster recovery)
# =============================================================================

# Peer recovery interval (seconds) - how often to probe retired peers
# Default: 15s - fast retries for quick recovery
export RINGRIFT_P2P_PEER_RECOVERY_INTERVAL="${RINGRIFT_P2P_PEER_RECOVERY_INTERVAL:-15}"

# Startup grace period (seconds) - time to allow slow state loading
# Default: 90s - gives time for large clusters to stabilize
export RINGRIFT_P2P_STARTUP_GRACE_PERIOD="${RINGRIFT_P2P_STARTUP_GRACE_PERIOD:-90}"

# Burst mode threshold - fraction of retired peers that triggers burst mode
# Default: 0.30 - when 30%+ peers are retired, accelerate recovery
export RINGRIFT_P2P_BURST_MODE_THRESHOLD="${RINGRIFT_P2P_BURST_MODE_THRESHOLD:-0.30}"

# Burst mode max probes - probes per cycle during burst mode
# Default: 50 - increased from normal 20 for faster recovery
export RINGRIFT_P2P_BURST_MODE_MAX_PROBES="${RINGRIFT_P2P_BURST_MODE_MAX_PROBES:-50}"

# Burst mode interval (seconds) - faster interval during burst mode
# Default: 5s - reduced from normal 15s
export RINGRIFT_P2P_BURST_MODE_INTERVAL="${RINGRIFT_P2P_BURST_MODE_INTERVAL:-5}"

# =============================================================================
# Memory Configuration (conservative for long-running clusters)
# =============================================================================

# Maximum gossip peers to track
export RINGRIFT_MAX_GOSSIP_PEERS="${RINGRIFT_MAX_GOSSIP_PEERS:-50}"

# Gossip state TTL (seconds) - how long to keep inactive peer state
# Default: 60s - fast cleanup of stale state
export RINGRIFT_GOSSIP_STATE_TTL="${RINGRIFT_GOSSIP_STATE_TTL:-60}"

# Job states TTL (seconds) - how long to keep completed job info
# Default: 1800s (30 min) - reduced for memory efficiency
export RINGRIFT_JOB_STATES_TTL="${RINGRIFT_JOB_STATES_TTL:-1800}"

# Recovery attempts TTL (seconds) - how long to track recovery history
# Default: 7200s (2 hours) - reduced from 6 hours
export RINGRIFT_RECOVERY_ATTEMPTS_TTL="${RINGRIFT_RECOVERY_ATTEMPTS_TTL:-7200}"

# Connection pool limits
export RINGRIFT_P2P_CONNECTION_LIMIT="${RINGRIFT_P2P_CONNECTION_LIMIT:-200}"

# =============================================================================
# Circuit Breaker Configuration
# =============================================================================

# Circuit breaker decay TTL (seconds) - auto-reset open circuits after this time
# Default: 3600s (1 hour) - prevents permanent node exclusion
export RINGRIFT_CB_DECAY_TTL="${RINGRIFT_CB_DECAY_TTL:-3600}"

# =============================================================================
# Logging
# =============================================================================

echo "=================================================================="
echo "Starting P2P Orchestrator with Stability Configuration"
echo "=================================================================="
echo ""
echo "Timeouts:"
echo "  Heartbeat interval:     ${RINGRIFT_P2P_HEARTBEAT_INTERVAL}s"
echo "  Peer timeout:           ${RINGRIFT_P2P_PEER_TIMEOUT}s"
echo "  NAT-blocked timeout:    ${RINGRIFT_P2P_PEER_TIMEOUT_NAT_BLOCKED}s"
echo "  Startup grace period:   ${RINGRIFT_P2P_STARTUP_GRACE_PERIOD}s"
echo ""
echo "Gossip:"
echo "  Fanout (leader):        ${RINGRIFT_P2P_GOSSIP_FANOUT_LEADER}"
echo "  Fanout (follower):      ${RINGRIFT_P2P_GOSSIP_FANOUT_FOLLOWER}"
echo "  Gossip interval:        ${RINGRIFT_P2P_GOSSIP_INTERVAL}s"
echo ""
echo "Recovery:"
echo "  Recovery interval:      ${RINGRIFT_P2P_PEER_RECOVERY_INTERVAL}s"
echo "  Burst mode threshold:   ${RINGRIFT_P2P_BURST_MODE_THRESHOLD}"
echo "  Burst mode probes:      ${RINGRIFT_P2P_BURST_MODE_MAX_PROBES}"
echo ""
echo "Memory:"
echo "  Gossip state TTL:       ${RINGRIFT_GOSSIP_STATE_TTL}s"
echo "  Job states TTL:         ${RINGRIFT_JOB_STATES_TTL}s"
echo "  Connection limit:       ${RINGRIFT_P2P_CONNECTION_LIMIT}"
echo ""
echo "=================================================================="
echo ""

# Change to ai-service directory
cd "$(dirname "$0")/.." || exit 1

# Ensure PYTHONPATH includes the ai-service directory
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Start P2P orchestrator with any additional arguments
exec python scripts/p2p_orchestrator.py "$@"
