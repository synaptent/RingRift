"""P2P Orchestrator Data Models.

This module contains dataclasses used throughout the P2P orchestrator.
Extracted from p2p_orchestrator.py for better modularity.
"""

from __future__ import annotations

import os
import time
from dataclasses import (
    asdict,
    dataclass,
    field,
    fields as dataclass_fields,
)
from typing import Any, Optional

from .constants import (
    DISK_WARNING_THRESHOLD,
    GPU_POWER_RANKINGS,
    LOAD_AVERAGE_MAX_MULTIPLIER,
    LOAD_MAX_FOR_NEW_JOBS,
    MAX_CONSECUTIVE_FAILURES,
    MEMORY_WARNING_THRESHOLD,
    PEER_TIMEOUT,
    RETRY_DEAD_NODE_INTERVAL,
    RETRY_RETIRED_NODE_INTERVAL,
    SUSPECT_TIMEOUT,
    get_cpu_adaptive_timeout,  # Jan 20, 2026: CPU-adaptive timeouts for busy nodes
)
from .types import JobType, NodeHealthState, NodeRole

# Jan 23, 2026: Phase 3 - Provider-aware timeouts for peer death detection
try:
    from app.config.provider_timeouts import ProviderTimeouts
except ImportError:
    ProviderTimeouts = None  # Graceful degradation if not available


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    host: str
    port: int
    scheme: str = "http"  # How to reach this node (http/https)
    role: NodeRole = NodeRole.FOLLOWER
    last_heartbeat: float = 0.0
    # Leader tracking: who this node believes is the current cluster leader
    leader_id: str = ""  # Dec 2025: Added for leader propagation in heartbeats
    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    selfplay_jobs: int = 0
    # Jan 2, 2026: Maximum selfplay slots for this node based on GPU capability
    # Used by WorkerPullLoop for slot-based capacity management
    max_selfplay_slots: int = 8
    training_jobs: int = 0
    has_gpu: bool = False
    gpu_name: str = ""
    memory_gb: int = 0
    capabilities: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    # Self-reported endpoint (may be different from the observed reachable
    # endpoint stored in `host`/`port`, e.g. overlays or containers).
    reported_host: str = ""
    reported_port: int = 0
    # LEARNED LESSONS - Track connection failures for adaptive retry
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    # LEARNED LESSONS - Track resource issues
    disk_cleanup_needed: bool = False
    oom_events: int = 0
    last_oom_time: float = 0.0
    # NAT/Relay support - nodes that can't be reached directly
    nat_blocked: bool = False
    nat_blocked_since: float = 0.0  # timestamp when nat_blocked was set (for recovery)
    last_nat_probe: float = 0.0     # timestamp of last NAT recovery probe attempt
    relay_via: str = ""  # node_id of the relay hub (usually leader)
    # Peer lifecycle: permanently gone nodes are marked retired so they don't
    # pollute scheduling, but they can be reactivated if they come back online.
    retired: bool = False
    retired_at: float = 0.0
    # Dec 29, 2025: Track alternate IPs for the same node (Tailscale, public, etc.)
    # This enables peer deduplication - multiple IPs map to single canonical entry
    alternate_ips: set[str] = field(default_factory=set)
    # Jan 13, 2026: Multi-address advertisement - all addresses this node is reachable at
    # Nodes advertise these in heartbeats so peers can try alternate paths if primary fails
    # Format: List of IPs/hostnames sorted by preference (first = primary)
    addresses: list[str] = field(default_factory=list)
    # Jan 13, 2026: Tailscale IP for direct VPN mesh connectivity
    tailscale_ip: str = ""
    # External work detection - work running outside P2P orchestrator tracking
    cmaes_running: bool = False
    gauntlet_running: bool = False
    tournament_running: bool = False
    data_merge_running: bool = False
    # Phase 6: Health broadcasting - comprehensive health metrics for leader aggregation
    nfs_accessible: bool = True
    code_version: str = ""  # Git commit hash for version mismatch detection
    errors_last_hour: int = 0  # Error count for anomaly detection
    disk_free_gb: float = 0.0  # Absolute free disk space
    active_job_count: int = 0  # Total active jobs (selfplay + training + external)
    # Dec 30, 2025: Endpoint validation tracking for stale IP detection
    # Prevents partition isolation from stale private IPs persisting after network changes
    endpoint_last_validated: float = 0.0  # Unix timestamp of last successful heartbeat
    # Dec 30, 2025: Reduced from 300s to 60s for faster stale IP recovery
    endpoint_ttl_seconds: int = 60  # 1 min TTL before endpoint considered stale
    # Jan 7, 2026: GPU job tracking for better dispatch decisions
    # Enables detecting GPU driver issues and tracking GPU job success/failure rates
    gpu_job_count: int = 0           # Currently running GPU jobs on this node
    cpu_job_count: int = 0           # Currently running CPU jobs on this node
    last_gpu_job_success: float = 0.0  # Unix timestamp of last successful GPU job
    last_gpu_job_failure: float = 0.0  # Unix timestamp of last failed GPU job
    gpu_failure_count: int = 0       # Consecutive GPU job failures (reset on success)
    # Jan 13, 2026: Node status from distributed_hosts.yaml (e.g., "proxy_only", "active")
    # Used by _is_leader_eligible() to reject proxy nodes from becoming cluster leaders
    status: str = ""
    # Jan 24, 2026: Number of visible peers for connectivity scoring
    # Used by _compute_connectivity_score() to determine leader eligibility
    visible_peers: int = 0
    # Jan 25, 2026: Broadcast effective timeout so all nodes agree on when to mark us dead
    # Computed from PEER_TIMEOUT * CPU_multiplier * Provider_multiplier
    effective_timeout: float = 0.0

    def get_health_state(self) -> NodeHealthState:
        """Get detailed health state based on heartbeat timing.

        Dec 2025: Added SUSPECT state for grace period handling.
        This reduces false-positive failures from transient network issues.

        Jan 20, 2026: Integrated CPU-adaptive timeouts. Nodes under high CPU load
        (selfplay, training) respond slowly to heartbeats. Using the peer's reported
        cpu_percent, we extend the timeout proportionally:
        - base_timeout if CPU < 80%
        - base_timeout * 1.5 if CPU >= 80%
        - base_timeout * 2.0 if CPU >= 95%
        This prevents marking busy nodes as dead due to slow heartbeat responses.

        Jan 23, 2026: Phase 3 - Integrated provider-specific timeouts. Different
        providers have different network characteristics (NAT, latency, reliability).
        Provider multipliers are applied after CPU-adaptive adjustment:
        - Vast.ai: 2.5x (consumer networks, NAT issues)
        - Lambda: 2.0x (NAT-blocked GH200s)
        - Nebius/Vultr: 1.5x (cloud infrastructure)
        - Hetzner: 1.0x (bare metal)

        Returns:
            NodeHealthState.ALIVE if heartbeat within SUSPECT_TIMEOUT
            NodeHealthState.SUSPECT if heartbeat between SUSPECT_TIMEOUT and adaptive timeout
            NodeHealthState.DEAD if no heartbeat for provider-adjusted timeout
        """
        elapsed = time.time() - self.last_heartbeat

        # Jan 25, 2026: Use peer's broadcast effective_timeout if available
        # This ensures all nodes agree on when this peer should be marked dead
        # The peer computes its timeout based on its own CPU load and provider
        if self.effective_timeout > 0:
            adaptive_timeout = self.effective_timeout
        else:
            # Fallback to local calculation for backward compatibility
            # Jan 20, 2026: Use CPU-adaptive timeout based on peer's reported CPU load
            # If peer reports high CPU, give it more time before marking dead
            cpu_load = self.cpu_percent / 100.0 if self.cpu_percent > 0 else None
            adaptive_timeout = get_cpu_adaptive_timeout(PEER_TIMEOUT, cpu_load)

            # Jan 23, 2026: Apply provider-specific timeout multiplier
            # This prevents false-positive deaths for nodes with slower networks
            if ProviderTimeouts is not None:
                provider_mult = ProviderTimeouts.get_multiplier(self.node_id)
                adaptive_timeout = adaptive_timeout * provider_mult

        if elapsed < SUSPECT_TIMEOUT:
            return NodeHealthState.ALIVE
        elif elapsed < adaptive_timeout:
            return NodeHealthState.SUSPECT
        return NodeHealthState.DEAD

    def is_alive(self) -> bool:
        """Check if node is considered alive based on last heartbeat.

        Note: SUSPECT nodes are still considered alive for job execution
        but may be treated differently for leader election.
        """
        return self.get_health_state() != NodeHealthState.DEAD

    def is_suspect(self) -> bool:
        """Check if node is in SUSPECT state (grace period)."""
        return self.get_health_state() == NodeHealthState.SUSPECT

    def is_available(self) -> bool:
        """Check if node is available for new work (ALIVE only, not SUSPECT).

        Use this for job scheduling decisions where SUSPECT nodes should be skipped.
        Use is_alive() for membership/heartbeat checks where SUSPECT is acceptable.

        Dec 2025: Added to standardize health checks across codebase.
        Previously, some code used get_health_state() == ALIVE (strict) while
        most used is_alive() (lenient), causing nodes to appear/disappear.
        """
        return self.get_health_state() == NodeHealthState.ALIVE

    def is_healthy(self) -> bool:
        """Check if node is healthy for new jobs (not just reachable)."""
        if not self.is_alive():
            return False
        if getattr(self, "retired", False):
            return False
        # LEARNED LESSONS - Don't start jobs on resource-constrained nodes
        if self.disk_percent >= DISK_WARNING_THRESHOLD:
            return False
        if self.memory_percent >= MEMORY_WARNING_THRESHOLD:
            return False
        return not self.get_load_score() >= LOAD_MAX_FOR_NEW_JOBS

    def is_endpoint_stale(self) -> bool:
        """Check if this node's endpoint needs revalidation.

        Dec 30, 2025: Added for P2P partition recovery. When a node's endpoint
        hasn't been validated for endpoint_ttl_seconds, we should probe alternate
        IPs (Tailscale, public IP) to check if the primary IP is stale.

        A stale endpoint may be:
        - A private IP from a previous network that's no longer routable
        - An IP that changed after container restart or VPN reconnection

        Returns:
            True if endpoint_last_validated is older than endpoint_ttl_seconds
        """
        if self.endpoint_last_validated <= 0:
            return True  # Never validated
        return time.time() - self.endpoint_last_validated > self.endpoint_ttl_seconds

    def mark_endpoint_validated(self) -> None:
        """Mark this node's endpoint as freshly validated.

        Called after successful heartbeat or gossip exchange.
        """
        self.endpoint_last_validated = time.time()

    def get_load_score(self) -> float:
        """Calculate a load score for load balancing (lower = less loaded).

        LEARNED LESSONS - Smart load balancing:
        - Weights CPU, memory, and job counts for overall load estimation
        - GPU utilization is weighted separately for GPU nodes
        - Returns 0-100 scale where 0 = idle, 100 = overloaded
        """
        # Base score from CPU and memory utilization
        cpu_weight = 0.4
        mem_weight = 0.3
        jobs_weight = 0.3

        # Normalize job count (assume 8 selfplay jobs = fully loaded)
        job_score = min(100.0, (self.selfplay_jobs + self.training_jobs * 2) * 12.5)

        load = (
            self.cpu_percent * cpu_weight +
            self.memory_percent * mem_weight +
            job_score * jobs_weight
        )

        # For GPU nodes, also consider GPU utilization
        if self.has_gpu and self.gpu_percent > 0:
            load = load * 0.7 + self.gpu_percent * 0.3

        return min(100.0, load)

    def is_gpu_node(self) -> bool:
        """Check if this node has a CUDA GPU (not Apple MPS)."""
        gpu_name = (self.gpu_name or "").upper()
        return self.has_gpu and "MPS" not in gpu_name and "APPLE" not in gpu_name

    def is_cpu_only_node(self) -> bool:
        """Check if this node is CPU-only (no accelerator)."""
        return not self.has_gpu

    def has_cuda_gpu(self) -> bool:
        """Check if this node has a CUDA-capable GPU.

        This is the authoritative method for determining if a node can run
        GPU-required engine modes (gumbel-mcts, mcts, nnue-guided, etc.).

        Returns:
            True if node has CUDA GPU, False for CPU-only or MPS-only nodes.
        """
        return self.is_gpu_node()

    def can_run_engine_mode(self, mode: str) -> bool:
        """Check if this node can run the given engine mode.

        Uses the GPU requirement metadata from selfplay_config to determine
        if the engine mode requires GPU and if this node has GPU capability.

        Args:
            mode: Engine mode string (e.g., "gumbel-mcts", "heuristic-only")

        Returns:
            True if this node can run the engine mode

        Example:
            >>> node = NodeInfo(node_id="hetzner-cpu1", has_gpu=False)
            >>> node.can_run_engine_mode("heuristic-only")
            True
            >>> node.can_run_engine_mode("gumbel-mcts")
            False

            >>> node = NodeInfo(node_id="lambda-gh200-1", has_gpu=True, gpu_name="GH200")
            >>> node.can_run_engine_mode("gumbel-mcts")
            True
        """
        try:
            from app.training.selfplay_config import engine_mode_requires_gpu
            if engine_mode_requires_gpu(mode):
                return self.has_cuda_gpu()
            return True  # CPU-compatible modes run anywhere
        except ImportError:
            # If selfplay_config not available, be conservative
            # Assume GPU modes need GPU
            gpu_modes = {"gumbel-mcts", "mcts", "nnue-guided", "policy-only",
                         "nn-minimax", "nn-descent", "gnn", "hybrid"}
            if mode.lower() in gpu_modes:
                return self.has_cuda_gpu()
            return True

    def has_external_work(self) -> bool:
        """Check if any external (untracked) work is running."""
        return (self.cmaes_running or self.gauntlet_running or
                self.tournament_running or self.data_merge_running)

    def is_misrouted(self) -> bool:
        """Check if this GPU node is running CPU-bound work with idle GPU.

        A node is misrouted if:
        - It has a GPU (is_gpu_node)
        - GPU utilization is low (<30%)
        - CPU utilization is high (>70%)
        - Running CPU-bound external work (CMA-ES, gauntlet, tournament)
        """
        if not self.is_gpu_node():
            return False
        if self.gpu_percent >= 30:
            return False
        if self.cpu_percent < 70:
            return False
        # Check for CPU-bound external work
        return self.cmaes_running or self.gauntlet_running or self.tournament_running

    def get_health_issues(self) -> list[tuple[str, str]]:
        """Get list of health issues for this node.

        Returns:
            List of (issue_code, description) tuples for any detected issues.
            Empty list means node is healthy.
        """
        issues = []

        # Check disk
        if self.disk_percent >= 95:
            issues.append(("disk_critical", f"Disk at {self.disk_percent:.0f}%"))
        elif self.disk_percent >= DISK_WARNING_THRESHOLD:
            issues.append(("disk_warning", f"Disk at {self.disk_percent:.0f}%"))

        # Check memory
        if self.memory_percent >= 95:
            issues.append(("memory_critical", f"Memory at {self.memory_percent:.0f}%"))
        elif self.memory_percent >= MEMORY_WARNING_THRESHOLD:
            issues.append(("memory_warning", f"Memory at {self.memory_percent:.0f}%"))

        # Check NFS (skip if NFS is disabled - RingRift doesn't use NFS)
        nfs_disabled = os.environ.get("RINGRIFT_DISABLE_NFS_CHECK", "1").lower() in ("1", "true", "yes")
        if not nfs_disabled and not self.nfs_accessible:
            issues.append(("nfs_unavailable", "NFS mount not accessible"))

        # Check errors
        if self.errors_last_hour >= 50:
            issues.append(("high_errors", f"{self.errors_last_hour} errors in last hour"))
        elif self.errors_last_hour >= 10:
            issues.append(("moderate_errors", f"{self.errors_last_hour} errors in last hour"))

        # Check OOM
        if self.oom_events > 0:
            issues.append(("oom_events", f"{self.oom_events} OOM events"))

        # Check if retired
        if self.retired:
            issues.append(("retired", "Node is retired"))

        # Check connection failures
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            issues.append(("unreachable", f"{self.consecutive_failures} consecutive failures"))

        return issues

    def should_retry(self) -> bool:
        """Check if we should retry connecting to a failed node."""
        if getattr(self, "retired", False):
            return time.time() - self.last_failure_time > RETRY_RETIRED_NODE_INTERVAL
        if self.consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            return True
        # LEARNED LESSONS - Retry dead nodes periodically
        return time.time() - self.last_failure_time > RETRY_DEAD_NODE_INTERVAL

    def gpu_power_score(self) -> int:
        """Get GPU processing power score based on GPU model.

        Higher score = more powerful GPU = better for training.
        Used for prioritizing which nodes receive selfplay data first.
        """
        if not self.has_gpu or not self.gpu_name:
            return 0

        gpu_upper = self.gpu_name.upper()

        # Check each GPU model pattern
        for gpu_key, score in GPU_POWER_RANKINGS.items():
            if gpu_key.upper() in gpu_upper:
                # Multi-GPU bonus: assume more memory = more GPUs
                # H100 80GB = 1x, 160GB = 2x, etc.
                if self.memory_gb and self.memory_gb > 100:
                    # Likely multi-GPU system
                    gpu_count = max(1, self.memory_gb // 80)
                    return score * gpu_count
                return score

        return GPU_POWER_RANKINGS.get("Unknown", 10)

    def cpu_power_score(self) -> int:
        """Get CPU processing power score based on core count and node type.

        Higher score = more CPU power = better for CPU-intensive tasks like
        NPZ export, data aggregation, and CMA-ES.

        Vast nodes typically have very high CPU counts (256-512 cores) and are
        ideal for CPU-intensive work, freeing GPU nodes for training/selfplay.
        """
        if not self.cpu_count or self.cpu_count <= 0:
            return 0

        # Base score is simply the CPU count
        score = self.cpu_count

        # Bonus for vast nodes (they're optimized for high CPU throughput)
        # Lambda nodes have high GPU power but limited CPU relative to vast
        node_id_lower = self.node_id.lower()
        if "vast" in node_id_lower:
            # Vast nodes get 50% bonus - they're ideal for CPU tasks
            score = int(score * 1.5)
        elif "lambda" in node_id_lower:
            # Lambda nodes should prioritize GPU work, not CPU-intensive tasks
            # Reduce their CPU score to prefer vast nodes for CPU work
            score = int(score * 0.5)
        elif "aws" in node_id_lower and "cpu" in node_id_lower:
            # AWS CPU-optimized instances get a small bonus
            score = int(score * 1.2)

        return score

    def check_load_average_safe(self) -> tuple[bool, str]:
        """Check if system load average is safe for spawning new processes.

        SAFEGUARD: Uses actual os.getloadavg() instead of just CPU percentage.
        On a 72-core machine, load 500 is catastrophic even if CPU% looks OK.

        Returns:
            (is_safe, reason) - True if safe to spawn, with explanation
        """
        try:
            load_1, load_5, _load_15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            max_load = cpu_count * LOAD_AVERAGE_MAX_MULTIPLIER

            if load_1 > max_load:
                return False, f"Load {load_1:.1f} > {max_load:.0f} (cpus={cpu_count} x {LOAD_AVERAGE_MAX_MULTIPLIER})"
            if load_5 > max_load * 1.5:
                return False, f"5min load {load_5:.1f} > {max_load * 1.5:.0f}"

            return True, f"Load OK: {load_1:.1f}/{max_load:.0f}"
        except Exception as e:
            # If we can't check load, be conservative
            return True, f"Load check unavailable: {e}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['role'] = self.role.value
        # Dec 29, 2025: Convert set to list for JSON serialization
        d['alternate_ips'] = list(self.alternate_ips) if self.alternate_ips else []
        # Jan 13, 2026: Multi-address advertisement
        d['addresses'] = list(self.addresses) if self.addresses else []
        d['tailscale_ip'] = self.tailscale_ip or ""
        # Derived metrics (not persisted as dataclass fields).
        d["load_score"] = self.get_load_score()
        d["gpu_power_score"] = self.gpu_power_score()
        d["cpu_power_score"] = self.cpu_power_score()
        d["is_cpu_only_node"] = self.is_cpu_only_node()
        d["is_cuda_gpu_node"] = self.is_gpu_node()
        d["has_external_work"] = self.has_external_work()
        d["is_misrouted"] = self.is_misrouted()
        return d

    def merge_from(self, other: "NodeInfo") -> None:
        """Merge information from another NodeInfo representing the same node.

        Dec 29, 2025: Used for peer deduplication - when we discover the same
        node via multiple IPs, merge them into a single canonical entry.

        Args:
            other: Another NodeInfo for the same node_id
        """
        if other.node_id != self.node_id:
            return  # Safety: only merge if same node

        # Collect all known IPs for this node
        all_ips = set()
        if self.host:
            all_ips.add(self.host)
        if self.reported_host:
            all_ips.add(self.reported_host)
        if other.host:
            all_ips.add(other.host)
        if other.reported_host:
            all_ips.add(other.reported_host)
        all_ips.update(self.alternate_ips)
        all_ips.update(other.alternate_ips)
        # Remove empty strings
        all_ips.discard("")

        # Pick the most recently updated node as primary
        if other.last_heartbeat > self.last_heartbeat:
            # Other is more recent - update our primary host
            self.host = other.host
            self.port = other.port
            self.scheme = other.scheme
            self.last_heartbeat = other.last_heartbeat
            # Copy resource metrics from more recent source
            self.cpu_percent = other.cpu_percent
            self.memory_percent = other.memory_percent
            self.disk_percent = other.disk_percent
            self.gpu_percent = other.gpu_percent
            self.gpu_memory_percent = other.gpu_memory_percent
            self.selfplay_jobs = other.selfplay_jobs
            self.training_jobs = other.training_jobs
            self.visible_peers = other.visible_peers

        # Remove primary host from alternate_ips to avoid duplication
        all_ips.discard(self.host)
        self.alternate_ips = all_ips

        # Merge capabilities (union)
        if other.capabilities:
            caps = set(self.capabilities)
            caps.update(other.capabilities)
            self.capabilities = list(caps)

        # Clear retired status if other is alive
        if other.is_alive() and not other.retired:
            self.retired = False
            self.retired_at = 0.0
            self.consecutive_failures = 0

    @classmethod
    def from_dict(cls, d: dict) -> NodeInfo:
        """Create from dictionary."""
        d = d.copy()
        d['role'] = NodeRole(d.get('role', 'follower'))
        # Handle missing new fields gracefully
        d.setdefault('scheme', 'http')
        d.setdefault('leader_id', '')  # Dec 2025: Leader propagation field
        d.setdefault('cpu_count', 0)
        d.setdefault('reported_host', '')
        d.setdefault('reported_port', 0)
        d.setdefault('consecutive_failures', 0)
        d.setdefault('last_failure_time', 0.0)
        d.setdefault('disk_cleanup_needed', False)
        d.setdefault('oom_events', 0)
        d.setdefault('last_oom_time', 0.0)
        d.setdefault('nat_blocked', False)
        d.setdefault('nat_blocked_since', 0.0)
        d.setdefault('last_nat_probe', 0.0)
        d.setdefault('relay_via', '')
        d.setdefault('retired', False)
        d.setdefault('retired_at', 0.0)
        # Dec 29, 2025: Handle alternate_ips (stored as list in JSON, convert to set)
        alt_ips = d.get('alternate_ips', [])
        d['alternate_ips'] = set(alt_ips) if isinstance(alt_ips, (list, set)) else set()
        # Jan 13, 2026: Multi-address advertisement
        addresses = d.get('addresses', [])
        d['addresses'] = list(addresses) if isinstance(addresses, (list, tuple)) else []
        d.setdefault('tailscale_ip', '')
        # External work detection fields
        d.setdefault('cmaes_running', False)
        d.setdefault('gauntlet_running', False)
        d.setdefault('tournament_running', False)
        d.setdefault('data_merge_running', False)
        # Phase 6: Health broadcasting fields
        d.setdefault('nfs_accessible', True)
        d.setdefault('code_version', '')
        d.setdefault('errors_last_hour', 0)
        d.setdefault('disk_free_gb', 0.0)
        d.setdefault('active_job_count', 0)
        # Dec 30, 2025: Endpoint validation fields
        d.setdefault('endpoint_last_validated', 0.0)
        d.setdefault('endpoint_ttl_seconds', 60)  # Match dataclass default (reduced Dec 30, 2025)
        # Dec 30, 2025: Capabilities field - critical for job scheduling
        # Without this, nodes may have empty capabilities and get skipped by schedulers
        d.setdefault('capabilities', [])
        # Jan 24, 2026: Visible peers count for connectivity scoring
        d.setdefault('visible_peers', 0)

        # Dec 30, 2025: Capability inference - if capabilities empty, infer from has_gpu
        # This ensures nodes get work even if they didn't explicitly advertise capabilities
        if not d.get('capabilities'):
            if d.get('has_gpu'):
                # GPU nodes can do selfplay, training, cmaes, gauntlet, and tournament
                d['capabilities'] = ['selfplay', 'training', 'cmaes', 'gauntlet', 'tournament']
            else:
                # CPU nodes can at least do selfplay (heuristic-based)
                d['capabilities'] = ['selfplay']

        # Ignore unknown keys for rolling upgrades.
        allowed = {f.name for f in dataclass_fields(cls)}
        d = {k: v for k, v in d.items() if k in allowed}
        return cls(**d)


@dataclass
class ClusterJob:
    """A job running in the cluster."""
    job_id: str
    job_type: JobType
    node_id: str
    board_type: str = "square8"
    num_players: int = 2
    engine_mode: str = "gumbel-mcts"  # Jan 2026: Default to high-quality GPU mode
    pid: int = 0
    started_at: float = 0.0
    status: str = "running"
    error_message: str = ""
    # Extended fields for distributed jobs
    coordinator_node: str = ""  # Node running coordinator (for worker jobs)
    worker_port: int = 8766     # Port for worker server
    config_json: str = ""       # JSON config for complex jobs

    def to_dict(self) -> dict:
        d = asdict(self)
        d['job_type'] = self.job_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ClusterJob:
        d = d.copy()
        d['job_type'] = JobType(d.get('job_type', 'selfplay'))
        # Handle missing new fields
        d.setdefault('coordinator_node', '')
        d.setdefault('worker_port', 8766)
        d.setdefault('config_json', '')
        d.setdefault('error_message', '')
        return cls(**d)


@dataclass
class DistributedCMAESState:
    """State for distributed CMA-ES job coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    generations: int = 100
    population_size: int = 20
    games_per_eval: int = 50
    current_generation: int = 0
    best_fitness: float = 0.0
    best_weights: dict[str, float] = field(default_factory=dict)
    worker_nodes: list[str] = field(default_factory=list)
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0
    results_file: str = ""
    # LEARNED LESSONS - Store pending results keyed by (generation, individual_idx)
    pending_results: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DistributedCMAESState:
        d.setdefault('pending_results', {})
        return cls(**d)


@dataclass
class DistributedTournamentState:
    """State for distributed tournament coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    agent_ids: list[str] = field(default_factory=list)
    games_per_pairing: int = 2
    total_matches: int = 0
    completed_matches: int = 0
    worker_nodes: list[str] = field(default_factory=list)
    pending_matches: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    final_ratings: dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DistributedTournamentState:
        return cls(**d)


@dataclass
class SSHTournamentRun:
    """State for an SSH-distributed difficulty-tier tournament (leader-triggered)."""

    job_id: str
    run_id: str
    tiers: str
    board: str
    games_per_matchup: int
    pid: int = 0
    status: str = "running"  # running, completed, failed, cancelled
    started_at: float = 0.0
    completed_at: float = 0.0
    output_root: str = ""
    manifest_path: str = ""
    checkpoint_path: str = ""
    report_path: str = ""
    log_path: str = ""
    command: list[str] = field(default_factory=list)
    return_code: int | None = None
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImprovementLoopState:
    """State for improvement loop coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    current_iteration: int = 0
    max_iterations: int = 50
    games_per_iteration: int = 1000
    phase: str = "idle"  # idle, selfplay, export, train, evaluate, promote
    best_model_path: str = ""
    best_winrate: float = 0.0
    consecutive_failures: int = 0
    worker_nodes: list[str] = field(default_factory=list)
    selfplay_progress: dict[str, int] = field(default_factory=dict)  # node_id -> games done
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ImprovementLoopState:
        return cls(**d)


# ============================================
# Phase 3: Training Pipeline Integration Types
# ============================================

@dataclass
class TrainingJob:
    """State for automatic training jobs dispatched by leader."""
    job_id: str
    job_type: str  # "nnue", "cmaes"
    board_type: str
    num_players: int
    status: str = "pending"  # pending, queued, running, completed, failed
    worker_node: str = ""    # Node where training is running
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    # Training configuration
    epochs: int = 100
    batch_size: int = 4096  # Increased for GH200/H100 GPUs
    learning_rate: float = 0.001
    # Data sources
    data_paths: list[str] = field(default_factory=list)
    data_games_count: int = 0
    # Output
    output_model_path: str = ""
    error_message: str = ""
    # Metrics
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    # TRAINING CHECKPOINTING: Track checkpoint state for resume capability
    checkpoint_path: str = ""        # Path to latest checkpoint
    checkpoint_epoch: int = 0        # Epoch of latest checkpoint
    checkpoint_loss: float = 0.0     # Loss at checkpoint
    checkpoint_updated_at: float = 0.0  # When checkpoint was last saved
    resume_from_checkpoint: bool = False  # Whether this job resumed from checkpoint

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainingJob:
        return cls(**d)


@dataclass
class TrainingThresholds:
    """Configuration for automatic training triggers with adaptive scaling."""
    # Minimum games required to trigger training
    min_games_nnue: int = 2000          # Start sooner; improves as more data arrives
    min_games_cmaes: int = 1000         # CMA-ES can work with fewer games
    # Incremental thresholds (trigger re-training when new data >= threshold)
    incremental_games_nnue: int = 1000  # Re-train every 1k new games
    incremental_games_cmaes: int = 500  # Re-optimize every 500 new games
    # Cooldown between training runs (seconds)
    cooldown_seconds: float = 1800.0    # 30 minutes
    # Auto-training enabled flags
    auto_nnue_enabled: bool = True
    auto_cmaes_enabled: bool = True
    # Adaptive scaling factors (updated dynamically based on cluster state)
    cluster_scale_factor: float = 1.0   # Adjusted based on GPU node count
    data_rate_scale_factor: float = 1.0 # Adjusted based on games/hour

    def get_effective_min_games(self, job_type: str = "nnue") -> int:
        """Get cluster-scaled minimum games threshold.

        More GPU nodes = can train more frequently with less data per iteration.
        """
        base = self.min_games_nnue if job_type == "nnue" else self.min_games_cmaes
        # Scale inversely with cluster size (more nodes = lower threshold)
        scaled = int(base * self.cluster_scale_factor)
        # Never go below minimum viable threshold
        return max(500 if job_type == "nnue" else 250, scaled)

    def get_effective_incremental(self, job_type: str = "nnue") -> int:
        """Get cluster-scaled incremental threshold."""
        base = self.incremental_games_nnue if job_type == "nnue" else self.incremental_games_cmaes
        scaled = int(base * self.cluster_scale_factor)
        return max(200 if job_type == "nnue" else 100, scaled)

    def get_effective_cooldown(self) -> float:
        """Get data-rate-scaled cooldown.

        Faster data generation = shorter cooldowns (train more often).
        """
        # Scale inversely with data rate (faster data = shorter cooldown)
        scaled = self.cooldown_seconds / max(1.0, self.data_rate_scale_factor)
        # Never go below 5 minutes or above 2 hours
        return max(300, min(7200, scaled))

    def update_from_cluster_state(self, gpu_node_count: int, games_per_hour: float = 0):
        """Update adaptive scaling factors based on cluster state.

        Args:
            gpu_node_count: Number of GPU nodes in cluster
            games_per_hour: Current data generation rate (0 = unknown)
        """
        # Scale factor: more GPUs = lower thresholds (train more often)
        # 1 GPU = 1.0, 2 GPUs = 0.7, 4 GPUs = 0.5, 8+ GPUs = 0.35
        if gpu_node_count <= 1:
            self.cluster_scale_factor = 1.0
        elif gpu_node_count <= 2:
            self.cluster_scale_factor = 0.7
        elif gpu_node_count <= 4:
            self.cluster_scale_factor = 0.5
        else:
            self.cluster_scale_factor = 0.35

        # Data rate factor: faster generation = shorter cooldowns
        if games_per_hour > 0:
            # Baseline: 1000 games/hour = factor 1.0
            self.data_rate_scale_factor = max(0.5, min(4.0, games_per_hour / 1000))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainingThresholds:
        return cls(**d)


# ============================================
# Phase 2: Distributed Data Sync Types
# ============================================

@dataclass
class DataFileInfo:
    """Information about a single data file for manifest collection."""
    path: str                    # Relative path from ai-service/data
    size_bytes: int              # File size in bytes
    modified_time: float         # Last modification time (Unix timestamp)
    file_hash: str = ""          # MD5 hash for verification (computed on demand)
    file_type: str = ""          # Type: selfplay, model, training, etc.
    board_type: str = ""         # Board type if applicable (square8, hex, etc.)
    num_players: int = 0         # Player count if applicable
    game_count: int = 0          # For selfplay JSONL: number of games (lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DataFileInfo:
        return cls(**d)


@dataclass
class NodeDataManifest:
    """Data manifest for a single node - lists all available data files."""
    node_id: str
    collected_at: float          # When this manifest was collected
    total_files: int = 0
    total_size_bytes: int = 0
    files: list[DataFileInfo] = field(default_factory=list)
    # Summary by type
    selfplay_games: int = 0      # Total selfplay games (from JSONL line counts)
    model_count: int = 0         # Number of model files
    training_data_size: int = 0  # Total training data size

    @property
    def files_by_path(self) -> dict[str, DataFileInfo]:
        return {f.path: f for f in self.files}

    def to_dict(self) -> dict:
        d = asdict(self)
        d['files'] = [f.to_dict() if hasattr(f, 'to_dict') else f for f in self.files]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NodeDataManifest:
        d = d.copy()
        d['files'] = [DataFileInfo.from_dict(f) if isinstance(f, dict) else f for f in d.get('files', [])]
        return cls(**d)


@dataclass
class ClusterDataManifest:
    """Aggregated data manifest for the entire cluster (leader-only)."""
    collected_at: float
    node_manifests: dict[str, NodeDataManifest] = field(default_factory=dict)
    # Cluster-wide totals
    total_nodes: int = 0
    total_files: int = 0
    total_size_bytes: int = 0
    total_selfplay_games: int = 0
    # Data distribution analysis
    files_by_node: dict[str, int] = field(default_factory=dict)
    unique_files: set[str] = field(default_factory=set)
    missing_from_nodes: dict[str, list[str]] = field(default_factory=dict)  # file -> list of nodes missing it
    # External storage metadata (OWC drive, S3 bucket) - Jan 2026
    external_storage: ExternalStorageManifest | None = None

    @property
    def manifests_by_node(self) -> dict[str, NodeDataManifest]:
        return self.node_manifests

    @property
    def by_board_type(self) -> dict[str, dict[str, Any]]:
        """Aggregate selfplay game counts by (board_type, num_players).

        Key format matches downstream training logic: `{board_type}_{num_players}p`.
        """
        totals: dict[str, dict[str, Any]] = {}

        for node_id, node_manifest in self.node_manifests.items():
            for f in node_manifest.files:
                if getattr(f, "file_type", "") != "selfplay":
                    continue
                if not getattr(f, "board_type", "") or not getattr(f, "num_players", 0):
                    continue
                key = f"{f.board_type}_{f.num_players}p"
                entry = totals.setdefault(key, {"total_games": 0, "nodes": set()})
                entry["total_games"] += int(getattr(f, "game_count", 0) or 0)
                entry["nodes"].add(node_id)

        return {k: {"total_games": v["total_games"], "nodes": sorted(v["nodes"])} for k, v in totals.items()}

    def to_dict(self) -> dict:
        d = {
            'collected_at': self.collected_at,
            'node_manifests': {k: v.to_dict() for k, v in self.node_manifests.items()},
            'total_nodes': self.total_nodes,
            'total_files': self.total_files,
            'total_size_bytes': self.total_size_bytes,
            'total_selfplay_games': self.total_selfplay_games,
            'files_by_node': self.files_by_node,
            'unique_files': list(self.unique_files),
            'missing_from_nodes': self.missing_from_nodes,
            'external_storage': self.external_storage.to_dict() if self.external_storage else None,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ClusterDataManifest:
        d = d.copy()
        d['node_manifests'] = {k: NodeDataManifest.from_dict(v) for k, v in d.get('node_manifests', {}).items()}
        d['unique_files'] = set(d.get('unique_files', []))
        # Handle external_storage if present
        if 'external_storage' in d and d['external_storage']:
            d['external_storage'] = ExternalStorageManifest.from_dict(d['external_storage'])
        return cls(**d)


@dataclass
class ExternalStorageManifest:
    """Metadata about external storage sources (OWC drive, S3 bucket).

    Jan 2026: Added for unified cluster data visibility.
    """
    collected_at: float = 0.0

    # OWC external drive (mac-studio)
    owc_available: bool = False
    owc_games_by_config: dict[str, int] = field(default_factory=dict)
    owc_total_games: int = 0
    owc_total_size_bytes: int = 0
    owc_last_scan: float = 0.0

    # S3 bucket
    s3_available: bool = False
    s3_games_by_config: dict[str, int] = field(default_factory=dict)
    s3_total_games: int = 0
    s3_total_size_bytes: int = 0
    s3_last_scan: float = 0.0
    s3_bucket: str = ""

    # Scan error tracking (Jan 2026: for debugging manifest issues)
    owc_scan_error: str = ""
    s3_scan_error: str = ""

    def to_dict(self) -> dict:
        return {
            'collected_at': self.collected_at,
            'owc_available': self.owc_available,
            'owc_games_by_config': self.owc_games_by_config,
            'owc_total_games': self.owc_total_games,
            'owc_total_size_bytes': self.owc_total_size_bytes,
            'owc_last_scan': self.owc_last_scan,
            'owc_scan_error': self.owc_scan_error,
            's3_available': self.s3_available,
            's3_games_by_config': self.s3_games_by_config,
            's3_total_games': self.s3_total_games,
            's3_total_size_bytes': self.s3_total_size_bytes,
            's3_last_scan': self.s3_last_scan,
            's3_bucket': self.s3_bucket,
            's3_scan_error': self.s3_scan_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExternalStorageManifest:
        return cls(**d)


@dataclass
class DataSyncJob:
    """Tracks a P2P data synchronization job between nodes."""
    job_id: str
    source_node: str            # Node that has the file(s)
    target_node: str            # Node that needs the file(s)
    files: list[str]            # List of file paths to sync (relative to data/)
    status: str = "pending"     # pending, running, completed, failed
    started_at: float = 0.0
    completed_at: float = 0.0
    bytes_transferred: int = 0
    files_completed: int = 0
    error_message: str = ""
    # Rsync process details
    rsync_pid: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DataSyncJob:
        return cls(**d)


@dataclass
class ClusterSyncPlan:
    """Leader-generated plan for synchronizing data across the cluster."""
    plan_id: str
    created_at: float
    total_files_to_sync: int = 0
    total_bytes_to_sync: int = 0
    sync_jobs: list[DataSyncJob] = field(default_factory=list)
    # Status tracking
    status: str = "pending"     # pending, running, completed, failed
    jobs_completed: int = 0
    jobs_failed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['sync_jobs'] = [j.to_dict() for j in self.sync_jobs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ClusterSyncPlan:
        d = d.copy()
        d['sync_jobs'] = [DataSyncJob.from_dict(j) for j in d.get('sync_jobs', [])]
        return cls(**d)


# ============================================
# Phase 1: P2P Hardening - Jan 3, 2026
# Per-peer circuit breaker and health scoring
# ============================================

@dataclass
class PeerCircuitBreaker:
    """Per-peer circuit breaker for isolating flaky nodes.

    Jan 3, 2026: Sprint 10+ P2P hardening. Provides finer-grained failure
    isolation than per-transport breakers. When a specific peer has repeated
    failures, we can stop routing requests to it without affecting other peers.

    States:
        - closed: Normal operation, requests allowed
        - open: Breaker tripped, requests blocked for recovery_timeout
        - half-open: Testing recovery, allow single request

    Usage:
        breaker = PeerCircuitBreaker(peer_id="vast-12345")
        if breaker.should_allow_request():
            try:
                result = await send_request(peer)
                breaker.record_success()
            except Exception:
                breaker.record_failure()
    """
    peer_id: str
    failure_count: int = 0
    success_count: int = 0
    state: str = "closed"  # closed, open, half-open
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_state_change: float = 0.0
    # Configuration
    failure_threshold: int = 3  # Failures before opening
    success_threshold: int = 2  # Successes in half-open to close
    recovery_timeout: float = 30.0  # Seconds before half-open

    def record_failure(self) -> None:
        """Record a failure and potentially trip the breaker."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0  # Reset success streak

        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.last_state_change = time.time()
        elif self.state == "half-open":
            # Any failure in half-open goes back to open
            self.state = "open"
            self.last_state_change = time.time()

    def record_success(self) -> None:
        """Record a success and potentially close the breaker."""
        self.success_count += 1
        self.last_success_time = time.time()

        if self.state == "half-open" and self.success_count >= self.success_threshold:
            self.state = "closed"
            self.failure_count = 0
            self.last_state_change = time.time()
        elif self.state == "closed":
            # Decay failure count on success in closed state
            self.failure_count = max(0, self.failure_count - 1)

    def should_allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        now = time.time()

        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if recovery timeout has passed
            if now - self.last_state_change > self.recovery_timeout:
                self.state = "half-open"
                self.last_state_change = now
                self.success_count = 0
                return True
            return False
        else:  # half-open
            # Allow test requests in half-open state
            return True

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()

    def is_open(self) -> bool:
        """Check if breaker is currently open (blocking requests)."""
        return self.state == "open"

    def is_half_open(self) -> bool:
        """Check if breaker is in half-open (testing) state."""
        return self.state == "half-open"

    def to_dict(self) -> dict:
        """Serialize for JSON/state persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PeerCircuitBreaker":
        """Deserialize from dict."""
        return cls(**d)


@dataclass
class PeerHealthScore:
    """Composite health score for a peer node.

    Jan 3, 2026: Sprint 10+ P2P hardening. Combines multiple metrics into
    a single health score (0.0-1.0) for proactive failure detection.
    Nodes with low health scores can be deprioritized for job dispatch
    before they become completely unreachable.

    Metrics:
        - success_rate: Recent request success rate (last 100 requests)
        - avg_latency_ms: Average response latency
        - availability_percent: Uptime in observation window

    Usage:
        score = PeerHealthScore(peer_id="vast-12345")
        score.record_request(success=True, latency_ms=50.0)
        if score.composite_score < 0.7:
            emit_event(PEER_HEALTH_DEGRADED, {"peer_id": score.peer_id})
    """
    peer_id: str
    # Rolling metrics (last 100 requests)
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    request_count: int = 0
    # Availability tracking
    first_seen: float = 0.0
    last_seen: float = 0.0
    offline_duration_seconds: float = 0.0
    # Score caching
    _cached_score: float = 1.0
    _cache_time: float = 0.0

    def __post_init__(self):
        if self.first_seen == 0.0:
            self.first_seen = time.time()
        if self.last_seen == 0.0:
            self.last_seen = time.time()

    @property
    def success_rate(self) -> float:
        """Calculate success rate from recent requests."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0  # Assume healthy if no data
        return self.success_count / total

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency from recent requests."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def availability_percent(self) -> float:
        """Calculate availability as percentage of time online."""
        now = time.time()
        total_time = now - self.first_seen
        if total_time <= 0:
            return 1.0
        online_time = total_time - self.offline_duration_seconds
        return max(0.0, min(1.0, online_time / total_time))

    @property
    def composite_score(self) -> float:
        """Calculate weighted composite health score (0.0-1.0).

        Weights:
            - success_rate: 50% (most important - actual request outcomes)
            - latency: 30% (performance indicator)
            - availability: 20% (historical reliability)
        """
        # Recalculate at most once per second
        now = time.time()
        if now - self._cache_time < 1.0:
            return self._cached_score

        # Success rate component (0-1)
        success_component = self.success_rate * 0.5

        # Latency component (0-1): 0ms = 1.0, 1000ms+ = 0.0
        latency_normalized = 1.0 - min(self.avg_latency_ms / 1000.0, 1.0)
        latency_component = latency_normalized * 0.3

        # Availability component (0-1)
        availability_component = self.availability_percent * 0.2

        self._cached_score = success_component + latency_component + availability_component
        self._cache_time = now
        return self._cached_score

    def record_request(self, success: bool, latency_ms: float = 0.0) -> None:
        """Record a request outcome for health tracking."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        if latency_ms > 0:
            self.total_latency_ms += latency_ms
            self.request_count += 1

        self.last_seen = time.time()

        # Decay old data - keep only ~100 recent requests
        total = self.success_count + self.failure_count
        if total > 150:
            decay_factor = 100.0 / total
            self.success_count = int(self.success_count * decay_factor)
            self.failure_count = int(self.failure_count * decay_factor)
            self.total_latency_ms *= decay_factor
            self.request_count = int(self.request_count * decay_factor)

    def record_offline(self, duration_seconds: float) -> None:
        """Record an offline period."""
        self.offline_duration_seconds += duration_seconds

    def is_degraded(self, threshold: float = 0.7) -> bool:
        """Check if peer health is below acceptable threshold."""
        return self.composite_score < threshold

    def is_critical(self, threshold: float = 0.4) -> bool:
        """Check if peer health is critically low."""
        return self.composite_score < threshold

    def to_dict(self) -> dict:
        """Serialize for JSON/API response."""
        return {
            "peer_id": self.peer_id,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "availability_percent": round(self.availability_percent, 3),
            "composite_score": round(self.composite_score, 3),
            "is_degraded": self.is_degraded(),
            "is_critical": self.is_critical(),
            "request_count": self.request_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PeerHealthScore":
        """Deserialize from persistence."""
        # Handle both full dict and simplified API response
        return cls(
            peer_id=d["peer_id"],
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            total_latency_ms=d.get("total_latency_ms", 0.0),
            request_count=d.get("request_count", 0),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
            offline_duration_seconds=d.get("offline_duration_seconds", 0.0),
        )
