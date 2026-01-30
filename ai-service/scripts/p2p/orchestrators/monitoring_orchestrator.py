"""Monitoring orchestrator for P2P validation and health servers.

January 2026: Created as part of Phase 4 P2POrchestrator decomposition.
Handles startup validation, health endpoints, and monitoring infrastructure.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aiohttp import web

from .base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)


class MonitoringOrchestrator(BaseOrchestrator):
    """Orchestrator for monitoring, validation, and health infrastructure.

    Responsibilities:
    - Startup validation (SWIM/Raft, managers, voters, PyTorch CUDA)
    - Isolated health server for liveness/readiness probes
    - Validation result tracking
    """

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the monitoring orchestrator."""
        super().__init__(p2p)
        self._validation_result: dict[str, Any] | None = None
        self._health_server_started = False

    @property
    def name(self) -> str:
        """Return the orchestrator name."""
        return "monitoring"

    def health_check(self) -> HealthCheckResult:
        """Check health of the monitoring orchestrator."""
        details = {
            "validation_completed": self._validation_result is not None,
            "health_server_started": self._health_server_started,
        }

        if self._validation_result:
            details["validation_warnings"] = len(self._validation_result.get("warnings", []))
            details["validation_errors"] = len(self._validation_result.get("errors", []))

        healthy = (
            self._validation_result is not None
            and len(self._validation_result.get("errors", [])) == 0
        )

        return HealthCheckResult(
            healthy=healthy,
            message="Monitoring orchestrator operational" if healthy else "Validation incomplete or has errors",
            details=details,
        )

    def validate_critical_subsystems(self) -> dict:
        """Validate critical subsystems at startup.

        Returns a status dict with protocol and manager availability.
        Logs clear messages about which protocols are active.

        December 2025: Added to address silent fallback behavior
        where operators couldn't tell if SWIM/Raft was running.
        """
        p2p = self._p2p

        # Import constants with fallback
        try:
            from app.p2p.constants import (
                SWIM_ENABLED, RAFT_ENABLED, MEMBERSHIP_MODE, CONSENSUS_MODE
            )
        except ImportError:
            SWIM_ENABLED = False
            RAFT_ENABLED = False
            MEMBERSHIP_MODE = "http"
            CONSENSUS_MODE = "bully"

        status: dict[str, Any] = {
            "protocols": {
                "membership_mode": MEMBERSHIP_MODE,
                "consensus_mode": CONSENSUS_MODE,
                "swim_enabled": SWIM_ENABLED,
                "raft_enabled": RAFT_ENABLED,
            },
            "managers": {},
            "warnings": [],
            "errors": [],
        }

        # Check SWIM availability
        try:
            from app.p2p.swim_adapter import SWIM_AVAILABLE
            status["protocols"]["swim_available"] = SWIM_AVAILABLE
            if SWIM_ENABLED and not SWIM_AVAILABLE:
                msg = "SWIM_ENABLED=true but swim-p2p not installed. Install: pip install swim-p2p>=1.2.0"
                status["warnings"].append(msg)
                self._log_warning(f"[Startup Validation] {msg}")
            elif SWIM_AVAILABLE:
                self._log_info(f"[Startup Validation] SWIM protocol available (membership_mode={MEMBERSHIP_MODE})")
        except ImportError:
            status["protocols"]["swim_available"] = False
            if SWIM_ENABLED:
                status["warnings"].append("swim_adapter import failed")

        # Check Raft availability
        try:
            from app.p2p.raft_state import PYSYNCOBJ_AVAILABLE
            status["protocols"]["raft_available"] = PYSYNCOBJ_AVAILABLE
            if RAFT_ENABLED and not PYSYNCOBJ_AVAILABLE:
                msg = "RAFT_ENABLED=true but pysyncobj not installed. Install: pip install pysyncobj>=0.3.14"
                status["warnings"].append(msg)
                self._log_warning(f"[Startup Validation] {msg}")
            elif PYSYNCOBJ_AVAILABLE:
                self._log_info(f"[Startup Validation] Raft protocol available (consensus_mode={CONSENSUS_MODE})")
        except ImportError:
            status["protocols"]["raft_available"] = False
            if RAFT_ENABLED:
                status["warnings"].append("raft_state import failed")

        # Log active protocol configuration
        self._log_info(
            f"[Startup Validation] Protocol config: membership={MEMBERSHIP_MODE}, consensus={CONSENSUS_MODE}"
        )

        # Check critical managers (lazy load check - don't fail, just report)
        manager_checks = [
            ("work_queue", "app.coordination.work_queue", "get_work_queue"),
            ("health_manager", "app.coordination.unified_health_manager", "get_unified_health_manager"),
            ("sync_router", "app.coordination.sync_router", "get_sync_router"),
        ]

        for name, module_path, getter_name in manager_checks:
            try:
                module = importlib.import_module(module_path)
                getter = getattr(module, getter_name, None)
                status["managers"][name] = getter is not None
                if getter:
                    self._log_debug(f"[Startup Validation] Manager {name} available")
            except ImportError as e:
                status["managers"][name] = False
                status["warnings"].append(f"{name} import failed: {e}")
                self._log_warning(f"[Startup Validation] Manager {name} unavailable: {e}")

        # December 2025: P2P voter connectivity validation
        voter_node_ids = getattr(p2p, "voter_node_ids", [])
        voter_quorum_size = getattr(p2p, "voter_quorum_size", 0)
        port = getattr(p2p, "port", 8770)

        status["voters"] = {
            "configured": len(voter_node_ids),
            "quorum": voter_quorum_size,
            "reachable": 0,
            "unreachable": [],
        }

        if voter_node_ids:
            reachable_count = 0
            for voter_id in voter_node_ids:
                try:
                    from app.config.cluster_config import get_cluster_nodes
                    nodes = get_cluster_nodes()
                    node = nodes.get(voter_id)
                    if node:
                        voter_ip = node.best_ip
                        if voter_ip:
                            with contextlib.suppress(Exception):
                                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                sock.settimeout(2.0)
                                result = sock.connect_ex((voter_ip, port))
                                sock.close()
                                if result == 0:
                                    reachable_count += 1
                                    continue
                    status["voters"]["unreachable"].append(voter_id)
                except (socket.error, socket.timeout, OSError, TimeoutError, ConnectionRefusedError):
                    status["voters"]["unreachable"].append(voter_id)

            status["voters"]["reachable"] = reachable_count

            if reachable_count < voter_quorum_size:
                msg = (
                    f"Only {reachable_count}/{len(voter_node_ids)} voters reachable, "
                    f"need {voter_quorum_size} for quorum. Unreachable: {status['voters']['unreachable']}"
                )
                status["warnings"].append(msg)
                self._log_warning(f"[Startup Validation] {msg}")
                # Emit QUORUM_VALIDATION_FAILED event
                try:
                    from app.distributed.data_events import DataEventType, get_event_bus
                    get_event_bus().emit(
                        DataEventType.QUORUM_VALIDATION_FAILED,
                        {
                            "node_id": self.node_id,
                            "reachable_voters": reachable_count,
                            "total_voters": len(voter_node_ids),
                            "quorum_required": voter_quorum_size,
                            "unreachable": status["voters"]["unreachable"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except (ImportError, Exception) as e:
                    self._log_debug(f"[Startup Validation] Could not emit quorum validation event: {e}")
            else:
                self._log_info(
                    f"[Startup Validation] Voter quorum OK: "
                    f"{reachable_count}/{len(voter_node_ids)} voters reachable"
                )
        else:
            self._log_info("[Startup Validation] No voters configured - quorum checks disabled")

        # Jan 9, 2026: PyTorch CUDA validation
        resource_detector = getattr(p2p, "_resource_detector", None)
        if resource_detector and hasattr(resource_detector, "validate_pytorch_cuda"):
            try:
                pytorch_status = resource_detector.validate_pytorch_cuda()
                status["pytorch"] = pytorch_status

                if pytorch_status.get("warning"):
                    status["warnings"].append(pytorch_status["warning"])
                    self._log_warning(f"[Startup Validation] {pytorch_status['warning']}")

                    try:
                        from app.distributed.data_events import DataEventType, get_event_bus
                        get_event_bus().emit(
                            DataEventType.PYTORCH_CUDA_MISMATCH,
                            {
                                "node_id": self.node_id,
                                "warning": pytorch_status["warning"],
                                "gpu_detected": pytorch_status.get("gpu_detected", False),
                                "pytorch_cuda_available": pytorch_status.get("pytorch_cuda_available", False),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except (ImportError, Exception) as e:
                        self._log_debug(f"[Startup Validation] Could not emit PyTorch CUDA event: {e}")
                elif pytorch_status.get("pytorch_cuda_available"):
                    self._log_info(
                        f"[Startup Validation] PyTorch CUDA OK: "
                        f"version={pytorch_status.get('pytorch_cuda_version')}, "
                        f"devices={pytorch_status.get('cuda_device_count')}"
                    )
                elif pytorch_status.get("error"):
                    self._log_debug(f"[Startup Validation] PyTorch not installed: {pytorch_status.get('error')}")
            except Exception as e:
                self._log_debug(f"[Startup Validation] PyTorch validation failed: {e}")

        # Summary log
        available_count = sum(1 for v in status["managers"].values() if v)
        total_count = len(status["managers"])
        if status["warnings"]:
            self._log_warning(
                f"[Startup Validation] Completed with {len(status['warnings'])} warnings. "
                f"Managers: {available_count}/{total_count} available"
            )
        else:
            self._log_info(
                f"[Startup Validation] All checks passed. "
                f"Managers: {available_count}/{total_count} available"
            )

        self._validation_result = status
        return status

    def start_isolated_health_server(self) -> None:
        """Start a lightweight health HTTP server in a separate thread.

        January 2026: This server runs in its own thread with its own event loop,
        guaranteeing that /health endpoints respond even when the main event loop
        is blocked by background tasks.

        The isolated server:
        - Listens on port + 2 (8772 for P2P on 8770)
        - Only serves /health and /ready endpoints
        - Does not access any state that requires the main event loop
        - Responds within 100ms even under heavy load
        """
        p2p = self._p2p
        health_port = getattr(p2p, "port", 8770) + 2

        def _run_health_server_in_thread() -> None:
            """Run the health server in a separate thread with its own event loop."""
            import asyncio as thread_asyncio

            async def handle_health(request: web.Request) -> web.Response:
                """Liveness probe - returns 200 if P2P process is alive."""
                uptime = time.time() - getattr(p2p, "start_time", time.time())
                role = getattr(p2p, "role", None)
                role_value = role.value if hasattr(role, 'value') else str(role)
                return web.json_response({
                    "alive": True,
                    "node_id": p2p.node_id,
                    "role": role_value,
                    "uptime_seconds": uptime,
                    "main_port": getattr(p2p, "port", 8770),
                    "isolated_health_server": True,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            async def handle_ready(request: web.Request) -> web.Response:
                """Readiness probe - returns 200 if P2P has started up."""
                uptime = time.time() - getattr(p2p, "start_time", time.time())
                is_ready = uptime >= 30.0
                return web.json_response({
                    "ready": is_ready,
                    "node_id": p2p.node_id,
                    "uptime_seconds": uptime,
                    "startup_complete": is_ready,
                    "timestamp": datetime.utcnow().isoformat(),
                }, status=200 if is_ready else 503)

            async def run_server() -> None:
                """Set up and run the health server."""
                app = web.Application()
                app.router.add_get('/health', handle_health)
                app.router.add_get('/ready', handle_ready)

                runner = web.AppRunner(app)
                await runner.setup()

                try:
                    site = web.TCPSite(runner, '0.0.0.0', health_port, reuse_address=True)
                    await site.start()
                    logger.info(f"Isolated health server started on 0.0.0.0:{health_port}")

                    while True:
                        await thread_asyncio.sleep(3600)
                except OSError as e:
                    if "Address already in use" in str(e):
                        logger.warning(f"Isolated health server port {health_port} in use, skipping")
                    else:
                        logger.error(f"Isolated health server failed: {e}")
                except Exception as e:
                    logger.error(f"Isolated health server error: {e}")

            loop = thread_asyncio.new_event_loop()
            thread_asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_server())
            except Exception as e:
                logger.error(f"Isolated health server thread failed: {e}")
            finally:
                loop.close()

        health_thread = threading.Thread(
            target=_run_health_server_in_thread,
            name="isolated-health-server",
            daemon=True,
        )
        health_thread.start()
        self._health_server_started = True
        self._log_info(f"Started isolated health server thread (port {health_port})")

    def create_self_info(self) -> "NodeInfo":
        """Create NodeInfo for this node.

        Jan 29, 2026: Implementation moved from P2POrchestrator._create_self_info().
        Creates the node's self-description used for heartbeats and cluster membership.

        Returns:
            NodeInfo with hardware capabilities, addresses, and node metadata.
        """
        # Lazy import to avoid circular dependency
        from scripts.p2p.models import NodeInfo

        p2p = self._p2p

        # Detect GPU via ResourceDetectorMixin
        has_gpu, gpu_name = p2p._detect_gpu()

        cpu_count = int(os.cpu_count() or 0)

        # Detect memory via ResourceDetectorMixin
        memory_gb = p2p._detect_memory()

        # Detect capabilities based on hardware
        # Dec 2025: RINGRIFT_IS_COORDINATOR=true restricts to coordinator-only
        # Dec 29, 2025: Also check distributed_hosts.yaml for role/enabled flags
        is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")

        # Check YAML config for this node's settings
        if not is_coordinator:
            try:
                from app.config.cluster_config import load_cluster_config
                config = load_cluster_config()
                # ClusterConfig stores hosts in hosts_raw attribute
                nodes = getattr(config, "hosts_raw", {}) or {}
                node_cfg = nodes.get(self.node_id, {})
                # Check role or explicit enabled flags
                if node_cfg.get("role") == "coordinator":
                    is_coordinator = True
                    logger.info(f"[P2P] Node {self.node_id} is coordinator (from YAML)")
                elif node_cfg.get("selfplay_enabled") is False and node_cfg.get("training_enabled") is False:
                    is_coordinator = True
                    logger.info(f"[P2P] Node {self.node_id} has selfplay/training disabled (from YAML)")
            except Exception as e:
                logger.debug(f"[P2P] Could not load cluster config: {e}")

        if is_coordinator:
            # Dec 30, 2025: Warn if GPU node is misconfigured as coordinator
            if has_gpu:
                logger.warning(
                    f"[P2P] GPU node {self.node_id} is marked as coordinator - "
                    f"this may be a misconfiguration. GPU: {gpu_name}. "
                    "Unset RINGRIFT_IS_COORDINATOR or remove role:coordinator from YAML "
                    "to enable training capabilities."
                )
            capabilities = []  # Coordinator nodes don't run compute tasks
            logger.info("[P2P] Coordinator-only mode: no selfplay/training/cmaes capabilities")
        else:
            capabilities = ["selfplay"]
            if has_gpu:
                capabilities.extend(["training", "cmaes", "gauntlet", "tournament"])
            if memory_gb >= 64:
                capabilities.append("large_boards")

        info = NodeInfo(
            node_id=self.node_id,
            host=p2p.advertise_host,
            port=p2p.advertise_port,
            role=p2p.role,
            last_heartbeat=time.time(),
            cpu_count=cpu_count,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            capabilities=capabilities,
            version=p2p.build_version,
        )

        # Advertise an alternate mesh endpoint (Tailscale) for NAT traversal and
        # multi-path retries. Peers persist the observed reachable endpoint in
        # `host`/`port` but keep our `reported_host`/`reported_port` as an
        # additional candidate (see `_heartbeat_loop` multi-path retry).
        ts_ip = p2p._get_tailscale_ip()
        if ts_ip and ts_ip != info.host:
            info.reported_host = ts_ip
            # Use the actual listening port for mesh endpoints (port-mapped
            # advertise ports may not be reachable inside overlays).
            info.reported_port = int(p2p.port)

        # Jan 2026: Populate alternate_ips with all reachable IPs for partition healing
        # Peers can try multiple IPs to reach us, improving mesh resilience
        info.alternate_ips = p2p._discover_all_ips(exclude_primary=info.host)

        # Jan 13, 2026: Multi-address advertisement for voter counting fix
        # Nodes advertise ALL addresses they're reachable at in heartbeats.
        # This fixes voter quorum issues where voters are listed by config name
        # but peers report via Tailscale/public IPs that don't match.
        info.tailscale_ip = ts_ip or ""
        info.addresses = self._collect_all_addresses(ts_ip, info.host)

        # Jan 24, 2026: Populate visible_peers for connectivity scoring
        # Used by _compute_connectivity_score() to determine leader eligibility
        info.visible_peers = len([peer for peer in p2p.peers.values() if peer.is_alive()])

        # Jan 25, 2026: Compute effective_timeout for broadcast to peers
        # This tells other nodes how long to wait before marking us dead
        try:
            from app.p2p.constants import PEER_TIMEOUT, get_cpu_adaptive_timeout
            from app.config.provider_timeouts import ProviderTimeouts
            cpu_load = info.cpu_percent / 100.0 if info.cpu_percent > 0 else 0.0
            base_timeout = get_cpu_adaptive_timeout(PEER_TIMEOUT, cpu_load)
            provider_mult = ProviderTimeouts.get_multiplier(self.node_id) if ProviderTimeouts else 1.0
            info.effective_timeout = base_timeout * provider_mult
        except Exception:
            info.effective_timeout = 180.0  # Fallback to default

        return info

    def _collect_all_addresses(
        self, tailscale_ip: str | None, primary_host: str
    ) -> list[str]:
        """Collect all addresses this node is reachable at.

        Jan 13, 2026: For multi-address advertisement to fix voter counting.
        Jan 29, 2026: Implementation moved from P2POrchestrator._collect_all_addresses().

        Returns addresses in priority order:
        1. Tailscale IP (100.x.x.x) - most reliable for P2P mesh
        2. Primary host (advertise_host) - what we're currently advertising
        3. SSH host from config - public/direct access
        4. Local interface IP - same-network access

        Args:
            tailscale_ip: Tailscale VPN IP if available
            primary_host: Current advertise_host

        Returns:
            List of addresses, deduplicated, in priority order
        """
        p2p = self._p2p
        addresses: list[str] = []
        seen: set[str] = set()

        def add_if_new(addr: str | None) -> None:
            if addr and addr not in seen and addr not in ("", "0.0.0.0", "127.0.0.1"):
                addresses.append(addr)
                seen.add(addr)

        # Priority 1: Tailscale IP (best for mesh)
        add_if_new(tailscale_ip)

        # Priority 2: Current advertise host
        add_if_new(primary_host)

        # Priority 3: SSH host from config (may be public IP)
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}
            node_cfg = nodes.get(self.node_id, {})
            if node_cfg:
                add_if_new(node_cfg.get("ssh_host"))
                add_if_new(node_cfg.get("tailscale_ip"))
        except Exception:
            pass

        # Priority 4: Local interface IPs
        for ip in p2p._discover_all_ips(exclude_primary=None):
            add_if_new(ip)

        return addresses

    def get_stability_metrics(self) -> dict:
        """Get current stability metrics for effectiveness tracking.

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_stability_metrics().

        Returns metrics used to evaluate whether recovery actions helped:
        - alive_count: Number of alive peers
        - total_count: Total peers in cluster
        - stability_score: 0-100 score based on alive ratio, leader presence, flapping
        """
        p2p = self._p2p
        alive_count = p2p._peer_query.alive_count(exclude_self=False).unwrap_or(0)
        total_count = len(p2p.peers) + 1  # Include self

        # Calculate stability score (0-100)
        stability_score = 0.0
        if total_count > 0:
            alive_ratio = alive_count / total_count
            stability_score = alive_ratio * 100

            # Bonus for having a leader
            if p2p.leader_id:
                stability_score += 10

            # Penalty for flapping peers
            if p2p._peer_state_tracker:
                try:
                    diag = p2p._peer_state_tracker.get_diagnostics()
                    flapping_count = len(diag.get("flapping_peers", []))
                    stability_score -= flapping_count * 5
                except Exception:
                    pass

        return {
            "alive_count": alive_count,
            "total_count": total_count,
            "stability_score": max(0, min(100, stability_score)),
        }
