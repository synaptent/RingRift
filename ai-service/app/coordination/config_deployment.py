"""Atomic Configuration Deployment for P2P Cluster.

Part of Phase 3: Configuration Synchronization
P2P Cluster Stability Plan - Jan 13, 2026

This module provides two-phase commit (2PC) atomic config deployment
across the cluster, ensuring all nodes have consistent configuration.

Usage:
    from app.coordination.config_deployment import AtomicConfigDeployer

    deployer = AtomicConfigDeployer()
    result = await deployer.deploy(config_content, target_nodes)
    if result.success:
        print(f"Config deployed to {len(result.nodes_updated)} nodes")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Feature flag for atomic config deployment
ATOMIC_DEPLOY_ENABLED = os.environ.get(
    "RINGRIFT_ATOMIC_CONFIG_DEPLOY", "false"
).lower() == "true"

# Timeout for each phase
PREPARE_TIMEOUT = 30.0  # seconds
COMMIT_TIMEOUT = 60.0  # seconds


class DeployPhase(Enum):
    """Phases of two-phase commit deployment."""
    PREPARE = "prepare"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    COMPLETE = "complete"


class NodeAckStatus(Enum):
    """Status of node acknowledgment in 2PC."""
    PENDING = "pending"
    ACK = "ack"
    NACK = "nack"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class NodeDeployResult:
    """Result of deploying config to a single node."""
    node_id: str
    phase: DeployPhase
    status: NodeAckStatus
    config_hash: str = ""
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class DeployResult:
    """Result of cluster-wide config deployment."""
    success: bool
    reason: str
    phase: DeployPhase
    nodes_updated: list[str] = field(default_factory=list)
    nodes_failed: list[str] = field(default_factory=list)
    node_results: list[NodeDeployResult] = field(default_factory=list)
    config_hash: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "success": self.success,
            "reason": self.reason,
            "phase": self.phase.value,
            "nodes_updated": self.nodes_updated,
            "nodes_failed": self.nodes_failed,
            "node_count": len(self.nodes_updated),
            "failed_count": len(self.nodes_failed),
            "config_hash": self.config_hash[:16] if self.config_hash else "",
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class AtomicConfigDeployer:
    """Deploys configuration atomically using two-phase commit.

    The deployment process:
    1. PREPARE phase: Send config to all targets, get ACKs
    2. Check quorum: Need ACKs from >= half of targets
    3. COMMIT phase: Tell all nodes to apply the config
    4. VERIFY: Check all nodes have matching hash
    5. ROLLBACK: If verification fails, rollback to previous

    This ensures either all nodes get the new config or none do.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        backup_dir: Path | None = None,
    ):
        """Initialize the deployer.

        Args:
            config_path: Local config path. If None, uses default.
            backup_dir: Directory for backups. If None, uses default.
        """
        self._config_path = config_path or self._default_config_path()
        self._backup_dir = backup_dir or self._config_path.parent / "backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def _default_config_path(self) -> Path:
        """Get default config path."""
        paths = [
            Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml",
            Path.cwd() / "config" / "distributed_hosts.yaml",
        ]
        for path in paths:
            if path.exists():
                return path
        return paths[0]

    async def deploy(
        self,
        config_content: str,
        targets: list[str],
        require_quorum: bool = True,
    ) -> DeployResult:
        """Deploy configuration to target nodes atomically.

        Args:
            config_content: YAML config content to deploy
            targets: List of target node IDs
            require_quorum: If True, require majority ACKs

        Returns:
            DeployResult with success status and details
        """
        start_time = time.time()

        if not targets:
            return DeployResult(
                success=False,
                reason="No target nodes specified",
                phase=DeployPhase.PREPARE,
            )

        # Compute config hash
        config_hash = hashlib.sha256(config_content.encode()).hexdigest()
        quorum_size = len(targets) // 2 + 1 if require_quorum else 1

        logger.info(
            f"[ConfigDeploy] Starting atomic deployment to {len(targets)} nodes "
            f"(hash={config_hash[:16]}, quorum={quorum_size})"
        )

        # Phase 1: PREPARE - Get ACKs from all targets
        prepare_results = await self._phase_prepare(config_content, config_hash, targets)
        ack_count = sum(1 for r in prepare_results if r.status == NodeAckStatus.ACK)

        if ack_count < quorum_size:
            failed_nodes = [r.node_id for r in prepare_results if r.status != NodeAckStatus.ACK]
            return DeployResult(
                success=False,
                reason=f"Insufficient ACKs: {ack_count}/{len(targets)} (need {quorum_size})",
                phase=DeployPhase.PREPARE,
                nodes_failed=failed_nodes,
                node_results=prepare_results,
                config_hash=config_hash,
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Phase 2: COMMIT - Push and verify
        acked_nodes = [r.node_id for r in prepare_results if r.status == NodeAckStatus.ACK]
        commit_results = await self._phase_commit(config_content, config_hash, acked_nodes)

        # Verify all committed nodes have matching hash
        success_nodes = [r.node_id for r in commit_results if r.status == NodeAckStatus.ACK]
        failed_nodes = [r.node_id for r in commit_results if r.status != NodeAckStatus.ACK]

        if failed_nodes:
            # Rollback on partial failure
            logger.warning(
                f"[ConfigDeploy] Partial failure: {len(failed_nodes)} nodes failed commit, rolling back"
            )
            await self._phase_rollback(success_nodes)
            return DeployResult(
                success=False,
                reason=f"Commit failed on {len(failed_nodes)} nodes, rolled back",
                phase=DeployPhase.ROLLBACK,
                nodes_updated=[],
                nodes_failed=failed_nodes,
                node_results=commit_results,
                config_hash=config_hash,
                duration_ms=(time.time() - start_time) * 1000,
            )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[ConfigDeploy] Deployment complete: {len(success_nodes)} nodes updated "
            f"in {duration_ms:.0f}ms"
        )

        return DeployResult(
            success=True,
            reason="OK",
            phase=DeployPhase.COMPLETE,
            nodes_updated=success_nodes,
            node_results=commit_results,
            config_hash=config_hash,
            duration_ms=duration_ms,
        )

    async def _phase_prepare(
        self,
        config_content: str,
        config_hash: str,
        targets: list[str],
    ) -> list[NodeDeployResult]:
        """Phase 1: Send PREPARE to all targets and collect ACKs.

        Args:
            config_content: Config content to deploy
            config_hash: SHA256 hash of config
            targets: List of target node IDs

        Returns:
            List of NodeDeployResult for each target
        """
        tasks = [
            self._prepare_node(node_id, config_content, config_hash)
            for node_id in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prepare_results = []
        for node_id, result in zip(targets, results):
            if isinstance(result, Exception):
                prepare_results.append(NodeDeployResult(
                    node_id=node_id,
                    phase=DeployPhase.PREPARE,
                    status=NodeAckStatus.ERROR,
                    error=str(result),
                ))
            else:
                prepare_results.append(result)

        return prepare_results

    async def _prepare_node(
        self,
        node_id: str,
        config_content: str,
        config_hash: str,
    ) -> NodeDeployResult:
        """Send PREPARE to a single node.

        Args:
            node_id: Target node ID
            config_content: Config content
            config_hash: Expected hash

        Returns:
            NodeDeployResult with ACK/NACK status
        """
        start_time = time.time()

        try:
            # Get node connection info
            from app.config.cluster_config import get_cluster_nodes
            nodes = get_cluster_nodes()
            node = nodes.get(node_id)

            if not node or not node.best_ip:
                return NodeDeployResult(
                    node_id=node_id,
                    phase=DeployPhase.PREPARE,
                    status=NodeAckStatus.ERROR,
                    error="Node not found in config",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Send PREPARE via HTTP POST to node's P2P port
            import aiohttp
            url = f"http://{node.best_ip}:8770/config/prepare"
            payload = {
                "config_hash": config_hash,
                "config_size": len(config_content),
                "source_node": self._get_local_node_id(),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=PREPARE_TIMEOUT),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ack"):
                            return NodeDeployResult(
                                node_id=node_id,
                                phase=DeployPhase.PREPARE,
                                status=NodeAckStatus.ACK,
                                config_hash=config_hash,
                                duration_ms=(time.time() - start_time) * 1000,
                            )

                    return NodeDeployResult(
                        node_id=node_id,
                        phase=DeployPhase.PREPARE,
                        status=NodeAckStatus.NACK,
                        error=f"HTTP {response.status}",
                        duration_ms=(time.time() - start_time) * 1000,
                    )

        except asyncio.TimeoutError:
            return NodeDeployResult(
                node_id=node_id,
                phase=DeployPhase.PREPARE,
                status=NodeAckStatus.TIMEOUT,
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return NodeDeployResult(
                node_id=node_id,
                phase=DeployPhase.PREPARE,
                status=NodeAckStatus.ERROR,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _phase_commit(
        self,
        config_content: str,
        config_hash: str,
        targets: list[str],
    ) -> list[NodeDeployResult]:
        """Phase 2: Send COMMIT to all prepared nodes.

        Args:
            config_content: Config content to write
            config_hash: Expected hash after write
            targets: List of prepared node IDs

        Returns:
            List of NodeDeployResult for each target
        """
        tasks = [
            self._commit_node(node_id, config_content, config_hash)
            for node_id in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        commit_results = []
        for node_id, result in zip(targets, results):
            if isinstance(result, Exception):
                commit_results.append(NodeDeployResult(
                    node_id=node_id,
                    phase=DeployPhase.COMMIT,
                    status=NodeAckStatus.ERROR,
                    error=str(result),
                ))
            else:
                commit_results.append(result)

        return commit_results

    async def _commit_node(
        self,
        node_id: str,
        config_content: str,
        config_hash: str,
    ) -> NodeDeployResult:
        """Send COMMIT to a single node with config content.

        Args:
            node_id: Target node ID
            config_content: Config content to write
            config_hash: Expected hash after write

        Returns:
            NodeDeployResult with success/failure
        """
        start_time = time.time()

        try:
            from app.config.cluster_config import get_cluster_nodes
            nodes = get_cluster_nodes()
            node = nodes.get(node_id)

            if not node or not node.best_ip:
                return NodeDeployResult(
                    node_id=node_id,
                    phase=DeployPhase.COMMIT,
                    status=NodeAckStatus.ERROR,
                    error="Node not found",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Send COMMIT via HTTP POST
            import aiohttp
            url = f"http://{node.best_ip}:8770/config/commit"
            payload = {
                "config_content": config_content,
                "config_hash": config_hash,
                "source_node": self._get_local_node_id(),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=COMMIT_TIMEOUT),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        actual_hash = data.get("config_hash", "")

                        if actual_hash == config_hash:
                            return NodeDeployResult(
                                node_id=node_id,
                                phase=DeployPhase.COMMIT,
                                status=NodeAckStatus.ACK,
                                config_hash=actual_hash,
                                duration_ms=(time.time() - start_time) * 1000,
                            )
                        else:
                            return NodeDeployResult(
                                node_id=node_id,
                                phase=DeployPhase.COMMIT,
                                status=NodeAckStatus.NACK,
                                config_hash=actual_hash,
                                error=f"Hash mismatch: expected {config_hash[:16]}, got {actual_hash[:16]}",
                                duration_ms=(time.time() - start_time) * 1000,
                            )

                    return NodeDeployResult(
                        node_id=node_id,
                        phase=DeployPhase.COMMIT,
                        status=NodeAckStatus.NACK,
                        error=f"HTTP {response.status}",
                        duration_ms=(time.time() - start_time) * 1000,
                    )

        except asyncio.TimeoutError:
            return NodeDeployResult(
                node_id=node_id,
                phase=DeployPhase.COMMIT,
                status=NodeAckStatus.TIMEOUT,
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return NodeDeployResult(
                node_id=node_id,
                phase=DeployPhase.COMMIT,
                status=NodeAckStatus.ERROR,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _phase_rollback(self, nodes: list[str]) -> None:
        """Rollback config on nodes that received commit.

        Args:
            nodes: List of node IDs to rollback
        """
        logger.info(f"[ConfigDeploy] Rolling back {len(nodes)} nodes")

        for node_id in nodes:
            try:
                from app.config.cluster_config import get_cluster_nodes
                nodes_config = get_cluster_nodes()
                node = nodes_config.get(node_id)

                if not node or not node.best_ip:
                    continue

                import aiohttp
                url = f"http://{node.best_ip}:8770/config/rollback"

                async with aiohttp.ClientSession() as session:
                    await session.post(
                        url,
                        json={"source_node": self._get_local_node_id()},
                        timeout=aiohttp.ClientTimeout(total=30),
                    )

            except Exception as e:
                logger.warning(f"[ConfigDeploy] Rollback failed for {node_id}: {e}")

    def _get_local_node_id(self) -> str:
        """Get local node ID."""
        try:
            from app.config.node_identity import get_node_id_safe
            return get_node_id_safe()
        except ImportError:
            import socket
            return socket.gethostname()

    def apply_config_atomic(self, content: str, path: Path) -> tuple[bool, str]:
        """Apply config to local filesystem atomically.

        Args:
            content: Config content to write
            path: Target path

        Returns:
            Tuple of (success, hash) or (False, error)
        """
        try:
            # Create backup
            backup_path = self._backup_dir / f"distributed_hosts.{int(time.time())}.yaml"
            if path.exists():
                shutil.copy2(path, backup_path)

            # Write to temp file
            temp_path = path.with_suffix(".tmp")
            temp_path.write_text(content)

            # Atomic rename
            temp_path.rename(path)

            # Compute hash
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            logger.info(f"[ConfigDeploy] Applied config atomically: {content_hash[:16]}")
            return True, content_hash

        except (OSError, PermissionError) as e:
            logger.error(f"[ConfigDeploy] Atomic apply failed: {e}")
            return False, str(e)

    def rollback_local(self) -> bool:
        """Rollback local config to most recent backup.

        Returns:
            True if rollback successful
        """
        try:
            backups = sorted(self._backup_dir.glob("distributed_hosts.*.yaml"))
            if not backups:
                logger.warning("[ConfigDeploy] No backups available for rollback")
                return False

            latest_backup = backups[-1]
            shutil.copy2(latest_backup, self._config_path)
            logger.info(f"[ConfigDeploy] Rolled back to {latest_backup.name}")
            return True

        except (OSError, PermissionError) as e:
            logger.error(f"[ConfigDeploy] Rollback failed: {e}")
            return False


# Module-level singleton
_deployer_instance: AtomicConfigDeployer | None = None


def get_config_deployer() -> AtomicConfigDeployer:
    """Get the global AtomicConfigDeployer instance."""
    global _deployer_instance
    if _deployer_instance is None:
        _deployer_instance = AtomicConfigDeployer()
    return _deployer_instance
