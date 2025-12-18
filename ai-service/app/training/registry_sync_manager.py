"""Model Registry Synchronization Manager.

Provides distributed synchronization of model registry state across cluster nodes.
Similar to EloSyncManager but for model lifecycle tracking.

Features:
- Multi-transport failover (Tailscale -> SSH -> HTTP) via ClusterTransport
- Merge-based conflict resolution (union of models)
- Periodic sync with cluster nodes
- Integration with UnifiedAILoop

Usage:
    sync_manager = RegistrySyncManager(registry_path=Path("data/model_registry.db"))
    await sync_manager.initialize()

    # Periodic sync:
    await sync_manager.sync_with_cluster()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Import CircuitBreaker from unified cluster transport layer
from app.coordination.cluster_transport import CircuitBreaker

logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = AI_SERVICE_ROOT / "data" / "model_registry.db"
SYNC_STATE_PATH = AI_SERVICE_ROOT / "data" / "registry_sync_state.json"


@dataclass
class SyncState:
    """Tracks registry synchronization state."""
    last_sync_timestamp: float = 0.0
    local_model_count: int = 0
    local_version_count: int = 0
    synced_nodes: Dict[str, float] = field(default_factory=dict)
    pending_syncs: List[str] = field(default_factory=list)


@dataclass
class NodeInfo:
    """Information about a cluster node for registry sync."""
    hostname: str
    registry_path: str = "ai-service/data/model_registry.db"
    last_seen: float = 0.0
    model_count: int = 0
    version_count: int = 0
    reachable: bool = True
    tailscale_ip: Optional[str] = None
    ssh_port: int = 22


class RegistrySyncManager:
    """
    Manages model registry synchronization across cluster nodes.

    Features:
    - Multi-transport failover (Tailscale -> SSH -> HTTP)
    - Merge-based conflict resolution (preserves all unique models/versions)
    - Local state tracking for offline sync
    - Integration with P2P orchestrator
    """

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        coordinator_host: str = "lambda-h100",
        sync_interval: int = 600,  # 10 minutes
        p2p_url: Optional[str] = None,
    ):
        self.registry_path = Path(registry_path)
        self.coordinator_host = coordinator_host
        self.sync_interval = sync_interval
        self.p2p_url = p2p_url or os.environ.get("P2P_URL", "https://p2p.ringrift.ai")

        self.state = SyncState()
        self.nodes: Dict[str, NodeInfo] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(CircuitBreaker)
        self._sync_lock = asyncio.Lock()
        self._running = False

        # Transport methods in priority order
        self.transport_methods = [
            ("tailscale", self._sync_via_tailscale),
            ("ssh", self._sync_via_ssh),
            ("http", self._sync_via_http),
        ]

        # Event callbacks
        self._on_sync_complete: List[Callable] = []
        self._on_sync_failed: List[Callable] = []

    async def initialize(self):
        """Initialize the sync manager."""
        self._load_state()
        await self._discover_nodes()
        self._update_local_stats()
        logger.info(f"RegistrySyncManager initialized: {self.state.local_model_count} models, "
                    f"{self.state.local_version_count} versions")

    def _load_state(self):
        """Load sync state from disk."""
        if SYNC_STATE_PATH.exists():
            try:
                with open(SYNC_STATE_PATH) as f:
                    data = json.load(f)
                    self.state = SyncState(**data)
            except Exception as e:
                logger.warning(f"Failed to load registry sync state: {e}")

    def _save_state(self):
        """Save sync state to disk."""
        try:
            SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SYNC_STATE_PATH, 'w') as f:
                json.dump({
                    'last_sync_timestamp': self.state.last_sync_timestamp,
                    'local_model_count': self.state.local_model_count,
                    'local_version_count': self.state.local_version_count,
                    'synced_nodes': self.state.synced_nodes,
                    'pending_syncs': self.state.pending_syncs,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save registry sync state: {e}")

    def _update_local_stats(self):
        """Update local registry statistics."""
        if not self.registry_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.registry_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM models")
            self.state.local_model_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM versions")
            self.state.local_version_count = cursor.fetchone()[0]

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update local stats: {e}")

    async def _discover_nodes(self):
        """Discover cluster nodes from hosts config."""
        hosts_config = AI_SERVICE_ROOT / "config" / "remote_hosts.yaml"
        if not hosts_config.exists():
            return

        try:
            import yaml
            with open(hosts_config) as f:
                config = yaml.safe_load(f)

            for host in config.get('hosts', []):
                hostname = host.get('name', host.get('hostname', ''))
                if hostname and hostname != os.uname().nodename:
                    self.nodes[hostname] = NodeInfo(
                        hostname=hostname,
                        tailscale_ip=host.get('tailscale_ip'),
                        ssh_port=host.get('ssh_port', 22),
                    )
        except Exception as e:
            logger.warning(f"Failed to discover nodes: {e}")

    async def sync_with_cluster(self) -> Dict[str, Any]:
        """Synchronize registry with all cluster nodes.

        Returns:
            Summary of sync results
        """
        async with self._sync_lock:
            results = {
                'success': False,
                'nodes_synced': 0,
                'nodes_failed': 0,
                'models_merged': 0,
                'versions_merged': 0,
            }

            for hostname, node in self.nodes.items():
                if not self.circuit_breakers[hostname].can_attempt():
                    continue

                try:
                    sync_result = await self._sync_with_node(hostname, node)
                    if sync_result['success']:
                        results['nodes_synced'] += 1
                        results['models_merged'] += sync_result.get('models_merged', 0)
                        results['versions_merged'] += sync_result.get('versions_merged', 0)
                        self.circuit_breakers[hostname].record_success()
                        self.state.synced_nodes[hostname] = time.time()
                    else:
                        results['nodes_failed'] += 1
                        self.circuit_breakers[hostname].record_failure()
                except Exception as e:
                    logger.error(f"Sync with {hostname} failed: {e}")
                    results['nodes_failed'] += 1
                    self.circuit_breakers[hostname].record_failure()

            self.state.last_sync_timestamp = time.time()
            self._save_state()

            results['success'] = results['nodes_synced'] > 0
            return results

    async def _sync_with_node(self, hostname: str, node: NodeInfo) -> Dict[str, Any]:
        """Sync registry with a single node using transport failover."""
        for transport_name, transport_fn in self.transport_methods:
            try:
                result = await transport_fn(hostname, node)
                if result['success']:
                    logger.info(f"Registry sync with {hostname} succeeded via {transport_name}")
                    return result
            except Exception as e:
                logger.debug(f"Transport {transport_name} failed for {hostname}: {e}")
                continue

        return {'success': False, 'error': 'All transports failed'}

    async def _sync_via_tailscale(self, hostname: str, node: NodeInfo) -> Dict[str, Any]:
        """Sync via Tailscale direct connection."""
        if not node.tailscale_ip:
            return {'success': False, 'error': 'No Tailscale IP'}

        remote_path = f"{node.tailscale_ip}:{node.registry_path}"
        return await self._rsync_and_merge(remote_path, hostname)

    async def _sync_via_ssh(self, hostname: str, node: NodeInfo) -> Dict[str, Any]:
        """Sync via SSH."""
        remote_path = f"{hostname}:{node.registry_path}"
        return await self._rsync_and_merge(remote_path, hostname, ssh_port=node.ssh_port)

    async def _sync_via_http(self, hostname: str, node: NodeInfo) -> Dict[str, Any]:
        """Sync via HTTP from P2P endpoint."""
        try:
            import aiohttp
            url = f"{self.p2p_url}/api/registry/export?host={hostname}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        return {'success': False, 'error': f'HTTP {resp.status}'}

                    data = await resp.json()
                    return await self._merge_registry_data(data, hostname)
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _rsync_and_merge(
        self,
        remote_path: str,
        hostname: str,
        ssh_port: int = 22
    ) -> Dict[str, Any]:
        """Rsync remote registry and merge with local."""
        with tempfile.TemporaryDirectory() as tmpdir:
            remote_db = Path(tmpdir) / "remote_registry.db"

            # Rsync the database
            cmd = [
                "rsync", "-az", "--timeout=30",
                "-e", f"ssh -p {ssh_port} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                remote_path,
                str(remote_db)
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)

            if proc.returncode != 0:
                return {'success': False, 'error': stderr.decode()[:200]}

            if not remote_db.exists():
                return {'success': False, 'error': 'Remote DB not found after rsync'}

            # Merge databases
            return await self._merge_databases(remote_db, hostname)

    async def _merge_databases(self, remote_db: Path, source_hostname: str) -> Dict[str, Any]:
        """Merge remote registry database into local."""
        models_merged = 0
        versions_merged = 0

        try:
            remote_conn = sqlite3.connect(str(remote_db))
            remote_conn.row_factory = sqlite3.Row
            local_conn = sqlite3.connect(str(self.registry_path))

            # Get existing local model IDs
            local_cursor = local_conn.cursor()
            local_cursor.execute("SELECT model_id FROM models")
            local_models = {row[0] for row in local_cursor.fetchall()}

            # Merge models
            remote_cursor = remote_conn.cursor()
            remote_cursor.execute("SELECT * FROM models")
            for row in remote_cursor.fetchall():
                model_id = row['model_id']
                if model_id not in local_models:
                    local_conn.execute("""
                        INSERT INTO models (model_id, name, description, model_type, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (row['model_id'], row['name'], row['description'],
                          row['model_type'], row['created_at'], row['updated_at']))
                    models_merged += 1

            # Get existing local versions
            local_cursor.execute("SELECT model_id, version FROM versions")
            local_versions = {(row[0], row[1]) for row in local_cursor.fetchall()}

            # Merge versions
            remote_cursor.execute("SELECT * FROM versions")
            for row in remote_cursor.fetchall():
                version_key = (row['model_id'], row['version'])
                if version_key not in local_versions:
                    local_conn.execute("""
                        INSERT INTO versions
                        (model_id, version, stage, file_path, file_hash, file_size_bytes,
                         metrics_json, training_config_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row['model_id'], row['version'], row['stage'], row['file_path'],
                          row['file_hash'], row['file_size_bytes'], row['metrics_json'],
                          row['training_config_json'], row['created_at'], row['updated_at']))
                    versions_merged += 1

            local_conn.commit()
            local_conn.close()
            remote_conn.close()

            self._update_local_stats()

            return {
                'success': True,
                'models_merged': models_merged,
                'versions_merged': versions_merged,
                'source': source_hostname
            }

        except Exception as e:
            logger.error(f"Database merge failed: {e}")
            return {'success': False, 'error': str(e)}

    async def _merge_registry_data(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Merge registry data received via HTTP."""
        # Similar to database merge but from JSON data
        models_merged = 0
        versions_merged = 0

        try:
            local_conn = sqlite3.connect(str(self.registry_path))
            cursor = local_conn.cursor()

            # Get existing IDs
            cursor.execute("SELECT model_id FROM models")
            local_models = {row[0] for row in cursor.fetchall()}

            cursor.execute("SELECT model_id, version FROM versions")
            local_versions = {(row[0], row[1]) for row in cursor.fetchall()}

            # Merge models from data
            for model in data.get('models', []):
                if model['model_id'] not in local_models:
                    cursor.execute("""
                        INSERT INTO models (model_id, name, description, model_type, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (model['model_id'], model['name'], model.get('description', ''),
                          model['model_type'], model['created_at'], model['updated_at']))
                    models_merged += 1

            # Merge versions
            for version in data.get('versions', []):
                version_key = (version['model_id'], version['version'])
                if version_key not in local_versions:
                    cursor.execute("""
                        INSERT INTO versions
                        (model_id, version, stage, file_path, file_hash, file_size_bytes,
                         metrics_json, training_config_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (version['model_id'], version['version'], version['stage'],
                          version['file_path'], version['file_hash'], version['file_size_bytes'],
                          version.get('metrics_json', '{}'), version.get('training_config_json', '{}'),
                          version['created_at'], version['updated_at']))
                    versions_merged += 1

            local_conn.commit()
            local_conn.close()

            self._update_local_stats()

            return {
                'success': True,
                'models_merged': models_merged,
                'versions_merged': versions_merged,
                'source': source
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def on_sync_complete(self, callback: Callable):
        """Register callback for sync completion."""
        self._on_sync_complete.append(callback)

    def on_sync_failed(self, callback: Callable):
        """Register callback for sync failure."""
        self._on_sync_failed.append(callback)

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        return {
            'last_sync': self.state.last_sync_timestamp,
            'local_models': self.state.local_model_count,
            'local_versions': self.state.local_version_count,
            'synced_nodes': self.state.synced_nodes,
            'nodes_available': len(self.nodes),
            'circuit_breakers': {
                h: {'state': cb.state, 'failures': cb.failures}
                for h, cb in self.circuit_breakers.items()
            }
        }
