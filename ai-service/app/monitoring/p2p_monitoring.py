"""P2P-integrated Monitoring Management for RingRift.

This module provides distributed Prometheus/Grafana management that
integrates with the P2P orchestrator's leader election:

- The cluster leader automatically runs Prometheus and Grafana
- Prometheus config is dynamically generated based on known peers
- When leadership changes, the new leader takes over monitoring
- Multiple fallback hosts ensure monitoring resilience

Usage:
    # In P2P orchestrator, on becoming leader:
    from app.monitoring.p2p_monitoring import MonitoringManager

    manager = MonitoringManager(orchestrator)
    await manager.start_as_leader()

    # On stepping down:
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # Avoid circular imports

logger = logging.getLogger(__name__)

# Import port configuration
try:
    from app.config.ports import P2P_DEFAULT_PORT
except ImportError:
    P2P_DEFAULT_PORT = 8770  # Fallback if ports module unavailable

# Default paths
DEFAULT_PROMETHEUS_PORT = 9090
DEFAULT_GRAFANA_PORT = 3000
DEFAULT_PROMETHEUS_DATA = "/var/lib/prometheus"
DEFAULT_GRAFANA_DATA = "/var/lib/grafana"

# Template for Prometheus config with federation support
PROMETHEUS_CONFIG_TEMPLATE = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'ringrift-p2p-monitor'
    leader_node: '{leader_node}'

rule_files:
  # - /etc/prometheus/alerting-rules.yaml

scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          instance: 'leader'

  # P2P orchestrator on all nodes
  - job_name: 'ringrift-p2p'
    static_configs:
{p2p_targets}

  # Node exporter on all nodes
  - job_name: 'node_exporter'
    static_configs:
{node_exporter_targets}

  # Unified AI loop (leader only by default)
  - job_name: 'ringrift-unified-loop'
    static_configs:
      - targets: ['localhost:9091']
        labels:
          instance: 'leader'
    scrape_interval: 15s

  # Elo metrics exporter (leader only)
  - job_name: 'ringrift-elo-metrics'
    static_configs:
      - targets: ['localhost:9092']
        labels:
          instance: 'leader'
    scrape_interval: 30s

  # Data quality monitor (leader only)
  - job_name: 'ringrift-data-quality'
    static_configs:
      - targets: ['localhost:9093']
        labels:
          instance: 'leader'
    scrape_interval: 60s

  # ==========================================================================
  # Prometheus Federation - scrape metrics from other Prometheus instances
  # This enables distributed monitoring where the leader aggregates metrics
  # from Prometheus instances running on other nodes (backup monitoring hosts)
  # ==========================================================================
  - job_name: 'prometheus-federation'
    honor_labels: true  # Preserve original labels from federated instances
    metrics_path: '/federate'
    params:
      'match[]':
        - '{{job=~"ringrift.*"}}'  # All RingRift metrics
        - '{{job="node_exporter"}}'  # Node hardware metrics
        - '{{__name__=~".*_total|.*_seconds|.*_bytes"}}'  # Standard metrics
    static_configs:
{federation_targets}
    scrape_interval: 30s
    scrape_timeout: 25s
"""


@dataclass
class PeerInfo:
    """Minimal peer info for monitoring config generation."""
    node_id: str
    host: str
    port: int = P2P_DEFAULT_PORT
    gpu_type: str | None = None
    is_alive: bool = True


class MonitoringManager:
    """Manages Prometheus and Grafana services for P2P cluster monitoring.

    When a node becomes the P2P cluster leader, it should start
    monitoring services. When it steps down, the new leader takes over.
    """

    def __init__(
        self,
        node_id: str,
        prometheus_port: int = DEFAULT_PROMETHEUS_PORT,
        grafana_port: int = DEFAULT_GRAFANA_PORT,
        config_dir: str | None = None,
        data_dir: str | None = None,
    ):
        """Initialize the monitoring manager.

        Args:
            node_id: ID of this node
            prometheus_port: Port for Prometheus (default 9090)
            grafana_port: Port for Grafana (default 3000)
            config_dir: Directory for config files (default /etc/prometheus)
            data_dir: Directory for data storage (default /var/lib)
        """
        self.node_id = node_id
        self.prometheus_port = prometheus_port
        self.grafana_port = grafana_port
        self.config_dir = Path(config_dir or "/etc/prometheus")
        self.data_dir = Path(data_dir or "/var/lib")

        self._prometheus_process: subprocess.Popen | None = None
        self._grafana_process: subprocess.Popen | None = None
        self._is_running = False
        self._peers: list[PeerInfo] = []

    def _load_monitoring_hosts(self) -> list[tuple]:
        """Load monitoring hosts from config/distributed_hosts.yaml."""
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"

        if not config_path.exists():
            logger.warning("[Monitoring] No config found, using empty monitoring hosts")
            return []

        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            hosts = []
            for name, info in config.get("hosts", {}).items():
                ip = info.get("tailscale_ip")
                if (ip and info.get("status") == "ready"
                        and (info.get("p2p_voter") or "primary" in info.get("role", ""))):
                    # Prefer hosts with monitoring role or p2p_voter
                    hosts.append((ip, name))

            return hosts[:3]  # Return top 3 as backup monitoring hosts
        except Exception as e:
            logger.warning(f"[Monitoring] Error loading config: {e}")
            return []

    def update_peers(self, peers: list[dict[str, Any]]):
        """Update the list of known peers for config generation.

        Args:
            peers: List of peer dicts with node_id, host, port, etc.
        """
        self._peers = [
            PeerInfo(
                node_id=p.get("node_id", "unknown"),
                host=p.get("host", p.get("tailscale_ip", p.get("ssh_host", "localhost"))),
                port=p.get("p2p_port", p.get("port", 8770)),
                gpu_type=p.get("gpu_type", p.get("gpu")),
                is_alive=p.get("is_alive", True),
            )
            for p in peers
        ]

    def generate_prometheus_config(self) -> str:
        """Generate Prometheus config based on known peers.

        Returns:
            YAML config string
        """
        # Build P2P targets
        p2p_lines = []
        node_exporter_lines = []
        federation_lines = []

        # Known backup monitoring hosts that may have Prometheus running
        # Loaded from config/distributed_hosts.yaml or hardcoded fallback
        backup_monitoring_hosts = self._load_monitoring_hosts()

        for peer in self._peers:
            if not peer.is_alive:
                continue

            host = peer.host
            p2p_port = peer.port
            node_exporter_port = 9100

            labels = f'node="{peer.node_id}"'
            if peer.gpu_type:
                labels += f', gpu_type="{peer.gpu_type}"'

            p2p_lines.append(
                f"      - targets: ['{host}:{p2p_port}']\n"
                f"        labels:\n"
                f"          {labels}"
            )

            node_exporter_lines.append(
                f"      - targets: ['{host}:{node_exporter_port}']\n"
                f"        labels:\n"
                f"          {labels}"
            )

        # Add self if not in peers
        self_in_peers = any(p.node_id == self.node_id for p in self._peers)
        if not self_in_peers:
            p2p_lines.append(
                f"      - targets: ['localhost:8770']\n"
                f"        labels:\n"
                f"          node=\"{self.node_id}\", role=\"leader\""
            )
            node_exporter_lines.append(
                f"      - targets: ['localhost:9100']\n"
                f"        labels:\n"
                f"          node=\"{self.node_id}\", role=\"leader\""
            )

        # Build federation targets (other Prometheus instances for redundancy)
        for host, node_name in backup_monitoring_hosts:
            # Skip self - we don't federate from ourselves
            if node_name == self.node_id:
                continue
            federation_lines.append(
                f"      - targets: ['{host}:9090']\n"
                f"        labels:\n"
                f"          source_node=\"{node_name}\""
            )

        p2p_targets = "\n".join(p2p_lines) if p2p_lines else "      - targets: ['localhost:8770']"
        node_exporter_targets = "\n".join(node_exporter_lines) if node_exporter_lines else "      - targets: ['localhost:9100']"
        federation_targets = "\n".join(federation_lines) if federation_lines else "      - targets: []  # No federation targets"

        return PROMETHEUS_CONFIG_TEMPLATE.format(
            leader_node=self.node_id,
            p2p_targets=p2p_targets,
            node_exporter_targets=node_exporter_targets,
            federation_targets=federation_targets,
        )

    async def start_as_leader(self) -> bool:
        """Start monitoring services when becoming the cluster leader.

        Returns:
            True if services started successfully
        """
        if self._is_running:
            logger.info(f"[Monitoring] Services already running on {self.node_id}")
            return True

        logger.info(f"[Monitoring] Starting services on leader {self.node_id}")

        try:
            # Generate and write Prometheus config
            config = self.generate_prometheus_config()
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.config_dir / "prometheus.yml"

            # Write atomically
            tmp_path = config_path.with_suffix(".tmp")
            tmp_path.write_text(config)
            tmp_path.rename(config_path)
            logger.info(f"[Monitoring] Wrote Prometheus config to {config_path}")

            # Try to start via systemd first (preferred on Linux servers)
            prometheus_started = await self._start_prometheus_systemd()
            if not prometheus_started:
                # Fall back to direct process
                prometheus_started = await self._start_prometheus_process()

            grafana_started = await self._start_grafana_systemd()
            if not grafana_started:
                grafana_started = await self._start_grafana_process()

            self._is_running = prometheus_started

            if self._is_running:
                logger.info(f"[Monitoring] Services started successfully on {self.node_id}")
            else:
                logger.warning(f"[Monitoring] Failed to start services on {self.node_id}")

            return self._is_running

        except Exception as e:
            logger.error(f"[Monitoring] Failed to start services: {e}")
            return False

    async def _start_prometheus_systemd(self) -> bool:
        """Try to start Prometheus via systemd."""
        try:
            result = subprocess.run(
                ["sudo", "systemctl", "start", "prometheus"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("[Monitoring] Started Prometheus via systemd")
                return True
            return False
        except Exception as e:
            logger.debug(f"[Monitoring] systemd prometheus not available: {e}")
            return False

    async def _start_prometheus_process(self) -> bool:
        """Start Prometheus as a subprocess."""
        try:
            prometheus_bin = shutil.which("prometheus")
            if not prometheus_bin:
                # Try common locations
                for path in ["/usr/local/bin/prometheus", "/usr/bin/prometheus"]:
                    if os.path.exists(path):
                        prometheus_bin = path
                        break

            if not prometheus_bin:
                logger.warning("[Monitoring] Prometheus binary not found")
                return False

            data_dir = self.data_dir / "prometheus"
            data_dir.mkdir(parents=True, exist_ok=True)

            self._prometheus_process = subprocess.Popen(
                [
                    prometheus_bin,
                    f"--config.file={self.config_dir / 'prometheus.yml'}",
                    f"--storage.tsdb.path={data_dir}",
                    f"--web.listen-address=0.0.0.0:{self.prometheus_port}",
                    "--storage.tsdb.retention.time=7d",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait a moment to check if it started
            await asyncio.sleep(2)
            if self._prometheus_process.poll() is None:
                logger.info(f"[Monitoring] Started Prometheus process (PID {self._prometheus_process.pid})")
                return True
            return False

        except Exception as e:
            logger.error(f"[Monitoring] Failed to start Prometheus process: {e}")
            return False

    async def _start_grafana_systemd(self) -> bool:
        """Try to start Grafana via systemd."""
        try:
            result = subprocess.run(
                ["sudo", "systemctl", "start", "grafana-server"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("[Monitoring] Started Grafana via systemd")
                return True
            return False
        except Exception as e:
            logger.debug(f"[Monitoring] systemd grafana not available: {e}")
            return False

    async def _start_grafana_process(self) -> bool:
        """Start Grafana as a subprocess (usually not needed if installed)."""
        # Grafana is typically run via systemd, so we just check if it's running
        try:
            result = subprocess.run(
                ["pgrep", "-f", "grafana"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info("[Monitoring] Grafana is already running")
                return True
            logger.warning("[Monitoring] Grafana not running and systemd start failed")
            return False
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def stop(self):
        """Stop monitoring services (when stepping down from leader)."""
        logger.info(f"[Monitoring] Stopping services on {self.node_id}")

        # Stop processes if we started them
        if self._prometheus_process:
            self._prometheus_process.terminate()
            try:
                self._prometheus_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._prometheus_process.kill()
            self._prometheus_process = None

        if self._grafana_process:
            self._grafana_process.terminate()
            try:
                self._grafana_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._grafana_process.kill()
            self._grafana_process = None

        self._is_running = False
        logger.info(f"[Monitoring] Services stopped on {self.node_id}")

    async def reload_config(self):
        """Reload Prometheus config after peer changes."""
        if not self._is_running:
            return

        logger.info("[Monitoring] Reloading Prometheus config")

        # Regenerate config
        config = self.generate_prometheus_config()
        config_path = self.config_dir / "prometheus.yml"

        tmp_path = config_path.with_suffix(".tmp")
        tmp_path.write_text(config)
        tmp_path.rename(config_path)

        # Try to reload via API
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:{self.prometheus_port}/-/reload") as resp:
                    if resp.status == 200:
                        logger.info("[Monitoring] Prometheus config reloaded via API")
                        return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

        # Try systemd reload
        try:
            subprocess.run(
                ["sudo", "systemctl", "reload", "prometheus"],
                capture_output=True,
                timeout=10,
            )
            logger.info("[Monitoring] Prometheus config reloaded via systemd")
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError):
            # Send SIGHUP to process
            if self._prometheus_process:
                self._prometheus_process.send_signal(1)  # SIGHUP
                logger.info("[Monitoring] Sent SIGHUP to Prometheus process")

    @property
    def is_running(self) -> bool:
        """Check if monitoring services are running."""
        return self._is_running


def install_prometheus(version: str = "2.47.0") -> bool:
    """Install Prometheus on the system.

    Args:
        version: Prometheus version to install

    Returns:
        True if installation succeeded
    """
    try:
        import platform

        arch = platform.machine()
        if arch == "x86_64":
            arch = "amd64"
        elif arch == "aarch64":
            arch = "arm64"

        url = f"https://github.com/prometheus/prometheus/releases/download/v{version}/prometheus-{version}.linux-{arch}.tar.gz"

        # Download and extract
        subprocess.run([
            "wget", "-q", "-O", "/tmp/prometheus.tar.gz", url
        ], check=True)

        subprocess.run([
            "tar", "xzf", "/tmp/prometheus.tar.gz", "-C", "/tmp"
        ], check=True)

        # Install
        prometheus_dir = f"/tmp/prometheus-{version}.linux-{arch}"
        subprocess.run([
            "sudo", "cp", f"{prometheus_dir}/prometheus", "/usr/local/bin/"
        ], check=True)
        subprocess.run([
            "sudo", "cp", f"{prometheus_dir}/promtool", "/usr/local/bin/"
        ], check=True)

        # Create user and directories
        subprocess.run(["sudo", "useradd", "--no-create-home", "--shell", "/bin/false", "prometheus"], check=False)
        subprocess.run(["sudo", "mkdir", "-p", "/etc/prometheus", "/var/lib/prometheus"], check=True)
        subprocess.run(["sudo", "chown", "prometheus:prometheus", "/var/lib/prometheus"], check=True)

        logger.info(f"[Monitoring] Installed Prometheus {version}")
        return True

    except Exception as e:
        logger.error(f"[Monitoring] Failed to install Prometheus: {e}")
        return False


def create_prometheus_systemd_service() -> bool:
    """Create systemd service file for Prometheus.

    Returns:
        True if service created successfully
    """
    service_content = """[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \\
    --config.file=/etc/prometheus/prometheus.yml \\
    --storage.tsdb.path=/var/lib/prometheus \\
    --web.listen-address=0.0.0.0:9090 \\
    --storage.tsdb.retention.time=30d \\
    --web.enable-lifecycle
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

    try:
        # Write service file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.service', delete=False) as f:
            f.write(service_content)
            tmp_path = f.name

        subprocess.run([
            "sudo", "mv", tmp_path, "/etc/systemd/system/prometheus.service"
        ], check=True)

        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "prometheus"], check=True)

        logger.info("[Monitoring] Created Prometheus systemd service")
        return True

    except Exception as e:
        logger.error(f"[Monitoring] Failed to create Prometheus service: {e}")
        return False
