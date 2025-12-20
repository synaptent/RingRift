#!/usr/bin/env python3
"""
RingRift GPU Cluster Manager - Unified GPU cluster management tool

Commands:
    status      - Quick health check of all GPU nodes
    monitor     - Long-running monitoring with alerts
    deploy      - Sync code to cluster nodes
    benchmark   - Run GPU benchmarks
    jobs        - Manage job queue (list, submit, cancel)

Usage:
    python gpu_cluster_manager.py status
    python gpu_cluster_manager.py monitor --hours 10
    python gpu_cluster_manager.py deploy --group primary_training
    python gpu_cluster_manager.py benchmark --node lambda-gh200-d
    python gpu_cluster_manager.py jobs list
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import urllib.request
import urllib.error

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Using fallback config.")

# Try to use unified modules
try:
    from scripts.lib.hosts import get_hosts, get_hosts_by_group
    USE_UNIFIED_HOSTS = True
except ImportError:
    USE_UNIFIED_HOSTS = False

try:
    from scripts.monitor.alerting import send_alert as unified_send_alert, AlertSeverity
    USE_UNIFIED_ALERTING = True
except ImportError:
    USE_UNIFIED_ALERTING = False

# ============================================================================
# Configuration
# ============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "config" / "cluster.yaml"
JOBS_DB_PATH = Path(__file__).parent.parent / "data" / "jobs.json"

@dataclass
class NodeConfig:
    host: str
    ssh_user: str = "ubuntu"
    gpu_type: str = "unknown"
    gpu_count: int = 1
    vram_gb: int = 0
    batch_multiplier: int = 8
    priority: int = 3
    roles: List[str] = field(default_factory=list)
    status: str = "unknown"
    tailscale_ip: str = ""
    notes: str = ""

@dataclass
class NodeStatus:
    name: str
    online: bool = False
    gpu_util: List[int] = field(default_factory=list)
    gpu_memory_used: List[int] = field(default_factory=list)
    gpu_memory_total: List[int] = field(default_factory=list)
    cpu_load: float = 0.0
    disk_percent: float = 0.0
    memory_percent: float = 0.0
    process_count: int = 0
    error: str = ""
    last_check: datetime = field(default_factory=datetime.now)

@dataclass
class Job:
    id: str
    job_type: str  # selfplay, training, gauntlet, benchmark
    status: str  # pending, running, completed, failed
    node: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

class ClusterConfig:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.nodes: Dict[str, NodeConfig] = {}
        self.groups: Dict[str, List[str]] = {}
        self.alerts = {}

        # Prefer unified hosts module if available
        if USE_UNIFIED_HOSTS:
            self._load_from_unified_hosts()
        else:
            self._load_default_nodes()
            if HAS_YAML and config_path.exists():
                self._load_yaml()

    def _load_from_unified_hosts(self):
        """Load nodes from unified hosts module."""
        for host in get_hosts():
            self.nodes[host.name] = NodeConfig(
                host=host.effective_ssh_host,
                ssh_user=host.ssh_user,
                gpu_type=getattr(host, 'gpu_type', 'unknown'),
                gpu_count=getattr(host, 'gpu_count', 1),
                vram_gb=getattr(host, 'vram_gb', 0),
                batch_multiplier=getattr(host, 'batch_multiplier', 8),
                priority=getattr(host, 'priority', 3),
                roles=list(host.all_roles),
                status=host.status or 'unknown',
                tailscale_ip=host.tailscale_ip or '',
                notes=getattr(host, 'notes', ''),
            )
        # Load groups from hosts module
        for group_name in ['primary_training', 'selfplay_pool', 'gauntlet_pool']:
            group_hosts = get_hosts_by_group(group_name)
            if group_hosts:
                self.groups[group_name] = [h.name for h in group_hosts]
        # Fallback group definitions
        if 'selfplay_pool' not in self.groups:
            self.groups['selfplay_pool'] = list(self.nodes.keys())

    def _load_default_nodes(self):
        """Fallback node configuration."""
        default_nodes = {
            "lambda-gh200-d": ("lambda-gh200-d", "GH200", 96, 64, ["selfplay", "training"]),
            "lambda-gh200-e": ("lambda-gh200-e", "GH200", 96, 64, ["selfplay", "training"]),
            "lambda-gh200-f": ("lambda-gh200-f", "GH200", 96, 64, ["selfplay", "training"]),
            "lambda-gh200-g": ("lambda-gh200-g", "GH200", 96, 64, ["selfplay", "training"]),
            "lambda-gh200-h": ("lambda-gh200-h", "GH200", 96, 64, ["selfplay", "gauntlet"]),
            "lambda-gh200-i": ("lambda-gh200-i", "GH200", 96, 64, ["selfplay", "training"]),
            "lambda-gh200-k": ("lambda-gh200-k", "GH200", 96, 64, ["selfplay", "gauntlet"]),
            "lambda-gh200-l": ("lambda-gh200-l", "GH200", 96, 64, ["selfplay", "gauntlet"]),
            "lambda-h100": ("lambda-h100", "H100", 80, 32, ["selfplay", "training"]),
            "lambda-2xh100": ("lambda-2xh100", "H100", 160, 32, ["selfplay", "training"]),
        }
        for name, (host, gpu_type, vram, mult, roles) in default_nodes.items():
            self.nodes[name] = NodeConfig(
                host=host, gpu_type=gpu_type, vram_gb=vram,
                batch_multiplier=mult, roles=roles, status="active"
            )

        self.groups = {
            "primary_training": ["lambda-gh200-d", "lambda-gh200-e", "lambda-gh200-f", "lambda-gh200-g", "lambda-gh200-i"],
            "selfplay_pool": list(self.nodes.keys()),
            "gauntlet_pool": ["lambda-gh200-h", "lambda-gh200-k", "lambda-gh200-l", "lambda-h100"],
        }

    def _load_yaml(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)

            for name, node_data in data.get("nodes", {}).items():
                self.nodes[name] = NodeConfig(
                    host=node_data.get("host", name),
                    ssh_user=node_data.get("ssh_user", "ubuntu"),
                    gpu_type=node_data.get("gpu_type", "unknown"),
                    gpu_count=node_data.get("gpu_count", 1),
                    vram_gb=node_data.get("vram_gb", 0),
                    batch_multiplier=node_data.get("batch_multiplier", 8),
                    priority=node_data.get("priority", 3),
                    roles=node_data.get("roles", []),
                    status=node_data.get("status", "unknown"),
                    tailscale_ip=node_data.get("tailscale_ip", ""),
                    notes=node_data.get("notes", ""),
                )

            for name, group_data in data.get("groups", {}).items():
                self.groups[name] = group_data.get("nodes", [])

            self.alerts = data.get("alerts", {})
        except Exception as e:
            print(f"Warning: Could not load YAML config: {e}")

    def get_nodes_by_group(self, group: str) -> List[str]:
        return self.groups.get(group, [])

    def get_active_nodes(self) -> List[str]:
        return [name for name, node in self.nodes.items()
                if node.status == "active"]

# ============================================================================
# Alerting
# ============================================================================

class AlertManager:
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.slack_webhook = os.environ.get("RINGRIFT_SLACK_WEBHOOK", "")
        self.discord_webhook = os.environ.get("RINGRIFT_DISCORD_WEBHOOK", "")
        self.alert_history: Dict[str, datetime] = {}
        self.cooldown_minutes = 15

    def send_alert(self, title: str, message: str, severity: str = "warning"):
        """Send alert to configured channels."""
        alert_key = f"{title}:{message[:50]}"

        if alert_key in self.alert_history:
            if datetime.now() - self.alert_history[alert_key] < timedelta(minutes=self.cooldown_minutes):
                return

        self.alert_history[alert_key] = datetime.now()

        severity_label = {"info": "INFO", "warning": "WARN", "critical": "CRIT"}.get(severity, "INFO")
        print(f"\n[{severity_label}] ALERT: {title}")
        print(f"   {message}\n")

        # Use unified alerting module if available
        if USE_UNIFIED_ALERTING:
            level_map = {"info": AlertSeverity.INFO, "warning": AlertSeverity.WARNING, "critical": AlertSeverity.CRITICAL}
            unified_send_alert(title=title, message=message, level=level_map.get(severity, AlertSeverity.WARNING))
            return

        # Fallback to local implementation
        colors = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}
        color = colors.get(severity, "#808080")

        if self.slack_webhook:
            self._send_slack(title, message, color)
        if self.discord_webhook:
            self._send_discord(title, message, color)

    def _send_slack(self, title: str, message: str, color: str):
        try:
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"RingRift Cluster: {title}",
                    "text": message,
                    "ts": int(time.time())
                }]
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.slack_webhook,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")

    def _send_discord(self, title: str, message: str, color: str):
        try:
            color_int = int(color.lstrip("#"), 16)
            payload = {
                "embeds": [{
                    "title": f"RingRift Cluster: {title}",
                    "description": message,
                    "color": color_int,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.discord_webhook,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"Failed to send Discord alert: {e}")

# ============================================================================
# Node Operations
# ============================================================================

def ssh_command(host: str, command: str, timeout: int = 10) -> tuple:
    """Execute SSH command and return (success, output)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes", host, command],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)

def check_node(name: str, config: NodeConfig) -> NodeStatus:
    """Check status of a single node."""
    status = NodeStatus(name=name)
    host = config.host

    success, output = ssh_command(host, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")

    if not success and config.tailscale_ip:
        host = f"{config.ssh_user}@{config.tailscale_ip}"
        success, output = ssh_command(host, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")

    if not success:
        status.error = output or "unreachable"
        return status

    status.online = True

    try:
        for line in output.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                status.gpu_util.append(int(parts[0]))
                status.gpu_memory_used.append(int(parts[1]))
                status.gpu_memory_total.append(int(parts[2]))
    except ValueError:
        status.error = f"parse error: {output}"

    _, proc_output = ssh_command(host, "pgrep -c python 2>/dev/null || echo 0")
    try:
        status.process_count = int(proc_output)
    except ValueError:
        pass

    _, load_output = ssh_command(host, "cat /proc/loadavg | cut -d' ' -f1")
    try:
        status.cpu_load = float(load_output)
    except ValueError:
        pass

    _, disk_output = ssh_command(host, "df / | tail -1 | awk '{print $5}' | tr -d '%'")
    try:
        status.disk_percent = float(disk_output)
    except ValueError:
        pass

    status.last_check = datetime.now()
    return status

def check_all_nodes(config: ClusterConfig, nodes: List[str] = None, parallel: bool = True) -> Dict[str, NodeStatus]:
    """Check status of all nodes."""
    if nodes is None:
        nodes = list(config.nodes.keys())

    results: Dict[str, NodeStatus] = {}

    if parallel:
        threads = []
        for name in nodes:
            if name in config.nodes:
                def check_thread(n=name):
                    results[n] = check_node(n, config.nodes[n])
                t = threading.Thread(target=check_thread)
                threads.append(t)
                t.start()

        for t in threads:
            t.join(timeout=15)
    else:
        for name in nodes:
            if name in config.nodes:
                results[name] = check_node(name, config.nodes[name])

    return results

# ============================================================================
# Job Queue
# ============================================================================

class JobQueue:
    def __init__(self, db_path: Path = JOBS_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Job] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    data = json.load(f)
                for jid, jdata in data.items():
                    self.jobs[jid] = Job(**jdata)
            except Exception:
                pass

    def _save(self):
        data = {}
        for jid, job in self.jobs.items():
            data[jid] = {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "node": job.node,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "params": job.params,
                "result": job.result,
            }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def submit(self, job_type: str, node: str, params: Dict[str, Any] = None) -> Job:
        import uuid
        jid = str(uuid.uuid4())[:8]
        job = Job(
            id=jid,
            job_type=job_type,
            status="pending",
            node=node,
            created_at=datetime.now().isoformat(),
            params=params or {},
        )
        self.jobs[jid] = job
        self._save()
        return job

    def update_status(self, jid: str, status: str, result: Dict = None):
        if jid in self.jobs:
            self.jobs[jid].status = status
            if status == "running":
                self.jobs[jid].started_at = datetime.now().isoformat()
            elif status in ("completed", "failed"):
                self.jobs[jid].completed_at = datetime.now().isoformat()
                if result:
                    self.jobs[jid].result = result
            self._save()

    def list_jobs(self, status: str = None) -> List[Job]:
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def get_pending_for_node(self, node: str) -> List[Job]:
        return [j for j in self.jobs.values()
                if j.node == node and j.status == "pending"]

# ============================================================================
# Commands
# ============================================================================

def cmd_status(args, config: ClusterConfig):
    """Quick status check of all nodes."""
    print(f"\n{'='*70}")
    print(f"  RingRift GPU Cluster Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    if args.group:
        nodes = config.get_nodes_by_group(args.group)
    else:
        nodes = config.get_active_nodes() if not args.all else list(config.nodes.keys())

    results = check_all_nodes(config, nodes)

    online = sum(1 for r in results.values() if r.online)
    high_util = sum(1 for r in results.values() if r.online and r.gpu_util and max(r.gpu_util) > 90)

    print(f"  Online: {online}/{len(results)} | High GPU Util (>90%): {high_util}\n")
    print(f"  {'Node':<18} {'Status':<10} {'GPU%':<8} {'VRAM':<14} {'Procs':<6} {'Load':<6} {'Disk':<6}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*14} {'-'*6} {'-'*6} {'-'*6}")

    for name in sorted(results.keys()):
        status = results[name]
        node_cfg = config.nodes.get(name)
        if status.online:
            gpu_str = "/".join(str(u) for u in status.gpu_util) if status.gpu_util else "-"
            used_gb = sum(status.gpu_memory_used) // 1024 if status.gpu_memory_used else 0
            total_gb = sum(status.gpu_memory_total) // 1024 if status.gpu_memory_total else 0
            vram_str = f"{used_gb}GB/{total_gb}GB"
            state = "OK"
            disk_str = f"{status.disk_percent:.0f}%"
        else:
            gpu_str = "-"
            vram_str = "-"
            state = f"ERR {status.error[:6]}"
            disk_str = "-"

        print(f"  {name:<18} {state:<10} {gpu_str:<8} {vram_str:<14} {status.process_count:<6} {status.cpu_load:<6.1f} {disk_str:<6}")

    print()

def cmd_monitor(args, config: ClusterConfig):
    """Long-running monitoring with alerts."""
    alert_manager = AlertManager(config)
    duration_hours = args.hours
    interval_sec = args.interval
    end_time = datetime.now() + timedelta(hours=duration_hours)

    if args.group:
        nodes = config.get_nodes_by_group(args.group)
    else:
        nodes = config.get_active_nodes()

    print(f"\n{'='*60}")
    print(f"  RingRift GPU Cluster Monitor")
    print(f"  Duration: {duration_hours} hours | Interval: {interval_sec}s")
    print(f"  Monitoring {len(nodes)} nodes until {end_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    offline_since: Dict[str, datetime] = {}
    low_gpu_since: Dict[str, datetime] = {}
    iteration = 0

    try:
        while datetime.now() < end_time:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            results = check_all_nodes(config, nodes)

            online = sum(1 for r in results.values() if r.online)
            total_gpu = sum(sum(r.gpu_util) for r in results.values() if r.online and r.gpu_util)
            gpu_count = sum(len(r.gpu_util) for r in results.values() if r.online and r.gpu_util)
            avg_gpu = total_gpu // gpu_count if gpu_count > 0 else 0

            print(f"[{timestamp}] #{iteration} | Online: {online}/{len(nodes)} | Avg GPU: {avg_gpu}%")

            for name, status in results.items():
                if not status.online:
                    if name not in offline_since:
                        offline_since[name] = datetime.now()
                    elif datetime.now() - offline_since[name] > timedelta(seconds=300):
                        alert_manager.send_alert(
                            f"Node Offline: {name}",
                            f"Node {name} offline >5 min. Error: {status.error}",
                            severity="critical"
                        )
                else:
                    offline_since.pop(name, None)

                    if status.gpu_util and status.process_count > 5:
                        max_gpu = max(status.gpu_util)
                        if max_gpu < 10:
                            if name not in low_gpu_since:
                                low_gpu_since[name] = datetime.now()
                            elif datetime.now() - low_gpu_since[name] > timedelta(seconds=300):
                                alert_manager.send_alert(
                                    f"Low GPU: {name}",
                                    f"{status.process_count} procs but GPU at {max_gpu}%",
                                    severity="warning"
                                )
                        else:
                            low_gpu_since.pop(name, None)

                    if status.disk_percent > 90:
                        alert_manager.send_alert(f"Disk Critical: {name}", f"Disk at {status.disk_percent}%", "critical")
                    elif status.disk_percent > 80:
                        alert_manager.send_alert(f"Disk Warning: {name}", f"Disk at {status.disk_percent}%", "warning")

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

    print(f"\nComplete. Iterations: {iteration}")

def cmd_deploy(args, config: ClusterConfig):
    """Deploy code to cluster nodes."""
    if args.group:
        nodes = config.get_nodes_by_group(args.group)
    elif args.node:
        nodes = [args.node]
    else:
        nodes = config.get_active_nodes()

    print(f"\nDeploying to {len(nodes)} nodes...")
    src_dir = Path(__file__).parent.parent

    for name in nodes:
        if name not in config.nodes:
            print(f"  ERR {name}: not found")
            continue

        node_config = config.nodes[name]
        host = node_config.host
        print(f"  {name}...", end=" ", flush=True)

        try:
            result = subprocess.run([
                "rsync", "-avz", "--exclude", "__pycache__", "--exclude", "*.pyc",
                "--exclude", ".git", "--exclude", "*.log", "--exclude", "training_data",
                f"{src_dir}/", f"{host}:~/RingRift/ai-service/"
            ], capture_output=True, text=True, timeout=120)

            print("OK" if result.returncode == 0 else f"ERR ({result.stderr[:30]})")
        except Exception as e:
            print(f"ERR ({e})")

    print("\nDeploy complete.")

def cmd_benchmark(args, config: ClusterConfig):
    """Run GPU benchmarks."""
    nodes = [args.node] if args.node else config.get_active_nodes()[:3]
    print(f"\nBenchmarking {len(nodes)} nodes...\n")

    for name in nodes:
        if name not in config.nodes:
            continue
        host = config.nodes[name].host
        print(f"--- {name} ({config.nodes[name].gpu_type}) ---")
        success, output = ssh_command(
            host,
            "cd ~/RingRift/ai-service && python scripts/benchmark_gpu_selfplay_cluster.py --games 100 2>&1 | tail -20",
            timeout=180
        )
        print(output if success else f"Failed: {output}")
        print()

def cmd_jobs(args, config: ClusterConfig):
    """Manage job queue."""
    queue = JobQueue()

    if args.action == "list":
        jobs = queue.list_jobs(status=args.status)
        print(f"\n{'ID':<10} {'Type':<12} {'Status':<10} {'Node':<18} {'Created'}")
        print("-" * 70)
        for job in jobs[:20]:
            print(f"{job.id:<10} {job.job_type:<12} {job.status:<10} {job.node:<18} {job.created_at[:19]}")
        print(f"\nTotal: {len(jobs)} jobs")

    elif args.action == "submit":
        if not args.type or not args.node:
            print("Error: --type and --node required")
            return
        job = queue.submit(args.type, args.node, {"games": args.games or 100})
        print(f"Submitted job {job.id}")

    elif args.action == "cancel":
        if not args.job_id:
            print("Error: --job-id required")
            return
        queue.update_status(args.job_id, "cancelled")
        print(f"Cancelled job {args.job_id}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RingRift GPU Cluster Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status
    sp = subparsers.add_parser("status", help="Quick health check")
    sp.add_argument("--group", "-g", help="Node group")
    sp.add_argument("--all", "-a", action="store_true", help="All nodes")

    # Monitor
    sp = subparsers.add_parser("monitor", help="Long-running monitoring")
    sp.add_argument("--hours", "-H", type=float, default=1, help="Duration (hours)")
    sp.add_argument("--interval", "-i", type=int, default=120, help="Interval (seconds)")
    sp.add_argument("--group", "-g", help="Node group")

    # Deploy
    sp = subparsers.add_parser("deploy", help="Deploy code")
    sp.add_argument("--group", "-g", help="Node group")
    sp.add_argument("--node", "-n", help="Specific node")

    # Benchmark
    sp = subparsers.add_parser("benchmark", help="Run benchmarks")
    sp.add_argument("--node", "-n", help="Specific node")

    # Jobs
    sp = subparsers.add_parser("jobs", help="Manage jobs")
    sp.add_argument("action", choices=["list", "submit", "cancel"])
    sp.add_argument("--status", "-s", help="Filter by status")
    sp.add_argument("--type", "-t", help="Job type (selfplay, training, gauntlet)")
    sp.add_argument("--node", "-n", help="Target node")
    sp.add_argument("--games", type=int, help="Number of games")
    sp.add_argument("--job-id", help="Job ID for cancel")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    config = ClusterConfig()

    if args.command == "status":
        cmd_status(args, config)
    elif args.command == "monitor":
        cmd_monitor(args, config)
    elif args.command == "deploy":
        cmd_deploy(args, config)
    elif args.command == "benchmark":
        cmd_benchmark(args, config)
    elif args.command == "jobs":
        cmd_jobs(args, config)

if __name__ == "__main__":
    main()
