"""Vast.ai cloud provider manager.

Manages Vast.ai GPU instances via the vastai CLI.
Supports auto-scaling, instance lifecycle, and data sync.

Requires: vastai CLI installed and configured
    pip install vastai && vastai set api-key <key>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.providers.base import (
    HealthCheckResult,
    InstanceState,
    Provider,
    ProviderInstance,
    ProviderManager,
)

logger = logging.getLogger(__name__)


def _parse_instance_state(api_state: str) -> InstanceState:
    """Convert Vast API state to unified state."""
    state_map = {
        "running": InstanceState.RUNNING,
        "loading": InstanceState.STARTING,
        "starting": InstanceState.STARTING,
        "exited": InstanceState.STOPPED,
        "stopped": InstanceState.STOPPED,
        "destroying": InstanceState.TERMINATED,
    }
    return state_map.get(api_state.lower(), InstanceState.UNKNOWN)


@dataclass
class VastOffer:
    """Vast.ai offer for renting."""

    offer_id: int
    gpu_name: str
    num_gpus: int
    gpu_memory_gb: int
    cpu_cores: int
    ram_gb: int
    hourly_cost: float
    reliability: float
    location: str


# GPU to board type mapping for workload assignment
GPU_BOARD_MAPPING = {
    # Small GPUs (<=8GB) - fast hex8 games
    "RTX 3070": "hex8",
    "RTX 2060S": "hex8",
    "RTX 2060 SUPER": "hex8",
    "RTX 3060 Ti": "hex8",
    "RTX 2080 Ti": "hex8",
    "RTX 3060": "hex8",
    # Mid-range GPUs (12-16GB)
    "RTX 4060 Ti": "square8",
    "RTX 4080S": "square19",
    "RTX 4080 SUPER": "square19",
    "RTX 5080": "square19",
    # High-end GPUs (24GB+)
    "A40": "hexagonal",
    "RTX 5090": "hexagonal",
    "RTX 5070": "hexagonal",
    "A10": "hexagonal",
    "H100": "hexagonal",
}


class VastManager(ProviderManager):
    """Manage Vast.ai instances via CLI.

    Usage:
        manager = VastManager()
        instances = await manager.list_instances()

        for inst in instances:
            health = await manager.check_health(inst)
            print(f"{inst.name}: {health.message}")
    """

    provider = Provider.VAST

    def __init__(self):
        """Initialize Vast manager."""
        self._vastai_cmd: str | None = None
        self._vastai_available: bool | None = None

    async def _find_vastai_cmd(self) -> str | None:
        """Find vastai executable."""
        if self._vastai_cmd:
            return self._vastai_cmd

        paths = [
            "vastai",
            os.path.expanduser("~/.local/bin/vastai"),
            "/usr/local/bin/vastai",
        ]

        # Add pyenv paths
        pyenv_root = os.environ.get("PYENV_ROOT", os.path.expanduser("~/.pyenv"))
        if os.path.exists(pyenv_root):
            # Find all pyenv versions
            versions_dir = Path(pyenv_root) / "versions"
            if versions_dir.exists():
                for version_dir in versions_dir.iterdir():
                    if version_dir.is_dir():
                        paths.append(str(version_dir / "bin" / "vastai"))

        for path in paths:
            try:
                proc = await asyncio.create_subprocess_exec(
                    path, "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=5)
                if proc.returncode == 0:
                    self._vastai_cmd = path
                    return path
            except (FileNotFoundError, asyncio.TimeoutError):
                continue

        return None

    async def _check_vastai_available(self) -> bool:
        """Check if vastai CLI is available."""
        if self._vastai_available is not None:
            return self._vastai_available

        self._vastai_cmd = await self._find_vastai_cmd()
        self._vastai_available = self._vastai_cmd is not None

        if not self._vastai_available:
            logger.warning("vastai CLI not available")

        return self._vastai_available

    async def _run_vastai(self, *args: str) -> list | dict | None:
        """Run vastai command and return JSON output."""
        if not await self._check_vastai_available():
            return None

        cmd = [self._vastai_cmd, *args, "--raw"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.error(f"vastai error: {stderr.decode()}")
                return None

            return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            logger.error("vastai command timed out")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vastai output: {e}")
            return None
        except Exception as e:
            logger.error(f"vastai command failed: {e}")
            return None

    async def _run_vastai_action(self, *args: str, timeout: int = 60) -> bool:
        """Run vastai command that doesn't return JSON."""
        if not await self._check_vastai_available():
            return False

        cmd = [self._vastai_cmd, *args]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            if proc.returncode != 0:
                logger.error(f"vastai action error: {stderr.decode()}")
                return False

            return True
        except Exception as e:
            logger.error(f"vastai action failed: {e}")
            return False

    async def list_instances(self) -> list[ProviderInstance]:
        """List all Vast instances."""
        result = await self._run_vastai("show", "instances")
        if not result:
            return []

        instances = []
        for inst_data in result:
            instance = self._parse_instance(inst_data)
            if instance:
                instances.append(instance)

        logger.info(f"Vast: found {len(instances)} instances")
        return instances

    async def get_instance(self, instance_id: str) -> ProviderInstance | None:
        """Get details of a specific instance."""
        instances = await self.list_instances()
        for inst in instances:
            if inst.instance_id == instance_id:
                return inst
        return None

    def _parse_instance(self, data: dict[str, Any]) -> ProviderInstance | None:
        """Parse vastai output into ProviderInstance."""
        try:
            instance_id = str(data.get("id", ""))
            gpu_name = data.get("gpu_name", "unknown")
            num_gpus = data.get("num_gpus", 1)

            # Determine board type for this GPU
            board_type = GPU_BOARD_MAPPING.get(gpu_name, "square8")

            # Vast instances use SSH tunnels
            ssh_host = data.get("ssh_host", "")
            ssh_port = data.get("ssh_port", 22)

            # Calculate GPU memory
            gpu_memory_per = data.get("gpu_ram", 0) / 1024  # MB to GB
            gpu_memory = int(gpu_memory_per * num_gpus)

            return ProviderInstance(
                instance_id=instance_id,
                provider=Provider.VAST,
                name=f"vast-{instance_id[:8]}",
                public_ip=ssh_host.replace("ssh", "") if ssh_host.startswith("ssh") else ssh_host,
                ssh_port=int(ssh_port) if ssh_port else 22,
                state=_parse_instance_state(data.get("actual_status", "unknown")),
                gpu_type=f"{num_gpus}x {gpu_name}" if num_gpus > 1 else gpu_name,
                gpu_count=num_gpus,
                gpu_memory_gb=gpu_memory,
                cpu_count=int(data.get("cpu_cores_effective", 0)),
                memory_gb=int(data.get("cpu_ram", 0) / 1024),  # MB to GB
                hourly_cost=data.get("dph_total", 0) or 0,
                metadata={
                    "ssh_user": "root",
                    "ssh_host": ssh_host,
                    "ssh_port": ssh_port,
                    "board_type": board_type,
                    "gpu_name": gpu_name,
                    "num_gpus": num_gpus,
                    "reliability": data.get("reliability", 0),
                    "disk_space": data.get("disk_space", 0),
                    "inet_up": data.get("inet_up", 0),
                    "inet_down": data.get("inet_down", 0),
                },
            )
        except Exception as e:
            logger.error(f"Failed to parse Vast instance: {e}")
            return None

    async def run_ssh_command(
        self,
        instance: ProviderInstance,
        command: str,
        timeout: int = 30,
    ) -> tuple[int, str, str]:
        """Run SSH command on Vast instance.

        Overrides base to handle Vast's SSH tunnel format.
        """
        ssh_host = instance.metadata.get("ssh_host", "")
        ssh_port = instance.metadata.get("ssh_port", instance.ssh_port)

        if not ssh_host:
            return -1, "", "No SSH host available"

        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-p", str(ssh_port),
            f"root@{ssh_host}",
            command,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            return proc.returncode or 0, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            return -1, "", f"SSH timeout after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    async def check_health(self, instance: ProviderInstance) -> HealthCheckResult:
        """Check health of a Vast instance."""
        import time

        start = time.time()

        # First check SSH
        code, stdout, stderr = await self.run_ssh_command(instance, "echo ok", timeout=15)
        if code != 0 or "ok" not in stdout:
            return HealthCheckResult(
                healthy=False,
                check_type="ssh",
                message=f"SSH failed: {stderr or 'no response'}",
                latency_ms=(time.time() - start) * 1000,
            )

        # Check for running workers
        code, stdout, _ = await self.run_ssh_command(
            instance,
            "pgrep -fa 'diverse_selfplay|selfplay|generate_data' | grep -v pgrep | wc -l",
            timeout=15,
        )
        workers_running = 0
        if code == 0:
            try:
                workers_running = int(stdout.strip())
            except ValueError:
                pass

        # Check for games
        code, stdout, _ = await self.run_ssh_command(
            instance,
            """python3 -c "
import sqlite3
import glob
total = 0
for path in glob.glob('/root/ringrift/ai-service/data/games/*.db'):
    try:
        conn = sqlite3.connect(path)
        count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        conn.close()
        total += count
    except: pass
print(total)
" """,
            timeout=20,
        )
        games_count = 0
        if code == 0:
            try:
                games_count = int(stdout.strip())
            except ValueError:
                pass

        latency = (time.time() - start) * 1000

        if workers_running == 0:
            return HealthCheckResult(
                healthy=False,
                check_type="workers",
                message="No selfplay workers running",
                latency_ms=latency,
                details={"workers": 0, "games": games_count},
            )

        return HealthCheckResult(
            healthy=True,
            check_type="composite",
            message=f"{workers_running} workers, {games_count} games",
            latency_ms=latency,
            details={"workers": workers_running, "games": games_count},
        )

    async def reboot_instance(self, instance_id: str) -> bool:
        """Reboot a Vast instance (restart container)."""
        # Vast doesn't have a direct reboot - we stop and start
        if not await self.stop_instance(instance_id):
            return False

        await asyncio.sleep(5)

        return await self.start_instance(instance_id)

    async def start_instance(self, instance_id: str) -> bool:
        """Start a stopped Vast instance."""
        success = await self._run_vastai_action("start", "instance", instance_id)
        if success:
            logger.info(f"Vast: started instance {instance_id}")
        return success

    async def stop_instance(self, instance_id: str) -> bool:
        """Stop a Vast instance."""
        success = await self._run_vastai_action("stop", "instance", instance_id)
        if success:
            logger.info(f"Vast: stopped instance {instance_id}")
        return success

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate (destroy) a Vast instance."""
        success = await self._run_vastai_action("destroy", "instance", instance_id)
        if success:
            logger.info(f"Vast: destroyed instance {instance_id}")
        return success

    async def search_offers(
        self,
        gpu_name: str | None = None,
        max_price: float | None = None,
        min_reliability: float = 0.95,
        num_gpus: int = 1,
        limit: int = 10,
    ) -> list[VastOffer]:
        """Search for available GPU offers.

        Args:
            gpu_name: GPU type to filter by
            max_price: Maximum hourly price
            min_reliability: Minimum reliability score (0-1)
            num_gpus: Number of GPUs required
            limit: Maximum offers to return
        """
        query_parts = [f"reliability>{min_reliability}", "rentable=true"]

        if gpu_name:
            query_parts.append(f"gpu_name={gpu_name}")
        if max_price:
            query_parts.append(f"dph<{max_price}")
        if num_gpus > 1:
            query_parts.append(f"num_gpus>={num_gpus}")

        query = " ".join(query_parts)

        # Build command
        cmd = [self._vastai_cmd, "search", "offers", query, "-o", "dph", "--raw"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.error(f"Search failed: {stderr.decode()}")
                return []

            offers_data = json.loads(stdout.decode())

            offers = []
            for data in offers_data[:limit]:
                try:
                    offers.append(VastOffer(
                        offer_id=data.get("id", 0),
                        gpu_name=data.get("gpu_name", "unknown"),
                        num_gpus=data.get("num_gpus", 1),
                        gpu_memory_gb=int(data.get("gpu_ram", 0) / 1024),
                        cpu_cores=int(data.get("cpu_cores_effective", 0)),
                        ram_gb=int(data.get("cpu_ram", 0) / 1024),
                        hourly_cost=data.get("dph_total", 0) or 0,
                        reliability=data.get("reliability", 0),
                        location=data.get("geolocation", ""),
                    ))
                except Exception:
                    continue

            return offers
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def launch_instance(self, config: dict[str, Any]) -> str | None:
        """Launch a new Vast instance.

        Args:
            config: Launch configuration with keys:
                - offer_id: Offer ID to rent (required)
                - disk_gb: Disk space in GB (default 50)
                - image: Docker image (default pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime)
                - onstart_cmd: Command to run on start

        Returns:
            Instance ID if successful, None otherwise.
        """
        offer_id = config.get("offer_id")
        if not offer_id:
            logger.error("Vast: offer_id required")
            return None

        args = [
            "create", "instance", str(offer_id),
            "--image", config.get("image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"),
            "--disk", str(config.get("disk_gb", 50)),
            "--ssh",
        ]

        if onstart := config.get("onstart_cmd"):
            args.extend(["--onstart-cmd", onstart])

        # Run and capture instance ID
        cmd = [self._vastai_cmd, *args]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                logger.error(f"Launch failed: {stderr.decode()}")
                return None

            # Parse instance ID from output
            output = stdout.decode() + stderr.decode()
            for word in output.split():
                if word.isdigit() and len(word) > 6:
                    logger.info(f"Vast: launched instance {word}")
                    return word

            return None
        except Exception as e:
            logger.error(f"Launch error: {e}")
            return None

    async def get_current_hourly_cost(self) -> float:
        """Get total hourly cost of all running instances."""
        instances = await self.list_instances()
        running = [i for i in instances if i.state == InstanceState.RUNNING]
        return sum(i.hourly_cost for i in running)

    async def restart_workers(
        self,
        instance: ProviderInstance,
        board_type: str | None = None,
        sync_code: bool = True,
    ) -> bool:
        """Restart selfplay workers on instance.

        Args:
            instance: Instance to restart workers on
            board_type: Board type to use (default from GPU mapping)
            sync_code: Whether to git pull first
        """
        if board_type is None:
            board_type = instance.metadata.get("board_type", "square8")

        gpu_name = instance.metadata.get("gpu_name", "unknown")
        logger.info(f"Vast: restarting workers on {instance.name} ({gpu_name}) with {board_type}")

        # Sync git repo first
        if sync_code:
            code, stdout, _ = await self.run_ssh_command(
                instance,
                "cd /root/ringrift && git fetch origin && git reset --hard origin/main && git log -1 --oneline",
                timeout=60,
            )
            if code == 0:
                logger.info(f"  Updated to: {stdout.strip()}")

        # Kill existing workers
        await self.run_ssh_command(
            instance,
            "pkill -f 'generate_data|selfplay' || true",
            timeout=15,
        )

        # Determine num_games based on board type
        num_games = {"hex8": 2000, "square8": 1500, "hexagonal": 500}.get(board_type, 1000)

        # Start new workers
        code, stdout, stderr = await self.run_ssh_command(
            instance,
            f"""cd /root/ringrift/ai-service &&
mkdir -p data/games logs models &&
source venv/bin/activate 2>/dev/null || true &&
PYTHONPATH=. RINGRIFT_DISABLE_TORCH_COMPILE=1 nohup python3 -m app.training.generate_data \
    --board-type {board_type} --num-games {num_games} \
    --engine descent \
    --record-db data/games/selfplay_{board_type}_{instance.name}.db \
    > logs/selfplay_{board_type}.log 2>&1 &
sleep 2 && pgrep -f generate_data | head -1
""",
            timeout=90,
        )

        if code == 0 and stdout.strip():
            logger.info(f"  Started PID {stdout.strip()}")
            return True

        logger.error(f"  Failed: {stderr}")
        return False

    async def sync_data(
        self,
        instance: ProviderInstance,
        dest_host: str,
        dest_path: str,
        dest_user: str = "ubuntu",
    ) -> int:
        """Sync game data from instance to destination.

        Returns number of games synced.
        """
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix=f"vast_sync_{instance.name}_")

        try:
            # Find DBs on instance
            code, stdout, _ = await self.run_ssh_command(
                instance,
                "find /root/ringrift/ai-service/data -name '*.db' -type f 2>/dev/null",
                timeout=15,
            )
            if code != 0:
                return 0

            db_paths = [p.strip() for p in stdout.split("\n") if p.strip() and ".db" in p]
            if not db_paths:
                return 0

            total_games = 0
            ssh_host = instance.metadata.get("ssh_host", "")
            ssh_port = instance.metadata.get("ssh_port", 22)

            for remote_path in db_paths:
                local_path = os.path.join(temp_dir, os.path.basename(remote_path))

                # Download
                proc = await asyncio.create_subprocess_exec(
                    "scp",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=10",
                    "-P", str(ssh_port),
                    f"root@{ssh_host}:{remote_path}",
                    local_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=60)

                if proc.returncode == 0 and os.path.exists(local_path):
                    # Count games
                    import sqlite3
                    try:
                        conn = sqlite3.connect(local_path)
                        count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                        conn.close()
                        total_games += count
                    except Exception:
                        pass

            return total_games
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def get_autoscaler_groups(self) -> list[dict]:
        """Get autoscaler groups."""
        result = await self._run_vastai("show", "autoscalers")
        return result if result else []

    async def create_autoscaler_group(
        self,
        name: str,
        gpu_name: str,
        target_instances: int = 5,
        min_instances: int = 2,
        max_instances: int = 10,
        max_price: float = 0.10,
        disk_gb: int = 50,
    ) -> str | None:
        """Create an autoscaler group for automatic instance management."""
        search_query = f"gpu_name={gpu_name} rentable=true dph<{max_price}"

        cmd = [
            self._vastai_cmd, "create", "autoscaler",
            "--search-query", search_query,
            "--image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            "--disk", str(disk_gb),
            "--min-instances", str(min_instances),
            "--max-instances", str(max_instances),
            "--target-instances", str(target_instances),
            "--onstart-cmd", "apt-get update && apt-get install -y git curl",
            "--name", name,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                logger.error(f"Create autoscaler failed: {stderr.decode()}")
                return None

            output = stdout.decode() + stderr.decode()
            logger.info(f"Created autoscaler group '{name}': {output}")

            # Parse group ID
            for word in output.split():
                if word.isdigit():
                    return word

            return output
        except Exception as e:
            logger.error(f"Create autoscaler error: {e}")
            return None

    async def update_autoscaler_group(
        self,
        group_id: str,
        target_instances: int | None = None,
        min_instances: int | None = None,
        max_instances: int | None = None,
    ) -> bool:
        """Update autoscaler group settings."""
        args = ["change", "autoscaler", group_id]

        if target_instances is not None:
            args.extend(["--target-instances", str(target_instances)])
        if min_instances is not None:
            args.extend(["--min-instances", str(min_instances)])
        if max_instances is not None:
            args.extend(["--max-instances", str(max_instances)])

        return await self._run_vastai_action(*args)


async def test_vast():
    """Test Vast CLI connectivity."""
    manager = VastManager()

    print("Testing Vast CLI...")
    instances = await manager.list_instances()

    if instances:
        print(f"\nFound {len(instances)} instances:")
        for inst in instances:
            print(f"  {inst}")
            health = await manager.check_health(inst)
            print(f"    Health: {health.message}")
    else:
        print("No instances found (or vastai not configured)")

    # Show cost
    cost = await manager.get_current_hourly_cost()
    print(f"\nTotal hourly cost: ${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(test_vast())
