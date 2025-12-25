"""AWS EC2 provider manager.

Manages AWS EC2 instances via the AWS CLI.
Primarily used for staging/proxy instances.

Requires: AWS CLI installed and configured
    https://aws.amazon.com/cli/
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    """Convert AWS API state to unified state."""
    state_map = {
        "pending": InstanceState.STARTING,
        "running": InstanceState.RUNNING,
        "shutting-down": InstanceState.STOPPING,
        "terminated": InstanceState.TERMINATED,
        "stopping": InstanceState.STOPPING,
        "stopped": InstanceState.STOPPED,
    }
    return state_map.get(api_state.lower(), InstanceState.UNKNOWN)


# Known AWS instance types
AWS_INSTANCE_TYPES = {
    "t3.micro": {"cpu": 2, "memory_gb": 1, "hourly_cost": 0.0104},
    "t3.small": {"cpu": 2, "memory_gb": 2, "hourly_cost": 0.0208},
    "t3.medium": {"cpu": 2, "memory_gb": 4, "hourly_cost": 0.0416},
    "t3.large": {"cpu": 2, "memory_gb": 8, "hourly_cost": 0.0832},
    "t3.xlarge": {"cpu": 4, "memory_gb": 16, "hourly_cost": 0.1664},
    "c5.large": {"cpu": 2, "memory_gb": 4, "hourly_cost": 0.085},
    "c5.xlarge": {"cpu": 4, "memory_gb": 8, "hourly_cost": 0.17},
    "m5.large": {"cpu": 2, "memory_gb": 8, "hourly_cost": 0.096},
    "m5.xlarge": {"cpu": 4, "memory_gb": 16, "hourly_cost": 0.192},
    "p3.2xlarge": {"cpu": 8, "memory_gb": 61, "hourly_cost": 3.06, "gpu": "V100", "gpu_count": 1},
    "p4d.24xlarge": {"cpu": 96, "memory_gb": 1152, "hourly_cost": 32.77, "gpu": "A100", "gpu_count": 8},
    "g4dn.xlarge": {"cpu": 4, "memory_gb": 16, "hourly_cost": 0.526, "gpu": "T4", "gpu_count": 1},
}


class AWSManager(ProviderManager):
    """Manage AWS EC2 instances via AWS CLI.

    Usage:
        manager = AWSManager()
        instances = await manager.list_instances()

        for inst in instances:
            health = await manager.check_health(inst)
            print(f"{inst.name}: {health.message}")
    """

    provider = Provider.AWS

    def __init__(self, region: str | None = None):
        """Initialize AWS manager.

        Args:
            region: AWS region. If not provided, uses default from CLI config.
        """
        self.region = region
        self._aws_available: bool | None = None

    async def _check_aws_available(self) -> bool:
        """Check if AWS CLI is available."""
        if self._aws_available is not None:
            return self._aws_available

        try:
            proc = await asyncio.create_subprocess_exec(
                "aws", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._aws_available = proc.returncode == 0
        except FileNotFoundError:
            self._aws_available = False

        if not self._aws_available:
            logger.warning("AWS CLI not available")

        return self._aws_available

    async def _run_aws(self, *args: str) -> dict[str, Any] | list | None:
        """Run AWS CLI command and return JSON output."""
        if not await self._check_aws_available():
            return None

        cmd = ["aws", *args, "--output", "json"]
        if self.region:
            cmd.extend(["--region", self.region])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.error(f"AWS CLI error: {stderr.decode()}")
                return None

            return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            logger.error("AWS CLI command timed out")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AWS output: {e}")
            return None
        except Exception as e:
            logger.error(f"AWS CLI command failed: {e}")
            return None

    async def _run_aws_action(self, *args: str) -> bool:
        """Run AWS CLI command that doesn't return JSON."""
        if not await self._check_aws_available():
            return False

        cmd = ["aws", *args]
        if self.region:
            cmd.extend(["--region", self.region])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                logger.error(f"AWS action error: {stderr.decode()}")
                return False

            return True
        except Exception as e:
            logger.error(f"AWS action failed: {e}")
            return False

    async def list_instances(
        self,
        filters: dict[str, str] | None = None,
    ) -> list[ProviderInstance]:
        """List EC2 instances.

        Args:
            filters: Optional filters like {"tag:Environment": "production"}
        """
        args = ["ec2", "describe-instances"]

        if filters:
            filter_args = []
            for key, value in filters.items():
                filter_args.append(f"Name={key},Values={value}")
            if filter_args:
                args.extend(["--filters", *filter_args])

        result = await self._run_aws(*args)
        if not result:
            return []

        instances = []
        for reservation in result.get("Reservations", []):
            for inst_data in reservation.get("Instances", []):
                instance = self._parse_instance(inst_data)
                if instance:
                    instances.append(instance)

        logger.info(f"AWS: found {len(instances)} instances")
        return instances

    async def get_instance(self, instance_id: str) -> ProviderInstance | None:
        """Get details of a specific instance."""
        result = await self._run_aws(
            "ec2", "describe-instances",
            "--instance-ids", instance_id,
        )
        if not result:
            return None

        for reservation in result.get("Reservations", []):
            for inst_data in reservation.get("Instances", []):
                return self._parse_instance(inst_data)

        return None

    def _parse_instance(self, data: dict[str, Any]) -> ProviderInstance | None:
        """Parse AWS API response into ProviderInstance."""
        try:
            instance_id = data.get("InstanceId", "")
            instance_type = data.get("InstanceType", "")

            # Get name from tags
            name = instance_id
            tailscale_ip = None
            tags = {t["Key"]: t["Value"] for t in data.get("Tags", [])}
            if "Name" in tags:
                name = tags["Name"]
            if "tailscale_ip" in tags:
                tailscale_ip = tags["tailscale_ip"]

            # Get IP addresses
            public_ip = data.get("PublicIpAddress")
            private_ip = data.get("PrivateIpAddress")

            # Get type info
            type_info = AWS_INSTANCE_TYPES.get(instance_type, {})

            # Get state
            state_data = data.get("State", {})
            state_name = state_data.get("Name", "unknown")

            return ProviderInstance(
                instance_id=instance_id,
                provider=Provider.AWS,
                name=name,
                public_ip=public_ip,
                private_ip=private_ip,
                tailscale_ip=tailscale_ip,
                state=_parse_instance_state(state_name),
                gpu_type=type_info.get("gpu"),
                gpu_count=type_info.get("gpu_count", 0),
                cpu_count=type_info.get("cpu", 0),
                memory_gb=type_info.get("memory_gb", 0),
                hourly_cost=type_info.get("hourly_cost", 0.0),
                metadata={
                    "ssh_user": "ubuntu",
                    "ssh_key": "~/.ssh/id_cluster",
                    "instance_type": instance_type,
                    "availability_zone": data.get("Placement", {}).get("AvailabilityZone", ""),
                    "tags": tags,
                    "launch_time": data.get("LaunchTime", ""),
                },
            )
        except Exception as e:
            logger.error(f"Failed to parse AWS instance: {e}")
            return None

    async def check_health(self, instance: ProviderInstance) -> HealthCheckResult:
        """Check health of an AWS instance."""
        # First check SSH
        ssh_result = await self.check_ssh_connectivity(instance)
        if not ssh_result.healthy:
            return ssh_result

        # Then check P2P daemon
        p2p_result = await self.check_p2p_health(instance)
        if not p2p_result.healthy:
            return p2p_result

        # Then check Tailscale
        ts_result = await self.check_tailscale(instance)

        all_healthy = ssh_result.healthy and p2p_result.healthy and ts_result.healthy
        return HealthCheckResult(
            healthy=all_healthy,
            check_type="composite",
            message="All checks passed" if all_healthy else "Some checks failed",
            details={
                "ssh": ssh_result.healthy,
                "p2p": p2p_result.healthy,
                "tailscale": ts_result.healthy,
            },
        )

    async def reboot_instance(self, instance_id: str) -> bool:
        """Reboot an EC2 instance."""
        success = await self._run_aws_action(
            "ec2", "reboot-instances",
            "--instance-ids", instance_id,
        )
        if success:
            logger.info(f"AWS: rebooting instance {instance_id}")
        return success

    async def start_instance(self, instance_id: str) -> bool:
        """Start a stopped EC2 instance."""
        success = await self._run_aws_action(
            "ec2", "start-instances",
            "--instance-ids", instance_id,
        )
        if success:
            logger.info(f"AWS: starting instance {instance_id}")
        return success

    async def stop_instance(self, instance_id: str) -> bool:
        """Stop an EC2 instance."""
        success = await self._run_aws_action(
            "ec2", "stop-instances",
            "--instance-ids", instance_id,
        )
        if success:
            logger.info(f"AWS: stopping instance {instance_id}")
        return success

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an EC2 instance."""
        success = await self._run_aws_action(
            "ec2", "terminate-instances",
            "--instance-ids", instance_id,
        )
        if success:
            logger.info(f"AWS: terminated instance {instance_id}")
        return success

    async def launch_instance(self, config: dict[str, Any]) -> str | None:
        """Launch a new EC2 instance.

        Args:
            config: Launch configuration with keys:
                - image_id: AMI ID (required)
                - instance_type: e.g., "t3.medium" (default)
                - key_name: SSH key pair name
                - security_group_ids: list of security group IDs
                - subnet_id: VPC subnet ID
                - tags: dict of tags

        Returns:
            Instance ID if successful, None otherwise.
        """
        image_id = config.get("image_id")
        if not image_id:
            logger.error("AWS: image_id required")
            return None

        args = [
            "ec2", "run-instances",
            "--image-id", image_id,
            "--instance-type", config.get("instance_type", "t3.medium"),
            "--count", "1",
        ]

        if key_name := config.get("key_name"):
            args.extend(["--key-name", key_name])

        if sg_ids := config.get("security_group_ids"):
            args.extend(["--security-group-ids", *sg_ids])

        if subnet_id := config.get("subnet_id"):
            args.extend(["--subnet-id", subnet_id])

        if tags := config.get("tags"):
            tag_specs = [f"Key={k},Value={v}" for k, v in tags.items()]
            args.extend([
                "--tag-specifications",
                f"ResourceType=instance,Tags=[{','.join(['{' + t + '}' for t in tag_specs])}]",
            ])

        result = await self._run_aws(*args)
        if result:
            instances = result.get("Instances", [])
            if instances:
                instance_id = instances[0].get("InstanceId")
                if instance_id:
                    logger.info(f"AWS: launched instance {instance_id}")
                    return instance_id

        return None

    async def add_tag(
        self,
        instance_id: str,
        key: str,
        value: str,
    ) -> bool:
        """Add/update a tag on an instance."""
        return await self._run_aws_action(
            "ec2", "create-tags",
            "--resources", instance_id,
            "--tags", f"Key={key},Value={value}",
        )

    async def restart_p2p_daemon(self, instance: ProviderInstance) -> bool:
        """Restart P2P daemon on instance via SSH."""
        cmd = """
cd ~/ringrift/ai-service && \
pkill -f 'app.p2p.orchestrator' 2>/dev/null; \
sleep 2; \
source venv/bin/activate && \
nohup python -m app.p2p.orchestrator > logs/p2p.log 2>&1 &
echo "P2P daemon restarted"
"""
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=30)
        if code == 0 and "restarted" in stdout:
            logger.info(f"AWS: restarted P2P on {instance.name}")
            return True
        logger.error(f"AWS: failed to restart P2P on {instance.name}: {stderr}")
        return False

    async def restart_tailscale(self, instance: ProviderInstance) -> bool:
        """Restart Tailscale daemon on instance."""
        cmd = "sudo systemctl restart tailscaled && sleep 2 && tailscale status"
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=30)
        if code == 0:
            logger.info(f"AWS: restarted Tailscale on {instance.name}")
            return True
        logger.error(f"AWS: failed to restart Tailscale on {instance.name}: {stderr}")
        return False


async def test_aws():
    """Test AWS CLI connectivity."""
    manager = AWSManager()

    print("Testing AWS CLI...")
    instances = await manager.list_instances()

    if instances:
        print(f"\nFound {len(instances)} instances:")
        for inst in instances:
            print(f"  {inst}")
    else:
        print("No instances found (or AWS CLI not configured)")


if __name__ == "__main__":
    asyncio.run(test_aws())
