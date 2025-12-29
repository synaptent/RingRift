"""Provider-specific state checkers for NodeAvailabilityDaemon.

Each provider implements a StateChecker subclass that:
1. Queries the provider API for instance states
2. Maps provider-specific states to ProviderInstanceState enum
3. Correlates instances with node names in distributed_hosts.yaml

December 2025: Added TailscaleChecker for mesh connectivity monitoring.
Unlike cloud provider checkers, TailscaleChecker checks actual P2P connectivity.
"""

from app.coordination.node_availability.providers.vast_checker import VastChecker
from app.coordination.node_availability.providers.lambda_checker import LambdaChecker
from app.coordination.node_availability.providers.runpod_checker import RunPodChecker
from app.coordination.node_availability.providers.tailscale_checker import TailscaleChecker

__all__ = [
    "VastChecker",
    "LambdaChecker",
    "RunPodChecker",
    "TailscaleChecker",
]
