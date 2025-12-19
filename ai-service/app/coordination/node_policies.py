"""Node Work Assignment Policies.

Loads and applies work assignment policies to prevent misrouting work
to inappropriate nodes (e.g., CPU-bound work on GPU-heavy nodes).

Usage:
    from app.coordination.node_policies import NodePolicyManager, get_policy_manager

    # Get singleton instance
    manager = get_policy_manager()

    # Check if work type is allowed on a node
    if manager.is_work_allowed("lambda-gh200-a", "cpu_cmaes"):
        start_cmaes(node)

    # Get best work type for a node
    work_type = manager.get_best_work_type("lambda-gh200-a", ["training", "cpu_cmaes", "selfplay"])

    # Get policy for a node
    policy = manager.get_node_policy("lambda-gh200-a")
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.utils.yaml_utils import safe_load_yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "node_policies.yaml"


@dataclass
class NodePolicy:
    """Policy for a node or node type."""
    name: str
    patterns: List[str] = field(default_factory=list)
    allowed: Set[str] = field(default_factory=set)
    denied: Set[str] = field(default_factory=set)
    priorities: Dict[str, int] = field(default_factory=dict)

    def is_allowed(self, work_type: str) -> bool:
        """Check if a work type is allowed."""
        if work_type in self.denied:
            return False
        if self.allowed and work_type not in self.allowed:
            return False
        return True

    def get_priority(self, work_type: str) -> int:
        """Get priority for a work type (higher = prefer)."""
        return self.priorities.get(work_type, 50)

    def matches_node(self, node_id: str) -> bool:
        """Check if this policy matches a node ID."""
        for pattern in self.patterns:
            if fnmatch.fnmatch(node_id.lower(), pattern.lower()):
                return True
        return False


class NodePolicyManager:
    """Manages work assignment policies for nodes."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.policies: Dict[str, NodePolicy] = {}
        self.overrides: Dict[str, NodePolicy] = {}
        self.default_policy: NodePolicy = NodePolicy(
            name="default",
            allowed={"training", "gpu_cmaes", "tournament", "selfplay"},
            priorities={"training": 100, "gpu_cmaes": 90, "tournament": 80, "selfplay": 70}
        )
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config = safe_load_yaml(self.config_path, default=None, log_errors=True)
        if config is None:
            if not self.config_path.exists():
                logger.warning(f"Node policies config not found: {self.config_path}")
            return

        # Load default policy
        if "default" in config:
            self.default_policy = self._parse_policy("default", config["default"])

        # Load named policies
        for name in ["gpu_heavy", "gpu_medium", "apple_silicon", "cpu_only", "vast_gpu"]:
            if name in config:
                self.policies[name] = self._parse_policy(name, config[name])

        # Load overrides for specific nodes
        if "overrides" in config and config["overrides"]:
            for node_id, policy_config in config["overrides"].items():
                self.overrides[node_id.lower()] = self._parse_policy(
                    f"override:{node_id}", policy_config
                )

        logger.info(f"Loaded {len(self.policies)} node policies, {len(self.overrides)} overrides")

    def _parse_policy(self, name: str, config: Dict[str, Any]) -> NodePolicy:
        """Parse a policy from config dict."""
        return NodePolicy(
            name=name,
            patterns=config.get("patterns", []),
            allowed=set(config.get("allowed", [])),
            denied=set(config.get("denied", [])),
            priorities=config.get("priorities", {}),
        )

    def get_node_policy(self, node_id: str) -> NodePolicy:
        """Get the effective policy for a node."""
        node_lower = node_id.lower()

        # Check overrides first
        if node_lower in self.overrides:
            return self.overrides[node_lower]

        # Check named policies by pattern match
        for policy in self.policies.values():
            if policy.matches_node(node_id):
                return policy

        # Return default
        return self.default_policy

    def is_work_allowed(self, node_id: str, work_type: str) -> bool:
        """Check if a work type is allowed on a node."""
        policy = self.get_node_policy(node_id)
        return policy.is_allowed(work_type)

    def get_allowed_work_types(self, node_id: str) -> Set[str]:
        """Get all allowed work types for a node."""
        policy = self.get_node_policy(node_id)
        if policy.allowed:
            return policy.allowed - policy.denied
        # If no explicit allowed list, return all except denied
        all_types = {"training", "gpu_cmaes", "cpu_cmaes", "tournament", "gauntlet", "selfplay", "data_merge"}
        return all_types - policy.denied

    def get_denied_work_types(self, node_id: str) -> Set[str]:
        """Get all denied work types for a node."""
        policy = self.get_node_policy(node_id)
        return policy.denied

    def get_best_work_type(self, node_id: str, available_types: List[str]) -> Optional[str]:
        """Get the best work type for a node from available options.

        Returns the highest priority allowed work type, or None if none allowed.
        """
        policy = self.get_node_policy(node_id)

        # Filter to allowed types and sort by priority
        allowed = [
            (work_type, policy.get_priority(work_type))
            for work_type in available_types
            if policy.is_allowed(work_type)
        ]

        if not allowed:
            return None

        # Sort by priority (descending) and return best
        allowed.sort(key=lambda x: -x[1])
        return allowed[0][0]

    def get_priority(self, node_id: str, work_type: str) -> int:
        """Get priority of a work type for a node."""
        policy = self.get_node_policy(node_id)
        return policy.get_priority(work_type)

    def should_preempt(self, node_id: str, current_work: str, new_work: str) -> bool:
        """Check if new work should preempt current work on a node.

        Returns True if new work has higher priority and current work is allowed to be preempted.
        """
        policy = self.get_node_policy(node_id)

        # Can't preempt if new work isn't allowed
        if not policy.is_allowed(new_work):
            return False

        current_priority = policy.get_priority(current_work)
        new_priority = policy.get_priority(new_work)

        # Preempt if new work has significantly higher priority (>20 points)
        return new_priority > current_priority + 20

    def reload(self) -> None:
        """Reload configuration from file."""
        self.policies.clear()
        self.overrides.clear()
        self._load_config()


# Singleton instance
_policy_manager: Optional[NodePolicyManager] = None


def get_policy_manager() -> NodePolicyManager:
    """Get the singleton NodePolicyManager instance."""
    global _policy_manager
    if _policy_manager is None:
        _policy_manager = NodePolicyManager()
    return _policy_manager


def is_work_allowed(node_id: str, work_type: str) -> bool:
    """Convenience function to check if work is allowed on a node."""
    return get_policy_manager().is_work_allowed(node_id, work_type)


def get_best_work_type(node_id: str, available_types: List[str]) -> Optional[str]:
    """Convenience function to get best work type for a node."""
    return get_policy_manager().get_best_work_type(node_id, available_types)
