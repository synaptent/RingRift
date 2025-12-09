"""AI Agent registry for tournament system."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
    HeuristicWeights,
)


class AgentType(str, Enum):
    """Types of AI agents."""
    MINIMAX = "minimax"
    MCTS = "mcts"
    RANDOM = "random"
    HYBRID = "hybrid"


@dataclass
class AIAgent:
    """Represents an AI agent configuration for tournament play."""

    agent_id: str
    name: str
    agent_type: AgentType
    weights: HeuristicWeights = field(default_factory=lambda: BASE_V1_BALANCED_WEIGHTS.copy())
    search_depth: int = 3
    mcts_simulations: int = 100
    description: str = ""
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "weights": self.weights,
            "search_depth": self.search_depth,
            "mcts_simulations": self.mcts_simulations,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIAgent":
        """Deserialize agent from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            agent_type=AgentType(data["agent_type"]),
            weights=data.get("weights", BASE_V1_BALANCED_WEIGHTS.copy()),
            search_depth=data.get("search_depth", 3),
            mcts_simulations=data.get("mcts_simulations", 100),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )


class AIAgentRegistry:
    """Registry for managing AI agents."""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize agent registry.

        Args:
            registry_path: Path to JSON file for persistent storage.
        """
        self._agents: Dict[str, AIAgent] = {}
        self._registry_path = registry_path

        if registry_path and registry_path.exists():
            self.load()
        else:
            self._register_builtin_agents()

    def _register_builtin_agents(self) -> None:
        """Register built-in reference agents."""
        self.register(AIAgent(
            agent_id="baseline_v1",
            name="Baseline V1",
            agent_type=AgentType.MINIMAX,
            weights=BASE_V1_BALANCED_WEIGHTS.copy(),
            search_depth=3,
            description="Baseline minimax agent with balanced weights",
            version="1.0",
        ))

        self.register(AIAgent(
            agent_id="random",
            name="Random Player",
            agent_type=AgentType.RANDOM,
            search_depth=0,
            description="Uniformly random move selection",
            version="1.0",
        ))

        self.register(AIAgent(
            agent_id="aggressive_v1",
            name="Aggressive V1",
            agent_type=AgentType.MINIMAX,
            weights={
                **BASE_V1_BALANCED_WEIGHTS,
                "WEIGHT_STACK_CONTROL": 8.0,
                "WEIGHT_CAPTURE_VALUE": 6.0,
                "WEIGHT_MOBILITY": 2.0,
            },
            search_depth=3,
            description="Aggressive minimax agent favoring captures",
            version="1.0",
        ))

        self.register(AIAgent(
            agent_id="defensive_v1",
            name="Defensive V1",
            agent_type=AgentType.MINIMAX,
            weights={
                **BASE_V1_BALANCED_WEIGHTS,
                "WEIGHT_STACK_CONTROL": 3.0,
                "WEIGHT_RING_SAFETY": 8.0,
                "WEIGHT_MOBILITY": 6.0,
            },
            search_depth=3,
            description="Defensive minimax agent prioritizing safety",
            version="1.0",
        ))

        self.register(AIAgent(
            agent_id="deep_search_v1",
            name="Deep Search V1",
            agent_type=AgentType.MINIMAX,
            weights=BASE_V1_BALANCED_WEIGHTS.copy(),
            search_depth=5,
            description="Deep search minimax (slower but stronger)",
            version="1.0",
        ))

    def register(self, agent: AIAgent) -> None:
        """Register an agent."""
        self._agents[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent. Returns True if found."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def get(self, agent_id: str) -> Optional[AIAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AIAgent]:
        """List all registered agents."""
        return list(self._agents.values())

    def list_agent_ids(self) -> List[str]:
        """List all agent IDs."""
        return list(self._agents.keys())

    def save(self) -> None:
        """Save registry to file."""
        if not self._registry_path:
            raise ValueError("No registry path configured")

        data = {
            "agents": [agent.to_dict() for agent in self._agents.values()]
        }

        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load registry from file."""
        if not self._registry_path or not self._registry_path.exists():
            self._register_builtin_agents()
            return

        with open(self._registry_path, 'r') as f:
            data = json.load(f)

        self._agents.clear()
        for agent_data in data.get("agents", []):
            agent = AIAgent.from_dict(agent_data)
            self._agents[agent.agent_id] = agent

    def create_from_weights(
        self,
        agent_id: str,
        name: str,
        weights: HeuristicWeights,
        agent_type: AgentType = AgentType.MINIMAX,
        search_depth: int = 3,
        description: str = "",
    ) -> AIAgent:
        """Create and register a new agent from weight configuration."""
        agent = AIAgent(
            agent_id=agent_id,
            name=name,
            agent_type=agent_type,
            weights=weights,
            search_depth=search_depth,
            description=description,
        )
        self.register(agent)
        return agent
