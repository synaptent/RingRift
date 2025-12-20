"""AI Agent registry for tournament system."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HeuristicWeights,
)


class AgentType(str, Enum):
    """Types of AI agents."""
    MINIMAX = "minimax"
    MCTS = "mcts"
    RANDOM = "random"
    HYBRID = "hybrid"
    NEURAL = "neural"  # Neural network policy/value model


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
    metadata: dict[str, Any] = field(default_factory=dict)
    model_path: str | None = None  # Path to neural network weights for NEURAL agents
    board_type: str | None = None  # Board type the model was trained for
    num_players: int | None = None  # Number of players the model supports

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent to dictionary."""
        data = {
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
        if self.model_path:
            data["model_path"] = self.model_path
        if self.board_type:
            data["board_type"] = self.board_type
        if self.num_players:
            data["num_players"] = self.num_players
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AIAgent:
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
            model_path=data.get("model_path"),
            board_type=data.get("board_type"),
            num_players=data.get("num_players"),
        )


class AIAgentRegistry:
    """Registry for managing AI agents."""

    def __init__(self, registry_path: Path | None = None):
        """Initialize agent registry.

        Args:
            registry_path: Path to JSON file for persistent storage.
        """
        self._agents: dict[str, AIAgent] = {}
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

    def get(self, agent_id: str) -> AIAgent | None:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> list[AIAgent]:
        """List all registered agents."""
        return list(self._agents.values())

    def list_agent_ids(self) -> list[str]:
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

        with open(self._registry_path) as f:
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

    def create_from_model(
        self,
        model_path: str,
        agent_id: str | None = None,
        name: str | None = None,
        board_type: str | None = None,
        num_players: int | None = None,
        description: str = "",
    ) -> AIAgent:
        """Create and register a neural network agent from model file.

        Args:
            model_path: Path to the .pth model file
            agent_id: Optional agent ID (derived from model filename if not provided)
            name: Optional display name (derived from agent_id if not provided)
            board_type: Board type the model was trained for (e.g., "hex8", "square8")
            num_players: Number of players the model supports
            description: Optional description

        Returns:
            The registered AIAgent
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Derive agent_id from filename if not provided
        if agent_id is None:
            # e.g., "ringrift_hex8_2p_v8.pth" -> "hex8_2p_v8_nn"
            stem = path.stem
            agent_id = f"{stem}_nn"

        # Derive name from agent_id if not provided
        if name is None:
            name = agent_id.replace("_", " ").title()

        # Try to extract board_type and num_players from filename
        if board_type is None or num_players is None:
            stem = path.stem.lower()
            # Common patterns: hex8_2p, square8_3p, hexagonal_2p
            import re
            match = re.search(r'(hex8|hexagonal|square\d+)_(\d+)p', stem)
            if match:
                if board_type is None:
                    board_type = match.group(1)
                if num_players is None:
                    num_players = int(match.group(2))

        agent = AIAgent(
            agent_id=agent_id,
            name=name,
            agent_type=AgentType.NEURAL,
            model_path=str(path.resolve()),
            board_type=board_type,
            num_players=num_players,
            description=description or f"Neural network agent from {path.name}",
        )
        self.register(agent)
        return agent
