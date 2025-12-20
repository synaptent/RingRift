"""Unified AI Factory for RingRift.

This module provides a centralized factory for creating AI instances,
consolidating the logic previously scattered across main.py and various
orchestrator scripts. All AI creation should go through this factory
to ensure consistent configuration and proper initialization.

Usage:
    from app.ai.factory import AIFactory, get_difficulty_profile

    # Create AI from difficulty level
    ai = AIFactory.create_from_difficulty(difficulty=5, player_number=1)

    # Create AI with explicit type and config
    ai = AIFactory.create(
        ai_type=AIType.MCTS,
        player_number=1,
        config=AIConfig(difficulty=5, think_time=5000),
    )

    # Get canonical difficulty profile
    profile = get_difficulty_profile(7)

    # Register custom AI implementation
    AIFactory.register("custom_ai", CustomAIClass)
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

from app.models.core import AIConfig, AIType

if TYPE_CHECKING:
    from app.ai.base import BaseAI

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Type definitions
# -----------------------------------------------------------------------------


class DifficultyProfile(TypedDict):
    """Canonical difficulty profile for a single ladder level.

    This mapping is the Python source of truth for how difficulty 1-11 map
    onto concrete AI configurations. TypeScript mirrors the same ladder in
    its own shared module so that lobby UIs, matchmaking, and the server
    agree on exactly which AI strengths are offered.
    """
    ai_type: AIType
    randomness: float
    think_time_ms: int
    profile_id: str
    use_neural_net: bool


# -----------------------------------------------------------------------------
# Canonical difficulty profiles (1-11)
# -----------------------------------------------------------------------------

# NOTE: think_time_ms is passed to the AI as a soft search-budget hint but
# must not be used to delay after a move has been selected.
#
# Benchmark Results (Dec 2025):
# - 2P: Gumbel+NN >> MCTS+NN > Descent+NN
# - 3P: Gumbel+NN >> (MaxN ≈ BRS ≈ MCTS+NN ≈ Descent+NN)
# - 4P: Gumbel+NN >> MaxN > BRS > MCTS+NN
#
# For 2P games, the default ladder below applies.
# For 3-4P games, use MULTIPLAYER_DIFFICULTY_OVERRIDES which swaps
# Minimax levels for MaxN/BRS (better suited for multi-player).
CANONICAL_DIFFICULTY_PROFILES: dict[int, DifficultyProfile] = {
    1: {
        # Beginner: pure random baseline
        "ai_type": AIType.RANDOM,
        "randomness": 0.5,
        "think_time_ms": 150,
        "profile_id": "v2-random-1",
        "use_neural_net": False,
    },
    2: {
        # Easy: shallow heuristic play with noticeable randomness
        "ai_type": AIType.HEURISTIC,
        "randomness": 0.3,
        "think_time_ms": 200,
        "profile_id": "v2-heuristic-2",
        "use_neural_net": False,
    },
    3: {
        # Lower-mid: Policy-only (neural policy without search)
        # Uses NN-trained move distribution, weaker than search-based
        "ai_type": AIType.POLICY_ONLY,
        "randomness": 0.15,
        "think_time_ms": 500,
        "profile_id": "v2-policy-3",
        "use_neural_net": True,
    },
    4: {
        # Mid-low: Minimax with heuristic evaluation (2P small boards only)
        # For large boards or 3-4P, use MULTIPLAYER_DIFFICULTY_OVERRIDES
        "ai_type": AIType.MINIMAX,
        "randomness": 0.08,
        "think_time_ms": 2000,
        "profile_id": "v2-minimax-4",
        "use_neural_net": False,
    },
    5: {
        # Mid: Minimax with NNUE neural evaluation (2P small boards)
        # For large boards: Descent+NN; For 3-4P: MaxN
        "ai_type": AIType.MINIMAX,
        "randomness": 0.05,
        "think_time_ms": 3000,
        "profile_id": "v2-minimax-5-nnue",
        "use_neural_net": True,
    },
    6: {
        # Upper-mid: Descent with neural guidance
        "ai_type": AIType.DESCENT,
        "randomness": 0.02,
        "think_time_ms": 4500,
        "profile_id": "v2-descent-6-neural",
        "use_neural_net": True,
    },
    7: {
        # High: MCTS with neural guidance
        "ai_type": AIType.MCTS,
        "randomness": 0.0,
        "think_time_ms": 6000,
        "profile_id": "v2-mcts-7-neural",
        "use_neural_net": True,
    },
    8: {
        # Expert: MCTS with neural guidance and larger budget
        "ai_type": AIType.MCTS,
        "randomness": 0.0,
        "think_time_ms": 9000,
        "profile_id": "v2-mcts-8-neural",
        "use_neural_net": True,
    },
    9: {
        # Master: Gumbel MCTS (strongest algorithm per benchmarks)
        "ai_type": AIType.GUMBEL_MCTS,
        "randomness": 0.0,
        "think_time_ms": 12000,
        "profile_id": "v2-gumbel-9-master",
        "use_neural_net": True,
    },
    10: {
        # Grandmaster: Gumbel MCTS with extended search
        "ai_type": AIType.GUMBEL_MCTS,
        "randomness": 0.0,
        "think_time_ms": 18000,
        "profile_id": "v2-gumbel-10-grandmaster",
        "use_neural_net": True,
    },
    11: {
        # Ultimate: Gumbel MCTS with maximum think time
        # Nearly unbeatable by humans
        "ai_type": AIType.GUMBEL_MCTS,
        "randomness": 0.0,
        "think_time_ms": 60000,  # 60 seconds per move
        "profile_id": "v2-gumbel-11-ultimate",
        "use_neural_net": True,
    },
    # -------------------------------------------------------------------------
    # Experimental/Research AI Slots (12-15)
    # These are novel approaches not yet proven superior to production ladder
    # -------------------------------------------------------------------------
    12: {
        # EBMO: Energy-Based Move Optimization
        # Uses gradient descent on action embeddings - novel energy-based approach
        # Trained on MCTS-labeled data, optimizes action manifold at inference time
        "ai_type": AIType.EBMO,
        "randomness": 0.1,
        "think_time_ms": 1000,
        "profile_id": "v3-ebmo-12-experimental",
        "use_neural_net": True,
    },
    13: {
        # GMO: Gradient Move Optimization
        # Entropy-guided gradient ascent in move embedding space
        # Explores action space via information-theoretic optimization
        "ai_type": AIType.GMO,
        "randomness": 0.1,
        "think_time_ms": 1500,
        "profile_id": "v3-gmo-13-experimental",
        "use_neural_net": True,
    },
    14: {
        # IG-GMO: Information-Gain GMO
        # Mutual information-based exploration with GNN state encoding
        # Most experimental of the gradient-based approaches
        "ai_type": AIType.IG_GMO,
        "randomness": 0.1,
        "think_time_ms": 2000,
        "profile_id": "v3-iggmo-14-experimental",
        "use_neural_net": True,
    },
    15: {
        # GPU Minimax: GPU-accelerated minimax with batched evaluation
        # Same algorithm as D5 but optimized for CUDA/MPS hardware
        "ai_type": AIType.GPU_MINIMAX,
        "randomness": 0.05,
        "think_time_ms": 3000,
        "profile_id": "v3-gpuminimax-15-experimental",
        "use_neural_net": True,
    },
}

# Overrides for 3-4 player games where MaxN/BRS outperform Minimax
# Benchmarks show: MaxN > BRS > MCTS in 4P; MaxN ≈ BRS ≈ MCTS in 3P
MULTIPLAYER_DIFFICULTY_OVERRIDES: dict[int, DifficultyProfile] = {
    4: {
        # For 3-4P: BRS (Best Reply Search) - faster than MaxN, good for 3P
        "ai_type": AIType.BRS,
        "randomness": 0.08,
        "think_time_ms": 2000,
        "profile_id": "v2-brs-4-mp",
        "use_neural_net": False,
    },
    5: {
        # For 3-4P: MaxN (better than BRS for 4P, equal for 3P)
        "ai_type": AIType.MAXN,
        "randomness": 0.05,
        "think_time_ms": 3000,
        "profile_id": "v2-maxn-5-mp",
        "use_neural_net": False,
    },
}

# Overrides for large boards where Minimax is too slow
LARGE_BOARD_DIFFICULTY_OVERRIDES: dict[int, DifficultyProfile] = {
    4: {
        # For large boards: Descent+NN (Minimax too slow)
        "ai_type": AIType.DESCENT,
        "randomness": 0.08,
        "think_time_ms": 2500,
        "profile_id": "v2-descent-4-large",
        "use_neural_net": True,
    },
    5: {
        # For large boards: Descent+NN with larger budget
        "ai_type": AIType.DESCENT,
        "randomness": 0.05,
        "think_time_ms": 3500,
        "profile_id": "v2-descent-5-large",
        "use_neural_net": True,
    },
}

# Difficulty level descriptions for UI/documentation
# Adapted based on player count and board size
DIFFICULTY_DESCRIPTIONS: dict[int, str] = {
    1: "Beginner - Pure random moves",
    2: "Easy - Simple heuristic with randomness",
    3: "Lower-mid - Policy-only (neural network move selection)",
    4: "Mid-low - Minimax/BRS (2P) or BRS/Descent (3-4P/large boards)",
    5: "Mid - Minimax+NNUE (2P small) / MaxN (3-4P) / Descent+NN (large)",
    6: "Upper-mid - Descent with neural guidance",
    7: "High - MCTS with neural guidance",
    8: "Expert - MCTS with larger search budget",
    9: "Master - Gumbel MCTS (strongest algorithm)",
    10: "Grandmaster - Gumbel MCTS with extended search",
    11: "Ultimate - Gumbel MCTS with 60s think time (nearly unbeatable)",
    # Experimental AI slots (not shown in standard UI)
    12: "Experimental - EBMO (Energy-Based Move Optimization)",
    13: "Experimental - GMO (Gradient Move Optimization)",
    14: "Experimental - IG-GMO (Information-Gain GMO)",
    15: "Experimental - GPU Minimax (GPU-accelerated)",
}

# Board types considered "large" (Minimax too slow)
LARGE_BOARD_TYPES = {"square19", "hexagonal", "hex"}

# Experimental EBMO override for D3
# Enable via environment variable: RINGRIFT_USE_EBMO=1
# EBMO uses gradient descent on action embeddings - novel energy-based approach
EBMO_DIFFICULTY_OVERRIDE: DifficultyProfile = {
    "ai_type": AIType.EBMO,
    "randomness": 0.1,
    "think_time_ms": 1000,
    "profile_id": "v3-ebmo-3-experimental",
    "use_neural_net": True,
}


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_difficulty_profile(
    difficulty: int,
    *,
    num_players: int = 2,
    board_type: str | None = None,
) -> DifficultyProfile:
    """Return the difficulty profile for the given level, player count, and board.

    Difficulty is clamped into [1, 11] so that out-of-range values still map
    to a well-defined profile instead of silently diverging between callers.

    The profile may be adjusted based on:
    - num_players: For 3-4 players, uses MaxN/BRS instead of Minimax at D4-5
    - board_type: For large boards, uses Descent instead of Minimax at D4-5
    - RINGRIFT_USE_EBMO env var: For D3, uses EBMO instead of PolicyOnly

    Args:
        difficulty: Difficulty level (1-11, clamped if out of range)
        num_players: Number of players (2-4, default 2)
        board_type: Board type string (e.g., "square8", "square19", "hexagonal")

    Returns:
        DifficultyProfile with ai_type, randomness, think_time_ms, etc.
    """
    effective = max(1, min(11, difficulty))

    # Start with canonical profile
    profile = CANONICAL_DIFFICULTY_PROFILES[effective]

    # Check for experimental EBMO override at D3
    # Enabled via RINGRIFT_USE_EBMO=1 environment variable
    if effective == 3 and os.environ.get("RINGRIFT_USE_EBMO", "").lower() in {"1", "true", "yes"}:
        logger.debug("Using experimental EBMO at D3 (RINGRIFT_USE_EBMO=1)")
        return EBMO_DIFFICULTY_OVERRIDE

    # Check for multiplayer override (3-4 players)
    if num_players >= 3 and effective in MULTIPLAYER_DIFFICULTY_OVERRIDES:
        return MULTIPLAYER_DIFFICULTY_OVERRIDES[effective]

    # Check for large board override (Minimax too slow)
    if board_type and board_type.lower() in LARGE_BOARD_TYPES and effective in LARGE_BOARD_DIFFICULTY_OVERRIDES:
        return LARGE_BOARD_DIFFICULTY_OVERRIDES[effective]

    return profile


def select_ai_type(difficulty: int) -> AIType:
    """Auto-select AI type based on canonical difficulty mapping.

    Args:
        difficulty: Difficulty level (1-11)

    Returns:
        AIType for the given difficulty level
    """
    return get_difficulty_profile(difficulty)["ai_type"]


def get_randomness_for_difficulty(difficulty: int) -> float:
    """Get randomness factor for difficulty level from canonical profile.

    Args:
        difficulty: Difficulty level (1-11)

    Returns:
        Randomness factor (0.0-1.0)
    """
    return get_difficulty_profile(difficulty)["randomness"]


def get_think_time_for_difficulty(difficulty: int) -> int:
    """Get think time in milliseconds for difficulty level.

    Args:
        difficulty: Difficulty level (1-11)

    Returns:
        Think time in milliseconds
    """
    return get_difficulty_profile(difficulty)["think_time_ms"]


def uses_neural_net(difficulty: int) -> bool:
    """Check if the given difficulty level uses neural network evaluation.

    Args:
        difficulty: Difficulty level (1-11)

    Returns:
        True if neural network is enabled for this difficulty
    """
    return get_difficulty_profile(difficulty)["use_neural_net"]


# -----------------------------------------------------------------------------
# AI Factory
# -----------------------------------------------------------------------------


class AIFactory:
    """Centralized factory for creating AI instances.

    This factory consolidates AI creation logic and supports:
    - Creating AIs from difficulty levels
    - Creating AIs with explicit type and configuration
    - Registering custom AI implementations
    - Lazy loading to avoid circular imports

    The factory maintains a registry of AI types and their constructors,
    allowing custom implementations to be registered at runtime.
    """

    # Registry for custom AI implementations
    # Maps string identifiers to callables that create AI instances
    _custom_registry: dict[str, Callable[..., BaseAI]] = {}

    # Cache for imported AI classes (lazy loading)
    _class_cache: dict[AIType, type[BaseAI]] = {}

    @classmethod
    def register(
        cls,
        identifier: str,
        constructor: Callable[..., BaseAI],
    ) -> None:
        """Register a custom AI implementation.

        Args:
            identifier: Unique string identifier for the AI type
            constructor: Callable that creates AI instances.
                         Should accept (player_number, config) arguments.
        """
        if identifier in cls._custom_registry:
            logger.warning(f"Overwriting existing custom AI: {identifier}")
        cls._custom_registry[identifier] = constructor
        logger.debug(f"Registered custom AI: {identifier}")

    @classmethod
    def unregister(cls, identifier: str) -> bool:
        """Unregister a custom AI implementation.

        Args:
            identifier: The identifier to unregister

        Returns:
            True if the identifier was found and removed, False otherwise
        """
        if identifier in cls._custom_registry:
            del cls._custom_registry[identifier]
            logger.debug(f"Unregistered custom AI: {identifier}")
            return True
        return False

    @classmethod
    def list_registered(cls) -> dict[str, str]:
        """List all registered AI types.

        Returns:
            Dict mapping identifiers to descriptions
        """
        result = {}

        # Built-in types
        for ai_type in AIType:
            result[ai_type.value] = f"Built-in: {ai_type.name}"

        # Custom implementations
        for identifier, constructor in cls._custom_registry.items():
            doc = getattr(constructor, "__doc__", None) or "Custom AI"
            result[identifier] = f"Custom: {doc.split(chr(10))[0]}"

        return result

    @classmethod
    def _get_ai_class(cls, ai_type: AIType) -> type[BaseAI]:
        """Get the AI class for a given type, with lazy loading.

        Args:
            ai_type: The AIType to get the class for

        Returns:
            The AI class

        Raises:
            ValueError: If the AI type is not supported
        """
        if ai_type in cls._class_cache:
            return cls._class_cache[ai_type]

        # Lazy imports to avoid circular dependencies
        if ai_type == AIType.RANDOM:
            from app.ai.random_ai import RandomAI
            ai_class = RandomAI
        elif ai_type == AIType.HEURISTIC:
            from app.ai.heuristic_ai import HeuristicAI
            ai_class = HeuristicAI
        elif ai_type == AIType.MINIMAX:
            from app.ai.minimax_ai import MinimaxAI
            ai_class = MinimaxAI
        elif ai_type == AIType.GPU_MINIMAX:
            from app.ai.gpu_minimax_ai import GPUMinimaxAI
            ai_class = GPUMinimaxAI
        elif ai_type == AIType.MAXN:
            from app.ai.maxn_ai import MaxNAI
            ai_class = MaxNAI
        elif ai_type == AIType.BRS:
            from app.ai.maxn_ai import BRSAI
            ai_class = BRSAI
        elif ai_type == AIType.MCTS:
            from app.ai.mcts_ai import MCTSAI
            ai_class = MCTSAI
        elif ai_type == AIType.DESCENT:
            from app.ai.descent_ai import DescentAI
            ai_class = DescentAI
        elif ai_type == AIType.NEURAL_DEMO:
            from app.ai.neural_net import NeuralNetAI
            ai_class = NeuralNetAI
        elif ai_type == AIType.POLICY_ONLY:
            from app.ai.policy_only_ai import PolicyOnlyAI
            ai_class = PolicyOnlyAI
        elif ai_type == AIType.GUMBEL_MCTS:
            from app.ai.gumbel_mcts_ai import GumbelMCTSAI
            ai_class = GumbelMCTSAI
        elif ai_type == AIType.EBMO:
            from app.ai.ebmo_ai import EBMO_AI
            ai_class = EBMO_AI
        elif ai_type == AIType.GMO:
            from app.ai.gmo_ai import GMOAI
            ai_class = GMOAI
        elif ai_type == AIType.IG_GMO:
            from app.ai.ig_gmo import IGGMO
            ai_class = IGGMO
        else:
            raise ValueError(f"Unsupported AI type: {ai_type}")

        cls._class_cache[ai_type] = ai_class
        return ai_class

    @classmethod
    def create(
        cls,
        ai_type: AIType,
        player_number: int,
        config: AIConfig,
        *,
        allow_neural_demo: bool | None = None,
    ) -> BaseAI:
        """Create an AI instance with explicit type and configuration.

        Args:
            ai_type: The type of AI to create
            player_number: The player number (1-indexed)
            config: AI configuration
            allow_neural_demo: Override env check for NEURAL_DEMO type.
                               If None, checks AI_ENGINE_NEURAL_DEMO_ENABLED env var.

        Returns:
            Configured AI instance

        Raises:
            ValueError: If the AI type is not supported or NEURAL_DEMO is blocked
        """
        # Handle NEURAL_DEMO gating
        if ai_type == AIType.NEURAL_DEMO:
            if allow_neural_demo is None:
                flag = os.getenv("AI_ENGINE_NEURAL_DEMO_ENABLED", "").lower()
                allow_neural_demo = flag in {"1", "true", "yes", "on"}

            if not allow_neural_demo:
                logger.warning(
                    "AIType.NEURAL_DEMO requested but AI_ENGINE_NEURAL_DEMO_ENABLED "
                    "is not set; falling back to HeuristicAI."
                )
                ai_type = AIType.HEURISTIC

        ai_class = cls._get_ai_class(ai_type)
        return ai_class(player_number, config)

    @classmethod
    def create_from_difficulty(
        cls,
        difficulty: int,
        player_number: int,
        *,
        num_players: int = 2,
        board_type: str | None = None,
        think_time_override: int | None = None,
        randomness_override: float | None = None,
        rng_seed: int | None = None,
        heuristic_profile_id: str | None = None,
        nn_model_id: str | None = None,
    ) -> BaseAI:
        """Create an AI instance from a difficulty level.

        This is the recommended way to create AIs for normal gameplay,
        as it uses the canonical difficulty profiles with appropriate
        overrides for player count and board size.

        Args:
            difficulty: Difficulty level (1-11)
            player_number: The player number (1-indexed)
            num_players: Total number of players (2-4, affects AI selection)
            board_type: Board type string (affects AI selection for large boards)
            think_time_override: Override default think time (ms)
            randomness_override: Override default randomness (0.0-1.0)
            rng_seed: Optional RNG seed for reproducibility
            heuristic_profile_id: Optional heuristic weight profile
            nn_model_id: Optional neural network model ID

        Returns:
            Configured AI instance
        """
        profile = get_difficulty_profile(
            difficulty,
            num_players=num_players,
            board_type=board_type,
        )

        config = AIConfig(
            difficulty=max(1, min(11, difficulty)),
            think_time=think_time_override or profile["think_time_ms"],
            randomness=randomness_override if randomness_override is not None else profile["randomness"],
            rng_seed=rng_seed,
            heuristic_profile_id=heuristic_profile_id,
            nn_model_id=nn_model_id,
            use_neural_net=profile["use_neural_net"],
        )

        return cls.create(profile["ai_type"], player_number, config)

    @classmethod
    def create_custom(
        cls,
        identifier: str,
        player_number: int,
        config: AIConfig,
        **kwargs: Any,
    ) -> BaseAI:
        """Create an AI instance from a registered custom implementation.

        Args:
            identifier: The registered identifier
            player_number: The player number (1-indexed)
            config: AI configuration
            **kwargs: Additional arguments passed to the constructor

        Returns:
            Configured AI instance

        Raises:
            ValueError: If the identifier is not registered
        """
        if identifier not in cls._custom_registry:
            raise ValueError(
                f"Unknown custom AI identifier: {identifier}. "
                f"Available: {list(cls._custom_registry.keys())}"
            )

        constructor = cls._custom_registry[identifier]
        return constructor(player_number, config, **kwargs)

    @classmethod
    def create_for_tournament(
        cls,
        agent_id: str,
        player_number: int,
        board_type: str = "square8",
        num_players: int = 2,
        *,
        rng_seed: int | None = None,
        nn_model_id: str | None = None,
    ) -> BaseAI:
        """Create an AI for tournament use.

        This method supports both built-in agent IDs and custom registered AIs.
        Built-in agent IDs include:
        - "random": RandomAI at difficulty 1
        - "heuristic": HeuristicAI at difficulty 2
        - "minimax": MinimaxAI at difficulty 3
        - "mcts_N": MCTS AI with N simulations budget
        - "difficulty_N": AI at difficulty level N

        Args:
            agent_id: Agent identifier (see above)
            player_number: The player number (1-indexed)
            board_type: Board type for context
            num_players: Number of players for context
            rng_seed: Optional RNG seed
            nn_model_id: Optional neural network model ID for NN-guided agents

        Returns:
            Configured AI instance
        """
        # Check custom registry first
        if agent_id in cls._custom_registry:
            config = AIConfig(difficulty=5, rng_seed=rng_seed, nn_model_id=nn_model_id)
            return cls.create_custom(agent_id, player_number, config)

        # Parse built-in agent IDs
        agent_lower = agent_id.lower()
        agent_key = agent_lower.replace("-", "_")

        if agent_key == "random":
            return cls.create_from_difficulty(1, player_number, rng_seed=rng_seed)

        if agent_key == "heuristic":
            return cls.create_from_difficulty(2, player_number, rng_seed=rng_seed)

        if agent_key == "minimax":
            return cls.create_from_difficulty(3, player_number, rng_seed=rng_seed, nn_model_id=nn_model_id)

        if agent_key.startswith("mcts_"):
            # Parse simulation count: mcts_100, mcts_500, etc.
            try:
                sims = int(agent_key.split("_")[1])
                # Map simulation count to approximate difficulty
                if sims <= 100:
                    difficulty = 5
                elif sims <= 500:
                    difficulty = 6
                elif sims <= 1000:
                    difficulty = 7
                else:
                    difficulty = 8

                # Enable neural net if model_id provided or difficulty >= 6
                use_nn = nn_model_id is not None or difficulty >= 6

                config = AIConfig(
                    difficulty=difficulty,
                    think_time=sims * 10,  # Rough heuristic
                    rng_seed=rng_seed,
                    nn_model_id=nn_model_id,
                    use_neural_net=use_nn,
                )
                return cls.create(AIType.MCTS, player_number, config)
            except (ValueError, IndexError):
                pass

        # Policy-only AI (direct NN policy without search)
        if agent_key == "policy_only" or agent_key.startswith("policy_"):
            from app.models import BoardType as BT
            bt = BT.SQUARE8 if board_type.lower() == "square8" else BT.HEXAGONAL

            # Parse optional temperature: policy_0.5, policy_1.0, etc.
            temperature = 1.0
            if "_" in agent_key:
                with contextlib.suppress(ValueError, IndexError):
                    temperature = float(agent_key.split("_")[1])

            config = AIConfig(
                difficulty=5,
                rng_seed=rng_seed,
                nn_model_id=nn_model_id,
                policy_temperature=temperature,
                allow_fresh_weights=False,
            )
            from app.ai.policy_only_ai import PolicyOnlyAI
            return PolicyOnlyAI(player_number, config, bt)

        # Gumbel MCTS AI (sample-efficient search)
        if agent_key == "gumbel_mcts" or agent_key.startswith("gumbel_"):
            from app.models import BoardType as BT
            bt = BT.SQUARE8 if board_type.lower() == "square8" else BT.HEXAGONAL

            # Parse optional budget: gumbel_100, gumbel_200, etc.
            budget = 150
            if "_" in agent_key and agent_key != "gumbel_mcts":
                try:
                    parts = agent_key.split("_")
                    if parts[1] != "mcts":
                        budget = int(parts[1])
                except (ValueError, IndexError):
                    pass

            config = AIConfig(
                difficulty=5,
                rng_seed=rng_seed,
                nn_model_id=nn_model_id,
                gumbel_simulation_budget=budget,
                allow_fresh_weights=False,
            )
            from app.ai.gumbel_mcts_ai import GumbelMCTSAI
            return GumbelMCTSAI(player_number, config, bt)

        # EBMO AI (Energy-Based Move Optimization)
        if agent_key == "ebmo" or agent_key.startswith("ebmo_"):
            # Parse optional model path: ebmo_modelpath
            model_path = None
            if "_" in agent_lower:
                try:
                    parts = agent_lower.split("_", 1)
                    if len(parts) > 1:
                        model_path = parts[1]
                except (ValueError, IndexError):
                    pass

            config = AIConfig(
                difficulty=5,
                rng_seed=rng_seed,
                nn_model_id=nn_model_id,
            )
            from app.ai.ebmo_ai import EBMO_AI
            return EBMO_AI(player_number, config, model_path=model_path)

        # GMO AI (Gradient Move Optimization - entropy-guided gradient ascent)
        if agent_key == "gmo" or agent_key.startswith("gmo_"):
            config = AIConfig(
                difficulty=6,
                rng_seed=rng_seed,
                nn_model_id=nn_model_id,
            )
            from app.ai.gmo_ai import GMOAI
            return GMOAI(player_number, config)

        if agent_key == "ig_gmo" or agent_key.startswith("ig_gmo_"):
            config = AIConfig(
                difficulty=6,
                rng_seed=rng_seed,
                nn_model_id=nn_model_id,
            )
            from app.ai.ig_gmo import IGGMO
            return IGGMO(player_number, config)

        if agent_key.startswith("difficulty_") or agent_key.startswith("level_"):
            # Parse difficulty level: difficulty_5, level_7, etc.
            try:
                level = int(agent_key.split("_")[1])
                return cls.create_from_difficulty(level, player_number, rng_seed=rng_seed, nn_model_id=nn_model_id)
            except (ValueError, IndexError):
                pass

        # Try parsing as AIType directly
        try:
            ai_type = AIType(agent_key)
            profile = get_difficulty_profile(5)  # Default to mid difficulty
            config = AIConfig(
                difficulty=5,
                think_time=profile["think_time_ms"],
                rng_seed=rng_seed,
                nn_model_id=nn_model_id,
            )
            return cls.create(ai_type, player_number, config)
        except ValueError:
            pass

        # Default fallback
        logger.warning(f"Unknown agent ID '{agent_id}', defaulting to heuristic")
        return cls.create_from_difficulty(2, player_number, rng_seed=rng_seed)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the class cache. Useful for testing."""
        cls._class_cache.clear()


# -----------------------------------------------------------------------------
# Convenience aliases
# -----------------------------------------------------------------------------

# Module-level function aliases for backwards compatibility
create_ai = AIFactory.create
create_ai_from_difficulty = AIFactory.create_from_difficulty
create_tournament_ai = AIFactory.create_for_tournament


def get_all_difficulties() -> dict[int, DifficultyProfile]:
    """Get all canonical difficulty profiles.

    Returns:
        Dict mapping difficulty levels (1-11) to profiles
    """
    return CANONICAL_DIFFICULTY_PROFILES.copy()


def get_difficulty_description(difficulty: int) -> str:
    """Get human-readable description for a difficulty level.

    Args:
        difficulty: Difficulty level (1-11)

    Returns:
        Human-readable description
    """
    effective = max(1, min(11, difficulty))
    return DIFFICULTY_DESCRIPTIONS.get(effective, f"Difficulty {effective}")
