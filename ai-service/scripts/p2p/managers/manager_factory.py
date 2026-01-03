"""ManagerFactory - Dependency injection for P2P managers.

January 2026: Phase 4.4 - Explicit dependency management.

This module provides a factory for creating P2P orchestrator managers
with explicit dependency injection. It replaces fragile initialization
order in p2p_orchestrator.__init__ with clear, lazy-loaded dependencies.

Benefits:
- Explicit dependency graph (no hidden init order issues)
- Lazy initialization (managers created on first access)
- Easy testing (mock individual managers)
- Clear error messages on circular dependencies
- Reset capability for testing

Usage:
    from scripts.p2p.managers import ManagerFactory, ManagerConfig

    # Create factory with config
    config = ManagerConfig(
        db_path=Path("data/p2p.db"),
        node_id="my-node",
        verbose=True,
    )
    factory = ManagerFactory(config)

    # Access managers (lazy-loaded)
    state = factory.state_manager  # Created on first access
    jobs = factory.job_manager      # Created on first access, gets state_manager

    # Reset for testing
    factory.reset()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from scripts.p2p.managers import (
        JobManager,
        NodeSelector,
        SelfplayScheduler,
        StateManager,
        SyncPlanner,
        TrainingCoordinator,
    )


@dataclass
class ManagerConfig:
    """Configuration for manager factory.

    All managers share this configuration, extracting what they need.
    """
    # Core paths
    db_path: Path = field(default_factory=lambda: Path("data/p2p.db"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Identity
    node_id: str = ""
    is_coordinator: bool = False

    # Behavior
    verbose: bool = False
    dry_run: bool = False

    # Network
    port: int = 8770
    bind_address: str = "0.0.0.0"

    # Timeouts
    job_timeout: int = 3600
    sync_timeout: int = 300

    # Feature flags
    enable_selfplay: bool = True
    enable_training: bool = True
    enable_sync: bool = True

    # Optional orchestrator reference (for backward compat)
    orchestrator: Any = None


class ManagerFactory:
    """Factory for P2P orchestrator managers with dependency injection.

    Managers are created lazily on first access. Dependencies between
    managers are resolved automatically through property access.

    Dependency Graph:
        StateManager (no deps)
            ↓
        JobManager (depends on: StateManager)
            ↓
        TrainingCoordinator (depends on: StateManager, JobManager)
            ↓
        SelfplayScheduler (depends on: StateManager, JobManager)
            ↓
        SyncPlanner (depends on: StateManager)
            ↓
        NodeSelector (depends on: StateManager, JobManager)
    """

    def __init__(self, config: ManagerConfig):
        self._config = config
        self._managers: dict[str, Any] = {}
        self._creating: set[str] = set()  # Track creation to detect cycles

    @property
    def config(self) -> ManagerConfig:
        """Get the factory configuration."""
        return self._config

    def reset(self) -> None:
        """Reset all managers.

        Use this in tests to get fresh manager instances.
        """
        # Clear cached properties
        for name in list(self.__dict__.keys()):
            if name.startswith("_cached_"):
                delattr(self, name)
        self._managers.clear()
        self._creating.clear()

    def _check_cycle(self, name: str) -> None:
        """Check for circular dependency."""
        if name in self._creating:
            cycle = " -> ".join(self._creating) + f" -> {name}"
            raise RuntimeError(f"Circular dependency detected: {cycle}")
        self._creating.add(name)

    def _done_creating(self, name: str) -> None:
        """Mark manager creation as complete."""
        self._creating.discard(name)

    # =========================================================================
    # Manager Properties (lazy-loaded with explicit dependencies)
    # =========================================================================

    @cached_property
    def state_manager(self) -> StateManager:
        """Get StateManager (no dependencies).

        Handles: SQLite persistence, cluster epoch tracking
        """
        self._check_cycle("state_manager")
        try:
            from scripts.p2p.managers import StateManager
            manager = StateManager(
                db_path=self._config.db_path,
                verbose=self._config.verbose,
            )
            self._managers["state_manager"] = manager
            logger.debug("Created StateManager")
            return manager
        finally:
            self._done_creating("state_manager")

    @cached_property
    def job_manager(self) -> JobManager:
        """Get JobManager (depends on: StateManager).

        Handles: Job spawning, lifecycle, cleanup
        """
        self._check_cycle("job_manager")
        try:
            from scripts.p2p.managers import JobManager
            manager = JobManager(
                state_manager=self.state_manager,  # Dependency
                orchestrator=self._config.orchestrator,
                db_path=self._config.db_path,
                verbose=self._config.verbose,
            )
            self._managers["job_manager"] = manager
            logger.debug("Created JobManager")
            return manager
        finally:
            self._done_creating("job_manager")

    @cached_property
    def training_coordinator(self) -> TrainingCoordinator:
        """Get TrainingCoordinator (depends on: StateManager, JobManager).

        Handles: Training dispatch, model promotion, gauntlet
        """
        self._check_cycle("training_coordinator")
        try:
            from scripts.p2p.managers import TrainingCoordinator
            manager = TrainingCoordinator(
                state_manager=self.state_manager,  # Dependency
                job_manager=self.job_manager,      # Dependency
                orchestrator=self._config.orchestrator,
                verbose=self._config.verbose,
            )
            self._managers["training_coordinator"] = manager
            logger.debug("Created TrainingCoordinator")
            return manager
        finally:
            self._done_creating("training_coordinator")

    @cached_property
    def selfplay_scheduler(self) -> SelfplayScheduler:
        """Get SelfplayScheduler (depends on: StateManager, JobManager).

        Handles: Config selection, diversity tracking, priority boost
        """
        self._check_cycle("selfplay_scheduler")
        try:
            from scripts.p2p.managers import SelfplayScheduler
            manager = SelfplayScheduler(
                state_manager=self.state_manager,  # Dependency
                job_manager=self.job_manager,      # Dependency
                orchestrator=self._config.orchestrator,
            )
            self._managers["selfplay_scheduler"] = manager
            logger.debug("Created SelfplayScheduler")
            return manager
        finally:
            self._done_creating("selfplay_scheduler")

    @cached_property
    def sync_planner(self) -> SyncPlanner:
        """Get SyncPlanner (depends on: StateManager).

        Handles: Manifest collection, sync planning, quality routing
        """
        self._check_cycle("sync_planner")
        try:
            from scripts.p2p.managers import SyncPlanner
            manager = SyncPlanner(
                state_manager=self.state_manager,  # Dependency
                orchestrator=self._config.orchestrator,
            )
            self._managers["sync_planner"] = manager
            logger.debug("Created SyncPlanner")
            return manager
        finally:
            self._done_creating("sync_planner")

    @cached_property
    def node_selector(self) -> NodeSelector:
        """Get NodeSelector (depends on: StateManager, JobManager).

        Handles: Node ranking, GPU selection, capacity tracking
        """
        self._check_cycle("node_selector")
        try:
            from scripts.p2p.managers import NodeSelector
            manager = NodeSelector(
                state_manager=self.state_manager,  # Dependency
                job_manager=self.job_manager,      # Dependency
                orchestrator=self._config.orchestrator,
            )
            self._managers["node_selector"] = manager
            logger.debug("Created NodeSelector")
            return manager
        finally:
            self._done_creating("node_selector")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_all_managers(self) -> dict[str, Any]:
        """Get all managers (forces initialization).

        Returns dict of name -> manager instance.
        """
        return {
            "state_manager": self.state_manager,
            "job_manager": self.job_manager,
            "training_coordinator": self.training_coordinator,
            "selfplay_scheduler": self.selfplay_scheduler,
            "sync_planner": self.sync_planner,
            "node_selector": self.node_selector,
        }

    def get_initialized_managers(self) -> dict[str, Any]:
        """Get only managers that have been initialized.

        Doesn't force initialization of unaccessed managers.
        """
        return dict(self._managers)


# =============================================================================
# Module-level factory (singleton pattern)
# =============================================================================

_factory: ManagerFactory | None = None


def get_manager_factory() -> ManagerFactory | None:
    """Get the global manager factory (if initialized)."""
    return _factory


def init_manager_factory(config: ManagerConfig) -> ManagerFactory:
    """Initialize the global manager factory.

    Args:
        config: Factory configuration

    Returns:
        The initialized factory
    """
    global _factory
    _factory = ManagerFactory(config)
    return _factory


def reset_manager_factory() -> None:
    """Reset the global manager factory (for testing)."""
    global _factory
    if _factory is not None:
        _factory.reset()
    _factory = None
