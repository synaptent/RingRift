"""P2P Orchestrator Background Loops.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This package contains background loop implementations extracted from p2p_orchestrator.py
for better modularity and testability.

Loop Categories:
- base.py: BaseLoop abstract class with error handling and backoff
- queue_populator_loop.py: Work queue population for selfplay jobs
- data_loops.py: JSONL conversion, model sync (TODO)
- network_loops.py: IP updates, Tailscale recovery (TODO)
- coordination_loops.py: Elo sync, auto-scaling (TODO)
- job_loops.py: Job reaper, idle detection (TODO)

Usage:
    from scripts.p2p.loops import BaseLoop, BackoffConfig, LoopManager, LoopStats
    from scripts.p2p.loops import QueuePopulatorLoop

    class MyLoop(BaseLoop):
        async def _run_once(self) -> None:
            # Your loop logic here
            pass

    # Single loop usage
    loop = MyLoop(name="my_loop", interval=60.0)
    await loop.run_forever()

    # Multiple loops with manager
    manager = LoopManager()
    manager.register(MyLoop(name="loop1", interval=30.0))
    manager.register(QueuePopulatorLoop(
        get_role=lambda: NodeRole.LEADER,
        get_selfplay_scheduler=lambda: scheduler,
    ))
    await manager.start_all()
    # ... later ...
    await manager.stop_all()
"""

from .base import (
    BackoffConfig,
    BaseLoop,
    LoopManager,
    LoopStats,
)
from .elo_sync_loop import EloSyncLoop
from .queue_populator_loop import QueuePopulatorLoop

__all__ = [
    "BackoffConfig",
    "BaseLoop",
    "EloSyncLoop",
    "LoopManager",
    "LoopStats",
    "QueuePopulatorLoop",
]
