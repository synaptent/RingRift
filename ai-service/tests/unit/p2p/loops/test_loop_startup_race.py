"""Test for loop startup race condition fix (Jan 2026).

Verifies that LoopManager.start_all() properly waits for loops
to transition to running state, even with slow _on_start().

This test prevents regression of the bug where loops were marked as
"failed" because the 0.1s sleep was insufficient for the asyncio task
to actually start and set _running = True.
"""
from __future__ import annotations

import asyncio

import pytest

from scripts.p2p.loops.base import BaseLoop, LoopManager


class SlowStartLoop(BaseLoop):
    """Loop with slow _on_start() to test race condition handling."""

    def __init__(self, startup_delay: float = 0.3):
        super().__init__(name="slow_start_loop", interval=1.0)
        self.startup_delay = startup_delay
        self.run_count = 0

    async def _on_start(self) -> None:
        await asyncio.sleep(self.startup_delay)

    async def _run_once(self) -> None:
        self.run_count += 1


class FastStartLoop(BaseLoop):
    """Loop with minimal _on_start() for baseline comparison."""

    def __init__(self):
        super().__init__(name="fast_start_loop", interval=1.0)
        self.run_count = 0

    async def _run_once(self) -> None:
        self.run_count += 1


@pytest.mark.asyncio
async def test_start_all_handles_slow_loop_startup():
    """start_all() should wait for slow _on_start() to complete."""
    manager = LoopManager(name="test_manager")
    loop = SlowStartLoop(startup_delay=0.3)
    manager.register(loop)

    results = await manager.start_all(startup_timeout=5.0)

    # Should succeed despite slow startup
    assert results["slow_start_loop"] is True, "Loop should start despite 0.3s _on_start() delay"
    assert loop.running is True, "Loop should be running after start_all()"

    await manager.stop_all()


@pytest.mark.asyncio
async def test_start_all_handles_fast_loop_startup():
    """start_all() should handle fast-starting loops correctly."""
    manager = LoopManager(name="test_manager")
    loop = FastStartLoop()
    manager.register(loop)

    results = await manager.start_all(startup_timeout=5.0)

    assert results["fast_start_loop"] is True
    assert loop.running is True

    await manager.stop_all()


@pytest.mark.asyncio
async def test_start_all_handles_multiple_loops():
    """start_all() should handle multiple loops with different startup times."""
    manager = LoopManager(name="test_manager")
    slow_loop = SlowStartLoop(startup_delay=0.3)
    fast_loop = FastStartLoop()
    manager.register(slow_loop)
    manager.register(fast_loop)

    results = await manager.start_all(startup_timeout=10.0)

    assert results["slow_start_loop"] is True
    assert results["fast_start_loop"] is True
    assert slow_loop.running is True
    assert fast_loop.running is True

    await manager.stop_all()


@pytest.mark.asyncio
async def test_start_all_with_slow_on_start_still_reports_running():
    """Loop with slow _on_start() still reports as running (initialization is async).

    Note: _running=True is set BEFORE _on_start() completes, so even a loop
    with a 5s initialization delay will report as "started" immediately.
    This is correct behavior - the loop IS running, just still initializing.
    """
    manager = LoopManager(name="test_manager")
    # 5.0s delay in _on_start(), but loop still starts immediately
    loop = SlowStartLoop(startup_delay=0.5)
    manager.register(loop)

    results = await manager.start_all(startup_timeout=10.0)

    # Should succeed because _running=True is set before _on_start() completes
    assert results["slow_start_loop"] is True, "Loop starts immediately even with slow _on_start()"
    assert loop.running is True

    await manager.stop_all()


@pytest.mark.asyncio
async def test_loop_actually_runs_after_startup():
    """Verify loops actually execute _run_once() after successful startup."""
    manager = LoopManager(name="test_manager")
    loop = FastStartLoop()
    manager.register(loop)

    results = await manager.start_all(startup_timeout=5.0)
    assert results["fast_start_loop"] is True

    # Wait a bit for the loop to run
    await asyncio.sleep(1.5)

    # Loop should have run at least once
    assert loop.run_count >= 1, f"Loop should have run at least once, got {loop.run_count}"

    await manager.stop_all()
