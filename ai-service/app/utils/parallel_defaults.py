"""Shared parallelism defaults for the codebase.

This module provides consistent defaults for parallel operations across
the ai-service codebase. The principle is: parallelism should be the default,
single-threaded should be optional (for debugging).

Usage:
    from app.utils.parallel_defaults import get_default_workers, get_dataloader_workers

    # For CPU-bound tasks (export, backfill, etc.)
    num_workers = get_default_workers()  # cpu_count - 1

    # For I/O-bound tasks (network, disk)
    num_workers = get_default_workers(divisor=2)  # cpu_count // 2

    # For PyTorch DataLoader (platform-aware)
    num_workers = get_dataloader_workers()  # 4 on CUDA, 0 on macOS, 2 on Linux CPU
"""

from __future__ import annotations

import os
import sys


def get_default_workers(divisor: int = 1, minimum: int = 1) -> int:
    """Get default worker count based on CPU cores.

    Args:
        divisor: Divide CPU count by this (use 2 for I/O-heavy tasks)
        minimum: Minimum workers to return (default 1)

    Returns:
        At least `minimum`, at most cpu_count - 1
    """
    cpu = os.cpu_count() or 4
    workers = max(minimum, (cpu - 1) // max(1, divisor))
    return workers


def get_dataloader_workers() -> int:
    """Get workers for PyTorch DataLoader (platform-aware).

    Returns:
        - 4 for CUDA systems (async loading helps GPU utilization)
        - 0 for macOS (memory mapping issues with multiprocessing)
        - 2 for Linux CPU (balanced default)
    """
    if sys.platform == "darwin":
        # macOS has memory mapping issues with torch multiprocessing
        return 0

    try:
        import torch

        if torch.cuda.is_available():
            return 4  # Async loading helps keep GPU fed
    except ImportError:
        pass

    return 2  # Default for Linux CPU


def get_tournament_workers() -> int:
    """Get workers for tournament/evaluation tasks.

    Uses fewer workers than CPU-bound tasks since tournaments
    often have GPU involvement and memory pressure.

    Returns:
        Half of available CPUs, minimum 2, maximum 8
    """
    cpu = os.cpu_count() or 4
    return min(8, max(2, cpu // 2))


# =============================================================================
# Parallel Execution Utilities (Jan 12, 2026)
# =============================================================================
#
# These utilities make parallelism the default for data-heavy and compute-heavy
# operations. Single-threaded execution is available as an opt-out for debugging.
#
# Environment Variables:
#   RINGRIFT_FORCE_SEQUENTIAL=true - Disable parallelism for debugging
#   RINGRIFT_MAX_WORKERS=N - Override max workers
# =============================================================================

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, TypeVar, Iterable, Any
import asyncio

T = TypeVar("T")
R = TypeVar("R")

# Environment variable overrides
FORCE_SEQUENTIAL = os.getenv("RINGRIFT_FORCE_SEQUENTIAL", "").lower() == "true"
MAX_WORKERS_OVERRIDE = int(os.getenv("RINGRIFT_MAX_WORKERS", "0")) or None

# February 2026: Cap worker counts on coordinator nodes to prevent OOM.
# Coordinator runs 45+ daemons and shouldn't spawn heavy process pools.
_IS_COORDINATOR = os.getenv("RINGRIFT_IS_COORDINATOR", "").lower() == "true"
_COORDINATOR_MAX_PROCESS_WORKERS = 2
_COORDINATOR_MAX_THREAD_WORKERS = 4

# Default worker counts based on hardware
DEFAULT_THREAD_WORKERS = min(
    _COORDINATOR_MAX_THREAD_WORKERS if _IS_COORDINATOR else 32,
    (os.cpu_count() or 4) * 2,
)
DEFAULT_PROCESS_WORKERS = min(
    _COORDINATOR_MAX_PROCESS_WORKERS if _IS_COORDINATOR else (os.cpu_count() or 4),
    os.cpu_count() or 4,
)


def parallel_map_threads(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
) -> list[R]:
    """Map function over items using thread pool (I/O-bound work).

    Use this for operations like database queries, file I/O, network calls.

    Args:
        func: Function to apply to each item
        items: Items to process
        max_workers: Max threads (default: 2x cpu_count, max 32)

    Returns:
        Results in the same order as input items

    Example:
        >>> def fetch_game_count(db_path):
        ...     return count_games(db_path)
        >>> counts = parallel_map_threads(fetch_game_count, db_paths)
    """
    items_list = list(items)
    if not items_list:
        return []

    workers = max_workers or MAX_WORKERS_OVERRIDE or DEFAULT_THREAD_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items_list))


def parallel_map_processes(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
) -> list[R]:
    """Map function over items using process pool (CPU-bound work).

    Use this for CPU-intensive operations like policy inference, data encoding.
    Note: func must be picklable (module-level function or lambda).

    Args:
        func: Function to apply to each item (must be picklable)
        items: Items to process
        max_workers: Max processes (default: cpu_count)

    Returns:
        Results in the same order as input items

    Example:
        >>> def encode_game(game_data):
        ...     return encode_to_npz(game_data)
        >>> encoded = parallel_map_processes(encode_game, games)
    """
    items_list = list(items)
    if not items_list:
        return []

    workers = max_workers or MAX_WORKERS_OVERRIDE or DEFAULT_PROCESS_WORKERS
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items_list))


def parallel_map_unordered(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
    use_processes: bool = False,
) -> list[R]:
    """Map function with results returned as they complete (unordered).

    Use when you don't need results in input order and want progress
    as items complete.

    Args:
        func: Function to apply
        items: Items to process
        max_workers: Max workers
        use_processes: True for ProcessPoolExecutor, False for ThreadPool

    Returns:
        Results in completion order (not input order)
    """
    items_list = list(items)
    if not items_list:
        return []

    workers = max_workers or MAX_WORKERS_OVERRIDE or (
        DEFAULT_PROCESS_WORKERS if use_processes else DEFAULT_THREAD_WORKERS
    )
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with executor_cls(max_workers=workers) as executor:
        futures = {executor.submit(func, item): item for item in items_list}
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results


async def parallel_async(
    coros: list,
    max_concurrent: int | None = None,
) -> list:
    """Run coroutines concurrently with optional concurrency limit.

    Use for async operations like HTTP requests, async database queries.

    Args:
        coros: List of coroutines to run
        max_concurrent: Max concurrent tasks (default: unlimited)

    Returns:
        Results in input order

    Example:
        >>> async def fetch_status(node):
        ...     return await http_get(f"http://{node}/status")
        >>> statuses = await parallel_async([fetch_status(n) for n in nodes])
    """
    if not coros:
        return []

    if max_concurrent is None:
        return await asyncio.gather(*coros, return_exceptions=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited(c) for c in coros], return_exceptions=True)


async def to_thread(func: Callable[..., R], *args, **kwargs) -> R:
    """Run blocking function in thread pool (async wrapper).

    Use to call blocking code from async context without blocking event loop.

    Args:
        func: Blocking function
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result

    Example:
        >>> count = await to_thread(count_games_in_db, db_path)
    """
    return await asyncio.to_thread(func, *args, **kwargs)


def maybe_parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    parallel_threshold: int = 4,
    use_processes: bool = False,
    max_workers: int | None = None,
) -> list[R]:
    """Smart parallel map - sequential for small inputs or debug mode.

    Automatically chooses between parallel and sequential execution based on:
    1. RINGRIFT_FORCE_SEQUENTIAL env var (debug mode)
    2. Number of items (below threshold = sequential)

    Args:
        func: Function to apply
        items: Items to process
        parallel_threshold: Minimum items for parallel execution (default: 4)
        use_processes: Use ProcessPoolExecutor instead of ThreadPool
        max_workers: Max workers (default: auto-scaled)

    Returns:
        Results in input order

    Example:
        >>> # Automatically parallel for large lists
        >>> counts = maybe_parallel_map(count_games, db_paths)
        >>> # Force sequential with env var:
        >>> # RINGRIFT_FORCE_SEQUENTIAL=true python script.py
    """
    items_list = list(items)
    if not items_list:
        return []

    # Use sequential for small inputs or debug mode
    if FORCE_SEQUENTIAL or len(items_list) < parallel_threshold:
        return [func(item) for item in items_list]

    # Use parallel execution
    mapper = parallel_map_processes if use_processes else parallel_map_threads
    return mapper(func, items_list, max_workers)


def get_parallel_games_default() -> int:
    """Get default parallel games count for gauntlet/tournament.

    Returns:
        16 for systems with 8+ cores, 8 for smaller systems
    """
    cpu = os.cpu_count() or 4
    return 16 if cpu >= 8 else 8


def should_use_parallel() -> bool:
    """Check if parallel execution is enabled.

    Returns:
        False if RINGRIFT_FORCE_SEQUENTIAL is set, True otherwise
    """
    return not FORCE_SEQUENTIAL
