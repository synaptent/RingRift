"""
HTTP client for distributed CMA-ES evaluation on local Mac cluster.

This module provides the WorkerClient and DistributedEvaluator classes
for distributing CMA-ES population evaluation across multiple worker
machines.

Usage:
------
    from app.distributed.client import DistributedEvaluator
    from app.distributed.discovery import discover_workers

    # Discover workers
    workers = discover_workers()

    # Create evaluator
    evaluator = DistributedEvaluator(
        workers=[w.url for w in workers],
        board_type="square8",
        num_players=2,
        games_per_eval=24,
    )

    # Evaluate a population of candidates
    fitness_scores = evaluator.evaluate_population(
        population=[weights_1, weights_2, ...],
        baseline_weights=baseline,
    )
"""
from __future__ import annotations

import json
import logging
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HeuristicWeights,
)

logger = logging.getLogger(__name__)


# Memory requirements per board type (in GB)
# Must be kept in sync with cluster_worker.py BOARD_MEMORY_REQUIREMENTS
BOARD_MEMORY_REQUIREMENTS: Dict[str, int] = {
    "square8": 8,      # 8GB minimum for 8x8 games
    "square19": 48,    # 48GB minimum for 19x19 games
    "hexagonal": 48,   # 48GB minimum for hex games
    "hex": 48,         # Alias for hexagonal
}


@dataclass
class TaskResult:
    """Result from a worker evaluation task."""

    task_id: str
    candidate_id: int
    fitness: float
    games_played: int
    evaluation_time_sec: float
    worker_id: str
    status: str  # "success" or "error"
    error: Optional[str] = None
    # Optional game replay data (only populated if task requested recording)
    game_replays: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        return cls(
            task_id=data.get("task_id", ""),
            candidate_id=data.get("candidate_id", -1),
            fitness=data.get("fitness", 0.0),
            games_played=data.get("games_played", 0),
            evaluation_time_sec=data.get("evaluation_time_sec", 0.0),
            worker_id=data.get("worker_id", "unknown"),
            status=data.get("status", "error"),
            error=data.get("error"),
            game_replays=data.get("game_replays"),
        )


@dataclass
class EvaluationStats:
    """Statistics from a distributed population evaluation."""

    total_candidates: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    total_games: int = 0
    total_time_sec: float = 0.0
    worker_task_counts: Dict[str, int] = field(default_factory=dict)
    # Aggregated game replays from all workers (populated when record_games=True)
    all_game_replays: List[Dict[str, Any]] = field(default_factory=list)

    def add_result(self, result: TaskResult) -> None:
        self.total_candidates += 1
        if result.status == "success":
            self.successful_evaluations += 1
            self.total_games += result.games_played
            # Aggregate game replays if present
            if result.game_replays:
                self.all_game_replays.extend(result.game_replays)
        else:
            self.failed_evaluations += 1

        worker = result.worker_id
        self.worker_task_counts[worker] = self.worker_task_counts.get(worker, 0) + 1


class WorkerClient:
    """
    HTTP client for communicating with a single worker.

    Provides methods for health checks, pool preloading, and task evaluation.
    """

    def __init__(
        self,
        worker_url: str,
        timeout: float = 300.0,
    ):
        """
        Initialize worker client.

        Parameters
        ----------
        worker_url : str
            Worker address in format "host:port" or full URL "http://host:port"
        timeout : float
            Request timeout in seconds
        """
        self.worker_url = worker_url
        self.timeout = timeout
        # Handle both "host:port" and "http://host:port" formats
        if worker_url.startswith("http://") or worker_url.startswith("https://"):
            self._base_url = worker_url
        else:
            self._base_url = f"http://{worker_url}"

    def health_check(self) -> Dict[str, Any]:
        """
        Check worker health.

        Returns
        -------
        Dict[str, Any]
            Health response or error dict
        """
        try:
            url = f"{self._base_url}/health"
            request = Request(url, method="GET")
            with urlopen(request, timeout=5.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            return {"status": "error", "error": f"HTTP {e.code}"}
        except URLError as e:
            return {"status": "error", "error": f"URL error: {e.reason}"}
        except (socket.timeout, TimeoutError):
            return {"status": "error", "error": "Request timeout"}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}
        except OSError as e:
            return {"status": "error", "error": f"Network error: {e}"}

    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        result = self.health_check()
        return result.get("status") == "healthy"

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get worker memory information.

        Returns dict with:
            - total_gb: Total RAM in GB
            - available_gb: Available RAM in GB
            - eligible_boards: List of board types worker can handle
        """
        health = self.health_check()
        if health.get("status") != "healthy":
            return {"total_gb": 0, "available_gb": 0, "eligible_boards": []}
        return health.get("memory", {"total_gb": 0, "available_gb": 0, "eligible_boards": []})

    def can_handle_board(self, board_type: str) -> bool:
        """
        Check if worker has enough memory to handle the given board type.

        Parameters
        ----------
        board_type : str
            Board type to check (e.g., "square8", "square19", "hex")

        Returns
        -------
        bool
            True if worker can handle the board type
        """
        memory_info = self.get_memory_info()
        eligible = memory_info.get("eligible_boards", [])

        # Check if board type is in eligible list
        if board_type in eligible:
            return True

        # Fall back to direct memory check
        total_gb = memory_info.get("total_gb", 0)
        required_gb = BOARD_MEMORY_REQUIREMENTS.get(board_type, 8)
        return total_gb >= required_gb

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed worker statistics."""
        try:
            url = f"{self._base_url}/stats"
            request = Request(url, method="GET")
            with urlopen(request, timeout=5.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            return {"status": "error", "error": f"HTTP {e.code}"}
        except URLError as e:
            return {"status": "error", "error": f"URL error: {e.reason}"}
        except (socket.timeout, TimeoutError):
            return {"status": "error", "error": "Request timeout"}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}
        except OSError as e:
            return {"status": "error", "error": f"Network error: {e}"}

    def preload_pool(
        self,
        board_type: str,
        num_players: int,
        pool_id: str,
    ) -> Dict[str, Any]:
        """
        Request worker to preload a state pool.

        Parameters
        ----------
        board_type : str
            Board type (e.g., "square8")
        num_players : int
            Number of players
        pool_id : str
            State pool identifier

        Returns
        -------
        Dict[str, Any]
            Response with pool info or error
        """
        try:
            url = f"{self._base_url}/preload-pool"
            data = json.dumps({
                "board_type": board_type,
                "num_players": num_players,
                "pool_id": pool_id,
            }).encode("utf-8")
            request = Request(
                url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=30.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            return {"status": "error", "error": f"HTTP {e.code}"}
        except URLError as e:
            return {"status": "error", "error": f"URL error: {e.reason}"}
        except (socket.timeout, TimeoutError):
            return {"status": "error", "error": "Request timeout"}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}
        except OSError as e:
            return {"status": "error", "error": f"Network error: {e}"}

    def evaluate(self, task: Dict[str, Any]) -> TaskResult:
        """
        Submit an evaluation task to the worker.

        Parameters
        ----------
        task : Dict[str, Any]
            Task specification containing weights and config

        Returns
        -------
        TaskResult
            Evaluation result
        """
        try:
            url = f"{self._base_url}/evaluate"
            data = json.dumps(task).encode("utf-8")
            request = Request(
                url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=self.timeout) as response:
                result_data = json.loads(response.read().decode("utf-8"))
                return TaskResult.from_dict(result_data)
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            return TaskResult(
                task_id=task.get("task_id", ""),
                candidate_id=task.get("candidate_id", -1),
                fitness=0.0,
                games_played=0,
                evaluation_time_sec=0.0,
                worker_id=self.worker_url,
                status="error",
                error=f"HTTP {e.code}: {error_body}",
            )
        except URLError as e:
            return TaskResult(
                task_id=task.get("task_id", ""),
                candidate_id=task.get("candidate_id", -1),
                fitness=0.0,
                games_played=0,
                evaluation_time_sec=0.0,
                worker_id=self.worker_url,
                status="error",
                error=f"URL error: {e.reason}",
            )
        except (socket.timeout, TimeoutError):
            return TaskResult(
                task_id=task.get("task_id", ""),
                candidate_id=task.get("candidate_id", -1),
                fitness=0.0,
                games_played=0,
                evaluation_time_sec=0.0,
                worker_id=self.worker_url,
                status="error",
                error="Request timeout",
            )
        except json.JSONDecodeError as e:
            return TaskResult(
                task_id=task.get("task_id", ""),
                candidate_id=task.get("candidate_id", -1),
                fitness=0.0,
                games_played=0,
                evaluation_time_sec=0.0,
                worker_id=self.worker_url,
                status="error",
                error=f"Invalid JSON response: {e}",
            )
        except OSError as e:
            # Catch-all for socket/network errors not covered by URLError
            return TaskResult(
                task_id=task.get("task_id", ""),
                candidate_id=task.get("candidate_id", -1),
                fitness=0.0,
                games_played=0,
                evaluation_time_sec=0.0,
                worker_id=self.worker_url,
                status="error",
                error=f"Network error: {e}",
            )


class DistributedEvaluator:
    """
    Distributed evaluator for CMA-ES population evaluation.

    Distributes candidate evaluation tasks across multiple workers
    using round-robin assignment with automatic retry on failure.
    """

    def __init__(
        self,
        workers: List[str],
        board_type: str = "square8",
        num_players: int = 2,
        games_per_eval: int = 24,
        eval_mode: str = "multi-start",
        state_pool_id: str = "v1",
        max_moves: int = 10000,
        eval_randomness: float = 0.0,
        seed: Optional[int] = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        fallback_fitness: float = 0.0,
        record_games: bool = False,
    ):
        """
        Initialize the distributed evaluator.

        Parameters
        ----------
        workers : List[str]
            List of worker URLs in format "host:port"
        board_type : str
            Board type for evaluation
        num_players : int
            Number of players
        games_per_eval : int
            Games per candidate evaluation
        eval_mode : str
            Evaluation mode ("initial-only" or "multi-start")
        state_pool_id : str
            State pool ID for multi-start mode
        max_moves : int
            Maximum moves per game
        eval_randomness : float
            Randomness parameter for evaluation
        seed : Optional[int]
            Base seed for reproducibility
        timeout : float
            Request timeout in seconds
        max_retries : int
            Maximum retries for failed tasks
        fallback_fitness : float
            Fitness to use when all retries fail
        record_games : bool
            If True, workers will record games and return them with results
        """
        self.workers = workers
        self.board_type = board_type
        self.num_players = num_players
        self.games_per_eval = games_per_eval
        self.eval_mode = eval_mode
        self.state_pool_id = state_pool_id
        self.max_moves = max_moves
        self.eval_randomness = eval_randomness
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_fitness = fallback_fitness
        self.record_games = record_games

        # Create clients for each worker
        self._clients = {url: WorkerClient(url, timeout) for url in workers}

        # Track worker health
        self._healthy_workers: List[str] = []
        self._worker_task_idx = 0

    def verify_workers(self, check_memory: bool = True) -> List[str]:
        """
        Verify which workers are healthy and can handle the configured board type.

        Parameters
        ----------
        check_memory : bool
            If True, also verify workers have enough RAM for the board type

        Returns list of healthy worker URLs that can handle the workload.
        """
        healthy = []
        eligible = []
        ineligible = []

        for url, client in self._clients.items():
            if not client.is_healthy():
                logger.warning(f"Worker {url} is not healthy")
                continue

            healthy.append(url)

            if check_memory:
                memory_info = client.get_memory_info()
                total_gb = memory_info.get("total_gb", 0)
                eligible_boards = memory_info.get("eligible_boards", [])

                can_handle = (
                    self.board_type in eligible_boards or
                    total_gb >= BOARD_MEMORY_REQUIREMENTS.get(self.board_type, 8)
                )

                if can_handle:
                    eligible.append(url)
                    logger.info(
                        f"Worker {url} is healthy and eligible for {self.board_type} "
                        f"({total_gb}GB RAM, eligible: {eligible_boards})"
                    )
                else:
                    ineligible.append(url)
                    logger.warning(
                        f"Worker {url} is healthy but cannot handle {self.board_type} "
                        f"(has {total_gb}GB RAM, needs {BOARD_MEMORY_REQUIREMENTS.get(self.board_type, 8)}GB)"
                    )
            else:
                eligible.append(url)
                logger.info(f"Worker {url} is healthy")

        if check_memory:
            self._healthy_workers = eligible
            if ineligible:
                logger.info(
                    f"Memory-aware filtering: {len(eligible)} workers eligible, "
                    f"{len(ineligible)} excluded for {self.board_type}"
                )
        else:
            self._healthy_workers = healthy

        return self._healthy_workers

    def preload_pools(self) -> Dict[str, Any]:
        """
        Preload state pools on all workers.

        Returns dict mapping worker URL to preload result.
        """
        results = {}
        for url, client in self._clients.items():
            result = client.preload_pool(
                self.board_type,
                self.num_players,
                self.state_pool_id,
            )
            results[url] = result
            if result.get("status") == "success":
                logger.info(
                    f"Preloaded pool on {url}: "
                    f"{result.get('pool_size', 0)} states"
                )
            else:
                logger.warning(
                    f"Failed to preload pool on {url}: "
                    f"{result.get('error', 'unknown error')}"
                )
        return results

    def _get_next_worker(self) -> str:
        """Get next worker URL in round-robin fashion."""
        if not self._healthy_workers:
            self._healthy_workers = self.verify_workers()
            if not self._healthy_workers:
                raise RuntimeError("No healthy workers available")

        worker = self._healthy_workers[self._worker_task_idx % len(self._healthy_workers)]
        self._worker_task_idx += 1
        return worker

    def _create_task(
        self,
        candidate_id: int,
        weights: HeuristicWeights,
        baseline_weights: HeuristicWeights,
    ) -> Dict[str, Any]:
        """Create a task specification for a candidate."""
        return {
            "task_id": str(uuid.uuid4()),
            "candidate_id": candidate_id,
            "weights": weights,
            "baseline_weights": baseline_weights,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "games_per_eval": self.games_per_eval,
            "eval_mode": self.eval_mode,
            "state_pool_id": self.state_pool_id,
            "max_moves": self.max_moves,
            "eval_randomness": self.eval_randomness,
            "seed": self.seed,
            "record_games": self.record_games,
        }

    def _evaluate_with_retry(
        self,
        task: Dict[str, Any],
        worker_url: str,
    ) -> TaskResult:
        """Evaluate a task with retry logic."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            client = self._clients.get(worker_url)
            if not client:
                logger.error(f"No client for worker {worker_url}")
                break

            result = client.evaluate(task)
            if result.status == "success":
                return result

            last_error = result.error
            logger.warning(
                f"Task {task['task_id']} failed on {worker_url} "
                f"(attempt {attempt + 1}/{self.max_retries + 1}): {last_error}"
            )

            # Try a different worker for retry
            if attempt < self.max_retries:
                worker_url = self._get_next_worker()

        # All retries failed
        return TaskResult(
            task_id=task.get("task_id", ""),
            candidate_id=task.get("candidate_id", -1),
            fitness=self.fallback_fitness,
            games_played=0,
            evaluation_time_sec=0.0,
            worker_id="fallback",
            status="error",
            error=f"All retries failed: {last_error}",
        )

    def evaluate_population(
        self,
        population: List[HeuristicWeights],
        baseline_weights: Optional[HeuristicWeights] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[float], EvaluationStats]:
        """
        Evaluate a population of candidates in parallel across workers.

        Parameters
        ----------
        population : List[HeuristicWeights]
            List of candidate weight dicts
        baseline_weights : Optional[HeuristicWeights]
            Baseline weights for evaluation (default: BASE_V1_BALANCED_WEIGHTS)
        progress_callback : Optional[Callable[[int, int], None]]
            Callback(completed, total) for progress reporting

        Returns
        -------
        Tuple[List[float], EvaluationStats]
            Tuple of (fitness_scores, statistics)
        """
        if not population:
            return [], EvaluationStats()

        if baseline_weights is None:
            baseline_weights = BASE_V1_BALANCED_WEIGHTS

        # Verify workers
        healthy = self.verify_workers()
        if not healthy:
            raise RuntimeError("No healthy workers available")

        logger.info(
            f"Evaluating {len(population)} candidates across "
            f"{len(healthy)} workers"
        )

        # Create tasks
        tasks = []
        for i, weights in enumerate(population):
            task = self._create_task(i, weights, baseline_weights)
            worker = self._get_next_worker()
            tasks.append((task, worker))

        # Execute in parallel
        start_time = time.time()
        results: Dict[int, TaskResult] = {}
        stats = EvaluationStats()

        with ThreadPoolExecutor(max_workers=len(healthy)) as executor:
            futures = {}
            for task, worker in tasks:
                future = executor.submit(
                    self._evaluate_with_retry, task, worker
                )
                futures[future] = task["candidate_id"]

            completed = 0
            for future in as_completed(futures):
                candidate_id = futures[future]
                try:
                    result = future.result()
                    results[result.candidate_id] = result
                    stats.add_result(result)
                except Exception as e:
                    logger.exception(f"Task for candidate {candidate_id} failed: {e}")
                    results[candidate_id] = TaskResult(
                        task_id="",
                        candidate_id=candidate_id,
                        fitness=self.fallback_fitness,
                        games_played=0,
                        evaluation_time_sec=0.0,
                        worker_id="error",
                        status="error",
                        error=str(e),
                    )
                    stats.failed_evaluations += 1

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(population))

        stats.total_time_sec = time.time() - start_time

        # Build fitness list in order
        fitness_scores = []
        for i in range(len(population)):
            if i in results:
                fitness_scores.append(results[i].fitness)
            else:
                logger.warning(f"No result for candidate {i}, using fallback")
                fitness_scores.append(self.fallback_fitness)

        logger.info(
            f"Population evaluation complete: "
            f"{stats.successful_evaluations}/{len(population)} successful, "
            f"{stats.total_games} games, {stats.total_time_sec:.1f}s"
        )

        return fitness_scores, stats

    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all workers."""
        stats = {}
        for url, client in self._clients.items():
            stats[url] = client.get_stats()
        return stats


class QueueDistributedEvaluator:
    """
    Queue-based distributed evaluator for cloud CMA-ES deployment.

    Unlike DistributedEvaluator (which uses direct HTTP to workers),
    this class publishes tasks to a queue (Redis or SQS) and collects
    results. Workers running cmaes_cloud_worker.py pull tasks from the
    queue and push results back.

    This mode is designed for:
    - Docker/ECS/K8s deployments
    - Auto-scaling worker pools
    - AWS Spot instances with graceful shutdown
    """

    def __init__(
        self,
        queue_backend: str = "redis",
        board_type: str = "square8",
        num_players: int = 2,
        games_per_eval: int = 24,
        eval_mode: str = "multi-start",
        state_pool_id: str = "v1",
        max_moves: int = 10000,
        eval_randomness: float = 0.0,
        seed: Optional[int] = None,
        timeout: float = 300.0,
        fallback_fitness: float = 0.0,
        run_id: Optional[str] = None,
        record_games: bool = False,
        **queue_kwargs,
    ):
        """
        Initialize the queue-based distributed evaluator.

        Parameters
        ----------
        queue_backend : str
            Queue backend type: "redis" or "sqs"
        board_type : str
            Board type for evaluation
        num_players : int
            Number of players
        games_per_eval : int
            Games per candidate evaluation
        eval_mode : str
            Evaluation mode ("initial-only" or "multi-start")
        state_pool_id : str
            State pool ID for multi-start mode
        max_moves : int
            Maximum moves per game
        eval_randomness : float
            Randomness parameter for evaluation
        seed : Optional[int]
            Base seed for reproducibility
        timeout : float
            Overall timeout for collecting results (seconds)
        fallback_fitness : float
            Fitness to use when task fails
        run_id : Optional[str]
            Run ID for tracking tasks
        record_games : bool
            If True, workers will record games and return them with results
        **queue_kwargs
            Additional arguments passed to get_task_queue()
        """
        from .queue import get_task_queue

        self.queue = get_task_queue(backend=queue_backend, **queue_kwargs)
        self.board_type = board_type
        self.num_players = num_players
        self.games_per_eval = games_per_eval
        self.eval_mode = eval_mode
        self.state_pool_id = state_pool_id
        self.max_moves = max_moves
        self.eval_randomness = eval_randomness
        self.seed = seed
        self.timeout = timeout
        self.fallback_fitness = fallback_fitness
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.record_games = record_games

        self._generation = 0

    def close(self) -> None:
        """Close the queue connection."""
        self.queue.close()

    def evaluate_population(
        self,
        population: List[HeuristicWeights],
        baseline_weights: Optional[HeuristicWeights] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        generation: int = 0,
    ) -> Tuple[List[float], EvaluationStats]:
        """
        Evaluate a population of candidates via queue-based workers.

        Publishes all tasks to the queue, then collects results as
        workers complete them.

        Parameters
        ----------
        population : List[HeuristicWeights]
            List of candidate weight dicts
        baseline_weights : Optional[HeuristicWeights]
            Baseline weights for evaluation (default: BASE_V1_BALANCED_WEIGHTS)
        progress_callback : Optional[Callable[[int, int], None]]
            Callback(completed, total) for progress reporting
        generation : int
            Current generation number (for task metadata)

        Returns
        -------
        Tuple[List[float], EvaluationStats]
            Tuple of (fitness_scores, statistics)
        """
        from .queue import EvalTask

        if not population:
            return [], EvaluationStats()

        if baseline_weights is None:
            baseline_weights = BASE_V1_BALANCED_WEIGHTS

        self._generation = generation
        population_size = len(population)

        logger.info(
            f"Publishing {population_size} tasks to queue for generation {generation}"
        )

        # Publish all tasks
        task_ids: Dict[str, int] = {}  # task_id -> candidate_id
        start_time = time.time()

        for candidate_id, weights in enumerate(population):
            task_id = f"{self.run_id}_gen{generation}_cand{candidate_id}"
            task = EvalTask(
                task_id=task_id,
                candidate_id=candidate_id,
                weights=weights,
                board_type=self.board_type,
                num_players=self.num_players,
                games_per_eval=self.games_per_eval,
                eval_mode=self.eval_mode,
                state_pool_id=self.state_pool_id,
                max_moves=self.max_moves,
                eval_randomness=self.eval_randomness,
                seed=self.seed,
                generation=generation,
                run_id=self.run_id,
                baseline_weights=baseline_weights,
                record_games=self.record_games,
            )
            self.queue.publish_task(task)
            task_ids[task_id] = candidate_id

        logger.info(f"Published {population_size} tasks, collecting results...")

        # Collect results
        results: Dict[int, float] = {}
        stats = EvaluationStats()
        stats.total_candidates = population_size

        def on_progress(results_so_far: list) -> None:
            if progress_callback:
                progress_callback(len(results_so_far), population_size)

        collected = self.queue.consume_results(
            count=population_size,
            timeout=self.timeout,
            progress_callback=on_progress,
        )

        for result in collected:
            candidate_id = result.candidate_id
            results[candidate_id] = result.fitness

            if result.status == "success":
                stats.successful_evaluations += 1
                stats.total_games += result.games_played
                # Aggregate game replays if present
                if result.game_replays:
                    stats.all_game_replays.extend(result.game_replays)
            else:
                stats.failed_evaluations += 1
                logger.warning(
                    f"Task {result.task_id} failed: {result.error}"
                )

            worker = result.worker_id
            stats.worker_task_counts[worker] = (
                stats.worker_task_counts.get(worker, 0) + 1
            )

        stats.total_time_sec = time.time() - start_time

        # Build fitness list in order, using fallback for missing results
        fitness_scores = []
        for candidate_id in range(population_size):
            if candidate_id in results:
                fitness_scores.append(results[candidate_id])
            else:
                logger.warning(
                    f"No result for candidate {candidate_id}, using fallback"
                )
                fitness_scores.append(self.fallback_fitness)
                stats.failed_evaluations += 1

        logger.info(
            f"Queue evaluation complete: "
            f"{stats.successful_evaluations}/{population_size} successful, "
            f"{stats.total_games} games, {stats.total_time_sec:.1f}s"
        )

        return fitness_scores, stats

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics for monitoring."""
        try:
            if hasattr(self.queue, "get_queue_lengths"):
                return self.queue.get_queue_lengths()
            elif hasattr(self.queue, "get_queue_attributes"):
                return self.queue.get_queue_attributes()
            return {}
        except (AttributeError, RuntimeError, TypeError) as e:
            # AttributeError: method doesn't exist, RuntimeError: queue invalid state
            # TypeError: method called with wrong arguments
            return {"error": str(e)}
