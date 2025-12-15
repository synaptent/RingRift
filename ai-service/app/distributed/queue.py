"""Queue abstraction layer for distributed CMA-ES evaluation.

Provides a unified interface for task distribution across:
- Redis (local development, small-scale cloud)
- AWS SQS (production cloud deployment)

Usage:
    from app.distributed.queue import get_task_queue, EvalTask, EvalResult

    # Get queue based on environment
    queue = get_task_queue()

    # Coordinator: publish tasks
    task = EvalTask(task_id="...", candidate_id=0, weights={...}, ...)
    queue.publish_task(task)

    # Worker: consume tasks
    task = queue.consume_task(timeout=30)
    result = evaluate(task)
    queue.publish_result(result)

    # Coordinator: collect results
    results = queue.consume_results(count=16, timeout=60)

Environment variables:
    QUEUE_BACKEND: redis, sqs (default: redis)
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    SQS_TASK_QUEUE_URL: SQS queue URL for tasks
    SQS_RESULT_QUEUE_URL: SQS queue URL for results
    SQS_REGION: AWS region (default: us-east-1)
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class EvalTask:
    """Evaluation task sent to workers."""

    task_id: str
    candidate_id: int
    weights: Dict[str, float]
    board_type: str = "square8"
    num_players: int = 2
    games_per_eval: int = 24
    eval_mode: str = "multi-start"
    state_pool_id: str = "v1"
    max_moves: int = 10000
    eval_randomness: float = 0.0
    seed: Optional[int] = None

    # Metadata for tracking
    generation: int = 0
    run_id: str = ""

    # Optional baseline weights for evaluation
    baseline_weights: Optional[Dict[str, float]] = None

    # Game recording options
    record_games: bool = False  # If True, worker returns serialized game data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalTask:
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> EvalTask:
        return cls.from_dict(json.loads(json_str))


@dataclass
class GameReplayData:
    """Serialized game replay data for transfer from worker to coordinator."""

    game_id: str
    initial_state: Dict[str, Any]  # Serialized GameState
    final_state: Dict[str, Any]  # Serialized GameState
    moves: List[Dict[str, Any]]  # Serialized moves
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameReplayData":
        return cls(**data)


@dataclass
class EvalResult:
    """Evaluation result returned by workers."""

    task_id: str
    candidate_id: int
    fitness: float
    games_played: int = 0
    wins: int = 0
    avg_moves: float = 0.0
    evaluation_time_sec: float = 0.0
    worker_id: str = ""
    status: str = "success"  # success, error, timeout
    error: Optional[str] = None

    # Optional game replay data (only populated if task.record_games=True)
    game_replays: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalResult:
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> EvalResult:
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Abstract Queue Interface
# ---------------------------------------------------------------------------


class TaskQueue(ABC):
    """Abstract base class for task queues."""

    @abstractmethod
    def publish_task(self, task: EvalTask) -> None:
        """Publish an evaluation task to the queue."""
        pass

    @abstractmethod
    def consume_task(self, timeout: float = 30.0) -> Optional[EvalTask]:
        """Consume a task from the queue.

        Args:
            timeout: Seconds to wait for a task

        Returns:
            EvalTask or None if timeout
        """
        pass

    @abstractmethod
    def ack_task(self, task_id: str) -> None:
        """Acknowledge task completion (for at-least-once delivery)."""
        pass

    @abstractmethod
    def publish_result(self, result: EvalResult) -> None:
        """Publish an evaluation result."""
        pass

    @abstractmethod
    def consume_result(self, timeout: float = 5.0) -> Optional[EvalResult]:
        """Consume a result from the queue."""
        pass

    def consume_results(
        self,
        count: int,
        timeout: float = 60.0,
        progress_callback: Optional[callable] = None,
    ) -> List[EvalResult]:
        """Consume multiple results with overall timeout.

        Args:
            count: Number of results to collect
            timeout: Total timeout for all results
            progress_callback: Optional callback(results_so_far)

        Returns:
            List of results (may be fewer than count on timeout)
        """
        results = []
        deadline = time.time() + timeout

        while len(results) < count and time.time() < deadline:
            remaining = deadline - time.time()
            result = self.consume_result(timeout=min(remaining, 5.0))
            if result:
                results.append(result)
                if progress_callback:
                    progress_callback(results)

        return results

    @abstractmethod
    def close(self) -> None:
        """Close the queue connection."""
        pass


# ---------------------------------------------------------------------------
# Redis Implementation
# ---------------------------------------------------------------------------


class RedisQueue(TaskQueue):
    """Redis-based task queue using Redis Lists and Pub/Sub.

    Suitable for local development and small-scale cloud deployments.
    Uses BRPOPLPUSH for reliable task delivery.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        task_queue: str = "ringrift:cmaes:tasks",
        result_queue: str = "ringrift:cmaes:results",
        processing_queue: str = "ringrift:cmaes:processing",
        visibility_timeout: float = 300.0,
    ):
        """Initialize Redis queue.

        Args:
            redis_url: Redis connection URL
            task_queue: Name of task queue
            result_queue: Name of result queue
            processing_queue: Name of processing queue (for reliability)
            visibility_timeout: Seconds before unacked task is requeued
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis is required for RedisQueue. Install with: pip install redis"
            )

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._processing_queue = processing_queue
        self._visibility_timeout = visibility_timeout

        # Test connection
        self._redis.ping()
        logger.info(f"Connected to Redis at {redis_url}")

    def publish_task(self, task: EvalTask) -> None:
        """Push task to the left of the task queue."""
        self._redis.lpush(self._task_queue, task.to_json())
        logger.debug(f"Published task {task.task_id}")

    def consume_task(self, timeout: float = 30.0) -> Optional[EvalTask]:
        """Pop task from the right (FIFO order)."""
        # Use BRPOP for blocking wait
        result = self._redis.brpop(self._task_queue, timeout=int(timeout))
        if result:
            _, json_str = result
            task = EvalTask.from_json(json_str)
            # Store in processing queue for reliability
            self._redis.hset(
                self._processing_queue,
                task.task_id,
                json.dumps({"task": json_str, "started": time.time()}),
            )
            logger.debug(f"Consumed task {task.task_id}")
            return task
        return None

    def ack_task(self, task_id: str) -> None:
        """Remove task from processing queue."""
        self._redis.hdel(self._processing_queue, task_id)
        logger.debug(f"Acked task {task_id}")

    def publish_result(self, result: EvalResult) -> None:
        """Push result to the result queue."""
        # Also ack the task
        self.ack_task(result.task_id)
        self._redis.lpush(self._result_queue, result.to_json())
        logger.debug(f"Published result for task {result.task_id}")

    def consume_result(self, timeout: float = 5.0) -> Optional[EvalResult]:
        """Pop result from the result queue."""
        result = self._redis.brpop(self._result_queue, timeout=int(timeout))
        if result:
            _, json_str = result
            return EvalResult.from_json(json_str)
        return None

    def get_queue_lengths(self) -> Dict[str, int]:
        """Get current queue lengths for monitoring."""
        return {
            "tasks": self._redis.llen(self._task_queue),
            "results": self._redis.llen(self._result_queue),
            "processing": self._redis.hlen(self._processing_queue),
        }

    def clear_all(self) -> None:
        """Clear all queues (use with caution)."""
        self._redis.delete(
            self._task_queue,
            self._result_queue,
            self._processing_queue,
        )
        logger.warning("Cleared all queues")

    def requeue_stale_tasks(self) -> int:
        """Requeue tasks that have been processing too long.

        Returns:
            Number of tasks requeued
        """
        requeued = 0
        now = time.time()

        for task_id in self._redis.hkeys(self._processing_queue):
            data = self._redis.hget(self._processing_queue, task_id)
            if data:
                info = json.loads(data)
                if now - info["started"] > self._visibility_timeout:
                    # Task has been processing too long, requeue it
                    self._redis.lpush(self._task_queue, info["task"])
                    self._redis.hdel(self._processing_queue, task_id)
                    requeued += 1
                    logger.warning(f"Requeued stale task {task_id}")

        return requeued

    def close(self) -> None:
        """Close Redis connection."""
        self._redis.close()


# ---------------------------------------------------------------------------
# AWS SQS Implementation
# ---------------------------------------------------------------------------


class SQSQueue(TaskQueue):
    """AWS SQS-based task queue for production cloud deployment.

    Uses two SQS queues: one for tasks, one for results.
    Supports automatic message visibility timeout and dead letter queues.
    """

    def __init__(
        self,
        task_queue_url: str,
        result_queue_url: str,
        region: str = "us-east-1",
        visibility_timeout: int = 300,
        max_receive_count: int = 3,
    ):
        """Initialize SQS queues.

        Args:
            task_queue_url: URL of the task SQS queue
            result_queue_url: URL of the result SQS queue
            region: AWS region
            visibility_timeout: Seconds before unacked message is redelivered
            max_receive_count: Max retries before dead-lettering
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for SQS. Install with: pip install boto3"
            )

        self._sqs = boto3.client("sqs", region_name=region)
        self._task_queue_url = task_queue_url
        self._result_queue_url = result_queue_url
        self._visibility_timeout = visibility_timeout

        # Map task_id to receipt handle for acknowledgement
        self._receipt_handles: Dict[str, str] = {}

        logger.info(f"Connected to SQS queues in {region}")

    def publish_task(self, task: EvalTask) -> None:
        """Send task message to SQS."""
        self._sqs.send_message(
            QueueUrl=self._task_queue_url,
            MessageBody=task.to_json(),
            MessageAttributes={
                "task_id": {"StringValue": task.task_id, "DataType": "String"},
                "candidate_id": {
                    "StringValue": str(task.candidate_id),
                    "DataType": "Number",
                },
                "generation": {
                    "StringValue": str(task.generation),
                    "DataType": "Number",
                },
            },
        )
        logger.debug(f"Published task {task.task_id} to SQS")

    def consume_task(self, timeout: float = 30.0) -> Optional[EvalTask]:
        """Receive task message from SQS."""
        # SQS WaitTimeSeconds maxes at 20
        wait_time = min(int(timeout), 20)

        response = self._sqs.receive_message(
            QueueUrl=self._task_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_time,
            VisibilityTimeout=self._visibility_timeout,
            MessageAttributeNames=["All"],
        )

        messages = response.get("Messages", [])
        if not messages:
            return None

        message = messages[0]
        task = EvalTask.from_json(message["Body"])

        # Store receipt handle for acknowledgement
        self._receipt_handles[task.task_id] = message["ReceiptHandle"]

        logger.debug(f"Consumed task {task.task_id} from SQS")
        return task

    def ack_task(self, task_id: str) -> None:
        """Delete message from queue (acknowledge processing)."""
        receipt_handle = self._receipt_handles.pop(task_id, None)
        if receipt_handle:
            self._sqs.delete_message(
                QueueUrl=self._task_queue_url,
                ReceiptHandle=receipt_handle,
            )
            logger.debug(f"Acked task {task_id}")

    def publish_result(self, result: EvalResult) -> None:
        """Send result message to SQS."""
        # Ack the original task first
        self.ack_task(result.task_id)

        self._sqs.send_message(
            QueueUrl=self._result_queue_url,
            MessageBody=result.to_json(),
            MessageAttributes={
                "task_id": {"StringValue": result.task_id, "DataType": "String"},
                "candidate_id": {
                    "StringValue": str(result.candidate_id),
                    "DataType": "Number",
                },
                "status": {"StringValue": result.status, "DataType": "String"},
            },
        )
        logger.debug(f"Published result for task {result.task_id}")

    def consume_result(self, timeout: float = 5.0) -> Optional[EvalResult]:
        """Receive result message from SQS."""
        wait_time = min(int(timeout), 20)

        response = self._sqs.receive_message(
            QueueUrl=self._result_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_time,
            MessageAttributeNames=["All"],
        )

        messages = response.get("Messages", [])
        if not messages:
            return None

        message = messages[0]
        result = EvalResult.from_json(message["Body"])

        # Immediately delete result messages (no retry needed)
        self._sqs.delete_message(
            QueueUrl=self._result_queue_url,
            ReceiptHandle=message["ReceiptHandle"],
        )

        return result

    def get_queue_attributes(self) -> Dict[str, Dict[str, str]]:
        """Get queue attributes for monitoring."""
        task_attrs = self._sqs.get_queue_attributes(
            QueueUrl=self._task_queue_url,
            AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
        )
        result_attrs = self._sqs.get_queue_attributes(
            QueueUrl=self._result_queue_url,
            AttributeNames=["ApproximateNumberOfMessages"],
        )
        return {
            "tasks": task_attrs.get("Attributes", {}),
            "results": result_attrs.get("Attributes", {}),
        }

    def close(self) -> None:
        """Close SQS client (no-op for boto3)."""
        pass


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def get_task_queue(
    backend: Optional[str] = None,
    **kwargs,
) -> TaskQueue:
    """Get a task queue based on configuration.

    Args:
        backend: Queue type: redis, sqs (default from env)
        **kwargs: Backend-specific configuration

    Returns:
        Configured TaskQueue instance

    Environment variables:
        QUEUE_BACKEND: redis, sqs (default: redis)
        REDIS_URL: Redis connection URL
        SQS_TASK_QUEUE_URL: SQS task queue URL
        SQS_RESULT_QUEUE_URL: SQS result queue URL
        SQS_REGION: AWS region
    """
    backend = backend or os.getenv("QUEUE_BACKEND", "redis")
    backend = backend.lower()

    if backend == "redis":
        redis_url = kwargs.get("redis_url") or os.getenv(
            "REDIS_URL", "redis://localhost:6379"
        )
        return RedisQueue(redis_url=redis_url, **kwargs)

    elif backend == "sqs":
        task_queue_url = kwargs.get("task_queue_url") or os.getenv(
            "SQS_TASK_QUEUE_URL"
        )
        result_queue_url = kwargs.get("result_queue_url") or os.getenv(
            "SQS_RESULT_QUEUE_URL"
        )
        region = kwargs.get("region") or os.getenv("SQS_REGION", "us-east-1")

        if not task_queue_url or not result_queue_url:
            raise ValueError(
                "SQS_TASK_QUEUE_URL and SQS_RESULT_QUEUE_URL are required for SQS backend"
            )

        return SQSQueue(
            task_queue_url=task_queue_url,
            result_queue_url=result_queue_url,
            region=region,
        )

    else:
        raise ValueError(
            f"Unknown queue backend: {backend}. Supported: redis, sqs"
        )
