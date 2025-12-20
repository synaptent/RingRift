from __future__ import annotations

"""
Async/batched neural network evaluation helpers.

This module provides a thread-safe wrapper around ``NeuralNetAI.evaluate_batch`` so
search engines can evaluate speculative GameStates asynchronously without risking
concurrent model access.

When enabled (default: CUDA), evaluations are routed through a global
micro-batching worker that coalesces multiple submit calls into larger batches,
improving GPU throughput while keeping model usage serialized.
"""

import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..models import GameState
from .neural_net import NeuralNetAI


def _is_truthy_env(name: str) -> bool | None:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return None
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return None


def _device_type(device: object) -> str:
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(getattr(device, "type", "cpu"))


def _should_use_microbatcher(neural_net: NeuralNetAI) -> bool:
    override = _is_truthy_env("RINGRIFT_NN_EVAL_QUEUE")
    if override is not None:
        return override
    return _device_type(getattr(neural_net, "device", "cpu")) == "cuda"


def _parse_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


@dataclass(frozen=True)
class _EvalRequest:
    neural_net: NeuralNetAI
    game_states: list[GameState]
    future: Future
    value_head: int | None = None


class _GlobalNNMicroBatcher:
    def __init__(
        self,
        *,
        max_batch_states: int,
        batch_timeout_ms: int,
    ) -> None:
        self._max_batch_states = max(1, int(max_batch_states))
        self._batch_timeout_s = max(0.0, float(batch_timeout_ms) / 1000.0)

        self._queue: queue.Queue[_EvalRequest] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ringrift-nn-microbatcher",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        neural_net: NeuralNetAI,
        game_states: list[GameState],
        *,
        value_head: int | None = None,
    ) -> Future:
        fut: Future = Future()
        self._queue.put(
            _EvalRequest(
                neural_net=neural_net,
                game_states=game_states,
                future=fut,
                value_head=value_head,
            )
        )
        return fut

    def shutdown(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        pending: deque[_EvalRequest] = deque()

        while not self._stop.is_set():
            if not pending:
                try:
                    # Block briefly for new work; this keeps CPU usage low when idle.
                    pending.append(self._queue.get(timeout=0.05))
                except queue.Empty:
                    continue

            if self._batch_timeout_s > 0:
                deadline = time.monotonic() + self._batch_timeout_s
                while time.monotonic() < deadline and len(pending) < 10_000:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        pending.append(self._queue.get(timeout=remaining))
                    except queue.Empty:
                        break

            # Process a compatible group starting from the oldest request.
            self._process_one_group(pending)

    def _process_one_group(self, pending: deque[_EvalRequest]) -> None:
        if not pending:
            return

        head = pending[0]
        try:
            head_key = self._request_key(head)
        except Exception as e:
            pending.popleft()
            head.future.set_exception(e)
            return

        group: list[_EvalRequest] = []
        deferred: deque[_EvalRequest] = deque()
        total_states = 0

        while pending:
            req = pending.popleft()
            try:
                key = self._request_key(req)
            except Exception as e:
                req.future.set_exception(e)
                continue

            req_states = len(req.game_states)
            if key == head_key and (
                total_states == 0 or total_states + req_states <= self._max_batch_states
            ):
                group.append(req)
                total_states += req_states
            else:
                deferred.append(req)

        pending.extendleft(reversed(deferred))

        if not group:
            return

        # If the head request alone exceeds max_batch_states, evaluate it alone.
        if len(group) > 1 and len(group[0].game_states) > self._max_batch_states:
            pending.extendleft(reversed(group[1:]))
            group = [group[0]]

        rep = group[0].neural_net
        combined_states: list[GameState] = []
        offsets: list[tuple[_EvalRequest, int, int]] = []
        for req in group:
            start = len(combined_states)
            combined_states.extend(req.game_states)
            end = len(combined_states)
            offsets.append((req, start, end))

        try:
            value_head = group[0].value_head
            values, policies = rep.evaluate_batch(combined_states, value_head=value_head)
            for req, start, end in offsets:
                req.future.set_result((values[start:end], policies[start:end]))
        except Exception as e:
            for req, _, _ in offsets:
                req.future.set_exception(e)

    def _request_key(self, req: _EvalRequest) -> tuple:
        game_states = req.game_states
        if not game_states:
            # Empty batches should not be common in callers; keep them isolated.
            return ("empty", id(req.neural_net))

        first_board = game_states[0].board
        first_type = first_board.type
        first_size = first_board.size
        for state in game_states[1:]:
            if state.board.type != first_type or state.board.size != first_size:
                raise ValueError(
                    "AsyncNeuralBatcher requires all game_states in a submit() "
                    "call to share the same board.type and board.size."
                )

        # Ensure the model exists so we can safely coalesce by underlying weights.
        req.neural_net._ensure_model_initialized(first_type)
        model = getattr(req.neural_net, "model", None)
        if model is None:
            raise RuntimeError("NeuralNetAI model was not initialized")

        device_key = str(getattr(req.neural_net, "device", "cpu"))
        history_len = int(getattr(req.neural_net, "history_length", 0) or 0)
        has_history = bool(getattr(req.neural_net, "game_history", {}))

        # Coalesce by model only when the wrapper has no per-game history state.
        share_key = ("instance", id(req.neural_net)) if has_history else ("model", id(model))

        return (
            device_key,
            str(first_type),
            int(first_size),
            history_len,
            share_key,
            req.value_head,
        )


_GLOBAL_BATCHER: _GlobalNNMicroBatcher | None = None
_GLOBAL_BATCHER_LOCK = threading.RLock()
_GLOBAL_BATCHER_REFS = 0


def _acquire_global_batcher() -> _GlobalNNMicroBatcher:
    global _GLOBAL_BATCHER, _GLOBAL_BATCHER_REFS
    with _GLOBAL_BATCHER_LOCK:
        if _GLOBAL_BATCHER is None:
            _GLOBAL_BATCHER = _GlobalNNMicroBatcher(
                max_batch_states=_parse_positive_int("RINGRIFT_NN_EVAL_MAX_BATCH", 256),
                batch_timeout_ms=_parse_positive_int("RINGRIFT_NN_EVAL_BATCH_TIMEOUT_MS", 2),
            )
        _GLOBAL_BATCHER_REFS += 1
        return _GLOBAL_BATCHER


def _release_global_batcher() -> None:
    global _GLOBAL_BATCHER, _GLOBAL_BATCHER_REFS
    with _GLOBAL_BATCHER_LOCK:
        _GLOBAL_BATCHER_REFS = max(0, _GLOBAL_BATCHER_REFS - 1)
        if _GLOBAL_BATCHER_REFS == 0 and _GLOBAL_BATCHER is not None:
            _GLOBAL_BATCHER.shutdown()
            _GLOBAL_BATCHER = None


class AsyncNeuralBatcher:
    """Thread-safe batcher for NeuralNetAI.

    - `evaluate(...)` runs a synchronous batched evaluation under a lock.
    - `submit(...)` schedules a batched evaluation on a single worker thread,
      also protected by the same lock.

    The lock ensures any direct calls to NeuralNetAI from the main thread
    will serialize safely with background evaluations.
    """

    def __init__(
        self,
        neural_net: NeuralNetAI,
        max_workers: int = 1,
    ) -> None:
        self.neural_net = neural_net
        self._lock = threading.RLock()
        self._use_microbatcher = _should_use_microbatcher(neural_net)
        self._global_batcher: _GlobalNNMicroBatcher | None = (
            _acquire_global_batcher() if self._use_microbatcher else None
        )
        self._executor: ThreadPoolExecutor | None = (
            None
            if self._use_microbatcher
            else ThreadPoolExecutor(max_workers=max_workers)
        )

    def evaluate(
        self,
        game_states: list[GameState],
        *,
        value_head: int | None = None,
    ) -> tuple[list[float], np.ndarray]:
        if self._use_microbatcher and self._global_batcher is not None:
            return self.submit(game_states, value_head=value_head).result()

        with self._lock:
            return self.neural_net.evaluate_batch(game_states, value_head=value_head)

    def submit(
        self,
        game_states: list[GameState],
        *,
        value_head: int | None = None,
    ) -> Future:
        if self._use_microbatcher and self._global_batcher is not None:
            return self._global_batcher.submit(
                self.neural_net,
                game_states,
                value_head=value_head,
            )

        if self._executor is None:
            raise RuntimeError("AsyncNeuralBatcher is shut down")

        def _run():
            with self._lock:
                return self.neural_net.evaluate_batch(game_states, value_head=value_head)

        return self._executor.submit(_run)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._global_batcher is not None:
            _release_global_batcher()
            self._global_batcher = None
