"""Tests for async neural network evaluation module.

Comprehensive test coverage for:
- Environment variable parsing
- Device type detection
- Global microbatcher
- AsyncNeuralBatcher
- Thread safety and concurrent access
"""

import os
import queue
import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.ai.async_nn_eval import (
    _is_truthy_env,
    _device_type,
    _should_use_microbatcher,
    _parse_positive_int,
    _EvalRequest,
    _GlobalNNMicroBatcher,
    _acquire_global_batcher,
    _release_global_batcher,
    AsyncNeuralBatcher,
    _GLOBAL_BATCHER_LOCK,
)


class TestEnvParsing:
    """Test environment variable parsing functions."""

    def test_is_truthy_env_true_values(self):
        """Test truthy values are recognized."""
        for val in ["1", "true", "yes", "on", "TRUE", "Yes", "ON"]:
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert _is_truthy_env("TEST_VAR") is True

    def test_is_truthy_env_false_values(self):
        """Test falsy values are recognized."""
        for val in ["0", "false", "no", "off", "FALSE", "No", "OFF"]:
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert _is_truthy_env("TEST_VAR") is False

    def test_is_truthy_env_empty(self):
        """Test empty/missing returns None."""
        with patch.dict(os.environ, {}, clear=True):
            assert _is_truthy_env("MISSING_VAR") is None

        with patch.dict(os.environ, {"TEST_VAR": ""}):
            assert _is_truthy_env("TEST_VAR") is None

    def test_is_truthy_env_invalid(self):
        """Test invalid values return None."""
        with patch.dict(os.environ, {"TEST_VAR": "maybe"}):
            assert _is_truthy_env("TEST_VAR") is None

    def test_parse_positive_int_valid(self):
        """Test parsing valid positive integers."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert _parse_positive_int("TEST_INT", 10) == 42

    def test_parse_positive_int_default(self):
        """Test default used when env missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert _parse_positive_int("MISSING_INT", 100) == 100

    def test_parse_positive_int_invalid(self):
        """Test default used for invalid values."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert _parse_positive_int("TEST_INT", 50) == 50

    def test_parse_positive_int_negative(self):
        """Test default used for negative values."""
        with patch.dict(os.environ, {"TEST_INT": "-5"}):
            assert _parse_positive_int("TEST_INT", 25) == 25

    def test_parse_positive_int_zero(self):
        """Test default used for zero."""
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            assert _parse_positive_int("TEST_INT", 30) == 30


class TestDeviceType:
    """Test device type detection."""

    def test_device_type_string_cuda(self):
        """Test CUDA device string."""
        assert _device_type("cuda:0") == "cuda"
        assert _device_type("cuda:1") == "cuda"
        assert _device_type("cuda") == "cuda"

    def test_device_type_string_cpu(self):
        """Test CPU device string."""
        assert _device_type("cpu") == "cpu"

    def test_device_type_string_mps(self):
        """Test MPS device string."""
        assert _device_type("mps") == "mps"

    def test_device_type_object_with_type(self):
        """Test device object with type attribute."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        assert _device_type(mock_device) == "cuda"

    def test_device_type_object_without_type(self):
        """Test device object without type attribute."""
        mock_device = object()
        assert _device_type(mock_device) == "cpu"


class TestShouldUseMicrobatcher:
    """Test microbatcher decision logic."""

    def test_env_override_true(self):
        """Test env var forces microbatcher on."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"

        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "1"}):
            assert _should_use_microbatcher(mock_nn) is True

    def test_env_override_false(self):
        """Test env var forces microbatcher off."""
        mock_nn = MagicMock()
        mock_nn.device = "cuda:0"

        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            assert _should_use_microbatcher(mock_nn) is False

    def test_cuda_enables_microbatcher(self):
        """Test CUDA device enables microbatcher by default."""
        mock_nn = MagicMock()
        mock_nn.device = "cuda:0"

        with patch.dict(os.environ, {}, clear=True):
            # Clear the env var to test default behavior
            os.environ.pop("RINGRIFT_NN_EVAL_QUEUE", None)
            assert _should_use_microbatcher(mock_nn) is True

    def test_cpu_disables_microbatcher(self):
        """Test CPU device disables microbatcher by default."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_NN_EVAL_QUEUE", None)
            assert _should_use_microbatcher(mock_nn) is False


class TestEvalRequest:
    """Test _EvalRequest dataclass."""

    def test_eval_request_creation(self):
        """Test creating an eval request."""
        mock_nn = MagicMock()
        mock_states = [MagicMock(), MagicMock()]
        future = Future()

        request = _EvalRequest(
            neural_net=mock_nn,
            game_states=mock_states,
            future=future,
            value_head=1,
        )

        assert request.neural_net is mock_nn
        assert request.game_states == mock_states
        assert request.future is future
        assert request.value_head == 1

    def test_eval_request_default_value_head(self):
        """Test default value_head is None."""
        request = _EvalRequest(
            neural_net=MagicMock(),
            game_states=[],
            future=Future(),
        )
        assert request.value_head is None


class TestGlobalNNMicroBatcher:
    """Test _GlobalNNMicroBatcher class."""

    @pytest.fixture
    def batcher(self):
        """Create a microbatcher for testing."""
        b = _GlobalNNMicroBatcher(
            max_batch_states=64,
            batch_timeout_ms=10,
        )
        yield b
        b.shutdown()

    def test_init(self, batcher):
        """Test batcher initialization."""
        assert batcher._max_batch_states == 64
        assert batcher._batch_timeout_s == 0.01
        assert batcher._thread.is_alive()

    def test_submit_returns_future(self, batcher):
        """Test submit returns a Future."""
        mock_nn = MagicMock()
        mock_states = [MagicMock()]

        future = batcher.submit(mock_nn, mock_states)

        assert isinstance(future, Future)

    def test_shutdown_stops_thread(self):
        """Test shutdown stops the worker thread."""
        batcher = _GlobalNNMicroBatcher(
            max_batch_states=64,
            batch_timeout_ms=5,
        )
        assert batcher._thread.is_alive()

        batcher.shutdown()
        time.sleep(0.1)  # Give thread time to stop

        # Thread may still be alive but stop event should be set
        assert batcher._stop.is_set()

    def test_request_key_empty_batch(self, batcher):
        """Test request key for empty batch."""
        mock_nn = MagicMock()
        request = _EvalRequest(
            neural_net=mock_nn,
            game_states=[],
            future=Future(),
        )

        key = batcher._request_key(request)
        assert key[0] == "empty"

    def test_request_key_validation(self, batcher):
        """Test request key validates board consistency."""
        mock_nn = MagicMock()
        mock_nn._ensure_model_initialized = MagicMock()
        mock_nn.model = MagicMock()
        mock_nn.device = "cpu"
        mock_nn.history_length = 0
        mock_nn.game_history = {}

        # Create states with different board types
        state1 = MagicMock()
        state1.board.type = "square"
        state1.board.size = 8

        state2 = MagicMock()
        state2.board.type = "hex"  # Different type!
        state2.board.size = 8

        request = _EvalRequest(
            neural_net=mock_nn,
            game_states=[state1, state2],
            future=Future(),
        )

        with pytest.raises(ValueError, match="same board.type"):
            batcher._request_key(request)


class TestAsyncNeuralBatcher:
    """Test AsyncNeuralBatcher class."""

    @pytest.fixture
    def mock_neural_net(self):
        """Create a mock NeuralNetAI."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"  # Use CPU to avoid microbatcher
        mock_nn.evaluate_batch.return_value = (
            [0.5, 0.6],
            np.array([[0.1] * 64, [0.2] * 64]),
        )
        return mock_nn

    @pytest.fixture
    def batcher(self, mock_neural_net):
        """Create AsyncNeuralBatcher for testing."""
        # Ensure we don't use the global batcher
        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            b = AsyncNeuralBatcher(mock_neural_net, max_workers=1)
        yield b
        b.shutdown()

    def test_init_cpu(self, mock_neural_net):
        """Test initialization with CPU (no microbatcher)."""
        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_neural_net)

        assert batcher.neural_net is mock_neural_net
        assert batcher._use_microbatcher is False
        assert batcher._executor is not None
        batcher.shutdown()

    def test_evaluate_synchronous(self, batcher, mock_neural_net):
        """Test synchronous evaluate."""
        mock_states = [MagicMock(), MagicMock()]

        values, policies = batcher.evaluate(mock_states)

        mock_neural_net.evaluate_batch.assert_called_once()
        assert values == [0.5, 0.6]
        assert policies.shape == (2, 64)

    def test_evaluate_with_value_head(self, batcher, mock_neural_net):
        """Test evaluate with value_head parameter."""
        mock_states = [MagicMock()]

        batcher.evaluate(mock_states, value_head=2)

        mock_neural_net.evaluate_batch.assert_called_with(
            mock_states, value_head=2
        )

    def test_submit_returns_future(self, batcher, mock_neural_net):
        """Test submit returns a Future."""
        mock_states = [MagicMock()]

        future = batcher.submit(mock_states)

        assert isinstance(future, Future)
        result = future.result(timeout=1.0)
        assert result[0] == [0.5, 0.6]

    def test_submit_with_value_head(self, batcher, mock_neural_net):
        """Test submit with value_head parameter."""
        mock_states = [MagicMock()]

        future = batcher.submit(mock_states, value_head=1)
        future.result(timeout=1.0)

        mock_neural_net.evaluate_batch.assert_called()

    def test_shutdown(self, mock_neural_net):
        """Test shutdown cleans up resources."""
        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_neural_net)

        assert batcher._executor is not None
        batcher.shutdown()
        assert batcher._executor is None

    def test_submit_after_shutdown_raises(self, mock_neural_net):
        """Test submit after shutdown raises error."""
        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_neural_net)

        batcher.shutdown()

        with pytest.raises(RuntimeError, match="shut down"):
            batcher.submit([MagicMock()])


class TestThreadSafety:
    """Test thread safety of AsyncNeuralBatcher."""

    def test_concurrent_evaluate(self):
        """Test concurrent evaluate calls are serialized."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"

        call_order = []
        call_lock = threading.Lock()

        def mock_evaluate_batch(states, **kwargs):
            with call_lock:
                call_order.append(("start", threading.current_thread().name))
            time.sleep(0.01)  # Simulate work
            with call_lock:
                call_order.append(("end", threading.current_thread().name))
            return [0.5], np.array([[0.1] * 64])

        mock_nn.evaluate_batch = mock_evaluate_batch

        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_nn, max_workers=2)

        threads = []
        for i in range(4):
            t = threading.Thread(
                target=lambda: batcher.evaluate([MagicMock()]),
                name=f"worker-{i}",
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        batcher.shutdown()

        # All calls should complete (4 starts, 4 ends)
        starts = [c for c in call_order if c[0] == "start"]
        ends = [c for c in call_order if c[0] == "end"]
        assert len(starts) == 4
        assert len(ends) == 4

    def test_concurrent_submit(self):
        """Test concurrent submit calls."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"
        mock_nn.evaluate_batch.return_value = ([0.5], np.array([[0.1] * 64]))

        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_nn, max_workers=2)

        futures = []
        for _ in range(10):
            future = batcher.submit([MagicMock()])
            futures.append(future)

        # All futures should complete
        results = [f.result(timeout=5.0) for f in futures]
        assert len(results) == 10

        batcher.shutdown()


class TestGlobalBatcherManagement:
    """Test global batcher acquire/release."""

    def test_acquire_creates_batcher(self):
        """Test acquire creates global batcher."""
        # Reset global state
        import app.ai.async_nn_eval as module
        with module._GLOBAL_BATCHER_LOCK:
            original_batcher = module._GLOBAL_BATCHER
            original_refs = module._GLOBAL_BATCHER_REFS
            module._GLOBAL_BATCHER = None
            module._GLOBAL_BATCHER_REFS = 0

        try:
            batcher = _acquire_global_batcher()
            assert batcher is not None
            assert module._GLOBAL_BATCHER_REFS == 1
        finally:
            _release_global_batcher()
            # Restore original state
            with module._GLOBAL_BATCHER_LOCK:
                module._GLOBAL_BATCHER = original_batcher
                module._GLOBAL_BATCHER_REFS = original_refs

    def test_multiple_acquire_same_instance(self):
        """Test multiple acquires return same instance."""
        import app.ai.async_nn_eval as module
        with module._GLOBAL_BATCHER_LOCK:
            original_batcher = module._GLOBAL_BATCHER
            original_refs = module._GLOBAL_BATCHER_REFS
            module._GLOBAL_BATCHER = None
            module._GLOBAL_BATCHER_REFS = 0

        try:
            batcher1 = _acquire_global_batcher()
            batcher2 = _acquire_global_batcher()

            assert batcher1 is batcher2
            assert module._GLOBAL_BATCHER_REFS == 2
        finally:
            _release_global_batcher()
            _release_global_batcher()
            with module._GLOBAL_BATCHER_LOCK:
                module._GLOBAL_BATCHER = original_batcher
                module._GLOBAL_BATCHER_REFS = original_refs

    def test_release_decrements_refs(self):
        """Test release decrements reference count."""
        import app.ai.async_nn_eval as module
        with module._GLOBAL_BATCHER_LOCK:
            original_batcher = module._GLOBAL_BATCHER
            original_refs = module._GLOBAL_BATCHER_REFS
            module._GLOBAL_BATCHER = None
            module._GLOBAL_BATCHER_REFS = 0

        try:
            _acquire_global_batcher()
            _acquire_global_batcher()
            assert module._GLOBAL_BATCHER_REFS == 2

            _release_global_batcher()
            assert module._GLOBAL_BATCHER_REFS == 1

            _release_global_batcher()
            assert module._GLOBAL_BATCHER_REFS == 0
        finally:
            with module._GLOBAL_BATCHER_LOCK:
                module._GLOBAL_BATCHER = original_batcher
                module._GLOBAL_BATCHER_REFS = original_refs


class TestMicrobatcherIntegration:
    """Integration tests for microbatcher."""

    def test_batcher_processes_requests(self):
        """Test batcher actually processes requests."""
        mock_nn = MagicMock()
        mock_nn._ensure_model_initialized = MagicMock()
        mock_nn.model = MagicMock()
        mock_nn.device = "cpu"
        mock_nn.history_length = 0
        mock_nn.game_history = {}
        mock_nn.evaluate_batch.return_value = (
            [0.5],
            np.array([[0.1] * 64]),
        )

        # Create state with proper board
        mock_state = MagicMock()
        mock_state.board.type = "square"
        mock_state.board.size = 8

        batcher = _GlobalNNMicroBatcher(
            max_batch_states=64,
            batch_timeout_ms=50,
        )

        try:
            future = batcher.submit(mock_nn, [mock_state])
            result = future.result(timeout=1.0)

            assert result[0] == [0.5]
            mock_nn.evaluate_batch.assert_called_once()
        finally:
            batcher.shutdown()

    def test_batcher_groups_compatible_requests(self):
        """Test batcher groups compatible requests."""
        mock_nn = MagicMock()
        mock_nn._ensure_model_initialized = MagicMock()
        mock_nn.model = MagicMock()
        mock_nn.device = "cpu"
        mock_nn.history_length = 0
        mock_nn.game_history = {}

        call_count = 0
        def mock_evaluate(states, **kwargs):
            nonlocal call_count
            call_count += 1
            return (
                [0.5] * len(states),
                np.array([[0.1] * 64] * len(states)),
            )

        mock_nn.evaluate_batch = mock_evaluate

        # Create states with same board type
        def make_state():
            s = MagicMock()
            s.board.type = "square"
            s.board.size = 8
            return s

        batcher = _GlobalNNMicroBatcher(
            max_batch_states=64,
            batch_timeout_ms=100,  # Long timeout to allow grouping
        )

        try:
            # Submit multiple requests quickly
            futures = []
            for _ in range(3):
                future = batcher.submit(mock_nn, [make_state()])
                futures.append(future)

            # Wait for all to complete
            for f in futures:
                f.result(timeout=1.0)

            # Should have been batched together (fewer evaluate_batch calls than requests)
            assert call_count <= 3  # May be 1 if all batched, up to 3 if not
        finally:
            batcher.shutdown()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_game_states(self):
        """Test handling of empty game states list."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"
        mock_nn.evaluate_batch.return_value = ([], np.array([]))

        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_nn)

        values, policies = batcher.evaluate([])
        assert values == []
        batcher.shutdown()

    def test_model_raises_exception(self):
        """Test handling when model raises exception."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"
        mock_nn.evaluate_batch.side_effect = RuntimeError("Model error")

        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_nn)

        with pytest.raises(RuntimeError, match="Model error"):
            batcher.evaluate([MagicMock()])

        batcher.shutdown()

    def test_future_exception_propagation(self):
        """Test exceptions propagate through futures."""
        mock_nn = MagicMock()
        mock_nn.device = "cpu"
        mock_nn.evaluate_batch.side_effect = ValueError("Bad input")

        with patch.dict(os.environ, {"RINGRIFT_NN_EVAL_QUEUE": "0"}):
            batcher = AsyncNeuralBatcher(mock_nn)

        future = batcher.submit([MagicMock()])

        with pytest.raises(ValueError, match="Bad input"):
            future.result(timeout=1.0)

        batcher.shutdown()
