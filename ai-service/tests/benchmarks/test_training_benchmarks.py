"""Performance benchmarks for training operations.

Benchmarks critical training paths:
- Neural network forward pass throughput
- Batch data loading speed
- Checkpoint save/load latency
- Move encoding/decoding throughput
- Policy layout operations

Run with: pytest tests/benchmarks/test_training_benchmarks.py -v --benchmark-only
Or without pytest-benchmark: python tests/benchmarks/test_training_benchmarks.py
"""

import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directories to path for standalone execution
_THIS_DIR = Path(__file__).resolve().parent
_AI_SERVICE_ROOT = _THIS_DIR.parents[1]
if str(_AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_SERVICE_ROOT))

import pytest

# Check if pytest-benchmark is available
try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


def simple_benchmark(func: Callable, iterations: int = 100) -> tuple[float, float, float]:
    """Simple benchmark without pytest-benchmark.

    Returns (min_time, avg_time, max_time) in milliseconds.
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return min(times), sum(times) / len(times), max(times)


def print_benchmark_result(name: str, min_t: float, avg_t: float, max_t: float, unit: str = "ms"):
    """Print benchmark result in a readable format."""
    print(f"{name:50s} | min: {min_t:8.3f}{unit} | avg: {avg_t:8.3f}{unit} | max: {max_t:8.3f}{unit}")


class SimpleTestModel(nn.Module):
    """Simple model matching production architecture for benchmarking."""

    def __init__(self, in_channels=21, hidden=64, policy_size=2048):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.fc = nn.Linear(hidden * 8 * 8, 256)
        self.policy_head = nn.Linear(256, policy_size)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    min_ms: float
    avg_ms: float
    max_ms: float
    throughput: float = 0.0  # items/sec if applicable


class TestNeuralNetBenchmarks:
    """Benchmarks for neural network operations."""

    @pytest.fixture
    def model(self):
        """Create model for benchmarking."""
        model = SimpleTestModel()
        model.eval()
        return model

    @pytest.fixture
    def batch_sizes(self):
        """Batch sizes to benchmark."""
        return [1, 8, 32, 64, 128]

    def test_forward_pass_latency(self, model, batch_sizes):
        """Benchmark forward pass latency at various batch sizes."""
        results = []

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 21, 8, 8)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model(x)

            def forward(*, _x=x):
                with torch.no_grad():
                    model(_x)

            min_t, avg_t, max_t = simple_benchmark(forward, iterations=100)
            throughput = (batch_size / avg_t) * 1000  # samples/sec

            results.append(BenchmarkResult(
                name=f"forward_batch_{batch_size}",
                min_ms=min_t,
                avg_ms=avg_t,
                max_ms=max_t,
                throughput=throughput,
            ))

        print("\n=== Neural Network Forward Pass Benchmarks ===")
        for r in results:
            print(f"Batch {r.name.split('_')[-1]:>3}: {r.avg_ms:6.2f}ms avg, {r.throughput:8.0f} samples/sec")

        # Assert reasonable performance
        assert results[0].avg_ms < 50  # Single sample < 50ms
        assert results[-1].throughput > 100  # At least 100 samples/sec at max batch

    def test_training_step_latency(self, model):
        """Benchmark complete training step (forward + backward + optimizer)."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        batch_size = 32
        x = torch.randn(batch_size, 21, 8, 8)
        target = torch.randint(0, 2048, (batch_size,))

        model.train()

        # Warmup
        for _ in range(5):
            policy, _value = model(x)
            loss = criterion(policy, target)
            # set_to_none avoids unnecessary memset and is the recommended fast path.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        def training_step():
            policy, _value = model(x)
            loss = criterion(policy, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        _min_t, avg_t, _max_t = simple_benchmark(training_step, iterations=50)
        throughput = (batch_size / avg_t) * 1000

        print(f"\n=== Training Step Benchmark (batch={batch_size}) ===")
        print(f"Latency: {avg_t:.2f}ms avg, Throughput: {throughput:.0f} samples/sec")

        assert avg_t < 200  # Complete training step < 200ms


class TestMoveEncodingBenchmarks:
    """Benchmarks for move encoding/decoding operations."""

    def test_policy_encoding_throughput(self):
        """Benchmark policy encoding speed."""
        try:
            from app.ai.neural_net import (
                SQUARE8_2P_POLICY_SIZE,
                _encode_move_for_policy,
            )
            from app.models import Move, Position
        except ImportError:
            pytest.skip("Neural net module not available")

        # Create sample moves
        moves = []
        for i in range(100):
            move = Move(
                id=f"move_{i}",
                player=1,
                from_pos=Position(x=i % 8, y=i // 8 % 8),
                to=Position(x=(i + 1) % 8, y=(i + 2) % 8),
            )
            moves.append(move)

        def encode_batch():
            for move in moves:
                _encode_move_for_policy(move, 8)

        _min_t, avg_t, _max_t = simple_benchmark(encode_batch, iterations=100)
        throughput = (len(moves) / avg_t) * 1000

        print("\n=== Move Encoding Benchmark ===")
        print(f"100 moves: {avg_t:.2f}ms, Throughput: {throughput:.0f} moves/sec")

        assert throughput > 10000  # At least 10k moves/sec


class TestCheckpointBenchmarks:
    """Benchmarks for checkpoint operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_checkpoint_save_latency(self, temp_dir):
        """Benchmark checkpoint save latency."""
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        checkpoint_path = Path(temp_dir) / "benchmark_checkpoint.pt"

        def save_checkpoint():
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 100,
                "metrics": {"loss": 0.5, "accuracy": 0.9},
            }, checkpoint_path)

        # Warmup
        save_checkpoint()

        _min_t, avg_t, _max_t = simple_benchmark(save_checkpoint, iterations=20)

        # Get file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        print("\n=== Checkpoint Save Benchmark ===")
        print(f"Latency: {avg_t:.2f}ms avg, File size: {file_size_mb:.2f}MB")

        assert avg_t < 500  # Save < 500ms

    def test_checkpoint_load_latency(self, temp_dir):
        """Benchmark checkpoint load latency."""
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        checkpoint_path = Path(temp_dir) / "benchmark_checkpoint.pt"

        # Save checkpoint first
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 100,
        }, checkpoint_path)

        def load_checkpoint():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

        # Warmup
        load_checkpoint()

        _min_t, avg_t, _max_t = simple_benchmark(load_checkpoint, iterations=20)

        print("\n=== Checkpoint Load Benchmark ===")
        print(f"Latency: {avg_t:.2f}ms avg")

        assert avg_t < 500  # Load < 500ms


class TestHeuristicWeightBenchmarks:
    """Benchmarks for heuristic weight operations."""

    def test_flatten_reconstruct_throughput(self):
        """Benchmark heuristic weight flatten/reconstruct."""
        try:
            from app.ai.heuristic_weights import (
                HEURISTIC_WEIGHT_KEYS,
                HEURISTIC_WEIGHT_PROFILES,
            )
            from app.training.train import (
                _flatten_heuristic_weights,
                _reconstruct_heuristic_profile,
            )
        except ImportError:
            pytest.skip("Heuristic weights not available")

        if not HEURISTIC_WEIGHT_PROFILES:
            pytest.skip("No profiles defined")

        profile = next(iter(HEURISTIC_WEIGHT_PROFILES.values()))

        def flatten_reconstruct():
            keys, values = _flatten_heuristic_weights(profile)
            _reconstruct_heuristic_profile(keys, values)

        _min_t, avg_t, _max_t = simple_benchmark(flatten_reconstruct, iterations=1000)
        throughput = (1 / avg_t) * 1000  # ops/sec

        print("\n=== Heuristic Weight Flatten/Reconstruct Benchmark ===")
        print(f"Latency: {avg_t:.4f}ms, Throughput: {throughput:.0f} ops/sec")

        assert throughput > 1000  # At least 1000 ops/sec


class TestAdvancedTrainingBenchmarks:
    """Benchmarks for advanced training utilities."""

    def test_pfsp_sampling_throughput(self):
        """Benchmark PFSP opponent sampling."""
        try:
            from app.training.advanced_training import PFSPOpponentPool
        except ImportError:
            pytest.skip("Advanced training not available")

        pool = PFSPOpponentPool(max_pool_size=50)

        # Add opponents
        for i in range(50):
            pool.add_opponent(
                f"/models/gen{i}.pth",
                elo=1500 + i * 10,
                generation=i,
            )
            # Add some game history
            for _ in range(10):
                pool.update_stats(f"/models/gen{i}.pth", won=i % 2 == 0)

        def sample_opponent():
            pool.sample_opponent(current_elo=1600, strategy="pfsp")

        _min_t, avg_t, _max_t = simple_benchmark(sample_opponent, iterations=1000)
        throughput = (1 / avg_t) * 1000

        print("\n=== PFSP Sampling Benchmark (50 opponents) ===")
        print(f"Latency: {avg_t:.4f}ms, Throughput: {throughput:.0f} samples/sec")

        assert throughput > 1000  # At least 1000 samples/sec

    def test_lr_finder_analysis_throughput(self):
        """Benchmark LR finder result analysis."""
        try:
            from app.training.advanced_training import LRFinder
        except ImportError:
            pytest.skip("Advanced training not available")

        model = SimpleTestModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(model, optimizer, criterion)

        # Generate synthetic LR finder data
        lrs = np.logspace(-7, 1, 200).tolist()
        losses = [1.0 - 0.3 * np.exp(-((np.log10(lr) + 3) ** 2) / 2) + 0.1 * np.random.random()
                  for lr in lrs]

        def analyze():
            finder._analyze_results(lrs, losses, 1e-7, 10.0)

        _min_t, avg_t, _max_t = simple_benchmark(analyze, iterations=100)
        throughput = (1 / avg_t) * 1000

        print("\n=== LR Finder Analysis Benchmark (200 points) ===")
        print(f"Latency: {avg_t:.4f}ms, Throughput: {throughput:.0f} analyses/sec")

        assert throughput > 100  # At least 100 analyses/sec


class TestDataLoadingBenchmarks:
    """Benchmarks for data loading operations."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        tmpdir = tempfile.mkdtemp()

        # Create synthetic training data
        num_samples = 1000
        features = np.random.randn(num_samples, 21, 8, 8).astype(np.float32)
        globals_data = np.random.randn(num_samples, 10).astype(np.float32)
        values = np.random.uniform(-1, 1, num_samples).astype(np.float32)

        np.savez_compressed(
            Path(tmpdir) / "train_data.npz",
            features=features,
            globals=globals_data,
            values=values,
        )

        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_numpy_load_throughput(self, temp_data_dir):
        """Benchmark NumPy data loading."""
        data_path = Path(temp_data_dir) / "train_data.npz"

        def load_data():
            data = np.load(data_path)
            _ = data["features"]
            _ = data["values"]

        _min_t, avg_t, _max_t = simple_benchmark(load_data, iterations=20)

        # Get file size
        file_size_mb = data_path.stat().st_size / (1024 * 1024)
        throughput_mb = file_size_mb / (avg_t / 1000)  # MB/sec

        print("\n=== NumPy Load Benchmark ===")
        print(f"Latency: {avg_t:.2f}ms, File: {file_size_mb:.2f}MB, Throughput: {throughput_mb:.0f}MB/sec")


# Standalone execution
if __name__ == "__main__":
    print("=" * 70)
    print("RingRift AI Training Performance Benchmarks")
    print("=" * 70)

    # Run benchmarks manually
    model = SimpleTestModel()
    model.eval()

    # Forward pass benchmark
    print("\n--- Forward Pass Benchmarks ---")
    for batch_size in [1, 8, 32, 64, 128]:
        x = torch.randn(batch_size, 21, 8, 8)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(x)

        def forward(*, _x=x):
            with torch.no_grad():
                model(_x)

        min_t, avg_t, max_t = simple_benchmark(forward, iterations=100)
        throughput = (batch_size / avg_t) * 1000
        print(f"Batch {batch_size:>3}: {avg_t:6.2f}ms avg, {throughput:8.0f} samples/sec")

    print("\nBenchmarks completed.")
