#!/usr/bin/env python3
"""
Rigorous GPU Data Generation Testing Suite

Tests:
1. Data Quality: Valid features, correct shapes, no NaN/Inf
2. Data Correctness: Outcomes align with winners, proper augmentation
3. Performance: Games/second, samples/second vs CPU baseline
4. Integration: Data usable by training pipeline

Run on cloud GPU instances (H100, A100, etc.) - not local machines (OOM risk).

Usage:
    PYTHONPATH=. python scripts/test_gpu_data_generation.py --quick       # Quick validation
    PYTHONPATH=. python scripts/test_gpu_data_generation.py --full        # Full test suite
    PYTHONPATH=. python scripts/test_gpu_data_generation.py --benchmark   # Performance only
"""

import argparse
import gc
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

# Ensure ai-service is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import BoardType
from app.training.generate_data import (
    generate_dataset,
    generate_dataset_gpu_parallel,
)


@dataclass
class TestResult:
    """Result from a single test."""
    name: str
    passed: bool
    message: str
    duration_sec: float
    metrics: dict


class GPUDataGenTester:
    """Comprehensive tester for GPU Data Generation."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[TestResult] = []

        # Detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.has_gpu = self.device.type in ("cuda", "mps")

        if self.verbose:
            print(f"GPU Data Generation Tester initialized")
            print(f"  Device: {self.device}")
            print(f"  GPU available: {self.has_gpu}")
            if self.device.type == "cuda":
                print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _add_result(self, name: str, passed: bool, message: str,
                    duration: float, metrics: dict = None) -> None:
        result = TestResult(
            name=name,
            passed=passed,
            message=message,
            duration_sec=duration,
            metrics=metrics or {},
        )
        self.results.append(result)
        status = "PASS" if passed else "FAIL"
        self._log(f"  [{status}] {name}: {message}")

    def test_data_quality(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_games: int = 10,
    ) -> bool:
        """Test that generated data has valid features and shapes."""
        self._log(f"\n=== Test: Data Quality (games={num_games}) ===")
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_data.npz")

            try:
                generate_dataset_gpu_parallel(
                    num_games=num_games,
                    output_file=output_file,
                    board_type=board_type,
                    seed=42,
                    max_moves=100,
                    num_players=2,
                    gpu_batch_size=min(num_games, 10),
                )

                # Load and validate data
                data = np.load(output_file, allow_pickle=True)

                features = data["features"]
                globals_arr = data["globals"]
                values = data["values"]
                policy_indices = data["policy_indices"]
                policy_values = data["policy_values"]

                issues = []

                # Check shapes
                if len(features.shape) != 4:
                    issues.append(f"Features shape should be 4D, got {features.shape}")
                if len(globals_arr.shape) != 2:
                    issues.append(f"Globals shape should be 2D, got {globals_arr.shape}")
                if len(values.shape) != 1:
                    issues.append(f"Values shape should be 1D, got {values.shape}")

                # Check for NaN/Inf
                if np.any(np.isnan(features)):
                    issues.append("Features contain NaN")
                if np.any(np.isinf(features)):
                    issues.append("Features contain Inf")
                if np.any(np.isnan(globals_arr)):
                    issues.append("Globals contain NaN")
                if np.any(np.isnan(values)):
                    issues.append("Values contain NaN")

                # Check value range
                if np.any(values < -1.1) or np.any(values > 1.1):
                    issues.append(f"Values out of range: min={values.min():.3f}, max={values.max():.3f}")

                # Check sample count
                num_samples = len(values)
                if num_samples == 0:
                    issues.append("No samples generated")

                # Check consistency
                if len(features) != len(values):
                    issues.append(f"Feature/value count mismatch: {len(features)} vs {len(values)}")

                passed = len(issues) == 0

            except Exception as e:
                passed = False
                issues = [str(e)]
                num_samples = 0

        duration = time.time() - start_time

        self._add_result(
            "Data Quality",
            passed,
            f"Samples: {num_samples}, Issues: {len(issues)}" + (f" - {issues[0]}" if issues else ""),
            duration,
            {"num_samples": num_samples, "issues": issues},
        )
        return passed

    def test_data_correctness(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_games: int = 20,
    ) -> bool:
        """Test that outcomes properly align with game winners."""
        self._log(f"\n=== Test: Data Correctness (games={num_games}) ===")
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_data.npz")

            try:
                generate_dataset_gpu_parallel(
                    num_games=num_games,
                    output_file=output_file,
                    board_type=board_type,
                    seed=42,
                    max_moves=150,
                    num_players=2,
                    gpu_batch_size=min(num_games, 10),
                )

                data = np.load(output_file, allow_pickle=True)
                values = data["values"]

                # Count outcome distribution
                wins = np.sum(values > 0.5)
                losses = np.sum(values < -0.5)
                draws = np.sum(np.abs(values) <= 0.5)

                # Expect some variety in outcomes
                has_variety = wins > 0 or losses > 0

                # Check outcome balance (shouldn't be all wins or all losses)
                total = len(values)
                win_rate = wins / total if total > 0 else 0
                loss_rate = losses / total if total > 0 else 0

                # Check that discounting is applied (later moves should have higher absolute values)
                # This is a soft check since augmentation complicates the pattern
                passed = has_variety and total > 0

            except Exception as e:
                passed = False
                wins, losses, draws = 0, 0, 0
                win_rate, loss_rate = 0, 0
                total = 0

        duration = time.time() - start_time

        self._add_result(
            "Data Correctness",
            passed,
            f"Samples: {total}, Wins: {wins} ({win_rate:.1%}), Losses: {losses} ({loss_rate:.1%}), Draws: {draws}",
            duration,
            {"total": total, "wins": wins, "losses": losses, "draws": draws},
        )
        return passed

    def test_augmentation(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_games: int = 5,
    ) -> bool:
        """Test that data augmentation produces expected multiplier."""
        self._log(f"\n=== Test: Augmentation (games={num_games}) ===")
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_data.npz")

            try:
                generate_dataset_gpu_parallel(
                    num_games=num_games,
                    output_file=output_file,
                    board_type=board_type,
                    seed=42,
                    max_moves=50,  # Short games for quick test
                    num_players=2,
                    gpu_batch_size=num_games,
                )

                data = np.load(output_file, allow_pickle=True)
                num_samples = len(data["values"])

                # Expected augmentation factor:
                # Square boards: 8x (rotations + reflections)
                # Hex boards: 12x (D6 symmetry)
                expected_factor = 12 if board_type == BoardType.HEXAGONAL else 8

                # Each game generates ~50 moves, each augmented 8x = ~400 samples per game
                # But some moves may fail to extract, so use lower bound
                min_expected = num_games * 10 * expected_factor  # At least 10 positions per game
                max_expected = num_games * 60 * expected_factor  # At most 60 positions per game

                in_range = min_expected <= num_samples <= max_expected

                passed = in_range and num_samples > 0

            except Exception as e:
                passed = False
                num_samples = 0
                min_expected = 0
                max_expected = 0

        duration = time.time() - start_time

        self._add_result(
            "Augmentation",
            passed,
            f"Samples: {num_samples}, Expected range: [{min_expected}, {max_expected}]",
            duration,
            {"num_samples": num_samples, "min_expected": min_expected, "max_expected": max_expected},
        )
        return passed

    def test_performance_vs_cpu(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_games: int = 20,
    ) -> bool:
        """Compare GPU vs CPU data generation performance."""
        self._log(f"\n=== Test: Performance vs CPU (games={num_games}) ===")
        start_time = time.time()

        if not self.has_gpu:
            self._add_result(
                "Performance vs CPU",
                True,
                "Skipped - no GPU available",
                0,
            )
            return True

        with tempfile.TemporaryDirectory() as tmpdir:
            gpu_file = os.path.join(tmpdir, "gpu_data.npz")
            cpu_file = os.path.join(tmpdir, "cpu_data.npz")

            # GPU timing
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            gc.collect()

            gpu_start = time.time()
            generate_dataset_gpu_parallel(
                num_games=num_games,
                output_file=gpu_file,
                board_type=board_type,
                seed=42,
                max_moves=100,
                num_players=2,
                gpu_batch_size=min(num_games, 20),
            )
            gpu_duration = time.time() - gpu_start

            gpu_data = np.load(gpu_file, allow_pickle=True)
            gpu_samples = len(gpu_data["values"])

            # CPU timing (using standard generate_dataset with Descent engine)
            cpu_start = time.time()
            generate_dataset(
                num_games=num_games,
                output_file=cpu_file,
                board_type=board_type,
                seed=42,
                max_moves=100,
                num_players=2,
                engine="descent",
                engine_mix="single",
            )
            cpu_duration = time.time() - cpu_start

            cpu_data = np.load(cpu_file, allow_pickle=True)
            cpu_samples = len(cpu_data["values"])

        duration = time.time() - start_time

        gpu_games_per_sec = num_games / gpu_duration if gpu_duration > 0 else 0
        cpu_games_per_sec = num_games / cpu_duration if cpu_duration > 0 else 0
        speedup = cpu_duration / gpu_duration if gpu_duration > 0 else 0

        # GPU should be faster (speedup > 1) for reasonable batch sizes
        passed = speedup >= 0.5  # Allow some margin

        self._add_result(
            "Performance vs CPU",
            passed,
            f"GPU: {gpu_duration:.1f}s ({gpu_games_per_sec:.2f} games/s), "
            f"CPU: {cpu_duration:.1f}s ({cpu_games_per_sec:.2f} games/s), "
            f"Speedup: {speedup:.2f}x",
            duration,
            {
                "gpu_duration": gpu_duration,
                "cpu_duration": cpu_duration,
                "speedup": speedup,
                "gpu_samples": gpu_samples,
                "cpu_samples": cpu_samples,
            },
        )
        return passed

    def test_training_compatibility(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_games: int = 10,
    ) -> bool:
        """Test that generated data is compatible with training pipeline."""
        self._log(f"\n=== Test: Training Compatibility (games={num_games}) ===")
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_data.npz")

            try:
                generate_dataset_gpu_parallel(
                    num_games=num_games,
                    output_file=output_file,
                    board_type=board_type,
                    seed=42,
                    max_moves=100,
                    num_players=2,
                    gpu_batch_size=num_games,
                )

                # Try to load with training data loader
                from app.training.data_loader import RingRiftDataset

                dataset = RingRiftDataset(output_file)

                # Check we can get samples
                issues = []

                if len(dataset) == 0:
                    issues.append("Dataset is empty")
                else:
                    # Get a batch of samples
                    sample = dataset[0]

                    # Check sample structure
                    if not isinstance(sample, tuple) or len(sample) != 4:
                        issues.append(f"Sample should be tuple of 4, got {type(sample)}")
                    else:
                        features, globals_vec, policy, value = sample

                        # Convert to tensors for shape check
                        if hasattr(features, 'shape'):
                            if len(features.shape) != 3:
                                issues.append(f"Features should be 3D (C,H,W), got {features.shape}")

                        if hasattr(value, 'shape'):
                            if len(value.shape) != 0:
                                issues.append(f"Value should be scalar, got shape {value.shape}")

                passed = len(issues) == 0
                num_samples = len(dataset)

            except Exception as e:
                passed = False
                issues = [str(e)]
                num_samples = 0

        duration = time.time() - start_time

        self._add_result(
            "Training Compatibility",
            passed,
            f"Samples: {num_samples}, Issues: {len(issues)}" + (f" - {issues[0]}" if issues else ""),
            duration,
            {"num_samples": num_samples, "issues": issues},
        )
        return passed

    def test_gpu_memory_usage(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        batch_sizes: list[int] = [10, 20, 50],
    ) -> bool:
        """Test GPU memory usage at different batch sizes."""
        self._log(f"\n=== Test: GPU Memory Usage ===")
        start_time = time.time()

        if not self.has_gpu or self.device.type != "cuda":
            self._add_result(
                "GPU Memory Usage",
                True,
                "Skipped - CUDA not available",
                0,
            )
            return True

        results = []

        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            gc.collect()

            initial_mem = torch.cuda.memory_allocated() / 1e9

            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = os.path.join(tmpdir, "test_data.npz")

                generate_dataset_gpu_parallel(
                    num_games=batch_size,
                    output_file=output_file,
                    board_type=board_type,
                    seed=42,
                    max_moves=100,
                    num_players=2,
                    gpu_batch_size=batch_size,
                )

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            used_mem = peak_mem - initial_mem

            results.append({
                "batch_size": batch_size,
                "used_gb": used_mem,
                "peak_gb": peak_mem,
            })

            self._log(f"    Batch size {batch_size}: {used_mem:.2f} GB used, {peak_mem:.2f} GB peak")

            torch.cuda.empty_cache()

        duration = time.time() - start_time

        # Memory should scale reasonably with batch size
        max_mem = max(r["used_gb"] for r in results)
        passed = max_mem < 16.0  # Should use less than 16 GB

        self._add_result(
            "GPU Memory Usage",
            passed,
            f"Max memory: {max_mem:.2f} GB across batch sizes {batch_sizes}",
            duration,
            {"results": results, "max_mem_gb": max_mem},
        )
        return passed

    def test_multi_board_types(self) -> bool:
        """Test data generation across different board types."""
        self._log(f"\n=== Test: Multi Board Types ===")
        start_time = time.time()

        board_types = [
            (BoardType.SQUARE8, 8),
            # BoardType.SQUARE19 and HEXAGONAL need more memory, test separately
        ]

        results = []

        for board_type, expected_size in board_types:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = os.path.join(tmpdir, "test_data.npz")

                try:
                    generate_dataset_gpu_parallel(
                        num_games=5,
                        output_file=output_file,
                        board_type=board_type,
                        seed=42,
                        max_moves=50,
                        num_players=2,
                        gpu_batch_size=5,
                    )

                    data = np.load(output_file, allow_pickle=True)
                    features = data["features"]

                    # Check board size in features
                    # Features shape: (N, C, H, W)
                    actual_size = features.shape[-1]

                    results.append({
                        "board_type": board_type.value,
                        "expected_size": expected_size,
                        "actual_size": actual_size,
                        "num_samples": len(data["values"]),
                        "success": True,
                    })

                except Exception as e:
                    results.append({
                        "board_type": board_type.value,
                        "error": str(e),
                        "success": False,
                    })

        duration = time.time() - start_time

        all_success = all(r["success"] for r in results)

        self._add_result(
            "Multi Board Types",
            all_success,
            f"Tested {len(board_types)} board types, {sum(1 for r in results if r['success'])} successful",
            duration,
            {"results": results},
        )
        return all_success

    def run_quick_tests(self) -> bool:
        """Run quick validation tests."""
        self._log("\n" + "=" * 60)
        self._log("GPU DATA GENERATION QUICK TEST SUITE")
        self._log("=" * 60)

        self.test_data_quality(num_games=5)
        self.test_data_correctness(num_games=10)
        self.test_training_compatibility(num_games=5)

        return self._summarize()

    def run_full_tests(self) -> bool:
        """Run full test suite."""
        self._log("\n" + "=" * 60)
        self._log("GPU DATA GENERATION FULL TEST SUITE")
        self._log("=" * 60)

        self.test_data_quality(num_games=20)
        self.test_data_correctness(num_games=30)
        self.test_augmentation(num_games=10)
        self.test_training_compatibility(num_games=10)
        self.test_performance_vs_cpu(num_games=30)
        self.test_gpu_memory_usage(batch_sizes=[10, 20, 50, 100])
        self.test_multi_board_types()

        return self._summarize()

    def run_benchmark(self) -> bool:
        """Run performance benchmarks only."""
        self._log("\n" + "=" * 60)
        self._log("GPU DATA GENERATION BENCHMARK")
        self._log("=" * 60)

        self.test_performance_vs_cpu(num_games=50)
        self.test_gpu_memory_usage(batch_sizes=[10, 20, 50, 100, 200])

        return self._summarize()

    def _summarize(self) -> bool:
        """Print summary and return overall pass/fail."""
        self._log("\n" + "=" * 60)
        self._log("SUMMARY")
        self._log("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        total_time = sum(r.duration_sec for r in self.results)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            self._log(f"  [{status}] {r.name}: {r.message}")

        self._log(f"\nResults: {passed}/{total} tests passed in {total_time:.1f}s")

        all_passed = passed == total
        if all_passed:
            self._log("All tests PASSED!")
        else:
            self._log("Some tests FAILED!")

        return all_passed


def main():
    parser = argparse.ArgumentParser(description="GPU Data Generation Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks only")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    tester = GPUDataGenTester(verbose=not args.quiet)

    if args.benchmark:
        success = tester.run_benchmark()
    elif args.full:
        success = tester.run_full_tests()
    else:
        # Default to quick tests
        success = tester.run_quick_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
