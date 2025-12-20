"""
Comprehensive Benchmark Suite for RingRift AI.

Provides reproducible benchmarks for model evaluation across
performance, quality, and position-specific metrics.
"""

import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    PERFORMANCE = "performance"      # Speed, throughput
    QUALITY = "quality"              # Win rate, Elo
    TACTICAL = "tactical"            # Specific tactical patterns
    STRATEGIC = "strategic"          # Long-term planning
    ROBUSTNESS = "robustness"        # Edge cases, adversarial
    EFFICIENCY = "efficiency"        # Memory, compute usage


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    benchmark_name: str
    category: BenchmarkCategory
    score: float
    unit: str
    higher_is_better: bool
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['category'] = self.category.value
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class BenchmarkSuiteResult:
    """Results from running a benchmark suite."""
    suite_name: str
    model_id: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_score(self, benchmark_name: str) -> float | None:
        """Get score for a specific benchmark."""
        for r in self.results:
            if r.benchmark_name == benchmark_name:
                return r.score
        return None

    def get_category_scores(self, category: BenchmarkCategory) -> dict[str, float]:
        """Get all scores for a category."""
        return {
            r.benchmark_name: r.score
            for r in self.results
            if r.category == category
        }

    def compute_aggregate_score(self) -> float:
        """Compute weighted aggregate score."""
        if not self.results:
            return 0.0

        # Normalize scores and compute mean
        normalized_scores = []
        for r in self.results:
            # Assume score is already normalized 0-1 or handle appropriately
            score = r.score
            if not r.higher_is_better:
                score = 1.0 / (1.0 + score)  # Invert for lower-is-better
            normalized_scores.append(score)

        return statistics.mean(normalized_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            'suite_name': self.suite_name,
            'model_id': self.model_id,
            'results': [r.to_dict() for r in self.results],
            'total_duration_seconds': self.total_duration_seconds,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'aggregate_score': self.compute_aggregate_score()
        }


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    def __init__(self, name: str, category: BenchmarkCategory):
        self.name = name
        self.category = category

    @abstractmethod
    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run the benchmark and return results."""
        pass


class InferenceBenchmark(Benchmark):
    """Benchmark inference speed and throughput."""

    def __init__(
        self,
        batch_sizes: list[int] | None = None,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100
    ):
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 64]
        super().__init__("inference_speed", BenchmarkCategory.PERFORMANCE)
        self.batch_sizes = batch_sizes
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run inference speed benchmark."""
        try:
            import numpy as np
            import torch
        except ImportError:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="samples/sec",
                higher_is_better=True,
                details={'error': 'PyTorch not available'}
            )

        results = {}
        device = kwargs.get('device', 'cpu')

        for batch_size in self.batch_sizes:
            # Create dummy input
            if hasattr(model, 'input_shape'):
                shape = (batch_size, *model.input_shape)
            else:
                shape = (batch_size, 18, 8, 8)  # Default shape

            x = torch.randn(shape).to(device)

            # Warmup
            model.eval()
            with torch.no_grad():
                for _ in range(self.warmup_iterations):
                    _ = model(x)

            # Benchmark
            if torch.cuda.is_available() and device != 'cpu':
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(self.benchmark_iterations):
                    _ = model(x)

            if torch.cuda.is_available() and device != 'cpu':
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            throughput = (batch_size * self.benchmark_iterations) / elapsed
            results[f'batch_{batch_size}'] = {
                'throughput': throughput,
                'latency_ms': (elapsed / self.benchmark_iterations) * 1000
            }

        # Use batch_size=1 throughput as main score
        main_score = results.get('batch_1', {}).get('throughput', 0.0)

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            score=main_score,
            unit="samples/sec",
            higher_is_better=True,
            details=results,
            duration_seconds=sum(r['latency_ms'] * self.benchmark_iterations / 1000
                                 for r in results.values())
        )


class MemoryBenchmark(Benchmark):
    """Benchmark memory usage."""

    def __init__(self, batch_sizes: list[int] | None = None):
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        super().__init__("memory_usage", BenchmarkCategory.EFFICIENCY)
        self.batch_sizes = batch_sizes

    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run memory benchmark."""
        try:
            import torch
        except ImportError:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="MB",
                higher_is_better=False,
                details={'error': 'PyTorch not available'}
            )

        device = kwargs.get('device', 'cpu')
        results = {}

        # Model parameter count
        param_count = sum(p.numel() for p in model.parameters())
        param_memory_mb = param_count * 4 / (1024 * 1024)  # Assume float32
        results['param_count'] = param_count
        results['param_memory_mb'] = param_memory_mb

        if torch.cuda.is_available() and device != 'cpu':
            torch.cuda.reset_peak_memory_stats()

            for batch_size in self.batch_sizes:
                if hasattr(model, 'input_shape'):
                    shape = (batch_size, *model.input_shape)
                else:
                    shape = (batch_size, 18, 8, 8)

                x = torch.randn(shape).to(device)

                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = model(x)

                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                results[f'batch_{batch_size}_peak_mb'] = peak_memory

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            score=param_memory_mb,
            unit="MB",
            higher_is_better=False,
            details=results
        )


class PolicyAccuracyBenchmark(Benchmark):
    """Benchmark policy prediction accuracy on known positions."""

    def __init__(self, test_positions: list[dict[str, Any]] | None = None):
        super().__init__("policy_accuracy", BenchmarkCategory.QUALITY)
        self.test_positions = test_positions or []

    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run policy accuracy benchmark."""
        if not self.test_positions:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="accuracy",
                higher_is_better=True,
                details={'error': 'No test positions provided'}
            )

        correct = 0
        total = 0
        top3_correct = 0
        details = {}

        for pos in self.test_positions:
            state = pos.get('state')
            best_move = pos.get('best_move')
            pos.get('good_moves', [best_move])

            if state is None or best_move is None:
                continue

            try:
                # Get model prediction
                policy = self._get_policy(model, state, **kwargs)

                # Find top moves
                top_moves = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:3]
                predicted_move = top_moves[0][0]

                if predicted_move == best_move:
                    correct += 1

                if best_move in [m[0] for m in top_moves]:
                    top3_correct += 1

                total += 1

            except Exception as e:
                logger.debug(f"Error evaluating position: {e}")

        accuracy = correct / total if total > 0 else 0.0
        top3_accuracy = top3_correct / total if total > 0 else 0.0

        details['accuracy'] = accuracy
        details['top3_accuracy'] = top3_accuracy
        details['correct'] = correct
        details['total'] = total

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            score=accuracy,
            unit="accuracy",
            higher_is_better=True,
            details=details
        )

    def _get_policy(self, model: Any, state: Any, **kwargs) -> list[float]:
        """Get policy from model for a state."""
        try:
            import numpy as np
            import torch
        except ImportError:
            return []

        model.eval()
        with torch.no_grad():
            if hasattr(state, 'to_tensor'):
                x = state.to_tensor().unsqueeze(0)
            else:
                x = torch.tensor(state).unsqueeze(0).float()

            device = kwargs.get('device', 'cpu')
            x = x.to(device)

            output = model(x)
            if isinstance(output, tuple):
                policy = output[0]
            else:
                policy = output

            return policy.squeeze().cpu().numpy().tolist()


class ValueAccuracyBenchmark(Benchmark):
    """Benchmark value prediction accuracy."""

    def __init__(self, test_positions: list[dict[str, Any]] | None = None):
        super().__init__("value_accuracy", BenchmarkCategory.QUALITY)
        self.test_positions = test_positions or []

    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run value accuracy benchmark."""
        if not self.test_positions:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="mse",
                higher_is_better=False,
                details={'error': 'No test positions provided'}
            )

        errors = []
        correct_sign = 0
        total = 0

        for pos in self.test_positions:
            state = pos.get('state')
            true_value = pos.get('value')

            if state is None or true_value is None:
                continue

            try:
                predicted_value = self._get_value(model, state, **kwargs)
                error = (predicted_value - true_value) ** 2
                errors.append(error)

                if (predicted_value > 0) == (true_value > 0):
                    correct_sign += 1

                total += 1

            except Exception as e:
                logger.debug(f"Error evaluating position: {e}")

        mse = statistics.mean(errors) if errors else 0.0
        sign_accuracy = correct_sign / total if total > 0 else 0.0

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            score=mse,
            unit="mse",
            higher_is_better=False,
            details={
                'mse': mse,
                'rmse': mse ** 0.5,
                'sign_accuracy': sign_accuracy,
                'total': total
            }
        )

    def _get_value(self, model: Any, state: Any, **kwargs) -> float:
        """Get value from model for a state."""
        try:
            import torch
        except ImportError:
            return 0.0

        model.eval()
        with torch.no_grad():
            if hasattr(state, 'to_tensor'):
                x = state.to_tensor().unsqueeze(0)
            else:
                x = torch.tensor(state).unsqueeze(0).float()

            device = kwargs.get('device', 'cpu')
            x = x.to(device)

            output = model(x)
            if isinstance(output, tuple):
                value = output[1]
            else:
                value = output

            return float(value.squeeze().cpu().item())


class TacticalBenchmark(Benchmark):
    """Benchmark tactical pattern recognition."""

    def __init__(self, tactical_puzzles: list[dict[str, Any]] | None = None):
        super().__init__("tactical_patterns", BenchmarkCategory.TACTICAL)
        self.tactical_puzzles = tactical_puzzles or []

    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run tactical benchmark."""
        if not self.tactical_puzzles:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="accuracy",
                higher_is_better=True,
                details={'error': 'No tactical puzzles provided'}
            )

        results_by_type = {}
        total_correct = 0
        total_puzzles = 0

        for puzzle in self.tactical_puzzles:
            puzzle_type = puzzle.get('type', 'unknown')
            state = puzzle.get('state')
            solution = puzzle.get('solution')

            if state is None or solution is None:
                continue

            try:
                # Get model's move
                predicted_move = self._get_best_move(model, state, **kwargs)
                is_correct = predicted_move == solution

                if puzzle_type not in results_by_type:
                    results_by_type[puzzle_type] = {'correct': 0, 'total': 0}

                results_by_type[puzzle_type]['total'] += 1
                if is_correct:
                    results_by_type[puzzle_type]['correct'] += 1
                    total_correct += 1

                total_puzzles += 1

            except Exception as e:
                logger.debug(f"Error on tactical puzzle: {e}")

        accuracy = total_correct / total_puzzles if total_puzzles > 0 else 0.0

        # Compute per-type accuracy
        type_accuracies = {}
        for t, counts in results_by_type.items():
            if counts['total'] > 0:
                type_accuracies[t] = counts['correct'] / counts['total']

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            score=accuracy,
            unit="accuracy",
            higher_is_better=True,
            details={
                'overall_accuracy': accuracy,
                'total_puzzles': total_puzzles,
                'type_accuracies': type_accuracies,
                'results_by_type': results_by_type
            }
        )

    def _get_best_move(self, model: Any, state: Any, **kwargs) -> int:
        """Get best move from model."""
        try:
            import numpy as np
            import torch
        except ImportError:
            return 0

        model.eval()
        with torch.no_grad():
            if hasattr(state, 'to_tensor'):
                x = state.to_tensor().unsqueeze(0)
            else:
                x = torch.tensor(state).unsqueeze(0).float()

            device = kwargs.get('device', 'cpu')
            x = x.to(device)

            output = model(x)
            if isinstance(output, tuple):
                policy = output[0]
            else:
                policy = output

            return int(policy.argmax().item())


class MCTSBenchmark(Benchmark):
    """Benchmark MCTS search quality."""

    def __init__(
        self,
        search_iterations: list[int] | None = None,
        test_positions: list[dict[str, Any]] | None = None
    ):
        if search_iterations is None:
            search_iterations = [100, 400, 800]
        super().__init__("mcts_quality", BenchmarkCategory.QUALITY)
        self.search_iterations = search_iterations
        self.test_positions = test_positions or []

    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run MCTS benchmark."""
        mcts_class = kwargs.get('mcts_class')
        if mcts_class is None or not self.test_positions:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="accuracy",
                higher_is_better=True,
                details={'error': 'MCTS class or positions not provided'}
            )

        results_by_iterations = {}

        for num_iters in self.search_iterations:
            correct = 0
            total = 0
            search_times = []

            for pos in self.test_positions:
                state = pos.get('state')
                best_move = pos.get('best_move')

                if state is None or best_move is None:
                    continue

                try:
                    mcts = mcts_class(model, num_simulations=num_iters)

                    start_time = time.perf_counter()
                    predicted_move = mcts.search(state)
                    elapsed = time.perf_counter() - start_time

                    search_times.append(elapsed)

                    if predicted_move == best_move:
                        correct += 1
                    total += 1

                except Exception as e:
                    logger.debug(f"MCTS error: {e}")

            accuracy = correct / total if total > 0 else 0.0
            avg_time = statistics.mean(search_times) if search_times else 0.0

            results_by_iterations[num_iters] = {
                'accuracy': accuracy,
                'avg_time_seconds': avg_time,
                'correct': correct,
                'total': total
            }

        # Use highest iteration count for main score
        main_score = results_by_iterations.get(
            max(self.search_iterations), {}
        ).get('accuracy', 0.0)

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            score=main_score,
            unit="accuracy",
            higher_is_better=True,
            details=results_by_iterations
        )


class RobustnessBenchmark(Benchmark):
    """Benchmark model robustness to perturbations."""

    def __init__(
        self,
        test_positions: list[dict[str, Any]] | None = None,
        noise_levels: list[float] | None = None
    ):
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1]
        super().__init__("robustness", BenchmarkCategory.ROBUSTNESS)
        self.test_positions = test_positions or []
        self.noise_levels = noise_levels

    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """Run robustness benchmark."""
        try:
            import numpy as np
            import torch
        except ImportError:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="stability",
                higher_is_better=True,
                details={'error': 'PyTorch not available'}
            )

        if not self.test_positions:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                score=0.0,
                unit="stability",
                higher_is_better=True,
                details={'error': 'No test positions provided'}
            )

        stability_scores = []
        results_by_noise = {}

        for noise_level in self.noise_levels:
            consistent = 0
            total = 0

            for pos in self.test_positions:
                state = pos.get('state')
                if state is None:
                    continue

                try:
                    # Get clean prediction
                    clean_move = self._get_best_move(model, state, **kwargs)

                    # Get noisy predictions
                    noisy_moves = []
                    for _ in range(5):
                        noisy_state = self._add_noise(state, noise_level)
                        noisy_move = self._get_best_move(model, noisy_state, **kwargs)
                        noisy_moves.append(noisy_move)

                    # Check consistency
                    if all(m == clean_move for m in noisy_moves):
                        consistent += 1
                    total += 1

                except Exception as e:
                    logger.debug(f"Robustness error: {e}")

            stability = consistent / total if total > 0 else 0.0
            stability_scores.append(stability)
            results_by_noise[noise_level] = {
                'stability': stability,
                'consistent': consistent,
                'total': total
            }

        # Average stability across noise levels
        avg_stability = statistics.mean(stability_scores) if stability_scores else 0.0

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            score=avg_stability,
            unit="stability",
            higher_is_better=True,
            details=results_by_noise
        )

    def _get_best_move(self, model: Any, state: Any, **kwargs) -> int:
        """Get best move from model."""
        try:
            import torch
        except ImportError:
            return 0

        model.eval()
        with torch.no_grad():
            if hasattr(state, 'to_tensor'):
                x = state.to_tensor().unsqueeze(0)
            else:
                x = torch.tensor(state).unsqueeze(0).float()

            device = kwargs.get('device', 'cpu')
            x = x.to(device)

            output = model(x)
            if isinstance(output, tuple):
                policy = output[0]
            else:
                policy = output

            return int(policy.argmax().item())

    def _add_noise(self, state: Any, noise_level: float) -> Any:
        """Add Gaussian noise to state."""
        try:
            import numpy as np
            import torch
        except ImportError:
            return state

        if hasattr(state, 'to_tensor'):
            tensor = state.to_tensor()
            noisy = tensor + torch.randn_like(tensor) * noise_level
            return noisy
        elif isinstance(state, np.ndarray):
            return state + np.random.randn(*state.shape) * noise_level
        else:
            return state


class BenchmarkSuite:
    """
    Main benchmark suite that runs multiple benchmarks.
    """

    def __init__(self, suite_name: str = "default"):
        self.suite_name = suite_name
        self.benchmarks: list[Benchmark] = []
        self.results_history: list[BenchmarkSuiteResult] = []

    def add_benchmark(self, benchmark: Benchmark):
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    def add_default_benchmarks(self):
        """Add default set of benchmarks."""
        self.benchmarks = [
            InferenceBenchmark(),
            MemoryBenchmark(),
            PolicyAccuracyBenchmark(),
            ValueAccuracyBenchmark(),
            TacticalBenchmark(),
            RobustnessBenchmark()
        ]

    def run(self, model: Any, model_id: str, **kwargs) -> BenchmarkSuiteResult:
        """
        Run all benchmarks in the suite.

        Args:
            model: The model to benchmark
            model_id: Identifier for the model
            **kwargs: Additional arguments for benchmarks

        Returns:
            BenchmarkSuiteResult with all results
        """
        start_time = time.perf_counter()
        results = []

        for benchmark in self.benchmarks:
            logger.info(f"Running benchmark: {benchmark.name}")
            try:
                result = benchmark.run(model, **kwargs)
                results.append(result)
                logger.info(f"  Score: {result.score:.4f} {result.unit}")
            except Exception as e:
                logger.error(f"Benchmark {benchmark.name} failed: {e}")
                results.append(BenchmarkResult(
                    benchmark_name=benchmark.name,
                    category=benchmark.category,
                    score=0.0,
                    unit="error",
                    higher_is_better=False,
                    details={'error': str(e)}
                ))

        total_duration = time.perf_counter() - start_time

        suite_result = BenchmarkSuiteResult(
            suite_name=self.suite_name,
            model_id=model_id,
            results=results,
            total_duration_seconds=total_duration,
            metadata=kwargs.get('metadata', {})
        )

        self.results_history.append(suite_result)
        return suite_result

    def compare_models(
        self,
        results_a: BenchmarkSuiteResult,
        results_b: BenchmarkSuiteResult
    ) -> dict[str, Any]:
        """
        Compare benchmark results between two models.

        Returns:
            Comparison report
        """
        comparison = {
            'model_a': results_a.model_id,
            'model_b': results_b.model_id,
            'benchmark_comparisons': {},
            'winner_counts': {'a': 0, 'b': 0, 'tie': 0},
            'aggregate_comparison': {}
        }

        # Compare individual benchmarks
        for result_a in results_a.results:
            result_b = next(
                (r for r in results_b.results if r.benchmark_name == result_a.benchmark_name),
                None
            )

            if result_b:
                diff = result_a.score - result_b.score
                if result_a.higher_is_better:
                    winner = 'a' if diff > 0 else 'b' if diff < 0 else 'tie'
                else:
                    winner = 'b' if diff > 0 else 'a' if diff < 0 else 'tie'

                comparison['benchmark_comparisons'][result_a.benchmark_name] = {
                    'score_a': result_a.score,
                    'score_b': result_b.score,
                    'difference': diff,
                    'winner': winner,
                    'unit': result_a.unit
                }
                comparison['winner_counts'][winner] += 1

        # Aggregate comparison
        agg_a = results_a.compute_aggregate_score()
        agg_b = results_b.compute_aggregate_score()
        comparison['aggregate_comparison'] = {
            'aggregate_a': agg_a,
            'aggregate_b': agg_b,
            'overall_winner': 'a' if agg_a > agg_b else 'b' if agg_b > agg_a else 'tie'
        }

        return comparison

    def save_results(self, result: BenchmarkSuiteResult, path: Path):
        """Save benchmark results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved benchmark results to {path}")

    def load_results(self, path: Path) -> BenchmarkSuiteResult:
        """Load benchmark results from file."""
        with open(path) as f:
            data = json.load(f)

        results = []
        for r in data['results']:
            results.append(BenchmarkResult(
                benchmark_name=r['benchmark_name'],
                category=BenchmarkCategory(r['category']),
                score=r['score'],
                unit=r['unit'],
                higher_is_better=r['higher_is_better'],
                details=r.get('details', {}),
                timestamp=datetime.fromisoformat(r['timestamp']),
                duration_seconds=r.get('duration_seconds', 0.0)
            ))

        return BenchmarkSuiteResult(
            suite_name=data['suite_name'],
            model_id=data['model_id'],
            results=results,
            total_duration_seconds=data['total_duration_seconds'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


def create_default_suite() -> BenchmarkSuite:
    """Create a default benchmark suite."""
    suite = BenchmarkSuite("ringrift_default")
    suite.add_default_benchmarks()
    return suite


def main():
    """Demonstrate benchmark suite."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available for demonstration")
        return

    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(18, 64, 3, padding=1)
            self.policy_head = nn.Linear(64 * 8 * 8, 64)
            self.value_head = nn.Linear(64 * 8 * 8, 1)
            self.input_shape = (18, 8, 8)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            policy = torch.softmax(self.policy_head(x), dim=1)
            value = torch.tanh(self.value_head(x))
            return policy, value

    # Create suite
    suite = BenchmarkSuite("demo_suite")
    suite.add_benchmark(InferenceBenchmark(batch_sizes=[1, 8]))
    suite.add_benchmark(MemoryBenchmark(batch_sizes=[1, 8]))
    suite.add_benchmark(RobustnessBenchmark())

    # Create model
    model = DummyModel()
    model.eval()

    # Generate some test positions for robustness benchmark
    test_positions = [
        {'state': torch.randn(18, 8, 8)}
        for _ in range(10)
    ]
    suite.benchmarks[-1].test_positions = test_positions

    # Run benchmarks
    print("Running benchmark suite...")
    result = suite.run(model, model_id="dummy_v1")

    # Print results
    print(f"\n=== {result.suite_name} Results ===")
    print(f"Model: {result.model_id}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")
    print(f"Aggregate Score: {result.compute_aggregate_score():.4f}")
    print()

    for r in result.results:
        print(f"{r.benchmark_name}:")
        print(f"  Score: {r.score:.4f} {r.unit}")
        print(f"  Category: {r.category.value}")

    # Save results
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        result_path = Path(f.name)

    suite.save_results(result, result_path)
    print(f"\nResults saved to {result_path}")

    # Load and verify
    loaded = suite.load_results(result_path)
    print(f"Loaded results for model: {loaded.model_id}")


if __name__ == "__main__":
    main()
