"""Performance benchmarks for PromotionController and EloReconciler.

Benchmarks the critical paths in the promotion and Elo reconciliation system:
- PromotionController.evaluate_promotion() latency
- EloReconciler.check_drift() with various database sizes
- EloReconciler._import_matches() throughput
- Metrics emission overhead

Run with: pytest tests/benchmarks/test_promotion_elo_benchmarks.py -v --benchmark-only
Or without pytest-benchmark: python tests/benchmarks/test_promotion_elo_benchmarks.py
"""

import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Tuple
from unittest.mock import MagicMock

# Add parent directories to path for standalone execution
_THIS_DIR = Path(__file__).resolve().parent
_AI_SERVICE_ROOT = _THIS_DIR.parents[1]
if str(_AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_SERVICE_ROOT))

import pytest

from app.training.elo_reconciliation import EloReconciler
from app.training.promotion_controller import (
    PromotionController,
    PromotionCriteria,
    PromotionType,
)


# Check if pytest-benchmark is available
try:
    import pytest_benchmark  # noqa
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


def simple_benchmark(func: Callable, iterations: int = 100) -> Tuple[float, float, float]:
    """Simple benchmark without pytest-benchmark."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    return avg, min_t, max_t


class TestPromotionControllerBenchmarks:
    """Benchmark PromotionController performance."""

    def setup_method(self):
        """Set up benchmark fixtures."""
        self.mock_elo = MagicMock()
        self.mock_rating = MagicMock()
        self.mock_rating.rating = 1550.0
        self.mock_rating.games_played = 100
        self.mock_rating.win_rate = 0.55

        self.mock_baseline = MagicMock()
        self.mock_baseline.rating = 1500.0

        self.mock_elo.get_rating.side_effect = lambda *args, **kwargs: (
            self.mock_rating if args[0] == "test_model" else self.mock_baseline
        )

        self.controller = PromotionController(
            criteria=PromotionCriteria(min_elo_improvement=25.0, min_games_played=50),
            elo_service=self.mock_elo,
        )

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
    def test_evaluate_promotion_latency(self, benchmark):
        """Benchmark evaluate_promotion latency."""
        benchmark(
            self.controller.evaluate_promotion,
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id="baseline",
        )

    def test_evaluate_promotion_latency_simple(self):
        """Simple benchmark for evaluate_promotion without pytest-benchmark."""
        def run_eval():
            self.controller.evaluate_promotion(
                model_id="test_model",
                promotion_type=PromotionType.PRODUCTION,
                baseline_model_id="baseline",
            )

        avg, min_t, max_t = simple_benchmark(run_eval, iterations=100)
        print(f"\nevaluate_promotion: avg={avg*1000:.3f}ms, min={min_t*1000:.3f}ms, max={max_t*1000:.3f}ms")

        # Assert reasonable performance (should be <10ms without DB)
        assert avg < 0.01, f"evaluate_promotion too slow: {avg*1000:.3f}ms"


class TestEloReconcilerBenchmarks:
    """Benchmark EloReconciler performance."""

    def setup_method(self):
        """Set up benchmark fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "benchmark_elo.db"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_db_with_participants(self, n_participants: int):
        """Create a database with n participants."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT PRIMARY KEY,
                board_type TEXT DEFAULT 'square8',
                num_players INTEGER DEFAULT 2,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_history (
                match_id TEXT PRIMARY KEY,
                player1_id TEXT NOT NULL,
                player2_id TEXT NOT NULL,
                winner_id TEXT,
                player1_rating_before REAL,
                player2_rating_before REAL,
                player1_rating_after REAL,
                player2_rating_after REAL,
                board_type TEXT,
                num_players INTEGER,
                game_length INTEGER,
                timestamp TEXT,
                source TEXT
            )
        """)

        # Insert participants in batches for efficiency
        batch_size = 1000
        for i in range(0, n_participants, batch_size):
            batch_end = min(i + batch_size, n_participants)
            data = [
                (f"model_{j}", 1500 + (j % 100), 50 + (j % 50))
                for j in range(i, batch_end)
            ]
            cursor.executemany(
                "INSERT INTO elo_ratings (participant_id, rating, games_played) VALUES (?, ?, ?)",
                data,
            )

        conn.commit()
        conn.close()

    def test_check_drift_100_participants(self):
        """Benchmark check_drift with 100 participants."""
        self._create_db_with_participants(100)
        reconciler = EloReconciler(local_db_path=self.db_path)

        def run_check():
            reconciler.check_drift()

        avg, min_t, max_t = simple_benchmark(run_check, iterations=50)
        print(f"\ncheck_drift(100 participants): avg={avg*1000:.3f}ms, min={min_t*1000:.3f}ms, max={max_t*1000:.3f}ms")

        # Should be fast (<50ms) for small DBs
        assert avg < 0.05, f"check_drift too slow for 100 participants: {avg*1000:.3f}ms"

    def test_check_drift_1000_participants(self):
        """Benchmark check_drift with 1000 participants."""
        self._create_db_with_participants(1000)
        reconciler = EloReconciler(local_db_path=self.db_path)

        def run_check():
            reconciler.check_drift()

        avg, min_t, max_t = simple_benchmark(run_check, iterations=20)
        print(f"\ncheck_drift(1000 participants): avg={avg*1000:.3f}ms, min={min_t*1000:.3f}ms, max={max_t*1000:.3f}ms")

        # Should scale reasonably (<100ms)
        assert avg < 0.1, f"check_drift too slow for 1000 participants: {avg*1000:.3f}ms"

    def test_import_matches_throughput(self):
        """Benchmark _import_matches throughput."""
        self._create_db_with_participants(10)  # Small participant set
        reconciler = EloReconciler(local_db_path=self.db_path)

        # Generate 1000 matches
        matches = [
            {
                "match_id": f"match_{i}",
                "player1_id": f"player_{i % 10}",
                "player2_id": f"player_{(i + 1) % 10}",
                "winner_id": f"player_{i % 10}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(1000)
        ]

        start = time.perf_counter()
        result = reconciler._import_matches("benchmark_host", "now", matches)
        elapsed = time.perf_counter() - start

        throughput = len(matches) / elapsed
        print(f"\n_import_matches: {len(matches)} matches in {elapsed*1000:.2f}ms ({throughput:.0f} matches/sec)")
        print(f"  Added: {result.matches_added}, Skipped: {result.matches_skipped}")

        # Should achieve >100 matches/sec
        assert throughput > 100, f"_import_matches throughput too low: {throughput:.0f} matches/sec"

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
    def test_check_drift_benchmark(self, benchmark):
        """pytest-benchmark version of check_drift."""
        self._create_db_with_participants(500)
        reconciler = EloReconciler(local_db_path=self.db_path)

        benchmark(reconciler.check_drift)


class TestMetricsOverhead:
    """Benchmark metrics emission overhead."""

    def test_promotion_decision_metrics_overhead(self):
        """Benchmark overhead of metrics emission in promotion decisions."""
        # Without metrics (mock)
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.rating = 1550.0
        mock_rating.games_played = 100
        mock_rating.win_rate = 0.55
        mock_baseline = MagicMock()
        mock_baseline.rating = 1500.0
        mock_elo.get_rating.side_effect = [mock_rating, mock_baseline] * 200

        controller = PromotionController(elo_service=mock_elo)

        # Measure with metrics enabled (default)
        def run_with_metrics():
            controller.evaluate_promotion(
                model_id="test",
                promotion_type=PromotionType.PRODUCTION,
                baseline_model_id="baseline",
            )

        avg_with, _, _ = simple_benchmark(run_with_metrics, iterations=100)

        # Metrics overhead should be <1ms
        print(f"\nPromotion evaluation with metrics: avg={avg_with*1000:.3f}ms")
        print(f"Metrics overhead is acceptable if total time <10ms")

        assert avg_with < 0.01, f"Metrics overhead too high: {avg_with*1000:.3f}ms"


class TestConcurrency:
    """Test concurrent access to EloReconciler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "concurrent_elo.db"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_db_with_data(self, n_participants: int = 100, n_matches: int = 500):
        """Create database with test data."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT PRIMARY KEY,
                board_type TEXT DEFAULT 'square8',
                num_players INTEGER DEFAULT 2,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_history (
                match_id TEXT PRIMARY KEY,
                player1_id TEXT NOT NULL,
                player2_id TEXT NOT NULL,
                winner_id TEXT,
                player1_rating_before REAL,
                player2_rating_before REAL,
                player1_rating_after REAL,
                player2_rating_after REAL,
                board_type TEXT,
                num_players INTEGER,
                game_length INTEGER,
                timestamp TEXT,
                source TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0
            )
        """)

        # Insert participants
        for i in range(n_participants):
            cursor.execute(
                "INSERT INTO elo_ratings (participant_id, rating, games_played) VALUES (?, ?, ?)",
                (f"model_{i}", 1500 + (i % 100), 50 + (i % 50)),
            )
            cursor.execute(
                "INSERT INTO participants (participant_id, rating, games_played) VALUES (?, ?, ?)",
                (f"model_{i}", 1500 + (i % 100), 50 + (i % 50)),
            )

        # Insert matches
        for i in range(n_matches):
            cursor.execute(
                """INSERT INTO match_history (match_id, player1_id, player2_id, winner_id, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    f"existing_match_{i}",
                    f"model_{i % n_participants}",
                    f"model_{(i + 1) % n_participants}",
                    f"model_{i % n_participants}",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        conn.commit()
        conn.close()

    def test_concurrent_drift_checks(self):
        """Test concurrent drift check calls don't cause issues."""
        import concurrent.futures

        self._create_db_with_data(100, 200)

        reconciler = EloReconciler(local_db_path=self.db_path)
        errors = []
        results = []

        def check_drift():
            try:
                drift = reconciler.check_drift()
                return drift.participants_in_source
            except Exception as e:
                errors.append(str(e))
                return None

        # Run 10 concurrent drift checks
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_drift) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        print(f"\nConcurrent drift checks: {len([r for r in results if r])} successful, {len(errors)} errors")

        # All should succeed
        assert len(errors) == 0, f"Concurrent drift checks had errors: {errors}"
        assert all(r == 100 for r in results if r is not None), "Inconsistent drift check results"

    def test_concurrent_match_imports(self):
        """Test concurrent match imports with conflict resolution."""
        import concurrent.futures
        from app.training.elo_reconciliation import ConflictResolution

        self._create_db_with_data(10, 50)

        reconciler = EloReconciler(
            local_db_path=self.db_path,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
        )
        errors = []
        results = []

        def import_batch(batch_id: int):
            try:
                matches = [
                    {
                        "match_id": f"concurrent_match_{batch_id}_{i}",
                        "player1_id": f"model_{i % 10}",
                        "player2_id": f"model_{(i + 1) % 10}",
                        "winner_id": f"model_{i % 10}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    for i in range(10)
                ]
                result = reconciler._import_matches(f"batch_{batch_id}", "now", matches)
                return result.matches_added
            except Exception as e:
                errors.append(str(e))
                return 0

        # Run 5 concurrent batch imports
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(import_batch, i) for i in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        total_added = sum(results)
        print(f"\nConcurrent imports: {total_added} matches added across {len(results)} batches")
        print(f"Errors: {len(errors)}")

        # Should have added 50 matches total (5 batches * 10 matches each)
        assert len(errors) == 0, f"Concurrent imports had errors: {errors}"
        assert total_added == 50, f"Expected 50 matches added, got {total_added}"

    def test_load_large_batch_import(self):
        """Test importing large batch of matches."""
        self._create_db_with_data(10, 0)  # Start with no matches

        reconciler = EloReconciler(local_db_path=self.db_path)

        # Generate 10,000 matches
        matches = [
            {
                "match_id": f"load_match_{i}",
                "player1_id": f"model_{i % 10}",
                "player2_id": f"model_{(i + 1) % 10}",
                "winner_id": f"model_{i % 10}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(10000)
        ]

        start = time.perf_counter()
        result = reconciler._import_matches("load_test", "now", matches)
        elapsed = time.perf_counter() - start

        throughput = result.matches_added / elapsed
        print(f"\nLarge batch import: {result.matches_added} matches in {elapsed:.2f}s ({throughput:.0f} matches/sec)")

        # Should achieve reasonable throughput
        assert result.matches_added == 10000, f"Expected 10000 matches, got {result.matches_added}"
        assert throughput > 1000, f"Throughput too low: {throughput:.0f} matches/sec"


def run_all_benchmarks():
    """Run all benchmarks without pytest."""
    print("=" * 60)
    print("Running Performance Benchmarks")
    print("=" * 60)

    # PromotionController benchmarks
    print("\n--- PromotionController Benchmarks ---")
    promo_bench = TestPromotionControllerBenchmarks()
    promo_bench.setup_method()
    promo_bench.test_evaluate_promotion_latency_simple()

    # EloReconciler benchmarks
    print("\n--- EloReconciler Benchmarks ---")
    elo_bench = TestEloReconcilerBenchmarks()

    elo_bench.setup_method()
    elo_bench.test_check_drift_100_participants()
    elo_bench.teardown_method()

    elo_bench.setup_method()
    elo_bench.test_check_drift_1000_participants()
    elo_bench.teardown_method()

    elo_bench.setup_method()
    elo_bench.test_import_matches_throughput()
    elo_bench.teardown_method()

    # Metrics overhead
    print("\n--- Metrics Overhead Benchmarks ---")
    metrics_bench = TestMetricsOverhead()
    metrics_bench.test_promotion_decision_metrics_overhead()

    # Concurrency tests
    print("\n--- Concurrency Benchmarks ---")
    concurrency_bench = TestConcurrency()

    concurrency_bench.setup_method()
    concurrency_bench.test_concurrent_drift_checks()
    concurrency_bench.teardown_method()

    concurrency_bench.setup_method()
    concurrency_bench.test_concurrent_match_imports()
    concurrency_bench.teardown_method()

    concurrency_bench.setup_method()
    concurrency_bench.test_load_large_batch_import()
    concurrency_bench.teardown_method()

    print("\n" + "=" * 60)
    print("All benchmarks completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
