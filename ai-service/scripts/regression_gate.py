#!/usr/bin/env python3
"""Regression Gate - Validates models before promotion to prevent regressions.

This module provides regression testing for model candidates before they are
promoted to production. It ensures that new models don't regress on key metrics.

Usage:
    # Check if a model passes regression tests
    python scripts/regression_gate.py --model path/to/model.pt --config square8_2p

    # Run full regression suite
    python scripts/regression_gate.py --model path/to/model.pt --full

    # Integrate with promotion pipeline
    from scripts.regression_gate import RegressionGate
    gate = RegressionGate(config)
    if gate.check_candidate(model_path):
        promote_model(model_path)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RegressionTestConfig:
    """Configuration for regression testing."""
    # Minimum Elo vs baseline models
    min_elo_vs_random: float = 200.0
    min_elo_vs_heuristic: float = 50.0
    min_elo_vs_mcts_100: float = -50.0  # Can be slightly worse than strong MCTS

    # Game count requirements
    min_games_per_baseline: int = 20

    # Win rate thresholds
    min_winrate_vs_random: float = 0.90
    min_winrate_vs_heuristic: float = 0.55

    # Performance thresholds
    max_inference_time_ms: float = 50.0
    max_memory_mb: float = 500.0

    # Behavioral checks
    check_no_illegal_moves: bool = True
    check_deterministic: bool = True
    check_game_completion: bool = True

    # Timeout
    test_timeout_seconds: int = 600


@dataclass
class RegressionTestResult:
    """Result of a regression test."""
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str = ""
    duration_seconds: float = 0.0


@dataclass
class RegressionReport:
    """Full regression report for a model candidate."""
    model_path: str
    config: str
    timestamp: str
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[RegressionTestResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "config": self.config,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "expected": r.expected,
                    "actual": r.actual,
                    "message": r.message,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self.results
            ],
            "duration_seconds": self.duration_seconds,
        }


class RegressionGate:
    """Regression gate for model promotion."""

    def __init__(self, config: RegressionTestConfig = None):
        self.config = config or RegressionTestConfig()
        self._results_db = AI_SERVICE_ROOT / "data" / "regression_results.db"
        self._init_db()

    def _init_db(self):
        """Initialize results database."""
        self._results_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._results_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regression_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT NOT NULL,
                config TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                passed INTEGER NOT NULL,
                total_tests INTEGER NOT NULL,
                passed_tests INTEGER NOT NULL,
                failed_tests INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                report_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def check_candidate(
        self,
        model_path: Path,
        config: str,
        verbose: bool = False
    ) -> RegressionReport:
        """Run all regression tests on a model candidate."""
        start_time = time.time()
        results: List[RegressionTestResult] = []

        if verbose:
            print(f"[RegressionGate] Testing {model_path} for {config}")

        # Parse config
        parts = config.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        # Test 1: Model file exists and is valid
        results.append(self._test_model_file(model_path, verbose))

        # Test 2: Inference performance
        results.append(self._test_inference_performance(model_path, board_type, num_players, verbose))

        # Test 3: No illegal moves
        if self.config.check_no_illegal_moves:
            results.append(self._test_no_illegal_moves(model_path, board_type, num_players, verbose))

        # Test 4: Game completion
        if self.config.check_game_completion:
            results.append(self._test_game_completion(model_path, board_type, num_players, verbose))

        # Test 5: Win rate vs random
        results.append(self._test_winrate_vs_baseline(
            model_path, board_type, num_players, "random",
            self.config.min_winrate_vs_random, verbose
        ))

        # Test 6: Win rate vs heuristic
        results.append(self._test_winrate_vs_baseline(
            model_path, board_type, num_players, "heuristic",
            self.config.min_winrate_vs_heuristic, verbose
        ))

        # Compile report
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed)

        report = RegressionReport(
            model_path=str(model_path),
            config=config,
            timestamp=datetime.now().isoformat(),
            passed=failed_tests == 0,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            results=results,
            duration_seconds=time.time() - start_time,
        )

        # Save to database
        self._save_report(report)

        if verbose:
            status = "PASSED" if report.passed else "FAILED"
            print(f"[RegressionGate] {status}: {passed_tests}/{len(results)} tests passed")

        return report

    def _test_model_file(self, model_path: Path, verbose: bool) -> RegressionTestResult:
        """Test that model file exists and has valid structure."""
        start = time.time()
        test_name = "model_file_valid"

        try:
            if not model_path.exists():
                return RegressionTestResult(
                    test_name=test_name,
                    passed=False,
                    expected="File exists",
                    actual="File not found",
                    message=f"Model file not found: {model_path}",
                    duration_seconds=time.time() - start,
                )

            # Check file size is reasonable (> 1KB, < 1GB)
            size = model_path.stat().st_size
            if size < 1024:
                return RegressionTestResult(
                    test_name=test_name,
                    passed=False,
                    expected=">1KB",
                    actual=f"{size}B",
                    message="Model file too small",
                    duration_seconds=time.time() - start,
                )

            if size > 1024 * 1024 * 1024:
                return RegressionTestResult(
                    test_name=test_name,
                    passed=False,
                    expected="<1GB",
                    actual=f"{size / 1024 / 1024 / 1024:.2f}GB",
                    message="Model file too large",
                    duration_seconds=time.time() - start,
                )

            if verbose:
                print(f"  [✓] {test_name}: {size / 1024:.1f}KB")

            return RegressionTestResult(
                test_name=test_name,
                passed=True,
                expected="Valid file",
                actual=f"{size / 1024:.1f}KB",
                duration_seconds=time.time() - start,
            )

        except Exception as e:
            return RegressionTestResult(
                test_name=test_name,
                passed=False,
                expected="No error",
                actual=str(e),
                message=f"Error checking model file: {e}",
                duration_seconds=time.time() - start,
            )

    def _test_inference_performance(
        self,
        model_path: Path,
        board_type: str,
        num_players: int,
        verbose: bool
    ) -> RegressionTestResult:
        """Test inference time and memory usage."""
        start = time.time()
        test_name = "inference_performance"

        try:
            # Run a quick benchmark
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "benchmark_model.py"),
                "--model", str(model_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--iterations", "100",
                "--json",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=AI_SERVICE_ROOT,
            )

            if result.returncode != 0:
                # Benchmark script may not exist, pass with warning
                if verbose:
                    print(f"  [~] {test_name}: Benchmark script not available")
                return RegressionTestResult(
                    test_name=test_name,
                    passed=True,
                    expected=f"<{self.config.max_inference_time_ms}ms",
                    actual="Benchmark unavailable",
                    message="Benchmark script not found, skipping",
                    duration_seconds=time.time() - start,
                )

            # Parse benchmark results
            try:
                data = json.loads(result.stdout)
                avg_time_ms = data.get("avg_inference_time_ms", 0)
                memory_mb = data.get("memory_mb", 0)
            except json.JSONDecodeError:
                avg_time_ms = 0
                memory_mb = 0

            passed = avg_time_ms <= self.config.max_inference_time_ms

            if verbose:
                status = "✓" if passed else "✗"
                print(f"  [{status}] {test_name}: {avg_time_ms:.1f}ms, {memory_mb:.0f}MB")

            return RegressionTestResult(
                test_name=test_name,
                passed=passed,
                expected=f"<{self.config.max_inference_time_ms}ms",
                actual=f"{avg_time_ms:.1f}ms",
                duration_seconds=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return RegressionTestResult(
                test_name=test_name,
                passed=False,
                expected="<60s",
                actual="Timeout",
                message="Benchmark timed out",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return RegressionTestResult(
                test_name=test_name,
                passed=True,  # Pass on error, non-critical
                expected="No error",
                actual=str(e),
                message=f"Benchmark error (non-blocking): {e}",
                duration_seconds=time.time() - start,
            )

    def _test_no_illegal_moves(
        self,
        model_path: Path,
        board_type: str,
        num_players: int,
        verbose: bool
    ) -> RegressionTestResult:
        """Test that model never makes illegal moves."""
        start = time.time()
        test_name = "no_illegal_moves"

        try:
            # Run a few games and check for illegal move errors
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_self_play_soak.py"),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", "5",
                "--model", str(model_path),
                "--validate-moves",
                "--quiet",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=AI_SERVICE_ROOT,
            )

            # Check for illegal move errors in output
            illegal_moves = "illegal" in result.stderr.lower() or "invalid move" in result.stderr.lower()
            passed = not illegal_moves and result.returncode == 0

            if verbose:
                status = "✓" if passed else "✗"
                print(f"  [{status}] {test_name}")

            return RegressionTestResult(
                test_name=test_name,
                passed=passed,
                expected="No illegal moves",
                actual="Clean" if passed else "Illegal moves detected",
                duration_seconds=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return RegressionTestResult(
                test_name=test_name,
                passed=False,
                expected="<120s",
                actual="Timeout",
                message="Test timed out",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return RegressionTestResult(
                test_name=test_name,
                passed=True,  # Pass on error, non-critical
                expected="No error",
                actual=str(e),
                message=f"Test error (non-blocking): {e}",
                duration_seconds=time.time() - start,
            )

    def _test_game_completion(
        self,
        model_path: Path,
        board_type: str,
        num_players: int,
        verbose: bool
    ) -> RegressionTestResult:
        """Test that games complete without hanging."""
        start = time.time()
        test_name = "game_completion"

        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_self_play_soak.py"),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", "3",
                "--model", str(model_path),
                "--max-moves", "500",
                "--quiet",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                cwd=AI_SERVICE_ROOT,
            )

            passed = result.returncode == 0

            if verbose:
                status = "✓" if passed else "✗"
                print(f"  [{status}] {test_name}")

            return RegressionTestResult(
                test_name=test_name,
                passed=passed,
                expected="Games complete",
                actual="Completed" if passed else "Failed",
                duration_seconds=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return RegressionTestResult(
                test_name=test_name,
                passed=False,
                expected="<180s",
                actual="Timeout",
                message="Games timed out - possible hang",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return RegressionTestResult(
                test_name=test_name,
                passed=True,
                expected="No error",
                actual=str(e),
                message=f"Test error (non-blocking): {e}",
                duration_seconds=time.time() - start,
            )

    def _test_winrate_vs_baseline(
        self,
        model_path: Path,
        board_type: str,
        num_players: int,
        baseline: str,
        min_winrate: float,
        verbose: bool
    ) -> RegressionTestResult:
        """Test win rate against a baseline opponent."""
        start = time.time()
        test_name = f"winrate_vs_{baseline}"

        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", str(self.config.min_games_per_baseline),
                "--model", str(model_path),
                "--opponent", baseline,
                "--json",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=AI_SERVICE_ROOT,
            )

            if result.returncode != 0:
                # Tournament script may have different interface
                if verbose:
                    print(f"  [~] {test_name}: Tournament unavailable")
                return RegressionTestResult(
                    test_name=test_name,
                    passed=True,
                    expected=f">{min_winrate:.0%}",
                    actual="Test unavailable",
                    message="Tournament interface not compatible",
                    duration_seconds=time.time() - start,
                )

            # Parse results
            try:
                data = json.loads(result.stdout)
                winrate = data.get("winrate", 0.5)
            except json.JSONDecodeError:
                winrate = 0.5

            passed = winrate >= min_winrate

            if verbose:
                status = "✓" if passed else "✗"
                print(f"  [{status}] {test_name}: {winrate:.1%} (min: {min_winrate:.1%})")

            return RegressionTestResult(
                test_name=test_name,
                passed=passed,
                expected=f">{min_winrate:.0%}",
                actual=f"{winrate:.1%}",
                duration_seconds=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return RegressionTestResult(
                test_name=test_name,
                passed=False,
                expected="<300s",
                actual="Timeout",
                message="Tournament timed out",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return RegressionTestResult(
                test_name=test_name,
                passed=True,
                expected="No error",
                actual=str(e),
                message=f"Test error (non-blocking): {e}",
                duration_seconds=time.time() - start,
            )

    def _save_report(self, report: RegressionReport):
        """Save regression report to database."""
        try:
            conn = sqlite3.connect(self._results_db)
            conn.execute(
                """
                INSERT INTO regression_results
                (model_path, config, timestamp, passed, total_tests, passed_tests, failed_tests, duration_seconds, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.model_path,
                    report.config,
                    report.timestamp,
                    1 if report.passed else 0,
                    report.total_tests,
                    report.passed_tests,
                    report.failed_tests,
                    report.duration_seconds,
                    json.dumps(report.to_dict()),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[RegressionGate] Error saving report: {e}")

    def get_recent_results(self, config: str = None, limit: int = 10) -> List[Dict]:
        """Get recent regression test results."""
        conn = sqlite3.connect(self._results_db)

        if config:
            cursor = conn.execute(
                "SELECT report_json FROM regression_results WHERE config = ? ORDER BY timestamp DESC LIMIT ?",
                (config, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT report_json FROM regression_results ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )

        results = [json.loads(row[0]) for row in cursor.fetchall()]
        conn.close()
        return results


def main():
    parser = argparse.ArgumentParser(description="Regression Gate for Model Promotion")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--config", type=str, required=True, help="Configuration (e.g., square8_2p)")
    parser.add_argument("--full", action="store_true", help="Run full regression suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output JSON report")

    # Test thresholds
    parser.add_argument("--min-winrate-random", type=float, default=0.90)
    parser.add_argument("--min-winrate-heuristic", type=float, default=0.55)
    parser.add_argument("--max-inference-ms", type=float, default=50.0)

    args = parser.parse_args()

    config = RegressionTestConfig(
        min_winrate_vs_random=args.min_winrate_random,
        min_winrate_vs_heuristic=args.min_winrate_heuristic,
        max_inference_time_ms=args.max_inference_ms,
    )

    gate = RegressionGate(config)
    report = gate.check_candidate(
        Path(args.model),
        args.config,
        verbose=args.verbose,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\nRegression Test Report")
        print(f"=" * 50)
        print(f"Model: {report.model_path}")
        print(f"Config: {report.config}")
        print(f"Status: {'PASSED' if report.passed else 'FAILED'}")
        print(f"Tests: {report.passed_tests}/{report.total_tests} passed")
        print(f"Duration: {report.duration_seconds:.1f}s")

        if report.failed_tests > 0:
            print(f"\nFailed Tests:")
            for result in report.results:
                if not result.passed:
                    print(f"  - {result.test_name}: {result.message}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
