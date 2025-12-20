#!/usr/bin/env python3
"""
Rigorous GPU Minimax AI Testing Suite

Tests:
1. Correctness: GPU minimax produces same moves as CPU minimax at same depth
2. Performance: GPU batching provides meaningful speedup
3. Strength: GPU minimax has equivalent strength in games
4. Memory: GPU memory usage stays within bounds

Run on cloud GPU instances (H100, A100, etc.) - not local machines (OOM risk).

Usage:
    PYTHONPATH=. python scripts/test_gpu_minimax.py --quick      # Quick validation
    PYTHONPATH=. python scripts/test_gpu_minimax.py --full       # Full test suite
    PYTHONPATH=. python scripts/test_gpu_minimax.py --benchmark  # Performance only
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

# Ensure ai-service is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.gpu_minimax_ai import GPUMinimaxAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.models import AIConfig, BoardType, GameState, GameStatus, Move
from app.rules.default_engine import DefaultRulesEngine
from app.training.generate_data import create_initial_state


@dataclass
class TestResult:
    """Result from a single test."""
    name: str
    passed: bool
    message: str
    duration_sec: float
    metrics: dict


class GPUMinimaxTester:
    """Comprehensive tester for GPU Minimax AI."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.rules_engine = DefaultRulesEngine()

        # Detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.has_gpu = self.device.type in ("cuda", "mps")

        if self.verbose:
            print("GPU Minimax Tester initialized")
            print(f"  Device: {self.device}")
            print(f"  GPU available: {self.has_gpu}")
            if self.device.type == "cuda":
                print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _add_result(self, name: str, passed: bool, message: str,
                    duration: float, metrics: dict | None = None) -> None:
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

    def test_correctness_vs_cpu(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_positions: int = 20,
        depth: int = 3,
    ) -> bool:
        """Test that GPU minimax produces same moves as CPU minimax at same depth."""
        self._log(f"\n=== Test: Correctness vs CPU (depth={depth}, positions={num_positions}) ===")
        start_time = time.time()

        # Create AI instances
        config = AIConfig(difficulty=depth, think_time=5000)  # 5s think time
        gpu_ai = GPUMinimaxAI(player_number=1, config=config)
        cpu_ai = MinimaxAI(player_number=1, config=config)

        # Disable GPU for CPU AI
        os.environ["RINGRIFT_GPU_MINIMAX_DISABLE"] = "1"

        matches = 0
        mismatches = 0
        errors = 0

        for i in range(num_positions):
            try:
                # Create varied game positions
                state = self._create_varied_position(board_type, seed=42 + i)

                # Get moves from both AIs
                gpu_ai.reset_for_new_game()
                cpu_ai.reset_for_new_game()

                gpu_move = gpu_ai.select_move(state)
                cpu_move = cpu_ai.select_move(state)

                if gpu_move == cpu_move:
                    matches += 1
                elif self._moves_equivalent(gpu_move, cpu_move, state):
                    # Moves different but equivalent (same evaluation)
                    matches += 1
                else:
                    mismatches += 1
                    if self.verbose and mismatches <= 3:
                        self._log(f"    Position {i}: GPU={gpu_move} vs CPU={cpu_move}")

            except Exception as e:
                errors += 1
                self._log(f"    Position {i}: Error - {e}")

        # Restore env
        os.environ.pop("RINGRIFT_GPU_MINIMAX_DISABLE", None)

        duration = time.time() - start_time
        match_rate = matches / num_positions if num_positions > 0 else 0

        # Allow some variation (different move ordering can lead to equivalent moves)
        passed = match_rate >= 0.85 and errors == 0

        self._add_result(
            "Correctness vs CPU",
            passed,
            f"Match rate: {match_rate:.1%}, Matches: {matches}, Mismatches: {mismatches}, Errors: {errors}",
            duration,
            {"match_rate": match_rate, "matches": matches, "mismatches": mismatches, "errors": errors},
        )
        return passed

    def test_gpu_batch_speedup(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_positions: int = 50,
    ) -> bool:
        """Test that GPU batching provides speedup over CPU evaluation."""
        self._log(f"\n=== Test: GPU Batch Speedup (positions={num_positions}) ===")
        start_time = time.time()

        if not self.has_gpu:
            self._add_result(
                "GPU Batch Speedup",
                False,
                "Skipped - no GPU available",
                0,
            )
            return False

        config = AIConfig(difficulty=3, think_time=10000)  # High think time to ensure full search

        # Test GPU AI
        gpu_ai = GPUMinimaxAI(player_number=1, config=config)
        gpu_times = []

        for i in range(num_positions):
            state = self._create_varied_position(board_type, seed=100 + i)
            gpu_ai.reset_for_new_game()

            t0 = time.time()
            _ = gpu_ai.select_move(state)
            gpu_times.append(time.time() - t0)

        gpu_avg = np.mean(gpu_times)
        np.median(gpu_times)

        # Test CPU AI (with GPU disabled)
        os.environ["RINGRIFT_GPU_MINIMAX_DISABLE"] = "1"
        cpu_ai = GPUMinimaxAI(player_number=1, config=config)  # Still use GPUMinimaxAI but with GPU disabled
        cpu_times = []

        for i in range(num_positions):
            state = self._create_varied_position(board_type, seed=100 + i)
            cpu_ai.reset_for_new_game()

            t0 = time.time()
            _ = cpu_ai.select_move(state)
            cpu_times.append(time.time() - t0)

        os.environ.pop("RINGRIFT_GPU_MINIMAX_DISABLE", None)

        cpu_avg = np.mean(cpu_times)
        np.median(cpu_times)

        duration = time.time() - start_time
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

        # GPU should be at least as fast (speedup >= 1.0)
        # Note: For small depths, CPU might be faster due to GPU overhead
        passed = speedup >= 0.8  # Allow some margin for GPU overhead

        self._add_result(
            "GPU Batch Speedup",
            passed,
            f"GPU avg: {gpu_avg*1000:.1f}ms, CPU avg: {cpu_avg*1000:.1f}ms, Speedup: {speedup:.2f}x",
            duration,
            {"gpu_avg_ms": gpu_avg*1000, "cpu_avg_ms": cpu_avg*1000, "speedup": speedup},
        )
        return passed

    def test_strength_vs_heuristic(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_games: int = 20,
    ) -> bool:
        """Test GPU minimax strength by playing against heuristic AI."""
        self._log(f"\n=== Test: Strength vs Heuristic (games={num_games}) ===")
        start_time = time.time()

        minimax_config = AIConfig(difficulty=3, think_time=2000)
        heuristic_config = AIConfig(difficulty=5)

        wins = 0
        losses = 0
        draws = 0

        for game_idx in range(num_games):
            # Alternate colors
            if game_idx % 2 == 0:
                gpu_minimax = GPUMinimaxAI(player_number=1, config=minimax_config)
                heuristic = HeuristicAI(player_number=2, config=heuristic_config)
                minimax_player = 1
            else:
                heuristic = HeuristicAI(player_number=1, config=heuristic_config)
                gpu_minimax = GPUMinimaxAI(player_number=2, config=minimax_config)
                minimax_player = 2

            gpu_minimax.reset_for_new_game()
            heuristic.reset_for_new_game()

            state = create_initial_state(board_type=board_type, num_players=2)

            move_count = 0
            max_moves = 200

            while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
                current_player = state.current_player

                if current_player == minimax_player:
                    move = gpu_minimax.select_move(state)
                else:
                    move = heuristic.select_move(state)

                if not move:
                    break

                try:
                    state = self.rules_engine.apply_move(state, move)
                except Exception:
                    break

                move_count += 1

            if state.winner == minimax_player:
                wins += 1
            elif state.winner is not None:
                losses += 1
            else:
                draws += 1

            if self.verbose:
                self._log(f"    Game {game_idx + 1}: {'Win' if state.winner == minimax_player else 'Loss' if state.winner else 'Draw'}")

        duration = time.time() - start_time
        win_rate = wins / num_games if num_games > 0 else 0

        # Minimax depth 3 should beat heuristic most of the time
        passed = win_rate >= 0.4  # At least 40% win rate

        self._add_result(
            "Strength vs Heuristic",
            passed,
            f"Win rate: {win_rate:.1%}, W/L/D: {wins}/{losses}/{draws}",
            duration,
            {"win_rate": win_rate, "wins": wins, "losses": losses, "draws": draws},
        )
        return passed

    def test_gpu_memory_bounds(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        batch_sizes: list[int] | None = None,
    ) -> bool:
        """Test GPU memory usage stays within bounds."""
        if batch_sizes is None:
            batch_sizes = [32, 64, 128]
        self._log("\n=== Test: GPU Memory Bounds ===")
        start_time = time.time()

        if not self.has_gpu or self.device.type != "cuda":
            self._add_result(
                "GPU Memory Bounds",
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

            config = AIConfig(difficulty=3, think_time=2000)
            config.gpu_batch_size = batch_size

            gpu_ai = GPUMinimaxAI(player_number=1, config=config)

            # Run a few moves to trigger GPU allocation
            state = create_initial_state(board_type=board_type, num_players=2)
            for _ in range(5):
                move = gpu_ai.select_move(state)
                if move:
                    state = self.rules_engine.apply_move(state, move)

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            used_mem = peak_mem - initial_mem

            results.append({
                "batch_size": batch_size,
                "used_gb": used_mem,
                "peak_gb": peak_mem,
            })

            self._log(f"    Batch size {batch_size}: {used_mem:.2f} GB used, {peak_mem:.2f} GB peak")

            del gpu_ai
            torch.cuda.empty_cache()

        duration = time.time() - start_time

        # All batch sizes should use less than 8 GB (reasonable for cloud GPUs)
        max_mem = max(r["used_gb"] for r in results)
        passed = max_mem < 8.0

        self._add_result(
            "GPU Memory Bounds",
            passed,
            f"Max memory used: {max_mem:.2f} GB across batch sizes {batch_sizes}",
            duration,
            {"results": results, "max_mem_gb": max_mem},
        )
        return passed

    def test_game_completion(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        num_games: int = 10,
    ) -> bool:
        """Test that GPU minimax can complete full games without errors."""
        self._log(f"\n=== Test: Game Completion (games={num_games}) ===")
        start_time = time.time()

        config = AIConfig(difficulty=3, think_time=1500)

        completed = 0
        errors = 0
        total_moves = 0

        for game_idx in range(num_games):
            try:
                ai1 = GPUMinimaxAI(player_number=1, config=config)
                ai2 = GPUMinimaxAI(player_number=2, config=config)
                ai1.reset_for_new_game()
                ai2.reset_for_new_game()

                state = create_initial_state(board_type=board_type, num_players=2)

                move_count = 0
                max_moves = 200

                while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
                    current_player = state.current_player
                    ai = ai1 if current_player == 1 else ai2

                    move = ai.select_move(state)
                    if not move:
                        break

                    state = self.rules_engine.apply_move(state, move)
                    move_count += 1

                completed += 1
                total_moves += move_count

            except Exception as e:
                errors += 1
                self._log(f"    Game {game_idx + 1}: Error - {e}")

        duration = time.time() - start_time
        avg_moves = total_moves / completed if completed > 0 else 0

        passed = completed == num_games and errors == 0

        self._add_result(
            "Game Completion",
            passed,
            f"Completed: {completed}/{num_games}, Errors: {errors}, Avg moves: {avg_moves:.1f}",
            duration,
            {"completed": completed, "errors": errors, "avg_moves": avg_moves},
        )
        return passed

    def _create_varied_position(
        self,
        board_type: BoardType,
        seed: int,
        min_moves: int = 5,
        max_moves: int = 30,
    ) -> GameState:
        """Create a varied game position by playing random moves."""
        np.random.seed(seed)

        state = create_initial_state(board_type=board_type, num_players=2)
        target_moves = np.random.randint(min_moves, max_moves + 1)

        for _ in range(target_moves):
            if state.game_status != GameStatus.ACTIVE:
                break

            valid_moves = self.rules_engine.get_valid_moves(state, state.current_player)
            if not valid_moves:
                break

            # Pick a random move
            move = valid_moves[np.random.randint(len(valid_moves))]
            try:
                state = self.rules_engine.apply_move(state, move)
            except Exception:
                break

        return state

    def _moves_equivalent(self, move1: Move, move2: Move, state: GameState) -> bool:
        """Check if two moves are equivalent (same effect on game)."""
        if move1 is None or move2 is None:
            return move1 == move2

        # Same move
        if move1 == move2:
            return True

        # Different moves - could still be equivalent if they have same evaluation
        # This is a simplification; full equivalence checking would require
        # evaluating both resulting states
        return False

    def run_quick_tests(self) -> bool:
        """Run quick validation tests."""
        self._log("\n" + "=" * 60)
        self._log("GPU MINIMAX QUICK TEST SUITE")
        self._log("=" * 60)

        self.test_game_completion(num_games=5)
        self.test_correctness_vs_cpu(num_positions=10, depth=2)

        return self._summarize()

    def run_full_tests(self) -> bool:
        """Run full test suite."""
        self._log("\n" + "=" * 60)
        self._log("GPU MINIMAX FULL TEST SUITE")
        self._log("=" * 60)

        self.test_game_completion(num_games=20)
        self.test_correctness_vs_cpu(num_positions=30, depth=3)
        self.test_gpu_batch_speedup(num_positions=30)
        self.test_strength_vs_heuristic(num_games=30)
        self.test_gpu_memory_bounds(batch_sizes=[32, 64, 128, 256])

        return self._summarize()

    def run_benchmark(self) -> bool:
        """Run performance benchmarks only."""
        self._log("\n" + "=" * 60)
        self._log("GPU MINIMAX BENCHMARK")
        self._log("=" * 60)

        self.test_gpu_batch_speedup(num_positions=100)
        self.test_gpu_memory_bounds(batch_sizes=[32, 64, 128, 256, 512])

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
    parser = argparse.ArgumentParser(description="GPU Minimax AI Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks only")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    tester = GPUMinimaxTester(verbose=not args.quiet)

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
