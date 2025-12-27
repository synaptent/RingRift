"""Gauntlet HTTP Handlers Mixin.

Provides HTTP endpoints for distributed gauntlet evaluation.
Enables parallel model evaluation across cluster nodes by distributing
game batches to worker nodes and aggregating results.

Usage:
    class P2POrchestrator(GauntletHandlersMixin, ...):
        pass

Endpoints:
    POST /gauntlet/execute - Execute a batch of gauntlet games (worker)
    POST /gauntlet/start - Start distributed gauntlet run (leader only)
    GET /gauntlet/status - Get gauntlet run status and progress
    GET /gauntlet/results - Get completed gauntlet results

Gauntlet Workflow:
    1. Leader starts gauntlet via /gauntlet/start with model and opponents
    2. Leader distributes game batches to workers via /gauntlet/execute
    3. Workers play games and return results
    4. Leader aggregates results and determines win rates
    5. Results used for model promotion decisions (vs RANDOM: 85%, vs HEURISTIC: 60%)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Event bridge import (with fallback)
try:
    from scripts.p2p.p2p_event_bridge import emit_p2p_gauntlet_completed
    HAS_EVENT_BRIDGE = True
except ImportError as e:
    HAS_EVENT_BRIDGE = False
    # Dec 2025: Log import failure so operators know events aren't being emitted
    logger.warning(f"[GauntletHandlers] Event bridge not available ({e}), gauntlet events will not be emitted")

    async def emit_p2p_gauntlet_completed(*args, **kwargs):
        pass


class GauntletHandlersMixin:
    """Mixin providing gauntlet HTTP handlers.

    Requires the implementing class to have:
    - node_id: str
    - self_info: NodeInfo
    - ringrift_path: str
    """

    # Type hints for IDE support
    node_id: str
    ringrift_path: str

    async def handle_gauntlet_execute(self, request: web.Request) -> web.Response:
        """POST /gauntlet/execute - Execute a batch of gauntlet games.

        Workers receive batches of games to play and return results.
        This endpoint allows distributed gauntlet evaluation across the cluster.

        Request body:
            {
                "run_id": "abc123",
                "config_key": "square8_2p",
                "tasks": [
                    {"task_id": "...", "model_id": "...", "baseline_id": "...", "game_num": 0},
                    ...
                ]
            }

        Response:
            {
                "success": true,
                "results": [
                    {"task_id": "...", "model_id": "...", "model_won": true, ...},
                    ...
                ]
            }
        """
        try:
            data = await request.json()
        except (AttributeError, ValueError, UnicodeDecodeError):
            # Dec 2025: Catch all JSON parsing errors
            return web.json_response({"error": "Invalid JSON"}, status=400)

        run_id = data.get("run_id", "unknown")
        config_key = data.get("config_key", "")
        tasks = data.get("tasks", [])

        if not config_key or not tasks:
            return web.json_response({
                "error": "config_key and tasks required"
            }, status=400)

        logger.info(f"Gauntlet: Executing {len(tasks)} games for {config_key} (run {run_id})")

        try:
            results = await self._execute_gauntlet_batch(config_key, tasks)

            return web.json_response({
                "success": True,
                "node_id": self.node_id,
                "run_id": run_id,
                "games_completed": len(results),
                "results": results,
            })

        except Exception as e:
            logger.info(f"Gauntlet execution error: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
            }, status=500)

    async def _execute_gauntlet_batch(
        self,
        config_key: str,
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute a batch of gauntlet games.

        Args:
            config_key: Config like "square8_2p"
            tasks: List of task dicts

        Returns:
            List of result dicts
        """
        results = []

        # Parse config
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        # Import game execution modules
        try:
            pass
        except ImportError as e:
            logger.info(f"Gauntlet: Import error: {e}")
            # Return simulated results if modules not available
            for task in tasks:
                results.append({
                    "task_id": task["task_id"],
                    "model_id": task["model_id"],
                    "baseline_id": task["baseline_id"],
                    "model_won": False,
                    "baseline_won": True,
                    "draw": False,
                    "game_length": 0,
                    "duration_sec": 0.0,
                    "error": "Game modules not available",
                })
            return results

        # Load model paths
        model_dir = Path(self.ringrift_path) / "ai-service" / "data" / "models"

        # Execute games concurrently in small batches
        batch_size = 4  # Run 4 games at a time
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start:batch_start + batch_size]

            batch_coros = [
                self._execute_single_gauntlet_game(
                    task, board_type, num_players, model_dir
                )
                for task in batch
            ]

            batch_results = await asyncio.gather(*batch_coros, return_exceptions=True)

            for task, result in zip(batch, batch_results, strict=False):
                if isinstance(result, Exception):
                    results.append({
                        "task_id": task["task_id"],
                        "model_id": task["model_id"],
                        "baseline_id": task["baseline_id"],
                        "model_won": False,
                        "baseline_won": False,
                        "draw": True,
                        "game_length": 0,
                        "duration_sec": 0.0,
                        "error": str(result),
                    })
                else:
                    results.append(result)

            # Progress update
            logger.info(f"Gauntlet: Completed {len(results)}/{len(tasks)} games")

        return results

    async def _execute_single_gauntlet_game(
        self,
        task: dict[str, Any],
        board_type: str,
        num_players: int,
        model_dir: Path,
    ) -> dict[str, Any]:
        """Execute a single gauntlet game.

        Args:
            task: Task dict with model_id, baseline_id, etc.
            board_type: Board type (square8, etc.)
            num_players: Number of players
            model_dir: Path to model files

        Returns:
            Result dict
        """
        start_time = time.time()
        task_id = task["task_id"]
        model_id = task["model_id"]
        baseline_id = task["baseline_id"]

        try:
            # Run game in thread pool to avoid blocking
            # Dec 2025: Use get_running_loop() in async context
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._run_gauntlet_game_sync,
                task_id, model_id, baseline_id,
                board_type, num_players, model_dir,
            )
            result["duration_sec"] = time.time() - start_time
            return result

        except Exception as e:
            return {
                "task_id": task_id,
                "model_id": model_id,
                "baseline_id": baseline_id,
                "model_won": False,
                "baseline_won": False,
                "draw": True,
                "game_length": 0,
                "duration_sec": time.time() - start_time,
                "error": str(e),
            }

    def _run_gauntlet_game_sync(
        self,
        task_id: str,
        model_id: str,
        baseline_id: str,
        board_type: str,
        num_players: int,
        model_dir: Path,
    ) -> dict[str, Any]:
        """Synchronously run a single gauntlet game.

        This runs in a thread pool executor. Uses GameExecutor for consistent
        game execution across all gauntlet modes.
        """
        try:
            from app.execution.game_executor import GameExecutor

            # Map model IDs to player configs
            player_configs = []

            # Model agent (player 0)
            # Use mcts_25 for fast gauntlet evaluation (25 simulations = ~0.15s/move)
            if model_id == "random_ai":
                player_configs.append({"ai_type": "random", "difficulty": 1})
            else:
                model_path = model_dir / f"{model_id}.pth"
                if model_path.exists():
                    # Use MCTS with neural guidance - 25 sims for speed
                    player_configs.append({
                        "ai_type": "mcts_25",
                        "difficulty": 5,
                        "nn_model_id": model_id,
                    })
                else:
                    # Model file not found, use MCTS fallback
                    player_configs.append({"ai_type": "mcts_25", "difficulty": 4})

            # Baseline agent (player 1)
            if baseline_id == "random_ai":
                player_configs.append({"ai_type": "random", "difficulty": 1})
            else:
                baseline_path = model_dir / f"{baseline_id}.pth"
                if baseline_path.exists():
                    player_configs.append({
                        "ai_type": "mcts_25",
                        "difficulty": 5,
                        "nn_model_id": baseline_id,
                    })
                else:
                    player_configs.append({"ai_type": "mcts_25", "difficulty": 4})

            # Add random players for 3p/4p games
            while len(player_configs) < num_players:
                player_configs.append({"ai_type": "random", "difficulty": 1})

            # Run game using GameExecutor
            max_moves = 2000 if "19" in board_type else 500
            executor = GameExecutor(board_type=board_type, num_players=num_players)
            result = executor.run_game(
                player_configs=player_configs,
                max_moves=max_moves,
            )

            game_length = result.move_count

            # Convert executor result to gauntlet result format
            # GameExecutor uses 1-indexed winner (1 = player 1 = model)
            if result.winner is None or result.outcome.value == "draw":
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "baseline_id": baseline_id,
                    "model_won": False,
                    "baseline_won": False,
                    "draw": True,
                    "game_length": game_length,
                    "duration_sec": 0.0,
                }
            elif result.winner == 1:  # Player 1 (model) won
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "baseline_id": baseline_id,
                    "model_won": True,
                    "baseline_won": False,
                    "draw": False,
                    "game_length": game_length,
                    "duration_sec": 0.0,
                }
            else:  # Player 2+ (baseline or other) won
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "baseline_id": baseline_id,
                    "model_won": False,
                    "baseline_won": True,
                    "draw": False,
                    "game_length": game_length,
                    "duration_sec": 0.0,
                }

        except Exception as e:
            return {
                "task_id": task_id,
                "model_id": model_id,
                "baseline_id": baseline_id,
                "model_won": False,
                "baseline_won": False,
                "draw": True,
                "game_length": 0,
                "duration_sec": 0.0,
                "error": str(e),
            }

    async def handle_gauntlet_status(self, request: web.Request) -> web.Response:
        """GET /gauntlet/status - Get current gauntlet execution status.

        Returns information about this node's gauntlet capabilities.
        """
        return web.json_response({
            "node_id": self.node_id,
            "available": True,
            "has_gpu": self.self_info.has_gpu if hasattr(self.self_info, "has_gpu") else False,
            "gpu_name": self.self_info.gpu_name if hasattr(self.self_info, "gpu_name") else "",
            "cpu_count": self.self_info.cpu_count if hasattr(self.self_info, "cpu_count") else 0,
        })

    async def handle_gauntlet_quick_eval(self, request: web.Request) -> web.Response:
        """POST /gauntlet/quick-eval - Run quick gauntlet evaluation.

        This endpoint is called by GPU nodes to offload gauntlet work to
        CPU-rich nodes (like Vast instances). Returns win rate and pass status.
        """
        try:
            data = await request.json()
            config_key = data.get("config_key")
            model_id = data.get("model_id")
            baseline_id = data.get("baseline_id")
            games_per_side = data.get("games_per_side", 4)

            if not all([config_key, model_id, baseline_id]):
                return web.json_response(
                    {"error": "Missing required fields"},
                    status=400
                )

            # Parse config
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2:
                return web.json_response({"error": "Invalid config_key"}, status=400)
            board_type = parts[0]
            num_players = int(parts[1].rstrip("p"))

            model_dir = Path(self.ringrift_path) / "ai-service" / "models"

            # Run games: model vs baseline from both sides
            wins = 0
            total_games = 0
            # Dec 2025: Use get_running_loop() in async context
            loop = asyncio.get_running_loop()

            for game_num in range(games_per_side * 2):
                try:
                    if game_num < games_per_side:
                        # Model plays first
                        result = await loop.run_in_executor(
                            None,
                            self._run_gauntlet_game_sync,
                            f"quick_eval_{game_num}", model_id, baseline_id,
                            board_type, num_players, model_dir
                        )
                        if result.get("model_won"):
                            wins += 1
                    else:
                        # Baseline plays first
                        result = await loop.run_in_executor(
                            None,
                            self._run_gauntlet_game_sync,
                            f"quick_eval_{game_num}", baseline_id, model_id,
                            board_type, num_players, model_dir
                        )
                        if result.get("baseline_won"):
                            wins += 1
                    total_games += 1
                except Exception as e:
                    logger.info(f"Quick eval game {game_num} error: {e}")
                    total_games += 1

            win_rate = wins / total_games if total_games > 0 else 0
            passed = win_rate >= 0.50

            # Emit gauntlet completion event to coordination EventRouter
            if HAS_EVENT_BRIDGE:
                await emit_p2p_gauntlet_completed(
                    model_id=model_id,
                    baseline_id=baseline_id,
                    config_key=config_key,
                    wins=wins,
                    total_games=total_games,
                    win_rate=win_rate,
                    passed=passed,
                    node_id=self.node_id,
                )

            return web.json_response({
                "success": True,
                "node_id": self.node_id,
                "model_id": model_id,
                "baseline_id": baseline_id,
                "wins": wins,
                "total_games": total_games,
                "win_rate": win_rate,
                "passed": passed,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
