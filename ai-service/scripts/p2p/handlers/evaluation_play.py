"""Evaluation Play HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 5a

This mixin provides HTTP handlers for playing Elo calibration matches
between AI configurations. Supports 2-4 player games with various AI types.

Must be mixed into a class that provides:
- self.node_id: str
- self._tournament_match_semaphore: asyncio.Semaphore | None
- self._current_match_holder: str | None
- self._get_ai_service_path() -> str
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EvaluationPlayHandlersMixin:
    """Mixin providing Elo match play HTTP handlers.

    Endpoints:
    - POST /play/elo-match - Play a single Elo calibration match between AI configurations
    """

    # Required attributes (provided by orchestrator)
    node_id: str
    _tournament_match_semaphore: asyncio.Semaphore | None
    _current_match_holder: str | None

    async def handle_play_elo_match(self, request: web.Request) -> web.Response:
        """Play a single Elo calibration match between AI configurations.

        This endpoint supports playing games between different AI types
        (random, heuristic, minimax, mcts, policy_only, gumbel_mcts, descent)
        for Elo calibration purposes. Supports 2-4 player games.

        Request body:
            match_id: Unique match identifier
            agent_a: Agent A identifier (e.g., "random", "mcts_neural")
            agent_b: Agent B identifier
            agent_a_config: Full AI configuration for agent A
            agent_b_config: Full AI configuration for agent B
            agents: List of agent identifiers for multiplayer (optional)
            agent_configs: List of AI configs for multiplayer (optional)
            board_type: Board type (default: square8)
            num_players: Number of players (default: 2)

        Returns:
            success: True if match completed
            winner: "agent_a", "agent_b", "agent_c", "agent_d", or "draw"
            game_length: Number of moves
            duration_sec: Game duration in seconds
        """
        try:
            logger.info("Tournament endpoint called, parsing request...")
            data = await request.json()
            logger.info(f"Request parsed: {data}")

            match_id = data.get("match_id", str(uuid.uuid4())[:8])
            board_type_str = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            # Support both legacy 2-player (agent_a/agent_b) and multiplayer (agents list)
            agents_list = data.get("agents")
            agent_configs_list = data.get("agent_configs")

            if agents_list and len(agents_list) >= 2:
                # Multiplayer mode: use agents list
                agents = agents_list[:num_players]
                if agent_configs_list and len(agent_configs_list) >= num_players:
                    agent_configs = agent_configs_list[:num_players]
                else:
                    # Build configs from agent names
                    agent_configs = [{"ai_type": a} for a in agents]
            else:
                # Legacy 2-player mode
                agent_a = data.get("agent_a", "random")
                agent_b = data.get("agent_b", "heuristic")
                agent_a_config = data.get("agent_a_config", {"ai_type": agent_a})
                agent_b_config = data.get("agent_b_config", {"ai_type": agent_b})
                agents = [agent_a, agent_b]
                agent_configs = [agent_a_config, agent_b_config]

            # Pad with random if fewer agents than players
            while len(agents) < num_players:
                agents.append("random")
                agent_configs.append({"ai_type": "random"})

            agents_desc = " vs ".join(agents)
            logger.info(f"Playing Elo match {match_id}: {agents_desc}")
            start_time = time.time()

            # Acquire semaphore to prevent concurrent matches (OOM protection)
            # Create lazily in async context to avoid event loop issues
            if self._tournament_match_semaphore is None:
                logger.info("Creating tournament semaphore...")
                self._tournament_match_semaphore = asyncio.Semaphore(1)

            # Try to acquire semaphore with timeout to avoid deadlocks
            # If we can't get the semaphore within 30 seconds, fail fast
            logger.info(f"Acquiring semaphore (current holder: {getattr(self, '_current_match_holder', 'none')})...")
            try:
                await asyncio.wait_for(
                    self._tournament_match_semaphore.acquire(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.info(f"Semaphore acquisition timed out after 30s (holder: {getattr(self, '_current_match_holder', 'unknown')})")
                return web.json_response({
                    "success": False,
                    "error": f"Server busy - another match in progress (holder: {getattr(self, '_current_match_holder', 'unknown')})",
                    "match_id": match_id,
                }, status=503)

            # Track who holds the semaphore for debugging
            self._current_match_holder = f"{match_id} ({agents_desc})"
            try:
                logger.info(f"Semaphore acquired, running match {match_id}...")
                # Run the match in a thread pool to avoid blocking
                # Add 5-minute timeout to prevent hung matches
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._play_elo_match_sync,
                            agent_configs,
                            board_type_str,
                            num_players,
                            match_id,
                            agents,
                        ),
                        timeout=300.0,  # 5 minute timeout for tournament matches
                    )
                except asyncio.TimeoutError:
                    logger.info(f"Elo match {match_id} timed out after 5 minutes")
                    return web.json_response({
                        "success": False,
                        "error": "Match timed out after 5 minutes",
                        "match_id": match_id,
                    }, status=504)
            finally:
                # Always release semaphore and clear holder
                self._current_match_holder = None
                self._tournament_match_semaphore.release()
                logger.info(f"Semaphore released for match {match_id}")

            duration = time.time() - start_time

            if result is None:
                return web.json_response({
                    "success": False,
                    "error": "Match failed to complete",
                    "match_id": match_id,
                }, status=500)

            # Map winner player number to agent label
            winner_player = result.get("winner_player")
            if winner_player is not None and winner_player > 0:
                agent_labels = ["agent_a", "agent_b", "agent_c", "agent_d"]
                winner = agent_labels[winner_player - 1] if winner_player <= len(agent_labels) else "draw"
            else:
                # Legacy format support
                winner_map = {"model_a": "agent_a", "model_b": "agent_b", "draw": "draw"}
                winner = winner_map.get(result.get("winner", "draw"), "draw")

            response = {
                "success": True,
                "match_id": match_id,
                "agents": agents,
                "agent_a": agents[0] if len(agents) > 0 else "unknown",
                "agent_b": agents[1] if len(agents) > 1 else "unknown",
                "winner": winner,
                "winner_player": winner_player,
                "game_length": result.get("game_length", 0),
                "duration_sec": duration,
                "worker_node": self.node_id,
            }

            logger.info(f"Elo match {match_id} complete: {agents_desc} -> {response['winner']} ({result.get('game_length', 0)} moves)")
            return web.json_response(response)

        except Exception as e:  # noqa: BLE001
            import traceback
            logger.info(f"Elo match error: {e}")
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    def _play_elo_match_sync(
        self,
        agent_configs: list[dict],
        board_type_str: str,
        num_players: int,
        match_id: str,
        agent_ids: list[str] | None = None,
    ) -> dict | None:
        """Synchronous wrapper for playing an Elo match.

        Uses a lightweight implementation for simple AI types (random, heuristic, minimax)
        to avoid loading heavy neural network dependencies that cause OOM.

        Args:
            agent_configs: List of AI configurations for each player
            board_type_str: Board type string
            num_players: Number of players in the game
        """
        try:
            import time as time_mod

            from app.db.unified_recording import (
                RecordingConfig,
                RecordSource,
                UnifiedGameRecorder,
                is_recording_enabled,
            )
            from app.game_engine import GameEngine
            from app.models import AIConfig, AIType, BoardType, GameStatus
            from app.training.initial_state import create_initial_state

            board_type = BoardType(board_type_str)
            start_time = time_mod.time()

            # Generate unique random seeds for each player
            import random as rand_mod
            match_seed = int(time_mod.time() * 1000000) % (2**31)
            rand_mod.seed(match_seed)
            seeds = [rand_mod.randint(0, 2**31 - 1) for _ in range(num_players)]

            # Create initial state
            state = create_initial_state(board_type, num_players)
            engine = GameEngine()

            # Map agent names to AI types
            def get_ai_type(agent_config: dict) -> str:
                ai_type = agent_config.get("ai_type", "random")
                if isinstance(ai_type, str):
                    return ai_type.lower()
                return str(ai_type).lower()

            def create_lightweight_ai(agent_config: dict, player_num: int, rng_seed: int):
                """Create AI without loading heavy dependencies."""
                ai_type = get_ai_type(agent_config)

                if ai_type in ("random", "aitype.random"):
                    from app.ai.random_ai import RandomAI
                    config = AIConfig(ai_type=AIType.RANDOM, board_type=board_type, difficulty=1, rng_seed=rng_seed)
                    return RandomAI(player_num, config)

                elif ai_type in ("heuristic", "aitype.heuristic"):
                    from app.ai.heuristic_ai import HeuristicAI
                    config = AIConfig(ai_type=AIType.HEURISTIC, board_type=board_type, difficulty=3, rng_seed=rng_seed)
                    return HeuristicAI(player_num, config)

                elif ai_type in ("minimax", "minimax_heuristic", "aitype.minimax"):
                    from app.ai.minimax_ai import MinimaxAI
                    use_nn = agent_config.get("use_neural_net", False)
                    max_depth = agent_config.get("max_depth", 3)
                    config = AIConfig(
                        ai_type=AIType.MINIMAX,
                        board_type=board_type,
                        difficulty=agent_config.get("difficulty", 3),
                        use_neural_net=use_nn,
                        max_depth=max_depth,
                        rng_seed=rng_seed,
                    )
                    return MinimaxAI(player_num, config)

                elif ai_type in ("mcts", "mcts_heuristic", "aitype.mcts"):
                    from app.ai.mcts_ai import MCTSAI
                    use_nn = agent_config.get("use_neural_net", False)
                    iters = agent_config.get("mcts_iterations", 100)
                    config = AIConfig(
                        ai_type=AIType.MCTS,
                        board_type=board_type,
                        difficulty=agent_config.get("difficulty", 5),
                        use_neural_net=use_nn,
                        mcts_iterations=iters,
                        rng_seed=rng_seed,
                    )
                    return MCTSAI(player_num, config)

                elif ai_type in ("descent", "aitype.descent"):
                    # Descent AI is CPU-based but can be slow at high difficulty
                    # Cap at difficulty 5 for tournament matches (~1.1s/move)
                    from app.ai.descent_ai import DescentAI
                    requested_diff = agent_config.get("difficulty", 5)
                    capped_diff = min(requested_diff, 5)  # Cap at 5 for tournaments
                    if capped_diff < requested_diff:
                        logger.info(f"Descent AI difficulty capped from {requested_diff} to {capped_diff} for tournament")
                    config = AIConfig(
                        ai_type=AIType.DESCENT,
                        board_type=board_type,
                        difficulty=capped_diff,
                        rng_seed=rng_seed,
                    )
                    return DescentAI(player_num, config)

                else:
                    # For neural-net based types (policy_only, gumbel_mcts, mcts_neural),
                    # check available memory before loading
                    import psutil
                    mem = psutil.virtual_memory()
                    available_gb = mem.available / (1024**3)

                    # Require at least 8GB free for neural network loading (conservative to prevent OOM)
                    if available_gb < 8.0:
                        logger.info(f"Skipping NN-based AI {ai_type}: only {available_gb:.1f}GB available (need 8GB)")
                        # Fall back to descent AI (CPU-based, no NN)
                        from app.ai.descent_ai import DescentAI
                        config = AIConfig(ai_type=AIType.DESCENT, board_type=board_type, difficulty=7, rng_seed=rng_seed)
                        return DescentAI(player_num, config)

                    # Safe to load neural network AI
                    try:
                        from scripts.run_model_elo_tournament import create_ai_from_model
                        return create_ai_from_model(agent_config, player_num, board_type)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Failed to create NN AI {ai_type}: {e}, falling back to heuristic")
                        from app.ai.heuristic_ai import HeuristicAI
                        config = AIConfig(ai_type=AIType.HEURISTIC, board_type=board_type, difficulty=7, rng_seed=rng_seed)
                        return HeuristicAI(player_num, config)

            # Create AIs for all players with unique seeds
            ais = {}
            for i in range(num_players):
                player_num = i + 1
                config = agent_configs[i] if i < len(agent_configs) else {"ai_type": "random"}
                seed = seeds[i] if i < len(seeds) else 0
                ais[player_num] = create_lightweight_ai(config, player_num, seed)

            # Keep initial state for training record
            initial_state = state

            if not agent_ids:
                agent_ids = [
                    str(cfg.get("agent_id") or cfg.get("ai_type") or f"player_{idx + 1}")
                    for idx, cfg in enumerate(agent_configs[:num_players])
                ]
            while len(agent_ids) < num_players:
                agent_ids.append(f"player_{len(agent_ids) + 1}")

            # Play game and record actual Move objects for training
            move_count = 0
            max_moves = 500
            recorded_moves = []  # List of actual Move objects for training
            termination_reason = "completed"
            recording_enabled = is_recording_enabled()
            tags = [
                "elo_tournament",
                f"node_{self.node_id}",
                f"board_{board_type.value}",
                f"players_{num_players}",
            ]
            for idx, cfg in enumerate(agent_configs[:num_players]):
                ai_type = cfg.get("ai_type", "unknown")
                tags.append(f"p{idx + 1}_{ai_type}")

            recording_config = RecordingConfig(
                board_type=board_type.value,
                num_players=num_players,
                source=RecordSource.TOURNAMENT,
                engine_mode="p2p_elo",
                db_prefix="tournament",
                db_dir="data/games",
                store_history_entries=True,
                fsm_validation=True,
                tags=tags,
            )
            recorder = UnifiedGameRecorder(recording_config, state, game_id=match_id) if recording_enabled else None

            try:
                if recorder is not None:
                    recorder.__enter__()

                while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
                    current_player = state.current_player

                    requirement = GameEngine.get_phase_requirement(state, current_player)
                    if requirement is not None:
                        move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                    else:
                        current_ai = ais.get(current_player)
                        if current_ai is None:
                            logger.warning(f"No AI for player {current_player}, using random fallback")
                            from app.ai.random_ai import RandomAI
                            config = AIConfig(ai_type=AIType.RANDOM, board_type=board_type, difficulty=1)
                            current_ai = RandomAI(current_player, config)
                            ais[current_player] = current_ai
                        move = current_ai.select_move(state)

                    if move is None:
                        termination_reason = "no_move"
                        break

                    # Get soft policy targets for training data
                    move_probs = None
                    if hasattr(current_ai, 'get_visit_distribution'):
                        try:
                            moves_dist, probs_dist = current_ai.get_visit_distribution()
                            if moves_dist and probs_dist:
                                move_probs = {}
                                for mv, prob in zip(moves_dist, probs_dist, strict=False):
                                    # Create move key in format: "{from_x},{from_y}->{to_x},{to_y}"
                                    if hasattr(mv, 'to') and mv.to is not None:
                                        move_key = f"{mv.to.x},{mv.to.y}"
                                        if hasattr(mv, 'from_pos') and mv.from_pos is not None:
                                            move_key = f"{mv.from_pos.x},{mv.from_pos.y}->{move_key}"
                                        move_probs[move_key] = float(prob)
                        except (ValueError, KeyError, IndexError, AttributeError):
                            pass  # Silently ignore if visit distribution fails

                    # Record actual Move object for training
                    recorded_moves.append(move)

                    state_before = state
                    state = engine.apply_move(state, move, trace_mode=True)
                    move_count += 1
                    if recorder is not None:
                        recorder.add_move(
                            move,
                            state_after=state,
                            state_before=state_before,
                            available_moves_count=None,
                            move_probs=move_probs,
                        )
            finally:
                if move_count >= max_moves and state.game_status == GameStatus.ACTIVE:
                    termination_reason = "max_moves"

                if recorder is not None:
                    winner_player = state.winner if state.winner else 0
                    winner_agent = None
                    if isinstance(winner_player, int) and winner_player > 0 and winner_player <= len(agent_ids):
                        winner_agent = agent_ids[winner_player - 1]

                    extra_metadata = {
                        "match_id": match_id,
                        "tournament_id": f"p2p_elo_{match_id}",
                        "node_id": self.node_id,
                        "match_seed": match_seed,
                        "agent_ids": agent_ids,
                        "agent_configs": agent_configs,
                        "winner_player": winner_player,
                        "winner_agent": winner_agent,
                        "game_length": move_count,
                        "duration_sec": time_mod.time() - start_time,
                        "termination_reason": termination_reason,
                    }
                    try:
                        recorder.finalize(state, extra_metadata=extra_metadata)
                    finally:
                        recorder.__exit__(None, None, None)

            duration = time_mod.time() - start_time

            # Determine winner (as player number)
            winner_player = state.winner if state.winner else 0

            # Legacy format for 2-player backward compatibility
            winner = "draw"
            if winner_player == 1:
                winner = "model_a"
            elif winner_player == 2:
                winner = "model_b"

            # Save game for training using proper GameRecord format
            if recorded_moves and winner_player > 0:
                try:
                    self._save_tournament_game_for_training(
                        initial_state=initial_state,
                        final_state=state,
                        moves=recorded_moves,
                        match_seed=match_seed,
                        agent_configs=agent_configs,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to save tournament game for training: {e}")

            return {
                "winner": winner,
                "winner_player": winner_player,
                "game_length": move_count,
                "duration_sec": duration,
            }

        except Exception as e:  # noqa: BLE001
            import traceback
            logger.info(f"_play_elo_match_sync error: {e}")
            traceback.print_exc()
            return None

    def _save_tournament_game_for_training(
        self,
        initial_state,  # GameState
        final_state,  # GameState
        moves: list,  # List of Move objects
        match_seed: int,
        agent_configs: list[dict] | None = None,  # Optional AI configs for metadata
    ) -> None:
        """Save a tournament game to JSONL format for training.

        Uses build_training_game_record to create proper GameRecord format
        compatible with the training pipeline. The saved games can be ingested
        by the training system alongside selfplay games.

        Saves games to data/tournament_games/{board_type}_{num_players}p/ directory.

        Args:
            initial_state: The initial game state
            final_state: The final game state after all moves
            moves: List of Move objects representing the game
            match_seed: RNG seed used for this match
            agent_configs: Optional list of AI configs for each player
        """
        import json
        import sys
        from datetime import datetime, timezone
        from pathlib import Path

        # Ensure app module is importable
        ai_service_path = str(Path(self._get_ai_service_path()))
        if ai_service_path not in sys.path:
            sys.path.insert(0, ai_service_path)

        try:
            from app.models.game_record import RecordSource
            from app.training.game_record_export import build_training_game_record
        except ImportError as e:
            logger.warning(f"Cannot import game record modules: {e}")
            return

        board_type_str = initial_state.board_type.value if hasattr(initial_state.board_type, 'value') else str(initial_state.board_type)
        num_players = len(initial_state.players)

        # Create output directory
        data_dir = Path(self._get_ai_service_path()) / "data" / "tournament_games"
        config_dir = data_dir / f"{board_type_str}_{num_players}p"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Create game ID with full metadata
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        game_id = f"tournament_{self.node_id}_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Build tags with detailed metadata for training filtering
        tags = [
            "elo_tournament",
            f"node_{self.node_id}",
            f"board_{board_type_str}",
            f"players_{num_players}",
        ]
        if agent_configs:
            for i, cfg in enumerate(agent_configs[:num_players]):
                ai_type = cfg.get("ai_type", "unknown")
                tags.append(f"player{i+1}_{ai_type}")

        # Build proper GameRecord using the training export function
        try:
            game_record = build_training_game_record(
                game_id=game_id,
                initial_state=initial_state,
                final_state=final_state,
                moves=moves,
                source=RecordSource.TOURNAMENT,
                rng_seed=match_seed,
                terminated_by_budget_only=False,
                created_at=datetime.now(timezone.utc),
                tags=tags,
                fsm_validated=None,  # Not FSM validated in tournament context
            )

            # Use the canonical to_jsonl_line() method for proper serialization
            jsonl_line = game_record.to_jsonl_line()

            # Append to daily file for this config
            daily_file = config_dir / f"tournament_{timestamp[:8]}.jsonl"
            with open(daily_file, "a", encoding="utf-8") as f:
                f.write(jsonl_line + "\n")

            logger.info(f"Saved tournament game {game_id} to {daily_file} ({len(moves)} moves, winner={final_state.winner})")

        except Exception as e:  # noqa: BLE001
            import traceback
            logger.warning(f"Failed to build/save tournament game record: {e}")
            traceback.print_exc()
