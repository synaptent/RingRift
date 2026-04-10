"""Large tournament execution helpers for TournamentDaemon."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_utils import make_config_key, parse_config_key
from app.training.composite_participant import extract_harness_type

logger = logging.getLogger(__name__)


class TournamentExecutionMixin:
    """Extracted helpers for TournamentDaemon."""

    async def _run_cross_nn_tournament(self) -> dict[str, Any]:
        """Run cross-NN version tournament to compare model generations (Dec 2025).

        Discovers all model versions for each configuration and runs tournaments
        between adjacent versions (e.g., v2 vs v3, v3 vs v4) to:
        - Validate newer models are stronger than older ones
        - Maintain accurate Elo ratings across model generations
        - Identify potential regressions in model quality

        Returns:
            Tournament results with per-pairing win rates and Elo updates
        """
        logger.info("Running cross-NN version tournament")
        start_time = time.time()

        results = {
            "tournament_id": str(uuid.uuid4()),
            "tournament_type": "cross_nn",
            "success": False,
            "pairings": {},
            "games_played": 0,
        }

        try:
            from app.training.game_gauntlet import play_single_game
            from app.models import BoardType
            from app.ai.neural_net import UnifiedNeuralNetFactory
            from app.training.elo_service import get_elo_service
            from pathlib import Path
            import re

            elo_service = get_elo_service()
            models_dir = Path("models")

            # Find all canonical models per config
            # Pattern: canonical_{board}_{n}p.pth or canonical_{board}_{n}p_v{version}.pth
            model_pattern = re.compile(
                r"canonical_(?P<board>\w+)_(?P<players>\d)p(?:_v(?P<version>\d+))?\.pth"
            )

            # Group models by config (board_type, num_players)
            config_models: dict[tuple[str, int], list[tuple[str, Path]]] = {}

            for model_path in models_dir.glob("canonical_*.pth"):
                match = model_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    version = match.group("version") or "base"
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    config_models[config_key].append((version, model_path))

            # Also check for versioned models like hex8_2p_v2.pth, hex8_2p_v3.pth
            version_pattern = re.compile(
                r"(?:canonical_)?(?P<board>\w+)_(?P<players>\d)p_v(?P<version>\d+)\.pth"
            )

            for model_path in models_dir.glob("*_v*.pth"):
                if "canonical" in model_path.name:
                    continue  # Already captured above
                match = version_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    version = f"v{match.group('version')}"
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    config_models[config_key].append((version, model_path))

            # Dec 31, 2025: Also discover named architecture variants (v5heavy, v5-heavy-large)
            # Pattern: canonical_{board}_{n}p_{variant}.pth
            variant_pattern = re.compile(
                r"canonical_(?P<board>\w+)_(?P<players>\d)p_(?P<variant>v5heavy|v5-heavy|v5-heavy-large|v4|nnue)\.pth"
            )

            for model_path in models_dir.glob("canonical_*_*.pth"):
                match = variant_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    variant = match.group("variant")
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    # Use variant name as version identifier
                    config_models[config_key].append((variant, model_path))

            games_per_pairing = self.config.cross_nn_games_per_pairing
            total_games = 0

            for (board, num_players), models in config_models.items():
                if len(models) < 2:
                    continue  # Need at least 2 versions to compare

                # Sort by version (base < v2 < v3 < ... < v4 < v5heavy < nnue)
                # Dec 31, 2025: Extended to handle named architecture variants
                def version_key(item: tuple[str, Path]) -> tuple[int, str]:
                    v = item[0]
                    # Known architectures in order of complexity/recency
                    version_order = {
                        "base": (0, ""),
                        "v2": (2, ""),
                        "v3": (3, ""),
                        "v4": (4, ""),
                        "v5heavy": (5, "heavy"),
                        "v5-heavy": (5, "heavy"),
                        "v5-heavy-large": (5, "heavy-large"),
                        "nnue": (6, ""),  # NNUE is evaluated separately
                    }
                    if v in version_order:
                        return version_order[v]
                    # Handle numeric versions like "v5", "v10"
                    if v.startswith("v") and v[1:].isdigit():
                        return (int(v[1:]), "")
                    return (100, v)  # Unknown versions sort last

                models.sort(key=version_key)

                # Get board type enum
                try:
                    board_type = BoardType(board)
                except ValueError:
                    logger.warning(f"Unknown board type: {board}")
                    continue

                # Create recording config if enabled (Dec 2025 - tournament games for training)
                recording_config = None
                if self.config.enable_game_recording:
                    try:
                        from app.db.unified_recording import RecordingConfig, RecordSource
                        recording_config = RecordingConfig(
                            board_type=board_type.value,
                            num_players=num_players,
                            source=RecordSource.TOURNAMENT,
                            engine_mode="cross_nn",
                            db_prefix=self.config.recording_db_prefix,
                            db_dir=self.config.recording_db_dir,
                            store_history_entries=True,
                        )
                    except ImportError:
                        logger.debug("Recording module not available for cross-NN tournament")

                # Run tournaments between adjacent versions
                for i in range(len(models) - 1):
                    older_version, older_path = models[i]
                    newer_version, newer_path = models[i + 1]

                    pairing_key = f"{board}_{num_players}p:{older_version}_vs_{newer_version}"
                    logger.info(f"Cross-NN pairing: {pairing_key}")

                    # Load models
                    try:
                        older_ai = UnifiedNeuralNetFactory.create(
                            str(older_path),
                            board_type=board_type,
                            num_players=num_players,
                        )
                        newer_ai = UnifiedNeuralNetFactory.create(
                            str(newer_path),
                            board_type=board_type,
                            num_players=num_players,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load models for {pairing_key}: {e}")
                        results["pairings"][pairing_key] = {"error": str(e)}
                        continue

                    wins_newer = 0
                    wins_older = 0

                    for game_num in range(games_per_pairing):
                        try:
                            # Alternate positions for fairness
                            if game_num % 2 == 0:
                                player_ais = [newer_ai, older_ai]
                                newer_player = 0
                            else:
                                player_ais = [older_ai, newer_ai]
                                newer_player = 1

                            game_result = play_single_game(
                                board_type=board_type,
                                num_players=num_players,
                                player_ais=player_ais,
                                timeout=self.config.game_timeout_seconds,
                                recording_config=recording_config,
                            )

                            winner = game_result.get("winner")
                            if winner == newer_player:
                                wins_newer += 1
                            elif winner is not None:
                                wins_older += 1

                            total_games += 1
                            self._tournament_stats.games_played += 1

                            # Record match for Elo update
                            if winner is not None:
                                # Jan 2026: Fixed incorrect parameter names (was winner_id/loser_id)
                                winner_model_id = newer_path.stem if winner == newer_player else older_path.stem
                                loser_model_id = older_path.stem if winner == newer_player else newer_path.stem
                                # January 2026: Extract harness_type for per-harness Elo tracking
                                # Default to gumbel_mcts for legacy model names without composite ID
                                harness_type = extract_harness_type(winner_model_id) or "gumbel_mcts"
                                elo_service.record_match(
                                    participant_a=winner_model_id,
                                    participant_b=loser_model_id,
                                    winner=winner_model_id,
                                    board_type=board,
                                    num_players=num_players,
                                    harness_type=harness_type,
                                )

                        except Exception as e:
                            logger.warning(f"Cross-NN game failed: {e}")

                    win_rate_newer = wins_newer / games_per_pairing if games_per_pairing > 0 else 0
                    # Newer model should win >50% if it's actually better
                    improvement_validated = win_rate_newer >= 0.5

                    results["pairings"][pairing_key] = {
                        "newer_wins": wins_newer,
                        "older_wins": wins_older,
                        "draws": games_per_pairing - wins_newer - wins_older,
                        "games": games_per_pairing,
                        "newer_win_rate": win_rate_newer,
                        "improvement_validated": improvement_validated,
                    }

                    if not improvement_validated:
                        logger.warning(
                            f"Potential regression: {newer_version} only {win_rate_newer:.1%} "
                            f"vs {older_version} in {board}_{num_players}p"
                        )

            results["success"] = True
            results["games_played"] = total_games
            results["configs_tested"] = len([k for k, v in config_models.items() if len(v) >= 2])

        except ImportError as e:
            logger.warning(f"Cross-NN tournament dependencies not available: {e}")
            results["error"] = str(e)
        except Exception as e:
            logger.error(f"Cross-NN tournament failed: {e}")
            results["error"] = str(e)
            self._tournament_stats.errors.append(str(e))

        results["duration_seconds"] = time.time() - start_time
        logger.info(f"Cross-NN tournament completed: {results.get('games_played', 0)} games")
        return results

    async def _run_cross_config_tournament(self) -> dict[str, Any]:
        """Run cross-config tournament within board families.

        Compares models trained on different player counts (2p vs 3p vs 4p)
        within the same board type to validate curriculum progression.

        Returns:
            Results dict with families evaluated and games played
        """
        results = {
            "success": False,
            "families_evaluated": 0,
            "total_games": 0,
            "family_results": {},
        }

        try:
            from app.models.discovery import find_tournament_models
            from app.training.game_gauntlet import play_single_game
            from app.training.elo_service import get_elo_service
            from app.models import BoardType

            models = find_tournament_models()
            elo_service = get_elo_service()
            games_per_matchup = self.config.cross_config_games_per_matchup

            for family in self.config.cross_config_families:
                family_models = []

                # Collect models for this family
                for config_key in family:
                    parsed = parse_config_key(config_key)
                    if not parsed:
                        continue

                    model_key = (parsed.board_type, parsed.num_players)
                    if model_key in models:
                        family_models.append({
                            "config_key": config_key,
                            "board_type": parsed.board_type,
                            "num_players": parsed.num_players,
                            "model_path": models[model_key],
                        })

                if len(family_models) < 2:
                    continue  # Need at least 2 models to compare

                family_key = "_".join(m["config_key"] for m in family_models)
                family_result = {
                    "matchups": {},
                    "games_played": 0,
                }

                # Run round-robin within family (comparing different player counts)
                # Note: This is primarily for transfer learning validation
                # We compare how 2p model performs when adapted to 4p scenarios, etc.
                for i, model_a in enumerate(family_models):
                    for model_b in family_models[i + 1:]:
                        matchup_key = f"{model_a['config_key']}_vs_{model_b['config_key']}"

                        # Use the larger player count for the matchup
                        num_players = max(model_a["num_players"], model_b["num_players"])
                        board_type = model_a["board_type"]

                        wins_a = 0
                        for game_num in range(games_per_matchup):
                            try:
                                # Alternate starting positions
                                if game_num % 2 == 0:
                                    player_models = [model_a["model_path"], model_b["model_path"]]
                                    model_a_player = 0
                                else:
                                    player_models = [model_b["model_path"], model_a["model_path"]]
                                    model_a_player = 1

                                # Create player AIs for additional players if needed
                                from app.training.game_gauntlet import (
                                    create_neural_ai,
                                    BaselineOpponent,
                                    create_baseline_ai,
                                )

                                player_ais = []
                                for p in range(num_players):
                                    if p < 2:
                                        # Use the competing models
                                        player_ais.append(
                                            create_neural_ai(
                                                str(player_models[p]),
                                                p + 1,
                                                BoardType(board_type),
                                                num_players,
                                            )
                                        )
                                    else:
                                        # Fill with heuristic for 3p/4p games
                                        player_ais.append(
                                            create_baseline_ai(
                                                BaselineOpponent.HEURISTIC,
                                                p + 1,
                                                BoardType(board_type),
                                                num_players,
                                            )
                                        )

                                game_result = play_single_game(
                                    board_type=BoardType(board_type),
                                    num_players=num_players,
                                    player_ais=player_ais,
                                    timeout=self.config.game_timeout_seconds,
                                )

                                winner = game_result.get("winner")
                                if winner == model_a_player:
                                    wins_a += 1

                                family_result["games_played"] += 1
                                results["total_games"] += 1
                                self._tournament_stats.games_played += 1

                            except Exception as e:
                                logger.warning(f"Cross-config game failed: {e}")

                        win_rate_a = wins_a / games_per_matchup if games_per_matchup > 0 else 0
                        family_result["matchups"][matchup_key] = {
                            "model_a": model_a["config_key"],
                            "model_b": model_b["config_key"],
                            "wins_a": wins_a,
                            "games": games_per_matchup,
                            "win_rate_a": win_rate_a,
                        }

                        # Feb 2026: Do NOT record cross-config matches in the main
                        # Elo database. Recording a 2p model as a participant in 4p
                        # games pollutes per-config Elo tracking and causes massive
                        # Elo regression (e.g., hex8_4p dropped from 1900 to 1508).
                        # Cross-config results are logged above in family_result for
                        # informational purposes only.
                        logger.info(
                            f"Cross-config: {model_a['config_key']} vs {model_b['config_key']} "
                            f"({board_type} {num_players}p) - win_rate_a={win_rate_a:.1%}"
                        )

                results["family_results"][family_key] = family_result
                results["families_evaluated"] += 1

            results["success"] = True

            # Emit event for cross-config results
            try:
                from app.distributed.data_events import DataEventType
                safe_emit_event(
                    DataEventType.CROSS_CONFIG_TOURNAMENT_COMPLETED.value,
                    {
                        "families_evaluated": results["families_evaluated"],
                        "total_games": results["total_games"],
                        "family_results": results["family_results"],
                    },
                    context="TournamentDaemon",
                )
            except ImportError:
                pass  # Event emission not critical

        except ImportError as e:
            logger.warning(f"Cross-config tournament dependencies not available: {e}")
            results["error"] = "import_error"
        except Exception as e:
            logger.error(f"Cross-config tournament failed: {e}")
            results["error"] = str(e)

        return results

    async def _run_topn_roundrobin_tournament(self) -> dict[str, Any]:
        """Run round-robin tournaments between top-rated models.

        For each configured board/player configuration, gets the top N
        models by Elo rating and runs a round-robin tournament where
        each model plays every other model.

        Returns:
            Results dict with configs evaluated, games played, and per-config results
        """
        results = {
            "success": False,
            "configs_evaluated": 0,
            "total_games": 0,
            "config_results": {},
        }

        try:
            from app.training.elo_service import get_elo_service
            from app.models.discovery import find_tournament_models
            from app.training.game_gauntlet import (
                play_single_game,
                create_neural_ai,
            )
            from app.models import BoardType
            from app.training.composite_participant import extract_harness_type

            elo_service = get_elo_service()
            available_models = find_tournament_models()
            games_per_matchup = self.config.topn_roundrobin_games_per_matchup
            top_n = self.config.topn_roundrobin_n
            min_games = self.config.topn_roundrobin_min_elo_games

            # Get configs to evaluate
            configs_to_evaluate = self.config.topn_roundrobin_configs
            if not configs_to_evaluate:
                # Use all available configs
                configs_to_evaluate = [
                    make_config_key(bt, np)
                    for (bt, np) in available_models.keys()
                ]

            for config_key in configs_to_evaluate:
                parsed = parse_config_key(config_key)
                if not parsed:
                    logger.warning(f"Invalid config key: {config_key}")
                    continue

                board_type = parsed.board_type
                num_players = parsed.num_players

                # Get top N models for this config
                try:
                    leaderboard = elo_service.get_leaderboard(
                        board_type=board_type,
                        num_players=num_players,
                        limit=top_n,
                        min_games=min_games,
                    )
                except Exception as e:
                    logger.warning(f"Failed to get leaderboard for {config_key}: {e}")
                    continue

                if len(leaderboard) < 2:
                    logger.debug(f"Not enough rated models for {config_key} round-robin")
                    continue

                config_result = {
                    "models": len(leaderboard),
                    "matchups": {},
                    "games_played": 0,
                }

                # Extract model paths from leaderboard entries
                top_models = []
                for entry in leaderboard:
                    # Mar 29, 2026: LeaderboardEntry is a dataclass, not a dict.
                    # Using .get() crashed with "no attribute 'get'".
                    participant_id = getattr(entry, "participant_id", None) or getattr(entry, "participant", None)
                    if not participant_id:
                        continue
                    # Find model path - try to match with available models
                    model_key = (board_type, num_players)
                    if model_key in available_models:
                        # Use canonical model path
                        model_path = available_models[model_key]
                    else:
                        # Try to construct path from participant_id
                        model_path = Path(f"models/{participant_id}.pth")
                        if not model_path.exists():
                            model_path = Path(f"models/canonical_{config_key}.pth")
                            if not model_path.exists():
                                continue

                    top_models.append({
                        "participant_id": participant_id,
                        "model_path": str(model_path),
                        "elo": getattr(entry, "rating", getattr(entry, "elo", 1500)),
                    })

                if len(top_models) < 2:
                    continue

                # Run round-robin tournament
                for i, model_a in enumerate(top_models):
                    for model_b in top_models[i + 1:]:
                        matchup_key = f"{model_a['participant_id']}_vs_{model_b['participant_id']}"
                        wins_a = 0

                        for game_num in range(games_per_matchup):
                            try:
                                # Alternate starting positions
                                if game_num % 2 == 0:
                                    path_a, path_b = model_a["model_path"], model_b["model_path"]
                                    model_a_player = 0
                                else:
                                    path_a, path_b = model_b["model_path"], model_a["model_path"]
                                    model_a_player = 1

                                # Create player AIs
                                player_ais = []
                                player_ais.append(create_neural_ai(
                                    path_a, 1, BoardType(board_type), num_players
                                ))
                                player_ais.append(create_neural_ai(
                                    path_b, 2, BoardType(board_type), num_players
                                ))

                                # Fill remaining slots with the top model
                                for p in range(2, num_players):
                                    player_ais.append(create_neural_ai(
                                        model_a["model_path"], p + 1,
                                        BoardType(board_type), num_players
                                    ))

                                game_result = play_single_game(
                                    board_type=BoardType(board_type),
                                    num_players=num_players,
                                    player_ais=player_ais,
                                    timeout=self.config.game_timeout_seconds,
                                )

                                winner = game_result.get("winner")
                                if winner == model_a_player:
                                    wins_a += 1

                                config_result["games_played"] += 1
                                results["total_games"] += 1
                                self._tournament_stats.games_played += 1

                            except Exception as e:
                                logger.warning(f"Top-N game failed: {e}")

                        # Record matchup results
                        win_rate_a = wins_a / games_per_matchup if games_per_matchup > 0 else 0
                        config_result["matchups"][matchup_key] = {
                            "model_a": model_a["participant_id"],
                            "model_b": model_b["participant_id"],
                            "wins_a": wins_a,
                            "games": games_per_matchup,
                            "win_rate_a": win_rate_a,
                        }

                        # Record in Elo service
                        try:
                            # January 2026: Default to gumbel_mcts for legacy model names
                            harness_type = extract_harness_type(model_a["participant_id"]) or "gumbel_mcts"
                            for _ in range(wins_a):
                                elo_service.record_match(
                                    participant_a=model_a["participant_id"],
                                    participant_b=model_b["participant_id"],
                                    winner=model_a["participant_id"],
                                    board_type=board_type,
                                    num_players=num_players,
                                    tournament_id=f"topn_roundrobin_{config_key}",
                                    harness_type=harness_type,
                                )
                            for _ in range(games_per_matchup - wins_a):
                                elo_service.record_match(
                                    participant_a=model_a["participant_id"],
                                    participant_b=model_b["participant_id"],
                                    winner=model_b["participant_id"],
                                    board_type=board_type,
                                    num_players=num_players,
                                    tournament_id=f"topn_roundrobin_{config_key}",
                                    harness_type=harness_type,
                                )
                        except Exception as e:
                            logger.warning(f"Failed to record top-N Elo: {e}")

                results["config_results"][config_key] = config_result
                results["configs_evaluated"] += 1

            results["success"] = True

            # Emit event for top-N round-robin results
            try:
                from app.distributed.data_events import DataEventType
                safe_emit_event(
                    DataEventType.TOPN_ROUNDROBIN_COMPLETED.value,
                    {
                        "configs_evaluated": results["configs_evaluated"],
                        "total_games": results["total_games"],
                        "config_results": results["config_results"],
                    },
                    context="TournamentDaemon",
                )
            except (ImportError, AttributeError):
                # Event type may not exist yet - emit generic
                safe_emit_event(
                    "TOPN_ROUNDROBIN_COMPLETED",
                    {
                        "configs_evaluated": results["configs_evaluated"],
                        "total_games": results["total_games"],
                    },
                    context="TournamentDaemon",
                )

        except ImportError as e:
            logger.warning(f"Top-N round-robin dependencies not available: {e}")
            results["error"] = "import_error"
        except Exception as e:
            logger.error(f"Top-N round-robin tournament failed: {e}")
            results["error"] = str(e)

        return results
