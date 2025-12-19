"""Tier-aware evaluation and gating runner for RingRift difficulty tiers.

This module orchestrates evaluation runs between a candidate AI and
tier-configured opponents using the canonical RingRiftEnv environment.
It is intentionally light-weight and data-driven; thresholds and tier
definitions live in tier_eval_config, while this module is responsible
for actually playing games and computing gating metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from app.models import AIConfig, AIType, BoardType, GameStatus
from app.training.env import TrainingEnvConfig, make_env
from app.training.tier_eval_config import (
    TierEvaluationConfig,
    TierOpponentConfig,
)
from app.training.significance import wilson_score_interval
from app.config.ladder_config import (
    LadderTierConfig,
    get_ladder_tier_config,
)
from app.game_engine import GameEngine
from app.main import (
    DifficultyProfile,
    _create_ai_instance,
    _get_difficulty_profile,
)


@dataclass
class MatchupStats:
    """Aggregated statistics for a single candidate-vs-opponent matchup."""

    opponent_id: str
    opponent_difficulty: int
    opponent_ai_type: Optional[str]
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_moves: int = 0
    victory_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.wins / float(self.games)

    @property
    def average_game_length(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.total_moves / float(self.games)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "opponent_difficulty": self.opponent_difficulty,
            "opponent_ai_type": self.opponent_ai_type,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "average_game_length": self.average_game_length,
            "victory_reasons": dict(self.victory_reasons),
        }


@dataclass
class TierEvaluationResult:
    """Structured result for a tier evaluation run."""

    tier_name: str
    board_type: BoardType
    num_players: int
    candidate_id: str
    candidate_difficulty: int
    total_games: int
    matchups: List[MatchupStats] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    criteria: Dict[str, Optional[bool]] = field(default_factory=dict)
    overall_pass: bool = True

    def to_dict(self) -> Dict[str, Any]:
        overall_victory_reasons: Dict[str, int] = {}
        total_moves = 0
        for matchup in self.matchups:
            for reason, count in matchup.victory_reasons.items():
                overall_victory_reasons[reason] = (
                    overall_victory_reasons.get(reason, 0) + count
                )
            total_moves += matchup.total_moves

        overall_avg_length = 0.0
        if self.total_games > 0:
            overall_avg_length = total_moves / float(self.total_games)

        return {
            "tier": self.tier_name,
            "board_type": self.board_type.value,
            "num_players": self.num_players,
            "candidate": {
                "id": self.candidate_id,
                "difficulty": self.candidate_difficulty,
            },
            "stats": {
                "total_games": self.total_games,
                "overall": {
                    "wins": sum(m.wins for m in self.matchups),
                    "losses": sum(m.losses for m in self.matchups),
                    "draws": sum(m.draws for m in self.matchups),
                    "average_game_length": overall_avg_length,
                    "victory_reasons": overall_victory_reasons,
                },
                "by_opponent": {
                    m.opponent_id: m.to_dict() for m in self.matchups
                },
            },
            "metrics": self.metrics,
            "criteria": self.criteria,
            "overall_pass": self.overall_pass,
        }


def _create_ladder_ai_instance(
    tier_config: TierEvaluationConfig,
    difficulty: int,
    player_number: int,
    time_budget_ms: Optional[int],
    ai_type_override: Optional[AIType] = None,
    candidate_override_id: Optional[str] = None,
    rng_seed: Optional[int] = None,
):
    """Create an AI instance using the canonical difficulty ladder.

    This helper mirrors the /ai/move construction logic in app.main so
    that tier evaluation uses the same underlying AI family, randomness,
    and search budgets as the production ladder. When a board-aware
    LadderTierConfig exists for (difficulty, board_type, num_players),
    its values take precedence; otherwise we fall back to the global
    difficulty profile.
    """
    profile: DifficultyProfile = _get_difficulty_profile(difficulty)

    ladder_config: Optional[LadderTierConfig] = None
    try:
        ladder_config = get_ladder_tier_config(
            difficulty,
            tier_config.board_type,
            tier_config.num_players,
        )
    except Exception:
        ladder_config = None

    base_ai_type = (
        ladder_config.ai_type
        if ladder_config is not None
        else profile["ai_type"]
    )
    ai_type = ai_type_override or base_ai_type

    randomness = (
        ladder_config.randomness
        if ladder_config is not None
        else profile["randomness"]
    )

    if time_budget_ms is not None:
        think_time_ms = time_budget_ms
    elif ladder_config is not None:
        think_time_ms = ladder_config.think_time_ms
    else:
        think_time_ms = profile["think_time_ms"]

    use_neural_net = (
        ladder_config.use_neural_net
        if ladder_config is not None
        else bool(profile.get("use_neural_net", False))
    )

    heuristic_profile_id: Optional[str] = None
    if ladder_config is not None and ladder_config.heuristic_profile_id:
        heuristic_profile_id = ladder_config.heuristic_profile_id
    elif ai_type == AIType.HEURISTIC:
        heuristic_profile_id = profile.get("profile_id")

    nn_model_id: Optional[str] = None
    if use_neural_net and ladder_config is not None and ladder_config.model_id:
        nn_model_id = ladder_config.model_id

    # Candidate overrides are used only by higher-level promotion tooling and
    # should not affect default ladder evaluation unless explicitly enabled.
    if candidate_override_id:
        if ai_type == AIType.HEURISTIC:
            heuristic_profile_id = candidate_override_id
        elif use_neural_net:
            nn_model_id = candidate_override_id

    config = AIConfig(
        difficulty=difficulty,
        randomness=randomness,
        think_time=think_time_ms,
        rngSeed=rng_seed,
        heuristic_profile_id=heuristic_profile_id,
        nn_model_id=nn_model_id,
        use_neural_net=use_neural_net,
    )
    return _create_ai_instance(ai_type, player_number, config)


def _play_matchup(
    tier_config: TierEvaluationConfig,
    opponent: TierOpponentConfig,
    base_seed: Optional[int],
    candidate_override_id: Optional[str] = None,
    num_games_override: Optional[int] = None,
    time_budget_ms_override: Optional[int] = None,
    max_moves_override: Optional[int] = None,
) -> MatchupStats:
    """Run games between the candidate tier and a single opponent.

    Evaluates the candidate configuration against a single configured
    opponent entry and returns aggregated statistics.
    """
    games_to_play = opponent.games or tier_config.num_games
    if num_games_override is not None:
        games_to_play = num_games_override

    resolved_ai_type = opponent.ai_type
    if resolved_ai_type is None:
        profile = _get_difficulty_profile(opponent.difficulty)
        resolved_ai_type = profile["ai_type"]

    stats = MatchupStats(
        opponent_id=opponent.id,
        opponent_difficulty=opponent.difficulty,
        opponent_ai_type=resolved_ai_type.value if resolved_ai_type else None,
    )

    logger.info(
        "[tier-gate] Starting matchup vs %s (difficulty=%d, ai=%s, games=%d)",
        opponent.id,
        opponent.difficulty,
        resolved_ai_type.value if resolved_ai_type else "default",
        games_to_play,
    )
    matchup_start = time.time()

    env_config = TrainingEnvConfig(
        board_type=tier_config.board_type,
        num_players=tier_config.num_players,
        max_moves=max_moves_override,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    for game_index in range(games_to_play):
        stats.games += 1
        # Rotate the candidate across all seats to reduce first-move bias.
        # In 3p/4p, all non-candidate seats are controlled by independent
        # opponent AIs configured from the same TierOpponentConfig.
        num_players = int(tier_config.num_players)
        if num_players < 2:
            raise ValueError(
                f"tier_eval_runner requires num_players>=2 (got {num_players})"
            )
        candidate_seat = (game_index % num_players) + 1

        game_seed: Optional[int] = None
        if base_seed is not None:
            game_seed = (base_seed * 1_000_003 + game_index) & 0x7FFFFFFF

        def _derive_player_seed(base: Optional[int], player: int) -> Optional[int]:
            if base is None:
                return None
            return (base * 1_000_003 + player * 97) & 0x7FFFFFFF

        candidate_ai = _create_ladder_ai_instance(
            tier_config=tier_config,
            difficulty=tier_config.candidate_difficulty,
            player_number=candidate_seat,
            time_budget_ms=(
                time_budget_ms_override
                if time_budget_ms_override is not None
                else tier_config.time_budget_ms
            ),
            ai_type_override=None,
            candidate_override_id=candidate_override_id,
            rng_seed=_derive_player_seed(game_seed, candidate_seat),
        )

        ai_by_player: Dict[int, Any] = {candidate_seat: candidate_ai}
        for player_number in range(1, num_players + 1):
            if player_number == candidate_seat:
                continue
            ai_by_player[player_number] = _create_ladder_ai_instance(
                tier_config=tier_config,
                difficulty=opponent.difficulty,
                player_number=player_number,
                time_budget_ms=(
                    time_budget_ms_override
                    if time_budget_ms_override is not None
                    else tier_config.time_budget_ms
                ),
                ai_type_override=resolved_ai_type,
                candidate_override_id=None,
                rng_seed=_derive_player_seed(game_seed, player_number),
            )

        game_state = env.reset(seed=game_seed)
        done = False
        last_info: Dict[str, Any] = {}
        moves_played = 0

        while not done:
            current_player = game_state.current_player
            current_ai = ai_by_player.get(current_player)
            if current_ai is None:
                raise RuntimeError(
                    f"tier_eval_runner: missing AI for player {current_player}"
                )
            move = current_ai.select_move(game_state)
            if move is None:
                # Per RR-CANON-R076: when get_valid_moves returns empty,
                # check for phase requirements that require bookkeeping moves
                # (NO_*_ACTION, FORCED_ELIMINATION, etc.)
                requirement = GameEngine.get_phase_requirement(
                    game_state,
                    current_player,
                )
                if requirement is not None:
                    # Synthesize the required bookkeeping move and continue
                    move = GameEngine.synthesize_bookkeeping_move(
                        requirement,
                        game_state,
                    )
                else:
                    # True "no moves" case - treat as immediate loss for the side that
                    # failed to produce a move.
                    if current_player == candidate_seat:
                        stats.losses += 1
                    else:
                        stats.wins += 1
                    stats.victory_reasons["ai_no_move"] = (
                        stats.victory_reasons.get("ai_no_move", 0) + 1
                    )
                    stats.total_moves += moves_played
                    break

            game_state, reward, done, info = env.step(move)
            last_info = info
            moves_played = info.get("move_count", moves_played + 1)

        if (
            not last_info
            and game_state is not None
            and game_state.game_status != GameStatus.ACTIVE
        ):
            # Game terminated without env.step providing info (should not
            # happen, but we guard defensively).
            last_info = {}

        if done:
            winner = game_state.winner
            if winner == candidate_seat:
                stats.wins += 1
                result_str = "WIN"
            elif winner is None:
                stats.draws += 1
                result_str = "DRAW"
            else:
                stats.losses += 1
                result_str = "LOSS"

            victory_reason = last_info.get("victory_reason", "unknown")
            stats.victory_reasons[victory_reason] = (
                stats.victory_reasons.get(victory_reason, 0) + 1
            )
            stats.total_moves += last_info.get("moves_played", moves_played)

            # Progress logging
            win_rate = stats.win_rate * 100.0
            logger.info(
                "[tier-gate] Game %d/%d vs %s: %s (%s, %d moves) | W/D/L: %d/%d/%d (%.1f%%)",
                game_index + 1,
                games_to_play,
                opponent.id,
                result_str,
                victory_reason,
                moves_played,
                stats.wins,
                stats.draws,
                stats.losses,
                win_rate,
            )

    elapsed = time.time() - matchup_start
    logger.info(
        "[tier-gate] Matchup vs %s completed in %.1fs | Final W/D/L: %d/%d/%d (%.1f%%)",
        opponent.id,
        elapsed,
        stats.wins,
        stats.draws,
        stats.losses,
        stats.win_rate * 100.0,
    )
    return stats


def run_tier_evaluation(
    tier_config: TierEvaluationConfig,
    candidate_id: str,
    *,
    candidate_override_id: Optional[str] = None,
    seed: Optional[int] = None,
    num_games_override: Optional[int] = None,
    time_budget_ms_override: Optional[int] = None,
    max_moves_override: Optional[int] = None,
) -> TierEvaluationResult:
    """Evaluate a candidate configuration for a given difficulty tier.

    Args:
        tier_config: TierEvaluationConfig describing the tier profile.
        candidate_id: Human/machine readable identifier for the candidate
            configuration under test (for example a model path, profile id,
            or git SHA). Only used for reporting; the AI implementation
            is derived from the canonical difficulty ladder.
        seed: Optional base RNG seed for reproducible evaluations.
        num_games_override: When provided, overrides tier_config.num_games
            for all opponents.
    """
    matchups: List[MatchupStats] = []
    total_games = 0

    baseline_wins = 0
    baseline_losses = 0
    baseline_draws = 0
    baseline_games = 0
    prev_wins = 0
    prev_games = 0

    num_opponents = len(tier_config.opponents)
    logger.info(
        "[tier-gate] Starting tier %s evaluation for candidate %s (%d opponents)",
        tier_config.tier_name,
        candidate_id,
        num_opponents,
    )
    eval_start = time.time()

    for opponent_idx, opponent in enumerate(tier_config.opponents, 1):
        logger.info(
            "[tier-gate] Opponent %d/%d: %s",
            opponent_idx,
            num_opponents,
            opponent.id,
        )
        stats = _play_matchup(
            tier_config=tier_config,
            opponent=opponent,
            base_seed=seed,
            candidate_override_id=candidate_override_id,
            num_games_override=num_games_override,
            time_budget_ms_override=time_budget_ms_override,
            max_moves_override=max_moves_override,
        )
        matchups.append(stats)
        total_games += stats.games

        if opponent.role == "baseline":
            baseline_wins += stats.wins
            baseline_losses += stats.losses
            baseline_draws += stats.draws
            baseline_games += stats.games
        if opponent.role == "previous_tier":
            prev_wins += stats.wins
            prev_games += stats.games

    metrics: Dict[str, Optional[float]] = {}
    criteria: Dict[str, Optional[bool]] = {}

    win_rate_vs_baseline: Optional[float] = None
    if baseline_games > 0:
        win_rate_vs_baseline = baseline_wins / float(baseline_games)
    metrics["win_rate_vs_baseline"] = win_rate_vs_baseline

    # Wilson confidence interval (decisive games only).
    confidence = getattr(tier_config, "promotion_confidence", 0.95)
    decisive_baseline_games = baseline_wins + baseline_losses
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    if confidence is not None and confidence > 0 and decisive_baseline_games > 0:
        ci_low, ci_high = wilson_score_interval(
            baseline_wins,
            decisive_baseline_games,
            confidence=float(confidence),
        )
    metrics["win_rate_vs_baseline_ci_low"] = ci_low
    metrics["win_rate_vs_baseline_ci_high"] = ci_high

    if (
        tier_config.min_win_rate_vs_baseline is not None
        and win_rate_vs_baseline is not None
    ):
        if ci_low is not None:
            criteria["min_win_rate_vs_baseline"] = (
                ci_low >= tier_config.min_win_rate_vs_baseline
            )
        else:
            criteria["min_win_rate_vs_baseline"] = (
                win_rate_vs_baseline >= tier_config.min_win_rate_vs_baseline
            )
    else:
        criteria["min_win_rate_vs_baseline"] = None

    win_rate_vs_prev: Optional[float] = None
    if prev_games > 0:
        win_rate_vs_prev = prev_wins / float(prev_games)
    metrics["win_rate_vs_previous_tier"] = win_rate_vs_prev

    if (
        tier_config.max_regression_vs_previous_tier is not None
        and win_rate_vs_prev is not None
    ):
        # Treat 50% as "no regression" baseline and allow the candidate to be
        # at most max_regression_vs_previous_tier below that.
        allowed_floor = 0.5 - tier_config.max_regression_vs_previous_tier
        criteria["no_major_regression_vs_previous_tier"] = (
            win_rate_vs_prev >= allowed_floor
        )
    else:
        criteria["no_major_regression_vs_previous_tier"] = None

    overall_pass = True
    for value in criteria.values():
        if value is False:
            overall_pass = False

    elapsed = time.time() - eval_start
    result_str = "PASSED" if overall_pass else "FAILED"
    total_wins = sum(m.wins for m in matchups)
    total_losses = sum(m.losses for m in matchups)
    total_draws = sum(m.draws for m in matchups)
    overall_win_rate = total_wins / total_games * 100.0 if total_games > 0 else 0.0
    logger.info(
        "[tier-gate] Tier %s evaluation %s in %.1fs | Total W/D/L: %d/%d/%d (%.1f%%) | Games: %d",
        tier_config.tier_name,
        result_str,
        elapsed,
        total_wins,
        total_draws,
        total_losses,
        overall_win_rate,
        total_games,
    )

    return TierEvaluationResult(
        tier_name=tier_config.tier_name,
        board_type=tier_config.board_type,
        num_players=tier_config.num_players,
        candidate_id=candidate_id,
        candidate_difficulty=tier_config.candidate_difficulty,
        total_games=total_games,
        matchups=matchups,
        metrics=metrics,
        criteria=criteria,
        overall_pass=overall_pass,
    )
