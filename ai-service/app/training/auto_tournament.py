"""
Automated Tournament Pipeline for AI Model Evaluation

This module provides an automated pipeline for:
- Running tournaments between model versions
- Automatic "champion" model promotion
- Elo rating tracking across versions
- Performance report generation

Integrates with ModelVersionManager for checkpoint loading and
tournament.py for match execution.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.training.model_versioning import (
    LegacyCheckpointError,
    ModelMetadata,
    ModelVersionManager,
)
from app.training.tournament import VICTORY_REASONS, Tournament

# Import canonical thresholds
try:
    from app.config.thresholds import (
        ELO_K_FACTOR,
        INITIAL_ELO_RATING,
        PROMOTION_WIN_RATE_THRESHOLD,
    )
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32
    PROMOTION_WIN_RATE_THRESHOLD = 0.80  # Dec 2025: raised from 0.60 for 2000+ Elo

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RegisteredModel:
    """Information about a registered model in the tournament system."""

    model_id: str
    model_path: str
    metadata: ModelMetadata
    elo_rating: float = 1500.0
    registered_at: str = ""
    is_champion: bool = False
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def __post_init__(self) -> None:
        if not self.registered_at:
            self.registered_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # ModelMetadata needs special handling
        data["metadata"] = self.metadata.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegisteredModel":
        """Create from dictionary."""
        data["metadata"] = ModelMetadata.from_dict(data["metadata"])
        return cls(**data)

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.wins / self.games_played) * 100


@dataclass
class MatchResult:
    """Result of a single match between two models."""

    model_a_id: str
    model_b_id: str
    winner_id: str | None  # None for draw
    victory_reason: str
    game_number: int
    played_at: str = ""

    def __post_init__(self) -> None:
        if not self.played_at:
            self.played_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TournamentResult:
    """Results from a full tournament."""

    tournament_id: str
    participants: list[str]
    matches: list[MatchResult]
    final_elo_ratings: dict[str, float]
    final_standings: list[tuple[str, float]]  # (model_id, rating) sorted
    started_at: str
    finished_at: str = ""
    victory_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tournament_id": self.tournament_id,
            "participants": self.participants,
            "matches": [asdict(m) for m in self.matches],
            "final_elo_ratings": self.final_elo_ratings,
            "final_standings": self.final_standings,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "victory_reasons": self.victory_reasons,
        }


@dataclass
class ChallengerResult:
    """Result of challenger evaluation against champion."""

    challenger_id: str
    champion_id: str
    challenger_wins: int
    champion_wins: int
    draws: int
    total_games: int
    challenger_win_rate: float
    champion_win_rate: float
    statistical_p_value: float
    is_statistically_significant: bool
    challenger_final_elo: float
    champion_final_elo: float
    should_promote: bool
    evaluation_time: str = ""
    victory_reasons: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evaluation_time:
            self.evaluation_time = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# =============================================================================
# Statistical Functions
# =============================================================================


def calculate_binomial_p_value(
    successes: int, trials: int, null_probability: float = 0.5
) -> float:
    """
    Calculate p-value for a binomial test.

    Tests whether win rate is greater than the null probability.
    Uses exact binomial calculation for accuracy.

    Args:
        successes: Number of successes (wins for challenger).
        trials: Total trials (games played, excluding draws).
        null_probability: Null hypothesis probability (0.5 = equal).

    Returns:
        One-tailed p-value for successes >= observed.
    """
    if trials == 0:
        return 1.0

    # Calculate probability of getting >= successes wins by chance
    # Using exact binomial calculation
    p_value = 0.0

    for k in range(successes, trials + 1):
        # Binomial coefficient: C(n, k) = n! / (k! * (n-k)!)
        coeff = _binomial_coefficient(trials, k)
        p_k = null_probability ** k
        p_nk = (1 - null_probability) ** (trials - k)
        prob = coeff * p_k * p_nk
        p_value += prob

    return p_value


def _binomial_coefficient(n: int, k: int) -> int:
    """Calculate binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # Use symmetry for efficiency
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def calculate_elo_change(
    rating_a: float, rating_b: float, score_a: float, k_factor: float = 32.0
) -> tuple[float, float]:
    """
    Calculate Elo rating changes for both players.

    Args:
        rating_a: Current rating of player A.
        rating_b: Current rating of player B.
        score_a: Actual score for player A (1.0 win, 0.5 draw, 0.0 loss).
        k_factor: K-factor for rating adjustment.

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    # Expected score for A
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a

    score_b = 1.0 - score_a

    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)

    return new_rating_a, new_rating_b


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


# =============================================================================
# AutoTournamentPipeline
# =============================================================================


class AutoTournamentPipeline:
    """
    Automated tournament pipeline for AI model evaluation.

    This pipeline manages:
    - Model registration with metadata
    - Tournament execution between registered models
    - Elo rating tracking and updates
    - Champion model promotion based on statistical significance
    - Performance report generation

    Example usage::

        pipeline = AutoTournamentPipeline(
            models_dir="./models",
            results_dir="./results"
        )

        # Register models
        pipeline.register_model("model_v1.pth", metadata)
        pipeline.register_model("model_v2.pth", metadata)

        # Run tournament
        result = pipeline.run_tournament(games_per_match=20)

        # Evaluate a new challenger against the champion
        challenge_result = pipeline.evaluate_challenger(
            "new_model.pth", games=50
        )

        if challenge_result.should_promote:
            pipeline.promote_champion(challenge_result.challenger_id)

        # Generate report
        report = pipeline.generate_report()
    """

    # Promotion thresholds - use canonical values from app.config.thresholds
    WIN_RATE_THRESHOLD = float(PROMOTION_WIN_RATE_THRESHOLD)
    PROMOTION_SIGNIFICANCE_LEVEL = 0.05  # p-value threshold
    DEFAULT_ELO = float(INITIAL_ELO_RATING)
    ELO_K = float(ELO_K_FACTOR)  # Renamed to avoid shadowing import

    def __init__(
        self,
        models_dir: str,
        results_dir: str,
        registry_file: str | None = None,
    ):
        """
        Initialize the tournament pipeline.

        Args:
            models_dir: Directory containing model checkpoints.
            results_dir: Directory for storing tournament results.
            registry_file: Optional path to model registry JSON file.
                If None, uses {results_dir}/model_registry.json.
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.registry_file = registry_file or os.path.join(
            results_dir, "model_registry.json"
        )

        # Registered models by ID
        self._models: dict[str, RegisteredModel] = {}

        # Model version manager for checkpoint operations
        self._version_manager = ModelVersionManager()

        # Tournament history
        self._tournament_history: list[TournamentResult] = []

        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Load existing registry if available
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)

                self._models = {
                    model_id: RegisteredModel.from_dict(model_data)
                    for model_id, model_data in data.get("models", {}).items()
                }

                logger.info(
                    f"Loaded {len(self._models)} models from registry: "
                    f"{self.registry_file}"
                )
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                self._models = {}

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        data = {
            "models": {
                model_id: model.to_dict()
                for model_id, model in self._models.items()
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved registry with {len(self._models)} models")

    def register_model(
        self,
        model_path: str,
        metadata: ModelMetadata | None = None,
        initial_elo: float | None = None,
    ) -> str:
        """
        Register a new model version for tournament.

        Args:
            model_path: Path to the model checkpoint file.
            metadata: Optional model metadata. If None, will be extracted
                from the checkpoint if it's a versioned checkpoint.
            initial_elo: Optional initial Elo rating. Defaults to 1500.

        Returns:
            The model_id assigned to the registered model.

        Raises:
            FileNotFoundError: If model_path doesn't exist.
            LegacyCheckpointError: If checkpoint has no metadata and
                none is provided.
        """
        # Resolve path
        if not os.path.isabs(model_path):
            full_path = os.path.join(self.models_dir, model_path)
        else:
            full_path = model_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model not found: {full_path}")

        # Extract or validate metadata
        if metadata is None:
            try:
                metadata = self._version_manager.get_metadata(full_path)
            except LegacyCheckpointError:
                raise LegacyCheckpointError(full_path)

        # Generate model ID
        model_name = os.path.splitext(os.path.basename(full_path))[0]
        model_id = f"{model_name}_{uuid.uuid4().hex[:8]}"

        # Check if this exact checkpoint is already registered (by checksum)
        for existing in self._models.values():
            if existing.metadata.checksum == metadata.checksum:
                logger.info(
                    f"Model with checksum {metadata.checksum[:16]}... "
                    f"already registered as {existing.model_id}"
                )
                return existing.model_id

        # Create registered model
        registered = RegisteredModel(
            model_id=model_id,
            model_path=full_path,
            metadata=metadata,
            elo_rating=initial_elo or self.DEFAULT_ELO,
            is_champion=len(self._models) == 0,  # First model is champion
        )

        self._models[model_id] = registered
        self._save_registry()

        logger.info(
            f"Registered model {model_id}\n"
            f"  Path: {full_path}\n"
            f"  Version: {metadata.architecture_version}\n"
            f"  Elo: {registered.elo_rating}\n"
            f"  Champion: {registered.is_champion}"
        )

        return model_id

    def get_champion(self) -> RegisteredModel | None:
        """Get the current champion model."""
        for model in self._models.values():
            if model.is_champion:
                return model
        return None

    def get_model(self, model_id: str) -> RegisteredModel | None:
        """Get a registered model by ID."""
        return self._models.get(model_id)

    def list_models(self) -> list[RegisteredModel]:
        """List all registered models sorted by Elo rating."""
        return sorted(
            self._models.values(),
            key=lambda m: m.elo_rating,
            reverse=True,
        )

    def run_tournament(
        self,
        participants: list[str] | None = None,
        games_per_match: int = 10,
    ) -> TournamentResult:
        """
        Run full tournament between registered models.

        Args:
            participants: Optional list of model IDs to include.
                If None, all registered models participate.
            games_per_match: Number of games per pairwise match.

        Returns:
            TournamentResult with all match results and final standings.

        Raises:
            ValueError: If fewer than 2 participants are available.
        """
        # Get participants
        if participants is None:
            participant_ids = list(self._models.keys())
        else:
            participant_ids = [p for p in participants if p in self._models]

        if len(participant_ids) < 2:
            raise ValueError(
                f"Need at least 2 participants, got {len(participant_ids)}"
            )

        logger.info(
            f"Starting tournament with {len(participant_ids)} participants, "
            f"{games_per_match} games per match"
        )

        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        tournament_id = f"tournament_{ts}"
        started_at = datetime.now(timezone.utc).isoformat()
        matches: list[MatchResult] = []
        victory_reasons: dict[str, int] = dict.fromkeys(VICTORY_REASONS, 0)

        # Round-robin tournament
        game_number = 0
        for i, model_a_id in enumerate(participant_ids):
            for model_b_id in participant_ids[i + 1:]:
                model_a = self._models[model_a_id]
                model_b = self._models[model_b_id]

                logger.info(
                    f"Match: {model_a_id} vs {model_b_id} "
                    f"({games_per_match} games)"
                )

                # Run the match using existing Tournament class
                tournament = Tournament(
                    model_path_a=model_a.model_path,
                    model_path_b=model_b.model_path,
                    num_games=games_per_match,
                    k_elo=int(self.ELO_K),
                )

                # Override initial ratings with current Elo
                tournament.ratings["A"] = model_a.elo_rating
                tournament.ratings["B"] = model_b.elo_rating

                results = tournament.run()

                # Record individual games
                for _g in range(results["A"]):
                    game_number += 1
                    matches.append(
                        MatchResult(
                            model_a_id=model_a_id,
                            model_b_id=model_b_id,
                            winner_id=model_a_id,
                            victory_reason="unknown",  # Detailed in tournament
                            game_number=game_number,
                        )
                    )

                for _g in range(results["B"]):
                    game_number += 1
                    matches.append(
                        MatchResult(
                            model_a_id=model_a_id,
                            model_b_id=model_b_id,
                            winner_id=model_b_id,
                            victory_reason="unknown",
                            game_number=game_number,
                        )
                    )

                for _g in range(results["Draw"]):
                    game_number += 1
                    matches.append(
                        MatchResult(
                            model_a_id=model_a_id,
                            model_b_id=model_b_id,
                            winner_id=None,
                            victory_reason="draw",
                            game_number=game_number,
                        )
                    )

                # Aggregate victory reasons (tolerate legacy/new keys)
                for reason, count in tournament.victory_reasons.items():
                    victory_reasons[reason] = victory_reasons.get(reason, 0) + count

                # Update Elo ratings
                model_a.elo_rating = tournament.ratings["A"]
                model_b.elo_rating = tournament.ratings["B"]

                # Update statistics
                model_a.games_played += games_per_match
                model_b.games_played += games_per_match
                model_a.wins += results["A"]
                model_b.wins += results["B"]
                model_a.losses += results["B"]
                model_b.losses += results["A"]
                model_a.draws += results["Draw"]
                model_b.draws += results["Draw"]

        # Create result
        final_standings = self.get_elo_rankings()
        final_elo_ratings = {
            model_id: self._models[model_id].elo_rating
            for model_id in participant_ids
        }

        result = TournamentResult(
            tournament_id=tournament_id,
            participants=participant_ids,
            matches=matches,
            final_elo_ratings=final_elo_ratings,
            final_standings=final_standings,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            victory_reasons=victory_reasons,
        )

        # Save results
        self._tournament_history.append(result)
        self._save_tournament_result(result)
        self._save_registry()

        standings_str = "\n".join(
            f"  {i+1}. {model_id}: {elo:.1f}"
            for i, (model_id, elo) in enumerate(final_standings)
        )
        logger.info(f"Tournament complete. Final standings:\n{standings_str}")

        return result

    def _save_tournament_result(self, result: TournamentResult) -> None:
        """Save tournament result to disk."""
        result_path = os.path.join(
            self.results_dir, f"{result.tournament_id}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved tournament result to {result_path}")

    def evaluate_challenger(
        self,
        challenger_path: str,
        games: int = 50,
        challenger_metadata: ModelMetadata | None = None,
    ) -> ChallengerResult:
        """
        Evaluate new model against current champion.

        Args:
            challenger_path: Path to the challenger model checkpoint.
            games: Number of games to play.
            challenger_metadata: Optional metadata for the challenger.
                If None, will be extracted from checkpoint.

        Returns:
            ChallengerResult with evaluation results and promotion decision.

        Raises:
            ValueError: If no champion is registered.
            FileNotFoundError: If challenger_path doesn't exist.
        """
        champion = self.get_champion()
        if champion is None:
            raise ValueError("No champion registered")

        # Resolve challenger path
        if not os.path.isabs(challenger_path):
            full_path = os.path.join(self.models_dir, challenger_path)
        else:
            full_path = challenger_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Challenger model not found: {full_path}")

        # Register challenger (temporarily with default Elo)
        challenger_id = self.register_model(
            full_path,
            metadata=challenger_metadata,
            initial_elo=self.DEFAULT_ELO,
        )
        challenger = self._models[challenger_id]

        logger.info(
            f"Evaluating challenger {challenger_id} against "
            f"champion {champion.model_id} ({games} games)"
        )

        # Run matches
        tournament = Tournament(
            model_path_a=challenger.model_path,
            model_path_b=champion.model_path,
            num_games=games,
            k_elo=int(self.ELO_K),
        )

        # Initialize with current ratings
        tournament.ratings["A"] = challenger.elo_rating
        tournament.ratings["B"] = champion.elo_rating

        results = tournament.run()

        challenger_wins = results["A"]
        champion_wins = results["B"]
        draws = results["Draw"]

        # Calculate win rates (excluding draws for fair comparison)
        decisive_games = challenger_wins + champion_wins
        if decisive_games > 0:
            challenger_win_rate = challenger_wins / decisive_games
            champion_win_rate = champion_wins / decisive_games
        else:
            challenger_win_rate = 0.5
            champion_win_rate = 0.5

        # Statistical significance test
        p_value = calculate_binomial_p_value(
            successes=challenger_wins,
            trials=decisive_games,
            null_probability=0.5,
        )
        is_significant = p_value < self.PROMOTION_SIGNIFICANCE_LEVEL

        # Update Elo ratings
        challenger.elo_rating = tournament.ratings["A"]
        champion.elo_rating = tournament.ratings["B"]

        # Update statistics
        challenger.games_played += games
        champion.games_played += games
        challenger.wins += challenger_wins
        challenger.losses += champion_wins
        challenger.draws += draws
        champion.wins += champion_wins
        champion.losses += challenger_wins
        champion.draws += draws

        # Determine if challenger should be promoted
        should_promote = self._should_promote(
            challenger_win_rate=challenger_win_rate,
            is_significant=is_significant,
            challenger_elo=challenger.elo_rating,
            champion_elo=champion.elo_rating,
        )

        result = ChallengerResult(
            challenger_id=challenger_id,
            champion_id=champion.model_id,
            challenger_wins=challenger_wins,
            champion_wins=champion_wins,
            draws=draws,
            total_games=games,
            challenger_win_rate=challenger_win_rate,
            champion_win_rate=champion_win_rate,
            statistical_p_value=p_value,
            is_statistically_significant=is_significant,
            challenger_final_elo=challenger.elo_rating,
            champion_final_elo=champion.elo_rating,
            should_promote=should_promote,
            victory_reasons=tournament.victory_reasons,
        )

        # Save registry with updated stats
        self._save_registry()

        # Save evaluation result
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        result_filename = f"challenge_{challenger_id}_{ts}.json"
        result_path = os.path.join(self.results_dir, result_filename)
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        sig_str = 'significant' if is_significant else 'not significant'
        logger.info(
            f"Challenger evaluation complete:\n"
            f"  Win rate: {challenger_win_rate:.1%} vs "
            f"{champion_win_rate:.1%}\n"
            f"  P-value: {p_value:.4f} ({sig_str})\n"
            f"  Final Elo: {challenger.elo_rating:.1f} vs "
            f"{champion.elo_rating:.1f}\n"
            f"  Should promote: {should_promote}"
        )

        return result

    def _should_promote(
        self,
        challenger_win_rate: float,
        is_significant: bool,
        challenger_elo: float,
        champion_elo: float,
    ) -> bool:
        """
        Determine if challenger should be promoted to champion.

        Promotion criteria:
        - Win rate >= 55%
        - Result is statistically significant (p < 0.05)
        - Challenger has higher Elo than current champion

        Args:
            challenger_win_rate: Challenger's win rate against champion.
            is_significant: Whether result is statistically significant.
            challenger_elo: Challenger's final Elo rating.
            champion_elo: Champion's final Elo rating.

        Returns:
            True if challenger should be promoted.
        """
        return (
            challenger_win_rate >= self.WIN_RATE_THRESHOLD
            and is_significant
            and challenger_elo > champion_elo
        )

    def promote_champion(self, model_id: str) -> None:
        """
        Promote model to champion status.

        Args:
            model_id: ID of model to promote.

        Raises:
            ValueError: If model_id is not registered.
        """
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")

        # Demote current champion
        for model in self._models.values():
            model.is_champion = False

        # Promote new champion
        self._models[model_id].is_champion = True
        self._save_registry()

        logger.info(f"Promoted {model_id} to champion")

    def get_elo_rankings(self) -> list[tuple[str, float]]:
        """
        Get current Elo rankings for all models.

        Returns:
            List of (model_id, elo_rating) tuples sorted by rating descending.
        """
        rankings = [
            (model.model_id, model.elo_rating)
            for model in self._models.values()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def generate_report(self) -> str:
        """
        Generate markdown performance report.

        Returns:
            Markdown-formatted report string.
        """
        lines = [
            "# AI Tournament Performance Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Current Champion",
            "",
        ]

        champion = self.get_champion()
        if champion:
            wins, losses, draws = champion.wins, champion.losses, champion.draws
            record = f"{wins}W / {losses}L / {draws}D"
            arch_ver = champion.metadata.architecture_version
            lines.extend([
                f"**{champion.model_id}**",
                "",
                f"- **Elo Rating:** {champion.elo_rating:.1f}",
                f"- **Games Played:** {champion.games_played}",
                f"- **Win Rate:** {champion.win_rate:.1f}%",
                f"- **Record:** {record}",
                f"- **Architecture Version:** {arch_ver}",
                f"- **Registered:** {champion.registered_at}",
                "",
            ])
        else:
            lines.extend(["*No champion registered*", ""])

        lines.extend([
            "## Elo Rankings",
            "",
            "| Rank | Model | Elo | Games | Win Rate | W/L/D |",
            "|------|-------|-----|-------|----------|-------|",
        ])

        for rank, (model_id, elo) in enumerate(self.get_elo_rankings(), 1):
            model = self._models[model_id]
            champion_marker = " 👑" if model.is_champion else ""
            lines.append(
                f"| {rank} | {model_id}{champion_marker} | {elo:.1f} | "
                f"{model.games_played} | {model.win_rate:.1f}% | "
                f"{model.wins}/{model.losses}/{model.draws} |"
            )

        lines.extend(["", "## Model Details", ""])

        for model in sorted(
            self._models.values(), key=lambda m: m.elo_rating, reverse=True
        ):
            lines.extend([
                f"### {model.model_id}",
                "",
                f"- **Path:** `{model.model_path}`",
                f"- **Model Class:** {model.metadata.model_class}",
                f"- **Architecture:** {model.metadata.architecture_version}",
                f"- **Checksum:** `{model.metadata.checksum[:16]}...`",
                "",
            ])

            if model.metadata.training_info:
                lines.append("**Training Info:**")
                for key, value in model.metadata.training_info.items():
                    lines.append(f"- {key}: {value}")
                lines.append("")

        # Tournament history summary
        if self._tournament_history:
            lines.extend([
                "## Tournament History",
                "",
                "| Tournament | Participants | Games | Winner |",
                "|------------|--------------|-------|--------|",
            ])

            for t in self._tournament_history[-10:]:  # Last 10 tournaments
                if t.final_standings:
                    winner_id = t.final_standings[0][0]
                else:
                    winner_id = "N/A"
                lines.append(
                    f"| {t.tournament_id} | {len(t.participants)} | "
                    f"{len(t.matches)} | {winner_id} |"
                )
            lines.append("")

        # Victory reasons summary
        lines.extend([
            "## Victory Types",
            "",
            "| Type | Count |",
            "|------|-------|",
        ])

        # Aggregate victory reasons from all tournaments
        total_reasons: dict[str, int] = dict.fromkeys(VICTORY_REASONS, 0)
        for t in self._tournament_history:
            for reason, count in t.victory_reasons.items():
                total_reasons[reason] = total_reasons.get(reason, 0) + count

        for reason, count in sorted(
            total_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            if count > 0:
                lines.append(f"| {reason} | {count} |")

        lines.extend([
            "", "---", "",
            "*Report generated by AutoTournamentPipeline*"
        ])

        return "\n".join(lines)

    def save_report(self, filename: str | None = None) -> str:
        """
        Save report to a markdown file.

        Args:
            filename: Optional filename. Defaults to report_YYYYMMDD.md.

        Returns:
            Path to the saved report.
        """
        if filename is None:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"report_{ts}.md"

        report_path = os.path.join(self.results_dir, filename)
        report = self.generate_report()

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Saved report to {report_path}")
        return report_path


# =============================================================================
# Helper Functions
# =============================================================================


def should_promote(challenger_result: ChallengerResult) -> bool:
    """
    Standalone function to check if challenger should be promoted.

    Challenger becomes champion if:
    - Wins >= 55% of games (statistically significant)
    - Has higher Elo than current champion

    Args:
        challenger_result: Result from evaluate_challenger().

    Returns:
        True if challenger should be promoted.
    """
    return challenger_result.should_promote
