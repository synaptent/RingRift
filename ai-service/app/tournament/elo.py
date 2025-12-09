"""Elo rating calculation for tournament system.

Supports both 2-player and multiplayer games. For multiplayer, games are
decomposed into virtual pairwise matchups based on final rankings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Sequence


@dataclass
class EloRating:
    """Elo rating for an agent."""

    agent_id: str
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_history: List[Tuple[datetime, float]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    @property
    def expected_score(self) -> float:
        """Expected score based on games played."""
        if self.games_played == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games_played

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "rating": self.rating,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 3),
        }


class EloCalculator:
    """Standard Elo rating calculator with configurable K-factor."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        k_factor_high_rated: float = 16.0,
        high_rated_threshold: float = 2400.0,
        k_factor_provisional: float = 40.0,
        provisional_games: int = 30,
    ):
        """Initialize Elo calculator.

        Args:
            initial_rating: Starting rating for new players.
            k_factor: K-factor for normal players.
            k_factor_high_rated: K-factor for high-rated players.
            high_rated_threshold: Rating threshold for high-rated K-factor.
            k_factor_provisional: K-factor for provisional (new) players.
            provisional_games: Number of games before player is established.
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.k_factor_high_rated = k_factor_high_rated
        self.high_rated_threshold = high_rated_threshold
        self.k_factor_provisional = k_factor_provisional
        self.provisional_games = provisional_games

        self._ratings: Dict[str, EloRating] = {}

    def get_rating(self, agent_id: str) -> EloRating:
        """Get or create rating for an agent."""
        if agent_id not in self._ratings:
            self._ratings[agent_id] = EloRating(
                agent_id=agent_id,
                rating=self.initial_rating,
            )
        return self._ratings[agent_id]

    def get_k_factor(self, rating: EloRating) -> float:
        """Get K-factor for a player based on rating and games played."""
        if rating.games_played < self.provisional_games:
            return self.k_factor_provisional
        elif rating.rating >= self.high_rated_threshold:
            return self.k_factor_high_rated
        return self.k_factor

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B.

        Returns probability of player A winning (0.0 to 1.0).
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        agent_a_id: str,
        agent_b_id: str,
        result: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """Update ratings after a match.

        Args:
            agent_a_id: First player's agent ID.
            agent_b_id: Second player's agent ID.
            result: Result from A's perspective (1.0=win, 0.5=draw, 0.0=loss).
            timestamp: Optional timestamp for rating history.

        Returns:
            Tuple of (new_rating_a, new_rating_b).
        """
        timestamp = timestamp or datetime.now()

        rating_a = self.get_rating(agent_a_id)
        rating_b = self.get_rating(agent_b_id)

        expected_a = self.expected_score(rating_a.rating, rating_b.rating)
        expected_b = 1.0 - expected_a

        k_a = self.get_k_factor(rating_a)
        k_b = self.get_k_factor(rating_b)

        result_b = 1.0 - result

        new_rating_a = rating_a.rating + k_a * (result - expected_a)
        new_rating_b = rating_b.rating + k_b * (result_b - expected_b)

        rating_a.rating = new_rating_a
        rating_b.rating = new_rating_b

        rating_a.games_played += 1
        rating_b.games_played += 1

        if result == 1.0:
            rating_a.wins += 1
            rating_b.losses += 1
        elif result == 0.0:
            rating_a.losses += 1
            rating_b.wins += 1
        else:
            rating_a.draws += 1
            rating_b.draws += 1

        rating_a.rating_history.append((timestamp, new_rating_a))
        rating_b.rating_history.append((timestamp, new_rating_b))

        return new_rating_a, new_rating_b

    def get_leaderboard(self) -> List[EloRating]:
        """Get ratings sorted by rating (descending)."""
        return sorted(
            self._ratings.values(),
            key=lambda r: r.rating,
            reverse=True,
        )

    def get_all_ratings(self) -> Dict[str, EloRating]:
        """Get all ratings."""
        return self._ratings.copy()

    def reset(self) -> None:
        """Reset all ratings."""
        self._ratings.clear()

    def update_multiplayer_ratings(
        self,
        rankings: Sequence[str],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Update ratings after a multiplayer game based on final rankings.

        Decomposes the multiplayer result into virtual pairwise matchups.
        If player A finishes ahead of player B, A "beat" B in a virtual head-to-head.

        Args:
            rankings: Ordered list of agent IDs by finish position.
                      rankings[0] = 1st place, rankings[1] = 2nd place, etc.
            timestamp: Optional timestamp for rating history.

        Returns:
            Dict mapping agent_id to new rating.

        Example:
            # Player A wins, B second, C third, D fourth
            update_multiplayer_ratings(["agent_a", "agent_b", "agent_c", "agent_d"])
            # Results in virtual matchups:
            # A beats B, A beats C, A beats D
            # B beats C, B beats D
            # C beats D
        """
        if len(rankings) < 2:
            raise ValueError("Need at least 2 players for multiplayer rating update")

        timestamp = timestamp or datetime.now()
        n_players = len(rankings)

        # Get all ratings and calculate rating changes
        ratings = {agent_id: self.get_rating(agent_id) for agent_id in rankings}
        rating_deltas: Dict[str, float] = {agent_id: 0.0 for agent_id in rankings}

        # Process all pairwise matchups
        # For each pair (i, j) where i < j: player at rank i beat player at rank j
        for i in range(n_players):
            for j in range(i + 1, n_players):
                winner_id = rankings[i]  # Higher rank (lower index) = winner
                loser_id = rankings[j]

                winner_rating = ratings[winner_id]
                loser_rating = ratings[loser_id]

                # Calculate expected scores
                expected_winner = self.expected_score(
                    winner_rating.rating, loser_rating.rating
                )
                expected_loser = 1.0 - expected_winner

                # Get K-factors (scaled down by number of opponents for stability)
                k_winner = self.get_k_factor(winner_rating) / (n_players - 1)
                k_loser = self.get_k_factor(loser_rating) / (n_players - 1)

                # Winner gets result=1.0, loser gets result=0.0
                rating_deltas[winner_id] += k_winner * (1.0 - expected_winner)
                rating_deltas[loser_id] += k_loser * (0.0 - expected_loser)

        # Apply all rating changes and update stats
        new_ratings = {}
        for idx, agent_id in enumerate(rankings):
            rating = ratings[agent_id]
            rating.rating += rating_deltas[agent_id]
            rating.games_played += 1
            rating.rating_history.append((timestamp, rating.rating))

            # Update win/loss based on position
            if idx == 0:
                rating.wins += 1  # 1st place = win
            elif idx == n_players - 1:
                rating.losses += 1  # Last place = loss
            # Middle positions are neither win nor loss (could track separately)

            new_ratings[agent_id] = rating.rating

        return new_ratings

    def update_multiplayer_with_ties(
        self,
        rankings: Sequence[Tuple[str, int]],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Update ratings for multiplayer with potential ties.

        Args:
            rankings: List of (agent_id, rank) tuples. Rank 1 = first place.
                      Multiple agents can share the same rank (ties).
            timestamp: Optional timestamp for rating history.

        Returns:
            Dict mapping agent_id to new rating.

        Example:
            # A and B tie for 1st, C gets 3rd
            update_multiplayer_with_ties([
                ("agent_a", 1), ("agent_b", 1), ("agent_c", 3)
            ])
        """
        if len(rankings) < 2:
            raise ValueError("Need at least 2 players")

        timestamp = timestamp or datetime.now()
        n_players = len(rankings)

        # Get all ratings and calculate rating changes
        agents = [r[0] for r in rankings]
        ranks = {r[0]: r[1] for r in rankings}
        ratings = {agent_id: self.get_rating(agent_id) for agent_id in agents}
        rating_deltas: Dict[str, float] = {agent_id: 0.0 for agent_id in agents}

        # Process all pairwise matchups
        for i in range(n_players):
            for j in range(i + 1, n_players):
                agent_i, agent_j = agents[i], agents[j]
                rank_i, rank_j = ranks[agent_i], ranks[agent_j]

                rating_i = ratings[agent_i]
                rating_j = ratings[agent_j]

                expected_i = self.expected_score(rating_i.rating, rating_j.rating)
                expected_j = 1.0 - expected_i

                k_i = self.get_k_factor(rating_i) / (n_players - 1)
                k_j = self.get_k_factor(rating_j) / (n_players - 1)

                # Determine actual result based on ranks
                if rank_i < rank_j:
                    # i finished higher (better rank = lower number)
                    result_i, result_j = 1.0, 0.0
                elif rank_i > rank_j:
                    # j finished higher
                    result_i, result_j = 0.0, 1.0
                else:
                    # Tie
                    result_i, result_j = 0.5, 0.5

                rating_deltas[agent_i] += k_i * (result_i - expected_i)
                rating_deltas[agent_j] += k_j * (result_j - expected_j)

        # Apply all rating changes and update stats
        new_ratings = {}
        min_rank = min(ranks.values())
        max_rank = max(ranks.values())

        for agent_id in agents:
            rating = ratings[agent_id]
            rating.rating += rating_deltas[agent_id]
            rating.games_played += 1
            rating.rating_history.append((timestamp, rating.rating))

            rank = ranks[agent_id]
            if rank == min_rank:
                rating.wins += 1
            elif rank == max_rank:
                rating.losses += 1
            # Ties in non-extreme positions don't count as win/loss

            new_ratings[agent_id] = rating.rating

        return new_ratings

    def to_dict(self) -> Dict:
        """Serialize calculator state."""
        return {
            "ratings": {
                agent_id: rating.to_dict()
                for agent_id, rating in self._ratings.items()
            },
            "config": {
                "initial_rating": self.initial_rating,
                "k_factor": self.k_factor,
                "k_factor_high_rated": self.k_factor_high_rated,
                "high_rated_threshold": self.high_rated_threshold,
                "k_factor_provisional": self.k_factor_provisional,
                "provisional_games": self.provisional_games,
            },
        }
