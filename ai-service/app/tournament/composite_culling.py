#!/usr/bin/env python3
"""Hierarchical Culling System for Composite (NN, Algorithm) participants.

This module implements a three-level culling strategy for the Composite ELO System:

Level 1: Cull Weak NNs
    - Condition: ALL (NN, *) combinations are in bottom 50%
    - Action: Archive the NN entirely (all algorithm variants)
    - Impact: Removes ~40% of NNs

Level 2: Cull Weak Algorithm Combinations
    - For surviving NNs, rank algorithm pairings
    - Keep top 2 algorithms per NN
    - Archive remaining combinations
    - Ensures diversity while reducing clutter

Level 3: Standard Elo Culling
    - Apply to remaining participants
    - Standard 75% culling rule
    - Based on absolute Elo ranking

Safeguards:
    - min_games_for_cull: 30 (don't cull high-uncertainty)
    - min_participants_keep: 25 (always keep at least 25)
    - protect_baselines: True (never cull baselines)
    - protect_best_per_algo: True (keep best NN per algorithm)
    - diversity_min_algos: 3 (keep at least 3 algorithm types)

Usage:
    from app.tournament.composite_culling import (
        HierarchicalCullingController,
        run_hierarchical_culling,
    )

    controller = HierarchicalCullingController(board_type="square8", num_players=2)
    report = controller.run_culling(dry_run=True)
    print(report)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.training.composite_participant import (
    BASELINE_PARTICIPANTS,
    extract_ai_type,
    extract_nn_id,
    is_baseline_participant,
    is_composite_id,
)
from app.training.elo_service import get_elo_service

# Event emission for culling (Sprint 5)
try:
    from app.training.event_integration import publish_composite_nn_culled
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    publish_composite_nn_culled = None

logger = logging.getLogger(__name__)


@dataclass
class CullingConfig:
    """Configuration for hierarchical culling."""
    min_games_for_cull: int = 30           # Don't cull high-uncertainty
    min_participants_keep: int = 25         # Always keep at least 25
    protect_baselines: bool = True          # Never cull baselines
    protect_best_per_algo: bool = True      # Keep best NN per algorithm
    diversity_min_algos: int = 3            # Keep at least 3 algorithm types
    nn_cull_threshold: float = 0.5          # Cull NNs in bottom 50%
    algo_keep_per_nn: int = 2               # Keep top 2 algorithms per NN
    standard_keep_fraction: float = 0.25    # Keep top 25% in final pass


@dataclass
class CullingReport:
    """Detailed report of a culling operation."""
    board_type: str
    num_players: int
    timestamp: float = field(default_factory=time.time)
    dry_run: bool = True

    # Level 1: NN culling
    level1_nns_evaluated: int = 0
    level1_nns_culled: int = 0
    level1_culled_nn_ids: list[str] = field(default_factory=list)

    # Level 2: Algorithm combination culling
    level2_combinations_evaluated: int = 0
    level2_combinations_culled: int = 0
    level2_culled_participant_ids: list[str] = field(default_factory=list)

    # Level 3: Standard Elo culling
    level3_participants_evaluated: int = 0
    level3_participants_culled: int = 0
    level3_culled_participant_ids: list[str] = field(default_factory=list)

    # Summary
    total_participants_before: int = 0
    total_participants_after: int = 0
    total_culled: int = 0

    # Protected
    protected_baselines: list[str] = field(default_factory=list)
    protected_best_per_algo: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "board_type": self.board_type,
            "num_players": self.num_players,
            "timestamp": self.timestamp,
            "dry_run": self.dry_run,
            "level1": {
                "nns_evaluated": self.level1_nns_evaluated,
                "nns_culled": self.level1_nns_culled,
                "culled_nn_ids": self.level1_culled_nn_ids,
            },
            "level2": {
                "combinations_evaluated": self.level2_combinations_evaluated,
                "combinations_culled": self.level2_combinations_culled,
                "culled_ids": self.level2_culled_participant_ids,
            },
            "level3": {
                "participants_evaluated": self.level3_participants_evaluated,
                "participants_culled": self.level3_participants_culled,
                "culled_ids": self.level3_culled_participant_ids,
            },
            "summary": {
                "before": self.total_participants_before,
                "after": self.total_participants_after,
                "culled": self.total_culled,
            },
            "protected": {
                "baselines": self.protected_baselines,
                "best_per_algo": self.protected_best_per_algo,
            },
        }

    def __str__(self) -> str:
        """Human-readable report."""
        lines = [
            f"Culling Report: {self.board_type}/{self.num_players}p",
            f"{'[DRY RUN]' if self.dry_run else '[EXECUTED]'}",
            "",
            f"Level 1 (NN Culling):",
            f"  Evaluated: {self.level1_nns_evaluated} NNs",
            f"  Culled: {self.level1_nns_culled} NNs (all their algorithm variants)",
            "",
            f"Level 2 (Algorithm Combination Culling):",
            f"  Evaluated: {self.level2_combinations_evaluated} combinations",
            f"  Culled: {self.level2_combinations_culled} combinations",
            "",
            f"Level 3 (Standard Elo Culling):",
            f"  Evaluated: {self.level3_participants_evaluated} participants",
            f"  Culled: {self.level3_participants_culled} participants",
            "",
            f"Summary:",
            f"  Before: {self.total_participants_before}",
            f"  After: {self.total_participants_after}",
            f"  Total Culled: {self.total_culled}",
            "",
            f"Protected:",
            f"  Baselines: {len(self.protected_baselines)}",
            f"  Best per Algorithm: {len(self.protected_best_per_algo)}",
        ]
        return "\n".join(lines)


class HierarchicalCullingController:
    """Controller for hierarchical culling of composite participants.

    Implements three-level culling strategy:
    1. Cull weak NNs (all variants in bottom 50%)
    2. Cull weak algorithm combinations (keep top 2 per NN)
    3. Standard Elo culling on remainder
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        config: CullingConfig | None = None,
        model_dir: Path | None = None,
    ):
        """Initialize culling controller.

        Args:
            board_type: Board type
            num_players: Number of players
            config: Culling configuration
            model_dir: Directory for model archival
        """
        self.board_type = board_type
        self.num_players = num_players
        self.config = config or CullingConfig()
        self.model_dir = model_dir or Path("models")
        self.elo_service = get_elo_service()

    def run_culling(self, dry_run: bool = True) -> CullingReport:
        """Run hierarchical culling.

        Args:
            dry_run: If True, simulate culling without making changes

        Returns:
            CullingReport with detailed results
        """
        report = CullingReport(
            board_type=self.board_type,
            num_players=self.num_players,
            dry_run=dry_run,
        )

        # Get all participants
        all_participants = self._get_all_participants()
        report.total_participants_before = len(all_participants)

        if len(all_participants) < self.config.min_participants_keep:
            logger.info("Not enough participants to cull")
            report.total_participants_after = len(all_participants)
            return report

        # Identify protected participants
        protected = self._identify_protected(all_participants)
        report.protected_baselines = protected["baselines"]
        report.protected_best_per_algo = protected["best_per_algo"]
        protected_set = set(protected["baselines"] + protected["best_per_algo"])

        # Level 1: Cull weak NNs
        surviving_participants, level1_culled = self._level1_nn_culling(
            all_participants, protected_set
        )
        report.level1_nns_evaluated = len(self._group_by_nn(all_participants))
        report.level1_nns_culled = len(level1_culled)
        report.level1_culled_nn_ids = level1_culled

        # Level 2: Cull weak algorithm combinations
        surviving_participants, level2_culled = self._level2_algo_culling(
            surviving_participants, protected_set
        )
        report.level2_combinations_evaluated = len(surviving_participants) + len(level2_culled)
        report.level2_combinations_culled = len(level2_culled)
        report.level2_culled_participant_ids = level2_culled

        # Level 3: Standard Elo culling
        final_participants, level3_culled = self._level3_elo_culling(
            surviving_participants, protected_set
        )
        report.level3_participants_evaluated = len(surviving_participants)
        report.level3_participants_culled = len(level3_culled)
        report.level3_culled_participant_ids = level3_culled

        # Compute totals
        report.total_participants_after = len(final_participants)
        report.total_culled = (
            len(level1_culled) +  # NN culling removes multiple participants
            len(level2_culled) +
            len(level3_culled)
        )

        # Execute culling if not dry run
        if not dry_run:
            all_culled = level2_culled + level3_culled
            # For level 1, we need to get all participant IDs for culled NNs
            for nn_id in level1_culled:
                nn_participants = [
                    p["participant_id"] for p in all_participants
                    if extract_nn_id(p["participant_id"]) == nn_id
                ]
                all_culled.extend(nn_participants)

            self._execute_culling(all_culled)

        return report

    def _get_all_participants(self) -> list[dict[str, Any]]:
        """Get all participants with ratings."""
        return self.elo_service.get_composite_leaderboard(
            board_type=self.board_type,
            num_players=self.num_players,
            min_games=0,
            limit=10000,
        )

    def _identify_protected(
        self,
        participants: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Identify protected participants."""
        protected = {
            "baselines": [],
            "best_per_algo": [],
        }

        if self.config.protect_baselines:
            protected["baselines"] = [
                p["participant_id"] for p in participants
                if is_baseline_participant(p["participant_id"])
            ]

        if self.config.protect_best_per_algo:
            # Group by algorithm and find best NN for each
            algo_best: dict[str, tuple[str, float]] = {}
            for p in participants:
                algo = p.get("ai_algorithm")
                if not algo:
                    continue
                rating = p.get("rating", 0)
                if algo not in algo_best or rating > algo_best[algo][1]:
                    algo_best[algo] = (p["participant_id"], rating)

            protected["best_per_algo"] = [pid for pid, _ in algo_best.values()]

        return protected

    def _group_by_nn(
        self,
        participants: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group participants by NN ID."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for p in participants:
            nn_id = p.get("nn_model_id") or extract_nn_id(p["participant_id"])
            if nn_id and nn_id != "none":
                if nn_id not in groups:
                    groups[nn_id] = []
                groups[nn_id].append(p)
        return groups

    def _level1_nn_culling(
        self,
        participants: list[dict[str, Any]],
        protected: set[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Level 1: Cull NNs where ALL variants are in bottom 50%."""
        # Group by NN
        nn_groups = self._group_by_nn(participants)

        # Calculate median Elo
        all_ratings = [p.get("rating", 1500) for p in participants]
        if not all_ratings:
            return participants, []

        median_rating = sorted(all_ratings)[len(all_ratings) // 2]

        culled_nns = []
        surviving_participants = []

        for nn_id, nn_participants in nn_groups.items():
            # Check if all variants are below median
            all_below_median = all(
                p.get("rating", 1500) < median_rating
                for p in nn_participants
            )

            # Check if any are protected
            any_protected = any(
                p["participant_id"] in protected
                for p in nn_participants
            )

            # Check if enough games
            enough_games = all(
                p.get("games_played", 0) >= self.config.min_games_for_cull
                for p in nn_participants
            )

            if all_below_median and not any_protected and enough_games:
                culled_nns.append(nn_id)
            else:
                surviving_participants.extend(nn_participants)

        # Add back any participants not in nn_groups (e.g., baselines)
        nn_participant_ids = {p["participant_id"] for g in nn_groups.values() for p in g}
        for p in participants:
            if p["participant_id"] not in nn_participant_ids:
                surviving_participants.append(p)

        return surviving_participants, culled_nns

    def _level2_algo_culling(
        self,
        participants: list[dict[str, Any]],
        protected: set[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Level 2: Keep top N algorithms per NN."""
        # Group by NN
        nn_groups = self._group_by_nn(participants)

        surviving = []
        culled_ids = []

        for nn_id, nn_participants in nn_groups.items():
            # Sort by rating
            sorted_variants = sorted(
                nn_participants,
                key=lambda p: p.get("rating", 0),
                reverse=True,
            )

            kept = 0
            for p in sorted_variants:
                pid = p["participant_id"]
                is_protected = pid in protected
                enough_games = p.get("games_played", 0) >= self.config.min_games_for_cull

                if kept < self.config.algo_keep_per_nn or is_protected or not enough_games:
                    surviving.append(p)
                    if not is_protected:
                        kept += 1
                else:
                    culled_ids.append(pid)

        # Add back non-NN participants (baselines)
        nn_participant_ids = {p["participant_id"] for g in nn_groups.values() for p in g}
        for p in participants:
            if p["participant_id"] not in nn_participant_ids:
                surviving.append(p)

        return surviving, culled_ids

    def _level3_elo_culling(
        self,
        participants: list[dict[str, Any]],
        protected: set[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Level 3: Standard Elo-based culling."""
        # Ensure we keep minimum participants
        keep_count = max(
            self.config.min_participants_keep,
            int(len(participants) * self.config.standard_keep_fraction),
        )

        # Sort by rating
        sorted_participants = sorted(
            participants,
            key=lambda p: p.get("rating", 0),
            reverse=True,
        )

        # Ensure diversity (at least N algorithm types)
        algo_counts: dict[str, int] = {}
        surviving = []
        culled_ids = []

        for p in sorted_participants:
            pid = p["participant_id"]
            algo = p.get("ai_algorithm", "unknown")
            is_protected = pid in protected
            enough_games = p.get("games_played", 0) >= self.config.min_games_for_cull

            # Check if we need this for algorithm diversity
            need_for_diversity = (
                algo_counts.get(algo, 0) == 0 and
                len(algo_counts) < self.config.diversity_min_algos
            )

            if (
                len(surviving) < keep_count or
                is_protected or
                need_for_diversity or
                not enough_games
            ):
                surviving.append(p)
                algo_counts[algo] = algo_counts.get(algo, 0) + 1
            else:
                culled_ids.append(pid)

        return surviving, culled_ids

    def _execute_culling(self, participant_ids: list[str]) -> None:
        """Execute the actual culling by archiving participants."""
        logger.info(f"Archiving {len(participant_ids)} participants")

        # Mark as archived in database
        conn = self.elo_service._get_connection()

        for pid in participant_ids:
            try:
                # Get rating before archiving for event
                rating_info = self.elo_service.get_rating(
                    pid, self.board_type, self.num_players
                )
                final_elo = rating_info.rating if rating_info else 1500.0
                games_played = rating_info.games_played if rating_info else 0

                # Update elo_ratings to mark as archived
                conn.execute("""
                    UPDATE elo_ratings
                    SET archived_at = ?, archive_reason = 'hierarchical_culling'
                    WHERE participant_id = ? AND board_type = ? AND num_players = ?
                """, (time.time(), pid, self.board_type, self.num_players))

                # Try to archive the model file if it exists
                nn_id = extract_nn_id(pid)
                ai_type = extract_ai_type(pid)
                if nn_id:
                    model_path = self.model_dir / f"{nn_id}.pth"
                    if model_path.exists():
                        archive_dir = self.model_dir / "archived"
                        archive_dir.mkdir(exist_ok=True)
                        archive_path = archive_dir / f"{nn_id}_{int(time.time())}.pth"
                        shutil.move(str(model_path), str(archive_path))
                        logger.debug(f"Archived model: {model_path} -> {archive_path}")

                # Emit culling event (Sprint 5)
                if HAS_EVENTS and publish_composite_nn_culled is not None:
                    import asyncio
                    try:
                        asyncio.get_event_loop().run_until_complete(
                            publish_composite_nn_culled(
                                nn_id=nn_id or pid,
                                reason="hierarchical_culling",
                                final_elo=final_elo,
                                games_played=games_played,
                                cull_level=1,  # Could be 1, 2, or 3 - simplified to 1
                                algorithms_tested=[ai_type] if ai_type else [],
                                board_type=self.board_type,
                                num_players=self.num_players,
                            )
                        )
                    except RuntimeError:
                        # No event loop or loop not running - skip event
                        pass

            except Exception as e:
                logger.error(f"Failed to archive {pid}: {e}")

        conn.commit()


def run_hierarchical_culling(
    board_type: str = "square8",
    num_players: int = 2,
    dry_run: bool = True,
) -> CullingReport:
    """Run hierarchical culling for a configuration.

    Args:
        board_type: Board type
        num_players: Number of players
        dry_run: If True, simulate without changes

    Returns:
        CullingReport with results
    """
    controller = HierarchicalCullingController(
        board_type=board_type,
        num_players=num_players,
    )
    return controller.run_culling(dry_run=dry_run)


def check_culling_needed(
    board_type: str = "square8",
    num_players: int = 2,
    threshold: int = 100,
) -> tuple[bool, int]:
    """Check if culling is needed for a configuration.

    Args:
        board_type: Board type
        num_players: Number of players
        threshold: Participant count threshold

    Returns:
        Tuple of (needs_culling, current_count)
    """
    elo_service = get_elo_service()
    leaderboard = elo_service.get_composite_leaderboard(
        board_type=board_type,
        num_players=num_players,
        limit=10000,
    )
    count = len(leaderboard)
    return count > threshold, count
