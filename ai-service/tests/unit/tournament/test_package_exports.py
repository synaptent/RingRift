"""Focused tests for app.tournament package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_tournament_surface() -> None:
    module = importlib.import_module("app.tournament")

    expected = [
        "ELO_DB_PATH",
        "AIAgent",
        "AIAgentRegistry",
        "AgentType",
        "EloCalculator",
        "EloDatabase",
        "EloRating",
        "EloService",
        "EvaluationResult",
        "Match",
        "MatchRecord",
        "MatchResult",
        "MatchStatus",
        "RoundRobinScheduler",
        "SwissScheduler",
        "TournamentOrchestrator",
        "TournamentResults",
        "TournamentRunner",
        "TournamentRecordingOptions",
        "TournamentScheduler",
        "TournamentSummary",
        "TournamentConfig",
        "UnifiedEloRating",
        "create_tournament_runner",
        "get_database_stats",
        "get_elo_database",
        "get_elo_service",
        "get_head_to_head",
        "get_match_history",
        "get_rating_history",
        "reset_elo_database",
        "run_elo_calibration",
        "run_quick_evaluation",
        "run_quick_tournament",
        "CompositeGauntlet",
        "CompositeGauntletConfig",
        "run_two_phase_gauntlet",
        "run_algorithm_tournament",
        "run_nn_tournament",
        "AlgorithmTournament",
        "NNTournament",
        "CombinedTournament",
        "TournamentScheduleManager",
        "TournamentType",
        "HierarchicalCullingController",
        "CullingReport",
        "run_hierarchical_culling",
        "check_culling_needed",
        "aggregate_by_nn",
        "aggregate_by_algorithm",
        "check_nn_ranking_consistency",
        "print_aggregation_report",
        "ConsistencyMonitor",
        "ConsistencyReport",
        "run_consistency_checks",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
