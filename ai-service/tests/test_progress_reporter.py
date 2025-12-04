"""Tests for progress reporter utility classes.

This module tests the progress reporter classes used for long-running AI jobs.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from app.utils.progress_reporter import (
    OptimizationProgressReporter,
    ProgressReporter,
    SoakProgressReporter,
    _format_duration,
)


class TestFormatDuration:
    """Tests for the _format_duration helper function."""

    def test_negative_returns_zero(self) -> None:
        """Negative durations return 0s."""
        assert _format_duration(-10) == "0s"

    def test_seconds_only(self) -> None:
        """Durations under 60s show seconds."""
        assert _format_duration(45) == "45s"
        assert _format_duration(0) == "0s"
        assert _format_duration(59.9) == "60s"  # rounds

    def test_minutes_and_seconds(self) -> None:
        """Durations 1-59 minutes show m:s format."""
        assert _format_duration(60) == "1m0s"
        assert _format_duration(90) == "1m30s"
        assert _format_duration(3599) == "59m59s"

    def test_hours_and_minutes(self) -> None:
        """Durations >= 1 hour show h:m format."""
        assert _format_duration(3600) == "1h0m"
        assert _format_duration(3660) == "1h1m"
        assert _format_duration(7200) == "2h0m"


class TestProgressReporter:
    """Tests for the general-purpose ProgressReporter class."""

    def test_creation(self) -> None:
        """Reporter can be created with required parameters."""
        reporter = ProgressReporter(total_units=100, unit_name="items")
        assert reporter.total_units == 100
        assert reporter.unit_name == "items"
        assert reporter.report_interval_sec == 10.0  # default
        assert reporter.context_label is None

    def test_update_respects_interval(self) -> None:
        """Updates only emit reports when interval elapsed."""
        reporter = ProgressReporter(
            total_units=10,
            unit_name="tests",
            report_interval_sec=1.0,
        )

        # First update - no report (interval not elapsed)
        output = io.StringIO()
        with redirect_stdout(output):
            reporter.update(completed=1)
        assert output.getvalue() == ""

    def test_update_force_report(self) -> None:
        """force_report=True emits a report immediately."""
        reporter = ProgressReporter(
            total_units=10,
            unit_name="items",
            report_interval_sec=100.0,  # Long interval
        )

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.update(completed=5, force_report=True)

        result = output.getvalue()
        assert "PROGRESS:" in result
        assert "5/10 items" in result
        assert "50.0%" in result

    def test_finish_emits_summary(self) -> None:
        """finish() emits a completion summary."""
        reporter = ProgressReporter(
            total_units=10,
            unit_name="items",
        )
        reporter._completed = 10

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.finish()

        result = output.getvalue()
        assert "COMPLETED:" in result
        assert "10 items" in result

    def test_context_label_in_output(self) -> None:
        """Context label appears in output when provided."""
        reporter = ProgressReporter(
            total_units=10,
            unit_name="items",
            context_label="test_run",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.update(completed=5, force_report=True)

        assert "[test_run]" in output.getvalue()

    def test_extra_metrics_in_output(self) -> None:
        """Extra metrics appear in progress output."""
        reporter = ProgressReporter(
            total_units=10,
            unit_name="items",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.update(
                completed=5,
                extra_metrics={"moves": 100, "accuracy": 0.95},
                force_report=True,
            )

        result = output.getvalue()
        assert "moves: 100" in result
        assert "accuracy: 0.95" in result

    def test_reset_clears_state(self) -> None:
        """reset() clears completed count and restarts timer."""
        reporter = ProgressReporter(
            total_units=10,
            unit_name="items",
        )
        reporter._completed = 5

        reporter.reset(total_units=20, context_label="phase2")

        assert reporter.total_units == 20
        assert reporter.context_label == "phase2"
        assert reporter._completed == 0


class TestSoakProgressReporter:
    """Tests for the SoakProgressReporter class."""

    def test_creation(self) -> None:
        """Reporter can be created with required parameters."""
        reporter = SoakProgressReporter(total_games=100)
        assert reporter.total_games == 100
        assert reporter.report_interval_sec == 10.0

    def test_record_game_tracks_stats(self) -> None:
        """record_game updates internal statistics."""
        reporter = SoakProgressReporter(
            total_games=10,
            report_interval_sec=100.0,  # Long interval
        )

        reporter.record_game(moves=50, duration_sec=1.5)
        reporter.record_game(moves=60, duration_sec=2.0)

        assert reporter._games_completed == 2
        assert reporter._total_moves == 110
        assert len(reporter._game_durations) == 2

    def test_record_game_force_report(self) -> None:
        """force_report=True emits a progress report."""
        reporter = SoakProgressReporter(
            total_games=10,
            report_interval_sec=100.0,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.record_game(moves=50, duration_sec=1.5, force_report=True)

        result = output.getvalue()
        assert "PROGRESS:" in result
        assert "1/10 games" in result

    def test_finish_emits_summary(self) -> None:
        """finish() emits comprehensive statistics."""
        reporter = SoakProgressReporter(total_games=10)
        reporter.record_game(moves=50, duration_sec=1.5)
        reporter.record_game(moves=60, duration_sec=2.0)

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.finish()

        result = output.getvalue()
        assert "COMPLETED:" in result
        assert "2 games" in result
        assert "total moves: 110" in result

    def test_decisions_tracking(self) -> None:
        """Decisions are tracked when provided."""
        reporter = SoakProgressReporter(total_games=10)
        reporter.record_game(moves=50, duration_sec=1.5, decisions=5)

        assert reporter._total_decisions == 5


class TestOptimizationProgressReporter:
    """Tests for the OptimizationProgressReporter class."""

    def test_creation(self) -> None:
        """Reporter can be created with required parameters."""
        reporter = OptimizationProgressReporter(
            total_generations=10,
            candidates_per_generation=16,
        )
        assert reporter.total_generations == 10
        assert reporter.candidates_per_generation == 16
        assert reporter.total_candidates == 160

    def test_start_generation_prints_header(self) -> None:
        """start_generation prints a generation header."""
        reporter = OptimizationProgressReporter(
            total_generations=10,
            candidates_per_generation=16,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.start_generation(1)

        assert "Generation 1/10" in output.getvalue()

    def test_record_candidate_tracks_stats(self) -> None:
        """record_candidate updates internal statistics."""
        reporter = OptimizationProgressReporter(
            total_generations=10,
            candidates_per_generation=16,
            report_interval_sec=100.0,
        )

        reporter.record_candidate(
            candidate_idx=1,
            fitness=0.75,
            states_processed=1000,
            games_played=10,
        )

        assert reporter._candidates_evaluated == 1
        assert reporter._best_fitness == 0.75
        assert reporter._total_states_processed == 1000
        assert reporter._total_games_played == 10

    def test_record_candidate_tracks_best_fitness(self) -> None:
        """Best fitness is tracked across candidates."""
        reporter = OptimizationProgressReporter(
            total_generations=10,
            candidates_per_generation=16,
            report_interval_sec=100.0,
        )

        reporter.record_candidate(candidate_idx=1, fitness=0.5)
        reporter.record_candidate(candidate_idx=2, fitness=0.9)
        reporter.record_candidate(candidate_idx=3, fitness=0.7)

        assert reporter._best_fitness == 0.9

    def test_finish_generation_prints_summary(self) -> None:
        """finish_generation prints generation summary."""
        reporter = OptimizationProgressReporter(
            total_generations=10,
            candidates_per_generation=16,
        )
        reporter._current_generation = 1

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.finish_generation(
                mean_fitness=0.7,
                best_fitness=0.9,
                std_fitness=0.1,
            )

        result = output.getvalue()
        assert "SUMMARY:" in result
        assert "mean=0.7000" in result
        assert "best=0.9000" in result

    def test_finish_prints_final_summary(self) -> None:
        """finish() prints final optimization summary."""
        reporter = OptimizationProgressReporter(
            total_generations=10,
            candidates_per_generation=16,
        )
        reporter._current_generation = 10
        reporter._candidates_evaluated = 160
        reporter._best_fitness = 0.95

        output = io.StringIO()
        with redirect_stdout(output):
            reporter.finish()

        result = output.getvalue()
        assert "OPTIMIZATION COMPLETED:" in result
        assert "10 generations" in result
        assert "160 candidates evaluated" in result


class TestImportFromPackage:
    """Tests that classes can be imported from the package."""

    def test_import_from_utils_package(self) -> None:
        """Classes can be imported from app.utils."""
        from app.utils import (
            OptimizationProgressReporter,
            ProgressReporter,
            SoakProgressReporter,
        )

        assert ProgressReporter is not None
        assert SoakProgressReporter is not None
        assert OptimizationProgressReporter is not None
