"""Progress reporter utility for long-running AI training and soak processes.

This module provides a reusable progress reporter that outputs meaningful
progress information to the terminal at configurable intervals (default: ~10s).

Usage
-----
    from app.utils.progress_reporter import ProgressReporter

    reporter = ProgressReporter(
        total_units=1000,
        unit_name="games",
        report_interval_sec=10.0,
    )

    for i in range(1000):
        # Do work...
        reporter.update(
            completed=i + 1,
            extra_metrics={"moves": total_moves, "decisions": total_decisions}
        )

    reporter.finish()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OptimizationProgressReporter",
    "ProgressReporter",
    "SoakProgressReporter",
]


@dataclass
class ProgressReporter:
    """Progress reporter with time-based output for long-running processes.

    Attributes
    ----------
    total_units : int
        Total number of units to process (e.g., games, generations, candidates).
    unit_name : str
        Name of the unit being processed (e.g., "games", "candidates").
    report_interval_sec : float
        Minimum seconds between progress reports (default: 10.0).
    context_label : Optional[str]
        Optional label for context (e.g., "generation 3", "candidate 5").
    """

    total_units: int
    unit_name: str = "units"
    report_interval_sec: float = 10.0
    context_label: str | None = None

    # Internal tracking
    _start_time: float = field(default_factory=time.time, init=False)
    _last_report_time: float = field(default=0.0, init=False)
    _completed: int = field(default=0, init=False)
    _metrics_history: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._start_time = time.time()
        self._last_report_time = self._start_time

    def reset(
        self,
        total_units: int | None = None,
        context_label: str | None = None,
    ) -> None:
        """Reset the reporter for a new phase/generation."""
        if total_units is not None:
            self.total_units = total_units
        if context_label is not None:
            self.context_label = context_label
        self._start_time = time.time()
        self._last_report_time = self._start_time
        self._completed = 0
        self._metrics_history = []

    def update(
        self,
        completed: int,
        extra_metrics: dict[str, Any] | None = None,
        force_report: bool = False,
    ) -> None:
        """Update progress and potentially emit a report.

        Parameters
        ----------
        completed : int
            Number of units completed so far.
        extra_metrics : Optional[Dict[str, Any]]
            Additional metrics to include in the report (e.g., moves, decisions).
        force_report : bool
            If True, emit a report regardless of time interval.
        """
        self._completed = completed
        now = time.time()
        elapsed_since_report = now - self._last_report_time

        if force_report or elapsed_since_report >= self.report_interval_sec:
            self._emit_report(extra_metrics)
            self._last_report_time = now

    def _emit_report(self, extra_metrics: dict[str, Any] | None = None) -> None:
        """Emit a progress report to stdout."""
        now = time.time()
        elapsed = now - self._start_time
        completed = self._completed
        total = self.total_units
        remaining = max(0, total - completed)

        # Calculate rate and ETA
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta_sec = remaining / rate if rate > 0 else 0.0

        # Format percentage
        pct = (completed / total * 100) if total > 0 else 0.0

        # Build the report line
        parts: list[str] = []

        # Context label if present
        if self.context_label:
            parts.append(f"[{self.context_label}]")

        parts.append("PROGRESS:")
        parts.append(f"{completed}/{total} {self.unit_name} ({pct:.1f}%)")
        parts.append(f"| elapsed: {_format_duration(elapsed)}")
        parts.append(f"| rate: {rate:.2f} {self.unit_name}/sec")
        parts.append(f"| ETA: {_format_duration(eta_sec)}")

        # Extra metrics
        if extra_metrics:
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    parts.append(f"| {key}: {value:.2f}")
                else:
                    parts.append(f"| {key}: {value}")

        line = " ".join(parts)
        print(line, flush=True)

    def finish(self, extra_metrics: dict[str, Any] | None = None) -> None:
        """Emit a final summary report."""
        now = time.time()
        elapsed = now - self._start_time
        completed = self._completed
        rate = completed / elapsed if elapsed > 0 else 0.0

        parts: list[str] = []

        if self.context_label:
            parts.append(f"[{self.context_label}]")

        parts.append("COMPLETED:")
        parts.append(f"{completed} {self.unit_name}")
        parts.append(f"| total time: {_format_duration(elapsed)}")
        parts.append(f"| avg rate: {rate:.2f} {self.unit_name}/sec")

        if extra_metrics:
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    parts.append(f"| {key}: {value:.2f}")
                else:
                    parts.append(f"| {key}: {value}")

        line = " ".join(parts)
        print(line, flush=True)


@dataclass
class SoakProgressReporter:
    """Specialized progress reporter for game-based soaks with rich metrics.

    Tracks games played, total moves, decisions, and timing statistics.
    """

    total_games: int
    report_interval_sec: float = 10.0
    context_label: str | None = None

    # Internal tracking
    _start_time: float = field(default_factory=time.time, init=False)
    _last_report_time: float = field(default=0.0, init=False)
    _games_completed: int = field(default=0, init=False)
    _total_moves: int = field(default=0, init=False)
    _total_decisions: int = field(default=0, init=False)
    _game_durations: list[float] = field(default_factory=list, init=False)
    _game_lengths: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._start_time = time.time()
        self._last_report_time = self._start_time

    def record_game(
        self,
        moves: int,
        duration_sec: float,
        decisions: int = 0,
        force_report: bool = False,
    ) -> None:
        """Record completion of a game and potentially emit a report.

        Parameters
        ----------
        moves : int
            Number of moves in the completed game.
        duration_sec : float
            How long the game took in seconds.
        decisions : int
            Number of AI decisions made (if tracked separately).
        force_report : bool
            If True, emit a report regardless of time interval.
        """
        self._games_completed += 1
        self._total_moves += moves
        self._total_decisions += decisions
        self._game_durations.append(duration_sec)
        self._game_lengths.append(moves)

        now = time.time()
        elapsed_since_report = now - self._last_report_time

        if force_report or elapsed_since_report >= self.report_interval_sec:
            self._emit_report()
            self._last_report_time = now

    def _emit_report(self) -> None:
        """Emit a progress report with soak-specific metrics."""
        now = time.time()
        elapsed = now - self._start_time
        games = self._games_completed
        total = self.total_games
        remaining = max(0, total - games)

        # Calculate rates
        games_per_sec = games / elapsed if elapsed > 0 else 0.0
        moves_per_sec = self._total_moves / elapsed if elapsed > 0 else 0.0
        decisions_per_sec = (
            self._total_decisions / elapsed if elapsed > 0 else 0.0
        )

        # Calculate averages
        avg_moves_per_game = (
            self._total_moves / games if games > 0 else 0.0
        )
        avg_sec_per_game = sum(self._game_durations) / games if games > 0 else 0.0

        # ETA
        eta_sec = remaining / games_per_sec if games_per_sec > 0 else 0.0

        # Percentage
        pct = (games / total * 100) if total > 0 else 0.0

        parts: list[str] = []

        if self.context_label:
            parts.append(f"[{self.context_label}]")

        parts.append("PROGRESS:")
        parts.append(f"{games}/{total} games ({pct:.1f}%)")
        parts.append(f"| elapsed: {_format_duration(elapsed)}")
        parts.append(f"| ETA: {_format_duration(eta_sec)}")
        parts.append(f"| games/sec: {games_per_sec:.2f}")
        parts.append(f"| moves/sec: {moves_per_sec:.1f}")
        if self._total_decisions > 0:
            parts.append(f"| decisions/sec: {decisions_per_sec:.1f}")
        parts.append(f"| avg moves/game: {avg_moves_per_game:.1f}")
        parts.append(f"| avg sec/game: {avg_sec_per_game:.2f}")

        line = " ".join(parts)
        print(line, flush=True)

    def finish(self) -> None:
        """Emit a final summary report."""
        now = time.time()
        elapsed = now - self._start_time
        games = self._games_completed

        games_per_sec = games / elapsed if elapsed > 0 else 0.0
        moves_per_sec = self._total_moves / elapsed if elapsed > 0 else 0.0
        decisions_per_sec = (
            self._total_decisions / elapsed if elapsed > 0 else 0.0
        )
        avg_moves_per_game = self._total_moves / games if games > 0 else 0.0
        avg_sec_per_game = sum(self._game_durations) / games if games > 0 else 0.0

        parts: list[str] = []

        if self.context_label:
            parts.append(f"[{self.context_label}]")

        parts.append("COMPLETED:")
        parts.append(f"{games} games")
        parts.append(f"| total time: {_format_duration(elapsed)}")
        parts.append(f"| total moves: {self._total_moves}")
        if self._total_decisions > 0:
            parts.append(f"| total decisions: {self._total_decisions}")
        parts.append(f"| avg games/sec: {games_per_sec:.2f}")
        parts.append(f"| avg moves/sec: {moves_per_sec:.1f}")
        if self._total_decisions > 0:
            parts.append(f"| avg decisions/sec: {decisions_per_sec:.1f}")
        parts.append(f"| avg moves/game: {avg_moves_per_game:.1f}")
        parts.append(f"| avg sec/game: {avg_sec_per_game:.2f}")

        line = " ".join(parts)
        print(line, flush=True)


@dataclass
class OptimizationProgressReporter:
    """Specialized progress reporter for optimization runs (CMA-ES, GA).

    Tracks generations, candidates evaluated, fitness scores, and states processed.
    """

    total_generations: int
    candidates_per_generation: int
    report_interval_sec: float = 10.0

    # Internal tracking
    _start_time: float = field(default_factory=time.time, init=False)
    _last_report_time: float = field(default=0.0, init=False)
    _current_generation: int = field(default=0, init=False)
    _candidates_evaluated: int = field(default=0, init=False)
    _total_states_processed: int = field(default=0, init=False)
    _total_games_played: int = field(default=0, init=False)
    _best_fitness: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._start_time = time.time()
        self._last_report_time = self._start_time

    @property
    def total_candidates(self) -> int:
        return self.total_generations * self.candidates_per_generation

    def start_generation(self, generation: int) -> None:
        """Mark the start of a new generation."""
        self._current_generation = generation
        print(
            f"\n=== Generation {generation}/{self.total_generations} ===",
            flush=True,
        )

    def update_candidates_evaluated(self, completed: int) -> None:
        """Update the count of evaluated candidates (for distributed mode).

        In distributed mode, candidates are evaluated in parallel on remote
        workers, so we track progress by count rather than per-candidate.

        Parameters
        ----------
        completed : int
            Total number of candidates completed so far in this generation.
        """
        self._candidates_evaluated = completed
        now = time.time()
        elapsed_since_report = now - self._last_report_time

        if elapsed_since_report >= self.report_interval_sec:
            self._emit_report(completed)
            self._last_report_time = now

    def record_candidate(
        self,
        candidate_idx: int,
        fitness: float | None = None,
        states_processed: int = 0,
        games_played: int = 0,
        force_report: bool = False,
    ) -> None:
        """Record evaluation of a candidate and potentially emit a report.

        Parameters
        ----------
        candidate_idx : int
            Index of the candidate within the current generation (1-based).
        fitness : Optional[float]
            Fitness score of the candidate if available.
        states_processed : int
            Number of game states processed for this candidate.
        games_played : int
            Number of games played for this candidate.
        force_report : bool
            If True, emit a report regardless of time interval.
        """
        self._candidates_evaluated += 1
        self._total_states_processed += states_processed
        self._total_games_played += games_played

        if fitness is not None and (self._best_fitness is None or fitness > self._best_fitness):
            self._best_fitness = fitness

        now = time.time()
        elapsed_since_report = now - self._last_report_time

        if force_report or elapsed_since_report >= self.report_interval_sec:
            self._emit_report(candidate_idx)
            self._last_report_time = now

    def _emit_report(self, current_candidate_idx: int) -> None:
        """Emit a progress report with optimization-specific metrics."""
        now = time.time()
        elapsed = now - self._start_time

        total_candidates = self.total_candidates
        candidates_remaining = max(0, total_candidates - self._candidates_evaluated)

        # Calculate rates
        candidates_per_sec = (
            self._candidates_evaluated / elapsed if elapsed > 0 else 0.0
        )
        states_per_sec = (
            self._total_states_processed / elapsed if elapsed > 0 else 0.0
        )
        games_per_sec = self._total_games_played / elapsed if elapsed > 0 else 0.0

        # ETA
        eta_sec = (
            candidates_remaining / candidates_per_sec
            if candidates_per_sec > 0
            else 0.0
        )

        # Percentage (overall)
        overall_pct = (
            self._candidates_evaluated / total_candidates * 100
            if total_candidates > 0
            else 0.0
        )

        # Percentage (generation)
        gen_pct = (
            current_candidate_idx / self.candidates_per_generation * 100
            if self.candidates_per_generation > 0
            else 0.0
        )

        parts: list[str] = []

        parts.append(f"[Gen {self._current_generation}]")
        parts.append("PROGRESS:")
        parts.append(
            f"candidate {current_candidate_idx}/{self.candidates_per_generation} "
            f"({gen_pct:.0f}%)"
        )
        parts.append(
            f"| overall: {self._candidates_evaluated}/{total_candidates} "
            f"({overall_pct:.1f}%)"
        )
        parts.append(f"| elapsed: {_format_duration(elapsed)}")
        parts.append(f"| ETA: {_format_duration(eta_sec)}")
        if self._total_states_processed > 0:
            parts.append(f"| states/sec: {states_per_sec:.1f}")
        if self._total_games_played > 0:
            parts.append(f"| games/sec: {games_per_sec:.2f}")
        if self._best_fitness is not None:
            parts.append(f"| best fitness: {self._best_fitness:.4f}")

        line = " ".join(parts)
        print(line, flush=True)

    def finish_generation(
        self,
        mean_fitness: float,
        best_fitness: float,
        std_fitness: float,
    ) -> None:
        """Emit a generation summary."""
        print(
            f"[Gen {self._current_generation}] SUMMARY: "
            f"mean={mean_fitness:.4f}, std={std_fitness:.4f}, best={best_fitness:.4f}",
            flush=True,
        )

    def finish(self) -> None:
        """Emit a final summary report."""
        now = time.time()
        elapsed = now - self._start_time

        candidates_per_sec = (
            self._candidates_evaluated / elapsed if elapsed > 0 else 0.0
        )
        states_per_sec = (
            self._total_states_processed / elapsed if elapsed > 0 else 0.0
        )
        games_per_sec = self._total_games_played / elapsed if elapsed > 0 else 0.0

        parts: list[str] = []

        parts.append("OPTIMIZATION COMPLETED:")
        parts.append(f"{self._current_generation} generations")
        parts.append(f"| {self._candidates_evaluated} candidates evaluated")
        parts.append(f"| total time: {_format_duration(elapsed)}")
        if self._total_states_processed > 0:
            parts.append(f"| total states: {self._total_states_processed}")
            parts.append(f"| avg states/sec: {states_per_sec:.1f}")
        if self._total_games_played > 0:
            parts.append(f"| total games: {self._total_games_played}")
            parts.append(f"| avg games/sec: {games_per_sec:.2f}")
        parts.append(f"| avg candidates/sec: {candidates_per_sec:.3f}")
        if self._best_fitness is not None:
            parts.append(f"| best fitness: {self._best_fitness:.4f}")

        line = " ".join(parts)
        print(line, flush=True)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes}m"
