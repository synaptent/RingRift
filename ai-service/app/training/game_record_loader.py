"""
JSONL loader and lightweight adapters for backend-exported GameRecords.

This module is intentionally offline-only: it provides helpers for reading
canonical GameRecord JSONL exports (one game per line) and mapping them into
Python models suitable for training, evaluation, and analysis tooling.

It does not modify training loops or runtime services; callers are expected
to import these helpers from ad-hoc scripts or notebooks.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from app.models import BoardType
from app.models.game_record import (
    FinalScore,
    GameOutcome,
    GameRecord,
    MoveRecord,
)

logger = logging.getLogger(__name__)


def _normalize_path(path: Path | str) -> Path:
    """Return *path* as a Path instance."""
    if isinstance(path, Path):
        return path
    return Path(path)


def iter_game_records_from_jsonl(
    path: Path | str,
    *,
    skip_invalid: bool = False,
) -> Iterator[GameRecord]:
    """
    Stream :class:`GameRecord` objects from a JSONL file.

    Parameters
    ----------
    path:
        Path to a JSONL file produced by the backend exporter, where each
        non-empty line contains a single GameRecord JSON object matching
        :mod:`GAME_RECORD_SPEC`.
    skip_invalid:
        When ``False`` (default), any parse or validation error raises a
        :class:`ValueError` that includes the file path and 1-based line
        number. When ``True``, malformed lines are logged at WARNING level
        and skipped.

    Yields
    ------
    GameRecord
        Parsed and validated records in file order.
    """
    jsonl_path = _normalize_path(path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                # Skip empty or whitespace-only lines to match other JSONL
                # helpers in the training stack.
                continue

            try:
                # Prefer the canonical GameRecord JSONL helper, which mirrors
                # the shared TypeScript implementation.
                record = GameRecord.from_jsonl_line(line)
            except Exception as exc:  # pragma: no cover
                # Exact exception type is not critical here; preserve context.
                msg = (
                    f"Failed to parse GameRecord from {jsonl_path} at line "
                    f"{line_no}: {exc}"
                )
                if skip_invalid:
                    logger.warning(msg)
                    continue
                raise ValueError(msg) from exc

            yield record


def load_game_records(
    path: Path | str,
    limit: int | None = None,
    *,
    skip_invalid: bool = False,
) -> list[GameRecord]:
    """
    Eagerly load :class:`GameRecord` objects from a JSONL file.

    This is a small convenience wrapper around
    :func:`iter_game_records_from_jsonl`.

    Parameters
    ----------
    path:
        JSONL file path containing one GameRecord per non-empty line.
    limit:
        Optional maximum number of records to load. When ``None`` all
        available records are loaded. When > 0, iteration stops once
        ``limit`` records have been collected.
    skip_invalid:
        Passed through to :func:`iter_game_records_from_jsonl`.

    Returns
    -------
    list[GameRecord]
        List of parsed records, in file order (possibly truncated by
        ``limit``).
    """
    records: list[GameRecord] = []
    if limit is not None and limit <= 0:
        return records

    for record in iter_game_records_from_jsonl(
        path,
        skip_invalid=skip_invalid,
    ):
        records.append(record)
        if limit is not None and limit > 0 and len(records) >= limit:
            break

    return records


@dataclass(frozen=True)
class RecordedEpisode:
    """
    Lightweight, env-agnostic view of a completed GameRecord.

    This bridge type exposes just the fields typically consumed in training
    and evaluation code without depending on :class:`RingRiftEnv` or
    :class:`GameState`. It is intentionally minimal; future slices can
    extend it if additional metadata becomes useful.

    Attributes
    ----------
    record:
        The underlying :class:`GameRecord` instance.
    board_type:
        Board geometry for the game.
    num_players:
        Number of active players in the game.
    winner:
        Winning player number, or ``None`` for draws / unfinished games.
    outcome:
        High-level outcome category, mirroring the shared schema.
    final_score:
        Final ring/territory/rings-remaining breakdown.
    moves:
        Sequence of stored :class:`MoveRecord` entries.
    num_moves:
        Total number of moves in the game, as recorded in ``total_moves``.
    """

    record: GameRecord
    board_type: BoardType
    num_players: int
    winner: int | None
    outcome: GameOutcome
    final_score: FinalScore
    moves: list[MoveRecord]
    num_moves: int


def game_record_to_recorded_episode(record: GameRecord) -> RecordedEpisode:
    """
    Convert a :class:`GameRecord` into a :class:`RecordedEpisode`.

    This does not attempt to reconstruct a step-by-step environment
    trajectory; instead it exposes the stored summary and move list in a
    form that is easy to consume from offline tooling.
    """
    return RecordedEpisode(
        record=record,
        board_type=record.board_type,
        num_players=record.num_players,
        winner=record.winner,
        outcome=record.outcome,
        final_score=record.final_score,
        moves=list(record.moves),
        num_moves=record.total_moves,
    )


def iter_recorded_episodes_from_jsonl(
    path: Path | str,
    *,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> Iterator[RecordedEpisode]:
    """
    Stream :class:`RecordedEpisode` objects directly from a GameRecord JSONL.

    This helper is a thin composition of
    :func:`iter_game_records_from_jsonl` and
    :func:`game_record_to_recorded_episode`.
    """
    if limit is not None and limit <= 0:
        return
        yield  # pragma: no cover

    count = 0
    for record in iter_game_records_from_jsonl(
        path,
        skip_invalid=skip_invalid,
    ):
        yield game_record_to_recorded_episode(record)
        count += 1
        if limit is not None and count >= limit:
            break


def _summarize_records(records: Iterable[GameRecord]) -> dict[str, Counter]:
    """
    Compute basic distribution summaries over a collection of GameRecords.

    Returns a mapping with keys ``"board_types"``, ``"outcomes"``, and
    ``"winners"``, each containing a :class:`collections.Counter`.
    """
    board_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()

    for record in records:
        board_counts[record.board_type.value] += 1
        outcome_counts[record.outcome.value] += 1
        if record.winner is None:
            winner_counts["draw_or_unknown"] += 1
        else:
            winner_counts[f"player_{record.winner}"] += 1

    return {
        "board_types": board_counts,
        "outcomes": outcome_counts,
        "winners": winner_counts,
    }


def main(argv: list[str] | None = None) -> int:
    """
    Minimal CLI for manual inspection of GameRecord JSONL files.

    Usage (from the ``ai-service`` project root)::

        python -m app.training.game_record_loader --input path/to/file.jsonl

    This CLI is intended for ad-hoc, offline exploration only. It is not
    wired into training or evaluation pipelines.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect backend-exported GameRecord JSONL files.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the GameRecord JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to load.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Log and skip invalid lines instead of failing fast.",
    )

    args = parser.parse_args(argv)

    records = load_game_records(
        args.input,
        limit=args.limit,
        skip_invalid=args.skip_invalid,
    )

    if not records:
        print(f"No GameRecord entries found in {args.input!r}")
        return 0

    summary = _summarize_records(records)

    print(f"Loaded {len(records)} GameRecord(s) from {args.input!r}")

    print("\nBoard types:")
    for board, count in sorted(summary["board_types"].items()):
        print(f"  {board}: {count}")

    print("\nOutcomes:")
    for outcome, count in sorted(summary["outcomes"].items()):
        print(f"  {outcome}: {count}")

    print("\nWinner summary:")
    for winner, count in sorted(summary["winners"].items()):
        print(f"  {winner}: {count}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
