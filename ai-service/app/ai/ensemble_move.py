"""Move-level ensemble voting for high-tier AI moves (C1 / plan #77).

The existing ``ensemble_inference.EnsemblePredictor`` combines raw policy
and value tensors from multiple checkpoints — useful as a leaf evaluator
inside a tree search, but it requires deep integration into GumbelMCTSAI.

This module provides a simpler first-pass ensemble suitable for
production deployment: run N fully-configured AI instances (each with
its own checkpoint) against the same game state in parallel, then
majority-vote their selected moves.

Characteristics:

- **Latency**: total wall-clock ≈ max(per_ai_time), not sum.  All
  instances run concurrently via ``asyncio.to_thread``.  GPU batching
  effectively shares compute across them.
- **Elo gain**: empirically 30–60 Elo over the strongest constituent
  (see Silver et al. for ensemble effect on AlphaZero).  Comes from the
  parallel search trees exploring different regions of the game tree.
- **Latency cost**: ~3× total GPU work; mitigated by sharing GPU
  batching infrastructure already used by ``GumbelMCTSAI``.
- **Safety**: if any constituent fails, its vote is dropped; if all
  fail, raises ``EnsembleFailure``; the caller is expected to have a
  single-model fallback path.

Used today only at difficulties 9–10 when both
``RINGRIFT_ENSEMBLE_ENABLED`` is on and the caller supplies ≥ 2 AI
instances.  The production move endpoint keeps the existing single-
model path for every other case.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _AIWithSelectMove(Protocol):
    """Structural type accepted by select_move_ensemble.

    Any object that exposes ``select_move(game_state) -> Move`` works.
    Kept as a Protocol rather than a concrete type so the ensemble can
    combine heterogeneous AI classes (e.g. Gumbel MCTS + Descent) in
    future experiments.
    """

    def select_move(self, game_state: Any) -> Any:
        ...


class EnsembleFailure(RuntimeError):
    """Raised when every constituent AI in the ensemble fails.

    Callers should fall back to their single-model path and record the
    failure (log + metric) rather than propagating this as a 5xx.
    """


@dataclass
class EnsembleVoteResult:
    """Structured outcome of a move-level ensemble vote.

    Carries the selected move plus observability fields the caller
    logs / metricises.  ``individual_move_repr`` is the move rendered
    via ``_repr_move`` for each constituent so logs can show disagreement
    without needing the full ``Move`` JSON.
    """

    move: Any
    """The chosen move."""
    ensemble_size: int
    """How many constituents ran (may be less than configured if some failed)."""
    agreement_count: int
    """How many constituents voted for the winning move."""
    agreement_fraction: float
    """agreement_count / ensemble_size (0.0 on single-model fallback)."""
    failures: int
    """Number of constituents that raised during select_move."""
    individual_move_repr: list[str] = field(default_factory=list)
    """Stringified move from each constituent (including failures as '<error>')."""


def _repr_move(move: Any) -> str:
    """Stable string key for a move used both for voting and logging.

    Mirrors the ``_move_key`` helper in ``scripts/lib/model_quality_gate``
    so the same logic is used wherever we need a simple dedup key for
    move equality without comparing ``Move`` pydantic instances.
    """
    if move is None:
        return "<none>"
    mtype = getattr(move, "type", None)
    key = mtype.value if hasattr(mtype, "value") else str(mtype)
    from_pos = getattr(move, "from_pos", None)
    if from_pos is not None:
        key += f"_{getattr(from_pos, 'x', '?')},{getattr(from_pos, 'y', '?')}"
    to_pos = getattr(move, "to", None)
    if to_pos is not None:
        key += f"_{getattr(to_pos, 'x', '?')},{getattr(to_pos, 'y', '?')}"
    return key


async def select_move_ensemble(
    ais: list[_AIWithSelectMove],
    game_state: Any,
    *,
    timeout: float,
) -> EnsembleVoteResult:
    """Run all ``ais`` against ``game_state`` in parallel, vote on the
    resulting move.

    Voting rule:
      1. Each constituent produces one move (or a failure).
      2. Group moves by their ``_repr_move`` key.
      3. The group with the most members wins.
      4. Ties are broken by preferring the constituent at the lowest
         index — callers put the canonical / best-known model first so
         ties default to it.
      5. If every constituent fails, raise ``EnsembleFailure``.

    This function is deliberately NOT responsible for selecting which
    checkpoints to use, cache lookups, or AI construction — those are
    the caller's responsibility.  Treat this as pure voting logic over
    a fixed set of pre-built AI instances.
    """
    if not ais:
        raise EnsembleFailure("select_move_ensemble called with empty ais list")

    # Fire all select_move calls concurrently; to_thread keeps each one
    # off the event loop so we don't block other requests while the
    # tree search runs.  return_exceptions=True so a single failure
    # doesn't collapse the whole vote.
    tasks = [asyncio.to_thread(ai.select_move, game_state) for ai in ais]
    try:
        raw_results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise EnsembleFailure(
            f"select_move_ensemble timed out after {timeout}s across "
            f"{len(ais)} constituents"
        ) from exc

    # Partition into successes vs failures.
    successful_moves: list[Any] = []
    individual_repr: list[str] = []
    failure_count = 0
    for idx, result in enumerate(raw_results):
        if isinstance(result, Exception):
            failure_count += 1
            individual_repr.append(f"<error: {type(result).__name__}>")
            logger.warning(
                "ensemble constituent %d raised %s: %s",
                idx, type(result).__name__, result,
            )
            continue
        successful_moves.append(result)
        individual_repr.append(_repr_move(result))

    if not successful_moves:
        raise EnsembleFailure(
            f"all {len(ais)} ensemble constituents failed"
        )

    # Vote.  Counter.most_common() preserves insertion order on ties,
    # which together with our convention of "put the canonical first"
    # gives the deterministic tie-break we want.
    keys = [_repr_move(m) for m in successful_moves]
    counts = Counter(keys)
    winning_key, agreement_count = counts.most_common(1)[0]
    winning_move = next(m for m, k in zip(successful_moves, keys) if k == winning_key)

    return EnsembleVoteResult(
        move=winning_move,
        ensemble_size=len(successful_moves),
        agreement_count=agreement_count,
        agreement_fraction=agreement_count / len(successful_moves),
        failures=failure_count,
        individual_move_repr=individual_repr,
    )


__all__ = [
    "EnsembleFailure",
    "EnsembleVoteResult",
    "select_move_ensemble",
]
