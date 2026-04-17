"""Tests for the move-level ensemble voting helper (C1 / plan #77)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.ai.ensemble_move import (
    EnsembleFailure,
    EnsembleVoteResult,
    _repr_move,
    select_move_ensemble,
)


def _move(move_type: str = "place", x: int = 0, y: int = 0):
    """Build a mock Move-like SimpleNamespace for voting tests."""
    return SimpleNamespace(
        type=SimpleNamespace(value=move_type),
        from_pos=None,
        to=SimpleNamespace(x=x, y=y),
    )


class _StaticAI:
    """Always returns the same move."""

    def __init__(self, move):
        self._move = move

    def select_move(self, _state):
        return self._move


class _FailingAI:
    """Raises a given exception when asked for a move."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    def select_move(self, _state):
        raise self._exc


class _SlowAI:
    """Sleeps then returns a move — used to exercise the timeout path."""

    def __init__(self, move, seconds: float):
        self._move = move
        self._seconds = seconds

    def select_move(self, _state):
        import time
        time.sleep(self._seconds)
        return self._move


class TestMoveRepr:
    def test_place_move_key_is_stable(self):
        m = _move("place", 3, 4)
        assert _repr_move(m) == "place_3,4"

    def test_none_move(self):
        assert _repr_move(None) == "<none>"

    def test_move_with_from_pos(self):
        m = SimpleNamespace(
            type=SimpleNamespace(value="move_stack"),
            from_pos=SimpleNamespace(x=1, y=2),
            to=SimpleNamespace(x=3, y=4),
        )
        assert _repr_move(m) == "move_stack_1,2_3,4"


class TestSelectMoveEnsemble:
    @pytest.mark.asyncio
    async def test_unanimous_vote_picks_the_move(self):
        m = _move("place", 0, 0)
        ais = [_StaticAI(m), _StaticAI(m), _StaticAI(m)]
        result = await select_move_ensemble(ais, game_state=None, timeout=5.0)
        assert result.move is m
        assert result.ensemble_size == 3
        assert result.agreement_count == 3
        assert result.agreement_fraction == 1.0
        assert result.failures == 0

    @pytest.mark.asyncio
    async def test_majority_vote_wins(self):
        winner = _move("place", 5, 5)
        loser = _move("place", 7, 7)
        ais = [_StaticAI(winner), _StaticAI(winner), _StaticAI(loser)]
        result = await select_move_ensemble(ais, None, timeout=5.0)
        assert result.move is winner
        assert result.agreement_count == 2
        assert result.failures == 0

    @pytest.mark.asyncio
    async def test_tie_broken_by_lowest_index(self):
        """Callers put the canonical / best-known model first, so a tie
        must resolve to that model's pick."""
        canonical = _move("place", 1, 1)
        challenger = _move("place", 2, 2)
        ais = [_StaticAI(canonical), _StaticAI(challenger)]
        result = await select_move_ensemble(ais, None, timeout=5.0)
        assert result.move is canonical

    @pytest.mark.asyncio
    async def test_failing_constituent_dropped_but_vote_succeeds(self):
        m = _move("place", 0, 0)
        ais = [
            _StaticAI(m),
            _FailingAI(RuntimeError("boom")),
            _StaticAI(m),
        ]
        result = await select_move_ensemble(ais, None, timeout=5.0)
        assert result.move is m
        assert result.ensemble_size == 2  # only successful ones counted
        assert result.failures == 1
        # individual_move_repr preserves ordering including failures
        assert result.individual_move_repr[1].startswith("<error:")

    @pytest.mark.asyncio
    async def test_all_failing_raises_ensemble_failure(self):
        ais = [
            _FailingAI(RuntimeError("a")),
            _FailingAI(ValueError("b")),
        ]
        with pytest.raises(EnsembleFailure) as exc_info:
            await select_move_ensemble(ais, None, timeout=5.0)
        assert "all 2 ensemble constituents failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_ais_raises_ensemble_failure(self):
        with pytest.raises(EnsembleFailure, match="empty ais list"):
            await select_move_ensemble([], None, timeout=5.0)

    @pytest.mark.asyncio
    async def test_timeout_raises_ensemble_failure(self):
        slow = _SlowAI(_move("place", 0, 0), seconds=1.0)
        with pytest.raises(EnsembleFailure, match="timed out"):
            await select_move_ensemble([slow, slow], None, timeout=0.05)

    @pytest.mark.asyncio
    async def test_concurrency_latency_is_max_not_sum(self):
        """With 3 constituents each sleeping 0.3s, total wall-clock
        should be ~0.3s not ~0.9s.  Proves asyncio.gather runs them in
        parallel rather than serially."""
        import time as _time
        slow = _SlowAI(_move("place", 0, 0), seconds=0.3)
        ais = [slow, slow, slow]
        t0 = _time.monotonic()
        result = await select_move_ensemble(ais, None, timeout=5.0)
        elapsed = _time.monotonic() - t0
        assert result.ensemble_size == 3
        # Generous upper bound: 3× serial would be ~0.9s; parallel ~0.3s
        # + overhead.  Fail only if serial (> 0.7s).
        assert elapsed < 0.7, f"ensemble ran serially ({elapsed:.2f}s)"

    @pytest.mark.asyncio
    async def test_vote_result_shape(self):
        m = _move("place", 2, 3)
        result = await select_move_ensemble([_StaticAI(m)], None, timeout=5.0)
        assert isinstance(result, EnsembleVoteResult)
        assert result.ensemble_size == 1
        assert result.agreement_count == 1
        assert result.failures == 0
        assert result.individual_move_repr == ["place_2,3"]
