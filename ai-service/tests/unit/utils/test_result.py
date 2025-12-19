"""Tests for the Result pattern utility."""

import pytest

from app.utils.result import (
    Ok,
    Err,
    Result,
    ResultError,
    result_from_exception,
    collect_results,
    partition_results,
    OperationResult,
)


class TestOk:
    """Tests for Ok type."""

    def test_is_ok(self):
        result = Ok(42)
        assert result.is_ok is True
        assert result.is_err is False

    def test_value(self):
        result = Ok(42)
        assert result.value == 42

    def test_unwrap(self):
        result = Ok(42)
        assert result.unwrap() == 42

    def test_unwrap_or(self):
        result = Ok(42)
        assert result.unwrap_or(0) == 42

    def test_unwrap_or_else(self):
        result = Ok(42)
        assert result.unwrap_or_else(lambda: 0) == 42

    def test_expect(self):
        result = Ok(42)
        assert result.expect("should have value") == 42

    def test_map(self):
        result = Ok(42)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok
        assert mapped.unwrap() == 84

    def test_map_err(self):
        result: Result[int, str] = Ok(42)
        mapped = result.map_err(lambda e: e.upper())
        assert mapped.is_ok
        assert mapped.unwrap() == 42

    def test_and_then(self):
        result = Ok(42)
        chained = result.and_then(lambda x: Ok(x * 2))
        assert chained.is_ok
        assert chained.unwrap() == 84

    def test_and_then_error(self):
        result = Ok(42)
        chained = result.and_then(lambda x: Err("failed"))
        assert chained.is_err
        assert chained.error == "failed"

    def test_or_else(self):
        result: Result[int, str] = Ok(42)
        fallback = result.or_else(lambda e: Ok(0))
        assert fallback.is_ok
        assert fallback.unwrap() == 42

    def test_iter(self):
        result = Ok(42)
        values = list(result)
        assert values == [42]

    def test_bool(self):
        result = Ok(42)
        assert bool(result) is True

    def test_repr(self):
        result = Ok(42)
        assert repr(result) == "Ok(42)"


class TestErr:
    """Tests for Err type."""

    def test_is_err(self):
        result = Err("error")
        assert result.is_ok is False
        assert result.is_err is True

    def test_error(self):
        result = Err("error")
        assert result.error == "error"

    def test_value_is_none(self):
        result = Err("error")
        assert result.value is None

    def test_unwrap_raises(self):
        result = Err("error")
        with pytest.raises(ResultError) as exc_info:
            result.unwrap()
        assert "error" in str(exc_info.value)

    def test_unwrap_or(self):
        result = Err("error")
        assert result.unwrap_or(42) == 42

    def test_unwrap_or_else(self):
        result = Err("error")
        assert result.unwrap_or_else(lambda: 42) == 42

    def test_expect_raises(self):
        result = Err("error")
        with pytest.raises(ResultError) as exc_info:
            result.expect("custom message")
        assert "custom message" in str(exc_info.value)
        assert "error" in str(exc_info.value)

    def test_map(self):
        result: Result[int, str] = Err("error")
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err
        assert mapped.error == "error"

    def test_map_err(self):
        result = Err("error")
        mapped = result.map_err(lambda e: e.upper())
        assert mapped.is_err
        assert mapped.error == "ERROR"

    def test_and_then(self):
        result: Result[int, str] = Err("error")
        chained = result.and_then(lambda x: Ok(x * 2))
        assert chained.is_err
        assert chained.error == "error"

    def test_or_else(self):
        result = Err("error")
        fallback = result.or_else(lambda e: Ok(42))
        assert fallback.is_ok
        assert fallback.unwrap() == 42

    def test_iter(self):
        result = Err("error")
        values = list(result)
        assert values == []

    def test_bool(self):
        result = Err("error")
        assert bool(result) is False

    def test_repr(self):
        result = Err("error")
        assert repr(result) == "Err('error')"


class TestResultFromException:
    """Tests for result_from_exception helper."""

    def test_success(self):
        result = result_from_exception(int, "42")
        assert result.is_ok
        assert result.unwrap() == 42

    def test_failure(self):
        result = result_from_exception(int, "not a number")
        assert result.is_err
        assert "invalid literal" in result.error

    def test_custom_exception_type(self):
        def raises_value_error():
            raise ValueError("test")

        result = result_from_exception(raises_value_error, catch=ValueError)
        assert result.is_err
        assert result.error == "test"


class TestCollectResults:
    """Tests for collect_results helper."""

    def test_all_ok(self):
        results = [Ok(1), Ok(2), Ok(3)]
        collected = collect_results(results)
        assert collected.is_ok
        assert collected.unwrap() == [1, 2, 3]

    def test_with_err(self):
        results = [Ok(1), Err("fail"), Ok(3)]
        collected = collect_results(results)
        assert collected.is_err
        assert collected.error == "fail"

    def test_empty(self):
        results = []
        collected = collect_results(results)
        assert collected.is_ok
        assert collected.unwrap() == []

    def test_first_err_returned(self):
        results = [Ok(1), Err("first"), Err("second")]
        collected = collect_results(results)
        assert collected.error == "first"


class TestPartitionResults:
    """Tests for partition_results helper."""

    def test_mixed_results(self):
        results = [Ok(1), Err("a"), Ok(2), Err("b")]
        oks, errs = partition_results(results)
        assert oks == [1, 2]
        assert errs == ["a", "b"]

    def test_all_ok(self):
        results = [Ok(1), Ok(2)]
        oks, errs = partition_results(results)
        assert oks == [1, 2]
        assert errs == []

    def test_all_err(self):
        results = [Err("a"), Err("b")]
        oks, errs = partition_results(results)
        assert oks == []
        assert errs == ["a", "b"]

    def test_empty(self):
        results = []
        oks, errs = partition_results(results)
        assert oks == []
        assert errs == []


class TestOperationResult:
    """Tests for OperationResult dataclass."""

    def test_ok_factory(self):
        result = OperationResult.ok(42, extra="data")
        assert result.success is True
        assert result.value == 42
        assert result.details == {"extra": "data"}

    def test_fail_factory(self):
        result = OperationResult.fail("something went wrong", code=500)
        assert result.success is False
        assert result.error == "something went wrong"
        assert result.details == {"code": 500}

    def test_is_ok_property(self):
        ok_result = OperationResult.ok(42)
        fail_result = OperationResult.fail("error")
        assert ok_result.is_ok is True
        assert fail_result.is_ok is False

    def test_is_err_property(self):
        ok_result = OperationResult.ok(42)
        fail_result = OperationResult.fail("error")
        assert ok_result.is_err is False
        assert fail_result.is_err is True

    def test_to_result(self):
        ok_result = OperationResult.ok(42)
        converted = ok_result.to_result()
        assert converted.is_ok
        assert converted.unwrap() == 42

        fail_result = OperationResult.fail("error")
        converted = fail_result.to_result()
        assert converted.is_err
        assert converted.error == "error"

    def test_to_dict(self):
        result = OperationResult.ok(42, extra="data")
        result.duration_ms = 100.5
        d = result.to_dict()
        assert d["success"] is True
        assert d["value"] == 42
        assert d["details"] == {"extra": "data"}
        assert d["duration_ms"] == 100.5
