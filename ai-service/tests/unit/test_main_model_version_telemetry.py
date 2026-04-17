"""Tests for D1 (plan #81): model-version telemetry on /ai/move.

Covers:
- _extract_model_version helper robustness across neural-net shapes
- MoveResponse exposes nn_model_version field
- AI_MOVES_BY_MODEL_VERSION counter is registered with expected labels

Intentionally lightweight: does not spin up a real AI instance. End-to-end
telemetry is exercised in integration tests when a model is actually loaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from prometheus_client import CollectorRegistry


class TestExtractModelVersion:
    """Verify _extract_model_version handles all expected shapes without raising."""

    def _import(self):
        from app.main import _extract_model_version
        return _extract_model_version

    def test_returns_none_when_no_neural_net(self):
        extract = self._import()
        ai = SimpleNamespace()  # no neural_net attribute
        assert extract(ai) is None

    def test_returns_none_when_neural_net_is_none(self):
        extract = self._import()
        ai = SimpleNamespace(neural_net=None)
        assert extract(ai) is None

    def test_returns_none_when_model_is_none(self):
        extract = self._import()
        ai = SimpleNamespace(neural_net=SimpleNamespace(model=None))
        assert extract(ai) is None

    def test_returns_explicit_architecture_version(self):
        """Models with ARCHITECTURE_VERSION attribute surface it directly."""
        extract = self._import()
        model = SimpleNamespace()
        # Deliberately bypass SimpleNamespace's attribute semantics to set a
        # class-level-like attribute via a real object.
        model.ARCHITECTURE_VERSION = "v4.0.0"
        ai = SimpleNamespace(neural_net=SimpleNamespace(model=model))
        assert extract(ai) == "v4.0.0"

    def test_falls_back_to_registry(self, monkeypatch):
        """Models without ARCHITECTURE_VERSION use the versioning registry."""
        extract = self._import()

        fake_model = SimpleNamespace()  # no ARCHITECTURE_VERSION
        ai = SimpleNamespace(neural_net=SimpleNamespace(model=fake_model))

        captured: list = []

        def fake_get_model_version(model):
            captured.append(model)
            return "v5-heavy"

        # Patch the import inside _extract_model_version.
        import app.training.model_versioning as mv
        monkeypatch.setattr(mv, "get_model_version", fake_get_model_version)

        assert extract(ai) == "v5-heavy"
        assert captured == [fake_model]

    def test_handles_exceptions_gracefully(self, monkeypatch):
        """Any exception in the version lookup returns None, never raises."""
        extract = self._import()
        model = MagicMock()
        # Access to ARCHITECTURE_VERSION raises AttributeError mid-read;
        # fallback path should also not raise.
        type(model).ARCHITECTURE_VERSION = property(
            lambda self: (_ for _ in ()).throw(AttributeError("boom"))
        )
        ai = SimpleNamespace(neural_net=SimpleNamespace(model=model))

        # Also make the registry path fail.
        import app.training.model_versioning as mv
        def boom(model):
            raise RuntimeError("registry unavailable")
        monkeypatch.setattr(mv, "get_model_version", boom)

        # Should not propagate — returns None.
        assert extract(ai) is None

    def test_ignores_empty_version_string(self):
        """An empty ARCHITECTURE_VERSION falls through to the registry path."""
        extract = self._import()
        model = SimpleNamespace()
        model.ARCHITECTURE_VERSION = ""
        ai = SimpleNamespace(neural_net=SimpleNamespace(model=model))
        # Without a registry entry for a SimpleNamespace class, falls back
        # to the registry's default "v0.0.0".
        result = extract(ai)
        # Acceptable: either None (if registry path fails) or the
        # registry's default string. What we really care about is: we do
        # NOT surface the empty string.
        assert result != ""


class TestMoveResponseSchema:
    """Verify the response model exposes the new nn_model_version field."""

    def test_nn_model_version_field_is_optional(self):
        from app.main import MoveResponse
        # Build a minimal response without the field — must succeed
        resp = MoveResponse(
            move=None,
            evaluation=0.0,
            thinking_time_ms=1,
            ai_type="random",
            difficulty=1,
        )
        assert resp.nn_model_version is None

    def test_nn_model_version_round_trips(self):
        from app.main import MoveResponse
        resp = MoveResponse(
            move=None,
            evaluation=0.0,
            thinking_time_ms=1,
            ai_type="gumbel_mcts",
            difficulty=10,
            nn_model_version="v4.0.0",
        )
        assert resp.nn_model_version == "v4.0.0"
        dumped = resp.model_dump()
        assert dumped["nn_model_version"] == "v4.0.0"


class TestAIMovesByModelVersionCounter:
    """Verify the Prometheus counter is registered with the expected shape."""

    def test_counter_has_expected_labels(self):
        from app.metrics_base import AI_MOVES_BY_MODEL_VERSION
        # Label order matters — downstream dashboards rely on it.
        assert AI_MOVES_BY_MODEL_VERSION._labelnames == (
            "model_version",
            "ai_type",
            "difficulty",
        )

    def test_counter_increments(self):
        """Incrementing should produce samples on scrape."""
        from app.metrics_base import AI_MOVES_BY_MODEL_VERSION
        sample = AI_MOVES_BY_MODEL_VERSION.labels("v4.0.0", "gumbel_mcts", "10")
        before = sample._value.get()
        sample.inc()
        after = sample._value.get()
        assert after == before + 1.0

    def test_none_model_version_is_accepted(self):
        """The 'none' label is used for non-neural AI moves."""
        from app.metrics_base import AI_MOVES_BY_MODEL_VERSION
        sample = AI_MOVES_BY_MODEL_VERSION.labels("none", "heuristic", "2")
        sample.inc()  # must not raise
