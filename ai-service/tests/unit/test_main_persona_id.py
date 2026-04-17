"""Tests for C2 (plan #80): persona_id on /ai/move.

Covers the server-side layer only:

- MoveRequest accepts persona_id and validates against the 4 known names
- _resolve_persona_profile_id respects the RINGRIFT_PERSONAS_ENABLED flag
- MoveResponse exposes persona_id

TS-side propagation is covered in tests/unit/aiPersona*.test.ts (phase 2)
and the end-to-end request test lives in an integration suite.
"""

from __future__ import annotations

import pytest


class TestMoveRequestAllowedPersonas:
    """The ALLOWED_PERSONA_IDS set is the single source of truth for which
    persona names the API accepts.  Lock it in.

    We validate the set + the validator behaviour without building a full
    GameState — the GameState model has many required fields that don't
    affect persona validation.  The _validate_persona_id model_validator
    runs unconditionally, so we test it through the MoveRequest class's
    validators directly.
    """

    def test_allowed_persona_ids_matches_spec(self):
        from app.main import _ALLOWED_PERSONA_IDS
        assert _ALLOWED_PERSONA_IDS == frozenset({
            "balanced", "aggressive", "territorial", "defensive",
        })

    @pytest.mark.parametrize(
        "persona",
        ["balanced", "aggressive", "territorial", "defensive"],
    )
    def test_valid_persona_is_in_allowed_set(self, persona):
        from app.main import _ALLOWED_PERSONA_IDS
        assert persona in _ALLOWED_PERSONA_IDS

    @pytest.mark.parametrize(
        "bad",
        ["unknown", "BALANCED", "", " aggressive", "heuristic_v1_balanced"],
    )
    def test_bad_persona_rejected_by_set(self, bad):
        from app.main import _ALLOWED_PERSONA_IDS
        assert bad not in _ALLOWED_PERSONA_IDS

    def test_persona_validator_rejects_unknown(self):
        """Exercise the actual model_validator via a MoveRequest stub that
        skips the player_number validator. Uses model_construct to bypass
        validators entirely, then invokes validate_persona_id on the
        constructed instance."""
        from app.main import MoveRequest
        # Construct-no-validate, then call the validator.
        instance = MoveRequest.model_construct(persona_id="unknown")
        with pytest.raises(ValueError, match="persona_id"):
            instance.validate_persona_id()

    def test_persona_validator_allows_none(self):
        from app.main import MoveRequest
        instance = MoveRequest.model_construct(persona_id=None)
        # Should return self without raising.
        result = instance.validate_persona_id()
        assert result is instance

    @pytest.mark.parametrize(
        "persona",
        ["balanced", "aggressive", "territorial", "defensive"],
    )
    def test_persona_validator_allows_valid_names(self, persona):
        from app.main import MoveRequest
        instance = MoveRequest.model_construct(persona_id=persona)
        result = instance.validate_persona_id()
        assert result is instance
        assert result.persona_id == persona


class TestPersonasFeatureFlag:
    """_personas_feature_enabled respects RINGRIFT_PERSONAS_ENABLED."""

    @pytest.mark.parametrize("raw", ["1", "true", "True", "YES", "on"])
    def test_truthy_values_enable_flag(self, raw, monkeypatch):
        from app import main
        monkeypatch.setenv("RINGRIFT_PERSONAS_ENABLED", raw)
        assert main._personas_feature_enabled() is True

    @pytest.mark.parametrize(
        "raw",
        ["", "0", "false", "no", "off", "maybe", "unset"],
    )
    def test_falsy_values_disable_flag(self, raw, monkeypatch):
        from app import main
        if raw == "unset":
            monkeypatch.delenv("RINGRIFT_PERSONAS_ENABLED", raising=False)
        else:
            monkeypatch.setenv("RINGRIFT_PERSONAS_ENABLED", raw)
        assert main._personas_feature_enabled() is False


class TestResolvePersonaProfileId:
    """_resolve_persona_profile_id is the single point where a persona
    request becomes a heuristic profile id. Verify the mapping."""

    def test_returns_none_when_no_persona(self, monkeypatch):
        from app import main
        monkeypatch.setenv("RINGRIFT_PERSONAS_ENABLED", "true")
        assert main._resolve_persona_profile_id(None) is None

    def test_returns_none_when_flag_off(self, monkeypatch):
        from app import main
        monkeypatch.setenv("RINGRIFT_PERSONAS_ENABLED", "false")
        # Even with a valid persona, the flag gates the resolution.
        assert main._resolve_persona_profile_id("aggressive") is None

    @pytest.mark.parametrize(
        "persona,expected",
        [
            ("balanced", "heuristic_v1_balanced"),
            ("aggressive", "heuristic_v1_aggressive"),
            ("territorial", "heuristic_v1_territorial"),
            ("defensive", "heuristic_v1_defensive"),
        ],
    )
    def test_maps_each_persona_to_heuristic_v1_id(self, persona, expected, monkeypatch):
        from app import main
        monkeypatch.setenv("RINGRIFT_PERSONAS_ENABLED", "true")
        assert main._resolve_persona_profile_id(persona) == expected

    def test_unknown_persona_returns_none_defensively(self, monkeypatch):
        """Even with the flag on, a persona that slipped past validation
        should fall back to None rather than produce an invalid profile id."""
        from app import main
        monkeypatch.setenv("RINGRIFT_PERSONAS_ENABLED", "true")
        assert main._resolve_persona_profile_id("unknown") is None


class TestMoveResponsePersonaField:
    """MoveResponse exposes persona_id; optional and defaults to None."""

    def test_default_is_none(self):
        from app.main import MoveResponse
        resp = MoveResponse(
            move=None,
            evaluation=0.0,
            thinking_time_ms=1,
            ai_type="heuristic",
            difficulty=2,
        )
        assert resp.persona_id is None

    def test_round_trip(self):
        from app.main import MoveResponse
        resp = MoveResponse(
            move=None,
            evaluation=0.0,
            thinking_time_ms=1,
            ai_type="heuristic",
            difficulty=2,
            persona_id="aggressive",
        )
        assert resp.persona_id == "aggressive"
        dumped = resp.model_dump()
        assert dumped["persona_id"] == "aggressive"


class TestHeuristicProfileIdMapping:
    """The 4 persona names must map to profile ids that actually exist in
    app.ai.heuristic_weights. Locks in the contract between the API and
    the underlying weight table."""

    @pytest.mark.parametrize(
        "persona",
        ["balanced", "aggressive", "territorial", "defensive"],
    )
    def test_persona_profile_id_resolves_to_real_weights(self, persona, monkeypatch):
        from app import main
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES
        monkeypatch.setenv("RINGRIFT_PERSONAS_ENABLED", "true")
        pid = main._resolve_persona_profile_id(persona)
        assert pid is not None
        assert pid in HEURISTIC_WEIGHT_PROFILES, (
            f"persona {persona!r} mapped to {pid!r} which is not in "
            f"HEURISTIC_WEIGHT_PROFILES; the C2 persona contract is broken"
        )
