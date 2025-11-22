import os
import sys

import pytest

# Ensure app package is importable when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.rules.default_engine import DefaultRulesEngine  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env_between_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the mutator-first env flag is unset unless a test sets it.

    This keeps tests isolated and avoids bleed-over from the outer
    process environment when running the suite under different configs.
    """
    monkeypatch.delenv("RINGRIFT_RULES_MUTATOR_FIRST", raising=False)


def test_mutator_first_default_false_when_env_unset() -> None:
    """By default, mutator-first mode is disabled when no env flag is set."""
    engine = DefaultRulesEngine()
    assert getattr(engine, "_mutator_first_enabled", False) is False


@pytest.mark.parametrize(
    "value",
    ["1", "true", "TRUE", "True", "yes", "on"],
)
def test_mutator_first_truthy_env_values_enable_flag(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """Truthy env values should enable mutator-first mode by default.

    The constructor argument remains the ultimate source of truth; these
    tests cover only the env  default mapping when ``mutator_first`` is
    omitted, assuming the server-level gate allows mutator-first.
    """
    monkeypatch.setenv("RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST", "1")
    monkeypatch.setenv("RINGRIFT_RULES_MUTATOR_FIRST", value)
    engine = DefaultRulesEngine()
    assert getattr(engine, "_mutator_first_enabled", False) is True


@pytest.mark.parametrize("value", ["0", "false", "", "off", "no", "random"])
def test_mutator_first_falsy_env_values_do_not_enable_flag(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """Non-truthy env values should leave mutator-first disabled.

    This is evaluated with the server-level gate permitting mutator-first
    so that the per-service flag is the deciding factor.
    """
    monkeypatch.setenv("RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST", "1")
    monkeypatch.setenv("RINGRIFT_RULES_MUTATOR_FIRST", value)
    engine = DefaultRulesEngine()
    assert getattr(engine, "_mutator_first_enabled", False) is False


def test_constructor_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit constructor argument must override any env configuration.

    When the server-level gate allows mutator-first, the constructor flag
    is still the ultimate source of truth for enabling/disabling the
    feature.
    """
    monkeypatch.setenv("RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST", "1")
    monkeypatch.setenv("RINGRIFT_RULES_MUTATOR_FIRST", "1")

    engine_false = DefaultRulesEngine(mutator_first=False)
    assert getattr(engine_false, "_mutator_first_enabled", True) is False

    engine_true = DefaultRulesEngine(mutator_first=True)
    assert getattr(engine_true, "_mutator_first_enabled", False) is True


def test_server_gate_blocks_mutator_first_even_with_env_and_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-level gate must disable mutator-first unless it is truthy.

    When ``RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST`` is unset or falsey, the
    per-service env flag and constructor argument are both ignored.
    """
    # Server gate unset/falsey: env + constructor should not be able to
    # turn on mutator-first.
    monkeypatch.setenv("RINGRIFT_RULES_MUTATOR_FIRST", "1")
    engine_env_only = DefaultRulesEngine()
    assert getattr(engine_env_only, "_mutator_first_enabled", True) is False

    engine_with_constructor = DefaultRulesEngine(mutator_first=True)
    assert getattr(engine_with_constructor, "_mutator_first_enabled", True) is False


def test_server_gate_and_env_enable_mutator_first_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server gate + service env truthy should enable mutator-first.

    When the server-level gate permits mutator-first and the per-service
    env flag is truthy, ``DefaultRulesEngine()`` (without constructor
    override) should enable the mutator-first path.
    """
    monkeypatch.setenv("RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST", "1")
    monkeypatch.setenv("RINGRIFT_RULES_MUTATOR_FIRST", "1")

    engine = DefaultRulesEngine()
    assert getattr(engine, "_mutator_first_enabled", False) is True


def test_server_gate_truthy_but_env_falsey_requires_constructor_to_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truthy server gate alone is not enough to enable mutator-first.

    With a truthy server gate but a falsey/absent per-service env flag,
    mutator-first should remain disabled unless the constructor explicitly
    enables it.
    """
    monkeypatch.setenv("RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST", "1")
    # Service-level flag is absent/falsey by default via the fixture.

    engine_default = DefaultRulesEngine()
    default_enabled = getattr(engine_default, "_mutator_first_enabled", True)
    assert default_enabled is False

    engine_enabled = DefaultRulesEngine(mutator_first=True)
    enabled_flag = getattr(engine_enabled, "_mutator_first_enabled", False)
    assert enabled_flag is True
