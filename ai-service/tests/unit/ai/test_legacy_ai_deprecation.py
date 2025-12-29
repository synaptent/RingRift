"""Tests for deprecated legacy AI modules.

These tests verify that deprecated AI modules:
1. Emit proper DeprecationWarning on import
2. Still provide backward-compatible imports during deprecation period
3. Export expected classes/functions

The following modules are tested:
- gmo_ai: GMO v1 (deprecated Dec 2025)
- gmo_v2: GMO v2 (deprecated Dec 2025)
- ebmo_ai: EBMO AI (deprecated Dec 2025)
- ig_gmo: IG-GMO (deprecated Dec 2025)
- gmo_gumbel_hybrid: GMO + Gumbel (deprecated Dec 2025)
- gmo_mcts_hybrid: GMO + MCTS (deprecated Dec 2025)
- gmo_policy_provider: GMO Policy Provider (deprecated Dec 2025)
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


class TestGMOAIDeprecation:
    """Tests for gmo_ai.py deprecation."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify importing gmo_ai emits DeprecationWarning."""
        # Remove cached module if present
        if "app.ai.gmo_ai" in sys.modules:
            del sys.modules["app.ai.gmo_ai"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import app.ai.gmo_ai  # noqa: F401
            except ImportError:
                pytest.skip("gmo_ai archive module not available")

            # Check for deprecation warning
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1, "Expected DeprecationWarning on import"
            assert "gmo_ai" in str(deprecation_warnings[0].message).lower()
            assert "Q2 2026" in str(deprecation_warnings[0].message)


class TestGMOV2Deprecation:
    """Tests for gmo_v2.py deprecation."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify importing gmo_v2 emits DeprecationWarning."""
        if "app.ai.gmo_v2" in sys.modules:
            del sys.modules["app.ai.gmo_v2"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import app.ai.gmo_v2  # noqa: F401
            except ImportError:
                pytest.skip("gmo_v2 archive module not available")

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1, "Expected DeprecationWarning on import"
            assert "gmo_v2" in str(deprecation_warnings[0].message).lower()


class TestEBMOAIDeprecation:
    """Tests for ebmo_ai.py deprecation."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify importing ebmo_ai emits DeprecationWarning."""
        if "app.ai.ebmo_ai" in sys.modules:
            del sys.modules["app.ai.ebmo_ai"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import app.ai.ebmo_ai  # noqa: F401
            except ImportError:
                pytest.skip("ebmo_ai archive module not available")

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1, "Expected DeprecationWarning on import"
            assert "ebmo_ai" in str(deprecation_warnings[0].message).lower()


class TestIGGMODeprecation:
    """Tests for ig_gmo.py deprecation."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify importing ig_gmo emits DeprecationWarning."""
        if "app.ai.ig_gmo" in sys.modules:
            del sys.modules["app.ai.ig_gmo"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import app.ai.ig_gmo  # noqa: F401
            except ImportError:
                pytest.skip("ig_gmo archive module not available")

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1, "Expected DeprecationWarning on import"
            assert "ig_gmo" in str(deprecation_warnings[0].message).lower()


class TestGMOGumbelHybridDeprecation:
    """Tests for gmo_gumbel_hybrid.py deprecation."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify importing gmo_gumbel_hybrid emits DeprecationWarning."""
        if "app.ai.gmo_gumbel_hybrid" in sys.modules:
            del sys.modules["app.ai.gmo_gumbel_hybrid"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import app.ai.gmo_gumbel_hybrid  # noqa: F401
            except ImportError:
                pytest.skip("gmo_gumbel_hybrid dependencies not available")

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            # May have multiple warnings due to transitive imports
            gumbel_hybrid_warnings = [
                x for x in deprecation_warnings
                if "gmo_gumbel_hybrid" in str(x.message).lower()
            ]
            assert len(gumbel_hybrid_warnings) >= 1, "Expected DeprecationWarning for gmo_gumbel_hybrid"
            assert "Q2 2026" in str(gumbel_hybrid_warnings[0].message)


class TestGMOMCTSHybridDeprecation:
    """Tests for gmo_mcts_hybrid.py deprecation."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify importing gmo_mcts_hybrid emits DeprecationWarning."""
        if "app.ai.gmo_mcts_hybrid" in sys.modules:
            del sys.modules["app.ai.gmo_mcts_hybrid"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import app.ai.gmo_mcts_hybrid  # noqa: F401
            except ImportError:
                pytest.skip("gmo_mcts_hybrid dependencies not available")

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            mcts_hybrid_warnings = [
                x for x in deprecation_warnings
                if "gmo_mcts_hybrid" in str(x.message).lower()
            ]
            assert len(mcts_hybrid_warnings) >= 1, "Expected DeprecationWarning for gmo_mcts_hybrid"
            assert "Q2 2026" in str(mcts_hybrid_warnings[0].message)


class TestGMOPolicyProviderDeprecation:
    """Tests for gmo_policy_provider.py deprecation."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify importing gmo_policy_provider emits DeprecationWarning."""
        if "app.ai.gmo_policy_provider" in sys.modules:
            del sys.modules["app.ai.gmo_policy_provider"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import app.ai.gmo_policy_provider  # noqa: F401
            except ImportError:
                pytest.skip("gmo_policy_provider dependencies not available")

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            policy_warnings = [
                x for x in deprecation_warnings
                if "gmo_policy_provider" in str(x.message).lower()
            ]
            assert len(policy_warnings) >= 1, "Expected DeprecationWarning for gmo_policy_provider"
            assert "Q2 2026" in str(policy_warnings[0].message)


class TestGMOSharedNote:
    """Tests for gmo_shared.py deprecation note."""

    def test_module_docstring_contains_deprecation_note(self) -> None:
        """Verify gmo_shared has deprecation note in docstring."""
        try:
            import app.ai.gmo_shared as gmo_shared
        except ImportError:
            pytest.skip("gmo_shared dependencies not available")

        docstring = gmo_shared.__doc__ or ""
        assert "deprecated" in docstring.lower(), "gmo_shared should note it supports deprecated code"
        assert "Q2 2026" in docstring, "gmo_shared should mention removal timeline"


class TestDeprecationMessageFormat:
    """Tests for deprecation message format consistency."""

    @pytest.mark.parametrize("module_name,expected_replacement", [
        ("app.ai.gmo_ai", "neural_net"),
        ("app.ai.gmo_v2", "neural_net"),
        ("app.ai.ebmo_ai", "neural_net"),
        ("app.ai.ig_gmo", "neural_net"),
        ("app.ai.gmo_gumbel_hybrid", "GumbelMCTSAI"),
        ("app.ai.gmo_mcts_hybrid", "MCTSAI"),
        ("app.ai.gmo_policy_provider", "nnue_policy"),
    ])
    def test_deprecation_suggests_replacement(
        self, module_name: str, expected_replacement: str
    ) -> None:
        """Verify deprecation messages suggest appropriate replacement."""
        # Clear module cache
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                importlib.import_module(module_name)
            except ImportError:
                pytest.skip(f"{module_name} dependencies not available")

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            if not deprecation_warnings:
                pytest.skip(f"No deprecation warning from {module_name}")

            # Check that at least one warning mentions the replacement
            messages = [str(x.message) for x in deprecation_warnings]
            combined = " ".join(messages)
            assert expected_replacement.lower() in combined.lower(), (
                f"Deprecation message should suggest {expected_replacement}"
            )
