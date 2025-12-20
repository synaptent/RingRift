"""Tests for app.interfaces module."""

import pytest

from app.interfaces import (
    EncodingProvider,
    HashProvider,
    ModelProvider,
    MoveCacheProvider,
)
from app.interfaces.protocols import HashValue, PolicyIndex


class TestProtocolImports:
    """Test that all protocols can be imported."""

    def test_hash_provider_import(self):
        """Test HashProvider can be imported."""
        assert HashProvider is not None

    def test_move_cache_provider_import(self):
        """Test MoveCacheProvider can be imported."""
        assert MoveCacheProvider is not None

    def test_model_provider_import(self):
        """Test ModelProvider can be imported."""
        assert ModelProvider is not None

    def test_encoding_provider_import(self):
        """Test EncodingProvider can be imported."""
        assert EncodingProvider is not None

    def test_type_aliases(self):
        """Test type aliases are defined."""
        assert HashValue is int
        assert PolicyIndex is int


class TestHashProviderProtocol:
    """Tests for HashProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that HashProvider is runtime checkable."""
        from typing import runtime_checkable

        # Should have @runtime_checkable decorator
        assert hasattr(HashProvider, "__protocol_attrs__") or hasattr(
            HashProvider, "_is_runtime_protocol"
        )

    def test_zobrist_implements_hash_provider(self):
        """Test that ZobristHash implements HashProvider interface."""
        from app.core.zobrist import ZobristHash

        zobrist = ZobristHash()

        # Check required methods exist
        assert hasattr(zobrist, "compute_initial_hash")
        assert hasattr(zobrist, "get_player_hash")
        assert hasattr(zobrist, "get_phase_hash")
        assert hasattr(zobrist, "get_marker_hash")
        assert hasattr(zobrist, "get_stack_hash")
        assert hasattr(zobrist, "get_collapsed_hash")

        # Check methods are callable
        assert callable(zobrist.compute_initial_hash)
        assert callable(zobrist.get_player_hash)
        assert callable(zobrist.get_phase_hash)
        assert callable(zobrist.get_marker_hash)
        assert callable(zobrist.get_stack_hash)
        assert callable(zobrist.get_collapsed_hash)


class TestMoveCacheProviderProtocol:
    """Tests for MoveCacheProvider protocol."""

    def test_protocol_methods_defined(self):
        """Test that MoveCacheProvider defines required methods."""
        import inspect

        # Get abstract methods
        [
            name
            for name, method in inspect.getmembers(MoveCacheProvider)
            if not name.startswith("_") and callable(method)
        ]

        # Should have get, set, clear
        assert "get" in dir(MoveCacheProvider)
        assert "set" in dir(MoveCacheProvider)
        assert "clear" in dir(MoveCacheProvider)


class TestModelProviderProtocol:
    """Tests for ModelProvider protocol."""

    def test_protocol_methods_defined(self):
        """Test that ModelProvider defines required methods."""
        assert "load" in dir(ModelProvider)
        assert "get_model_type" in dir(ModelProvider)


class TestEncodingProviderProtocol:
    """Tests for EncodingProvider protocol."""

    def test_protocol_methods_defined(self):
        """Test that EncodingProvider defines required methods."""
        assert "encode_move" in dir(EncodingProvider)
        assert "decode_move" in dir(EncodingProvider)


class TestAllExports:
    """Test __all__ exports are correct."""

    def test_interfaces_all(self):
        """Test that __all__ in interfaces matches exports."""
        from app.interfaces import __all__

        expected = [
            "HashProvider",
            "MoveCacheProvider",
            "ModelProvider",
            "EncodingProvider",
        ]

        for name in expected:
            assert name in __all__, f"{name} should be in __all__"
