"""Tests for app.core.zobrist module."""

import pytest

from app.core.zobrist import ZobristHash


class TestZobristHash:
    """Tests for ZobristHash class."""

    def test_singleton_pattern(self):
        """Test that ZobristHash uses singleton pattern."""
        hash1 = ZobristHash()
        hash2 = ZobristHash()
        assert hash1 is hash2

    def test_get_stack_hash_deterministic(self):
        """Test that stack hash is deterministic."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_stack_hash("a1", player=1, height=2, rings=(1, 2))
        hash2 = zobrist.get_stack_hash("a1", player=1, height=2, rings=(1, 2))
        assert hash1 == hash2

    def test_get_stack_hash_different_for_different_inputs(self):
        """Test that different inputs produce different hashes."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_stack_hash("a1", player=1, height=2, rings=(1, 2))
        hash2 = zobrist.get_stack_hash("a2", player=1, height=2, rings=(1, 2))
        hash3 = zobrist.get_stack_hash("a1", player=2, height=2, rings=(1, 2))
        hash4 = zobrist.get_stack_hash("a1", player=1, height=3, rings=(1, 2))
        hash5 = zobrist.get_stack_hash("a1", player=1, height=2, rings=(1, 3))

        # All should be different
        hashes = [hash1, hash2, hash3, hash4, hash5]
        assert len(set(hashes)) == len(hashes), "Hashes should be unique for different inputs"

    def test_get_marker_hash_deterministic(self):
        """Test that marker hash is deterministic."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_marker_hash("b2", player=1)
        hash2 = zobrist.get_marker_hash("b2", player=1)
        assert hash1 == hash2

    def test_get_marker_hash_different_for_different_inputs(self):
        """Test that different marker inputs produce different hashes."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_marker_hash("a1", player=1)
        hash2 = zobrist.get_marker_hash("a2", player=1)
        hash3 = zobrist.get_marker_hash("a1", player=2)

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_get_collapsed_hash_deterministic(self):
        """Test that collapsed hash is deterministic."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_collapsed_hash("c3")
        hash2 = zobrist.get_collapsed_hash("c3")
        assert hash1 == hash2

    def test_get_collapsed_hash_different_for_positions(self):
        """Test that different positions produce different collapsed hashes."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_collapsed_hash("a1")
        hash2 = zobrist.get_collapsed_hash("b2")
        assert hash1 != hash2

    def test_get_player_hash_deterministic(self):
        """Test that player hash is deterministic."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_player_hash(1)
        hash2 = zobrist.get_player_hash(1)
        assert hash1 == hash2

    def test_get_player_hash_different_for_players(self):
        """Test that different players produce different hashes."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_player_hash(1)
        hash2 = zobrist.get_player_hash(2)
        assert hash1 != hash2

    def test_get_phase_hash_deterministic(self):
        """Test that phase hash is deterministic."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_phase_hash("PLACEMENT")
        hash2 = zobrist.get_phase_hash("PLACEMENT")
        assert hash1 == hash2

    def test_get_phase_hash_different_for_phases(self):
        """Test that different phases produce different hashes."""
        zobrist = ZobristHash()
        hash1 = zobrist.get_phase_hash("PLACEMENT")
        hash2 = zobrist.get_phase_hash("MOVEMENT")
        assert hash1 != hash2

    def test_hash_values_are_64_bit(self):
        """Test that hash values are within 64-bit range."""
        zobrist = ZobristHash()

        # Check various hash types
        stack_hash = zobrist.get_stack_hash("a1", 1, 2, (1, 2))
        marker_hash = zobrist.get_marker_hash("a1", 1)
        collapsed_hash = zobrist.get_collapsed_hash("a1")
        player_hash = zobrist.get_player_hash(1)
        phase_hash = zobrist.get_phase_hash("PLACEMENT")

        max_64bit = (1 << 64) - 1
        assert 0 <= stack_hash <= max_64bit
        assert 0 <= marker_hash <= max_64bit
        assert 0 <= collapsed_hash <= max_64bit
        assert 0 <= player_hash <= max_64bit
        assert 0 <= phase_hash <= max_64bit

    def test_xor_property(self):
        """Test that XOR operations work correctly for incremental updates."""
        zobrist = ZobristHash()

        # Get some hashes
        h1 = zobrist.get_marker_hash("a1", 1)
        h2 = zobrist.get_marker_hash("a2", 2)

        # XOR should be reversible
        combined = h1 ^ h2
        assert combined ^ h1 == h2
        assert combined ^ h2 == h1

        # XOR with self should give 0
        assert h1 ^ h1 == 0

    def test_index_for_is_deterministic(self):
        """Test that internal _index_for is deterministic."""
        zobrist = ZobristHash()

        # Access internal method for testing determinism
        idx1 = zobrist._index_for("test_key")
        idx2 = zobrist._index_for("test_key")
        assert idx1 == idx2

    def test_index_for_distribution(self):
        """Test that _index_for produces well-distributed indices."""
        zobrist = ZobristHash()

        indices = []
        for i in range(100):
            idx = zobrist._index_for(f"key_{i}")
            indices.append(idx)

        # All indices should be within table size
        assert all(0 <= idx < zobrist.table_size for idx in indices)

        # Should have reasonable distribution (no duplicates for distinct keys)
        assert len(set(indices)) == len(indices)


class TestZobristHashThreadSafety:
    """Tests for thread safety of ZobristHash."""

    def test_concurrent_access(self):
        """Test that concurrent access doesn't cause issues."""
        import threading

        results = []
        errors = []

        def worker():
            try:
                zobrist = ZobristHash()
                # Perform some operations
                for i in range(100):
                    zobrist.get_marker_hash(f"pos_{i}", i % 4)
                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(results) == 10


class TestZobristReExport:
    """Tests for re-export from app.ai.zobrist."""

    def test_import_from_ai(self):
        """Test that importing from app.ai.zobrist works."""
        from app.ai.zobrist import ZobristHash as ZobristHashAI
        from app.core.zobrist import ZobristHash as ZobristHashCore

        # Both should return the same singleton
        assert ZobristHashAI() is ZobristHashCore()
