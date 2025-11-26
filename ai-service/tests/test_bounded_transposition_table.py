"""Unit tests for BoundedTranspositionTable."""

import pytest

from app.ai.bounded_transposition_table import BoundedTranspositionTable


class TestBasicOperations:
    """Tests for basic get/put operations."""

    def test_put_and_get(self) -> None:
        """Should store and retrieve values."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("key1", "value1")
        assert table.get("key1") == "value1"

    def test_get_nonexistent_key_returns_none(self) -> None:
        """Should return None for keys not in table."""
        table = BoundedTranspositionTable(max_entries=100)
        assert table.get("nonexistent") is None

    def test_put_updates_existing_key(self) -> None:
        """Should update value when key already exists."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("key1", "value1")
        table.put("key1", "value2")
        assert table.get("key1") == "value2"

    def test_contains_operator(self) -> None:
        """Should support 'in' operator for membership check."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("key1", "value1")
        assert "key1" in table
        assert "key2" not in table

    def test_len_operator(self) -> None:
        """Should support len() for entry count."""
        table = BoundedTranspositionTable(max_entries=100)
        assert len(table) == 0
        table.put("key1", "value1")
        assert len(table) == 1
        table.put("key2", "value2")
        assert len(table) == 2

    def test_clear(self) -> None:
        """Should clear all entries and reset stats."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("key1", "value1")
        table.put("key2", "value2")
        table.get("key1")  # Hit
        table.get("missing")  # Miss

        table.clear()

        assert len(table) == 0
        assert table.hits == 0
        assert table.misses == 0
        assert table.evictions == 0

    def test_various_key_types(self) -> None:
        """Should handle various hashable key types."""
        table = BoundedTranspositionTable(max_entries=100)

        # String key
        table.put("string_key", 1)
        assert table.get("string_key") == 1

        # Integer key
        table.put(42, 2)
        assert table.get(42) == 2

        # Tuple key
        table.put((1, 2, 3), 3)
        assert table.get((1, 2, 3)) == 3

    def test_various_value_types(self) -> None:
        """Should handle various value types."""
        table = BoundedTranspositionTable(max_entries=100)

        table.put("str", "string_value")
        table.put("int", 42)
        table.put("float", 3.14)
        table.put("dict", {"a": 1, "b": 2})
        table.put("list", [1, 2, 3])
        table.put("tuple", (1, 2, 3))

        assert table.get("str") == "string_value"
        assert table.get("int") == 42
        assert table.get("float") == 3.14
        assert table.get("dict") == {"a": 1, "b": 2}
        assert table.get("list") == [1, 2, 3]
        assert table.get("tuple") == (1, 2, 3)


class TestLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_eviction_at_capacity(self) -> None:
        """Should evict oldest entry when at capacity."""
        table = BoundedTranspositionTable(max_entries=3)
        table.put("a", 1)
        table.put("b", 2)
        table.put("c", 3)

        # Table is now at capacity
        assert len(table) == 3

        # Adding new entry should evict oldest (a)
        table.put("d", 4)

        assert len(table) == 3
        assert "a" not in table
        assert "b" in table
        assert "c" in table
        assert "d" in table

    def test_access_moves_to_end(self) -> None:
        """Accessing an entry should move it to end (most recently used)."""
        table = BoundedTranspositionTable(max_entries=3)
        table.put("a", 1)
        table.put("b", 2)
        table.put("c", 3)

        # Access 'a' - moves it to end
        table.get("a")

        # Now add 'd' - should evict 'b' (oldest now)
        table.put("d", 4)

        assert "a" in table  # Was accessed, moved to end
        assert "b" not in table  # Was oldest, got evicted
        assert "c" in table
        assert "d" in table

    def test_update_moves_to_end(self) -> None:
        """Updating an entry should move it to end."""
        table = BoundedTranspositionTable(max_entries=3)
        table.put("a", 1)
        table.put("b", 2)
        table.put("c", 3)

        # Update 'a' - moves it to end
        table.put("a", 10)

        # Now add 'd' - should evict 'b'
        table.put("d", 4)

        assert "a" in table
        assert table.get("a") == 10
        assert "b" not in table
        assert "c" in table
        assert "d" in table

    def test_eviction_count(self) -> None:
        """Should track eviction count correctly."""
        table = BoundedTranspositionTable(max_entries=2)
        table.put("a", 1)
        table.put("b", 2)
        assert table.evictions == 0

        table.put("c", 3)  # Evicts 'a'
        assert table.evictions == 1

        table.put("d", 4)  # Evicts 'b'
        assert table.evictions == 2

    def test_lru_order_preserved_across_operations(self) -> None:
        """LRU order should be preserved across multiple operations."""
        table = BoundedTranspositionTable(max_entries=5)

        # Add entries in order
        for i in range(5):
            table.put(f"key{i}", i)

        # Access some entries to change order
        table.get("key0")  # Move to end
        table.get("key2")  # Move to end

        # Order now: key1, key3, key4, key0, key2

        # Add 3 new entries, evicting key1, key3, key4
        table.put("new1", 100)
        table.put("new2", 200)
        table.put("new3", 300)

        assert "key0" in table
        assert "key2" in table
        assert "key1" not in table
        assert "key3" not in table
        assert "key4" not in table


class TestMemoryLimitSizing:
    """Tests for memory limit sizing."""

    def test_from_memory_limit_basic(self) -> None:
        """Should calculate max_entries from memory limit."""
        # 1MB with 200 bytes per entry = 5000 entries (1024*1024/200 = 5242)
        table = BoundedTranspositionTable.from_memory_limit(
            memory_limit_bytes=1024 * 1024, entry_size_estimate=200
        )
        expected = (1024 * 1024) // 200
        assert table.max_entries == expected

    def test_from_memory_limit_minimum_entries(self) -> None:
        """Should have minimum of 1000 entries."""
        # Very small memory limit
        table = BoundedTranspositionTable.from_memory_limit(
            memory_limit_bytes=100, entry_size_estimate=200
        )
        assert table.max_entries == 1000

    def test_from_memory_limit_custom_entry_size(self) -> None:
        """Should use custom entry size estimate."""
        table = BoundedTranspositionTable.from_memory_limit(
            memory_limit_bytes=1024 * 1024, entry_size_estimate=100
        )
        expected = (1024 * 1024) // 100
        assert table.max_entries == expected

    def test_from_memory_limit_large_memory(self) -> None:
        """Should handle large memory limits."""
        # 1GB limit
        one_gb = 1024**3
        table = BoundedTranspositionTable.from_memory_limit(
            memory_limit_bytes=one_gb, entry_size_estimate=200
        )
        expected = one_gb // 200
        assert table.max_entries == expected


class TestStatsTracking:
    """Tests for stats tracking."""

    def test_hit_tracking(self) -> None:
        """Should track hits correctly."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("key1", "value1")

        assert table.hits == 0
        table.get("key1")
        assert table.hits == 1
        table.get("key1")
        assert table.hits == 2

    def test_miss_tracking(self) -> None:
        """Should track misses correctly."""
        table = BoundedTranspositionTable(max_entries=100)

        assert table.misses == 0
        table.get("nonexistent1")
        assert table.misses == 1
        table.get("nonexistent2")
        assert table.misses == 2

    def test_eviction_tracking(self) -> None:
        """Should track evictions correctly."""
        table = BoundedTranspositionTable(max_entries=2)
        table.put("a", 1)
        table.put("b", 2)

        assert table.evictions == 0
        table.put("c", 3)  # Evicts 'a'
        assert table.evictions == 1

    def test_stats_method(self) -> None:
        """Should return comprehensive stats."""
        table = BoundedTranspositionTable(
            max_entries=100, entry_size_estimate=200
        )
        table.put("a", 1)
        table.put("b", 2)
        table.get("a")  # Hit
        table.get("c")  # Miss

        stats = table.stats()

        assert stats["entries"] == 2
        assert stats["max_entries"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == pytest.approx(0.5)
        # 2 entries * 200 bytes / 1024^2 = 0.000381 MB
        expected_mb = 2 * 200 / (1024**2)
        assert stats["estimated_memory_mb"] == pytest.approx(expected_mb)

    def test_hit_rate_zero_lookups(self) -> None:
        """Hit rate should be 0.0 when no lookups performed."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("a", 1)

        stats = table.stats()
        assert stats["hit_rate"] == 0.0

    def test_hit_rate_all_hits(self) -> None:
        """Hit rate should be 1.0 when all lookups are hits."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("a", 1)
        table.get("a")
        table.get("a")
        table.get("a")

        stats = table.stats()
        assert stats["hit_rate"] == pytest.approx(1.0)

    def test_hit_rate_all_misses(self) -> None:
        """Hit rate should be 0.0 when all lookups are misses."""
        table = BoundedTranspositionTable(max_entries=100)
        table.get("x")
        table.get("y")
        table.get("z")

        stats = table.stats()
        assert stats["hit_rate"] == pytest.approx(0.0)

    def test_stats_after_clear(self) -> None:
        """Stats should be reset after clear."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("a", 1)
        table.get("a")
        table.get("b")

        table.clear()
        stats = table.stats()

        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 0.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_entry_table(self) -> None:
        """Should work with max_entries=1."""
        table = BoundedTranspositionTable(max_entries=1)
        table.put("a", 1)
        assert table.get("a") == 1

        table.put("b", 2)
        assert "a" not in table
        assert table.get("b") == 2

    def test_none_value(self) -> None:
        """Should handle None as a value."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("key", None)

        # This is tricky - get returns None for missing keys too
        # But we can check containment
        assert "key" in table
        assert table.get("key") is None

    def test_empty_string_key(self) -> None:
        """Should handle empty string as key."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put("", "empty_key_value")
        assert table.get("") == "empty_key_value"

    def test_zero_key(self) -> None:
        """Should handle 0 as key."""
        table = BoundedTranspositionTable(max_entries=100)
        table.put(0, "zero_key_value")
        assert table.get(0) == "zero_key_value"

    def test_large_number_of_entries(self) -> None:
        """Should handle many entries efficiently."""
        table = BoundedTranspositionTable(max_entries=10000)

        # Add 10000 entries
        for i in range(10000):
            table.put(f"key{i}", i)

        assert len(table) == 10000

        # All should be retrievable
        for i in range(10000):
            assert table.get(f"key{i}") == i