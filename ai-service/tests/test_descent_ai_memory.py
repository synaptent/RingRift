"""Integration tests for DescentAI memory limiting behavior."""

from app.ai.bounded_transposition_table import BoundedTranspositionTable
from app.ai.descent_ai import DescentAI
from app.models import AIConfig
from app.utils.memory_config import MemoryConfig


class TestDescentAIMemoryIntegration:
    """Tests for DescentAI memory limiting integration."""

    def test_default_memory_config_from_env(self) -> None:
        """Verify DescentAI uses MemoryConfig.from_env() by default."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        ai = DescentAI(player_number=1, config=config)

        assert ai.memory_config is not None
        assert isinstance(ai.memory_config, MemoryConfig)
        # Default should come from env or default 16GB
        assert ai.memory_config.max_memory_gb > 0

    def test_custom_memory_config(self) -> None:
        """Verify DescentAI accepts custom MemoryConfig."""
        custom_config = MemoryConfig(
            max_memory_gb=4.0,
            training_allocation=0.5,
            inference_allocation=0.4,
            system_reserve=0.1,
        )
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        ai = DescentAI(
            player_number=1,
            config=config,
            memory_config=custom_config,
        )

        assert ai.memory_config is custom_config
        assert ai.memory_config.max_memory_gb == 4.0

    def test_transposition_table_is_bounded(self) -> None:
        """Verify the transposition table is BoundedTranspositionTable."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        ai = DescentAI(player_number=1, config=config)

        assert isinstance(ai.transposition_table, BoundedTranspositionTable)
        assert ai.transposition_table.max_entries > 0

    def test_transposition_table_respects_memory_limit(self) -> None:
        """Verify table max_entries is derived from memory config."""
        # Use a small memory config to verify scaling
        small_config = MemoryConfig(
            max_memory_gb=1.0,
            training_allocation=0.6,
            inference_allocation=0.3,
            system_reserve=0.1,
        )
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        ai = DescentAI(
            player_number=1,
            config=config,
            memory_config=small_config,
        )

        # With 1GB max memory, 30% inference, 50% of that for TT:
        # 1GB * 0.3 * 0.5 = 0.15GB = ~157MB for TT
        # At 200 bytes per entry, that's ~785,000 entries
        expected_tt_bytes = small_config.get_transposition_table_limit_bytes()
        expected_max_entries = max(1000, expected_tt_bytes // 200)

        assert ai.transposition_table.max_entries == expected_max_entries

    def test_transposition_table_put_and_get_operations(self) -> None:
        """Verify put/get operations work on the bounded table."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        ai = DescentAI(player_number=1, config=config)

        # Simulate storing a transposition table entry
        test_key = 123456789
        test_value = (0.5, {"move1": ("move", 0.6, 0.3)}, None)

        ai.transposition_table.put(test_key, test_value)
        retrieved = ai.transposition_table.get(test_key)

        assert retrieved is not None
        assert retrieved == test_value

    def test_transposition_table_tracks_stats(self) -> None:
        """Verify stats are tracked during table operations."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        ai = DescentAI(player_number=1, config=config)

        # Store some entries
        for i in range(10):
            ai.transposition_table.put(i, (0.5, {}, None))

        # Access some entries (hits)
        for i in range(5):
            ai.transposition_table.get(i)

        # Access non-existent entries (misses)
        for i in range(100, 105):
            ai.transposition_table.get(i)

        stats = ai.transposition_table.stats()

        assert stats["entries"] == 10
        assert stats["hits"] == 5
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 0.5

    def test_transposition_table_evicts_when_full(self) -> None:
        """Verify LRU eviction works when table reaches capacity."""
        # Create AI with very small memory limit to force eviction
        tiny_config = MemoryConfig(max_memory_gb=0.001)  # ~1MB
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        ai = DescentAI(
            player_number=1,
            config=config,
            memory_config=tiny_config,
        )

        max_entries = ai.transposition_table.max_entries

        # Fill table to capacity
        for i in range(max_entries):
            ai.transposition_table.put(i, (0.5, {}, None))

        # Table should be at capacity
        assert len(ai.transposition_table) == max_entries
        assert ai.transposition_table.evictions == 0

        # Add more entries to trigger eviction
        for i in range(max_entries, max_entries + 100):
            ai.transposition_table.put(i, (0.5, {}, None))

        # Should have evicted old entries
        stats = ai.transposition_table.stats()
        assert stats["evictions"] == 100
        assert stats["entries"] == max_entries

        # Oldest entries should be evicted (LRU)
        assert ai.transposition_table.get(0) is None
        assert ai.transposition_table.get(max_entries + 99) is not None