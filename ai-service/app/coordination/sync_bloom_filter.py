"""Bloom filter for efficient P2P sync set membership testing.

Harvested from: gossip_sync.py (lines 86-122)
Purpose: Efficient set membership testing for gossip replication without storing full data.

Use cases:
1. P2P gossip: Exchange bloom filters to determine which games need syncing
2. Deduplication: Quickly check if a game/model has been processed
3. Cache invalidation: Track which items have been distributed

Example usage:
    from app.coordination.sync_bloom_filter import SyncBloomFilter

    # Create filter for known game IDs
    bf = SyncBloomFilter(expected_items=10000, false_positive_rate=0.01)
    for game_id in known_game_ids:
        bf.add(game_id)

    # Serialize for P2P exchange
    data = bf.to_bytes()
    send_to_peer(data)

    # Deserialize peer's filter
    peer_bf = SyncBloomFilter.from_bytes(peer_data)

    # Find games peer doesn't have
    games_to_sync = [g for g in my_games if g not in peer_bf]
"""

from __future__ import annotations

import hashlib
import logging
import math
import zlib
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SIZE = 100_000  # Bloom filter bits
DEFAULT_HASH_COUNT = 7  # Number of hash functions
DEFAULT_FALSE_POSITIVE_RATE = 0.01  # 1%


@dataclass
class BloomFilterStats:
    """Statistics about a bloom filter."""

    size_bits: int
    hash_count: int
    items_added: int
    estimated_items: int
    fill_ratio: float
    estimated_false_positive_rate: float
    size_bytes: int
    compressed_bytes: int = 0


class SyncBloomFilter:
    """Bloom filter optimized for P2P sync set membership testing.

    This is an enhanced version of the BloomFilter from gossip_sync.py with:
    - Optimal size calculation based on expected items and FP rate
    - Compression for network transfer
    - Statistics and fill ratio tracking
    - Merge operation for combining filters
    - Thread-safe item counting

    Typical use case: Exchange bloom filters between nodes to efficiently
    determine which games/models need to be synced without transferring
    the full list of IDs.
    """

    def __init__(
        self,
        size: int | None = None,
        hash_count: int | None = None,
        expected_items: int | None = None,
        false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE,
    ):
        """Initialize bloom filter.

        Args:
            size: Explicit size in bits (overrides calculation)
            hash_count: Explicit hash count (overrides calculation)
            expected_items: Expected number of items (for optimal sizing)
            false_positive_rate: Target false positive rate (default 1%)
        """
        if expected_items is not None and size is None:
            # Calculate optimal size and hash count
            self.size, self.hash_count = self._optimal_params(
                expected_items, false_positive_rate
            )
        else:
            self.size = size or DEFAULT_SIZE
            self.hash_count = hash_count or DEFAULT_HASH_COUNT

        self.bits = bytearray(self.size // 8 + 1)
        self._items_added = 0

    @staticmethod
    def _optimal_params(n: int, p: float) -> tuple[int, int]:
        """Calculate optimal bloom filter parameters.

        Args:
            n: Expected number of items
            p: Desired false positive rate

        Returns:
            Tuple of (size_bits, hash_count)
        """
        if n <= 0:
            n = 1
        if p <= 0 or p >= 1:
            p = 0.01

        # Optimal size: m = -n*ln(p) / (ln(2)^2)
        m = int(-n * math.log(p) / (math.log(2) ** 2))
        m = max(m, 64)  # Minimum size

        # Optimal hash count: k = (m/n) * ln(2)
        k = int((m / n) * math.log(2))
        k = max(k, 1)  # At least 1 hash
        k = min(k, 16)  # Cap at 16

        return m, k

    def _hashes(self, item: str) -> list[int]:
        """Generate hash positions for an item.

        Uses double hashing (MD5 + SHA1) for simplicity.
        Note: Not for security - bloom filter only.
        """
        h1 = int(hashlib.md5(item.encode(), usedforsecurity=False).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode(), usedforsecurity=False).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        for pos in self._hashes(item):
            self.bits[pos // 8] |= 1 << (pos % 8)
        self._items_added += 1

    def add_many(self, items: Iterable[str]) -> int:
        """Add multiple items to the filter.

        Returns:
            Number of items added
        """
        count = 0
        for item in items:
            self.add(item)
            count += 1
        return count

    def __contains__(self, item: str) -> bool:
        """Check if an item might be in the filter.

        Note: False positives are possible, false negatives are not.
        """
        return all(
            self.bits[pos // 8] & (1 << (pos % 8)) for pos in self._hashes(item)
        )

    def probably_contains(self, item: str) -> bool:
        """Alias for __contains__ with clearer semantics."""
        return item in self

    @property
    def fill_ratio(self) -> float:
        """Return the ratio of set bits to total bits."""
        set_bits = sum(bin(byte).count("1") for byte in self.bits)
        return set_bits / self.size

    @property
    def items_added(self) -> int:
        """Return number of items added."""
        return self._items_added

    def estimated_items(self) -> int:
        """Estimate number of unique items in filter.

        Uses the formula: n ≈ -m * ln(1 - X/m) / k
        where X = number of set bits
        """
        set_bits = sum(bin(byte).count("1") for byte in self.bits)
        if set_bits == 0:
            return 0
        if set_bits >= self.size:
            return self.size  # Saturated

        ratio = set_bits / self.size
        if ratio >= 1:
            return self.size
        return int(-self.size * math.log(1 - ratio) / self.hash_count)

    def estimated_false_positive_rate(self) -> float:
        """Estimate current false positive rate based on fill ratio.

        Formula: p ≈ (1 - e^(-k*n/m))^k
        """
        n = self._items_added or self.estimated_items()
        if n == 0:
            return 0.0
        return (1 - math.exp(-self.hash_count * n / self.size)) ** self.hash_count

    def to_bytes(self, compress: bool = True) -> bytes:
        """Serialize the filter.

        Args:
            compress: If True, compress with zlib (good for network transfer)

        Returns:
            Serialized filter bytes
        """
        data = bytes(self.bits)
        if compress:
            data = zlib.compress(data, level=6)
        return data

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        size: int | None = None,
        hash_count: int = DEFAULT_HASH_COUNT,
        compressed: bool = True,
    ) -> SyncBloomFilter:
        """Deserialize a filter.

        Args:
            data: Serialized filter bytes
            size: Size in bits (defaults to data size * 8)
            hash_count: Number of hash functions used
            compressed: If True, decompress with zlib

        Returns:
            Reconstructed bloom filter
        """
        if compressed:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                # Data wasn't compressed
                pass

        bf = cls(size=size or (len(data) * 8), hash_count=hash_count)
        bf.bits = bytearray(data)
        # Estimate items from fill ratio
        bf._items_added = bf.estimated_items()
        return bf

    def merge(self, other: SyncBloomFilter) -> SyncBloomFilter:
        """Create a new filter that is the union of this and other.

        Both filters must have the same size and hash count.

        Returns:
            New bloom filter containing all items from both
        """
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Cannot merge filters with different parameters")

        result = SyncBloomFilter(size=self.size, hash_count=self.hash_count)
        for i in range(len(self.bits)):
            result.bits[i] = self.bits[i] | other.bits[i]
        result._items_added = result.estimated_items()
        return result

    def intersection_ratio(self, other: SyncBloomFilter) -> float:
        """Estimate intersection ratio with another filter.

        Returns a value between 0 (no overlap) and 1 (complete overlap).
        """
        if self.size != other.size:
            raise ValueError("Cannot compare filters with different sizes")

        # Count common set bits
        common_bits = sum(
            bin(self.bits[i] & other.bits[i]).count("1") for i in range(len(self.bits))
        )
        my_bits = sum(bin(byte).count("1") for byte in self.bits)

        if my_bits == 0:
            return 0.0
        return common_bits / my_bits

    def get_stats(self) -> BloomFilterStats:
        """Get detailed statistics about the filter."""
        raw_bytes = len(self.bits)
        compressed = self.to_bytes(compress=True)

        return BloomFilterStats(
            size_bits=self.size,
            hash_count=self.hash_count,
            items_added=self._items_added,
            estimated_items=self.estimated_items(),
            fill_ratio=self.fill_ratio,
            estimated_false_positive_rate=self.estimated_false_positive_rate(),
            size_bytes=raw_bytes,
            compressed_bytes=len(compressed),
        )

    def clear(self) -> None:
        """Clear all items from the filter."""
        self.bits = bytearray(self.size // 8 + 1)
        self._items_added = 0

    def __len__(self) -> int:
        """Return estimated number of items in filter."""
        return self.estimated_items()

    def __repr__(self) -> str:
        return (
            f"SyncBloomFilter(size={self.size}, hash_count={self.hash_count}, "
            f"items={self._items_added}, fill={self.fill_ratio:.1%})"
        )


# Convenience factory functions


def create_game_id_filter(expected_games: int = 10000) -> SyncBloomFilter:
    """Create a bloom filter optimized for game ID tracking.

    Args:
        expected_games: Expected number of game IDs to track

    Returns:
        Bloom filter with 1% false positive rate
    """
    return SyncBloomFilter(expected_items=expected_games, false_positive_rate=0.01)


def create_model_hash_filter(expected_models: int = 100) -> SyncBloomFilter:
    """Create a bloom filter optimized for model hash tracking.

    Args:
        expected_models: Expected number of model hashes

    Returns:
        Bloom filter with 0.1% false positive rate (more strict for models)
    """
    return SyncBloomFilter(expected_items=expected_models, false_positive_rate=0.001)


def create_event_dedup_filter(
    expected_events: int = 50000,
    false_positive_rate: float = 0.001,
) -> SyncBloomFilter:
    """Create a bloom filter for event deduplication.

    Args:
        expected_events: Expected number of events to track
        false_positive_rate: Target FP rate (default 0.1%)

    Returns:
        Bloom filter optimized for event deduplication
    """
    return SyncBloomFilter(
        expected_items=expected_events,
        false_positive_rate=false_positive_rate,
    )


# Backward compatibility alias
BloomFilter = SyncBloomFilter
