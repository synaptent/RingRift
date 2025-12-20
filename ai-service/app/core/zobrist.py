"""
Zobrist Hashing implementation for RingRift.
Provides O(1) state hashing for transposition tables.
"""

import random
import threading


class ZobristHash:
    """
    Thread-safe singleton class to manage Zobrist hash keys.

    The singleton pattern uses a lock to ensure thread-safety during
    initialization, which is important when multiple game threads
    access the ZobristHash simultaneously (e.g., in tournament play).
    """
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        # Double-checked locking pattern for thread-safe singleton
        if cls._instance is None:
            with cls._lock:
                # Check again inside lock to handle race condition
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self):
        """Initialize random bitstrings for all board features.

        This uses a private Random instance so that constructing a
        ZobristHash never mutates the module-level ``random`` state.
        """

        # Private RNG seeded with a fixed constant for reproducibility.
        self._rng = random.Random(42)

        # Board features to hash:
        # - Stacks: (position_key, controlling_player, height, rings_tuple)
        # - Markers: (position_key, player)
        # - Collapsed spaces: (position_key)
        # - Current player
        # - Current phase

        self.stack_keys: dict[str, int] = {}
        self.marker_keys: dict[str, int] = {}
        self.collapsed_keys: dict[str, int] = {}
        self.player_keys: dict[int, int] = {}
        self.phase_keys: dict[str, int] = {}

        # Pre-generate keys for common positions (lazy load others if needed,
        # but better to pre-gen for max speed). We back this with a large
        # table of 64‑bit values and index into it using a stable, salted‑hash‑
        # independent function so results do not depend on PYTHONHASHSEED.
        self.table_size = 1000000
        self.table = [
            self._rng.getrandbits(64)
            for _ in range(self.table_size)
        ]

    def _index_for(self, key: str) -> int:
        """Compute a stable table index for a feature key.

        We avoid Python's built-in ``hash()`` here because it is salted via
        ``PYTHONHASHSEED`` and therefore not stable across processes. A small
        64‑bit FNV‑1a style hash is sufficient and fully deterministic.
        """
        # 64‑bit FNV‑1a parameters.
        h = 1469598103934665603
        for b in key.encode("utf-8"):
            h ^= b
            h = (h * 1099511628211) & ((1 << 64) - 1)
        return h % self.table_size

    def get_stack_hash(
        self,
        pos_key: str,
        player: int,
        height: int,
        rings: tuple[int, ...]
    ) -> int:
        """Get hash for a stack"""
        rings_str = ",".join(str(r) for r in rings)
        key = f"{pos_key}|stack|{player}|{height}|{rings_str}"
        return self.table[self._index_for(key)]

    def get_marker_hash(self, pos_key: str, player: int) -> int:
        """Get hash for a marker"""
        key = f"{pos_key}|marker|{player}"
        return self.table[self._index_for(key)]

    def get_collapsed_hash(self, pos_key: str) -> int:
        """Get hash for a collapsed space"""
        key = f"{pos_key}|collapsed"
        return self.table[self._index_for(key)]

    def get_player_hash(self, player: int) -> int:
        """Get hash for current player"""
        key = f"player|{player}"
        return self.table[self._index_for(key)]

    def get_phase_hash(self, phase: str) -> int:
        """Get hash for current phase"""
        key = f"phase|{phase}"
        return self.table[self._index_for(key)]

    def compute_initial_hash(self, game_state) -> int:
        """Compute full hash from scratch (expensive, O(N))"""
        h = 0

        # Stacks
        for pos_key, stack in game_state.board.stacks.items():
            h ^= self.get_stack_hash(
                pos_key,
                stack.controlling_player,
                stack.stack_height,
                tuple(stack.rings)
            )

        # Markers
        for pos_key, marker in game_state.board.markers.items():
            h ^= self.get_marker_hash(pos_key, marker.player)

        # Collapsed
        for pos_key in game_state.board.collapsed_spaces:
            h ^= self.get_collapsed_hash(pos_key)

        # Global state
        h ^= self.get_player_hash(game_state.current_player)
        h ^= self.get_phase_hash(game_state.current_phase)

        return h
