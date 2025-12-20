"""Zobrist Hashing implementation for RingRift.

This module re-exports ZobristHash from app.core.zobrist for backwards
compatibility. The canonical implementation is in app.core.zobrist.

This allows game_engine to import from app.core.zobrist without creating
a circular dependency through app.ai.
"""

from app.core.zobrist import ZobristHash

__all__ = ["ZobristHash"]
