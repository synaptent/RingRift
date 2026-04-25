"""Backward-compatible shim for the split unified-loop package."""

from scripts import unified_loop as _unified_loop

for _name in _unified_loop.__all__:
    globals()[_name] = getattr(_unified_loop, _name)

__all__ = list(_unified_loop.__all__)
