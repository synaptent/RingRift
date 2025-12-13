"""Cloud integration helpers for RingRift.

This package intentionally hosts code that may talk to external services (AWS,
Vast.ai, etc.). Keep core rules/training logic under `app/` free of external
network dependencies; wire cloud APIs through scripts and helpers here.
"""

