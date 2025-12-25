"""API Routes for RingRift AI Service.

This package contains modular route definitions organized by domain:
- replay: Game replay and recording endpoints
- ai: AI move selection and evaluation endpoints (to be extracted)
- admin: Administrative endpoints (to be extracted)

Usage:
    from app.routes import replay_router, include_all_routes

    # Include individual routers
    app.include_router(replay_router, prefix="/replay")

    # Or include all routes at once
    include_all_routes(app)

Route Organization (December 2025):
    Routes are organized to keep main.py lean while maintaining
    backwards compatibility. Large endpoint groups should be
    extracted to their own modules.

Migration Guide:
    To extract routes from main.py:

    1. Create a new module (e.g., routes/ai.py)
    2. Create a router: router = APIRouter(prefix="/ai", tags=["ai"])
    3. Move endpoint functions with their decorators
    4. Update imports in the new module
    5. Add router to this __init__.py
    6. Call include_all_routes(app) in main.py or include individually
"""

from fastapi import FastAPI

# Import existing routers
from app.routes.replay import router as replay_router
from app.routes.cluster import router as cluster_router

__all__ = [
    "include_all_routes",
    "replay_router",
    "cluster_router",
]


def include_all_routes(app: FastAPI) -> None:
    """Include all route modules in the FastAPI app.

    This function provides a single entry point for including all
    modular routes. New route modules should be added here.

    Args:
        app: FastAPI application instance
    """
    # Replay routes
    app.include_router(replay_router, tags=["replay"])

    # Cluster monitoring routes (added Dec 2025)
    app.include_router(cluster_router, prefix="/api", tags=["cluster"])

    # Note: AI and admin routes are still in main.py
    # They should be extracted as separate routers following
    # the same pattern as replay_router.
    #
    # Extraction priority (by endpoint count):
    # 1. Admin routes (/admin/*) - 10 endpoints
    # 2. AI routes (/ai/*) - 6 endpoints
    # 3. Rules routes (/rules/*) - 1 endpoint
    # 4. Internal routes (/internal/*) - 1 endpoint
