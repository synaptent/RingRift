"""HTTP-based file download handlers for P2P cluster.

Provides endpoints for downloading files when SSH/SCP is unreliable:
- GET /files/models/<path> - Download model files
- GET /files/data/<path> - Download data files (databases, NPZ)
- GET /files/list - List available files

This is a permanent workaround for SSH connectivity issues on nodes like Nebius
where connection resets are frequent.

December 2025: Created as alternative to SSH-based sync.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore

try:
    from .timeout_decorator import handler_timeout, HANDLER_TIMEOUT_DELIVERY
except ImportError:
    # Fallback if decorator not available
    def handler_timeout(seconds):
        def decorator(func):
            return func
        return decorator
    HANDLER_TIMEOUT_DELIVERY = 120.0

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)


class FileDownloadHandler:
    """Handler for HTTP-based file downloads.

    Allows nodes to pull files via HTTP instead of SSH/SCP when connectivity
    is unreliable. This is especially useful for:
    - Nebius nodes with frequent connection resets
    - Nodes behind NAT/firewalls
    - Environments without SSH keys configured
    """

    ALLOWED_EXTENSIONS = {
        '.pth', '.pt', '.onnx',  # Model files
        '.db', '.sqlite', '.sqlite3',  # Database files
        '.npz', '.npy',  # Training data
        '.json', '.yaml', '.yml',  # Config files
    }

    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

    def __init__(self, orchestrator):
        """Initialize handler with P2P orchestrator reference."""
        self.orchestrator = orchestrator
        # Paths relative to ai-service root
        self.ai_service_root = self._find_ai_service_root()

    def _find_ai_service_root(self) -> Path:
        """Find the ai-service directory root."""
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent.parent,  # From handlers/
            Path.cwd(),
            Path(os.environ.get("RINGRIFT_AI_SERVICE_ROOT", ".")),
            Path.home() / "ringrift" / "ai-service",
            Path("/workspace/ringrift/ai-service"),
            Path("/root/ringrift/ai-service"),
        ]
        for candidate in candidates:
            if (candidate / "models").exists() or (candidate / "data").exists():
                return candidate.resolve()
        return Path.cwd().resolve()

    def _validate_path(self, base_dir: Path, rel_path: str) -> tuple[bool, Path | str]:
        """Validate a file path is safe and within allowed directory.

        Returns:
            (True, resolved_path) if valid
            (False, error_message) if invalid
        """
        if not rel_path:
            return False, "Path is required"

        # Clean the path
        rel_path = rel_path.lstrip("/").replace("..", "")

        try:
            full_path = (base_dir / rel_path).resolve()
            # Ensure path is within base_dir
            full_path.relative_to(base_dir.resolve())
        except (ValueError, OSError):
            return False, "Path traversal not allowed"

        if not full_path.exists():
            return False, f"File not found: {rel_path}"

        if not full_path.is_file():
            return False, f"Not a file: {rel_path}"

        # Check extension
        if full_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            return False, f"File type not allowed: {full_path.suffix}"

        # Check size
        size = full_path.stat().st_size
        if size > self.MAX_FILE_SIZE:
            return False, f"File too large: {size / 1024 / 1024:.1f} MB (max {self.MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"

        return True, full_path

    def _json_response(self, data: dict, status: int = 200) -> "web.Response":
        """Create a JSON response."""
        import json
        return web.Response(
            text=json.dumps(data),
            content_type="application/json",
            status=status,
        )

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_model_download(self, request: "web.Request") -> "web.StreamResponse":
        """GET /files/models/{path:.*} - Download a model file.

        Example:
            GET /files/models/canonical_hex8_2p.pth
            GET /files/models/checkpoints/latest.pth
        """
        rel_path = request.match_info.get("path", "")
        models_dir = self.ai_service_root / "models"

        valid, result = self._validate_path(models_dir, rel_path)
        if not valid:
            return self._json_response({"error": result}, status=400 if "required" in str(result) else 404)

        file_path: Path = result  # type: ignore
        return await self._stream_file(request, file_path)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_data_download(self, request: "web.Request") -> "web.StreamResponse":
        """GET /files/data/{path:.*} - Download a data file (database, NPZ).

        Example:
            GET /files/data/unified_elo.db
            GET /files/data/games/canonical_hex8_2p.db
            GET /files/data/training/hex8_2p.npz
        """
        rel_path = request.match_info.get("path", "")
        data_dir = self.ai_service_root / "data"

        valid, result = self._validate_path(data_dir, rel_path)
        if not valid:
            return self._json_response({"error": result}, status=400 if "required" in str(result) else 404)

        file_path: Path = result  # type: ignore
        return await self._stream_file(request, file_path)

    async def _stream_file(self, request: "web.Request", file_path: Path) -> "web.StreamResponse":
        """Stream a file to the client with chunked transfer."""
        stat = file_path.stat()

        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(stat.st_size),
                "Content-Disposition": f'attachment; filename="{file_path.name}"',
                "X-File-Size": str(stat.st_size),
                "X-File-Name": file_path.name,
            },
        )
        await resp.prepare(request)

        # Stream in 1MB chunks
        chunk_size = 1024 * 1024
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                await resp.write(chunk)

        await resp.write_eof()
        logger.info(f"Served file via HTTP: {file_path.name} ({stat.st_size / 1024 / 1024:.1f} MB)")
        return resp

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_list_files(self, request: "web.Request") -> "web.Response":
        """GET /files/list - List available files for download.

        Query params:
            type: "models" | "data" | "all" (default: "all")
            pattern: glob pattern (default: "*")
        """
        file_type = request.query.get("type", "all")
        pattern = request.query.get("pattern", "*")

        files = []

        if file_type in ("models", "all"):
            models_dir = self.ai_service_root / "models"
            if models_dir.exists():
                for ext in [".pth", ".pt", ".onnx"]:
                    for f in models_dir.rglob(f"{pattern}{ext}"):
                        if f.is_file():
                            rel_path = f.relative_to(models_dir)
                            files.append({
                                "type": "model",
                                "path": str(rel_path),
                                "size": f.stat().st_size,
                                "url": f"/files/models/{rel_path}",
                            })

        if file_type in ("data", "all"):
            data_dir = self.ai_service_root / "data"
            if data_dir.exists():
                for ext in [".db", ".npz", ".sqlite"]:
                    for f in data_dir.rglob(f"{pattern}{ext}"):
                        if f.is_file():
                            rel_path = f.relative_to(data_dir)
                            files.append({
                                "type": "data",
                                "path": str(rel_path),
                                "size": f.stat().st_size,
                                "url": f"/files/data/{rel_path}",
                            })

        # Sort by size descending
        files.sort(key=lambda x: x["size"], reverse=True)

        return self._json_response({
            "files": files,
            "count": len(files),
            "node_id": getattr(self.orchestrator, "node_id", "unknown"),
        })

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_file_info(self, request: "web.Request") -> "web.Response":
        """GET /files/info - Get info about a specific file.

        Query params:
            path: File path (relative to models/ or data/)
            type: "model" | "data"
        """
        rel_path = request.query.get("path", "")
        file_type = request.query.get("type", "model")

        if file_type == "model":
            base_dir = self.ai_service_root / "models"
        else:
            base_dir = self.ai_service_root / "data"

        valid, result = self._validate_path(base_dir, rel_path)
        if not valid:
            return self._json_response({"error": result}, status=404)

        file_path: Path = result  # type: ignore
        stat = file_path.stat()

        # Compute checksum for verification
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return self._json_response({
            "path": rel_path,
            "type": file_type,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "sha256": sha256.hexdigest(),
            "url": f"/files/{file_type}s/{rel_path}",
        })


def register_file_download_routes(app: "web.Application", orchestrator) -> int:
    """Register file download routes on the aiohttp application.

    Returns:
        Number of routes registered
    """
    handler = FileDownloadHandler(orchestrator)

    # Store handler on orchestrator for access by other components
    orchestrator._file_download_handler = handler

    app.router.add_get("/files/models/{path:.*}", handler.handle_model_download)
    app.router.add_get("/files/data/{path:.*}", handler.handle_data_download)
    app.router.add_get("/files/list", handler.handle_list_files)
    app.router.add_get("/files/info", handler.handle_file_info)

    logger.info("Registered 4 file download routes")
    return 4
