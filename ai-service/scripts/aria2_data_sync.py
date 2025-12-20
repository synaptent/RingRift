#!/usr/bin/env python3
"""
aria2-based Data Sync Infrastructure for RingRift Distributed Training

Extends aria2 support beyond model distribution to include:
- Training data (selfplay databases, game records)
- ELO databases and tournament results
- Training checkpoints and metrics
- Bidirectional P2P gossip sync

This eliminates data silos by enabling any node to pull data from any other
node without manual SCP/rsync operations.

Usage:
    # Start data server on a node
    python aria2_data_sync.py serve --port 8766

    # Sync selfplay data from cluster
    python aria2_data_sync.py sync-games --sources "http://host1:8766,http://host2:8766"

    # Sync training data (latest batches)
    python aria2_data_sync.py sync-training --days 7

    # Full cluster sync (models + data)
    python aria2_data_sync.py cluster-sync

    # Export data inventory
    python aria2_data_sync.py inventory
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import shutil
import socketserver
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlopen

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("aria2_data_sync")


# ============================================
# Configuration
# ============================================

DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"

try:
    from app.distributed.storage_provider import get_storage_provider
except Exception:
    get_storage_provider = None


def _resolve_storage_paths() -> tuple[list[Path], Path, Path, list[Path]]:
    if get_storage_provider:
        provider = get_storage_provider()
        selfplay_dir = provider.selfplay_dir
        training_dir = provider.training_dir
        models_dir = provider.models_dir
        elo_db_paths = [provider.paths.elo_database]
    else:
        selfplay_dir = DATA_DIR / "selfplay"
        training_dir = DATA_DIR / "training"
        models_dir = ROOT / "models"
        elo_db_paths = [DATA_DIR / "unified_elo.db"]

    games_dirs: list[Path] = []
    for candidate in [DATA_DIR / "games", selfplay_dir]:
        if candidate not in games_dirs:
            games_dirs.append(candidate)

    fallback_elo = DATA_DIR / "unified_elo.db"
    if fallback_elo not in elo_db_paths:
        elo_db_paths.append(fallback_elo)

    return games_dirs, models_dir, training_dir, elo_db_paths


GAMES_DIRS, MODELS_DIR, TRAINING_DIR, ELO_DB_PATHS = _resolve_storage_paths()

try:
    from app.config.unified_config import get_config
    DEFAULT_DATA_PORT = get_config().distributed.data_server_port
except Exception:
    DEFAULT_DATA_PORT = 8766
DEFAULT_MODEL_PORT = 8765

ARIA2_CONNECTIONS = 16
ARIA2_SPLIT = 4
ARIA2_TIMEOUT = 120


def _load_hosts_from_config():
    """Load cluster data sources from config/distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print("[Script] Warning: No config found")
        return []
    try:
        from app.sync.cluster_hosts import load_hosts_config
        config = load_hosts_config()
    except Exception:
        config = {}

    try:
        # Extract hosts with data servers (prefer Tailscale IPs)
        sources = []
        for name, host_config in config.get("hosts", {}).items():
            if host_config.get("status") == "terminated":
                continue

            data_url = host_config.get("data_server_url")
            if data_url:
                sources.append(data_url)
                continue

            ip = host_config.get("tailscale_ip") or host_config.get("ssh_host")
            port = host_config.get("data_server_port", DEFAULT_DATA_PORT)
            if ip:
                host = ip.split("@", 1)[1] if "@" in ip else ip
                sources.append(f"http://{host}:{port}")

        return sources
    except Exception as e:
        print(f"[Script] Error loading config: {e}")
        return []


# Cluster hosts with data servers
CLUSTER_DATA_SOURCES = _load_hosts_from_config() or []


# ============================================
# Data Classes
# ============================================

@dataclass
class DataFile:
    """Information about a data file."""
    name: str
    path: str
    size_bytes: int
    mtime: float
    category: str  # 'games', 'models', 'training', 'elo', 'logs'
    checksum: str | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def age_hours(self) -> float:
        return (time.time() - self.mtime) / 3600


@dataclass
class NodeDataInventory:
    """Data inventory for a node."""
    url: str
    hostname: str = ""
    files: dict[str, DataFile] = field(default_factory=dict)
    reachable: bool = False
    last_check: str | None = None
    total_size_mb: float = 0

    def add_file(self, f: DataFile):
        self.files[f.path] = f
        self.total_size_mb += f.size_bytes / (1024 * 1024)


@dataclass
class SyncPlan:
    """Plan for syncing data between nodes."""
    files_to_download: list[DataFile] = field(default_factory=list)
    files_to_skip: list[str] = field(default_factory=list)
    total_size_mb: float = 0
    estimated_time_min: float = 0


# ============================================
# Inventory Building
# ============================================

def get_file_checksum(path: Path, quick: bool = True) -> str:
    """Calculate file checksum. Quick mode uses first/last 1MB."""
    hasher = hashlib.md5()
    size = path.stat().st_size

    if quick and size > 2 * 1024 * 1024:
        # Quick mode: hash first and last 1MB
        with open(path, "rb") as f:
            hasher.update(f.read(1024 * 1024))
            f.seek(-1024 * 1024, 2)
            hasher.update(f.read())
        return f"quick:{hasher.hexdigest()}"
    else:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


def build_local_inventory() -> NodeDataInventory:
    """Build inventory of local data files."""
    import socket

    inventory = NodeDataInventory(
        url=f"http://localhost:{DEFAULT_DATA_PORT}",
        hostname=socket.gethostname(),
        reachable=True,
        last_check=datetime.now().isoformat(),
    )

    # Scan game databases (prefer data/games, then selfplay DBs)
    seen_game_names: set[str] = set()
    for games_dir in GAMES_DIRS:
        if not games_dir.exists():
            continue
        for db_file in games_dir.glob("*.db"):
            if db_file.name in seen_game_names:
                continue
            seen_game_names.add(db_file.name)
            stat = db_file.stat()
            inventory.add_file(DataFile(
                name=db_file.name,
                path=f"games/{db_file.name}",
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                category="games",
            ))

    # Scan models
    if MODELS_DIR.exists():
        for model_file in list(MODELS_DIR.glob("*.pth")) + list(MODELS_DIR.glob("*.onnx")):
            stat = model_file.stat()
            inventory.add_file(DataFile(
                name=model_file.name,
                path=f"models/{model_file.name}",
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                category="models",
            ))

    # Scan ELO databases
    for elo_db in ELO_DB_PATHS:
        if elo_db.exists():
            stat = elo_db.stat()
            inventory.add_file(DataFile(
                name=elo_db.name,
                path=f"elo/{elo_db.name}",
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                category="elo",
            ))
            break

    # Scan training data
    if TRAINING_DIR.exists():
        for batch_file in TRAINING_DIR.glob("*.npz"):
            stat = batch_file.stat()
            inventory.add_file(DataFile(
                name=batch_file.name,
                path=f"training/{batch_file.name}",
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                category="training",
            ))

    return inventory


def count_games_in_db(db_path: Path) -> int:
    """Count games in a selfplay database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


# ============================================
# HTTP Server for Data Serving
# ============================================

class DataHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves data directory with inventory endpoint."""

    def __init__(self, *args, data_root: Path = ROOT, **kwargs):
        self.data_root = data_root
        super().__init__(*args, directory=str(data_root), **kwargs)

    def do_GET(self):
        if self.path == "/inventory.json":
            self.send_inventory()
        elif self.path == "/health":
            self.send_health()
        elif self.path.startswith("/games/"):
            # Serve from games directory
            self.serve_file(GAMES_DIRS, self.path[7:])
        elif self.path.startswith("/models/"):
            # Serve from models directory
            self.serve_file([MODELS_DIR], self.path[8:])
        elif self.path.startswith("/training/"):
            # Serve from training directory
            self.serve_file([TRAINING_DIR], self.path[10:])
        elif self.path.startswith("/elo/"):
            self.serve_elo_file(self.path[5:])
        elif self.path == "/unified_elo.db":
            self.serve_elo_file("unified_elo.db")
        else:
            super().do_GET()

    def serve_file(self, base_dirs: list[Path], filename: str):
        """Serve a specific file from a directory."""
        for base_dir in base_dirs:
            filepath = base_dir / filename
            if filepath.exists() and filepath.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(filepath.stat().st_size))
                self.end_headers()
                with open(filepath, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
        self.send_error(404, f"File not found: {filename}")

    def serve_elo_file(self, filename: str):
        """Serve the unified ELO database (supports /elo/ and legacy path)."""
        for elo_path in ELO_DB_PATHS:
            if elo_path.exists() and elo_path.name == filename:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(elo_path.stat().st_size))
                self.end_headers()
                with open(elo_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
        self.send_error(404, f"File not found: {filename}")

    def send_inventory(self):
        """Send JSON inventory of available data."""
        inventory = build_local_inventory()

        categories = {cat: [] for cat in ["games", "models", "training", "elo"]}
        files_map: dict[str, dict[str, Any]] = {}

        for path, f in inventory.files.items():
            entry = {
                "name": f.name,
                "path": f.path,
                "size_bytes": f.size_bytes,
                "mtime": f.mtime,
                "category": f.category,
                "age_hours": round(f.age_hours, 1),
            }
            files_map[path] = entry
            if f.category in categories:
                categories[f.category].append(entry)

        response = {
            "hostname": inventory.hostname,
            "timestamp": inventory.last_check,
            "total_size_mb": round(inventory.total_size_mb, 2),
            "files": files_map,
            "games": categories["games"],
            "models": categories["models"],
            "training": categories["training"],
            "elo": categories["elo"],
            "summary": {
                cat: len(categories[cat]) for cat in ["games", "models", "training", "elo"]
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response, indent=2).encode())

    def send_health(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def log_message(self, format, *args):
        logger.debug(f"HTTP: {args[0]}")


def run_data_server(port: int = DEFAULT_DATA_PORT):
    """Run HTTP server to serve data files."""
    handler = lambda *args, **kwargs: DataHTTPHandler(*args, data_root=ROOT, **kwargs)

    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        logger.info(f"Data server running on port {port}")
        logger.info(f"Inventory: http://0.0.0.0:{port}/inventory.json")
        logger.info(f"Health: http://0.0.0.0:{port}/health")

        inventory = build_local_inventory()
        logger.info(f"Serving {len(inventory.files)} files ({inventory.total_size_mb:.1f} MB)")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")


# ============================================
# Node Discovery
# ============================================

def check_data_node(url: str, timeout: int = 10) -> NodeDataInventory | None:
    """Check if a data node is reachable and get its inventory."""
    inventory = NodeDataInventory(url=url)

    try:
        inv_url = urljoin(url.rstrip("/") + "/", "inventory.json")
        with urlopen(inv_url, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            inventory.reachable = True
            inventory.hostname = data.get("hostname", "")
            inventory.last_check = datetime.now().isoformat()
            inventory.total_size_mb = data.get("total_size_mb", 0)

            for path, info in data.get("files", {}).items():
                inventory.files[path] = DataFile(
                    name=info.get("name", path),
                    path=path,
                    size_bytes=info.get("size_bytes", 0),
                    mtime=info.get("mtime", 0),
                    category=info.get("category", "unknown"),
                    sources=[urljoin(url.rstrip("/") + "/", path)],
                )

    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.debug(f"Node {url} unreachable: {e}")
        inventory.reachable = False

    return inventory


def discover_data_sources(source_urls: list[str]) -> dict[str, NodeDataInventory]:
    """Discover all available data sources in parallel."""
    logger.info(f"Discovering {len(source_urls)} data sources...")
    inventories = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(check_data_node, url): url for url in source_urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                inv = future.result()
                inventories[url] = inv
                if inv.reachable:
                    logger.info(f"  ✓ {inv.hostname or url}: {len(inv.files)} files ({inv.total_size_mb:.1f} MB)")
                else:
                    logger.debug(f"  ✗ {url}: unreachable")
            except Exception as e:
                logger.error(f"  ✗ {url}: error - {e}")

    reachable = sum(1 for inv in inventories.values() if inv.reachable)
    logger.info(f"Found {reachable}/{len(source_urls)} reachable nodes")

    return inventories


def aggregate_sources(inventories: dict[str, NodeDataInventory]) -> dict[str, DataFile]:
    """Aggregate files from all sources, collecting multiple sources per file."""
    all_files: dict[str, DataFile] = {}

    for url, inv in inventories.items():
        if not inv.reachable:
            continue

        for path, file in inv.files.items():
            if path not in all_files:
                all_files[path] = DataFile(
                    name=file.name,
                    path=path,
                    size_bytes=file.size_bytes,
                    mtime=file.mtime,
                    category=file.category,
                    sources=[],
                )

            source_url = urljoin(url.rstrip("/") + "/", path)
            if source_url not in all_files[path].sources:
                all_files[path].sources.append(source_url)

            # Use newest mtime
            if file.mtime > all_files[path].mtime:
                all_files[path].mtime = file.mtime
                all_files[path].size_bytes = file.size_bytes

    return all_files


# ============================================
# aria2 Download Functions
# ============================================

def check_aria2() -> bool:
    """Check if aria2c is available."""
    return shutil.which("aria2c") is not None


def download_with_aria2(
    file_path: str,
    sources: list[str],
    output_dir: Path,
    connections: int = ARIA2_CONNECTIONS,
    split: int = ARIA2_SPLIT,
) -> tuple[bool, str]:
    """Download a file using aria2c with multiple sources."""
    if not sources:
        return False, "No sources available"

    # Determine output path based on file path
    rel_path = Path(file_path)
    full_output_dir = output_dir / rel_path.parent
    full_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = rel_path.name

    cmd = [
        "aria2c",
        "--max-connection-per-server", str(connections),
        "--split", str(split),
        "--min-split-size", "1M",
        "--timeout", str(ARIA2_TIMEOUT),
        "--connect-timeout", "10",
        "--max-tries", "5",
        "--retry-wait", "2",
        "--auto-file-renaming", "false",
        "--allow-overwrite", "true",
        "--continue", "true",
        "--dir", str(full_output_dir),
        "--out", output_file,
        "--quiet", "true",
    ]

    cmd.extend(sources)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        final_path = full_output_dir / output_file
        if result.returncode == 0 and final_path.exists():
            size_mb = final_path.stat().st_size / (1024 * 1024)
            return True, f"Downloaded {file_path} ({size_mb:.1f} MB)"
        else:
            return False, result.stderr[:200] if result.stderr else "Download failed"

    except subprocess.TimeoutExpired:
        return False, "Download timeout"
    except Exception as e:
        return False, str(e)[:200]


def generate_aria2_input_file(
    files: list[DataFile],
    output_path: Path,
    base_dir: Path,
) -> int:
    """Generate aria2 input file for batch download."""
    lines = []

    for f in files:
        if not f.sources:
            continue

        # First line: primary source
        lines.append(f.sources[0])

        # Output path
        rel_path = Path(f.path)
        out_dir = base_dir / rel_path.parent
        lines.append(f"  dir={out_dir}")
        lines.append(f"  out={rel_path.name}")

        # Additional sources as mirrors
        for source in f.sources[1:]:
            lines.append(f"  {source}")

        lines.append("")  # Blank line between entries

    output_path.write_text("\n".join(lines))
    return len(files)


def batch_download_aria2(
    input_file: Path,
    connections: int = ARIA2_CONNECTIONS,
    max_concurrent: int = 5,
) -> tuple[int, int]:
    """Run batch download using aria2c input file."""
    if not input_file.exists():
        return 0, 0

    cmd = [
        "aria2c",
        "--input-file", str(input_file),
        "--max-connection-per-server", str(connections),
        "--split", str(ARIA2_SPLIT),
        "--max-concurrent-downloads", str(max_concurrent),
        "--continue", "true",
        "--auto-file-renaming", "false",
        "--allow-overwrite", "true",
        "--summary-interval", "10",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        # Parse output for success/failure counts
        success = result.stdout.count("Download complete")
        failure = result.stdout.count("Download failed") + result.stdout.count("ERROR")

        return success, failure

    except subprocess.TimeoutExpired:
        logger.error("Batch download timeout")
        return 0, 1
    except Exception as e:
        logger.error(f"Batch download error: {e}")
        return 0, 1


# ============================================
# Sync Operations
# ============================================

def plan_sync(
    remote_files: dict[str, DataFile],
    local_inventory: NodeDataInventory,
    categories: list[str] | None = None,
    max_age_hours: float | None = None,
    newer_only: bool = True,
) -> SyncPlan:
    """Plan which files to download."""
    plan = SyncPlan()

    for path, remote_file in remote_files.items():
        # Category filter
        if categories and remote_file.category not in categories:
            plan.files_to_skip.append(f"{path} (category)")
            continue

        # Age filter
        if max_age_hours and remote_file.age_hours > max_age_hours:
            plan.files_to_skip.append(f"{path} (too old)")
            continue

        # Check if we have it locally
        local_file = local_inventory.files.get(path)

        if local_file:
            if newer_only and remote_file.mtime <= local_file.mtime:
                plan.files_to_skip.append(f"{path} (up to date)")
                continue

            if remote_file.size_bytes == local_file.size_bytes:
                plan.files_to_skip.append(f"{path} (same size)")
                continue

        # Add to download list
        plan.files_to_download.append(remote_file)
        plan.total_size_mb += remote_file.size_bytes / (1024 * 1024)

    # Estimate time (assume 50 MB/s average with aria2)
    plan.estimated_time_min = plan.total_size_mb / 50 / 60

    return plan


def sync_data(
    source_urls: list[str],
    categories: list[str] | None = None,
    max_age_hours: float | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Sync data from cluster sources."""
    if not check_aria2():
        logger.error("aria2c not found. Install with: apt install aria2")
        return 0, 0

    # Discover sources
    inventories = discover_data_sources(source_urls)
    remote_files = aggregate_sources(inventories)

    if not remote_files:
        logger.info("No remote files found")
        return 0, 0

    # Build local inventory
    local_inventory = build_local_inventory()

    # Plan sync
    plan = plan_sync(
        remote_files,
        local_inventory,
        categories=categories,
        max_age_hours=max_age_hours,
    )

    logger.info(f"Sync plan: {len(plan.files_to_download)} files to download ({plan.total_size_mb:.1f} MB)")
    logger.info(f"Skipping: {len(plan.files_to_skip)} files")

    if dry_run:
        for f in plan.files_to_download:
            logger.info(f"  Would download: {f.path} ({f.size_bytes / 1024 / 1024:.1f} MB)")
        return len(plan.files_to_download), 0

    if not plan.files_to_download:
        logger.info("Nothing to sync")
        return 0, 0

    # Generate aria2 input file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        input_file = Path(f.name)

    generate_aria2_input_file(plan.files_to_download, input_file, ROOT)

    logger.info(f"Starting download of {len(plan.files_to_download)} files...")
    success, failure = batch_download_aria2(input_file)

    # Cleanup
    input_file.unlink(missing_ok=True)

    logger.info(f"Sync complete: {success} succeeded, {failure} failed")
    return success, failure


def sync_games(source_urls: list[str], max_age_hours: float = 168, dry_run: bool = False):
    """Sync selfplay game databases."""
    return sync_data(
        source_urls,
        categories=["games"],
        max_age_hours=max_age_hours,
        dry_run=dry_run,
    )


def sync_models(source_urls: list[str], dry_run: bool = False):
    """Sync model files."""
    return sync_data(
        source_urls,
        categories=["models"],
        dry_run=dry_run,
    )


def sync_training(source_urls: list[str], max_age_hours: float = 24, dry_run: bool = False):
    """Sync training data."""
    return sync_data(
        source_urls,
        categories=["training"],
        max_age_hours=max_age_hours,
        dry_run=dry_run,
    )


def cluster_sync(source_urls: list[str], dry_run: bool = False):
    """Full cluster sync - models and recent data."""
    logger.info("=== Full Cluster Sync ===")

    # Sync models (no age limit)
    logger.info("\n--- Syncing Models ---")
    m_success, m_fail = sync_models(source_urls, dry_run=dry_run)

    # Sync recent games (last week)
    logger.info("\n--- Syncing Games (last 7 days) ---")
    g_success, g_fail = sync_games(source_urls, max_age_hours=168, dry_run=dry_run)

    # Sync training data (last day)
    logger.info("\n--- Syncing Training Data (last 24 hours) ---")
    t_success, t_fail = sync_training(source_urls, max_age_hours=24, dry_run=dry_run)

    total_success = m_success + g_success + t_success
    total_fail = m_fail + g_fail + t_fail

    logger.info(f"\n=== Sync Complete: {total_success} files synced, {total_fail} failed ===")
    return total_success, total_fail


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(description="aria2-based Data Sync for RingRift")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start data server")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_DATA_PORT)

    # sync-games command
    games_parser = subparsers.add_parser("sync-games", help="Sync selfplay games")
    games_parser.add_argument("--sources", type=str, help="Comma-separated source URLs")
    games_parser.add_argument("--days", type=float, default=7, help="Max age in days")
    games_parser.add_argument("--dry-run", action="store_true")

    # sync-models command
    models_parser = subparsers.add_parser("sync-models", help="Sync model files")
    models_parser.add_argument("--sources", type=str, help="Comma-separated source URLs")
    models_parser.add_argument("--dry-run", action="store_true")

    # sync-training command
    training_parser = subparsers.add_parser("sync-training", help="Sync training data")
    training_parser.add_argument("--sources", type=str, help="Comma-separated source URLs")
    training_parser.add_argument("--hours", type=float, default=24, help="Max age in hours")
    training_parser.add_argument("--dry-run", action="store_true")

    # cluster-sync command
    cluster_parser = subparsers.add_parser("cluster-sync", help="Full cluster sync")
    cluster_parser.add_argument("--sources", type=str, help="Comma-separated source URLs")
    cluster_parser.add_argument("--dry-run", action="store_true")

    # inventory command
    inv_parser = subparsers.add_parser("inventory", help="Show local data inventory")
    inv_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # discover command
    disc_parser = subparsers.add_parser("discover", help="Discover cluster nodes")
    disc_parser.add_argument("--sources", type=str, help="Comma-separated source URLs")

    args = parser.parse_args()

    def get_sources(args_sources: str | None) -> list[str]:
        if args_sources:
            return [s.strip() for s in args_sources.split(",")]
        return CLUSTER_DATA_SOURCES

    if args.command == "serve":
        run_data_server(port=args.port)

    elif args.command == "sync-games":
        sources = get_sources(args.sources)
        sync_games(sources, max_age_hours=args.days * 24, dry_run=args.dry_run)

    elif args.command == "sync-models":
        sources = get_sources(args.sources)
        sync_models(sources, dry_run=args.dry_run)

    elif args.command == "sync-training":
        sources = get_sources(args.sources)
        sync_training(sources, max_age_hours=args.hours, dry_run=args.dry_run)

    elif args.command == "cluster-sync":
        sources = get_sources(args.sources)
        cluster_sync(sources, dry_run=args.dry_run)

    elif args.command == "inventory":
        inventory = build_local_inventory()
        if args.json:
            output = {
                "hostname": inventory.hostname,
                "total_size_mb": round(inventory.total_size_mb, 2),
                "files": {
                    path: {
                        "name": f.name,
                        "size_mb": round(f.size_bytes / 1024 / 1024, 2),
                        "category": f.category,
                        "age_hours": round(f.age_hours, 1),
                    }
                    for path, f in inventory.files.items()
                },
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Local Data Inventory ({inventory.hostname})")
            print(f"Total: {len(inventory.files)} files, {inventory.total_size_mb:.1f} MB")
            print()
            for cat in ["games", "models", "training", "elo"]:
                files = [f for f in inventory.files.values() if f.category == cat]
                if files:
                    total_mb = sum(f.size_bytes for f in files) / 1024 / 1024
                    print(f"  {cat}: {len(files)} files ({total_mb:.1f} MB)")

    elif args.command == "discover":
        sources = get_sources(args.sources)
        inventories = discover_data_sources(sources)

        print("\nCluster Data Summary:")
        for url, inv in inventories.items():
            if inv.reachable:
                print(f"  {inv.hostname or url}: {len(inv.files)} files ({inv.total_size_mb:.1f} MB)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
