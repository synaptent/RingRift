#!/usr/bin/env python3
"""P2P Model Distribution using aria2 for resilient multi-source downloads.

DEPRECATED (2025-12-16): This script should be consolidated into aria2_data_sync.py.
Use aria2_data_sync.py for all aria2-based sync operations.
See docs/RESOURCE_MANAGEMENT.md for sync tool consolidation notes.

This script enables efficient model distribution across distributed nodes using:
- HTTP serving mode: Any node can serve models over HTTP
- aria2c downloads: Parallel, resumable downloads from multiple sources
- Metalink generation: Create download manifests listing all available sources
- Auto-discovery: Find which nodes have which models

Features:
- Resilient to connection drops (aria2 auto-resume)
- Multi-source parallel download (faster than single-source)
- Works with unstable SSH connections (uses HTTP instead)
- Simple HTTP server - no complex setup required

Usage:
    # Start HTTP server on a node (serves models directory)
    python p2p_model_sync.py serve --port 8765

    # Download models from available sources
    python p2p_model_sync.py download --sources "http://host1:8765,http://host2:8765"

    # Generate metalink file for aria2c
    python p2p_model_sync.py metalink --output models.metalink

    # Sync all models across cluster
    python p2p_model_sync.py sync

    # Install aria2 on current node
    python p2p_model_sync.py install-aria2
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import logging
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin
from urllib.request import urlopen
from urllib.error import URLError

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

MODELS_DIR = ROOT / "models"
DEFAULT_HTTP_PORT = 8765
ARIA2_CONNECTIONS = 16  # Connections per server
ARIA2_SPLIT = 4  # Segments per file
ARIA2_TIMEOUT = 60

# Known cluster hosts (can be extended via config)
# Format: name -> (http_url, ssh_info)
DEFAULT_SOURCES = [
    # Tailscale IPs
    "http://100.78.101.123:8765",  # Lambda-H100
    "http://100.123.183.70:8765",  # Lambda-2xH100
    # Vast instances (via reverse tunnel or direct IP)
]


# ============================================
# Data Classes
# ============================================

@dataclass
class ModelFile:
    """Information about a model file."""
    name: str
    size_bytes: int
    md5_hash: Optional[str] = None
    sources: List[str] = field(default_factory=list)


@dataclass
class NodeInventory:
    """Model inventory for a node."""
    url: str
    models: Dict[str, ModelFile] = field(default_factory=dict)
    reachable: bool = False
    last_check: Optional[str] = None


# ============================================
# HTTP Server for Model Serving
# ============================================

class ModelHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves models directory and inventory."""

    def __init__(self, *args, models_dir: Path = MODELS_DIR, **kwargs):
        self.models_dir = models_dir
        super().__init__(*args, directory=str(models_dir), **kwargs)

    def do_GET(self):
        if self.path == "/inventory.json":
            self.send_inventory()
        else:
            super().do_GET()

    def send_inventory(self):
        """Send JSON inventory of available models."""
        inventory = {}
        for pth in self.models_dir.glob("*.pth"):
            inventory[pth.name] = {
                "size": pth.stat().st_size,
                "mtime": pth.stat().st_mtime,
            }
        for pth in self.models_dir.glob("*.onnx"):
            inventory[pth.name] = {
                "size": pth.stat().st_size,
                "mtime": pth.stat().st_mtime,
            }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(inventory, indent=2).encode())

    def log_message(self, format, *args):
        logger.debug(f"HTTP: {args[0]}")


def run_http_server(port: int = DEFAULT_HTTP_PORT, models_dir: Path = MODELS_DIR):
    """Run HTTP server to serve models."""
    handler = lambda *args, **kwargs: ModelHTTPHandler(*args, models_dir=models_dir, **kwargs)

    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        logger.info(f"Serving models from {models_dir} on port {port}")
        logger.info(f"Inventory: http://0.0.0.0:{port}/inventory.json")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")


# ============================================
# Node Discovery
# ============================================

def check_node(url: str, timeout: int = 5) -> Optional[NodeInventory]:
    """Check if a node is reachable and get its inventory."""
    inventory = NodeInventory(url=url)

    try:
        inv_url = urljoin(url.rstrip("/") + "/", "inventory.json")
        with urlopen(inv_url, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            inventory.reachable = True
            inventory.last_check = datetime.now().isoformat()

            for name, info in data.items():
                inventory.models[name] = ModelFile(
                    name=name,
                    size_bytes=info.get("size", 0),
                    sources=[urljoin(url.rstrip("/") + "/", name)],
                )

    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.debug(f"Node {url} unreachable: {e}")
        inventory.reachable = False

    return inventory


def discover_sources(source_urls: List[str]) -> Dict[str, NodeInventory]:
    """Discover all available model sources in parallel."""
    inventories = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(check_node, url): url for url in source_urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                inv = future.result()
                inventories[url] = inv
                if inv.reachable:
                    logger.info(f"  {url}: {len(inv.models)} models available")
                else:
                    logger.debug(f"  {url}: unreachable")
            except Exception as e:
                logger.error(f"  {url}: error - {e}")

    return inventories


def get_all_models_with_sources(inventories: Dict[str, NodeInventory]) -> Dict[str, ModelFile]:
    """Aggregate models from all sources."""
    all_models: Dict[str, ModelFile] = {}

    for url, inv in inventories.items():
        if not inv.reachable:
            continue

        for name, model in inv.models.items():
            if name not in all_models:
                all_models[name] = ModelFile(
                    name=name,
                    size_bytes=model.size_bytes,
                    sources=[],
                )
            # Add this source
            source_url = urljoin(url.rstrip("/") + "/", name)
            if source_url not in all_models[name].sources:
                all_models[name].sources.append(source_url)

    return all_models


# ============================================
# aria2c Integration
# ============================================

def check_aria2():
    """Check if aria2c is available."""
    return shutil.which("aria2c") is not None


def install_aria2():
    """Install aria2c on the current system."""
    logger.info("Installing aria2...")

    # Try apt (Debian/Ubuntu)
    result = subprocess.run(
        ["apt-get", "update", "-qq"],
        capture_output=True,
        timeout=60,
    )
    result = subprocess.run(
        ["apt-get", "install", "-y", "-qq", "aria2"],
        capture_output=True,
        timeout=120,
    )
    if result.returncode == 0:
        logger.info("aria2 installed successfully via apt")
        return True

    # Try yum (RHEL/CentOS)
    result = subprocess.run(
        ["yum", "install", "-y", "aria2"],
        capture_output=True,
        timeout=120,
    )
    if result.returncode == 0:
        logger.info("aria2 installed successfully via yum")
        return True

    # Try brew (macOS)
    result = subprocess.run(
        ["brew", "install", "aria2"],
        capture_output=True,
        timeout=120,
    )
    if result.returncode == 0:
        logger.info("aria2 installed successfully via brew")
        return True

    logger.error("Could not install aria2. Please install manually.")
    return False


def download_with_aria2(
    model_name: str,
    sources: List[str],
    output_dir: Path,
    connections: int = ARIA2_CONNECTIONS,
    split: int = ARIA2_SPLIT,
) -> Tuple[bool, str]:
    """Download a file using aria2c with multiple sources."""
    if not sources:
        return False, "No sources available"

    output_path = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build aria2c command
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
        "--continue", "true",  # Resume partial downloads
        "--dir", str(output_dir),
        "--out", model_name,
    ]

    # Add all sources (aria2 will try them in parallel)
    cmd.extend(sources)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per file
        )

        if result.returncode == 0 and output_path.exists():
            return True, f"Downloaded {model_name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
        else:
            return False, result.stderr[:200] if result.stderr else "Download failed"

    except subprocess.TimeoutExpired:
        return False, "Download timeout"
    except Exception as e:
        return False, str(e)[:200]


def generate_metalink(
    models: Dict[str, ModelFile],
    output_path: Path,
):
    """Generate a metalink XML file for aria2c."""
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<metalink xmlns="urn:ietf:params:xml:ns:metalink">',
    ]

    for name, model in models.items():
        if not model.sources:
            continue

        xml_lines.append(f'  <file name="{name}">')
        xml_lines.append(f'    <size>{model.size_bytes}</size>')

        for i, url in enumerate(model.sources):
            priority = i + 1  # Lower is better
            xml_lines.append(f'    <url priority="{priority}">{url}</url>')

        xml_lines.append('  </file>')

    xml_lines.append('</metalink>')

    output_path.write_text("\n".join(xml_lines))
    logger.info(f"Generated metalink: {output_path}")


def download_metalink(
    metalink_path: Path,
    output_dir: Path,
    connections: int = ARIA2_CONNECTIONS,
) -> Tuple[int, int]:
    """Download all files from a metalink."""
    cmd = [
        "aria2c",
        "--max-connection-per-server", str(connections),
        "--split", str(ARIA2_SPLIT),
        "--continue", "true",
        "--dir", str(output_dir),
        "-M", str(metalink_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        # Parse output to get success/failure counts
        return 1 if result.returncode == 0 else 0, 0 if result.returncode == 0 else 1
    except Exception as e:
        logger.error(f"Metalink download failed: {e}")
        return 0, 1


# ============================================
# Sync Operations
# ============================================

def sync_models(
    source_urls: List[str],
    output_dir: Path = MODELS_DIR,
    max_parallel: int = 4,
) -> Tuple[int, int, List[str]]:
    """Sync all missing models from available sources."""
    if not check_aria2():
        logger.warning("aria2c not found. Attempting to install...")
        if not install_aria2():
            return 0, 0, ["aria2c not available"]

    # Discover available sources
    logger.info(f"Discovering models from {len(source_urls)} potential sources...")
    inventories = discover_sources(source_urls)

    reachable_count = sum(1 for inv in inventories.values() if inv.reachable)
    if reachable_count == 0:
        return 0, 0, ["No reachable sources found"]

    logger.info(f"Found {reachable_count} reachable sources")

    # Get all available models with their sources
    all_models = get_all_models_with_sources(inventories)
    logger.info(f"Total models available: {len(all_models)}")

    # Find models we're missing locally
    local_models = set()
    if output_dir.exists():
        local_models = {f.name for f in output_dir.glob("*.pth")}
        local_models.update(f.name for f in output_dir.glob("*.onnx"))

    missing_models = {
        name: model for name, model in all_models.items()
        if name not in local_models
    }

    if not missing_models:
        logger.info("All models already present locally")
        return 0, 0, []

    logger.info(f"Downloading {len(missing_models)} missing models...")

    # Download missing models
    downloaded = 0
    failed = 0
    errors = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_model = {
            executor.submit(
                download_with_aria2,
                name,
                model.sources,
                output_dir,
            ): name
            for name, model in missing_models.items()
        }

        for future in as_completed(future_to_model):
            name = future_to_model[future]
            try:
                success, msg = future.result()
                if success:
                    downloaded += 1
                    logger.info(f"  {msg}")
                else:
                    failed += 1
                    errors.append(f"{name}: {msg}")
                    logger.warning(f"  Failed: {name} - {msg}")
            except Exception as e:
                failed += 1
                errors.append(f"{name}: {e}")

    return downloaded, failed, errors


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="P2P Model Distribution with aria2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="Port number")
    serve_parser.add_argument("--dir", type=str, default=str(MODELS_DIR), help="Models directory")

    # download command
    download_parser = subparsers.add_parser("download", help="Download models from sources")
    download_parser.add_argument("--sources", type=str, required=True, help="Comma-separated source URLs")
    download_parser.add_argument("--output", type=str, default=str(MODELS_DIR), help="Output directory")
    download_parser.add_argument("--parallel", type=int, default=4, help="Parallel downloads")

    # metalink command
    metalink_parser = subparsers.add_parser("metalink", help="Generate metalink file")
    metalink_parser.add_argument("--sources", type=str, required=True, help="Comma-separated source URLs")
    metalink_parser.add_argument("--output", type=str, default="models.metalink", help="Output file")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync all models from available sources")
    sync_parser.add_argument("--sources", type=str, help="Comma-separated source URLs (optional)")
    sync_parser.add_argument("--output", type=str, default=str(MODELS_DIR), help="Output directory")
    sync_parser.add_argument("--parallel", type=int, default=4, help="Parallel downloads")

    # install-aria2 command
    subparsers.add_parser("install-aria2", help="Install aria2c")

    # check command
    check_parser = subparsers.add_parser("check", help="Check available sources")
    check_parser.add_argument("--sources", type=str, required=True, help="Comma-separated source URLs")

    args = parser.parse_args()

    if args.command == "serve":
        run_http_server(port=args.port, models_dir=Path(args.dir))

    elif args.command == "download":
        sources = [s.strip() for s in args.sources.split(",")]
        downloaded, failed, errors = sync_models(
            sources,
            output_dir=Path(args.output),
            max_parallel=args.parallel,
        )
        print(f"\nDownloaded: {downloaded}, Failed: {failed}")
        if errors:
            print("Errors:")
            for err in errors[:10]:
                print(f"  - {err}")

    elif args.command == "metalink":
        sources = [s.strip() for s in args.sources.split(",")]
        inventories = discover_sources(sources)
        all_models = get_all_models_with_sources(inventories)
        generate_metalink(all_models, Path(args.output))

    elif args.command == "sync":
        sources = DEFAULT_SOURCES.copy()
        if args.sources:
            sources.extend(s.strip() for s in args.sources.split(","))

        downloaded, failed, errors = sync_models(
            sources,
            output_dir=Path(args.output),
            max_parallel=args.parallel,
        )
        print(f"\nSync complete: {downloaded} downloaded, {failed} failed")

    elif args.command == "install-aria2":
        if check_aria2():
            print("aria2c is already installed")
        else:
            install_aria2()

    elif args.command == "check":
        sources = [s.strip() for s in args.sources.split(",")]
        inventories = discover_sources(sources)
        print("\nSource Status:")
        for url, inv in inventories.items():
            status = f"{len(inv.models)} models" if inv.reachable else "UNREACHABLE"
            print(f"  {url}: {status}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
