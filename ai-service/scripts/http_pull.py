#!/usr/bin/env python3
"""HTTP-based file pull utility for RingRift cluster.

Downloads files from P2P nodes via HTTP when SSH/SCP is unreliable.
This is a permanent workaround for nodes with connectivity issues.

Usage:
    # Pull a model from the leader
    python scripts/http_pull.py --model canonical_hex8_2p.pth

    # Pull from a specific node
    python scripts/http_pull.py --model canonical_hex8_2p.pth --source 192.168.1.100:8770

    # Pull Elo database
    python scripts/http_pull.py --data unified_elo.db

    # Pull training data
    python scripts/http_pull.py --data training/hex8_2p.npz

    # List available files
    python scripts/http_pull.py --list

    # Pull all canonical models
    python scripts/http_pull.py --sync-models

    # Pull Elo DB from any healthy node
    python scripts/http_pull.py --sync-elo

December 2025: Created as SSH connectivity workaround.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# HTTP client - try aiohttp first, fall back to urllib
try:
    import aiohttp
    import asyncio
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    import urllib.request
    import urllib.error


def get_default_source() -> str:
    """Get default source node from environment or P2P leader."""
    # Check environment
    source = os.environ.get("RINGRIFT_HTTP_PULL_SOURCE")
    if source:
        return source

    # Try to get leader from local P2P status
    try:
        import urllib.request
        import json
        with urllib.request.urlopen("http://localhost:8770/status", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            leader_id = data.get("leader_id")
            peers = data.get("peers", {})
            if leader_id and leader_id in peers:
                peer = peers[leader_id]
                return f"{peer.get('host', '127.0.0.1')}:{peer.get('port', 8770)}"
    except Exception:
        pass

    return "localhost:8770"


def download_file_urllib(url: str, dest_path: Path, verify_sha256: Optional[str] = None) -> bool:
    """Download file using urllib (sync, fallback when aiohttp unavailable)."""
    import urllib.request
    import urllib.error

    try:
        logger.info(f"Downloading {url} -> {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with progress
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            sha256 = hashlib.sha256() if verify_sha256 else None
            start_time = time.time()

            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    if sha256:
                        sha256.update(chunk)
                    downloaded += len(chunk)

                    # Progress indicator
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        print(f"\r  {pct:.1f}% ({downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB) @ {speed:.1f} MB/s", end="", flush=True)

            print()  # Newline after progress

            # Verify checksum
            if verify_sha256 and sha256:
                actual_sha256 = sha256.hexdigest()
                if actual_sha256 != verify_sha256:
                    logger.error(f"Checksum mismatch: expected {verify_sha256}, got {actual_sha256}")
                    dest_path.unlink()
                    return False
                logger.info(f"Checksum verified: {actual_sha256[:16]}...")

            logger.info(f"Downloaded {dest_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
            return True

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error: {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        logger.error(f"URL error: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


async def download_file_aiohttp(url: str, dest_path: Path, verify_sha256: Optional[str] = None) -> bool:
    """Download file using aiohttp (async, preferred)."""
    try:
        logger.info(f"Downloading {url} -> {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                if resp.status != 200:
                    logger.error(f"HTTP {resp.status}: {await resp.text()}")
                    return False

                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                sha256 = hashlib.sha256() if verify_sha256 else None
                start_time = time.time()

                with open(dest_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                        f.write(chunk)
                        if sha256:
                            sha256.update(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            pct = downloaded / total_size * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                            print(f"\r  {pct:.1f}% ({downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB) @ {speed:.1f} MB/s", end="", flush=True)

                print()

                if verify_sha256 and sha256:
                    actual_sha256 = sha256.hexdigest()
                    if actual_sha256 != verify_sha256:
                        logger.error(f"Checksum mismatch: expected {verify_sha256}, got {actual_sha256}")
                        dest_path.unlink()
                        return False
                    logger.info(f"Checksum verified: {actual_sha256[:16]}...")

                logger.info(f"Downloaded {dest_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
                return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_file(url: str, dest_path: Path, verify_sha256: Optional[str] = None) -> bool:
    """Download file using best available method."""
    if HAS_AIOHTTP:
        return asyncio.run(download_file_aiohttp(url, dest_path, verify_sha256))
    else:
        return download_file_urllib(url, dest_path, verify_sha256)


def list_files(source: str, file_type: str = "all") -> list[dict]:
    """List available files from source node."""
    import urllib.request
    import json

    url = f"http://{source}/files/list?type={file_type}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("files", [])
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return []


def get_file_info(source: str, path: str, file_type: str) -> Optional[dict]:
    """Get info about a specific file."""
    import urllib.request
    import json

    url = f"http://{source}/files/info?path={path}&type={file_type}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        return None


def find_healthy_node_with_file(path: str, file_type: str) -> Optional[str]:
    """Find any healthy P2P node that has the requested file."""
    import urllib.request
    import json

    try:
        # Get cluster status
        with urllib.request.urlopen("http://localhost:8770/status", timeout=10) as resp:
            status = json.loads(resp.read().decode())

        peers = status.get("peers", {})
        for node_id, peer in peers.items():
            if peer.get("status") != "alive":
                continue

            host = peer.get("host")
            port = peer.get("port", 8770)
            source = f"{host}:{port}"

            # Check if node has the file
            info = get_file_info(source, path, file_type)
            if info and "error" not in info:
                logger.info(f"Found {path} on node {node_id}")
                return source

    except Exception as e:
        logger.warning(f"Failed to search cluster: {e}")

    return None


def sync_canonical_models(source: str, dest_dir: Path) -> int:
    """Sync all canonical models from source."""
    files = list_files(source, "models")
    synced = 0

    for f in files:
        path = f["path"]
        if not path.startswith("canonical_"):
            continue

        dest_path = dest_dir / path
        if dest_path.exists():
            # Check if sizes match
            if dest_path.stat().st_size == f["size"]:
                logger.info(f"Skipping {path} (already exists)")
                continue

        url = f"http://{source}{f['url']}"
        info = get_file_info(source, path, "model")
        sha256 = info.get("sha256") if info else None

        if download_file(url, dest_path, sha256):
            synced += 1

    return synced


def sync_elo_database(dest_path: Path) -> bool:
    """Sync Elo database from any healthy node."""
    source = find_healthy_node_with_file("unified_elo.db", "data")
    if not source:
        # Fall back to leader
        source = get_default_source()

    url = f"http://{source}/files/data/unified_elo.db"
    info = get_file_info(source, "unified_elo.db", "data")
    sha256 = info.get("sha256") if info else None

    return download_file(url, dest_path, sha256)


def main():
    parser = argparse.ArgumentParser(
        description="HTTP-based file pull for RingRift cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pull a model
    python scripts/http_pull.py --model canonical_hex8_2p.pth

    # Pull from specific node
    python scripts/http_pull.py --model canonical_hex8_2p.pth --source 192.168.1.100:8770

    # List available files
    python scripts/http_pull.py --list

    # Sync all canonical models
    python scripts/http_pull.py --sync-models

    # Sync Elo database
    python scripts/http_pull.py --sync-elo
        """,
    )

    parser.add_argument("--source", default=None, help="Source node (host:port). Default: auto-detect leader")
    parser.add_argument("--model", help="Model file to download (relative to models/)")
    parser.add_argument("--data", help="Data file to download (relative to data/)")
    parser.add_argument("--output", "-o", help="Output path (default: models/ or data/)")
    parser.add_argument("--list", action="store_true", help="List available files")
    parser.add_argument("--type", choices=["models", "data", "all"], default="all", help="File type for --list")
    parser.add_argument("--sync-models", action="store_true", help="Sync all canonical models")
    parser.add_argument("--sync-elo", action="store_true", help="Sync Elo database")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify SHA256 checksum")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="Skip checksum verification")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source = args.source or get_default_source()
    logger.info(f"Using source: {source}")

    # Find ai-service root
    ai_service_root = Path(__file__).parent.parent
    models_dir = ai_service_root / "models"
    data_dir = ai_service_root / "data"

    if args.list:
        files = list_files(source, args.type)
        if not files:
            print("No files found")
            return 1

        print(f"\nAvailable files ({len(files)} total):\n")
        for f in files:
            size_mb = f["size"] / 1024 / 1024
            print(f"  [{f['type']:6}] {f['path']:40} ({size_mb:7.1f} MB)")
        return 0

    if args.sync_models:
        logger.info("Syncing canonical models...")
        synced = sync_canonical_models(source, models_dir)
        logger.info(f"Synced {synced} models")
        return 0 if synced >= 0 else 1

    if args.sync_elo:
        logger.info("Syncing Elo database...")
        dest_path = data_dir / "unified_elo.db"
        success = sync_elo_database(dest_path)
        return 0 if success else 1

    if args.model:
        info = get_file_info(source, args.model, "model") if args.verify else None
        sha256 = info.get("sha256") if info else None

        url = f"http://{source}/files/models/{args.model}"
        dest_path = Path(args.output) if args.output else models_dir / args.model

        success = download_file(url, dest_path, sha256)
        return 0 if success else 1

    if args.data:
        info = get_file_info(source, args.data, "data") if args.verify else None
        sha256 = info.get("sha256") if info else None

        url = f"http://{source}/files/data/{args.data}"
        dest_path = Path(args.output) if args.output else data_dir / args.data

        success = download_file(url, dest_path, sha256)
        return 0 if success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
