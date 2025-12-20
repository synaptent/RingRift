#!/usr/bin/env python3
"""
P2P Model Distribution Utilities

Fast model distribution using aria2c multi-connection downloads with BitTorrent fallback.
Supports HTTP server mode (source), aria2 download mode (destination), and BitTorrent swarm sync.

Usage:
    # On source machine (has models):
    python p2p_model_distribution.py serve --models-dir /path/to/models --port 8765

    # On destination machine (needs models):
    python p2p_model_distribution.py download --source-ip <SOURCE_IP> --port 8765 \
        --dest-dir /dev/shm/ringrift/ai-service/models --connections 16

    # Generate URL list for manual aria2 use:
    python p2p_model_distribution.py generate-urls --source-ip <SOURCE_IP> --port 8765 \
        --models-dir /path/to/models --output urls.txt

    # Create torrent for models (BitTorrent distribution):
    python p2p_model_distribution.py create-torrent --models-dir /path/to/models \
        --output models.torrent --tracker http://tracker.example.com:6969/announce

    # Seed models via BitTorrent:
    python p2p_model_distribution.py seed --torrent models.torrent --models-dir /path/to/models

    # Download models via BitTorrent:
    python p2p_model_distribution.py bt-download --torrent models.torrent \
        --dest-dir /path/to/dest --seeders node1:51413,node2:51413
"""

import argparse
import hashlib
import http.server
import json
import socketserver
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

# Try to import bencodepy, fall back to simple implementation
try:
    import bencodepy
    BENCODE_AVAILABLE = True
except ImportError:
    bencodepy = None  # type: ignore
    BENCODE_AVAILABLE = False


def bencode(data: Any) -> bytes:
    """Bencode data for torrent files. Pure Python fallback if bencodepy unavailable."""
    if BENCODE_AVAILABLE:
        return bencodepy.encode(data)

    # Pure Python bencode implementation
    if isinstance(data, int):
        return f"i{data}e".encode('ascii')
    elif isinstance(data, bytes):
        return f"{len(data)}:".encode('ascii') + data
    elif isinstance(data, str):
        encoded = data.encode('utf-8')
        return f"{len(encoded)}:".encode('ascii') + encoded
    elif isinstance(data, list):
        result = b'l'
        for item in data:
            result += bencode(item)
        return result + b'e'
    elif isinstance(data, dict):
        result = b'd'
        # Keys must be sorted in bencoding
        for key in sorted(data.keys()):
            if isinstance(key, str):
                key_bytes = key.encode('utf-8')
            else:
                key_bytes = key
            result += bencode(key_bytes)
            result += bencode(data[key])
        return result + b'e'
    else:
        raise TypeError(f"Cannot bencode {type(data)}")


def bdecode(data: bytes) -> Any:
    """Decode bencoded data. Pure Python fallback if bencodepy unavailable."""
    if BENCODE_AVAILABLE:
        return bencodepy.decode(data)

    def decode_next(data: bytes, idx: int) -> tuple[Any, int]:
        if data[idx:idx+1] == b'i':
            # Integer
            end = data.index(b'e', idx)
            return int(data[idx+1:end]), end + 1
        elif data[idx:idx+1] == b'l':
            # List
            result = []
            idx += 1
            while data[idx:idx+1] != b'e':
                item, idx = decode_next(data, idx)
                result.append(item)
            return result, idx + 1
        elif data[idx:idx+1] == b'd':
            # Dictionary
            result = {}
            idx += 1
            while data[idx:idx+1] != b'e':
                key, idx = decode_next(data, idx)
                value, idx = decode_next(data, idx)
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                result[key] = value
            return result, idx + 1
        elif data[idx:idx+1].isdigit():
            # String/bytes
            colon = data.index(b':', idx)
            length = int(data[idx:colon])
            start = colon + 1
            return data[start:start+length], start + length
        else:
            raise ValueError(f"Invalid bencoded data at {idx}")

    result, _ = decode_next(data, 0)
    return result


# BitTorrent constants
PIECE_SIZE = 256 * 1024  # 256 KB pieces (good for model files)
BT_PORT = 51413  # Default BitTorrent port


def get_tailscale_ip() -> str | None:
    """Get the local Tailscale IP address."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass
    return None


def list_models(models_dir: Path) -> list[str]:
    """List all .pth model files in directory."""
    return sorted([f.name for f in models_dir.glob("*.pth")])


class TorrentCreator:
    """Create .torrent files for model distribution."""

    def __init__(
        self,
        piece_size: int = PIECE_SIZE,
        trackers: list[str] | None = None,
        comment: str = "RingRift model distribution",
    ):
        self.piece_size = piece_size
        self.trackers = trackers or []
        self.comment = comment

    def create_torrent(
        self,
        models_dir: Path,
        output_path: Path,
        web_seeds: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a torrent file for all models in directory.

        Args:
            models_dir: Directory containing .pth model files
            output_path: Where to save the .torrent file
            web_seeds: Optional HTTP URLs for web seeding (hybrid HTTP+BT)

        Returns:
            Dict with torrent info (info_hash, file_count, total_size)
        """
        models = sorted(models_dir.glob("*.pth"))
        if not models:
            raise ValueError(f"No .pth files found in {models_dir}")

        # Build file list and calculate pieces
        files_info = []
        all_pieces = b""
        current_piece_data = b""
        total_size = 0

        for model_path in models:
            file_size = model_path.stat().st_size
            files_info.append({
                b"length": file_size,
                b"path": [model_path.name.encode("utf-8")],
            })
            total_size += file_size

            # Read file and calculate pieces
            with open(model_path, "rb") as f:
                while True:
                    chunk = f.read(self.piece_size - len(current_piece_data))
                    if not chunk:
                        break
                    current_piece_data += chunk
                    if len(current_piece_data) >= self.piece_size:
                        all_pieces += hashlib.sha1(current_piece_data).digest()
                        current_piece_data = b""

        # Handle final piece
        if current_piece_data:
            all_pieces += hashlib.sha1(current_piece_data).digest()

        # Build torrent info dictionary
        info = {
            b"name": models_dir.name.encode("utf-8"),
            b"piece length": self.piece_size,
            b"pieces": all_pieces,
            b"files": files_info,
        }

        # Calculate info hash (this is the torrent identifier)
        info_encoded = bencode(info)
        info_hash = hashlib.sha1(info_encoded).hexdigest()

        # Build full torrent dictionary
        torrent = {
            b"info": info,
            b"created by": b"RingRift P2P Model Distribution",
            b"creation date": int(time.time()),
            b"comment": self.comment.encode("utf-8"),
        }

        # Add trackers
        if self.trackers:
            if len(self.trackers) == 1:
                torrent[b"announce"] = self.trackers[0].encode("utf-8")
            else:
                torrent[b"announce"] = self.trackers[0].encode("utf-8")
                torrent[b"announce-list"] = [
                    [t.encode("utf-8")] for t in self.trackers
                ]

        # Add web seeds for hybrid HTTP+BT downloads
        if web_seeds:
            torrent[b"url-list"] = [ws.encode("utf-8") for ws in web_seeds]

        # Write torrent file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(bencode(torrent))

        return {
            "info_hash": info_hash,
            "file_count": len(files_info),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "torrent_path": str(output_path),
        }


def create_torrent_file(
    models_dir: Path,
    output_path: Path,
    trackers: list[str] | None = None,
    web_seeds: list[str] | None = None,
) -> dict[str, Any]:
    """Create a .torrent file for models.

    For trackerless operation (DHT-only), omit trackers.
    For hybrid HTTP+BT, provide web_seeds pointing to HTTP server.
    """
    creator = TorrentCreator(trackers=trackers)
    return creator.create_torrent(models_dir, output_path, web_seeds)


def seed_torrent(
    torrent_path: Path,
    models_dir: Path,
    port: int = BT_PORT,
    enable_dht: bool = True,
    listen_all: bool = False,
) -> subprocess.Popen:
    """Start seeding models via BitTorrent using aria2c.

    Args:
        torrent_path: Path to .torrent file
        models_dir: Directory containing model files
        port: BitTorrent listen port
        enable_dht: Enable DHT for trackerless operation
        listen_all: Listen on all interfaces (for external connections)

    Returns:
        Popen process object for the aria2c seeder
    """
    cmd = [
        "aria2c",
        f"--dir={models_dir}",
        f"--listen-port={port}",
        "--seed-ratio=0",  # Seed forever
        "--bt-seed-unverified=true",  # Don't re-hash existing files
        "--bt-hash-check-seed=false",  # Skip hash check for seeding
        "--enable-peer-exchange=true",  # Enable PEX for peer discovery
        "--console-log-level=warn",
    ]

    if enable_dht:
        cmd.append("--enable-dht=true")
        cmd.append("--dht-listen-port=" + str(port))

    if listen_all:
        cmd.append("--interface=0.0.0.0")

    cmd.append(str(torrent_path))

    print(f"Starting BitTorrent seeder on port {port}")
    print(f"Seeding: {models_dir}")

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def download_via_bittorrent(
    torrent_path: Path,
    dest_dir: Path,
    seeders: list[str] | None = None,
    port: int = BT_PORT,
    enable_dht: bool = True,
    web_seed_fallback: bool = True,
) -> bool:
    """Download models via BitTorrent using aria2c.

    Args:
        torrent_path: Path to .torrent file
        dest_dir: Destination directory for downloaded models
        seeders: Optional list of known seeders (ip:port format)
        port: BitTorrent listen port
        enable_dht: Enable DHT for peer discovery
        web_seed_fallback: Fall back to web seeds in torrent if available

    Returns:
        True if download successful, False otherwise
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        f"--dir={dest_dir}",
        f"--listen-port={port}",
        "--seed-ratio=0.0",  # Don't seed after download (can change to seed)
        "--bt-stop-timeout=60",  # Stop 60s after download complete
        "--enable-peer-exchange=true",
        "--console-log-level=warn",
        "--summary-interval=10",
    ]

    if enable_dht:
        cmd.append("--enable-dht=true")
        cmd.append("--dht-listen-port=" + str(port))

    # Add known seeders for faster initial connection
    if seeders:
        seeder_str = ",".join(seeders)
        cmd.append(f"--bt-external-ip={get_tailscale_ip() or '0.0.0.0'}")

    if not web_seed_fallback:
        cmd.append("--bt-enable-hook-after-hash-check=false")

    cmd.append(str(torrent_path))

    print(f"Starting BitTorrent download to {dest_dir}")
    if seeders:
        print(f"Known seeders: {seeders}")

    try:
        result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("BitTorrent download timed out")
        return False
    except Exception as e:
        print(f"BitTorrent download failed: {e}")
        return False


def sync_between_nodes(
    nodes: list[str],
    models_dir: str,
    torrent_dir: str | None = None,
    http_port: int = 8765,
    bt_port: int = BT_PORT,
) -> dict[str, Any]:
    """Sync models between multiple nodes using BitTorrent swarm.

    Creates a hybrid HTTP+BitTorrent distribution:
    1. Creates torrent with web seeds pointing to HTTP servers
    2. Each node with models becomes a seeder
    3. Nodes download from swarm + HTTP fallback

    Args:
        nodes: List of node IPs (Tailscale IPs recommended)
        models_dir: Local models directory
        torrent_dir: Where to store .torrent files (default: models_dir/.torrents)
        http_port: HTTP server port for web seeding
        bt_port: BitTorrent port

    Returns:
        Dict with sync results
    """
    models_path = Path(models_dir)
    torrent_path = Path(torrent_dir or models_dir) / ".torrents"
    torrent_path.mkdir(parents=True, exist_ok=True)

    local_ip = get_tailscale_ip()
    if not local_ip:
        print("Warning: Could not determine Tailscale IP, using localhost")
        local_ip = "localhost"

    # Check which nodes have models
    nodes_with_models = []
    for node in nodes:
        try:
            import urllib.request
            url = f"http://{node}:{http_port}/"
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    nodes_with_models.append(node)
        except Exception:
            pass

    if not nodes_with_models:
        print("No nodes with available models found")
        return {"success": False, "error": "No sources available"}

    # Create torrent with web seeds from all available nodes
    web_seeds = [f"http://{node}:{http_port}/" for node in nodes_with_models]
    seeders = [f"{node}:{bt_port}" for node in nodes_with_models]

    # Check if we have models locally
    local_models = list(models_path.glob("*.pth"))

    if local_models:
        # We have models, create torrent and seed
        torrent_file = torrent_path / "models.torrent"

        print(f"Creating torrent for {len(local_models)} local models...")
        info = create_torrent_file(
            models_dir=models_path,
            output_path=torrent_file,
            web_seeds=web_seeds,
        )
        print(f"Torrent created: {info['info_hash'][:16]}... ({info['total_size_mb']:.1f} MB)")

        # Start seeding in background
        seeder_proc = seed_torrent(
            torrent_path=torrent_file,
            models_dir=models_path,
            port=bt_port,
            enable_dht=True,
        )

        return {
            "success": True,
            "mode": "seeding",
            "info_hash": info["info_hash"],
            "file_count": info["file_count"],
            "total_size_mb": info["total_size_mb"],
            "seeder_pid": seeder_proc.pid,
        }
    else:
        # We need models, download from swarm
        # First, try to fetch torrent from a node
        torrent_file = torrent_path / "models.torrent"

        for node in nodes_with_models:
            try:
                import urllib.request
                torrent_url = f"http://{node}:{http_port}/.torrents/models.torrent"
                urllib.request.urlretrieve(torrent_url, torrent_file)
                print(f"Downloaded torrent from {node}")
                break
            except Exception:
                continue
        else:
            # Fall back to HTTP download if no torrent available
            print("No torrent file available, falling back to HTTP download")
            download_models(
                source_ip=nodes_with_models[0],
                port=http_port,
                dest_dir=models_path,
            )
            return {
                "success": True,
                "mode": "http_fallback",
                "source": nodes_with_models[0],
            }

        # Download via BitTorrent with HTTP fallback
        print("Downloading via BitTorrent swarm...")
        success = download_via_bittorrent(
            torrent_path=torrent_file,
            dest_dir=models_path,
            seeders=seeders,
            port=bt_port,
            enable_dht=True,
            web_seed_fallback=True,
        )

        return {
            "success": success,
            "mode": "bittorrent" if success else "failed",
            "seeders": seeders,
        }


def generate_urls(source_ip: str, port: int, models: list[str]) -> list[str]:
    """Generate download URLs for all models."""
    return [f"http://{source_ip}:{port}/{model}" for model in models]


def serve_models(models_dir: Path, port: int):
    """Start HTTP server to serve model files."""
    os.chdir(models_dir)

    handler = http.server.SimpleHTTPRequestHandler

    class QuietHandler(handler):
        def log_message(self, format, *args):
            # Only log errors, not every request
            if args[1] != '200':
                super().log_message(format, *args)

    with http.server.ThreadingHTTPServer(("", port), QuietHandler) as httpd:
        local_ip = get_tailscale_ip() or "localhost"
        models = list_models(models_dir)
        print(f"Serving {len(models)} models from {models_dir}")
        print(f"HTTP server running at http://{local_ip}:{port}")
        print(f"Other nodes can download using:")
        print(f"  python p2p_model_distribution.py download --source-ip {local_ip} --port {port} --dest-dir /path/to/dest")
        print()
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


def download_models(
    source_ip: str,
    port: int,
    dest_dir: Path,
    connections: int = 16,
    max_concurrent: int = 5,
    models_filter: list[str] | None = None
):
    """Download models using aria2c with multiple connections."""

    # First, get the list of available models from source
    import urllib.request
    import re

    try:
        index_url = f"http://{source_ip}:{port}/"
        with urllib.request.urlopen(index_url, timeout=10) as response:
            html = response.read().decode('utf-8')
            # Parse model names from directory listing
            models = re.findall(r'href="([^"]+\.pth)"', html)
    except Exception as e:
        print(f"Error fetching model list: {e}")
        print("Make sure the source server is running and accessible")
        sys.exit(1)

    if not models:
        print("No .pth files found on source server")
        sys.exit(1)

    # Filter models if specified
    if models_filter:
        models = [m for m in models if any(f in m for f in models_filter)]

    # Check which models we already have
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in dest_dir.glob("*.pth"))
    to_download = [m for m in models if m not in existing]

    print(f"Found {len(models)} models on source, {len(existing)} already downloaded")
    print(f"Need to download: {len(to_download)} models")

    if not to_download:
        print("All models already downloaded!")
        return

    # Generate URL file for aria2
    urls = generate_urls(source_ip, port, to_download)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for url in urls:
            f.write(url + '\n')
        url_file = f.name

    try:
        # Check if aria2c is available
        result = subprocess.run(["which", "aria2c"], capture_output=True)
        if result.returncode != 0:
            print("aria2c not found. Installing...")
            subprocess.run(["apt-get", "update", "-qq"], check=False)
            subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], check=True)

        # Download using aria2c
        print(f"\nStarting aria2c download with {connections} connections per file, {max_concurrent} concurrent downloads")
        print(f"Destination: {dest_dir}")
        print()

        cmd = [
            "aria2c",
            f"--input-file={url_file}",
            f"--dir={dest_dir}",
            f"--max-connection-per-server={connections}",
            f"--max-concurrent-downloads={max_concurrent}",
            "--split=16",
            "--min-split-size=1M",
            "--continue=true",
            "--auto-file-renaming=false",
            "--console-log-level=warn",
            "--summary-interval=10",
        ]

        subprocess.run(cmd)

    finally:
        os.unlink(url_file)

    # Verify downloads
    final_count = len(list(dest_dir.glob("*.pth")))
    print(f"\nDownload complete. {final_count} models now in {dest_dir}")


def generate_url_file(source_ip: str, port: int, models_dir: Path, output: Path):
    """Generate aria2 URL file for manual use."""
    models = list_models(models_dir)
    urls = generate_urls(source_ip, port, models)

    with open(output, 'w') as f:
        for url in urls:
            f.write(url + '\n')

    print(f"Generated {len(urls)} URLs in {output}")
    print(f"\nTo download on remote machine:")
    print(f"  aria2c --input-file={output} --dir=/path/to/dest --max-connection-per-server=16 --max-concurrent-downloads=5")


def main():
    parser = argparse.ArgumentParser(description="P2P Model Distribution with BitTorrent support")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Serve command (HTTP)
    serve_parser = subparsers.add_parser('serve', help='Start HTTP server to serve models')
    serve_parser.add_argument('--models-dir', type=Path, required=True, help='Directory containing .pth files')
    serve_parser.add_argument('--port', type=int, default=8765, help='Port to serve on (default: 8765)')

    # Download command (HTTP via aria2)
    dl_parser = subparsers.add_parser('download', help='Download models using aria2c')
    dl_parser.add_argument('--source-ip', required=True, help='Tailscale IP of source machine')
    dl_parser.add_argument('--port', type=int, default=8765, help='Source server port')
    dl_parser.add_argument('--dest-dir', type=Path, required=True, help='Destination directory')
    dl_parser.add_argument('--connections', type=int, default=16, help='Connections per file (default: 16)')
    dl_parser.add_argument('--concurrent', type=int, default=5, help='Concurrent downloads (default: 5)')
    dl_parser.add_argument('--filter', nargs='+', help='Only download models matching these patterns')

    # Generate URLs command
    gen_parser = subparsers.add_parser('generate-urls', help='Generate URL list file')
    gen_parser.add_argument('--source-ip', required=True, help='IP of source machine')
    gen_parser.add_argument('--port', type=int, default=8765, help='Source server port')
    gen_parser.add_argument('--models-dir', type=Path, required=True, help='Directory containing .pth files')
    gen_parser.add_argument('--output', type=Path, required=True, help='Output URL file')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show distribution status')
    status_parser.add_argument('--models-dir', type=Path, required=True, help='Local models directory')

    # === BitTorrent commands ===

    # Create torrent command
    torrent_parser = subparsers.add_parser('create-torrent', help='Create .torrent file for models')
    torrent_parser.add_argument('--models-dir', type=Path, required=True, help='Directory containing .pth files')
    torrent_parser.add_argument('--output', type=Path, required=True, help='Output .torrent file path')
    torrent_parser.add_argument('--tracker', nargs='*', help='Tracker URLs (optional, DHT used if omitted)')
    torrent_parser.add_argument('--web-seed', nargs='*', help='HTTP URLs for web seeding (hybrid mode)')

    # Seed torrent command
    seed_parser = subparsers.add_parser('seed', help='Seed models via BitTorrent')
    seed_parser.add_argument('--torrent', type=Path, required=True, help='Path to .torrent file')
    seed_parser.add_argument('--models-dir', type=Path, required=True, help='Directory containing model files')
    seed_parser.add_argument('--port', type=int, default=BT_PORT, help=f'BitTorrent listen port (default: {BT_PORT})')
    seed_parser.add_argument('--no-dht', action='store_true', help='Disable DHT (require tracker)')
    seed_parser.add_argument('--listen-all', action='store_true', help='Listen on all interfaces')

    # Download via BitTorrent command
    bt_dl_parser = subparsers.add_parser('bt-download', help='Download models via BitTorrent')
    bt_dl_parser.add_argument('--torrent', type=Path, required=True, help='Path to .torrent file')
    bt_dl_parser.add_argument('--dest-dir', type=Path, required=True, help='Destination directory')
    bt_dl_parser.add_argument('--seeders', help='Comma-separated list of seeders (ip:port)')
    bt_dl_parser.add_argument('--port', type=int, default=BT_PORT, help=f'BitTorrent listen port (default: {BT_PORT})')
    bt_dl_parser.add_argument('--no-dht', action='store_true', help='Disable DHT')
    bt_dl_parser.add_argument('--no-web-seed', action='store_true', help='Disable web seed fallback')

    # Swarm sync command
    swarm_parser = subparsers.add_parser('swarm-sync', help='Sync models using BitTorrent swarm')
    swarm_parser.add_argument('--nodes', required=True, help='Comma-separated list of node IPs')
    swarm_parser.add_argument('--models-dir', type=Path, required=True, help='Local models directory')
    swarm_parser.add_argument('--http-port', type=int, default=8765, help='HTTP server port (default: 8765)')
    swarm_parser.add_argument('--bt-port', type=int, default=BT_PORT, help=f'BitTorrent port (default: {BT_PORT})')

    args = parser.parse_args()

    if args.command == 'serve':
        serve_models(args.models_dir, args.port)

    elif args.command == 'download':
        download_models(
            args.source_ip, args.port, args.dest_dir,
            args.connections, args.concurrent, args.filter
        )

    elif args.command == 'generate-urls':
        generate_url_file(args.source_ip, args.port, args.models_dir, args.output)

    elif args.command == 'status':
        models = list_models(args.models_dir)
        print(f"Found {len(models)} models in {args.models_dir}")
        ts_ip = get_tailscale_ip()
        if ts_ip:
            print(f"Tailscale IP: {ts_ip}")

    elif args.command == 'create-torrent':
        try:
            info = create_torrent_file(
                models_dir=args.models_dir,
                output_path=args.output,
                trackers=args.tracker,
                web_seeds=args.web_seed,
            )
            print(f"Created torrent: {args.output}")
            print(f"  Info hash: {info['info_hash']}")
            print(f"  Files: {info['file_count']}")
            print(f"  Total size: {info['total_size_mb']:.1f} MB")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == 'seed':
        if not args.torrent.exists():
            print(f"Torrent file not found: {args.torrent}")
            sys.exit(1)
        proc = seed_torrent(
            torrent_path=args.torrent,
            models_dir=args.models_dir,
            port=args.port,
            enable_dht=not args.no_dht,
            listen_all=args.listen_all,
        )
        print(f"Seeder started (PID: {proc.pid})")
        print("Press Ctrl+C to stop")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\nStopped")

    elif args.command == 'bt-download':
        if not args.torrent.exists():
            print(f"Torrent file not found: {args.torrent}")
            sys.exit(1)
        seeders = args.seeders.split(',') if args.seeders else None
        success = download_via_bittorrent(
            torrent_path=args.torrent,
            dest_dir=args.dest_dir,
            seeders=seeders,
            port=args.port,
            enable_dht=not args.no_dht,
            web_seed_fallback=not args.no_web_seed,
        )
        if success:
            print("Download complete!")
        else:
            print("Download failed")
            sys.exit(1)

    elif args.command == 'swarm-sync':
        nodes = [n.strip() for n in args.nodes.split(',')]
        result = sync_between_nodes(
            nodes=nodes,
            models_dir=str(args.models_dir),
            http_port=args.http_port,
            bt_port=args.bt_port,
        )
        print(json.dumps(result, indent=2))
        if not result.get('success'):
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
