#!/usr/bin/env python3
"""
P2P Model Distribution Utilities

Fast model distribution using aria2c multi-connection downloads.
Supports HTTP server mode (source) and aria2 download mode (destination).

Usage:
    # On source machine (has models):
    python p2p_model_distribution.py serve --models-dir /path/to/models --port 8765

    # On destination machine (needs models):
    python p2p_model_distribution.py download --source-ip 100.77.77.122 --port 8765 \
        --dest-dir /dev/shm/ringrift/ai-service/models --connections 16

    # Generate URL list for manual aria2 use:
    python p2p_model_distribution.py generate-urls --source-ip 100.77.77.122 --port 8765 \
        --models-dir /path/to/models --output urls.txt
"""

import argparse
import http.server
import socketserver
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import List, Optional
import threading
import time


def get_tailscale_ip() -> Optional[str]:
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


def list_models(models_dir: Path) -> List[str]:
    """List all .pth model files in directory."""
    return sorted([f.name for f in models_dir.glob("*.pth")])


def generate_urls(source_ip: str, port: int, models: List[str]) -> List[str]:
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

    with socketserver.TCPServer(("", port), QuietHandler) as httpd:
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
    models_filter: Optional[List[str]] = None
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


def sync_between_nodes(nodes: List[str], models_dir: str):
    """Sync models between multiple nodes using BitTorrent-like sharing."""
    # Each node that has models becomes a seeder
    # Nodes download from any available seeder
    # TODO: Implement using aria2c's BitTorrent support or a lightweight BT client
    raise NotImplementedError("BitTorrent sync not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="P2P Model Distribution")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start HTTP server to serve models')
    serve_parser.add_argument('--models-dir', type=Path, required=True, help='Directory containing .pth files')
    serve_parser.add_argument('--port', type=int, default=8765, help='Port to serve on (default: 8765)')

    # Download command
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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
