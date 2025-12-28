"""Torrent Manager - Create, seed, and track torrents for cluster distribution.

Provides resilient P2P file sharing for nodes with flaky connections:
1. Automatic torrent creation for large files (databases, models)
2. DHT-based peer discovery (no central tracker needed)
3. Magnet link generation and sharing
4. Web seed support for HTTP fallback
5. Integration with cluster inventory system

BitTorrent is ideal for cluster sync because:
- Downloads resume from any peer after disconnection
- Multiple peers provide redundancy
- DHT enables decentralized peer discovery
- Seeding helps spread data across the cluster

Usage:
    manager = TorrentManager()

    # Create torrent for a database
    info = await manager.create_torrent(
        path=Path("data/games/canonical_hex8_2p.db"),
        web_seeds=["http://node1:8766/games/", "http://node2:8766/games/"],
    )

    # Get magnet link for sharing
    print(f"Magnet: {info.magnet_link}")

    # Start seeding
    await manager.start_seeding(info.torrent_path)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Piece sizes optimized for different file types
PIECE_SIZE_SMALL = 256 * 1024       # 256KB for files <100MB
PIECE_SIZE_MEDIUM = 512 * 1024      # 512KB for files 100MB-500MB
PIECE_SIZE_LARGE = 1024 * 1024      # 1MB for files >500MB
PIECE_SIZE_XLARGE = 2 * 1024 * 1024  # 2MB for files >1GB

# BitTorrent ports
BT_LISTEN_PORT = 51413
DHT_LISTEN_PORT = 6881

# Public DHT bootstrap nodes (for initial peer discovery)
DHT_BOOTSTRAP_NODES = [
    "router.bittorrent.com:6881",
    "router.utorrent.com:6881",
    "dht.transmissionbt.com:6881",
]


# Try to import bencodepy, fall back to pure Python implementation
try:
    import bencodepy
    HAS_BENCODEPY = True
except ImportError:
    bencodepy = None  # type: ignore
    HAS_BENCODEPY = False


def bencode(data: Any) -> bytes:
    """Bencode data for torrent files. Pure Python fallback if bencodepy unavailable."""
    if HAS_BENCODEPY:
        return bencodepy.encode(data)

    # Pure Python bencode implementation
    if isinstance(data, int):
        return f"i{data}e".encode("utf-8")
    elif isinstance(data, bytes):
        return f"{len(data)}:".encode("utf-8") + data
    elif isinstance(data, str):
        data_bytes = data.encode("utf-8")
        return f"{len(data_bytes)}:".encode("utf-8") + data_bytes
    elif isinstance(data, list):
        result = b"l"
        for item in data:
            result += bencode(item)
        result += b"e"
        return result
    elif isinstance(data, dict):
        result = b"d"
        # Keys must be sorted
        for key in sorted(data.keys()):
            key_bytes = key if isinstance(key, bytes) else key.encode("utf-8")
            result += bencode(key_bytes)
            result += bencode(data[key])
        result += b"e"
        return result
    else:
        raise TypeError(f"Cannot bencode {type(data)}")


@dataclass
class TorrentInfo:
    """Information about a created torrent."""
    info_hash: str
    torrent_path: Path
    magnet_link: str
    file_count: int
    total_size: int
    piece_size: int
    name: str
    web_seeds: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class TorrentManagerConfig:
    """Configuration for torrent manager."""
    cache_dir: Path = field(default_factory=lambda: Path("data/torrents"))
    bt_port: int = BT_LISTEN_PORT
    dht_port: int = DHT_LISTEN_PORT
    enable_dht: bool = True
    enable_lpd: bool = True   # Local Peer Discovery
    enable_pex: bool = True   # Peer Exchange
    seed_ratio: float = 2.0   # Seed until 2:1 upload ratio
    seed_time: int = 3600     # Minimum seed time (seconds)
    min_file_size_for_torrent: int = 10 * 1024 * 1024  # 10MB


class TorrentManager:
    """Manages torrent creation, seeding, and distribution for cluster sync."""

    def __init__(self, config: TorrentManagerConfig | None = None):
        self.config = config or TorrentManagerConfig()
        self._torrents: dict[str, TorrentInfo] = {}  # info_hash -> TorrentInfo
        self._seeding_processes: dict[str, asyncio.subprocess.Process] = {}
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure torrent cache directory exists."""
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _select_piece_size(self, file_size: int) -> int:
        """Select optimal piece size based on file size."""
        if file_size < 100 * 1024 * 1024:  # < 100MB
            return PIECE_SIZE_SMALL
        elif file_size < 500 * 1024 * 1024:  # < 500MB
            return PIECE_SIZE_MEDIUM
        elif file_size < 1024 * 1024 * 1024:  # < 1GB
            return PIECE_SIZE_LARGE
        else:
            return PIECE_SIZE_XLARGE

    async def create_torrent(
        self,
        path: Path,
        web_seeds: list[str] | None = None,
        trackers: list[str] | None = None,
        output_path: Path | None = None,
        comment: str | None = None,
    ) -> TorrentInfo:
        """Create a .torrent file for a file or directory.

        Args:
            path: Path to file or directory to create torrent for
            web_seeds: Optional HTTP URLs for hybrid HTTP+BT downloads
            trackers: Optional tracker URLs (omit for DHT-only trackerless operation)
            output_path: Where to save .torrent file (default: cache_dir/name.torrent)
            comment: Optional comment in torrent metadata

        Returns:
            TorrentInfo with info_hash, magnet link, etc.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Determine output path
        if output_path is None:
            output_path = self.config.cache_dir / f"{path.name}.torrent"
        output_path = Path(output_path)

        # Calculate file info and pieces
        if path.is_file():
            files_info, all_pieces, total_size, piece_size = self._process_single_file(path)
            torrent_name = path.name
        else:
            files_info, all_pieces, total_size, piece_size = self._process_directory(path)
            torrent_name = path.name

        # Build torrent info dictionary
        info: dict[bytes, Any] = {
            b"name": torrent_name.encode("utf-8"),
            b"piece length": piece_size,
            b"pieces": all_pieces,
        }

        if path.is_file():
            # Single file mode
            info[b"length"] = total_size
        else:
            # Multi-file mode
            info[b"files"] = files_info

        # Calculate info hash
        info_encoded = bencode(info)
        info_hash = hashlib.sha1(info_encoded).hexdigest()

        # Build full torrent dictionary
        torrent: dict[bytes, Any] = {
            b"info": info,
            b"created by": b"RingRift Cluster Sync",
            b"creation date": int(time.time()),
        }

        if comment:
            torrent[b"comment"] = comment.encode("utf-8")

        # Add trackers (if any)
        if trackers:
            if len(trackers) == 1:
                torrent[b"announce"] = trackers[0].encode("utf-8")
            else:
                torrent[b"announce"] = trackers[0].encode("utf-8")
                torrent[b"announce-list"] = [
                    [t.encode("utf-8")] for t in trackers
                ]

        # Add web seeds for hybrid HTTP+BT downloads (critical for flaky connections)
        if web_seeds:
            # Ensure web seeds point to the file, not just the directory
            resolved_seeds = []
            for ws in web_seeds:
                ws = ws.rstrip("/")
                if path.is_file() and not ws.endswith(path.name):
                    ws = f"{ws}/{path.name}"
                resolved_seeds.append(ws)
            torrent[b"url-list"] = [ws.encode("utf-8") for ws in resolved_seeds]

        # Write torrent file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(bencode(torrent))

        # Generate magnet link
        magnet_link = self._generate_magnet_link(
            info_hash=info_hash,
            name=torrent_name,
            trackers=trackers,
            web_seeds=web_seeds,
        )

        torrent_info = TorrentInfo(
            info_hash=info_hash,
            torrent_path=output_path,
            magnet_link=magnet_link,
            file_count=1 if path.is_file() else len(files_info),
            total_size=total_size,
            piece_size=piece_size,
            name=torrent_name,
            web_seeds=web_seeds or [],
        )

        self._torrents[info_hash] = torrent_info
        logger.info(f"Created torrent: {info_hash[:16]}... for {path.name} ({total_size / 1024 / 1024:.1f} MB)")

        return torrent_info

    def _process_single_file(self, path: Path) -> tuple[list, bytes, int, int]:
        """Process a single file for torrent creation."""
        file_size = path.stat().st_size
        piece_size = self._select_piece_size(file_size)

        all_pieces = b""
        with open(path, "rb") as f:
            while True:
                piece_data = f.read(piece_size)
                if not piece_data:
                    break
                all_pieces += hashlib.sha1(piece_data).digest()

        return [], all_pieces, file_size, piece_size

    def _process_directory(self, path: Path) -> tuple[list, bytes, int, int]:
        """Process a directory for multi-file torrent creation."""
        files = sorted(path.rglob("*"))
        files = [f for f in files if f.is_file()]

        if not files:
            raise ValueError(f"No files found in {path}")

        total_size = sum(f.stat().st_size for f in files)
        piece_size = self._select_piece_size(total_size)

        files_info = []
        all_pieces = b""
        current_piece_data = b""

        for file_path in files:
            file_size = file_path.stat().st_size
            rel_path = file_path.relative_to(path)

            files_info.append({
                b"length": file_size,
                b"path": [p.encode("utf-8") for p in rel_path.parts],
            })

            # Read file and calculate pieces
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(piece_size - len(current_piece_data))
                    if not chunk:
                        break
                    current_piece_data += chunk
                    if len(current_piece_data) >= piece_size:
                        all_pieces += hashlib.sha1(current_piece_data).digest()
                        current_piece_data = b""

        # Handle final piece
        if current_piece_data:
            all_pieces += hashlib.sha1(current_piece_data).digest()

        return files_info, all_pieces, total_size, piece_size

    def _generate_magnet_link(
        self,
        info_hash: str,
        name: str,
        trackers: list[str] | None = None,
        web_seeds: list[str] | None = None,
    ) -> str:
        """Generate a magnet link for a torrent.

        Magnet links allow sharing torrents without the .torrent file.
        DHT is used to discover peers.
        """
        params = [
            f"xt=urn:btih:{info_hash}",
            f"dn={urllib.parse.quote(name)}",
        ]

        # Add trackers
        if trackers:
            for tracker in trackers:
                params.append(f"tr={urllib.parse.quote(tracker)}")

        # Add web seeds
        if web_seeds:
            for ws in web_seeds:
                params.append(f"ws={urllib.parse.quote(ws)}")

        return f"magnet:?{'&'.join(params)}"

    def get_magnet_link(self, info_hash: str) -> str | None:
        """Get magnet link for a known torrent."""
        info = self._torrents.get(info_hash)
        return info.magnet_link if info else None

    def get_torrent_info(self, info_hash: str) -> TorrentInfo | None:
        """Get info for a known torrent."""
        return self._torrents.get(info_hash)

    async def start_seeding(
        self,
        torrent_path: Path,
        data_dir: Path | None = None,
    ) -> bool:
        """Start seeding a torrent using aria2c.

        Seeding is critical for cluster resilience - it allows other nodes
        to download the file even when the original source is unavailable.

        Args:
            torrent_path: Path to .torrent file
            data_dir: Directory containing the actual files (default: auto-detect)

        Returns:
            True if seeding started successfully
        """
        torrent_path = Path(torrent_path)
        if not torrent_path.exists():
            logger.error(f"Torrent file not found: {torrent_path}")
            return False

        # Build aria2c command for seeding
        cmd = [
            "aria2c",
            str(torrent_path),
            "--seed-ratio=0",  # Seed indefinitely
            "--seed-time=0",   # Seed indefinitely
            f"--listen-port={self.config.bt_port}",
            "--file-allocation=none",
            "--console-log-level=warn",
            "--quiet=true",
        ]

        if self.config.enable_dht:
            cmd.extend([
                "--enable-dht=true",
                f"--dht-listen-port={self.config.dht_port}",
                f"--dht-file-path={self.config.cache_dir}/dht.dat",
            ])

        if self.config.enable_lpd:
            cmd.append("--bt-enable-lpd=true")

        if self.config.enable_pex:
            cmd.append("--enable-peer-exchange=true")

        if data_dir:
            cmd.append(f"--dir={data_dir}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Read torrent to get info_hash for tracking
            info_hash = self._get_info_hash_from_file(torrent_path)
            if info_hash:
                self._seeding_processes[info_hash] = process
                logger.info(f"Started seeding: {info_hash[:16]}...")
            else:
                logger.warning(f"Started seeding but couldn't parse info_hash from {torrent_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to start seeding: {e}")
            return False

    def _get_info_hash_from_file(self, torrent_path: Path) -> str | None:
        """Extract info_hash from a torrent file."""
        try:
            with open(torrent_path, "rb") as f:
                data = f.read()

            # Simple bdecode for info dict extraction
            if HAS_BENCODEPY:
                torrent = bencodepy.decode(data)
                info = torrent.get(b"info", {})
            else:
                # Find info dict position (simplified parsing)
                # This is a basic implementation - full bdecode is complex
                return None

            info_encoded = bencode(info)
            return hashlib.sha1(info_encoded).hexdigest()

        except Exception as e:
            logger.debug(f"Failed to parse torrent file: {e}")
            return None

    async def stop_seeding(self, info_hash: str) -> None:
        """Stop seeding a specific torrent."""
        process = self._seeding_processes.pop(info_hash, None)
        if process:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
            logger.info(f"Stopped seeding: {info_hash[:16]}...")

    async def stop_all_seeding(self) -> None:
        """Stop all seeding processes."""
        for info_hash in list(self._seeding_processes.keys()):
            await self.stop_seeding(info_hash)

    def get_active_torrents(self) -> list[TorrentInfo]:
        """Get list of actively seeding torrents."""
        return [
            info for info_hash, info in self._torrents.items()
            if info_hash in self._seeding_processes
        ]

    async def refresh_torrent_cache(self) -> None:
        """Scan cache directory and reload torrent metadata."""
        for torrent_file in self.config.cache_dir.glob("*.torrent"):
            info_hash = self._get_info_hash_from_file(torrent_file)
            if info_hash and info_hash not in self._torrents:
                # Basic info for cached torrents
                self._torrents[info_hash] = TorrentInfo(
                    info_hash=info_hash,
                    torrent_path=torrent_file,
                    magnet_link=f"magnet:?xt=urn:btih:{info_hash}",
                    file_count=0,  # Unknown from file alone
                    total_size=0,
                    piece_size=0,
                    name=torrent_file.stem,
                )


def get_default_web_seeds(
    file_path: Path,
    data_server_port: int = 8766,
) -> list[str]:
    """Generate web seed URLs for a file from known cluster nodes.

    Web seeds provide HTTP fallback when no BitTorrent peers are available.
    This is critical for nodes with intermittent connectivity.
    """
    # Determine category from path
    path_str = str(file_path)
    if "games" in path_str or file_path.suffix == ".db":
        category = "games"
    elif "models" in path_str or file_path.suffix in (".pth", ".onnx"):
        category = "models"
    elif "training" in path_str or file_path.suffix == ".npz":
        category = "training"
    else:
        category = "data"

    # Try to get cluster nodes from config
    try:
        from app.config.cluster_config import get_cluster_nodes
        nodes = get_cluster_nodes()
    except ImportError:
        # Fall back to reading config directly
        try:
            import yaml
            config_path = Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                nodes = config.get("hosts", {})
            else:
                nodes = {}
        except (yaml.YAMLError, OSError, TypeError) as e:
            # Dec 2025: YAML parse or file read errors
            logger.debug(f"[TorrentManager] Config load failed, using empty nodes: {e}")
            nodes = {}

    web_seeds = []
    for name, node in nodes.items():
        if isinstance(node, dict):
            status = node.get("status", "")
            if status == "terminated":
                continue
            ip = node.get("tailscale_ip") or node.get("ssh_host")
            if ip and ip != "localhost":
                web_seeds.append(f"http://{ip}:{data_server_port}/{category}/{file_path.name}")

    return web_seeds
