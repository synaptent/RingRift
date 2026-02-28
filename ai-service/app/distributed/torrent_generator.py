"""Torrent Generator - Create .torrent files for P2P file distribution.

This module provides torrent generation for resilient P2P file sync across
the cluster. It integrates with aria2's BitTorrent support to enable:
- Multi-source downloads from multiple seeders
- Automatic resume after connection drops
- DHT-based peer discovery (trackerless operation)
- Web seeds for hybrid HTTP+BitTorrent downloads

Usage:
    generator = TorrentGenerator()
    torrent_path, info_hash = generator.create_torrent(
        Path("data/training/hex8_2p.npz"),
        web_seeds=["http://node1:8766/hex8_2p.npz"],
    )
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import bencodepy for faster encoding
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


# Default piece size: 256KB (good for large files)
DEFAULT_PIECE_SIZE = 256 * 1024

# Piece size tiers based on file size (for optimal performance)
PIECE_SIZE_TIERS = [
    (50 * 1024 * 1024, 256 * 1024),      # < 50MB: 256KB pieces
    (200 * 1024 * 1024, 512 * 1024),     # < 200MB: 512KB pieces
    (500 * 1024 * 1024, 1024 * 1024),    # < 500MB: 1MB pieces
    (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024),  # < 2GB: 2MB pieces
    (float('inf'), 4 * 1024 * 1024),     # >= 2GB: 4MB pieces
]


def get_optimal_piece_size(file_size: int) -> int:
    """Calculate optimal piece size based on file size.

    Larger files use larger pieces to reduce overhead.
    """
    for size_threshold, piece_size in PIECE_SIZE_TIERS:
        if file_size < size_threshold:
            return piece_size
    return DEFAULT_PIECE_SIZE


@dataclass
class TorrentInfo:
    """Information about a created torrent."""
    info_hash: str
    file_path: str
    torrent_path: str
    file_size: int
    piece_size: int
    piece_count: int
    web_seeds: list[str]
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "info_hash": self.info_hash,
            "file_path": self.file_path,
            "torrent_path": self.torrent_path,
            "file_size": self.file_size,
            "piece_size": self.piece_size,
            "piece_count": self.piece_count,
            "web_seeds": self.web_seeds,
            "created_at": self.created_at,
        }


class TorrentGenerator:
    """Generate .torrent files for P2P file distribution.

    Features:
    - Single file torrent creation (optimized for training data)
    - Automatic piece size selection based on file size
    - Web seed support for hybrid HTTP+BitTorrent downloads
    - DHT-friendly (trackerless) operation
    - Caching of generated torrents

    Example:
        generator = TorrentGenerator()
        torrent_path, info_hash = generator.create_torrent(
            Path("data/training/hex8_2p.npz"),
            web_seeds=["http://node1:8766/hex8_2p.npz"],
        )
    """

    def __init__(
        self,
        torrents_dir: Path | str | None = None,
        piece_size: int | None = None,
        comment: str = "RingRift distributed training data",
    ):
        """Initialize the torrent generator.

        Args:
            torrents_dir: Directory to store .torrent files (default: data/torrents)
            piece_size: Fixed piece size (default: auto-select based on file size)
            comment: Comment to include in torrent files
        """
        if torrents_dir is None:
            # Default to data/torrents relative to ai-service root
            root = Path(__file__).parent.parent.parent
            self.torrents_dir = root / "data" / "torrents"
        else:
            self.torrents_dir = Path(torrents_dir)

        self.piece_size = piece_size  # None = auto-select
        self.comment = comment

        # Feb 2026: In-memory cache to prevent concurrent regeneration of the same
        # torrent by multiple threads (was causing 42K+ "corrupt torrent" warnings).
        import threading
        self._torrent_cache: dict[str, tuple[Path, str]] = {}  # path -> (torrent_path, info_hash)
        self._torrent_locks: dict[str, threading.Lock] = {}
        self._torrent_locks_lock = threading.Lock()

        # Ensure torrents directory exists
        self.torrents_dir.mkdir(parents=True, exist_ok=True)

    def get_torrent_path(self, file_path: Path) -> Path:
        """Get the expected .torrent path for a data file.

        Args:
            file_path: Path to the data file

        Returns:
            Path where the .torrent file would be stored
        """
        # Use filename with .torrent extension
        return self.torrents_dir / f"{file_path.name}.torrent"

    def torrent_exists(self, file_path: Path) -> bool:
        """Check if a torrent already exists for this file.

        Note: Does NOT verify the torrent matches the current file.
        Use verify_torrent() for that.
        """
        return self.get_torrent_path(file_path).exists()

    def compute_info_hash(self, file_path: Path) -> str:
        """Compute the info_hash for a file without creating a torrent.

        This is useful for checking if a torrent needs to be regenerated
        (if the file has changed, the info_hash will differ).

        Args:
            file_path: Path to the data file

        Returns:
            SHA1 hex digest of the bencoded info dictionary
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        piece_size = self.piece_size or get_optimal_piece_size(file_size)

        # Calculate piece hashes
        pieces = self._hash_file_pieces(file_path, piece_size)

        # Build info dictionary (same as in create_torrent)
        info = {
            b"name": file_path.name.encode("utf-8"),
            b"length": file_size,
            b"piece length": piece_size,
            b"pieces": pieces,
        }

        # Calculate info hash
        info_encoded = bencode(info)
        return hashlib.sha1(info_encoded).hexdigest()

    def _hash_file_pieces(self, file_path: Path, piece_size: int) -> bytes:
        """Hash file contents into pieces.

        Returns concatenated SHA1 digests of each piece.
        """
        pieces = b""
        with open(file_path, "rb") as f:
            while True:
                piece_data = f.read(piece_size)
                if not piece_data:
                    break
                pieces += hashlib.sha1(piece_data).digest()
        return pieces

    def create_torrent(
        self,
        file_path: Path | str,
        web_seeds: list[str] | None = None,
        trackers: list[str] | None = None,
        private: bool = False,
        force: bool = False,
    ) -> tuple[Path, str]:
        """Create a .torrent file for a data file.

        Args:
            file_path: Path to the file to create torrent for
            web_seeds: Optional HTTP URLs for hybrid HTTP+BT downloads
            trackers: Optional tracker URLs (empty for DHT-only)
            private: If True, disable DHT and require tracker
            force: If True, overwrite existing torrent

        Returns:
            Tuple of (torrent_path, info_hash)

        Raises:
            FileNotFoundError: If file_path doesn't exist
            FileExistsError: If torrent exists and force=False
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        torrent_path = self.get_torrent_path(file_path)
        cache_key = str(torrent_path)

        # Feb 2026: Check in-memory cache first to avoid concurrent disk reads
        if not force and cache_key in self._torrent_cache:
            cached_path, cached_hash = self._torrent_cache[cache_key]
            if cached_path.exists():
                return cached_path, cached_hash

        # Per-file lock to prevent concurrent regeneration
        with self._torrent_locks_lock:
            if cache_key not in self._torrent_locks:
                import threading
                self._torrent_locks[cache_key] = threading.Lock()
            file_lock = self._torrent_locks[cache_key]

        with file_lock:
            # Re-check cache after acquiring lock (another thread may have finished)
            if not force and cache_key in self._torrent_cache:
                cached_path, cached_hash = self._torrent_cache[cache_key]
                if cached_path.exists():
                    return cached_path, cached_hash

            if torrent_path.exists() and not force:
                # Torrent already exists - read info_hash from it
                try:
                    with open(torrent_path, "rb") as f:
                        torrent_data = bdecode(f.read())
                    info_encoded = bencode(torrent_data[b"info"])
                    info_hash = hashlib.sha1(info_encoded).hexdigest()
                    self._torrent_cache[cache_key] = (torrent_path, info_hash)
                    logger.debug(f"Torrent already exists: {torrent_path}")
                    return torrent_path, info_hash
                except Exception as e:
                    logger.warning(f"Corrupt torrent file, regenerating: {e}")

            file_size = file_path.stat().st_size
            piece_size = self.piece_size or get_optimal_piece_size(file_size)

            logger.info(
                f"Creating torrent for {file_path.name} "
                f"({file_size / 1024 / 1024:.1f}MB, {piece_size // 1024}KB pieces)"
            )

            # Calculate piece hashes
            pieces = self._hash_file_pieces(file_path, piece_size)
            piece_count = len(pieces) // 20  # SHA1 is 20 bytes

            # Build info dictionary (single file format)
            info = {
                b"name": file_path.name.encode("utf-8"),
                b"length": file_size,
                b"piece length": piece_size,
                b"pieces": pieces,
            }

            if private:
                info[b"private"] = 1

            # Calculate info hash
            info_encoded = bencode(info)
            info_hash = hashlib.sha1(info_encoded).hexdigest()

            # Build full torrent dictionary
            torrent = {
                b"info": info,
                b"created by": b"RingRift TorrentGenerator",
                b"creation date": int(time.time()),
                b"comment": self.comment.encode("utf-8"),
            }

            # Add trackers
            if trackers:
                if len(trackers) == 1:
                    torrent[b"announce"] = trackers[0].encode("utf-8")
                else:
                    torrent[b"announce"] = trackers[0].encode("utf-8")
                    torrent[b"announce-list"] = [
                        [t.encode("utf-8")] for t in trackers
                    ]

            # Add web seeds for hybrid HTTP+BT downloads
            if web_seeds:
                torrent[b"url-list"] = [ws.encode("utf-8") for ws in web_seeds]

            # Write torrent file atomically to prevent read-during-write corruption.
            # Feb 2026: Multiple concurrent threads call create_torrent() on the same
            # file, causing 42K+ "Corrupt torrent file" warnings from partial reads.
            torrent_path.parent.mkdir(parents=True, exist_ok=True)
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=torrent_path.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "wb") as f:
                    f.write(bencode(torrent))
                Path(tmp_path).replace(torrent_path)  # Atomic on POSIX
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.info(
                f"Created torrent: {torrent_path.name} "
                f"(info_hash: {info_hash[:16]}..., {piece_count} pieces)"
            )

            self._torrent_cache[cache_key] = (torrent_path, info_hash)
            return torrent_path, info_hash

    def create_torrent_info(
        self,
        file_path: Path | str,
        web_seeds: list[str] | None = None,
        trackers: list[str] | None = None,
        **kwargs,
    ) -> TorrentInfo:
        """Create a torrent and return detailed info.

        Same as create_torrent() but returns TorrentInfo object with
        all metadata for storage in ClusterManifest.
        """
        file_path = Path(file_path)
        torrent_path, info_hash = self.create_torrent(
            file_path, web_seeds, trackers, **kwargs
        )

        file_size = file_path.stat().st_size
        piece_size = self.piece_size or get_optimal_piece_size(file_size)
        piece_count = (file_size + piece_size - 1) // piece_size

        return TorrentInfo(
            info_hash=info_hash,
            file_path=str(file_path),
            torrent_path=str(torrent_path),
            file_size=file_size,
            piece_size=piece_size,
            piece_count=piece_count,
            web_seeds=web_seeds or [],
            created_at=time.time(),
        )

    def verify_torrent(self, file_path: Path, torrent_path: Path | None = None) -> bool:
        """Verify that a torrent matches the current file.

        Compares the stored info_hash with freshly computed hash.

        Args:
            file_path: Path to the data file
            torrent_path: Path to torrent (default: auto-detect)

        Returns:
            True if torrent matches file, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False

        torrent_path = torrent_path or self.get_torrent_path(file_path)
        if not torrent_path.exists():
            return False

        try:
            # Read stored info_hash
            with open(torrent_path, "rb") as f:
                torrent_data = bdecode(f.read())
            stored_info_encoded = bencode(torrent_data[b"info"])
            stored_hash = hashlib.sha1(stored_info_encoded).hexdigest()

            # Compute current info_hash
            current_hash = self.compute_info_hash(file_path)

            return stored_hash == current_hash
        except Exception as e:
            logger.warning(f"Error verifying torrent: {e}")
            return False

    def list_torrents(self) -> list[Path]:
        """List all torrent files in the torrents directory."""
        return sorted(self.torrents_dir.glob("*.torrent"))

    def cleanup_orphaned(self, data_dirs: list[Path]) -> int:
        """Remove torrents for files that no longer exist.

        Args:
            data_dirs: Directories to search for data files

        Returns:
            Number of orphaned torrents removed
        """
        removed = 0
        for torrent_path in self.list_torrents():
            # Extract original filename
            data_filename = torrent_path.stem  # Remove .torrent extension

            # Check if file exists in any data directory
            found = False
            for data_dir in data_dirs:
                if (data_dir / data_filename).exists():
                    found = True
                    break

            if not found:
                logger.info(f"Removing orphaned torrent: {torrent_path.name}")
                torrent_path.unlink()
                removed += 1

        return removed


# Singleton instance for convenience
_generator: TorrentGenerator | None = None


def get_torrent_generator() -> TorrentGenerator:
    """Get the global TorrentGenerator instance."""
    global _generator
    if _generator is None:
        _generator = TorrentGenerator()
    return _generator
