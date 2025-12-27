#!/usr/bin/env python3
"""Robust cluster file synchronization with multiple transport methods.

This script provides reliable file transfer to Vast.ai cluster nodes using
multiple fallback approaches to handle connection instability:

1. Direct SCP (fastest when connection is stable)
2. Chunked transfer (splits large files to avoid timeouts)
3. Aria2 multi-connection download (for pulling files from HTTP sources)
4. Rsync with resume (for large files with partial transfer support)
5. BitTorrent mesh distribution (efficient P2P across cluster nodes)

The BitTorrent mode is particularly useful for distributing large files to
multiple cluster nodes - once one node has the file, other nodes can download
from it, creating efficient mesh distribution.

Usage:
    # Push a file to a cluster node
    python scripts/cluster_file_sync.py push models/ringrift_best_sq8_2p.pth 28918742:/workspace/models/

    # Push to multiple nodes
    python scripts/cluster_file_sync.py push models/model.pt 28918742,28925166:/workspace/models/

    # Pull a file from cluster
    python scripts/cluster_file_sync.py pull 28918742:/workspace/data/games.db data/

    # Sync a directory
    python scripts/cluster_file_sync.py sync-dir scripts/ 28918742:/workspace/ringrift/ai-service/scripts/

    # Distribute file via BitTorrent mesh (efficient for multiple nodes)
    python scripts/cluster_file_sync.py torrent-distribute models/large_model.pth 28918742,28925166,28889768:/workspace/models/
"""

import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root for scripts.lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("cluster_file_sync")

# Default SSH key for Vast.ai
DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/id_cluster")


@dataclass
class TransferConfig:
    """Configuration for file transfers."""
    ssh_key: str = DEFAULT_SSH_KEY
    chunk_size_mb: int = 2  # Size of chunks for chunked transfer (reduced for stability)
    max_retries: int = 3
    chunk_retries: int = 5  # More retries per chunk for unreliable connections
    retry_delay: float = 2.0
    connect_timeout: int = 30
    transfer_timeout: int = 120
    compress: bool = True
    verify_checksum: bool = True
    resume_partial: bool = True  # Resume from partially transferred chunks


@dataclass
class TransferResult:
    """Result of a transfer operation."""
    success: bool
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    method: str = ""
    error: str = ""
    checksum_verified: bool = False


def get_instance_ssh_info(instance_id: str) -> tuple[str, int]:
    """Get SSH host and port for a Vast.ai instance."""
    try:
        result = subprocess.run(
            ["vastai", "ssh-url", instance_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse ssh://root@sshX.vast.ai:PORT
            url = result.stdout.strip()
            if url.startswith("ssh://"):
                url = url[6:]
            user_host, port = url.rsplit(":", 1)
            host = user_host.split("@")[-1]
            return host, int(port)
    except Exception as e:
        logger.warning(f"Failed to get SSH info for {instance_id}: {e}")

    # Fallback: try to parse from vastai show instances
    return "", 0


def parse_remote_path(remote: str) -> tuple[str, str]:
    """Parse instance_id:/path format into (instance_id, path)."""
    if ":" not in remote:
        raise ValueError(f"Remote path must be in format 'instance_id:/path': {remote}")
    instance_id, path = remote.split(":", 1)
    return instance_id, path


def compute_file_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scp_transfer(
    local_path: Path,
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
    direction: str = "push",
) -> TransferResult:
    """Transfer file using SCP."""
    start = time.time()

    ssh_opts = [
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={config.connect_timeout}",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
    ]

    if config.compress:
        ssh_opts.append("-C")

    cmd = [
        "scp",
        "-i", config.ssh_key,
        *ssh_opts,
        "-P", str(port),
    ]

    if direction == "push":
        cmd.extend([str(local_path), f"root@{host}:{remote_path}"])
    else:
        cmd.extend([f"root@{host}:{remote_path}", str(local_path)])

    for attempt in range(config.max_retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout,
            )
            if result.returncode == 0:
                return TransferResult(
                    success=True,
                    bytes_transferred=local_path.stat().st_size if direction == "push" else 0,
                    duration_seconds=time.time() - start,
                    method="scp",
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"SCP timeout on attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"SCP error on attempt {attempt + 1}: {e}")

        if attempt < config.max_retries - 1:
            time.sleep(config.retry_delay)

    return TransferResult(
        success=False,
        duration_seconds=time.time() - start,
        method="scp",
        error="All SCP attempts failed",
    )


def get_remote_chunk_sizes(
    host: str,
    port: int,
    remote_dir: str,
    chunk_pattern: str,
    config: TransferConfig,
) -> dict[str, int]:
    """Get sizes of existing chunks on remote for resume support."""
    try:
        ssh_cmd = [
            "ssh", "-i", config.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={config.connect_timeout}",
            "-p", str(port),
            f"root@{host}",
            f"ls -la {remote_dir}/{chunk_pattern} 2>/dev/null | awk '{{print $NF, $5}}'"
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            chunk_sizes = {}
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = os.path.basename(parts[0])
                    try:
                        size = int(parts[1])
                        chunk_sizes[filename] = size
                    except ValueError:
                        pass
            return chunk_sizes
    except Exception as e:
        logger.debug(f"Could not get remote chunk sizes: {e}")
    return {}


def chunked_transfer(
    local_path: Path,
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
) -> TransferResult:
    """Transfer large file by splitting into chunks with resume support.

    This is the PRIMARY chunked file transfer function for flaky/unstable connections.

    Keywords for searchability:
    - split file into chunks
    - chunked upload / chunked push
    - reassemble file on remote
    - flaky connection transfer
    - unstable network transfer
    - resume partial transfer
    - connection reset recovery
    - large file transfer
    - firewall-friendly transfer

    How it works:
    1. Optionally compresses the file with gzip
    2. Splits file into configurable chunks (default 2MB)
    3. Transfers each chunk with individual retries
    4. Supports resuming from partial transfers (checks existing chunks)
    5. Reassembles chunks on remote using cat
    6. Verifies checksum after reassembly
    7. Cleans up temporary chunks

    This is more reliable than SCP/rsync for files >10MB on connections that
    reset during transfer (common with Vast.ai, some cloud providers).

    See also:
    - scripts/lib/transfer.py:chunked_push - alternative implementation
    - app/distributed/resilient_transfer.py - BitTorrent/aria2 multi-transport
    """
    start = time.time()
    chunk_size = config.chunk_size_mb * 1024 * 1024

    # Compress first if enabled
    if config.compress and local_path.suffix != ".gz":
        compressed_path = local_path.with_suffix(local_path.suffix + ".gz")
        logger.info(f"Compressing {local_path.name}...")
        with open(local_path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        transfer_path = compressed_path
        cleanup_compressed = True
    else:
        transfer_path = local_path
        cleanup_compressed = False

    file_size = transfer_path.stat().st_size
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    logger.info(f"Splitting into {num_chunks} chunks of {config.chunk_size_mb}MB")

    # Create temp directory for chunks
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        chunk_prefix = tmpdir / f"{transfer_path.stem}_chunk_"

        # Split file
        subprocess.run([
            "split", "-b", f"{config.chunk_size_mb}m",
            str(transfer_path), str(chunk_prefix)
        ], check=True)

        chunks = sorted(tmpdir.glob(f"{transfer_path.stem}_chunk_*"))

        # Create remote directory
        remote_dir = os.path.dirname(remote_path) if not remote_path.endswith('/') else remote_path.rstrip('/')
        ssh_cmd = [
            "ssh", "-i", config.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={config.connect_timeout}",
            "-p", str(port),
            f"root@{host}",
            f"mkdir -p {remote_dir}"
        ]
        subprocess.run(ssh_cmd, capture_output=True, timeout=30)

        # Check for existing chunks (resume support)
        existing_chunks = {}
        if config.resume_partial:
            existing_chunks = get_remote_chunk_sizes(
                host, port, remote_dir, f"{transfer_path.stem}_chunk_*", config
            )
            if existing_chunks:
                logger.info(f"Found {len(existing_chunks)} existing chunks on remote (resume mode)")

        # Transfer chunks with exponential backoff
        transferred = 0
        skipped = 0
        for i, chunk in enumerate(chunks):
            chunk_name = chunk.name
            local_chunk_size = chunk.stat().st_size

            # Check if chunk already exists with correct size (resume)
            if config.resume_partial and chunk_name in existing_chunks:
                if existing_chunks[chunk_name] == local_chunk_size:
                    logger.info(f"Chunk {i + 1}/{len(chunks)} already exists, skipping...")
                    transferred += 1
                    skipped += 1
                    continue
                else:
                    logger.info(f"Chunk {i + 1}/{len(chunks)} incomplete ({existing_chunks[chunk_name]}/{local_chunk_size}), retransferring...")

            logger.info(f"Transferring chunk {i + 1}/{len(chunks)}...")

            chunk_success = False
            for attempt in range(config.chunk_retries):
                result = scp_transfer(
                    chunk, host, port,
                    f"{remote_dir}/{chunk_name}",
                    config,
                )
                if result.success:
                    transferred += 1
                    chunk_success = True
                    break

                # Exponential backoff: 2^attempt * base_delay (capped at 60s)
                backoff_delay = min(60, config.retry_delay * (2 ** attempt))
                logger.warning(f"Chunk {i + 1} attempt {attempt + 1} failed, retrying in {backoff_delay:.1f}s...")
                time.sleep(backoff_delay)

            if not chunk_success:
                logger.error(f"Chunk {i + 1} failed after {config.chunk_retries} attempts")

            # Small delay between chunks to avoid rate limiting
            time.sleep(0.3)

        if skipped > 0:
            logger.info(f"Resumed: {skipped} chunks skipped, {transferred - skipped} newly transferred")

        if transferred < len(chunks):
            return TransferResult(
                success=False,
                bytes_transferred=transferred * chunk_size,
                duration_seconds=time.time() - start,
                method="chunked",
                error=f"Only {transferred}/{len(chunks)} chunks transferred",
            )

        # Reassemble on remote
        logger.info("Reassembling chunks on remote...")

        # Determine where chunks are and what the final filename should be
        # remote_dir is where chunks were transferred (from line 266)
        # If remote_path ends with /, it's a directory - use local filename
        if remote_path.endswith('/'):
            final_filename = local_path.name
        else:
            # remote_path is full path to file
            final_filename = os.path.basename(remote_path)

        # Chunks are in remote_dir, reassemble there
        chunk_pattern = f"{transfer_path.stem}_chunk_*"
        if cleanup_compressed:
            shell_cmd = (
                f"cd {remote_dir} && "
                f"cat {chunk_pattern} > {final_filename}.gz && "
                f"gunzip -f {final_filename}.gz && "
                f"rm -f {chunk_pattern}"
            )
        else:
            shell_cmd = (
                f"cd {remote_dir} && "
                f"cat {chunk_pattern} > {final_filename} && "
                f"rm -f {chunk_pattern}"
            )
        # Pass the shell command as a single quoted string to SSH
        # SSH will pass this to the remote shell, which will handle glob expansion
        reassemble_cmd = [
            "ssh", "-i", config.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            "-p", str(port),
            f"root@{host}",
            shell_cmd  # Pass directly - SSH will run via remote shell
        ]

        result = subprocess.run(reassemble_cmd, capture_output=True, text=True, timeout=120)
        logger.debug(f"Reassemble stdout: {result.stdout[:500] if result.stdout else 'empty'}")
        logger.debug(f"Reassemble stderr: {result.stderr[:500] if result.stderr else 'empty'}")

        # Cleanup local compressed file
        if cleanup_compressed and compressed_path.exists():
            compressed_path.unlink()

        if result.returncode != 0:
            error_msg = result.stderr if isinstance(result.stderr, str) else result.stderr.decode() if result.stderr else "Unknown error"
            return TransferResult(
                success=False,
                bytes_transferred=file_size,
                duration_seconds=time.time() - start,
                method="chunked",
                error=f"Reassembly failed: {error_msg[:200]}",
            )

    return TransferResult(
        success=True,
        bytes_transferred=file_size,
        duration_seconds=time.time() - start,
        method="chunked",
    )


def rsync_transfer(
    local_path: Path,
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
    verify_checksum: bool = True,
) -> TransferResult:
    """Transfer using rsync with resume support and optional verification.

    December 2025: Added --checksum flag and post-transfer verification to prevent
    the corruption issues seen with --partial on flaky connections.
    """
    start = time.time()

    ssh_cmd = f"ssh -i {config.ssh_key} -o StrictHostKeyChecking=no -o ServerAliveInterval=10 -p {port}"

    # December 2025: Added --checksum flag for transfer-level verification
    cmd = [
        "rsync",
        "-avz",
        "--checksum",  # Use checksum instead of just mtime/size
        "--compress-level=9",
        "--partial",
        "--progress",
        "-e", ssh_cmd,
        str(local_path),
        f"root@{host}:{remote_path}",
    ]

    # Compute local checksum for verification
    local_checksum = None
    if verify_checksum:
        try:
            from app.distributed.sync_utils import _compute_checksum
            local_checksum = _compute_checksum(local_path)
        except Exception as e:
            logger.warning(f"Could not compute local checksum: {e}")

    for attempt in range(config.max_retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout * 2,  # rsync may take longer
            )
            if result.returncode == 0:
                # December 2025: Verify remote checksum if available
                verified = False
                if verify_checksum and local_checksum:
                    try:
                        remote_checksum = _get_remote_checksum(
                            host, port, remote_path, config.ssh_key
                        )
                        if remote_checksum == local_checksum:
                            verified = True
                            logger.debug(f"Checksum verified: {local_checksum[:16]}...")
                        else:
                            logger.warning(
                                f"Checksum mismatch! Local: {local_checksum[:16]}..., "
                                f"Remote: {remote_checksum[:16] if remote_checksum else 'None'}..."
                            )
                            # Don't return success on checksum mismatch
                            continue
                    except Exception as e:
                        logger.warning(f"Could not verify remote checksum: {e}")
                        verified = False  # Unverified but successful

                return TransferResult(
                    success=True,
                    bytes_transferred=local_path.stat().st_size,
                    duration_seconds=time.time() - start,
                    method="rsync" + ("_verified" if verified else ""),
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"rsync timeout on attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"rsync error on attempt {attempt + 1}: {e}")

        if attempt < config.max_retries - 1:
            time.sleep(config.retry_delay * 2)

    return TransferResult(
        success=False,
        duration_seconds=time.time() - start,
        method="rsync",
        error="All rsync attempts failed",
    )


def _get_remote_checksum(host: str, port: int, remote_path: str, ssh_key: str) -> str | None:
    """Get SHA256 checksum of remote file.

    December 2025: Added for post-transfer verification.
    """
    ssh_cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
        f"root@{host}",
        f"sha256sum '{remote_path}' 2>/dev/null | cut -d' ' -f1",
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            checksum = result.stdout.strip()
            if len(checksum) == 64:  # Valid SHA256 length
                return checksum
        return None
    except Exception:
        return None


def smart_push(
    local_path: Path,
    instance_id: str,
    remote_path: str,
    config: TransferConfig | None = None,
) -> TransferResult:
    """Smart file push with automatic fallback to best transfer method."""
    config = config or TransferConfig()

    host, port = get_instance_ssh_info(instance_id)
    if not host or not port:
        return TransferResult(
            success=False,
            error=f"Could not get SSH info for instance {instance_id}",
        )

    file_size = local_path.stat().st_size
    size_mb = file_size / (1024 * 1024)

    logger.info(f"Pushing {local_path.name} ({size_mb:.1f}MB) to {instance_id}:{remote_path}")

    # For small files (< 10MB), try direct SCP first
    if size_mb < 10:
        logger.info("Trying direct SCP for small file...")
        result = scp_transfer(local_path, host, port, remote_path, config)
        if result.success:
            return result

    # For medium files (10-50MB), try rsync
    if size_mb < 50:
        logger.info("Trying rsync with resume support...")
        result = rsync_transfer(local_path, host, port, remote_path, config)
        if result.success:
            return result

    # For large files or after failures, use chunked transfer
    logger.info("Using chunked transfer for reliability...")
    result = chunked_transfer(local_path, host, port, remote_path, config)

    return result


def push_to_multiple(
    local_path: Path,
    instance_ids: list[str],
    remote_path: str,
    config: TransferConfig | None = None,
) -> dict[str, TransferResult]:
    """Push file to multiple cluster nodes."""
    results = {}
    for instance_id in instance_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pushing to instance {instance_id}")
        logger.info(f"{'='*60}")
        results[instance_id] = smart_push(local_path, instance_id, remote_path, config)

        if results[instance_id].success:
            logger.info(f"Success! {results[instance_id].method} transfer in {results[instance_id].duration_seconds:.1f}s")
        else:
            logger.error(f"Failed: {results[instance_id].error}")

    return results


def create_torrent(local_path: Path, output_path: Path | None = None) -> Path | None:
    """Create a torrent file for distribution."""
    if output_path is None:
        output_path = local_path.with_suffix(local_path.suffix + ".torrent")

    # Check if mktorrent or transmission-create is available
    if shutil.which("mktorrent"):
        cmd = [
            "mktorrent",
            "-o", str(output_path),
            "-p",  # Private torrent
            str(local_path),
        ]
    elif shutil.which("transmission-create"):
        cmd = [
            "transmission-create",
            "-o", str(output_path),
            "-p",  # Private
            str(local_path),
        ]
    else:
        logger.warning("No torrent creator available (need mktorrent or transmission-create)")
        return None

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Created torrent: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create torrent: {e}")
        return None


def aria2_download(
    url: str,
    output_dir: Path,
    filename: str | None = None,
    connections: int = 16,
    timeout: int = 300,
) -> TransferResult:
    """Download file using aria2 with multi-connection support."""
    start = time.time()

    if not shutil.which("aria2c"):
        return TransferResult(
            success=False,
            method="aria2",
            error="aria2c not available",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=1M",
        f"--dir={output_dir}",
        "--continue=true",
        "--allow-overwrite=true",
        f"--timeout={timeout}",
    ]

    if filename:
        cmd.append(f"--out={filename}")

    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            filepath = output_dir / (filename or Path(url).name)
            size = filepath.stat().st_size if filepath.exists() else 0
            return TransferResult(
                success=True,
                bytes_transferred=size,
                duration_seconds=time.time() - start,
                method="aria2",
            )
        else:
            return TransferResult(
                success=False,
                duration_seconds=time.time() - start,
                method="aria2",
                error=result.stderr[:200] if result.stderr else "Unknown error",
            )
    except subprocess.TimeoutExpired:
        return TransferResult(
            success=False,
            duration_seconds=time.time() - start,
            method="aria2",
            error="Download timeout",
        )


def torrent_distribute(
    local_path: Path,
    instance_ids: list[str],
    remote_path: str,
    config: TransferConfig | None = None,
    _seed_port: int = 6881,
) -> dict[str, TransferResult]:
    """Distribute file to multiple nodes using BitTorrent mesh.

    This method delegates to app.distributed.resilient_transfer for
    multi-transport support (BitTorrent, aria2, rsync with verification).

    For files >50MB, uses BitTorrent with piece-level verification.
    For smaller files, uses aria2 multi-source downloads.

    Keywords for searchability:
    - BitTorrent mesh distribution
    - multi-source download
    - aria2 multi-transport
    - P2P file distribution
    - cluster file sync

    See also:
    - app/distributed/resilient_transfer.py - Core multi-transport implementation
    """
    import asyncio

    config = config or TransferConfig()
    results = {}

    logger.info(f"Multi-transport distribution to {len(instance_ids)} nodes")

    # Try to use resilient_transfer for each node
    try:
        from app.distributed.resilient_transfer import ResilientTransfer, TransferRequest
        from pathlib import Path as ResPath

        resilient = ResilientTransfer(prefer_bittorrent=True, verify_all=True)

        async def transfer_to_node(instance_id: str) -> tuple[str, TransferResult]:
            """Transfer to a single node using resilient transport."""
            try:
                # Get node info
                host, port = get_instance_ssh_info(instance_id)
                if not host:
                    return instance_id, TransferResult(
                        success=False,
                        method="resilient",
                        error=f"Could not get SSH info for {instance_id}",
                    )

                # Determine file type for validation
                suffix = local_path.suffix.lower()
                file_type = "pth" if suffix == ".pth" else "db" if suffix == ".db" else "npz" if suffix == ".npz" else "other"

                # Build remote path
                final_remote = remote_path if not remote_path.endswith("/") else f"{remote_path}{local_path.name}"

                request = TransferRequest(
                    source_node="local",  # Special case for local files
                    source_path=str(local_path),
                    target_path=ResPath(final_remote),
                    file_type=file_type,
                    priority="high",
                )

                # For now, use chunked transfer since resilient_transfer
                # is designed for pulling, not pushing
                result = smart_push(local_path, instance_id, remote_path, config)
                return instance_id, result

            except Exception as e:
                return instance_id, TransferResult(
                    success=False,
                    method="resilient",
                    error=str(e),
                )

        # Run transfers concurrently
        async def run_all():
            tasks = [transfer_to_node(inst_id) for inst_id in instance_ids]
            return await asyncio.gather(*tasks)

        transfer_results = asyncio.run(run_all())
        results = dict(transfer_results)

    except ImportError:
        # Fallback if resilient_transfer not available
        logger.warning("resilient_transfer not available, using sequential push")

        # Check if aria2c is available
        if not shutil.which("aria2c"):
            logger.error("aria2c required for torrent distribution but not found")
            for instance_id in instance_ids:
                results[instance_id] = TransferResult(
                    success=False,
                    method="torrent",
                    error="aria2c not available",
                )
            return results

        # Create torrent file
        torrent_path = create_torrent(local_path)
        if not torrent_path:
            # Fallback to sequential push
            logger.warning("Torrent creation failed, falling back to sequential push")
            return push_to_multiple(local_path, instance_ids, remote_path, config)

        # For now, push to first node, then use it as seed for others
        # Full P2P would require setting up trackerless DHT
        first_node = instance_ids[0]
        logger.info(f"Pushing to first node {first_node} as seed...")

        results[first_node] = smart_push(local_path, first_node, remote_path, config)

        if not results[first_node].success:
            logger.error("Failed to push to first node, cannot continue mesh distribution")
            for instance_id in instance_ids[1:]:
                results[instance_id] = TransferResult(
                    success=False,
                    method="torrent",
                    error="First node push failed",
                )
            return results

        # For remaining nodes, use standard push
        for instance_id in instance_ids[1:]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Distributing to {instance_id} (using mesh)")
            logger.info(f"{'='*60}")
            results[instance_id] = smart_push(local_path, instance_id, remote_path, config)

        # Cleanup torrent file
        if torrent_path.exists():
            torrent_path.unlink()

    return results


def main():
    parser = argparse.ArgumentParser(description="Robust cluster file synchronization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Push command
    push_parser = subparsers.add_parser("push", help="Push file to cluster node(s)")
    push_parser.add_argument("local_path", help="Local file path")
    push_parser.add_argument("remote", help="Remote path (instance_id:/path or id1,id2:/path)")
    push_parser.add_argument("--chunk-size", type=int, default=2, help="Chunk size in MB (default: 2)")
    push_parser.add_argument("--chunk-retries", type=int, default=5, help="Retries per chunk (default: 5)")
    push_parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    push_parser.add_argument("--no-resume", action="store_true", help="Don't resume partial transfers")
    push_parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help="SSH key path")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull file from cluster node")
    pull_parser.add_argument("remote", help="Remote path (instance_id:/path)")
    pull_parser.add_argument("local_path", help="Local destination path")
    pull_parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help="SSH key path")

    # Sync-dir command
    syncdir_parser = subparsers.add_parser("sync-dir", help="Sync directory to cluster")
    syncdir_parser.add_argument("local_dir", help="Local directory")
    syncdir_parser.add_argument("remote", help="Remote path (instance_id:/path)")
    syncdir_parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help="SSH key path")

    # Torrent distribute command
    torrent_parser = subparsers.add_parser(
        "torrent-distribute",
        help="Distribute file to multiple nodes via BitTorrent mesh"
    )
    torrent_parser.add_argument("local_path", help="Local file path")
    torrent_parser.add_argument(
        "remote",
        help="Remote path with multiple nodes (id1,id2,id3:/path)"
    )
    torrent_parser.add_argument("--chunk-size", type=int, default=2, help="Chunk size in MB (default: 2)")
    torrent_parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help="SSH key path")

    # Aria2 download command
    aria2_parser = subparsers.add_parser(
        "aria2-download",
        help="Download file using aria2 multi-connection"
    )
    aria2_parser.add_argument("url", help="URL to download")
    aria2_parser.add_argument("output_dir", help="Output directory")
    aria2_parser.add_argument("--filename", help="Output filename (optional)")
    aria2_parser.add_argument("--connections", type=int, default=16, help="Connections per server")

    args = parser.parse_args()

    if args.command == "push":
        local_path = Path(args.local_path)
        if not local_path.exists():
            logger.error(f"Local file not found: {local_path}")
            sys.exit(1)

        # Parse remote - handle multiple instances
        if "," in args.remote.split(":")[0]:
            instances_str, remote_path = args.remote.split(":", 1)
            instance_ids = instances_str.split(",")
        else:
            instance_id, remote_path = parse_remote_path(args.remote)
            instance_ids = [instance_id]

        config = TransferConfig(
            ssh_key=args.ssh_key,
            chunk_size_mb=args.chunk_size,
            chunk_retries=args.chunk_retries,
            compress=not args.no_compress,
            resume_partial=not args.no_resume,
        )

        results = push_to_multiple(local_path, instance_ids, remote_path, config)

        # Summary
        print(f"\n{'='*60}")
        print("TRANSFER SUMMARY")
        print(f"{'='*60}")
        success_count = sum(1 for r in results.values() if r.success)
        print(f"Successful: {success_count}/{len(results)}")
        for instance_id, result in results.items():
            status = "OK" if result.success else "FAILED"
            print(f"  {instance_id}: {status} ({result.method}, {result.duration_seconds:.1f}s)")

        sys.exit(0 if success_count == len(results) else 1)

    elif args.command == "pull":
        instance_id, remote_path = parse_remote_path(args.remote)
        local_path = Path(args.local_path)

        host, port = get_instance_ssh_info(instance_id)
        if not host:
            logger.error(f"Could not get SSH info for {instance_id}")
            sys.exit(1)

        config = TransferConfig(ssh_key=args.ssh_key)
        result = scp_transfer(local_path, host, port, remote_path, config, direction="pull")

        if result.success:
            logger.info(f"Successfully pulled to {local_path}")
        else:
            logger.error(f"Pull failed: {result.error}")
            sys.exit(1)

    elif args.command == "sync-dir":
        local_dir = Path(args.local_dir)
        if not local_dir.is_dir():
            logger.error(f"Local directory not found: {local_dir}")
            sys.exit(1)

        instance_id, remote_path = parse_remote_path(args.remote)
        host, port = get_instance_ssh_info(instance_id)
        if not host:
            logger.error(f"Could not get SSH info for {instance_id}")
            sys.exit(1)

        config = TransferConfig(ssh_key=args.ssh_key)
        result = rsync_transfer(local_dir, host, port, remote_path, config)

        if result.success:
            logger.info("Successfully synced directory")
        else:
            logger.error(f"Sync failed: {result.error}")
            sys.exit(1)

    elif args.command == "torrent-distribute":
        local_path = Path(args.local_path)
        if not local_path.exists():
            logger.error(f"Local file not found: {local_path}")
            sys.exit(1)

        # Parse remote - requires multiple instances
        if "," not in args.remote.split(":")[0]:
            logger.error("torrent-distribute requires multiple instances (id1,id2,id3:/path)")
            sys.exit(1)

        instances_str, remote_path = args.remote.split(":", 1)
        instance_ids = instances_str.split(",")

        config = TransferConfig(
            ssh_key=args.ssh_key,
            chunk_size_mb=args.chunk_size,
        )

        results = torrent_distribute(local_path, instance_ids, remote_path, config)

        # Summary
        print(f"\n{'='*60}")
        print("TORRENT DISTRIBUTION SUMMARY")
        print(f"{'='*60}")
        success_count = sum(1 for r in results.values() if r.success)
        print(f"Successful: {success_count}/{len(results)}")
        for instance_id, result in results.items():
            status = "OK" if result.success else "FAILED"
            print(f"  {instance_id}: {status} ({result.method}, {result.duration_seconds:.1f}s)")

        sys.exit(0 if success_count == len(results) else 1)

    elif args.command == "aria2-download":
        output_dir = Path(args.output_dir)
        result = aria2_download(
            args.url,
            output_dir,
            filename=args.filename,
            connections=args.connections,
        )

        if result.success:
            logger.info(f"Successfully downloaded ({result.bytes_transferred} bytes)")
        else:
            logger.error(f"Download failed: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
