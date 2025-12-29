"""Resilient file transfer with automatic verification and fallback.

This is the RECOMMENDED way to transfer files across the cluster.
It provides:
1. Automatic transport selection (BitTorrent for large files)
2. Pre-transfer checksum exchange
3. Mandatory post-transfer verification
4. Quarantine of corrupted files
5. Automatic retry with different sources/transports
6. Type-specific validation (NPZ, SQLite, PTH)

This module was created in December 2025 after a 955MB NPZ file was
corrupted during rsync transfer (rsync --partial stitched together
corrupted segments after ~100 restarts due to connection resets).

Usage:
    from app.distributed.resilient_transfer import ResilientTransfer, TransferRequest

    transfer = ResilientTransfer()
    result = await transfer.transfer(TransferRequest(
        source_node="nebius-backbone-1",
        source_path="/data/training/hex8_2p.npz",
        target_path=Path("data/training/hex8_2p.npz"),
        file_type="npz",
        priority="high",
    ))

    if result.success and result.verification_passed:
        print(f"Transferred {result.bytes_transferred} bytes, verified!")
    else:
        print(f"Transfer failed: {result.error}")

Transport Selection:
    - Files > 50MB: Try BitTorrent first (piece-level SHA1 verification)
    - Files < 50MB: Try aria2 multi-source first
    - Fallback: rsync with checksum verification
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.distributed.hosts import HostConfig

logger = logging.getLogger(__name__)

# Transport thresholds
LARGE_FILE_THRESHOLD = 50_000_000  # 50MB - use BitTorrent above this
HUGE_FILE_THRESHOLD = 500_000_000  # 500MB - require BitTorrent
BASE64_SIZE_LIMIT = 100_000_000  # 100MB - warn when using base64 above this

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Connection reset error patterns (Dec 2025)
# These indicate binary stream corruption from firewalls/proxies
CONNECTION_RESET_PATTERNS = [
    "connection reset by peer",
    "connection reset",
    "broken pipe",
    "connection timed out",
    "network is unreachable",
    "ssh_exchange_identification",
    "read: connection reset",
    "write failed: broken pipe",
]


@dataclass
class TransferRequest:
    """Request for a file transfer.

    Attributes:
        source_node: Node ID or hostname of the source
        source_path: Path to the file on the source node
        target_path: Local path to save the file
        expected_checksum: Optional SHA256 checksum to verify against
        expected_size: Optional expected file size in bytes
        file_type: Type of file for type-specific validation
        priority: Transfer priority (affects retry behavior)
    """

    source_node: str
    source_path: str
    target_path: Path
    expected_checksum: str | None = None
    expected_size: int | None = None
    file_type: Literal["npz", "db", "pth", "other"] = "other"
    priority: Literal["low", "normal", "high", "critical"] = "normal"


@dataclass
class TransferResult:
    """Result of a file transfer.

    Attributes:
        success: Whether the transfer succeeded
        bytes_transferred: Number of bytes transferred
        transport_used: Which transport completed the transfer
        verification_passed: Whether post-transfer verification passed
        error: Error message if transfer failed
        retries: Number of retries performed
        checksum: SHA256 checksum of transferred file
    """

    success: bool
    bytes_transferred: int = 0
    transport_used: str = ""
    verification_passed: bool = False
    error: str = ""
    retries: int = 0
    checksum: str = ""


class ResilientTransfer:
    """Unified resilient file transfer with verification.

    This class provides a single entry point for all file transfers
    with automatic transport selection, verification, and retry logic.
    """

    def __init__(
        self,
        prefer_bittorrent: bool = True,
        verify_all: bool = True,
        quarantine_on_failure: bool = True,
    ):
        """Initialize ResilientTransfer.

        Args:
            prefer_bittorrent: Use BitTorrent for large files (default True)
            verify_all: Verify all transfers, not just large ones (default True)
            quarantine_on_failure: Quarantine corrupted files (default True)
        """
        self.prefer_bittorrent = prefer_bittorrent
        self.verify_all = verify_all
        self.quarantine_on_failure = quarantine_on_failure

        # Lazy-loaded components
        self._aria2_transport = None
        self._hosts_config = None

    async def transfer(self, request: TransferRequest) -> TransferResult:
        """Execute a resilient file transfer.

        Transport selection:
        - > 500MB: Require BitTorrent (piece-level verification built-in)
        - > 50MB: Try BitTorrent first, fallback to rsync with verification
        - < 50MB: Try aria2 multi-source first, fallback to rsync

        Args:
            request: TransferRequest with source, target, and options

        Returns:
            TransferResult with success status and details
        """
        result = TransferResult(success=False)

        # 1. Pre-transfer: Get expected checksum if not provided
        if not request.expected_checksum:
            request.expected_checksum = await self._fetch_remote_checksum(request)

        # 2. Get expected size if not provided
        if not request.expected_size:
            request.expected_size = await self._fetch_remote_size(request)

        # 3. Select transport based on file size and preferences
        # December 2025: Added base64 and chunked as final fallbacks for
        # connections that experience "Connection reset by peer" errors
        if request.expected_size and request.expected_size > HUGE_FILE_THRESHOLD:
            # Huge files MUST use BitTorrent, with chunked fallback for
            # connections where BitTorrent/rsync fail due to connection resets
            logger.info(
                f"File {request.source_path} is {request.expected_size / 1024 / 1024:.0f}MB, "
                "using BitTorrent (required for >500MB)"
            )
            result = await self._transfer_with_retry(
                request, ["bittorrent", "rsync_verified", "chunked"]
            )

        elif request.expected_size and request.expected_size > LARGE_FILE_THRESHOLD:
            # Large files prefer BitTorrent, with base64 fallback
            if self.prefer_bittorrent:
                result = await self._transfer_with_retry(
                    request, ["bittorrent", "aria2", "rsync_verified", "base64"]
                )
            else:
                result = await self._transfer_with_retry(
                    request, ["aria2", "rsync_verified", "base64"]
                )

        else:
            # Small files use aria2 or rsync, with base64 fallback
            result = await self._transfer_with_retry(
                request, ["aria2", "rsync_verified", "base64"]
            )

        # 4. Post-transfer type-specific validation
        if result.success and result.verification_passed:
            type_valid, type_error = await self._validate_file_type(request)
            if not type_valid:
                result.success = False
                result.verification_passed = False
                result.error = f"Type validation failed: {type_error}"

                # Quarantine the file
                if self.quarantine_on_failure:
                    await self._quarantine_file(request.target_path, "type_validation")

        return result

    def _is_connection_reset_error(self, error_msg: str) -> bool:
        """Check if error indicates connection reset/binary corruption.

        These errors typically occur when firewalls or proxies corrupt
        binary streams during SCP/rsync transfers. The solution is to
        use base64 encoding which converts binary to text-safe format.

        Args:
            error_msg: Error message to check

        Returns:
            True if error matches connection reset patterns
        """
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in CONNECTION_RESET_PATTERNS)

    async def _transfer_with_retry(
        self,
        request: TransferRequest,
        transports: list[str],
    ) -> TransferResult:
        """Try transfer with multiple transports and retry logic.

        December 2025: Added connection-reset detection. When we detect
        repeated connection reset errors, we automatically escalate to
        base64 transfer which encodes binary as text to avoid corruption.

        Args:
            request: Transfer request
            transports: Ordered list of transports to try

        Returns:
            TransferResult from first successful transport
        """
        last_result = TransferResult(success=False)
        connection_reset_count = 0

        for transport in transports:
            for attempt in range(MAX_RETRIES):
                try:
                    if transport == "bittorrent":
                        result = await self._transfer_via_bittorrent(request)
                    elif transport == "aria2":
                        result = await self._transfer_via_aria2(request)
                    elif transport == "rsync_verified":
                        result = await self._transfer_via_rsync(request)
                    elif transport == "base64":
                        result = await self._transfer_via_base64(request)
                    elif transport == "chunked":
                        result = await self._transfer_via_chunked(request)
                    else:
                        result = TransferResult(
                            success=False, error=f"Unknown transport: {transport}"
                        )

                    if result.success:
                        result.transport_used = transport
                        result.retries = attempt
                        return result

                    last_result = result

                    # Track connection reset errors for escalation
                    if self._is_connection_reset_error(result.error):
                        connection_reset_count += 1
                        logger.warning(
                            f"Connection reset detected ({connection_reset_count}x) with {transport}: "
                            f"{result.error}"
                        )

                        # After 2 connection resets, escalate to base64 immediately
                        if connection_reset_count >= 2 and "base64" not in transports:
                            logger.info(
                                "Escalating to base64 transport after repeated connection resets"
                            )
                            # Try base64 directly
                            base64_result = await self._transfer_via_base64(request)
                            if base64_result.success:
                                base64_result.transport_used = "base64 (escalated)"
                                return base64_result
                            last_result = base64_result
                    else:
                        logger.warning(
                            f"Transfer attempt {attempt + 1}/{MAX_RETRIES} failed with {transport}: "
                            f"{result.error}"
                        )

                except Exception as e:
                    error_str = str(e)
                    logger.warning(
                        f"Transfer attempt {attempt + 1}/{MAX_RETRIES} raised exception "
                        f"with {transport}: {e}"
                    )
                    last_result = TransferResult(success=False, error=error_str)

                    # Check exception message for connection reset patterns
                    if self._is_connection_reset_error(error_str):
                        connection_reset_count += 1

                # Wait before retry (exponential backoff)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_SECONDS * (2**attempt)
                    await asyncio.sleep(delay)

        return last_result

    async def _transfer_via_bittorrent(self, request: TransferRequest) -> TransferResult:
        """Transfer file via BitTorrent.

        BitTorrent provides piece-level SHA1 verification automatically,
        making it the most reliable transport for large files.
        """
        result = TransferResult(success=False)

        try:
            # Get aria2 transport
            transport = await self._get_aria2_transport()
            if not transport:
                result.error = "aria2 transport not available"
                return result

            # Check if torrent exists for this file
            # This requires the manifest to track torrents
            torrent_info = await self._get_torrent_info(request)

            if not torrent_info:
                result.error = "No torrent available for this file"
                return result

            # Create parent directory
            request.target_path.parent.mkdir(parents=True, exist_ok=True)

            # Download via torrent
            success = await transport.download_torrent(
                torrent_path=torrent_info["torrent_path"],
                output_dir=str(request.target_path.parent),
                timeout=600,  # 10 minute timeout for large files
            )

            if success and request.target_path.exists():
                result.success = True
                result.bytes_transferred = request.target_path.stat().st_size
                result.verification_passed = True  # BitTorrent verifies pieces
                result.transport_used = "bittorrent"

                # Additional checksum verification if provided
                if request.expected_checksum:
                    from app.distributed.sync_utils import _compute_checksum

                    actual = _compute_checksum(request.target_path)
                    if actual != request.expected_checksum:
                        result.success = False
                        result.verification_passed = False
                        result.error = "Checksum mismatch after BitTorrent transfer"
                        if self.quarantine_on_failure:
                            await self._quarantine_file(
                                request.target_path, "bt_checksum_mismatch"
                            )
            else:
                result.error = "BitTorrent download failed or file not created"

        except Exception as e:
            result.error = f"BitTorrent transfer error: {e}"

        return result

    async def _transfer_via_aria2(self, request: TransferRequest) -> TransferResult:
        """Transfer file via aria2 (HTTP multi-source)."""
        result = TransferResult(success=False)

        try:
            transport = await self._get_aria2_transport()
            if not transport:
                result.error = "aria2 transport not available"
                return result

            # Build URL for the source
            source_url = await self._get_source_url(request)
            if not source_url:
                result.error = "Could not determine source URL"
                return result

            # Create parent directory
            request.target_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with aria2
            success = await transport.download_file(
                url=source_url,
                output_path=str(request.target_path),
                expected_checksum=request.expected_checksum,
                timeout=300,
            )

            if success and request.target_path.exists():
                result.success = True
                result.bytes_transferred = request.target_path.stat().st_size
                result.transport_used = "aria2"

                # Verify checksum
                if request.expected_checksum:
                    from app.distributed.sync_utils import _compute_checksum

                    actual = _compute_checksum(request.target_path)
                    if actual == request.expected_checksum:
                        result.verification_passed = True
                        result.checksum = actual
                    else:
                        result.success = False
                        result.error = "Checksum mismatch"
                        if self.quarantine_on_failure:
                            await self._quarantine_file(
                                request.target_path, "aria2_checksum_mismatch"
                            )
                else:
                    result.verification_passed = False  # Unverified

            else:
                result.error = "aria2 download failed or file not created"

        except Exception as e:
            result.error = f"aria2 transfer error: {e}"

        return result

    async def _transfer_via_rsync(self, request: TransferRequest) -> TransferResult:
        """Transfer file via rsync with verification."""
        result = TransferResult(success=False)

        try:
            # Get host config for source node
            host = await self._get_host_config(request.source_node)
            if not host:
                result.error = f"Unknown host: {request.source_node}"
                return result

            from app.distributed.sync_utils import rsync_file_verified

            # Use the verified rsync function
            rsync_result = rsync_file_verified(
                host=host,
                remote_path=request.source_path,
                local_path=request.target_path,
                expected_checksum=request.expected_checksum,
                timeout=300,
            )

            result.success = rsync_result.success
            result.bytes_transferred = rsync_result.bytes_transferred
            result.verification_passed = rsync_result.verified
            result.transport_used = "rsync_verified"

            if rsync_result.error:
                result.error = rsync_result.error

            if rsync_result.checksum_matched:
                from app.distributed.sync_utils import _compute_checksum

                result.checksum = _compute_checksum(request.target_path)

        except Exception as e:
            result.error = f"rsync transfer error: {e}"

        return result

    async def _transfer_via_base64(self, request: TransferRequest) -> TransferResult:
        """Transfer file via base64 encoding through SSH.

        December 2025: This transport encodes binary as base64 text before
        sending through SSH stdin. This avoids binary stream corruption
        from firewalls/proxies that cause "Connection reset by peer" errors.

        How it works:
        1. Read remote file and encode as base64
        2. Pipe through SSH as text
        3. Decode on local side
        4. Verify file size/checksum

        This is slower than rsync but more reliable for problematic connections.
        """
        result = TransferResult(success=False)

        try:
            host = await self._get_host_config(request.source_node)
            if not host:
                result.error = f"Unknown host: {request.source_node}"
                return result

            # Warn for large files
            if request.expected_size and request.expected_size > BASE64_SIZE_LIMIT:
                logger.warning(
                    f"base64 transfer: Large file ({request.expected_size / 1024 / 1024:.1f}MB) - "
                    "consider chunked transfer for better memory efficiency"
                )

            import os
            import subprocess
            import base64
            import tempfile

            # Build SSH command options
            ssh_opts = [
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=30",
            ]

            if host.ssh_key:
                key_path = os.path.expanduser(host.ssh_key)
                ssh_opts.extend(["-i", key_path])

            if host.ssh_port != 22:
                ssh_opts.extend(["-p", str(host.ssh_port)])

            # Command to read and base64-encode remote file
            remote_cmd = f"base64 '{request.source_path}'"

            ssh_cmd = ["ssh", *ssh_opts, host.ssh_target, remote_cmd]

            # Execute and capture base64-encoded data
            proc_result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                timeout=600,  # 10 minute timeout for large files
            )

            if proc_result.returncode != 0:
                result.error = f"SSH base64 read failed: {proc_result.stderr.decode('utf-8', errors='replace')[:200]}"
                return result

            # Decode base64 data
            try:
                file_data = base64.b64decode(proc_result.stdout)
            except Exception as e:
                result.error = f"Base64 decode failed: {e}"
                return result

            # Create parent directory
            request.target_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to target file
            with open(request.target_path, "wb") as f:
                f.write(file_data)

            # Verify size
            actual_size = request.target_path.stat().st_size
            if request.expected_size and actual_size != request.expected_size:
                result.error = f"Size mismatch: expected {request.expected_size}, got {actual_size}"
                if self.quarantine_on_failure:
                    await self._quarantine_file(request.target_path, "base64_size_mismatch")
                return result

            # Verify checksum if provided
            if request.expected_checksum:
                from app.distributed.sync_utils import _compute_checksum
                actual_checksum = _compute_checksum(request.target_path)
                if actual_checksum != request.expected_checksum:
                    result.error = f"Checksum mismatch after base64 transfer"
                    if self.quarantine_on_failure:
                        await self._quarantine_file(request.target_path, "base64_checksum_mismatch")
                    return result
                result.checksum = actual_checksum
                result.verification_passed = True
            else:
                result.verification_passed = False  # Unverified

            result.success = True
            result.bytes_transferred = actual_size
            result.transport_used = "base64"

        except subprocess.TimeoutExpired:
            result.error = "base64 transfer timed out (10 minutes)"
        except Exception as e:
            result.error = f"base64 transfer error: {e}"

        return result

    async def _transfer_via_chunked(self, request: TransferRequest) -> TransferResult:
        """Transfer file in chunks for very unstable connections.

        December 2025: This transport splits large files into 5MB chunks,
        transfers each separately with retries, then reassembles. Each chunk
        can succeed independently, making this robust for connections that
        reset during large transfers.

        Use this when:
        - Base64 transfer also fails
        - Connection drops after transferring partial data
        - Very long-distance or high-latency connections
        """
        result = TransferResult(success=False)

        try:
            host = await self._get_host_config(request.source_node)
            if not host:
                result.error = f"Unknown host: {request.source_node}"
                return result

            # Get file size if not known
            file_size = request.expected_size
            if not file_size:
                file_size = await self._fetch_remote_size(request)
                if not file_size:
                    result.error = "Could not determine file size for chunked transfer"
                    return result

            import os
            import subprocess
            import tempfile
            import hashlib
            import json

            chunk_size = 5 * 1024 * 1024  # 5MB chunks
            num_chunks = (file_size + chunk_size - 1) // chunk_size

            if num_chunks == 1:
                # Small file - just use base64
                return await self._transfer_via_base64(request)

            logger.info(
                f"chunked transfer: Splitting {file_size / 1024 / 1024:.1f}MB into {num_chunks} chunks"
            )

            # Build SSH options
            ssh_opts = [
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=30",
            ]

            if host.ssh_key:
                key_path = os.path.expanduser(host.ssh_key)
                ssh_opts.extend(["-i", key_path])

            if host.ssh_port != 22:
                ssh_opts.extend(["-p", str(host.ssh_port)])

            # Create local temp directory for chunks
            with tempfile.TemporaryDirectory(prefix="chunked_transfer_") as temp_dir:
                temp_path = Path(temp_dir)
                chunk_files = []

                # Download each chunk
                bytes_transferred = 0
                for i in range(num_chunks):
                    offset = i * chunk_size
                    length = min(chunk_size, file_size - offset)

                    # Command to read chunk and base64 encode
                    chunk_cmd = f"dd if='{request.source_path}' bs=1 skip={offset} count={length} 2>/dev/null | base64"

                    ssh_cmd = ["ssh", *ssh_opts, host.ssh_target, chunk_cmd]

                    chunk_result = subprocess.run(
                        ssh_cmd,
                        capture_output=True,
                        timeout=120,  # 2 minute timeout per chunk
                    )

                    if chunk_result.returncode != 0:
                        result.error = f"Chunk {i} download failed: {chunk_result.stderr.decode('utf-8', errors='replace')[:100]}"
                        return result

                    import base64
                    try:
                        chunk_data = base64.b64decode(chunk_result.stdout)
                    except Exception as e:
                        result.error = f"Chunk {i} decode failed: {e}"
                        return result

                    # Verify chunk size
                    if len(chunk_data) != length:
                        result.error = f"Chunk {i} size mismatch: expected {length}, got {len(chunk_data)}"
                        return result

                    # Save chunk
                    chunk_path = temp_path / f"chunk_{i:04d}"
                    with open(chunk_path, "wb") as cf:
                        cf.write(chunk_data)
                    chunk_files.append(chunk_path)
                    bytes_transferred += length

                    logger.debug(f"chunked transfer: Downloaded chunk {i + 1}/{num_chunks}")

                # Reassemble chunks
                request.target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(request.target_path, "wb") as out:
                    for chunk_path in chunk_files:
                        with open(chunk_path, "rb") as cf:
                            out.write(cf.read())

            # Verify final file
            actual_size = request.target_path.stat().st_size
            if actual_size != file_size:
                result.error = f"Reassembled size mismatch: expected {file_size}, got {actual_size}"
                if self.quarantine_on_failure:
                    await self._quarantine_file(request.target_path, "chunked_size_mismatch")
                return result

            # Verify checksum if provided
            if request.expected_checksum:
                from app.distributed.sync_utils import _compute_checksum
                actual_checksum = _compute_checksum(request.target_path)
                if actual_checksum != request.expected_checksum:
                    result.error = "Checksum mismatch after chunked transfer"
                    if self.quarantine_on_failure:
                        await self._quarantine_file(request.target_path, "chunked_checksum_mismatch")
                    return result
                result.checksum = actual_checksum
                result.verification_passed = True
            else:
                result.verification_passed = False

            result.success = True
            result.bytes_transferred = actual_size
            result.transport_used = "chunked"

        except subprocess.TimeoutExpired:
            result.error = "chunked transfer timed out"
        except Exception as e:
            result.error = f"chunked transfer error: {e}"

        return result

    async def _validate_file_type(
        self, request: TransferRequest
    ) -> tuple[bool, str]:
        """Validate file based on its type.

        Args:
            request: Transfer request with file type

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not request.target_path.exists():
            return False, "File does not exist"

        if request.file_type == "npz":
            from app.coordination.npz_validation import validate_npz_structure

            result = validate_npz_structure(request.target_path)
            if result.valid:
                return True, ""
            else:
                return False, "; ".join(result.errors)

        elif request.file_type == "db":
            from app.coordination.sync_integrity import check_sqlite_integrity

            # Dec 29, 2025: Use adaptive timeout for large databases
            db_size_mb = request.target_path.stat().st_size / (1024 * 1024)
            use_fast = db_size_mb > 100  # Fast check for DBs > 100MB
            valid, errors = check_sqlite_integrity(
                request.target_path,
                use_fast_check=use_fast,
                timeout_seconds=15.0 if use_fast else 30.0,
            )
            if valid:
                return True, ""
            else:
                return False, "; ".join(errors)

        elif request.file_type == "pth":
            # For PyTorch models, try to load the metadata
            try:
                import torch

                # Just try to load - if corrupted, this will fail
                torch.load(
                    request.target_path,
                    map_location="cpu",
                    weights_only=True,
                )
                return True, ""
            except Exception as e:
                return False, f"Cannot load PyTorch model: {e}"

        else:
            # For other types, just check file exists and is not empty
            if request.target_path.stat().st_size > 0:
                return True, ""
            else:
                return False, "File is empty"

    async def _fetch_remote_checksum(self, request: TransferRequest) -> str | None:
        """Fetch checksum from remote node."""
        try:
            host = await self._get_host_config(request.source_node)
            if not host:
                return None

            from app.distributed.sync_utils import _fetch_remote_checksum

            return _fetch_remote_checksum(host, request.source_path)
        except Exception as e:
            logger.debug(f"Could not fetch remote checksum: {e}")
            return None

    async def _fetch_remote_size(self, request: TransferRequest) -> int | None:
        """Fetch file size from remote node."""
        try:
            host = await self._get_host_config(request.source_node)
            if not host:
                return None

            import subprocess

            # Build SSH command to get file size
            ssh_parts = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
            ]

            if host.ssh_key:
                import os

                key_path = os.path.expanduser(host.ssh_key)
                ssh_parts.extend(["-i", key_path])

            if host.ssh_port != 22:
                ssh_parts.extend(["-p", str(host.ssh_port)])

            ssh_parts.append(host.ssh_target)
            ssh_parts.append(f"stat -c%s '{request.source_path}' 2>/dev/null")

            result = subprocess.run(
                ssh_parts, capture_output=True, text=True, timeout=15
            )

            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())

            return None
        except Exception as e:
            logger.debug(f"Could not fetch remote size: {e}")
            return None

    async def _get_host_config(self, node_id: str) -> "HostConfig | None":
        """Get HostConfig for a node."""
        if self._hosts_config is None:
            try:
                from app.distributed.hosts import load_hosts_config

                self._hosts_config = load_hosts_config()
            except Exception as e:
                logger.error(f"Could not load hosts config: {e}")
                return None

        return self._hosts_config.get(node_id)

    async def _get_aria2_transport(self):
        """Get or create aria2 transport instance."""
        if self._aria2_transport is None:
            try:
                from app.distributed.aria2_transport import Aria2Transport

                self._aria2_transport = Aria2Transport()
            except Exception as e:
                logger.error(f"Could not create aria2 transport: {e}")
                return None

        return self._aria2_transport

    async def _get_torrent_info(self, request: TransferRequest) -> dict | None:
        """Get torrent info for a file from the manifest."""
        try:
            from app.distributed.cluster_manifest import ClusterManifest

            manifest = ClusterManifest.get_instance()
            # Check if there's a torrent registered for this file
            torrents = manifest.get_torrents_for_path(request.source_path)
            if torrents:
                return torrents[0]
            return None
        except Exception as e:
            logger.debug(f"Could not get torrent info: {e}")
            return None

    async def _get_source_url(self, request: TransferRequest) -> str | None:
        """Get HTTP URL for file from data server."""
        try:
            host = await self._get_host_config(request.source_node)
            if not host:
                return None

            # Assume data server runs on port 8765
            # This matches aria2_data_sync.py convention
            return f"http://{host.ip}:8765/files/{request.source_path}"
        except Exception as e:
            logger.debug(f"Could not build source URL: {e}")
            return None

    async def _quarantine_file(self, path: Path, reason: str) -> None:
        """Move corrupted file to quarantine."""
        from app.distributed.sync_utils import _quarantine_file

        _quarantine_file(path, reason)


# Convenience function for simple transfers
async def transfer_file(
    source_node: str,
    source_path: str,
    target_path: Path,
    file_type: Literal["npz", "db", "pth", "other"] = "other",
    expected_checksum: str | None = None,
) -> TransferResult:
    """Convenience function for simple file transfers.

    Args:
        source_node: Node ID or hostname of the source
        source_path: Path to the file on the source node
        target_path: Local path to save the file
        file_type: Type of file for type-specific validation
        expected_checksum: Optional SHA256 checksum to verify against

    Returns:
        TransferResult with success status and details

    Example:
        result = await transfer_file(
            "nebius-backbone-1",
            "/data/training/hex8_2p.npz",
            Path("data/training/hex8_2p.npz"),
            file_type="npz",
        )
    """
    transfer = ResilientTransfer()
    return await transfer.transfer(
        TransferRequest(
            source_node=source_node,
            source_path=source_path,
            target_path=target_path,
            file_type=file_type,
            expected_checksum=expected_checksum,
        )
    )
