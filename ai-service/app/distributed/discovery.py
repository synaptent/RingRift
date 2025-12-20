"""
Worker discovery via Bonjour/mDNS for local Mac cluster.

This module provides automatic discovery of RingRift training workers
on the local network using Bonjour (mDNS/DNS-SD), which is natively
supported on macOS.

Usage:
------
    from app.distributed.discovery import discover_workers, wait_for_workers

    # Quick discovery (waits 2 seconds for responses)
    workers = discover_workers()
    print(f"Found {len(workers)} workers: {workers}")

    # Wait for minimum number of workers
    workers = wait_for_workers(min_workers=3, timeout=30)
    if len(workers) < 3:
        print("Warning: Not enough workers found")

Requirements:
-------------
    pip install zeroconf
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# Service type for RingRift workers
SERVICE_TYPE = "_ringrift-worker._tcp.local."


@dataclass
class WorkerInfo:
    """Information about a discovered worker."""

    worker_id: str
    address: str
    port: int
    hostname: str
    discovered_at: datetime = field(default_factory=datetime.now)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        """Get the worker URL for HTTP requests."""
        return f"{self.address}:{self.port}"

    def __hash__(self) -> int:
        return hash((self.address, self.port))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkerInfo):
            return False
        return self.address == other.address and self.port == other.port


class WorkerDiscovery:
    """
    Discover RingRift training workers on the local network via Bonjour/mDNS.

    This class uses the zeroconf library to discover workers that have
    registered themselves as Bonjour services.

    Usage:
        discovery = WorkerDiscovery()
        discovery.start()
        time.sleep(2)  # Wait for discovery
        workers = discovery.get_workers()
        discovery.stop()
    """

    def __init__(self, service_type: str = SERVICE_TYPE):
        self.service_type = service_type
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = threading.RLock()
        self._zeroconf = None
        self._browser = None
        self._running = False

    def start(self) -> bool:
        """
        Start the worker discovery service.

        Returns True if started successfully, False if zeroconf is not available.
        """
        try:
            from zeroconf import ServiceBrowser, Zeroconf
        except ImportError:
            logger.warning(
                "zeroconf not installed. Run 'pip install zeroconf' for "
                "automatic worker discovery. Using manual worker configuration."
            )
            return False

        try:
            self._zeroconf = Zeroconf()
            self._browser = ServiceBrowser(
                self._zeroconf,
                self.service_type,
                self,
            )
            self._running = True
            logger.info(f"Started worker discovery for {self.service_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to start worker discovery: {e}")
            return False

    def stop(self) -> None:
        """Stop the worker discovery service."""
        self._running = False
        if self._zeroconf:
            try:
                self._zeroconf.close()
            except Exception as e:
                logger.error(f"Error closing zeroconf: {e}")
            self._zeroconf = None
            self._browser = None
        logger.info("Stopped worker discovery")

    def add_service(
        self,
        zc: Any,  # Zeroconf instance
        type_: str,
        name: str,
    ) -> None:
        """Called when a new worker service is discovered."""
        try:
            info = zc.get_service_info(type_, name)
            if info:
                addresses = info.parsed_addresses()
                if addresses:
                    address = addresses[0]
                    port = info.port

                    # Extract properties
                    properties = {}
                    if info.properties:
                        for key, value in info.properties.items():
                            if isinstance(key, bytes):
                                key = key.decode("utf-8")
                            if isinstance(value, bytes):
                                value = value.decode("utf-8")
                            properties[key] = value

                    worker_id = properties.get("worker_id", name.split(".")[0])
                    hostname = properties.get("hostname", "unknown")

                    worker = WorkerInfo(
                        worker_id=worker_id,
                        address=address,
                        port=port,
                        hostname=hostname,
                        properties=properties,
                    )

                    with self._lock:
                        self._workers[worker.url] = worker

                    logger.info(
                        f"Discovered worker: {worker_id} at {worker.url}"
                    )
        except Exception as e:
            logger.error(f"Error processing discovered service {name}: {e}")

    def remove_service(
        self,
        zc: Any,
        type_: str,
        name: str,
    ) -> None:
        """Called when a worker service goes offline."""
        # Find and remove the worker by name
        with self._lock:
            to_remove = None
            for url, worker in self._workers.items():
                if worker.worker_id in name:
                    to_remove = url
                    break
            if to_remove:
                removed = self._workers.pop(to_remove, None)
                if removed:
                    logger.info(f"Worker offline: {removed.worker_id}")

    def update_service(
        self,
        zc: Any,
        type_: str,
        name: str,
    ) -> None:
        """Called when a worker service is updated."""
        # Re-add to update info
        self.add_service(zc, type_, name)

    def get_workers(self) -> list[WorkerInfo]:
        """Get list of currently discovered workers."""
        with self._lock:
            return list(self._workers.values())

    def get_worker_urls(self) -> list[str]:
        """Get list of worker URLs (address:port)."""
        with self._lock:
            return [w.url for w in self._workers.values()]

    def __enter__(self) -> WorkerDiscovery:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def discover_workers(
    discovery_time: float = 2.0,
    service_type: str = SERVICE_TYPE,
) -> list[WorkerInfo]:
    """
    Discover workers on the local network.

    This is a convenience function that creates a WorkerDiscovery instance,
    waits for the specified time, and returns the discovered workers.

    Parameters
    ----------
    discovery_time : float
        Time to wait for worker discovery (seconds).
    service_type : str
        Bonjour service type to search for.

    Returns
    -------
    List[WorkerInfo]
        List of discovered workers.
    """
    with WorkerDiscovery(service_type) as discovery:
        time.sleep(discovery_time)
        return discovery.get_workers()


def wait_for_workers(
    min_workers: int = 1,
    timeout: float = 30.0,
    check_interval: float = 1.0,
    service_type: str = SERVICE_TYPE,
) -> list[WorkerInfo]:
    """
    Wait until at least `min_workers` are discovered or timeout is reached.

    Parameters
    ----------
    min_workers : int
        Minimum number of workers to wait for.
    timeout : float
        Maximum time to wait (seconds).
    check_interval : float
        How often to check for new workers (seconds).
    service_type : str
        Bonjour service type to search for.

    Returns
    -------
    List[WorkerInfo]
        List of discovered workers. May be less than min_workers if timeout.
    """
    with WorkerDiscovery(service_type) as discovery:
        start_time = time.time()

        while time.time() - start_time < timeout:
            workers = discovery.get_workers()
            if len(workers) >= min_workers:
                logger.info(
                    f"Found {len(workers)} workers "
                    f"(required: {min_workers})"
                )
                return workers

            elapsed = time.time() - start_time
            logger.info(
                f"Found {len(workers)}/{min_workers} workers "
                f"({elapsed:.1f}s/{timeout}s)"
            )
            time.sleep(check_interval)

        workers = discovery.get_workers()
        logger.warning(
            f"Timeout waiting for workers. "
            f"Found {len(workers)}, required {min_workers}"
        )
        return workers


def parse_manual_workers(worker_list: str) -> list[WorkerInfo]:
    """
    Parse a comma-separated list of worker addresses.

    This is useful for manual worker configuration when Bonjour
    discovery is not available.

    Parameters
    ----------
    worker_list : str
        Comma-separated list of worker addresses (e.g., "192.168.1.10:8765,192.168.1.11:8765")

    Returns
    -------
    List[WorkerInfo]
        List of WorkerInfo objects.
    """
    workers = []
    for addr in worker_list.split(","):
        addr = addr.strip()
        if not addr:
            continue

        if ":" in addr:
            address, port_str = addr.rsplit(":", 1)
            port = int(port_str)
        else:
            address = addr
            port = 8765  # Default port

        workers.append(
            WorkerInfo(
                worker_id=f"manual-{address}",
                address=address,
                port=port,
                hostname=address,
            )
        )

    return workers


def verify_worker_health(worker: WorkerInfo, timeout: float = 5.0) -> bool:
    """
    Check if a worker is healthy by calling its /health endpoint.

    Parameters
    ----------
    worker : WorkerInfo
        Worker to check.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    bool
        True if worker is healthy, False otherwise.
    """
    try:
        import json
        import urllib.request

        url = f"http://{worker.url}/health"
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("status") == "healthy"
    except Exception as e:
        logger.warning(f"Worker {worker.url} health check failed: {e}")
        return False


def filter_healthy_workers(
    workers: list[WorkerInfo],
    timeout: float = 5.0,
) -> list[WorkerInfo]:
    """
    Filter workers to only include those that respond to health checks.

    Parameters
    ----------
    workers : List[WorkerInfo]
        List of workers to check.
    timeout : float
        Request timeout per worker.

    Returns
    -------
    List[WorkerInfo]
        List of healthy workers.
    """
    healthy = []
    for worker in workers:
        if verify_worker_health(worker, timeout):
            healthy.append(worker)
        else:
            logger.warning(f"Excluding unhealthy worker: {worker.url}")
    return healthy
