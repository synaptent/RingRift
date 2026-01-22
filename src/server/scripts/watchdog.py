#!/usr/bin/env python3
"""
External watchdog for RingRift services.

Runs independently of PM2, monitors health endpoints, triggers restarts.
This provides a second layer of resilience beyond PM2's built-in monitoring.

Features:
- Independent health monitoring (not tied to PM2's lifecycle)
- Exponential backoff on restarts (prevents restart storms)
- Configurable failure thresholds per service
- Automatic restart count reset after sustained health

Usage:
    python watchdog.py

Recommended: Run as a systemd service for automatic restart on failure.
See the plan file for systemd service configuration.
"""
import subprocess
import time
import logging
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests module not installed. Run: pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WATCHDOG] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a monitored service."""
    name: str
    health_url: str
    pm2_name: str
    timeout: float = 5.0
    max_failures: int = 3
    backoff_base: float = 30.0
    backoff_max: float = 300.0


# Services to monitor
SERVICES = [
    ServiceConfig(
        name='AI Service',
        health_url='http://localhost:8765/health',
        pm2_name='ringrift-ai',
        timeout=10.0,
        max_failures=3,  # Restart after 3 consecutive failures (45s)
    ),
    ServiceConfig(
        name='Web Server',
        health_url='http://localhost:3001/health',
        pm2_name='ringrift-server',
        timeout=5.0,
        max_failures=5,  # More tolerant for web server (75s)
    ),
]

# How often to check health (seconds)
CHECK_INTERVAL = 15

# How long service must be healthy before resetting restart count (seconds)
STABILITY_THRESHOLD = 600  # 10 minutes


class ServiceWatchdog:
    """Monitors a single service and triggers restarts when unhealthy."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.consecutive_failures = 0
        self.last_restart: Optional[float] = None
        self.restart_count = 0

    def check_health(self) -> bool:
        """Check if the service is healthy via HTTP health endpoint."""
        try:
            resp = requests.get(
                self.config.health_url,
                timeout=self.config.timeout
            )
            return resp.status_code == 200
        except requests.exceptions.Timeout:
            logger.warning(f"{self.config.name} health check timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning(f"{self.config.name} connection refused")
            return False
        except Exception as e:
            logger.warning(f"{self.config.name} health check failed: {e}")
            return False

    def get_backoff_delay(self) -> float:
        """Calculate exponential backoff delay with cap."""
        delay = self.config.backoff_base * (2 ** min(self.restart_count, 5))
        return min(delay, self.config.backoff_max)

    def should_restart(self) -> bool:
        """Determine if we should trigger a restart."""
        if self.consecutive_failures < self.config.max_failures:
            return False
        if self.last_restart is None:
            return True
        elapsed = time.time() - self.last_restart
        backoff = self.get_backoff_delay()
        if elapsed < backoff:
            logger.debug(
                f"{self.config.name} in backoff period ({elapsed:.0f}s < {backoff:.0f}s)"
            )
            return False
        return True

    def restart_service(self):
        """Trigger a PM2 restart for the service."""
        logger.warning(
            f"Restarting {self.config.name} (attempt {self.restart_count + 1})"
        )
        try:
            result = subprocess.run(
                ['pm2', 'restart', self.config.pm2_name],
                check=True,
                capture_output=True,
                timeout=60,
                text=True
            )
            self.last_restart = time.time()
            self.restart_count += 1
            self.consecutive_failures = 0
            logger.info(f"{self.config.name} restart initiated successfully")
            logger.debug(f"PM2 output: {result.stdout}")
        except subprocess.TimeoutExpired:
            logger.error(f"PM2 restart timed out for {self.config.name}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to restart {self.config.name}: {e.stderr}"
            )
        except FileNotFoundError:
            logger.error("PM2 not found. Is it installed and in PATH?")
        except Exception as e:
            logger.error(f"Unexpected error restarting {self.config.name}: {e}")

    def tick(self):
        """Perform one health check cycle."""
        healthy = self.check_health()

        if healthy:
            if self.consecutive_failures > 0:
                logger.info(
                    f"{self.config.name} recovered after "
                    f"{self.consecutive_failures} failures"
                )
            self.consecutive_failures = 0

            # Reset restart count after sustained health
            if self.restart_count > 0 and self.last_restart:
                if time.time() - self.last_restart > STABILITY_THRESHOLD:
                    logger.info(
                        f"{self.config.name} stable for "
                        f"{STABILITY_THRESHOLD}s, resetting restart count"
                    )
                    self.restart_count = 0
        else:
            self.consecutive_failures += 1
            logger.warning(
                f"{self.config.name} failure "
                f"{self.consecutive_failures}/{self.config.max_failures}"
            )
            if self.should_restart():
                self.restart_service()


def main():
    """Main watchdog loop."""
    logger.info("RingRift Watchdog starting...")
    logger.info(f"Monitoring {len(SERVICES)} services every {CHECK_INTERVAL}s")

    for svc in SERVICES:
        logger.info(
            f"  - {svc.name}: {svc.health_url} "
            f"(max_failures={svc.max_failures})"
        )

    watchdogs = [ServiceWatchdog(cfg) for cfg in SERVICES]

    try:
        while True:
            for wd in watchdogs:
                wd.tick()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Watchdog stopped by user")
    except Exception as e:
        logger.error(f"Watchdog crashed: {e}")
        raise


if __name__ == '__main__':
    main()
