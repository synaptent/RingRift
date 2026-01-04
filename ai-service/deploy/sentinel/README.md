# RingRift Sentinel

Minimal C binary for OS-level process supervision. Part of the hierarchical supervision architecture:

```
launchd/systemd (OS-level, always running)
    |
    v KeepAlive
ringrift_sentinel (this binary)
    |
    v monitors /tmp/ringrift_watchdog.heartbeat
master_loop_watchdog.py
    |
    v supervises
master_loop.py
    |
    v supervises
DaemonManager -> 109 daemon types
```

## Why a C Binary?

- **No Python dependencies**: Survives Python interpreter crashes
- **Minimal footprint**: ~1MB memory, no external libraries
- **OS-managed lifecycle**: launchd/systemd keeps it running
- **Simple logic**: Just checks file mtime, restarts watchdog if stale

## Building

```bash
cd deploy/sentinel
make
```

## Installation (macOS)

```bash
# Build and install binary
cd deploy/sentinel
make
sudo make install

# Copy launchd plist
sudo cp ../../config/launchd/com.ringrift.sentinel.plist /Library/LaunchDaemons/

# Load the service
sudo launchctl load /Library/LaunchDaemons/com.ringrift.sentinel.plist

# Verify it's running
sudo launchctl list | grep ringrift.sentinel
```

## Uninstallation

```bash
sudo launchctl unload /Library/LaunchDaemons/com.ringrift.sentinel.plist
sudo rm /Library/LaunchDaemons/com.ringrift.sentinel.plist
sudo make uninstall
```

## Configuration

Environment variables (set in launchd plist or export):

| Variable                            | Default                            | Description                      |
| ----------------------------------- | ---------------------------------- | -------------------------------- |
| `RINGRIFT_SENTINEL_HEARTBEAT_PATH`  | `/tmp/ringrift_watchdog.heartbeat` | Path to heartbeat file           |
| `RINGRIFT_SENTINEL_CHECK_INTERVAL`  | `30`                               | Seconds between checks           |
| `RINGRIFT_SENTINEL_STALE_THRESHOLD` | `120`                              | Seconds until heartbeat is stale |
| `RINGRIFT_SENTINEL_STARTUP_GRACE`   | `60`                               | Seconds to wait after restart    |

## Logs

The sentinel logs to syslog. View logs with:

```bash
# macOS
log stream --predicate 'subsystem == "ringrift_sentinel"'

# Or check system log
cat /var/log/system.log | grep ringrift_sentinel
```

## Testing

```bash
# Run in foreground (Ctrl+C to stop)
make test

# Or run binary directly
./ringrift_sentinel
```

## How It Works

1. **Startup**: Waits `STARTUP_GRACE` seconds for system to stabilize
2. **Main loop**: Every `CHECK_INTERVAL` seconds:
   - Checks heartbeat file mtime
   - If file is missing or mtime > `STALE_THRESHOLD` seconds old:
     - Kill any existing watchdog processes
     - Start fresh watchdog process
     - Wait for grace period before next check
3. **Shutdown**: Gracefully exits on SIGTERM/SIGINT

## Heartbeat Protocol

The watchdog writes a JSON heartbeat file every 30 seconds:

```json
{
  "timestamp": 1704312000.123,
  "pid": 12345,
  "node_id": "mac-studio",
  "iteration": 42,
  "status": "running",
  "memory_percent": 45.2
}
```

The sentinel only checks the file's mtime (modification time), not the content.
This keeps the sentinel simple and avoids JSON parsing in C.

## Session 16 (January 2026)

Created as part of the Cluster Resilience Architecture to prevent the 4+ hour
cluster outage caused by memory exhaustion killing the watchdog with no
automatic recovery mechanism.
