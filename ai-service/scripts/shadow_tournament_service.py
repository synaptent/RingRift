#!/usr/bin/env python3
"""Shadow Tournament Service - Continuous lightweight model evaluation.

This service provides quick feedback on model quality by running mini-tournaments
every 15 minutes instead of waiting for full 6-hour Elo calibration.

Features:
1. Shadow tournaments: 10-20 games every 15 minutes for quick feedback
2. Full tournaments: 50 games every hour for comprehensive evaluation
3. Checkpoint monitoring: Watch training directories for new models
4. Early regression detection: Alert if model performance drops
5. Event emission: Notify downstream systems of evaluation results

Usage:
    # Run as standalone service
    python scripts/shadow_tournament_service.py

    # Watch a specific training run
    python scripts/shadow_tournament_service.py --watch-dir runs/training_001/models

    # Run single evaluation
    python scripts/shadow_tournament_service.py --once --config square8_2p

    # Quick mode (10 games)
    python scripts/shadow_tournament_service.py --quick --config square8_2p
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Import event bus helpers (consolidated imports)
from app.distributed.event_helpers import (
    has_event_bus,
    get_event_bus_safe,
    emit_evaluation_completed_safe,
    emit_error_safe,
)
HAS_EVENT_BUS = has_event_bus()

# For backwards compatibility, get the raw functions if available
if HAS_EVENT_BUS:
    from app.distributed.data_events import get_event_bus, emit_evaluation_completed, emit_error
else:
    get_event_bus = get_event_bus_safe
    emit_evaluation_completed = emit_evaluation_completed_safe
    emit_error = emit_error_safe

# Prometheus metrics for observability
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    HAS_PROMETHEUS = True

    # Tournament metrics
    TOURNAMENTS_TOTAL = Counter(
        'shadow_tournaments_total',
        'Total tournaments run',
        ['tournament_type', 'config', 'success']
    )
    TOURNAMENT_DURATION = Histogram(
        'shadow_tournament_duration_seconds',
        'Tournament duration in seconds',
        ['tournament_type', 'config'],
        buckets=[10, 30, 60, 120, 300, 600, 1200]
    )
    TOURNAMENT_WIN_RATE = Gauge(
        'shadow_tournament_win_rate',
        'Win rate from last tournament',
        ['config']
    )
    TOURNAMENT_ELO = Gauge(
        'shadow_tournament_elo_estimate',
        'Elo estimate from last tournament',
        ['config']
    )
    GAMES_PLAYED_TOTAL = Counter(
        'shadow_tournament_games_total',
        'Total games played in tournaments',
        ['config']
    )
    REGRESSION_DETECTED = Counter(
        'shadow_tournament_regressions_total',
        'Total regressions detected',
        ['config']
    )
    SERVICE_UP = Gauge(
        'shadow_tournament_service_up',
        'Whether the shadow tournament service is running'
    )
except ImportError:
    HAS_PROMETHEUS = False

# Try to import HealthRegistry for distributed health awareness
try:
    from app.distributed.health_registry import register_health
    HAS_HEALTH_REGISTRY = True
except ImportError:
    HAS_HEALTH_REGISTRY = False

# Try to import canonical config
try:
    from app.config.unified_config import (
        get_shadow_tournament_interval,
        get_full_tournament_interval,
        get_tournament_games_per_matchup,
        get_regression_elo_threshold,
    )
    HAS_UNIFIED_CONFIG = True
except ImportError:
    HAS_UNIFIED_CONFIG = False


# Board/player configurations
ALL_CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


def _get_default_shadow_interval() -> int:
    """Get default shadow interval from canonical config or fallback."""
    if HAS_UNIFIED_CONFIG:
        return get_shadow_tournament_interval()
    return 300  # 5 minutes fallback


def _get_default_full_interval() -> int:
    """Get default full interval from canonical config or fallback."""
    if HAS_UNIFIED_CONFIG:
        return get_full_tournament_interval()
    return 3600  # 1 hour fallback


def _get_default_games() -> int:
    """Get default games per matchup from canonical config or fallback."""
    if HAS_UNIFIED_CONFIG:
        return get_tournament_games_per_matchup()
    return 20  # fallback


@dataclass
class TournamentConfig:
    """Configuration for shadow tournaments."""
    shadow_interval_seconds: int = field(default_factory=_get_default_shadow_interval)
    shadow_games: int = 15  # OPTIMIZED: 15 games (was 10) for better signal
    full_interval_seconds: int = field(default_factory=_get_default_full_interval)
    full_games: int = 50
    include_baselines: bool = True
    baseline_models: list[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100"])
    timeout_seconds: int = 600  # Per tournament
    concurrent_tournaments: int = 4  # OPTIMIZED: 4 parallel tournaments (was 1)


@dataclass
class EvaluationResult:
    """Result of a tournament evaluation."""
    config: str
    board_type: str
    num_players: int
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    elo_estimate: float
    duration_seconds: float
    timestamp: float
    tournament_type: str  # "shadow" or "full"
    success: bool
    error: str | None = None


class ShadowTournamentService:
    """Continuous lightweight model evaluation service."""

    def __init__(self, config: TournamentConfig, http_port: int = 8771, prometheus_port: int = 8772):
        self.config = config
        self.http_port = http_port
        self.prometheus_port = prometheus_port
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_shadow: dict[str, float] = {}
        self._last_full: float = 0.0
        self._watched_dirs: set[Path] = set()
        self._known_checkpoints: set[str] = set()
        self._results_history: list[EvaluationResult] = []
        self._elo_baselines: dict[str, float] = {}  # Baseline Elo for regression detection
        # Use canonical config for regression threshold if available
        self._regression_threshold: float = (
            get_regression_elo_threshold() if HAS_UNIFIED_CONFIG else 30.0
        )
        self._app: Any | None = None
        self._http_runner: Any | None = None
        self._health_registered = False

    async def run_shadow_tournament(
        self,
        board_type: str,
        num_players: int,
        model_path: str | None = None,
    ) -> EvaluationResult:
        """Run a quick shadow tournament for a configuration."""
        config_key = f"{board_type}_{num_players}p"
        start_time = time.time()

        try:
            # Build command
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", str(self.config.shadow_games),
                "--quick",
            ]

            if self.config.include_baselines:
                cmd.append("--include-baselines")

            if model_path:
                cmd.extend(["--model", model_path])

            # Run tournament
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds
            )

            duration = time.time() - start_time
            success = process.returncode == 0

            # Parse results
            result = EvaluationResult(
                config=config_key,
                board_type=board_type,
                num_players=num_players,
                games_played=self.config.shadow_games,
                wins=0,
                losses=0,
                draws=0,
                win_rate=0.5,
                elo_estimate=1500.0,
                duration_seconds=duration,
                timestamp=time.time(),
                tournament_type="shadow",
                success=success,
                error=None if success else stderr.decode()[:500],
            )

            # Try to parse actual results from stdout
            try:
                output = stdout.decode()
                for line in output.split("\n"):
                    if "Win rate:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            result.win_rate = float(parts[1].strip().rstrip("%")) / 100
                    if "Elo:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            result.elo_estimate = float(parts[1].strip().split()[0])
            except Exception:
                pass

            self._results_history.append(result)
            self._last_shadow[config_key] = time.time()

            # Record Prometheus metrics
            if HAS_PROMETHEUS:
                TOURNAMENTS_TOTAL.labels(
                    tournament_type="shadow",
                    config=config_key,
                    success=str(success).lower()
                ).inc()
                TOURNAMENT_DURATION.labels(
                    tournament_type="shadow",
                    config=config_key
                ).observe(duration)
                if success:
                    TOURNAMENT_WIN_RATE.labels(config=config_key).set(result.win_rate)
                    TOURNAMENT_ELO.labels(config=config_key).set(result.elo_estimate)
                    GAMES_PLAYED_TOTAL.labels(config=config_key).inc(result.games_played)

            # Emit event
            if HAS_EVENT_BUS:
                await emit_evaluation_completed(
                    config_key,
                    result.elo_estimate,
                    result.games_played,
                    result.win_rate,
                    source="shadow_tournament_service",
                )

            return result

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            result = EvaluationResult(
                config=config_key,
                board_type=board_type,
                num_players=num_players,
                games_played=0,
                wins=0, losses=0, draws=0,
                win_rate=0.0,
                elo_estimate=0.0,
                duration_seconds=duration,
                timestamp=time.time(),
                tournament_type="shadow",
                success=False,
                error="Timeout",
            )
            self._results_history.append(result)

            # Record failure metric
            if HAS_PROMETHEUS:
                TOURNAMENTS_TOTAL.labels(
                    tournament_type="shadow",
                    config=config_key,
                    success="false"
                ).inc()

            if HAS_EVENT_BUS:
                await emit_error("shadow_tournament", f"Timeout for {config_key}", source="shadow_tournament_service")

            return result

        except Exception as e:
            duration = time.time() - start_time
            result = EvaluationResult(
                config=config_key,
                board_type=board_type,
                num_players=num_players,
                games_played=0,
                wins=0, losses=0, draws=0,
                win_rate=0.0,
                elo_estimate=0.0,
                duration_seconds=duration,
                timestamp=time.time(),
                tournament_type="shadow",
                success=False,
                error=str(e),
            )
            self._results_history.append(result)

            # Record failure metric
            if HAS_PROMETHEUS:
                TOURNAMENTS_TOTAL.labels(
                    tournament_type="shadow",
                    config=config_key,
                    success="false"
                ).inc()

            return result

    async def run_parallel_shadow_tournaments(
        self,
        configs: list[tuple] | None = None,
        max_concurrent: int | None = None,
    ) -> list[EvaluationResult]:
        """Run shadow tournaments in parallel for faster evaluation.

        OPTIMIZED: Run up to N tournaments concurrently to maximize throughput.
        This significantly reduces total evaluation time when cluster has capacity.

        Args:
            configs: List of (board_type, num_players) tuples to evaluate
            max_concurrent: Max concurrent tournaments (default: config.concurrent_tournaments)

        Returns:
            List of EvaluationResults for all configs
        """
        if configs is None:
            # Get configs that need evaluation
            configs = []
            now = time.time()
            for board_type, num_players in ALL_CONFIGS:
                config_key = f"{board_type}_{num_players}p"
                last = self._last_shadow.get(config_key, 0)
                if now - last >= self.config.shadow_interval_seconds:
                    configs.append((board_type, num_players))

        if not configs:
            return []

        max_concurrent = max_concurrent or self.config.concurrent_tournaments
        results = []

        # Process in batches of max_concurrent
        for i in range(0, len(configs), max_concurrent):
            batch = configs[i:i + max_concurrent]
            print(f"[ShadowTournament] Running parallel batch: {[f'{b[0]}_{b[1]}p' for b in batch]}")

            # Create concurrent tasks for this batch
            tasks = [
                self.run_shadow_tournament(board_type, num_players)
                for board_type, num_players in batch
            ]

            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    board_type, num_players = batch[j]
                    config_key = f"{board_type}_{num_players}p"
                    error_result = EvaluationResult(
                        config=config_key,
                        board_type=board_type,
                        num_players=num_players,
                        games_played=0,
                        wins=0, losses=0, draws=0,
                        win_rate=0.0,
                        elo_estimate=0.0,
                        duration_seconds=0.0,
                        timestamp=time.time(),
                        tournament_type="shadow",
                        success=False,
                        error=str(result),
                    )
                    results.append(error_result)
                else:
                    results.append(result)
                    if result.success:
                        # Check for regression
                        regression = self.check_regression(result)
                        if regression:
                            print(f"[ShadowTournament] ⚠️  REGRESSION: {result.config} dropped {regression['drop']:.0f} Elo")
                            # Record regression metric
                            if HAS_PROMETHEUS:
                                REGRESSION_DETECTED.labels(config=result.config).inc()

        print(f"[ShadowTournament] Parallel evaluation complete: {len([r for r in results if r.success])}/{len(results)} succeeded")
        return results

    async def run_full_tournament(
        self,
        configs: list[tuple] | None = None,
    ) -> list[EvaluationResult]:
        """Run a full tournament across configurations."""
        if configs is None:
            configs = ALL_CONFIGS

        results = []
        start_time = time.time()

        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
                "--all-configs",
                "--games", str(self.config.full_games),
            ]

            if self.config.include_baselines:
                cmd.append("--include-baselines")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds * len(configs)
            )

            duration = time.time() - start_time
            success = process.returncode == 0

            for board_type, num_players in configs:
                config_key = f"{board_type}_{num_players}p"
                result = EvaluationResult(
                    config=config_key,
                    board_type=board_type,
                    num_players=num_players,
                    games_played=self.config.full_games,
                    wins=0, losses=0, draws=0,
                    win_rate=0.5,
                    elo_estimate=1500.0,
                    duration_seconds=duration / len(configs),
                    timestamp=time.time(),
                    tournament_type="full",
                    success=success,
                    error=None if success else stderr.decode()[:200],
                )
                results.append(result)
                self._results_history.append(result)

            self._last_full = time.time()

            # Record Prometheus metrics for full tournament
            if HAS_PROMETHEUS:
                TOURNAMENTS_TOTAL.labels(
                    tournament_type="full",
                    config="all",
                    success=str(success).lower()
                ).inc()
                TOURNAMENT_DURATION.labels(
                    tournament_type="full",
                    config="all"
                ).observe(duration)
                for result in results:
                    if result.success:
                        TOURNAMENT_WIN_RATE.labels(config=result.config).set(result.win_rate)
                        TOURNAMENT_ELO.labels(config=result.config).set(result.elo_estimate)
                        GAMES_PLAYED_TOTAL.labels(config=result.config).inc(result.games_played)

            if HAS_EVENT_BUS:
                for result in results:
                    await emit_evaluation_completed(
                        result.config,
                        result.elo_estimate,
                        result.games_played,
                        result.win_rate,
                        source="shadow_tournament_service",
                    )

        except Exception as e:
            for board_type, num_players in configs:
                config_key = f"{board_type}_{num_players}p"
                result = EvaluationResult(
                    config=config_key,
                    board_type=board_type,
                    num_players=num_players,
                    games_played=0,
                    wins=0, losses=0, draws=0,
                    win_rate=0.0,
                    elo_estimate=0.0,
                    duration_seconds=0.0,
                    timestamp=time.time(),
                    tournament_type="full",
                    success=False,
                    error=str(e),
                )
                results.append(result)

            # Record failure metric
            if HAS_PROMETHEUS:
                TOURNAMENTS_TOTAL.labels(
                    tournament_type="full",
                    config="all",
                    success="false"
                ).inc()

            if HAS_EVENT_BUS:
                await emit_error("full_tournament", str(e), source="shadow_tournament_service")

        return results

    async def run_checkpoint_evaluation(self, checkpoint_path: Path) -> EvaluationResult:
        """Evaluate a specific training checkpoint."""
        board_type = "square8"
        num_players = 2

        path_str = str(checkpoint_path).lower()
        for bt in ["square8", "square19", "hexagonal"]:
            if bt in path_str:
                board_type = bt
                break
        for np in [2, 3, 4]:
            if f"_{np}p" in path_str or f"{np}p" in path_str:
                num_players = np
                break

        return await self.run_shadow_tournament(
            board_type, num_players, str(checkpoint_path)
        )

    def add_watch_dir(self, dir_path: Path):
        """Add a directory to watch for new checkpoints."""
        self._watched_dirs.add(dir_path)

    async def _check_watched_dirs(self):
        """Check watched directories for new checkpoints."""
        for watch_dir in self._watched_dirs:
            if not watch_dir.exists():
                continue

            for pth_file in watch_dir.glob("*.pth"):
                file_key = f"{pth_file}:{pth_file.stat().st_mtime}"
                if file_key not in self._known_checkpoints:
                    self._known_checkpoints.add(file_key)
                    print(f"[ShadowTournament] New checkpoint: {pth_file.name}")
                    await self.run_checkpoint_evaluation(pth_file)

    def get_needs_shadow_eval(self) -> tuple | None:
        """Get next configuration needing shadow evaluation."""
        now = time.time()

        for board_type, num_players in ALL_CONFIGS:
            config_key = f"{board_type}_{num_players}p"
            last = self._last_shadow.get(config_key, 0)

            if now - last >= self.config.shadow_interval_seconds:
                return (board_type, num_players)

        return None

    def get_all_needs_shadow_eval(self) -> list[tuple]:
        """Get all configurations needing shadow evaluation.

        OPTIMIZED: Returns all configs that are due for evaluation,
        allowing parallel processing instead of one at a time.
        """
        now = time.time()
        configs = []

        for board_type, num_players in ALL_CONFIGS:
            config_key = f"{board_type}_{num_players}p"
            last = self._last_shadow.get(config_key, 0)

            if now - last >= self.config.shadow_interval_seconds:
                configs.append((board_type, num_players))

        return configs

    def needs_full_eval(self) -> bool:
        """Check if full tournament is due."""
        return time.time() - self._last_full >= self.config.full_interval_seconds

    def get_recent_results(self, limit: int = 50) -> list[EvaluationResult]:
        """Get recent evaluation results."""
        return self._results_history[-limit:]

    def get_elo_trend(self, config: str, window: int = 10) -> float:
        """Get Elo trend for a configuration (positive = improving)."""
        config_results = [r for r in self._results_history if r.config == config and r.success]
        if len(config_results) < 2:
            return 0.0

        recent = config_results[-window:]
        if len(recent) < 2:
            return 0.0

        elos = [r.elo_estimate for r in recent]
        avg_early = sum(elos[:len(elos)//2]) / (len(elos)//2)
        avg_late = sum(elos[len(elos)//2:]) / (len(elos) - len(elos)//2)

        return avg_late - avg_early

    async def run(self):
        """Main service loop.

        OPTIMIZED: Uses parallel evaluation to maximize throughput.
        """
        self._running = True
        print(f"[ShadowTournament] Starting - shadow every {self.config.shadow_interval_seconds}s, "
              f"full every {self.config.full_interval_seconds}s, "
              f"max_concurrent={self.config.concurrent_tournaments}")

        # Start Prometheus metrics server
        if HAS_PROMETHEUS:
            try:
                start_http_server(self.prometheus_port)
                SERVICE_UP.set(1)
                print(f"[ShadowTournament] Prometheus metrics on port {self.prometheus_port}")
            except Exception as e:
                print(f"[ShadowTournament] Failed to start Prometheus server: {e}")

        # Register with HealthRegistry for distributed health awareness
        if HAS_HEALTH_REGISTRY and not self._health_registered:
            try:
                register_health(
                    component="shadow_tournament_service",
                    check_fn=lambda: {"status": "healthy" if self._running else "stopped"},
                    interval_seconds=30,
                )
                self._health_registered = True
                print("[ShadowTournament] Registered with HealthRegistry")
            except Exception as e:
                print(f"[ShadowTournament] Failed to register health: {e}")

        # Start HTTP API
        await self._setup_http()

        while self._running:
            try:
                if self._watched_dirs:
                    await self._check_watched_dirs()

                # OPTIMIZED: Use parallel evaluation instead of sequential
                # Get all configs that need evaluation
                configs_to_eval = self.get_all_needs_shadow_eval()
                if configs_to_eval:
                    print(f"[ShadowTournament] {len(configs_to_eval)} configs need evaluation, running in parallel")
                    results = await self.run_parallel_shadow_tournaments(configs_to_eval)

                    # Log successful results
                    for result in results:
                        if result.success:
                            print(f"[ShadowTournament] {result.config}: win_rate={result.win_rate:.2%}, elo~{result.elo_estimate:.0f}")

                if self.needs_full_eval():
                    print("[ShadowTournament] Running full tournament")
                    await self.run_full_tournament()

            except Exception as e:
                print(f"[ShadowTournament] Error: {e}")

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
                break
            except asyncio.TimeoutError:
                pass

        # Cleanup HTTP
        await self._cleanup_http()

        # Update Prometheus metric on shutdown
        if HAS_PROMETHEUS:
            SERVICE_UP.set(0)

        print("[ShadowTournament] Stopped")

    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()
        if HAS_PROMETHEUS:
            SERVICE_UP.set(0)

    def check_regression(self, result: EvaluationResult) -> dict[str, Any] | None:
        """Check if result shows regression from baseline."""
        if not result.success:
            return None

        config_key = result.config
        elo = result.elo_estimate

        # Initialize or update baseline
        if config_key not in self._elo_baselines:
            self._elo_baselines[config_key] = elo
            return None

        baseline = self._elo_baselines[config_key]
        drop = baseline - elo

        # Update baseline if improving
        if elo > baseline:
            self._elo_baselines[config_key] = elo
            return None

        # Check for regression
        if drop >= self._regression_threshold:
            return {
                "config": config_key,
                "baseline_elo": baseline,
                "current_elo": elo,
                "drop": drop,
                "timestamp": result.timestamp,
            }

        return None

    # HTTP API methods
    async def _setup_http(self):
        """Set up HTTP API server."""
        try:
            from aiohttp import web
        except ImportError:
            print("[ShadowTournament] aiohttp not installed, HTTP API disabled")
            return

        self._app = web.Application()
        self._app.router.add_get('/health', self._handle_health)
        self._app.router.add_get('/status', self._handle_status)
        self._app.router.add_get('/results', self._handle_results)
        self._app.router.add_get('/trends', self._handle_trends)
        self._app.router.add_post('/trigger', self._handle_trigger)

        self._http_runner = web.AppRunner(self._app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, '0.0.0.0', self.http_port)
        await site.start()
        print(f"[ShadowTournament] HTTP API listening on port {self.http_port}")

    async def _cleanup_http(self):
        """Clean up HTTP server."""
        if self._http_runner:
            await self._http_runner.cleanup()

    async def _handle_health(self, request) -> Any:
        """GET /health - Health check."""
        from aiohttp import web
        return web.json_response({"status": "healthy", "running": self._running})

    async def _handle_status(self, request) -> Any:
        """GET /status - Service status."""
        from aiohttp import web

        now = time.time()
        status = {
            "running": self._running,
            "shadow_interval": self.config.shadow_interval_seconds,
            "full_interval": self.config.full_interval_seconds,
            "results_count": len(self._results_history),
            "watched_dirs": [str(d) for d in self._watched_dirs],
            "last_shadow": {
                k: now - v for k, v in self._last_shadow.items()
            },
            "time_since_full": now - self._last_full if self._last_full else None,
            "elo_baselines": self._elo_baselines,
        }
        return web.json_response(status)

    async def _handle_results(self, request) -> Any:
        """GET /results - Recent evaluation results."""
        from aiohttp import web

        limit = int(request.query.get('limit', '20'))
        config_filter = request.query.get('config')

        results = self._results_history[-limit:]
        if config_filter:
            results = [r for r in results if r.config == config_filter]

        return web.json_response([{
            "config": r.config,
            "win_rate": r.win_rate,
            "elo_estimate": r.elo_estimate,
            "games_played": r.games_played,
            "tournament_type": r.tournament_type,
            "success": r.success,
            "error": r.error,
            "timestamp": r.timestamp,
            "duration": r.duration_seconds,
        } for r in results])

    async def _handle_trends(self, request) -> Any:
        """GET /trends - Elo trends for each config."""
        from aiohttp import web

        trends = {}
        for board_type, num_players in ALL_CONFIGS:
            config_key = f"{board_type}_{num_players}p"
            trend = self.get_elo_trend(config_key)
            trends[config_key] = {
                "trend": trend,
                "baseline": self._elo_baselines.get(config_key, 0),
                "improving": trend > 5,
                "regressing": trend < -10,
            }

        return web.json_response(trends)

    async def _handle_trigger(self, request) -> Any:
        """POST /trigger - Trigger evaluation manually."""
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:
            data = {}

        config = data.get('config', 'square8_2p')
        full = data.get('full', False)

        if full:
            asyncio.create_task(self.run_full_tournament())
            return web.json_response({"triggered": "full", "status": "started"})
        else:
            parts = config.split("_")
            board_type = parts[0]
            num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2
            asyncio.create_task(self.run_shadow_tournament(board_type, num_players))
            return web.json_response({"triggered": config, "status": "started"})


def main():
    parser = argparse.ArgumentParser(description="Shadow Tournament Service")
    parser.add_argument("--config", type=str, help="Config key (e.g., square8_2p)")
    parser.add_argument("--watch-dir", type=str, help="Directory to watch for checkpoints")
    parser.add_argument("--once", action="store_true", help="Run one evaluation and exit")
    parser.add_argument("--quick", action="store_true", help="Run quick (10 games) evaluation")
    parser.add_argument("--full", action="store_true", help="Run full tournament")
    parser.add_argument("--shadow-interval", type=int, default=900, help="Shadow interval seconds")
    parser.add_argument("--shadow-games", type=int, default=10, help="Games per shadow tournament")
    parser.add_argument("--full-interval", type=int, default=3600, help="Full tournament interval seconds")
    parser.add_argument("--full-games", type=int, default=50, help="Games per full tournament")
    parser.add_argument("--http-port", type=int, default=8771, help="HTTP API port")
    parser.add_argument("--regression-threshold", type=float, default=30.0, help="Elo drop threshold for regression alert")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = TournamentConfig(
        shadow_interval_seconds=args.shadow_interval,
        shadow_games=args.shadow_games,
        full_interval_seconds=args.full_interval,
        full_games=args.full_games,
    )

    if args.quick:
        config.shadow_games = 10

    service = ShadowTournamentService(config, http_port=args.http_port)
    service._regression_threshold = args.regression_threshold

    if args.watch_dir:
        service.add_watch_dir(Path(args.watch_dir))

    import signal

    def signal_handler(sig, frame):
        print("\n[ShadowTournament] Shutdown requested")
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.once:
        if args.config:
            parts = args.config.split("_")
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            result = asyncio.run(service.run_shadow_tournament(board_type, num_players))
            print(f"Result: {result}")
        elif args.full:
            results = asyncio.run(service.run_full_tournament())
            for r in results:
                print(f"  {r.config}: win_rate={r.win_rate:.2%}, success={r.success}")
        else:
            asyncio.run(service.run_shadow_tournament("square8", 2))
    else:
        asyncio.run(service.run())


if __name__ == "__main__":
    main()
