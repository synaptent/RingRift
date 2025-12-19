#!/usr/bin/env python3
"""
Data Quality Monitor for Selfplay Training Data

Monitors training data for quality issues and exports Prometheus metrics.
Designed to catch contamination issues like timeout games before they affect training.

Usage:
    python scripts/data_quality_monitor.py --port 9093

    # One-shot audit mode
    python scripts/data_quality_monitor.py --audit-only

    # JSON output
    python scripts/data_quality_monitor.py --audit-only --json

Metrics exported:
    - ringrift_data_games_total: Total games in database
    - ringrift_data_games_with_winner: Games that ended with a winner
    - ringrift_data_games_draws: Games that ended in draw
    - ringrift_data_games_high_moves: Games with moves > 1000 (potential timeout)
    - ringrift_data_draw_rate: Current draw rate (0-1)
    - ringrift_data_quality_score: Overall quality score (0-100)

Alerts:
    - High draw rate (>20%)
    - Games hitting move limits
    - Sudden changes in game statistics
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.logging_config import (
    setup_script_logging,
    get_logger,
    get_metrics_logger,
)

logger = get_logger(__name__)

# Prometheus client
try:
    from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("prometheus_client not installed - metrics export disabled")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of data quality alerts."""
    HIGH_DRAW_RATE = "high_draw_rate"
    GAMES_AT_MOVE_LIMIT = "games_at_move_limit"
    NO_GAMES = "no_games"
    DATABASE_ERROR = "database_error"
    HIGH_MOVES_COUNT = "high_moves_count"


@dataclass
class QualityAlert:
    """A data quality alert."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    board_type: Optional[str] = None
    value: Optional[float] = None
    count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "board_type": self.board_type,
            "value": self.value,
            "count": self.count,
        }


@dataclass
class BoardTypeStats:
    """Statistics for a single board type."""
    board_type: str
    total: int
    with_winner: int
    draws: int
    draw_rate: float
    avg_moves: float
    max_moves: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatabaseAnalysis:
    """Analysis result for a single database."""
    db_name: str
    db_path: str
    timestamp: str
    total_games: int = 0
    quality_score: int = 100
    overall_draw_rate: float = 0.0
    board_types: Dict[str, BoardTypeStats] = field(default_factory=dict)
    alerts: List[QualityAlert] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_name": self.db_name,
            "db_path": self.db_path,
            "timestamp": self.timestamp,
            "total_games": self.total_games,
            "quality_score": self.quality_score,
            "overall_draw_rate": self.overall_draw_rate,
            "board_types": {k: v.to_dict() for k, v in self.board_types.items()},
            "alerts": [a.to_dict() for a in self.alerts],
            "error": self.error,
        }


@dataclass
class AuditSummary:
    """Summary of a full audit run."""
    databases_checked: int
    total_games: int
    total_draws: int
    overall_draw_rate: float
    total_alerts: int
    critical_alerts: int
    results: List[DatabaseAnalysis]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "databases_checked": self.databases_checked,
            "total_games": self.total_games,
            "total_draws": self.total_draws,
            "overall_draw_rate": self.overall_draw_rate,
            "total_alerts": self.total_alerts,
            "critical_alerts": self.critical_alerts,
            "results": [r.to_dict() for r in self.results],
        }


class DataQualityConfig:
    """Configuration for data quality monitoring."""
    DRAW_RATE_THRESHOLD = 0.20  # Alert if draw rate > 20%
    HIGH_MOVES_THRESHOLD = 1000  # Flag games with > 1000 moves
    MOVE_LIMIT_VALUES = [500, 10000]  # Known move limits to watch for


class PrometheusMetrics:
    """Prometheus metrics for data quality monitoring."""

    def __init__(self):
        if not HAS_PROMETHEUS:
            return

        self.games_total = Gauge(
            'ringrift_data_games_total',
            'Total games in database',
            ['db_name', 'board_type']
        )
        self.games_with_winner = Gauge(
            'ringrift_data_games_with_winner',
            'Games that ended with a winner',
            ['db_name', 'board_type']
        )
        self.games_draws = Gauge(
            'ringrift_data_games_draws',
            'Games that ended in draw',
            ['db_name', 'board_type']
        )
        self.games_high_moves = Gauge(
            'ringrift_data_games_high_moves',
            'Games with suspiciously high move counts',
            ['db_name', 'threshold']
        )
        self.games_at_limit = Gauge(
            'ringrift_data_games_at_limit',
            'Games hitting exact move limits',
            ['db_name', 'limit']
        )
        self.draw_rate = Gauge(
            'ringrift_data_draw_rate',
            'Current draw rate',
            ['db_name']
        )
        self.avg_moves = Gauge(
            'ringrift_data_avg_moves',
            'Average moves per game',
            ['db_name', 'board_type']
        )
        self.quality_score = Gauge(
            'ringrift_data_quality_score',
            'Overall data quality score (0-100)',
            ['db_name']
        )
        self.last_check = Gauge(
            'ringrift_data_last_check_timestamp',
            'Timestamp of last quality check',
            ['db_name']
        )

    def export(self, analysis: DatabaseAnalysis) -> None:
        """Export analysis results to Prometheus metrics."""
        if not HAS_PROMETHEUS:
            return

        db_name = analysis.db_name

        for bt_name, bt_stats in analysis.board_types.items():
            self.games_total.labels(db_name=db_name, board_type=bt_name).set(bt_stats.total)
            self.games_with_winner.labels(db_name=db_name, board_type=bt_name).set(bt_stats.with_winner)
            self.games_draws.labels(db_name=db_name, board_type=bt_name).set(bt_stats.draws)
            self.avg_moves.labels(db_name=db_name, board_type=bt_name).set(bt_stats.avg_moves)

        self.draw_rate.labels(db_name=db_name).set(analysis.overall_draw_rate)
        self.quality_score.labels(db_name=db_name).set(analysis.quality_score)
        self.last_check.labels(db_name=db_name).set(time.time())


class DatabaseAnalyzer:
    """Analyzes SQLite databases for data quality issues."""

    def __init__(
        self,
        config: Optional[DataQualityConfig] = None,
        prometheus_metrics: Optional[PrometheusMetrics] = None,
    ):
        self.config = config or DataQualityConfig()
        self.prometheus = prometheus_metrics
        self.metrics = get_metrics_logger("data_quality", log_interval=60)

    def analyze(self, db_path: Path) -> DatabaseAnalysis:
        """Analyze a single database for quality metrics."""
        result = DatabaseAnalysis(
            db_name=db_path.stem,
            db_path=str(db_path),
            timestamp=datetime.now().isoformat(),
        )

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if games table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
            )
            if not cursor.fetchone():
                result.error = "No games table found"
                return result

            # Get column info
            cursor.execute("PRAGMA table_info(games)")
            columns = {row['name'] for row in cursor.fetchall()}

            has_winner = 'winner' in columns
            has_total_moves = 'total_moves' in columns
            has_board_type = 'board_type' in columns

            # Total games
            cursor.execute("SELECT COUNT(*) as cnt FROM games")
            total = cursor.fetchone()['cnt']
            result.total_games = total

            if total == 0:
                result.quality_score = 0
                result.alerts.append(QualityAlert(
                    alert_type=AlertType.NO_GAMES,
                    severity=AlertSeverity.WARNING,
                    message="Database has no games",
                ))
                return result

            # Get board types
            if has_board_type:
                cursor.execute(
                    "SELECT DISTINCT board_type FROM games WHERE board_type IS NOT NULL"
                )
                board_types = [row['board_type'] for row in cursor.fetchall()]
            else:
                board_types = ['unknown']

            # Analyze each board type
            for board_type in board_types:
                bt_stats = self._analyze_board_type(
                    cursor, board_type, has_board_type, has_winner, has_total_moves
                )
                result.board_types[board_type] = bt_stats

                # Check for high draw rate
                if bt_stats.draw_rate > self.config.DRAW_RATE_THRESHOLD:
                    result.alerts.append(QualityAlert(
                        alert_type=AlertType.HIGH_DRAW_RATE,
                        severity=AlertSeverity.WARNING,
                        message=f"High draw rate {bt_stats.draw_rate:.1%} for {board_type}",
                        board_type=board_type,
                        value=bt_stats.draw_rate,
                    ))
                    result.quality_score -= 20

            # Check for move limit hits (timeout contamination)
            if has_total_moves and has_winner:
                for limit in self.config.MOVE_LIMIT_VALUES:
                    at_limit = self._count_games_at_limit(cursor, limit)
                    if at_limit > 0:
                        result.alerts.append(QualityAlert(
                            alert_type=AlertType.GAMES_AT_MOVE_LIMIT,
                            severity=AlertSeverity.CRITICAL,
                            message=f"{at_limit} games hit {limit} move limit (timeout contamination)",
                            count=at_limit,
                            value=float(limit),
                        ))
                        result.quality_score -= min(50, at_limit // 100)

                    if self.prometheus:
                        self.prometheus.games_at_limit.labels(
                            db_name=result.db_name,
                            limit=str(limit),
                        ).set(at_limit)

                # Check for high move games
                high_moves = self._count_high_moves_games(cursor)
                if self.prometheus:
                    self.prometheus.games_high_moves.labels(
                        db_name=result.db_name,
                        threshold=str(self.config.HIGH_MOVES_THRESHOLD),
                    ).set(high_moves)

            # Overall draw rate
            if has_winner:
                cursor.execute("SELECT COUNT(*) as cnt FROM games WHERE winner IS NULL")
                total_draws = cursor.fetchone()['cnt']
                result.overall_draw_rate = total_draws / total if total > 0 else 0
            else:
                result.overall_draw_rate = 0

            result.quality_score = max(0, result.quality_score)
            conn.close()

            # Export to Prometheus
            if self.prometheus:
                self.prometheus.export(result)

            # Update internal metrics
            self.metrics.set("last_analysis_games", result.total_games)
            self.metrics.set("last_analysis_quality", result.quality_score)

        except Exception as e:
            result.error = str(e)
            result.quality_score = 0
            logger.error(f"Error analyzing {db_path}: {e}")

        return result

    def _analyze_board_type(
        self,
        cursor: sqlite3.Cursor,
        board_type: str,
        has_board_type: bool,
        has_winner: bool,
        has_total_moves: bool,
    ) -> BoardTypeStats:
        """Analyze statistics for a single board type."""
        bt_filter = (
            f"WHERE board_type = '{board_type}'"
            if has_board_type and board_type != 'unknown'
            else ""
        )

        # Games with winner vs draws
        if has_winner:
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM games {bt_filter} "
                f"{'AND' if bt_filter else 'WHERE'} winner IS NOT NULL"
            )
            with_winner = cursor.fetchone()['cnt']
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM games {bt_filter} "
                f"{'AND' if bt_filter else 'WHERE'} winner IS NULL"
            )
            draws = cursor.fetchone()['cnt']
        else:
            cursor.execute(f"SELECT COUNT(*) as cnt FROM games {bt_filter}")
            with_winner = cursor.fetchone()['cnt']
            draws = 0

        # Average moves
        if has_total_moves:
            cursor.execute(
                f"SELECT AVG(total_moves) as avg, MAX(total_moves) as max "
                f"FROM games {bt_filter}"
            )
            row = cursor.fetchone()
            avg_moves = row['avg'] or 0
            max_moves = row['max'] or 0
        else:
            avg_moves = 0
            max_moves = 0

        total = with_winner + draws
        draw_rate = draws / total if total > 0 else 0

        return BoardTypeStats(
            board_type=board_type,
            total=total,
            with_winner=with_winner,
            draws=draws,
            draw_rate=draw_rate,
            avg_moves=avg_moves,
            max_moves=max_moves,
        )

    def _count_games_at_limit(self, cursor: sqlite3.Cursor, limit: int) -> int:
        """Count games that hit a specific move limit."""
        cursor.execute(
            f"SELECT COUNT(*) as cnt FROM games "
            f"WHERE total_moves = {limit} AND winner IS NULL"
        )
        return cursor.fetchone()['cnt']

    def _count_high_moves_games(self, cursor: sqlite3.Cursor) -> int:
        """Count games with high move counts."""
        cursor.execute(
            f"SELECT COUNT(*) as cnt FROM games "
            f"WHERE total_moves > {self.config.HIGH_MOVES_THRESHOLD}"
        )
        return cursor.fetchone()['cnt']


class DataQualityAuditor:
    """Runs comprehensive audits of training data."""

    DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
    DATABASE_PATTERNS = ['*.db', 'games/*.db', 'selfplay/*.db', 'canonical/*.db']

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        config: Optional[DataQualityConfig] = None,
    ):
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR
        self.analyzer = DatabaseAnalyzer(config=config)

    def find_databases(self) -> List[Path]:
        """Find all game databases to analyze."""
        dbs = []
        for pattern in self.DATABASE_PATTERNS:
            dbs.extend(self.data_dir.glob(pattern))
        return list(set(dbs))

    def run_audit(self, verbose: bool = True) -> AuditSummary:
        """Run a full audit of all databases."""
        databases = self.find_databases()

        if verbose:
            logger.info(f"Found {len(databases)} databases to analyze")

        results: List[DatabaseAnalysis] = []
        total_games = 0
        total_draws = 0
        total_alerts = 0
        critical_alerts = 0

        for db_path in sorted(databases):
            result = self.analyzer.analyze(db_path)
            results.append(result)

            if result.error:
                if verbose:
                    logger.warning(f"{db_path.name}: ERROR - {result.error}")
                continue

            total_games += result.total_games
            total_alerts += len(result.alerts)
            critical_alerts += sum(
                1 for a in result.alerts if a.severity == AlertSeverity.CRITICAL
            )

            for bt_stats in result.board_types.values():
                total_draws += bt_stats.draws

            if verbose:
                self._print_result(result)

        overall_draw_rate = total_draws / total_games if total_games > 0 else 0

        summary = AuditSummary(
            databases_checked=len(databases),
            total_games=total_games,
            total_draws=total_draws,
            overall_draw_rate=overall_draw_rate,
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            results=results,
        )

        if verbose:
            self._print_summary(summary)

        return summary

    def _print_result(self, result: DatabaseAnalysis) -> None:
        """Print analysis result for a single database."""
        print(f"\n{result.db_name}:")
        print(f"  Total games: {result.total_games:,}")
        print(f"  Quality score: {result.quality_score}/100")

        for bt_name, bt_stats in result.board_types.items():
            print(
                f"  {bt_name}: {bt_stats.total:,} games, "
                f"{bt_stats.draw_rate:.1%} draws, "
                f"avg {bt_stats.avg_moves:.0f} moves"
            )

        for alert in result.alerts:
            severity = alert.severity.value.upper()
            print(f"  [{severity}] {alert.message}")

    def _print_summary(self, summary: AuditSummary) -> None:
        """Print audit summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Databases checked: {summary.databases_checked}")
        print(f"Total games: {summary.total_games:,}")
        print(f"Total draws: {summary.total_draws:,} ({summary.overall_draw_rate:.1%})")
        print(f"Total alerts: {summary.total_alerts} ({summary.critical_alerts} critical)")


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    auditor: DataQualityAuditor
    prometheus: PrometheusMetrics
    last_refresh: float = 0
    refresh_interval: int = 60

    def do_GET(self):
        if self.path == '/metrics':
            self._handle_metrics()
        elif self.path == '/health':
            self._handle_health()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_metrics(self):
        """Handle /metrics endpoint."""
        if not HAS_PROMETHEUS:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Prometheus client not installed')
            return

        # Refresh metrics if needed
        now = time.time()
        if now - MetricsHTTPHandler.last_refresh > MetricsHTTPHandler.refresh_interval:
            self._refresh_metrics()
            MetricsHTTPHandler.last_refresh = now

        self.send_response(200)
        self.send_header('Content-Type', CONTENT_TYPE_LATEST)
        self.end_headers()
        self.wfile.write(generate_latest())

    def _handle_health(self):
        """Handle /health endpoint."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')

    def _refresh_metrics(self):
        """Refresh metrics by analyzing all databases."""
        databases = MetricsHTTPHandler.auditor.find_databases()
        analyzer = DatabaseAnalyzer(prometheus_metrics=MetricsHTTPHandler.prometheus)

        for db_path in databases:
            analyzer.analyze(db_path)

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def run_server(
    port: int,
    data_dir: Path,
    refresh_interval: int = 60,
) -> None:
    """Run the metrics server."""
    if not HAS_PROMETHEUS:
        logger.error("prometheus_client not installed. Run: pip install prometheus_client")
        return

    MetricsHTTPHandler.auditor = DataQualityAuditor(data_dir=data_dir)
    MetricsHTTPHandler.prometheus = PrometheusMetrics()
    MetricsHTTPHandler.refresh_interval = refresh_interval

    # Initial scan
    databases = MetricsHTTPHandler.auditor.find_databases()
    logger.info(f"Monitoring {len(databases)} databases")

    analyzer = DatabaseAnalyzer(prometheus_metrics=MetricsHTTPHandler.prometheus)
    for db_path in databases:
        result = analyzer.analyze(db_path)
        logger.info(
            f"  {db_path.name}: {result.total_games:,} games, "
            f"score={result.quality_score}"
        )

    MetricsHTTPHandler.last_refresh = time.time()

    server = HTTPServer(('0.0.0.0', port), MetricsHTTPHandler)
    logger.info(f"Data quality metrics server running on http://0.0.0.0:{port}/metrics")
    logger.info(f"Refresh interval: {refresh_interval}s")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Quality Monitor for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --port 9093                    # Run metrics server
  %(prog)s --audit-only                   # One-shot audit
  %(prog)s --audit-only --json            # Audit with JSON output
  %(prog)s --audit-only --output report.json
        """,
    )

    parser.add_argument(
        '--port',
        type=int,
        default=9093,
        help='Port for metrics server (default: 9093)',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help='Data directory to monitor',
    )
    parser.add_argument(
        '--audit-only',
        action='store_true',
        help='Run one-shot audit and exit',
    )
    parser.add_argument(
        '--refresh-interval',
        type=int,
        default=60,
        help='Metrics refresh interval in seconds (default: 60)',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for audit results',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output audit results as JSON',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging',
    )
    parser.add_argument(
        '--json-logs',
        action='store_true',
        help='Use JSON format for log files',
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_script_logging(
        script_name="data_quality_monitor",
        level=log_level,
        json_logs=args.json_logs,
    )

    data_dir = Path(args.data_dir)

    if args.audit_only:
        auditor = DataQualityAuditor(data_dir=data_dir)
        summary = auditor.run_audit(verbose=not args.json)

        if args.json or args.output:
            output_data = summary.to_dict()
            output_str = json.dumps(output_data, indent=2)

            if args.json:
                print(output_str)

            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_str)
                logger.info(f"Results saved to {args.output}")

        return 1 if summary.critical_alerts > 0 else 0
    else:
        run_server(args.port, data_dir, args.refresh_interval)
        return 0


if __name__ == '__main__':
    sys.exit(main())
