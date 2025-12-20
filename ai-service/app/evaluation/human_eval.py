"""
Human Evaluation Interface for RingRift AI.

Provides tools for collecting human feedback on AI play quality,
comparing models, and gathering annotations for training improvement.
"""

import json
import logging
import sqlite3
import threading
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    """Types of human evaluation."""
    MOVE_QUALITY = "move_quality"          # Rate individual moves
    POSITION_ASSESSMENT = "position"        # Evaluate position understanding
    HUMAN_LIKENESS = "human_likeness"       # How human-like is play
    PREFERENCE = "preference"               # A/B model comparison
    ANNOTATION = "annotation"               # Free-form position annotations
    GAME_QUALITY = "game_quality"           # Overall game assessment


class MoveQuality(Enum):
    """Move quality ratings."""
    BRILLIANT = 5
    EXCELLENT = 4
    GOOD = 3
    ACCEPTABLE = 2
    QUESTIONABLE = 1
    BLUNDER = 0


@dataclass
class EvaluationTask:
    """A single evaluation task for a human evaluator."""
    task_id: str
    eval_type: EvaluationType
    model_id: str
    position_data: dict[str, Any]
    ai_move: int | None = None
    alternatives: list[int] | None = None
    comparison_model_id: str | None = None
    comparison_move: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['eval_type'] = self.eval_type.value
        d['created_at'] = self.created_at.isoformat()
        return d


@dataclass
class EvaluationResponse:
    """Human evaluator's response to a task."""
    task_id: str
    evaluator_id: str
    rating: int | None = None
    preference: str | None = None  # 'a', 'b', or 'tie'
    annotation: str | None = None
    suggested_move: int | None = None
    confidence: float | None = None
    time_spent_seconds: float | None = None
    submitted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['submitted_at'] = self.submitted_at.isoformat()
        return d


@dataclass
class EvaluatorProfile:
    """Profile for a human evaluator."""
    evaluator_id: str
    name: str
    skill_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    evaluations_completed: int = 0
    agreement_score: float = 1.0  # How often they agree with consensus
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d


class EvaluationDatabase:
    """SQLite database for human evaluations."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evaluators (
                    evaluator_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    skill_level TEXT DEFAULT 'intermediate',
                    evaluations_completed INTEGER DEFAULT 0,
                    agreement_score REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    eval_type TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    position_data TEXT NOT NULL,
                    ai_move INTEGER,
                    alternatives TEXT,
                    comparison_model_id TEXT,
                    comparison_move INTEGER,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL,
                    rating INTEGER,
                    preference TEXT,
                    annotation TEXT,
                    suggested_move INTEGER,
                    confidence REAL,
                    time_spent_seconds REAL,
                    submitted_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id),
                    FOREIGN KEY (evaluator_id) REFERENCES evaluators(evaluator_id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    evaluator_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    game_state TEXT,
                    moves_played TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    game_result TEXT,
                    feedback TEXT,
                    FOREIGN KEY (evaluator_id) REFERENCES evaluators(evaluator_id)
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_model ON tasks(model_id);
                CREATE INDEX IF NOT EXISTS idx_responses_task ON responses(task_id);
                CREATE INDEX IF NOT EXISTS idx_responses_evaluator ON responses(evaluator_id);
            """)

    def create_evaluator(self, profile: EvaluatorProfile):
        """Create a new evaluator profile."""
        self.conn.execute("""
            INSERT INTO evaluators (evaluator_id, name, skill_level, created_at)
            VALUES (?, ?, ?, ?)
        """, (profile.evaluator_id, profile.name, profile.skill_level,
              profile.created_at.isoformat()))
        self.conn.commit()

    def get_evaluator(self, evaluator_id: str) -> EvaluatorProfile | None:
        """Get evaluator profile."""
        cursor = self.conn.execute(
            "SELECT * FROM evaluators WHERE evaluator_id = ?", (evaluator_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return EvaluatorProfile(
            evaluator_id=row['evaluator_id'],
            name=row['name'],
            skill_level=row['skill_level'],
            evaluations_completed=row['evaluations_completed'],
            agreement_score=row['agreement_score'],
            created_at=datetime.fromisoformat(row['created_at'])
        )

    def create_task(self, task: EvaluationTask):
        """Create a new evaluation task."""
        self.conn.execute("""
            INSERT INTO tasks
            (task_id, eval_type, model_id, position_data, ai_move, alternatives,
             comparison_model_id, comparison_move, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.eval_type.value, task.model_id,
            json.dumps(task.position_data), task.ai_move,
            json.dumps(task.alternatives) if task.alternatives else None,
            task.comparison_model_id, task.comparison_move,
            task.created_at.isoformat(), json.dumps(task.metadata)
        ))
        self.conn.commit()

    def get_task(self, task_id: str) -> EvaluationTask | None:
        """Get a task by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return EvaluationTask(
            task_id=row['task_id'],
            eval_type=EvaluationType(row['eval_type']),
            model_id=row['model_id'],
            position_data=json.loads(row['position_data']),
            ai_move=row['ai_move'],
            alternatives=json.loads(row['alternatives']) if row['alternatives'] else None,
            comparison_model_id=row['comparison_model_id'],
            comparison_move=row['comparison_move'],
            created_at=datetime.fromisoformat(row['created_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    def submit_response(self, response: EvaluationResponse):
        """Submit an evaluation response."""
        self.conn.execute("""
            INSERT INTO responses
            (task_id, evaluator_id, rating, preference, annotation,
             suggested_move, confidence, time_spent_seconds, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            response.task_id, response.evaluator_id, response.rating,
            response.preference, response.annotation, response.suggested_move,
            response.confidence, response.time_spent_seconds,
            response.submitted_at.isoformat()
        ))

        # Update evaluator stats
        self.conn.execute("""
            UPDATE evaluators SET evaluations_completed = evaluations_completed + 1
            WHERE evaluator_id = ?
        """, (response.evaluator_id,))

        self.conn.commit()

    def get_responses(self, task_id: str) -> list[EvaluationResponse]:
        """Get all responses for a task."""
        cursor = self.conn.execute(
            "SELECT * FROM responses WHERE task_id = ?", (task_id,)
        )
        responses = []
        for row in cursor.fetchall():
            responses.append(EvaluationResponse(
                task_id=row['task_id'],
                evaluator_id=row['evaluator_id'],
                rating=row['rating'],
                preference=row['preference'],
                annotation=row['annotation'],
                suggested_move=row['suggested_move'],
                confidence=row['confidence'],
                time_spent_seconds=row['time_spent_seconds'],
                submitted_at=datetime.fromisoformat(row['submitted_at'])
            ))
        return responses

    def get_model_statistics(self, model_id: str) -> dict[str, Any]:
        """Get evaluation statistics for a model."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_evals,
                AVG(r.rating) as avg_rating,
                AVG(r.confidence) as avg_confidence,
                COUNT(CASE WHEN r.rating >= 3 THEN 1 END) as good_moves,
                COUNT(CASE WHEN r.rating <= 1 THEN 1 END) as poor_moves
            FROM responses r
            JOIN tasks t ON r.task_id = t.task_id
            WHERE t.model_id = ?
        """, (model_id,))

        row = cursor.fetchone()
        return {
            'total_evaluations': row['total_evals'],
            'average_rating': row['avg_rating'],
            'average_confidence': row['avg_confidence'],
            'good_move_count': row['good_moves'],
            'poor_move_count': row['poor_moves']
        }


class TaskGenerator:
    """Generates evaluation tasks from game data."""

    def __init__(self, db: EvaluationDatabase):
        self.db = db

    def create_move_quality_task(
        self,
        model_id: str,
        position_data: dict[str, Any],
        ai_move: int,
        alternatives: list[int] | None = None
    ) -> EvaluationTask:
        """Create a move quality evaluation task."""
        task = EvaluationTask(
            task_id=str(uuid.uuid4()),
            eval_type=EvaluationType.MOVE_QUALITY,
            model_id=model_id,
            position_data=position_data,
            ai_move=ai_move,
            alternatives=alternatives
        )
        self.db.create_task(task)
        return task

    def create_preference_task(
        self,
        model_a_id: str,
        model_b_id: str,
        position_data: dict[str, Any],
        move_a: int,
        move_b: int
    ) -> EvaluationTask:
        """Create an A/B preference task between two models."""
        task = EvaluationTask(
            task_id=str(uuid.uuid4()),
            eval_type=EvaluationType.PREFERENCE,
            model_id=model_a_id,
            position_data=position_data,
            ai_move=move_a,
            comparison_model_id=model_b_id,
            comparison_move=move_b
        )
        self.db.create_task(task)
        return task

    def create_annotation_task(
        self,
        model_id: str,
        position_data: dict[str, Any],
        ai_move: int | None = None
    ) -> EvaluationTask:
        """Create a position annotation task."""
        task = EvaluationTask(
            task_id=str(uuid.uuid4()),
            eval_type=EvaluationType.ANNOTATION,
            model_id=model_id,
            position_data=position_data,
            ai_move=ai_move
        )
        self.db.create_task(task)
        return task

    def create_batch_from_games(
        self,
        model_id: str,
        games: list[dict[str, Any]],
        tasks_per_game: int = 3,
        eval_type: EvaluationType = EvaluationType.MOVE_QUALITY
    ) -> list[EvaluationTask]:
        """Create evaluation tasks from a batch of games."""
        import random

        tasks = []
        for game in games:
            moves = game.get('moves', [])
            positions = game.get('positions', [])

            if len(moves) < tasks_per_game:
                continue

            # Select random positions to evaluate
            indices = random.sample(range(len(moves)), min(tasks_per_game, len(moves)))

            for idx in indices:
                position_data = {
                    'game_id': game.get('game_id'),
                    'move_number': idx,
                    'position': positions[idx] if idx < len(positions) else None,
                    'previous_moves': moves[:idx]
                }

                task = EvaluationTask(
                    task_id=str(uuid.uuid4()),
                    eval_type=eval_type,
                    model_id=model_id,
                    position_data=position_data,
                    ai_move=moves[idx]
                )
                self.db.create_task(task)
                tasks.append(task)

        return tasks


class EvaluationAnalyzer:
    """Analyzes human evaluation data."""

    def __init__(self, db: EvaluationDatabase):
        self.db = db

    def compute_inter_rater_agreement(self, task_ids: list[str]) -> float:
        """
        Compute inter-rater agreement (Fleiss' kappa).
        """
        ratings_by_task = {}
        for task_id in task_ids:
            responses = self.db.get_responses(task_id)
            if responses:
                ratings_by_task[task_id] = [r.rating for r in responses if r.rating is not None]

        if not ratings_by_task:
            return 0.0

        # Simplified agreement: proportion of tasks where raters agree
        agreed = 0
        total = 0

        for ratings in ratings_by_task.values():
            if len(ratings) >= 2:
                # Check if all ratings are the same
                if len(set(ratings)) == 1:
                    agreed += 1
                total += 1

        return agreed / total if total > 0 else 0.0

    def compute_model_preference(
        self,
        model_a_id: str,
        model_b_id: str
    ) -> dict[str, Any]:
        """Compute preference statistics between two models."""
        cursor = self.db.conn.execute("""
            SELECT r.preference, COUNT(*) as count
            FROM responses r
            JOIN tasks t ON r.task_id = t.task_id
            WHERE t.model_id = ? AND t.comparison_model_id = ?
            GROUP BY r.preference
        """, (model_a_id, model_b_id))

        counts = {'a': 0, 'b': 0, 'tie': 0}
        for row in cursor.fetchall():
            if row['preference'] in counts:
                counts[row['preference']] = row['count']

        total = sum(counts.values())
        if total == 0:
            return {'error': 'No preference data'}

        return {
            'model_a_wins': counts['a'],
            'model_b_wins': counts['b'],
            'ties': counts['tie'],
            'total': total,
            'model_a_win_rate': counts['a'] / total,
            'model_b_win_rate': counts['b'] / total,
            'tie_rate': counts['tie'] / total,
            'preferred_model': model_a_id if counts['a'] > counts['b'] else model_b_id
        }

    def get_problematic_positions(
        self,
        model_id: str,
        rating_threshold: int = 2
    ) -> list[dict[str, Any]]:
        """Get positions where the AI received poor ratings."""
        cursor = self.db.conn.execute("""
            SELECT t.task_id, t.position_data, t.ai_move, AVG(r.rating) as avg_rating,
                   GROUP_CONCAT(r.annotation) as annotations
            FROM tasks t
            JOIN responses r ON t.task_id = r.task_id
            WHERE t.model_id = ? AND r.rating <= ?
            GROUP BY t.task_id
            ORDER BY avg_rating ASC
            LIMIT 100
        """, (model_id, rating_threshold))

        problems = []
        for row in cursor.fetchall():
            problems.append({
                'task_id': row['task_id'],
                'position': json.loads(row['position_data']),
                'ai_move': row['ai_move'],
                'avg_rating': row['avg_rating'],
                'annotations': row['annotations'].split(',') if row['annotations'] else []
            })

        return problems

    def generate_training_signal(self, min_confidence: float = 0.7) -> list[dict[str, Any]]:
        """
        Generate training signals from human evaluations.

        Returns positions with suggested improvements.
        """
        cursor = self.db.conn.execute("""
            SELECT t.position_data, t.ai_move, r.suggested_move, r.rating, r.confidence
            FROM tasks t
            JOIN responses r ON t.task_id = r.task_id
            WHERE r.suggested_move IS NOT NULL
              AND r.suggested_move != t.ai_move
              AND r.confidence >= ?
            ORDER BY r.rating ASC
        """, (min_confidence,))

        signals = []
        for row in cursor.fetchall():
            signals.append({
                'position': json.loads(row['position_data']),
                'ai_move': row['ai_move'],
                'better_move': row['suggested_move'],
                'ai_rating': row['rating'],
                'confidence': row['confidence']
            })

        return signals


class HumanEvalServer:
    """
    HTTP server for human evaluation interface.
    """

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RingRift AI Evaluation</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .board { display: grid; grid-template-columns: repeat(8, 50px); gap: 2px; margin: 20px 0; }
            .cell { width: 50px; height: 50px; border: 1px solid #333; display: flex;
                    align-items: center; justify-content: center; cursor: pointer;
                    font-size: 24px; background: #f0f0f0; }
            .cell.highlight { background: #ffeb3b; }
            .cell.ai-move { background: #4caf50; color: white; }
            .rating-buttons { margin: 20px 0; }
            .rating-btn { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
            .rating-btn:hover { opacity: 0.8; }
            .brilliant { background: #4caf50; color: white; }
            .excellent { background: #8bc34a; color: white; }
            .good { background: #cddc39; }
            .acceptable { background: #ffeb3b; }
            .questionable { background: #ff9800; color: white; }
            .blunder { background: #f44336; color: white; }
            .task-info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .preference-btn { padding: 15px 30px; margin: 10px; font-size: 18px; cursor: pointer; }
            #feedback { width: 100%; height: 100px; margin: 10px 0; }
            .stats { background: #fff3e0; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>RingRift AI Evaluation</h1>
        <div id="evaluator-info"></div>
        <div id="task-container"></div>
        <div id="stats-container"></div>

        <script>
            let currentTask = null;
            let evaluatorId = localStorage.getItem('evaluatorId') || '';
            let startTime = null;

            async function loadTask() {
                const response = await fetch('/api/task?evaluator=' + evaluatorId);
                currentTask = await response.json();
                startTime = Date.now();
                renderTask();
            }

            function renderTask() {
                const container = document.getElementById('task-container');
                if (!currentTask || currentTask.error) {
                    container.innerHTML = '<p>No tasks available. Thank you for your help!</p>';
                    return;
                }

                let html = '<div class="task-info">';
                html += '<h2>Task: ' + currentTask.eval_type + '</h2>';
                html += '<p>Model: ' + currentTask.model_id + '</p>';
                html += '</div>';

                // Render board (placeholder)
                html += '<div class="board">';
                for (let i = 0; i < 64; i++) {
                    const isAiMove = currentTask.ai_move === i;
                    html += '<div class="cell' + (isAiMove ? ' ai-move' : '') + '" data-idx="' + i + '">';
                    if (isAiMove) html += '★';
                    html += '</div>';
                }
                html += '</div>';

                if (currentTask.eval_type === 'move_quality') {
                    html += renderMoveQualityButtons();
                } else if (currentTask.eval_type === 'preference') {
                    html += renderPreferenceButtons();
                } else if (currentTask.eval_type === 'annotation') {
                    html += renderAnnotationForm();
                }

                container.innerHTML = html;
            }

            function renderMoveQualityButtons() {
                return `
                    <div class="rating-buttons">
                        <p>Rate the AI's move (marked with ★):</p>
                        <button class="rating-btn brilliant" onclick="submitRating(5)">Brilliant</button>
                        <button class="rating-btn excellent" onclick="submitRating(4)">Excellent</button>
                        <button class="rating-btn good" onclick="submitRating(3)">Good</button>
                        <button class="rating-btn acceptable" onclick="submitRating(2)">Acceptable</button>
                        <button class="rating-btn questionable" onclick="submitRating(1)">Questionable</button>
                        <button class="rating-btn blunder" onclick="submitRating(0)">Blunder</button>
                    </div>
                    <div>
                        <label>Better move (click on board or enter number): </label>
                        <input type="number" id="suggested-move" min="0" max="63">
                    </div>
                `;
            }

            function renderPreferenceButtons() {
                return `
                    <div>
                        <p>Which move is better?</p>
                        <button class="preference-btn" onclick="submitPreference('a')">Move A (★)</button>
                        <button class="preference-btn" onclick="submitPreference('tie')">Equal</button>
                        <button class="preference-btn" onclick="submitPreference('b')">Move B (☆)</button>
                    </div>
                `;
            }

            function renderAnnotationForm() {
                return `
                    <div>
                        <p>Describe this position and the AI's play:</p>
                        <textarea id="feedback" placeholder="Enter your analysis..."></textarea>
                        <button onclick="submitAnnotation()">Submit</button>
                    </div>
                `;
            }

            async function submitRating(rating) {
                const timeSpent = (Date.now() - startTime) / 1000;
                const suggestedMove = document.getElementById('suggested-move')?.value;

                await fetch('/api/response', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        task_id: currentTask.task_id,
                        evaluator_id: evaluatorId,
                        rating: rating,
                        suggested_move: suggestedMove ? parseInt(suggestedMove) : null,
                        time_spent_seconds: timeSpent
                    })
                });

                loadTask();
            }

            async function submitPreference(pref) {
                const timeSpent = (Date.now() - startTime) / 1000;

                await fetch('/api/response', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        task_id: currentTask.task_id,
                        evaluator_id: evaluatorId,
                        preference: pref,
                        time_spent_seconds: timeSpent
                    })
                });

                loadTask();
            }

            async function submitAnnotation() {
                const annotation = document.getElementById('feedback').value;
                const timeSpent = (Date.now() - startTime) / 1000;

                await fetch('/api/response', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        task_id: currentTask.task_id,
                        evaluator_id: evaluatorId,
                        annotation: annotation,
                        time_spent_seconds: timeSpent
                    })
                });

                loadTask();
            }

            async function loadStats() {
                const response = await fetch('/api/stats?evaluator=' + evaluatorId);
                const stats = await response.json();

                const container = document.getElementById('stats-container');
                container.innerHTML = `
                    <div class="stats">
                        <h3>Your Statistics</h3>
                        <p>Evaluations completed: ${stats.completed || 0}</p>
                        <p>Agreement score: ${(stats.agreement * 100).toFixed(1)}%</p>
                    </div>
                `;
            }

            // Initialize
            if (!evaluatorId) {
                evaluatorId = 'eval_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('evaluatorId', evaluatorId);
            }

            document.getElementById('evaluator-info').innerHTML = `
                <p>Evaluator ID: ${evaluatorId}</p>
            `;

            loadTask();
            loadStats();
        </script>
    </body>
    </html>
    """

    def __init__(self, db: EvaluationDatabase, host: str = "0.0.0.0", port: int = 8081):
        self.db = db
        self.host = host
        self.port = port
        self.task_generator = TaskGenerator(db)
        self.pending_tasks: list[str] = []

    def create_handler(self):
        """Create request handler with access to database."""
        db = self.db
        pending_tasks = self.pending_tasks
        html_template = self.HTML_TEMPLATE

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)

                if parsed.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html_template.encode())

                elif parsed.path == '/api/task':
                    params = urllib.parse.parse_qs(parsed.query)
                    evaluator_id = params.get('evaluator', [''])[0]

                    # Get a pending task
                    task = None
                    if pending_tasks:
                        task_id = pending_tasks.pop(0)
                        task = db.get_task(task_id)

                    if task:
                        response = task.to_dict()
                    else:
                        response = {'error': 'No tasks available'}

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

                elif parsed.path == '/api/stats':
                    params = urllib.parse.parse_qs(parsed.query)
                    evaluator_id = params.get('evaluator', [''])[0]

                    evaluator = db.get_evaluator(evaluator_id)
                    if evaluator:
                        response = {
                            'completed': evaluator.evaluations_completed,
                            'agreement': evaluator.agreement_score
                        }
                    else:
                        response = {'completed': 0, 'agreement': 1.0}

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/api/response':
                    content_length = int(self.headers['Content-Length'])
                    body = self.rfile.read(content_length).decode()
                    data = json.loads(body)

                    # Create or get evaluator
                    evaluator_id = data.get('evaluator_id', '')
                    if not db.get_evaluator(evaluator_id):
                        db.create_evaluator(EvaluatorProfile(
                            evaluator_id=evaluator_id,
                            name=f"Evaluator {evaluator_id[:8]}"
                        ))

                    # Submit response
                    response = EvaluationResponse(
                        task_id=data.get('task_id', ''),
                        evaluator_id=evaluator_id,
                        rating=data.get('rating'),
                        preference=data.get('preference'),
                        annotation=data.get('annotation'),
                        suggested_move=data.get('suggested_move'),
                        confidence=data.get('confidence'),
                        time_spent_seconds=data.get('time_spent_seconds')
                    )
                    db.submit_response(response)

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': True}).encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logging

        return Handler

    def add_tasks(self, tasks: list[EvaluationTask]):
        """Add tasks to the pending queue."""
        for task in tasks:
            self.pending_tasks.append(task.task_id)

    def run(self):
        """Run the evaluation server."""
        handler = self.create_handler()
        server = HTTPServer((self.host, self.port), handler)
        logger.info(f"Human evaluation server running at http://{self.host}:{self.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()


def main():
    """Demonstrate human evaluation interface."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "eval.db"

        # Initialize database
        db = EvaluationDatabase(db_path)

        # Create task generator
        generator = TaskGenerator(db)

        # Create some sample tasks
        print("Creating evaluation tasks...")

        tasks = []
        for i in range(5):
            task = generator.create_move_quality_task(
                model_id="test_model_v1",
                position_data={'board': [[0] * 8 for _ in range(8)], 'move_number': i * 5},
                ai_move=i * 8 + 3,
                alternatives=[i * 8 + 2, i * 8 + 4]
            )
            tasks.append(task)
            print(f"  Created task: {task.task_id[:8]}...")

        # Create preference task
        pref_task = generator.create_preference_task(
            model_a_id="model_v1",
            model_b_id="model_v2",
            position_data={'board': [[0] * 8 for _ in range(8)]},
            move_a=20,
            move_b=28
        )
        tasks.append(pref_task)
        print(f"  Created preference task: {pref_task.task_id[:8]}...")

        # Create annotation task
        annot_task = generator.create_annotation_task(
            model_id="test_model_v1",
            position_data={'board': [[0] * 8 for _ in range(8)], 'note': 'Complex position'},
            ai_move=35
        )
        tasks.append(annot_task)
        print(f"  Created annotation task: {annot_task.task_id[:8]}...")

        # Simulate some responses
        print("\nSimulating evaluator responses...")

        # Create an evaluator
        evaluator = EvaluatorProfile(
            evaluator_id="test_evaluator_1",
            name="Test Evaluator",
            skill_level="advanced"
        )
        db.create_evaluator(evaluator)

        # Submit responses
        for i, task in enumerate(tasks[:3]):
            response = EvaluationResponse(
                task_id=task.task_id,
                evaluator_id=evaluator.evaluator_id,
                rating=3 + (i % 3),  # Ratings 3, 4, 5
                confidence=0.8,
                time_spent_seconds=15.5 + i
            )
            db.submit_response(response)
            print(f"  Submitted rating {response.rating} for task {task.task_id[:8]}")

        # Analyze results
        print("\n=== Analysis ===")
        EvaluationAnalyzer(db)

        stats = db.get_model_statistics("test_model_v1")
        print("Model Statistics:")
        print(f"  Total evaluations: {stats['total_evaluations']}")
        print(f"  Average rating: {stats['average_rating']:.2f}")
        print(f"  Good moves: {stats['good_move_count']}")
        print(f"  Poor moves: {stats['poor_move_count']}")

        # Show evaluator stats
        evaluator = db.get_evaluator("test_evaluator_1")
        print("\nEvaluator Stats:")
        print(f"  Name: {evaluator.name}")
        print(f"  Evaluations completed: {evaluator.evaluations_completed}")

        print("\n=== Server Demo ===")
        print("To start the evaluation server, run:")
        print("  server = HumanEvalServer(db)")
        print("  server.add_tasks(tasks)")
        print("  server.run()")
        print("\nThen open http://localhost:8081 in a browser")


if __name__ == "__main__":
    main()
