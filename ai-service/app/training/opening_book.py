"""Opening Book for Diverse Selfplay Positions.

Generates and manages diverse opening positions to prevent mode collapse
during selfplay training. Instead of always starting from the initial
position, games can start from varied positions in the opening book.

Benefits:
1. Prevents overfitting to narrow opening lines
2. Exposes the model to diverse board patterns
3. Faster exploration of position space
4. Better generalization to varied play styles

Usage:
    from app.training.opening_book import OpeningBook, OpeningGenerator

    # Generate openings
    generator = OpeningGenerator(board_type="square8", num_players=2)
    openings = generator.generate_openings(count=1000, max_moves=8)

    # Use in selfplay
    book = OpeningBook("square8", 2)
    book.add_openings(openings)
    opening = book.sample_opening()
    game_state = opening.get_game_state()
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_BOOK_DIR = "data/opening_books"


@dataclass
class Opening:
    """Represents an opening sequence of moves."""
    opening_id: str
    board_type: str
    num_players: int
    moves: list[dict[str, Any]]  # List of move dicts
    move_count: int
    # Statistics from usage
    times_used: int = 0
    avg_game_length: float = 0.0
    win_rate_p1: float = 0.5
    diversity_score: float = 0.0  # How different from other openings

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "opening_id": self.opening_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "moves": self.moves,
            "move_count": self.move_count,
            "times_used": self.times_used,
            "avg_game_length": self.avg_game_length,
            "win_rate_p1": self.win_rate_p1,
            "diversity_score": self.diversity_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Opening:
        """Create from dictionary."""
        return cls(
            opening_id=data["opening_id"],
            board_type=data["board_type"],
            num_players=data["num_players"],
            moves=data["moves"],
            move_count=data["move_count"],
            times_used=data.get("times_used", 0),
            avg_game_length=data.get("avg_game_length", 0.0),
            win_rate_p1=data.get("win_rate_p1", 0.5),
            diversity_score=data.get("diversity_score", 0.0),
        )


class OpeningGenerator:
    """Generates diverse opening sequences for selfplay.

    Uses random rollouts to explore the opening space and create
    a diverse set of starting positions.
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        seed: int | None = None,
    ):
        """Initialize the opening generator.

        Args:
            board_type: Type of board
            num_players: Number of players
            seed: Random seed for reproducibility
        """
        self.board_type = board_type
        self.num_players = num_players

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_openings(
        self,
        count: int = 1000,
        min_moves: int = 4,
        max_moves: int = 10,
        unique_positions: bool = True,
    ) -> list[Opening]:
        """Generate diverse opening sequences.

        Args:
            count: Number of openings to generate
            min_moves: Minimum moves in opening
            max_moves: Maximum moves in opening
            unique_positions: Whether to ensure unique final positions

        Returns:
            List of Opening objects
        """
        try:
            from app.models import BoardType
            from app.training.env import RingRiftEnv
        except ImportError:
            logger.error("Could not import RingRiftEnv")
            return []

        openings = []
        seen_positions: set[str] = set()

        # Convert board type
        try:
            board_type_enum = BoardType(self.board_type)
        except ValueError:
            logger.error(f"Invalid board type: {self.board_type}")
            return []

        attempts = 0
        max_attempts = count * 10

        while len(openings) < count and attempts < max_attempts:
            attempts += 1

            # Create environment
            env = RingRiftEnv(
                board_type=board_type_enum,
                num_players=self.num_players,
            )

            state = env.reset()
            moves = []
            num_moves = random.randint(min_moves, max_moves)

            for _ in range(num_moves):
                legal_moves = state.get_legal_moves()
                if not legal_moves:
                    break

                # Random move selection
                move = random.choice(legal_moves)

                # Record move
                move_dict = {
                    "player": state.current_player,
                    "from": move.from_position if hasattr(move, "from_position") else None,
                    "to": move.to_position if hasattr(move, "to_position") else None,
                    "move_type": str(type(move).__name__),
                }
                moves.append(move_dict)

                # Apply move
                state, _, done, _ = env.step(move)
                if done:
                    break

            if len(moves) < min_moves:
                continue

            # Check uniqueness
            if unique_positions:
                # Create position hash
                pos_hash = self._hash_position(state)
                if pos_hash in seen_positions:
                    continue
                seen_positions.add(pos_hash)

            # Create opening
            opening_id = f"{self.board_type}_{self.num_players}p_{len(openings):05d}"
            opening = Opening(
                opening_id=opening_id,
                board_type=self.board_type,
                num_players=self.num_players,
                moves=moves,
                move_count=len(moves),
            )
            openings.append(opening)

        logger.info(f"Generated {len(openings)} openings in {attempts} attempts")
        return openings

    def _hash_position(self, state: Any) -> str:
        """Create a hash of the board position."""
        # Simple hash based on board state
        if hasattr(state, "board") and hasattr(state.board, "tobytes"):
            return state.board.tobytes().hex()[:32]
        elif hasattr(state, "__str__"):
            return str(state)[:100]
        else:
            return str(hash(str(state)))

    def generate_systematic_openings(
        self,
        depth: int = 6,
        branch_factor: int = 3,
    ) -> list[Opening]:
        """Generate openings systematically by exploring the game tree.

        Args:
            depth: How many moves deep to explore
            branch_factor: How many moves to consider at each position

        Returns:
            List of Opening objects
        """
        try:
            from app.models import BoardType
            from app.training.env import RingRiftEnv
        except ImportError:
            logger.error("Could not import RingRiftEnv")
            return []

        openings = []

        try:
            board_type_enum = BoardType(self.board_type)
        except ValueError:
            return []

        def explore(env, state, moves_so_far, current_depth):
            if current_depth >= depth:
                # Create opening from this path
                if len(moves_so_far) >= 4:
                    opening_id = f"{self.board_type}_{self.num_players}p_sys_{len(openings):05d}"
                    opening = Opening(
                        opening_id=opening_id,
                        board_type=self.board_type,
                        num_players=self.num_players,
                        moves=moves_so_far.copy(),
                        move_count=len(moves_so_far),
                    )
                    openings.append(opening)
                return

            legal_moves = state.get_legal_moves()
            if not legal_moves:
                return

            # Sample moves to explore
            sample_size = min(branch_factor, len(legal_moves))
            selected_moves = random.sample(legal_moves, sample_size)

            for move in selected_moves:
                # Record move
                move_dict = {
                    "player": state.current_player,
                    "from": move.from_position if hasattr(move, "from_position") else None,
                    "to": move.to_position if hasattr(move, "to_position") else None,
                    "move_type": str(type(move).__name__),
                }
                new_moves = [*moves_so_far, move_dict]

                # Apply move
                new_env = RingRiftEnv(
                    board_type=board_type_enum,
                    num_players=self.num_players,
                )
                new_state = new_env.reset()

                # Replay moves
                for m in new_moves:
                    legal = new_state.get_legal_moves()
                    # Find matching move
                    matched = None
                    for lm in legal:
                        if (hasattr(lm, "to_position") and
                            lm.to_position == m.get("to")):
                            matched = lm
                            break
                    if matched:
                        new_state, _, done, _ = new_env.step(matched)
                        if done:
                            break

                explore(new_env, new_state, new_moves, current_depth + 1)

        # Start exploration
        env = RingRiftEnv(
            board_type=board_type_enum,
            num_players=self.num_players,
        )
        state = env.reset()
        explore(env, state, [], 0)

        logger.info(f"Generated {len(openings)} systematic openings")
        return openings


class OpeningBook:
    """Database of opening positions for diverse selfplay.

    Stores and manages opening sequences with usage statistics
    and sampling weights for balanced exploration.
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        book_dir: str = DEFAULT_BOOK_DIR,
    ):
        """Initialize the opening book.

        Args:
            board_type: Type of board
            num_players: Number of players
            book_dir: Directory for book storage
        """
        self.board_type = board_type
        self.num_players = num_players
        self.config_key = f"{board_type}_{num_players}p"

        # Storage
        self.book_dir = Path(book_dir)
        self.book_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.book_dir / f"openings_{self.config_key}.db"

        # In-memory cache
        self.openings: list[Opening] = []
        self.weights: np.ndarray = np.array([])

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS openings (
                opening_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                moves TEXT,
                move_count INTEGER,
                times_used INTEGER DEFAULT 0,
                avg_game_length REAL DEFAULT 0,
                win_rate_p1 REAL DEFAULT 0.5,
                diversity_score REAL DEFAULT 0,
                created_at REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_config
            ON openings(board_type, num_players)
        """)
        conn.commit()
        conn.close()

        # Load into memory
        self._load_from_db()

    def _load_from_db(self):
        """Load openings from database into memory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT * FROM openings
            WHERE board_type = ? AND num_players = ?
            ORDER BY diversity_score DESC
        """, (self.board_type, self.num_players))

        self.openings = []
        for row in cursor:
            opening = Opening(
                opening_id=row["opening_id"],
                board_type=row["board_type"],
                num_players=row["num_players"],
                moves=json.loads(row["moves"]),
                move_count=row["move_count"],
                times_used=row["times_used"],
                avg_game_length=row["avg_game_length"],
                win_rate_p1=row["win_rate_p1"],
                diversity_score=row["diversity_score"],
            )
            self.openings.append(opening)

        conn.close()

        # Update weights
        self._update_weights()

        logger.info(f"Loaded {len(self.openings)} openings for {self.config_key}")

    def _update_weights(self):
        """Update sampling weights based on usage and diversity."""
        if not self.openings:
            self.weights = np.array([])
            return

        # Weight by diversity and inversely by usage
        diversity_scores = np.array([o.diversity_score + 0.1 for o in self.openings])
        usage_penalty = np.array([1.0 / (1 + o.times_used * 0.1) for o in self.openings])

        weights = diversity_scores * usage_penalty
        self.weights = weights / weights.sum()

    def add_openings(self, openings: list[Opening]):
        """Add openings to the book.

        Args:
            openings: List of Opening objects
        """
        conn = sqlite3.connect(str(self.db_path))
        timestamp = time.time()

        for opening in openings:
            # Compute diversity score
            diversity = self._compute_diversity(opening)
            opening.diversity_score = diversity

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO openings
                    (opening_id, board_type, num_players, moves, move_count,
                     times_used, avg_game_length, win_rate_p1, diversity_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    opening.opening_id,
                    opening.board_type,
                    opening.num_players,
                    json.dumps(opening.moves),
                    opening.move_count,
                    opening.times_used,
                    opening.avg_game_length,
                    opening.win_rate_p1,
                    opening.diversity_score,
                    timestamp,
                ))
            except Exception as e:
                logger.warning(f"Failed to add opening {opening.opening_id}: {e}")

        conn.commit()
        conn.close()

        # Reload
        self._load_from_db()

        logger.info(f"Added {len(openings)} openings to book")

    def _compute_diversity(self, opening: Opening) -> float:
        """Compute diversity score for an opening.

        Higher score = more different from existing openings.
        """
        if not self.openings:
            return 1.0

        # Simple diversity: check how many existing openings share prefix
        prefix_len = min(4, opening.move_count)
        prefix = json.dumps(opening.moves[:prefix_len])

        matches = 0
        for existing in self.openings:
            existing_prefix = json.dumps(existing.moves[:prefix_len])
            if prefix == existing_prefix:
                matches += 1

        # Fewer matches = higher diversity
        diversity = 1.0 / (1 + matches * 0.5)
        return diversity

    def sample_opening(self) -> Opening | None:
        """Sample an opening weighted by diversity and usage.

        Returns:
            Selected Opening or None if book is empty
        """
        if not self.openings:
            return None

        idx = np.random.choice(len(self.openings), p=self.weights)
        opening = self.openings[idx]

        # Update usage count
        opening.times_used += 1
        self._update_usage_in_db(opening.opening_id)

        return opening

    def _update_usage_in_db(self, opening_id: str):
        """Update usage count in database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            UPDATE openings SET times_used = times_used + 1
            WHERE opening_id = ?
        """, (opening_id,))
        conn.commit()
        conn.close()

    def update_stats(
        self,
        opening_id: str,
        game_length: int,
        p1_won: bool,
    ):
        """Update statistics after a game using this opening.

        Args:
            opening_id: Opening that was used
            game_length: Length of the completed game
            p1_won: Whether player 1 won
        """
        # Find opening
        opening = None
        for o in self.openings:
            if o.opening_id == opening_id:
                opening = o
                break

        if not opening:
            return

        # Update rolling average game length
        n = opening.times_used
        old_avg = opening.avg_game_length
        opening.avg_game_length = (old_avg * (n - 1) + game_length) / n

        # Update win rate
        old_wr = opening.win_rate_p1
        opening.win_rate_p1 = (old_wr * (n - 1) + (1 if p1_won else 0)) / n

        # Update in database
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            UPDATE openings
            SET avg_game_length = ?, win_rate_p1 = ?
            WHERE opening_id = ?
        """, (opening.avg_game_length, opening.win_rate_p1, opening_id))
        conn.commit()
        conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the opening book."""
        if not self.openings:
            return {"count": 0}

        return {
            "count": len(self.openings),
            "total_uses": sum(o.times_used for o in self.openings),
            "avg_moves": np.mean([o.move_count for o in self.openings]),
            "avg_diversity": np.mean([o.diversity_score for o in self.openings]),
            "balanced_openings": len([o for o in self.openings if 0.4 < o.win_rate_p1 < 0.6]),
        }

    def get_unused_openings(self, limit: int = 100) -> list[Opening]:
        """Get openings that haven't been used yet."""
        unused = [o for o in self.openings if o.times_used == 0]
        return unused[:limit]


def generate_default_book(
    board_type: str = "square8",
    num_players: int = 2,
    count: int = 1000,
) -> OpeningBook:
    """Generate a default opening book for a configuration.

    Args:
        board_type: Board type
        num_players: Number of players
        count: Number of openings to generate

    Returns:
        Populated OpeningBook
    """
    logger.info(f"Generating {count} openings for {board_type}_{num_players}p")

    generator = OpeningGenerator(board_type, num_players)
    openings = generator.generate_openings(count=count)

    book = OpeningBook(board_type, num_players)
    book.add_openings(openings)

    return book


def get_opening_book(
    board_type: str = "square8",
    num_players: int = 2,
    auto_generate: bool = True,
    min_openings: int = 100,
) -> OpeningBook:
    """Get or create an opening book for a configuration.

    Args:
        board_type: Board type
        num_players: Number of players
        auto_generate: Whether to generate openings if book is empty
        min_openings: Minimum openings to generate

    Returns:
        OpeningBook instance
    """
    book = OpeningBook(board_type, num_players)

    if auto_generate and len(book.openings) < min_openings:
        logger.info(f"Auto-generating {min_openings} openings for {board_type}_{num_players}p")
        generator = OpeningGenerator(board_type, num_players)
        openings = generator.generate_openings(count=min_openings)
        book.add_openings(openings)

    return book
