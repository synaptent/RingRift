"""Move Explanation System for RingRift AI.

Generates human-readable explanations for AI move decisions.
Helps players understand AI reasoning and supports debugging.

Features:
- Tactical explanations (threats, captures, defense)
- Strategic explanations (territory, influence, development)
- Confidence analysis
- Alternative move suggestions
- Heat map visualizations of influence

Usage:
    from app.ai.move_explanation import MoveExplainer

    explainer = MoveExplainer(model_path="models/best.pt")

    # Get explanation for a move
    explanation = explainer.explain_move(
        board_state=state,
        move=selected_move,
        player=current_player,
    )

    print(explanation.summary)
    print(explanation.tactical_reasons)
    print(explanation.strategic_reasons)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MoveCategory(Enum):
    """Categories of move motivations."""
    OFFENSIVE = "offensive"       # Attacking opponent
    DEFENSIVE = "defensive"       # Protecting own position
    TERRITORIAL = "territorial"   # Controlling territory
    DEVELOPMENT = "development"   # Developing position
    TACTICAL = "tactical"         # Specific tactical pattern
    STRATEGIC = "strategic"       # Long-term planning
    FORCED = "forced"            # Only legal move
    EXPLORATORY = "exploratory"  # Exploration/uncertainty


class ThreatType(Enum):
    """Types of tactical threats."""
    LINE_THREAT = "line_threat"         # Creating a line
    DOUBLE_THREAT = "double_threat"     # Multiple threats
    BLOCK = "block"                     # Blocking opponent
    CONNECT = "connect"                 # Connecting pieces
    CUT = "cut"                         # Cutting opponent
    CAPTURE = "capture"                 # Capturing opponent
    TERRITORY = "territory"             # Territory capture


@dataclass
class TacticalFactor:
    """A tactical consideration in move evaluation."""
    threat_type: ThreatType
    importance: float  # 0-1 scale
    description: str
    affected_squares: List[Tuple[int, int]]
    player_benefit: int  # Which player benefits (0, 1, ...)


@dataclass
class StrategicFactor:
    """A strategic consideration in move evaluation."""
    name: str
    importance: float  # 0-1 scale
    description: str
    value_contribution: float  # How much this affects value estimate


@dataclass
class MoveExplanation:
    """Complete explanation of a move decision."""
    move: Tuple[int, int]
    player: int
    confidence: float  # 0-1

    # Primary summary
    summary: str

    # Category and reasons
    primary_category: MoveCategory
    tactical_factors: List[TacticalFactor]
    strategic_factors: List[StrategicFactor]

    # Policy analysis
    move_probability: float
    rank_in_policy: int  # 1 = best, 2 = second best, etc.
    alternatives: List[Tuple[Tuple[int, int], float, str]]  # (move, prob, reason)

    # Value analysis
    value_estimate: float  # -1 to 1 (win probability for player)
    value_delta: float  # Change from previous position

    # Uncertainty
    uncertainty: float
    is_forced: bool

    # Optional visualizations
    influence_map: Optional[np.ndarray] = None
    threat_map: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return self.summary


@dataclass
class InfluenceMap:
    """Spatial influence analysis of the board."""
    player_influence: np.ndarray  # Per-player influence maps
    total_influence: np.ndarray   # Combined influence
    contested_areas: List[Tuple[int, int]]  # Highly contested squares
    controlled_areas: Dict[int, List[Tuple[int, int]]]  # Player -> controlled squares


class BoardAnalyzer:
    """Analyzes board state for explanation generation."""

    def __init__(self, board_type: str = "square8"):
        self.board_type = board_type
        if board_type == "square8":
            self.size = 8
        elif board_type == "square19":
            self.size = 19
        elif board_type == "hexagonal":
            self.size = 11
        else:
            self.size = 8

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get neighboring positions."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    neighbors.append((nr, nc))
        return neighbors

    def count_lines(
        self,
        board: np.ndarray,
        player: int,
        min_length: int = 2,
    ) -> List[Dict[str, Any]]:
        """Count lines of consecutive pieces for a player."""
        lines = []

        # Horizontal lines
        for r in range(self.size):
            count = 0
            start_c = 0
            for c in range(self.size):
                if board[r, c] == player + 1:
                    if count == 0:
                        start_c = c
                    count += 1
                else:
                    if count >= min_length:
                        lines.append({
                            "direction": "horizontal",
                            "start": (r, start_c),
                            "end": (r, start_c + count - 1),
                            "length": count,
                        })
                    count = 0
            if count >= min_length:
                lines.append({
                    "direction": "horizontal",
                    "start": (r, start_c),
                    "end": (r, start_c + count - 1),
                    "length": count,
                })

        # Vertical lines
        for c in range(self.size):
            count = 0
            start_r = 0
            for r in range(self.size):
                if board[r, c] == player + 1:
                    if count == 0:
                        start_r = r
                    count += 1
                else:
                    if count >= min_length:
                        lines.append({
                            "direction": "vertical",
                            "start": (start_r, c),
                            "end": (start_r + count - 1, c),
                            "length": count,
                        })
                    count = 0
            if count >= min_length:
                lines.append({
                    "direction": "vertical",
                    "start": (start_r, c),
                    "end": (start_r + count - 1, c),
                    "length": count,
                })

        # Diagonal lines (both directions)
        for direction in ["diagonal_down", "diagonal_up"]:
            for start_r in range(self.size):
                for start_c in range(self.size):
                    count = 0
                    r, c = start_r, start_c
                    positions = []

                    while 0 <= r < self.size and 0 <= c < self.size:
                        if board[r, c] == player + 1:
                            count += 1
                            positions.append((r, c))
                        else:
                            if count >= min_length:
                                lines.append({
                                    "direction": direction,
                                    "start": positions[0],
                                    "end": positions[-1],
                                    "length": count,
                                })
                            count = 0
                            positions = []

                        if direction == "diagonal_down":
                            r += 1
                            c += 1
                        else:
                            r += 1
                            c -= 1

                    if count >= min_length:
                        lines.append({
                            "direction": direction,
                            "start": positions[0],
                            "end": positions[-1],
                            "length": count,
                        })

        return lines

    def find_threats(
        self,
        board: np.ndarray,
        player: int,
    ) -> List[TacticalFactor]:
        """Find tactical threats for a player."""
        threats = []

        # Find lines that could become winning
        lines = self.count_lines(board, player, min_length=2)

        for line in lines:
            length = line["length"]
            start = line["start"]
            end = line["end"]

            # Check if line can be extended
            importance = min(length / 5, 1.0)  # Assuming 5 in a row wins

            if length >= 3:
                threats.append(TacticalFactor(
                    threat_type=ThreatType.LINE_THREAT,
                    importance=importance,
                    description=f"Line of {length} in {line['direction']} direction",
                    affected_squares=[start, end],
                    player_benefit=player,
                ))

        return threats

    def calculate_influence(
        self,
        board: np.ndarray,
        num_players: int = 2,
    ) -> InfluenceMap:
        """Calculate spatial influence for each player."""
        influence = np.zeros((num_players, self.size, self.size))

        # For each piece, add influence to surrounding squares
        for r in range(self.size):
            for c in range(self.size):
                piece = board[r, c]
                if piece > 0:
                    player = piece - 1
                    # Direct influence
                    influence[player, r, c] += 1.0

                    # Decay with distance
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.size and 0 <= nc < self.size:
                                dist = max(abs(dr), abs(dc))
                                if dist > 0:
                                    influence[player, nr, nc] += 1.0 / (dist + 1)

        # Normalize
        max_val = influence.max()
        if max_val > 0:
            influence /= max_val

        # Calculate total influence (who controls each square)
        total = np.sum(influence, axis=0)

        # Find contested areas
        contested = []
        controlled = {p: [] for p in range(num_players)}

        for r in range(self.size):
            for c in range(self.size):
                if total[r, c] > 0:
                    player_influences = influence[:, r, c]
                    max_player = np.argmax(player_influences)
                    max_influence = player_influences[max_player]
                    second_influence = sorted(player_influences)[-2] if num_players > 1 else 0

                    if max_influence > 0 and second_influence / max_influence > 0.7:
                        contested.append((r, c))
                    elif max_influence > 0.3:
                        controlled[max_player].append((r, c))

        return InfluenceMap(
            player_influence=influence,
            total_influence=total,
            contested_areas=contested,
            controlled_areas=controlled,
        )


class MoveExplainer:
    """Generates explanations for AI moves."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        board_type: str = "square8",
        device: str = "cpu",
    ):
        """Initialize explainer.

        Args:
            model_path: Path to model for policy/value analysis
            board_type: Board type
            device: Device for inference
        """
        self.model_path = model_path
        self.board_type = board_type
        self.device = device

        self.analyzer = BoardAnalyzer(board_type)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model for inference."""
        if not self.model_path:
            return

        try:
            import torch
            if self.model_path.exists():
                self.model = torch.load(self.model_path, map_location=self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
                logger.info(f"Loaded model from {self.model_path}")
        except ImportError:
            logger.warning("PyTorch not available")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def _get_model_output(
        self,
        board: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Get policy and value from model."""
        if self.model is None:
            # Return uniform policy and neutral value
            size = board.shape[0]
            return np.ones(size * size) / (size * size), 0.5

        try:
            import torch
            import torch.nn.functional as F

            # Convert to features
            features = self._board_to_features(board)
            input_tensor = torch.tensor(features, dtype=torch.float32)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, tuple):
                    policy_logits, value = output
                else:
                    policy_logits = output
                    value = torch.tensor([0.5])

                policy = F.softmax(policy_logits, dim=-1)
                return (
                    policy.cpu().numpy().flatten(),
                    float(value.cpu().numpy().flatten()[0])
                )

        except Exception as e:
            logger.error(f"Error getting model output: {e}")
            size = board.shape[0]
            return np.ones(size * size) / (size * size), 0.5

    def _board_to_features(self, board: np.ndarray) -> np.ndarray:
        """Convert board to model input features."""
        size = board.shape[0]
        features = np.zeros((4, size, size), dtype=np.float32)

        features[0] = (board == 0).astype(np.float32)
        features[1] = (board == 1).astype(np.float32)
        features[2] = (board == 2).astype(np.float32)
        features[3] = (board == 3).astype(np.float32)

        return features

    def _move_to_index(self, move: Tuple[int, int], board_size: int) -> int:
        """Convert move coordinates to policy index."""
        return move[0] * board_size + move[1]

    def _index_to_move(self, index: int, board_size: int) -> Tuple[int, int]:
        """Convert policy index to move coordinates."""
        return (index // board_size, index % board_size)

    def explain_move(
        self,
        board: np.ndarray,
        move: Tuple[int, int],
        player: int,
        previous_board: Optional[np.ndarray] = None,
    ) -> MoveExplanation:
        """Generate explanation for a move.

        Args:
            board: Current board state AFTER the move
            move: The move that was made (row, col)
            player: Player who made the move
            previous_board: Board state before the move (optional)

        Returns:
            MoveExplanation with detailed analysis
        """
        board_size = board.shape[0]

        # Get board before move if not provided
        if previous_board is None:
            previous_board = board.copy()
            previous_board[move] = 0

        # Get model policy and value
        policy, value = self._get_model_output(previous_board)
        move_index = self._move_to_index(move, board_size)
        move_prob = float(policy[move_index])

        # Calculate move rank
        sorted_indices = np.argsort(policy)[::-1]
        rank = int(np.where(sorted_indices == move_index)[0][0]) + 1

        # Find alternative moves
        alternatives = []
        for i, idx in enumerate(sorted_indices[:5]):
            if idx != move_index:
                alt_move = self._index_to_move(idx, board_size)
                if previous_board[alt_move] == 0:  # Legal move
                    alternatives.append((
                        alt_move,
                        float(policy[idx]),
                        self._describe_move_briefly(alt_move, previous_board, player)
                    ))
        alternatives = alternatives[:3]

        # Analyze tactical factors
        tactical_factors = self._analyze_tactical_factors(
            board, previous_board, move, player
        )

        # Analyze strategic factors
        strategic_factors = self._analyze_strategic_factors(
            board, previous_board, move, player
        )

        # Determine primary category
        primary_category = self._determine_category(
            tactical_factors, strategic_factors, move_prob
        )

        # Calculate value delta
        _, previous_value = self._get_model_output(previous_board)
        value_delta = value - previous_value

        # Calculate uncertainty (using entropy)
        entropy = -np.sum(policy * np.log(policy + 1e-10))
        max_entropy = np.log(len(policy))
        uncertainty = entropy / max_entropy

        # Check if forced move
        legal_moves = np.where(previous_board.flatten() == 0)[0]
        is_forced = len(legal_moves) <= 1

        # Calculate influence maps
        influence_map = self.analyzer.calculate_influence(board)

        # Generate summary
        summary = self._generate_summary(
            move, player, primary_category, tactical_factors,
            strategic_factors, move_prob, rank, value, value_delta
        )

        return MoveExplanation(
            move=move,
            player=player,
            confidence=move_prob,
            summary=summary,
            primary_category=primary_category,
            tactical_factors=tactical_factors,
            strategic_factors=strategic_factors,
            move_probability=move_prob,
            rank_in_policy=rank,
            alternatives=alternatives,
            value_estimate=value,
            value_delta=value_delta,
            uncertainty=uncertainty,
            is_forced=is_forced,
            influence_map=influence_map.player_influence,
            threat_map=None,
        )

    def _analyze_tactical_factors(
        self,
        board: np.ndarray,
        previous_board: np.ndarray,
        move: Tuple[int, int],
        player: int,
    ) -> List[TacticalFactor]:
        """Analyze tactical considerations for the move."""
        factors = []

        # Check for line extension
        lines_before = self.analyzer.count_lines(previous_board, player)
        lines_after = self.analyzer.count_lines(board, player)

        # Did we extend a line?
        max_before = max([l["length"] for l in lines_before], default=0)
        max_after = max([l["length"] for l in lines_after], default=0)

        if max_after > max_before:
            factors.append(TacticalFactor(
                threat_type=ThreatType.LINE_THREAT,
                importance=min(max_after / 5, 1.0),
                description=f"Extended line to {max_after} pieces",
                affected_squares=[move],
                player_benefit=player,
            ))

        # Check for blocking opponent
        opponent = 1 - player
        opp_lines_before = self.analyzer.count_lines(previous_board, opponent)
        opp_max_before = max([l["length"] for l in opp_lines_before], default=0)

        # Check neighbors for blocking
        neighbors = self.analyzer.get_neighbors(move[0], move[1])
        opp_neighbors = sum(1 for n in neighbors if previous_board[n] == opponent + 1)

        if opp_neighbors >= 2 and opp_max_before >= 2:
            factors.append(TacticalFactor(
                threat_type=ThreatType.BLOCK,
                importance=min(opp_max_before / 4, 1.0),
                description=f"Blocked opponent's line development",
                affected_squares=[move] + [(n[0], n[1]) for n in neighbors[:3]],
                player_benefit=player,
            ))

        # Check for connection
        own_neighbors = sum(1 for n in neighbors if previous_board[n] == player + 1)
        if own_neighbors >= 2:
            factors.append(TacticalFactor(
                threat_type=ThreatType.CONNECT,
                importance=min(own_neighbors / 3, 1.0),
                description=f"Connected {own_neighbors} groups",
                affected_squares=[move],
                player_benefit=player,
            ))

        return factors

    def _analyze_strategic_factors(
        self,
        board: np.ndarray,
        previous_board: np.ndarray,
        move: Tuple[int, int],
        player: int,
    ) -> List[StrategicFactor]:
        """Analyze strategic considerations for the move."""
        factors = []

        board_size = board.shape[0]
        center = board_size // 2

        # Central control
        dist_to_center = max(abs(move[0] - center), abs(move[1] - center))
        if dist_to_center <= board_size // 4:
            factors.append(StrategicFactor(
                name="Central Control",
                importance=1.0 - dist_to_center / (board_size // 2),
                description="Occupying central territory",
                value_contribution=0.1 * (1.0 - dist_to_center / center),
            ))

        # Territory expansion
        influence_before = self.analyzer.calculate_influence(previous_board)
        influence_after = self.analyzer.calculate_influence(board)

        own_territory_before = len(influence_before.controlled_areas.get(player, []))
        own_territory_after = len(influence_after.controlled_areas.get(player, []))

        if own_territory_after > own_territory_before:
            factors.append(StrategicFactor(
                name="Territory Expansion",
                importance=min((own_territory_after - own_territory_before) / 10, 1.0),
                description=f"Gained {own_territory_after - own_territory_before} squares of influence",
                value_contribution=0.05 * (own_territory_after - own_territory_before),
            ))

        # Contest opponent territory
        contested = len(influence_after.contested_areas)
        if contested > 0:
            factors.append(StrategicFactor(
                name="Contested Areas",
                importance=min(contested / 20, 1.0),
                description=f"Contesting {contested} squares",
                value_contribution=0.03 * contested,
            ))

        # Development (early game)
        total_pieces = np.sum(board > 0)
        if total_pieces < board_size * board_size // 3:
            # Early game - value spreading out
            neighbors = self.analyzer.get_neighbors(move[0], move[1])
            own_neighbors = sum(1 for n in neighbors if board[n] == player + 1)
            if own_neighbors <= 1:
                factors.append(StrategicFactor(
                    name="Development",
                    importance=0.6,
                    description="Developing to new area",
                    value_contribution=0.08,
                ))

        return factors

    def _determine_category(
        self,
        tactical_factors: List[TacticalFactor],
        strategic_factors: List[StrategicFactor],
        move_prob: float,
    ) -> MoveCategory:
        """Determine the primary category of the move."""
        # High probability, clear tactical reason
        if move_prob > 0.5 and tactical_factors:
            max_tactical = max(tactical_factors, key=lambda f: f.importance)
            if max_tactical.threat_type == ThreatType.LINE_THREAT:
                return MoveCategory.OFFENSIVE
            elif max_tactical.threat_type == ThreatType.BLOCK:
                return MoveCategory.DEFENSIVE

        # Check strategic factors
        if strategic_factors:
            max_strategic = max(strategic_factors, key=lambda f: f.importance)
            if "Territory" in max_strategic.name:
                return MoveCategory.TERRITORIAL
            elif "Development" in max_strategic.name:
                return MoveCategory.DEVELOPMENT
            elif "Central" in max_strategic.name:
                return MoveCategory.STRATEGIC

        # Low probability move
        if move_prob < 0.1:
            return MoveCategory.EXPLORATORY

        # Default
        return MoveCategory.TACTICAL

    def _describe_move_briefly(
        self,
        move: Tuple[int, int],
        board: np.ndarray,
        player: int,
    ) -> str:
        """Generate brief description of a move."""
        neighbors = self.analyzer.get_neighbors(move[0], move[1])
        own_neighbors = sum(1 for n in neighbors if board[n] == player + 1)
        opp_neighbors = sum(1 for n in neighbors if board[n] != 0 and board[n] != player + 1)

        if own_neighbors >= 2:
            return "Connecting pieces"
        elif opp_neighbors >= 2:
            return "Blocking opponent"
        else:
            board_size = board.shape[0]
            center = board_size // 2
            if abs(move[0] - center) <= 2 and abs(move[1] - center) <= 2:
                return "Central position"
            return "Developing"

    def _generate_summary(
        self,
        move: Tuple[int, int],
        player: int,
        category: MoveCategory,
        tactical_factors: List[TacticalFactor],
        strategic_factors: List[StrategicFactor],
        move_prob: float,
        rank: int,
        value: float,
        value_delta: float,
    ) -> str:
        """Generate human-readable summary of the move."""
        # Start with move description
        pos_str = f"({move[0]}, {move[1]})"

        # Category description
        category_desc = {
            MoveCategory.OFFENSIVE: "an attacking move",
            MoveCategory.DEFENSIVE: "a defensive move",
            MoveCategory.TERRITORIAL: "a territorial move",
            MoveCategory.DEVELOPMENT: "a developing move",
            MoveCategory.TACTICAL: "a tactical move",
            MoveCategory.STRATEGIC: "a strategic move",
            MoveCategory.FORCED: "the only move",
            MoveCategory.EXPLORATORY: "an exploratory move",
        }

        summary = f"Move to {pos_str} is {category_desc[category]}"

        # Add confidence
        if move_prob > 0.7:
            summary += f" (very confident, {move_prob:.0%})"
        elif move_prob > 0.3:
            summary += f" (confident, {move_prob:.0%})"
        else:
            summary += f" (uncertain, {move_prob:.0%})"

        # Add primary reason
        if tactical_factors:
            top_tactical = max(tactical_factors, key=lambda f: f.importance)
            summary += f". {top_tactical.description}"

        if strategic_factors:
            top_strategic = max(strategic_factors, key=lambda f: f.importance)
            if not tactical_factors or top_strategic.importance > 0.5:
                summary += f". {top_strategic.description}"

        # Add value assessment
        if value > 0.7:
            summary += ". Position looks winning."
        elif value > 0.55:
            summary += ". Position is favorable."
        elif value < 0.3:
            summary += ". Position is difficult."
        elif value < 0.45:
            summary += ". Position is challenging."

        if value_delta > 0.1:
            summary += f" (improved by {value_delta:.0%})"
        elif value_delta < -0.1:
            summary += f" (worsened by {abs(value_delta):.0%})"

        return summary

    def explain_game(
        self,
        move_history: List[Dict[str, Any]],
        board_type: str = "square8",
        num_players: int = 2,
    ) -> List[MoveExplanation]:
        """Generate explanations for all moves in a game.

        Args:
            move_history: List of moves with position and player
            board_type: Board type
            num_players: Number of players

        Returns:
            List of explanations for each move
        """
        if board_type == "square8":
            size = 8
        elif board_type == "square19":
            size = 19
        else:
            size = 8

        explanations = []
        board = np.zeros((size, size), dtype=np.int8)

        for move_data in move_history:
            if isinstance(move_data, dict):
                pos = move_data.get("position") or move_data.get("pos")
                player = move_data.get("player", 0)
            else:
                pos = move_data[:2]
                player = move_data[2] if len(move_data) > 2 else 0

            if pos is None:
                continue

            move = (int(pos[0]), int(pos[1]))
            previous_board = board.copy()

            # Make the move
            board[move] = player + 1

            # Generate explanation
            explanation = self.explain_move(
                board=board,
                move=move,
                player=player,
                previous_board=previous_board,
            )
            explanations.append(explanation)

        return explanations
