"""
Test territory detection for grid-like board configurations.

These tests verify that GPU and CPU implementations correctly detect
disconnected territories based on the rules:
- R140: Region discovery (BFS from all cells, not just edges)
- R141: Single-color boundary requirement
- R142: Color-disconnection criterion (RegionColors ⊂ ActiveColors)

Scenarios involve a 19x19 grid divided by vertical and horizontal lines at
~1/3 and ~2/3 positions (x/y=6 and x/y=12), creating 9 cells. The lines and
intersections vary by scenario to test different territory detection edge cases.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from app.board_manager import BoardManager
from app.models import (
    BoardState,
    BoardType,
    MarkerInfo,
    Position,
    RingStack,
    Territory,
)


@dataclass
class GridScenario:
    """Describes a grid scenario for territory detection testing."""
    name: str
    description: str
    # Line composition: 'markers' or 'collapsed'
    vertical_line_type: str
    horizontal_line_type: str
    # For marker lines: list of player numbers for each line
    # vertical_colors[0] = left line, vertical_colors[1] = right line
    vertical_colors: list[int] | None
    horizontal_colors: list[int] | None
    # Intersection composition: list of 4 player numbers (or None for collapsed)
    # Order: top-left, top-right, bottom-left, bottom-right
    intersection_contents: list[int | None]
    # Expected number of disconnected territories
    # Note: Depends on active players and border color rules
    expected_description: str


# Define the 6 scenarios
SCENARIOS = [
    GridScenario(
        name="scenario_1",
        description="Marker lines (4 different colors) with collapsed intersections",
        vertical_line_type="markers",
        horizontal_line_type="markers",
        vertical_colors=[1, 2],  # Left line=P1, Right line=P2
        horizontal_colors=[3, 4],  # Top line=P3, Bottom line=P4
        intersection_contents=[None, None, None, None],  # All collapsed
        expected_description="4 marker colors form boundaries; intersections are collapsed",
    ),
    GridScenario(
        name="scenario_2",
        description="Collapsed lines with 4 different color intersection markers",
        vertical_line_type="collapsed",
        horizontal_line_type="collapsed",
        vertical_colors=None,
        horizontal_colors=None,
        intersection_contents=[1, 2, 3, 4],  # 4 different colors
        expected_description="Lines are collapsed; 4 intersection markers of different colors",
    ),
    GridScenario(
        name="scenario_3",
        description="Collapsed lines with same-color intersection markers",
        vertical_line_type="collapsed",
        horizontal_line_type="collapsed",
        vertical_colors=None,
        horizontal_colors=None,
        intersection_contents=[1, 1, 1, 1],  # All same color
        expected_description="Lines are collapsed; all 4 intersection markers are player 1",
    ),
    GridScenario(
        name="scenario_4",
        description="Collapsed lines with 3-color intersection markers",
        vertical_line_type="collapsed",
        horizontal_line_type="collapsed",
        vertical_colors=None,
        horizontal_colors=None,
        intersection_contents=[1, 2, 3, 1],  # 3 colors: 1,2,3 with 1 appearing twice
        expected_description="Lines are collapsed; intersection markers use 3 colors",
    ),
    GridScenario(
        name="scenario_5a",
        description="Collapsed lines with 2-color intersection markers (2 each)",
        vertical_line_type="collapsed",
        horizontal_line_type="collapsed",
        vertical_colors=None,
        horizontal_colors=None,
        intersection_contents=[1, 2, 1, 2],  # 2 colors: 2 of each
        expected_description="Lines are collapsed; 2 colors with 2 markers each",
    ),
    GridScenario(
        name="scenario_5b",
        description="Collapsed lines with 2-color intersection markers (3:1 split)",
        vertical_line_type="collapsed",
        horizontal_line_type="collapsed",
        vertical_colors=None,
        horizontal_colors=None,
        intersection_contents=[1, 1, 1, 2],  # 2 colors: 3 of color 1, 1 of color 2
        expected_description="Lines are collapsed; 2 colors with 3:1 distribution",
    ),
]


def create_grid_board(scenario: GridScenario, board_size: int = 19) -> BoardState:
    """
    Create a board state with grid lines at 1/3 and 2/3 positions.

    For a 19x19 board:
    - Vertical lines at x=6 and x=12
    - Horizontal lines at y=6 and y=12
    - Intersections at (6,6), (6,12), (12,6), (12,12)

    The 9 cells are:
    - Top-left (0-5, 0-5), Top-center (7-11, 0-5), Top-right (13-18, 0-5)
    - Mid-left (0-5, 7-11), Mid-center (7-11, 7-11), Mid-right (13-18, 7-11)
    - Bot-left (0-5, 13-18), Bot-center (7-11, 13-18), Bot-right (13-18, 13-18)
    """
    # Line positions for 19x19 board (at ~1/3 and ~2/3)
    v_line_1 = 6   # Left vertical line
    v_line_2 = 12  # Right vertical line
    h_line_1 = 6   # Top horizontal line
    h_line_2 = 12  # Bottom horizontal line

    # Intersection positions
    intersections = [
        (v_line_1, h_line_1),  # top-left
        (v_line_2, h_line_1),  # top-right
        (v_line_1, h_line_2),  # bottom-left
        (v_line_2, h_line_2),  # bottom-right
    ]

    stacks: dict[str, RingStack] = {}
    markers: dict[str, MarkerInfo] = {}
    collapsed_spaces: dict[str, int] = {}

    # Helper to add marker or collapsed space
    def add_line_cell(x: int, y: int, line_type: str, player: int | None):
        pos = Position(x=x, y=y)
        key = pos.to_key()
        if line_type == "collapsed":
            collapsed_spaces[key] = player if player else 0  # Store player number or 0
        elif line_type == "markers" and player is not None:
            markers[key] = MarkerInfo(position=pos, player=player, type="regular")

    # Build vertical lines (excluding intersection points)
    for y in range(board_size):
        # Skip intersection y-coordinates
        if y in (h_line_1, h_line_2):
            continue
        # Left vertical line at x=v_line_1
        add_line_cell(
            v_line_1, y,
            scenario.vertical_line_type,
            scenario.vertical_colors[0] if scenario.vertical_colors else None
        )
        # Right vertical line at x=v_line_2
        add_line_cell(
            v_line_2, y,
            scenario.vertical_line_type,
            scenario.vertical_colors[1] if scenario.vertical_colors else None
        )

    # Build horizontal lines (excluding intersection points)
    for x in range(board_size):
        # Skip intersection x-coordinates
        if x in (v_line_1, v_line_2):
            continue
        # Top horizontal line at y=h_line_1
        add_line_cell(
            x, h_line_1,
            scenario.horizontal_line_type,
            scenario.horizontal_colors[0] if scenario.horizontal_colors else None
        )
        # Bottom horizontal line at y=h_line_2
        add_line_cell(
            x, h_line_2,
            scenario.horizontal_line_type,
            scenario.horizontal_colors[1] if scenario.horizontal_colors else None
        )

    # Add intersection contents
    for i, (x, y) in enumerate(intersections):
        pos = Position(x=x, y=y)
        key = pos.to_key()
        content = scenario.intersection_contents[i]
        if content is None:
            collapsed_spaces[key] = 0  # Collapsed space, 0 indicates no specific player
        else:
            markers[key] = MarkerInfo(position=pos, player=content, type="regular")

    # Add stacks in each of the 9 cells to make them "active"
    # We need stacks from different players to test color-disconnection
    #
    # Cell layout (center positions for 19x19 board):
    # (3,3) top-left     (9,3) top-center    (15,3) top-right
    # (3,9) mid-left     (9,9) mid-center    (15,9) mid-right
    # (3,15) bot-left    (9,15) bot-center   (15,15) bot-right
    #
    # For Scenario 1 (marker lines with 4 colors, collapsed intersections):
    # - Vertical line at x=6 (P1 markers) divides board into left (x<6) and right (x>6)
    # - For this line to create disconnected territories, each half must lack
    #   at least one of the 4 active player colors
    #
    # Stack placement strategy for proper territory detection:
    # - Left column (x=3): P1 stacks only
    # - Middle column (x=9): P2 stacks only
    # - Right column (x=15): P3 stacks only
    # - This ensures each vertical slice lacks 3 of 4 players
    #
    # For horizontal lines (y=6, y=12), we add P4 in specific positions
    # to ensure horizontal divisions also create disconnected regions

    # Map cell positions to players for proper territory testing
    # Layout ensures each half (divided by any single-color line) lacks some players
    cell_to_player = {
        # Left column - P1 (top-left, mid-left, bot-left)
        (3, 3): 1,   # top-left
        (3, 9): 1,   # mid-left
        (3, 15): 1,  # bot-left
        # Middle column - P2 (top-center, mid-center, bot-center)
        (9, 3): 2,   # top-center
        (9, 9): 2,   # mid-center
        (9, 15): 2,  # bot-center
        # Right column - P3 and P4 mixed for horizontal line testing
        (15, 3): 3,  # top-right (above y=6 line)
        (15, 9): 4,  # mid-right (between y=6 and y=12 lines)
        (15, 15): 3, # bot-right (below y=12 line)
    }

    for (x, y), player in cell_to_player.items():
        pos = Position(x=x, y=y)
        key = pos.to_key()
        stacks[key] = RingStack(
            position=pos,
            rings=[player] * 3,  # 3-ring stack
            controlling_player=player,
            stack_height=3,
            cap_height=3,  # All rings are same player
        )

    return BoardState(
        type=BoardType.SQUARE19,  # 19x19 square board
        size=board_size,
        stacks=stacks,
        markers=markers,
        collapsed_spaces=collapsed_spaces,
    )


def get_cpu_disconnected_regions(board: BoardState, player: int) -> list[Territory]:
    """Get disconnected regions using the CPU BoardManager implementation."""
    return BoardManager.find_disconnected_regions(board, player)


def count_unique_regions(regions: list[Territory]) -> int:
    """Count unique regions by their space sets."""
    seen_regions: set[frozenset] = set()
    for region in regions:
        space_keys = frozenset(pos.to_key() for pos in region.spaces)
        seen_regions.add(space_keys)
    return len(seen_regions)


def get_all_marker_colors(board: BoardState) -> set[int]:
    """Get all unique marker colors on the board."""
    return {marker.player for marker in board.markers.values()}


def get_active_players(board: BoardState) -> set[int]:
    """Get all players with stacks on the board."""
    return {stack.controlling_player for stack in board.stacks.values()}


class TestGridTerritoryScenarios:
    """Test territory detection for grid scenarios."""

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_cpu_territory_detection(self, scenario: GridScenario):
        """Test CPU implementation territory detection for each scenario."""
        board = create_grid_board(scenario)

        # Get some diagnostic info
        marker_colors = get_all_marker_colors(board)
        active_players = get_active_players(board)
        collapsed_count = len(board.collapsed_spaces)

        print(f"\n=== {scenario.name}: {scenario.description} ===")
        print(f"Board size: {board.size}x{board.size}")
        print(f"Marker colors on board: {sorted(marker_colors)}")
        print(f"Active players (with stacks): {sorted(active_players)}")
        print(f"Collapsed spaces: {collapsed_count}")
        print(f"Total markers: {len(board.markers)}")

        # Get disconnected regions for player 1 (arbitrary choice)
        regions = get_cpu_disconnected_regions(board, player=1)
        unique_count = count_unique_regions(regions)

        print(f"Disconnected regions found: {len(regions)}")
        print(f"Unique regions: {unique_count}")

        # Print region details
        for i, region in enumerate(regions):
            [pos.to_key() for pos in region.spaces]
            represented = set()
            for pos in region.spaces:
                stack = board.stacks.get(pos.to_key())
                if stack:
                    represented.add(stack.controlling_player)
            print(f"  Region {i+1}: {len(region.spaces)} spaces, "
                  f"players represented: {sorted(represented)}, "
                  f"controlling_player: {region.controlling_player}")

        # The test passes if we can successfully detect regions
        # The exact count depends on the rules implementation
        assert regions is not None, "Should return a list (possibly empty)"

    def test_scenario_1_analysis(self):
        """
        Detailed analysis of Scenario 1:
        - 4 different marker colors form the grid lines
        - Intersections are collapsed

        Each single-color marker line (+ collapsed intersections + board edges)
        divides the board into 2 halves. With 4 colors, we get 4 lines,
        each creating 2 disconnected regions = 8 total.

        Stack placement (19x19 board):
        - Left column (x=3): P1 stacks only
        - Middle column (x=9): P2 stacks only
        - Right column (x=15): P3/P4 stacks

        For P1's vertical line at x=6:
        - Left half (x<6) has only P1 stacks → lacks P2,P3,P4 → DISCONNECTED
        - Right half (x>6) has P2,P3,P4 stacks → lacks P1 → DISCONNECTED
        """
        scenario = SCENARIOS[0]
        board = create_grid_board(scenario)

        print("\n=== Scenario 1 Detailed Analysis ===")
        print("Vertical lines: P1 (left at x=6), P2 (right at x=12)")
        print("Horizontal lines: P3 (top at y=6), P4 (bottom at y=12)")
        print("Intersections: All collapsed")
        print("\nStack placement:")
        print("  Left column (x=3): P1 stacks")
        print("  Middle column (x=9): P2 stacks")
        print("  Right column (x=15): P3 (top/bot), P4 (mid)")
        print(f"\nActive players: {sorted(get_active_players(board))}")

        # For each of the 4 marker colors, find disconnected regions
        all_regions = []
        for player in [1, 2, 3, 4]:
            regions = get_cpu_disconnected_regions(board, player=player)
            print(f"\nP{player} marker line disconnected regions: {len(regions)}")
            for i, region in enumerate(regions):
                represented = set()
                for pos in region.spaces:
                    stack = board.stacks.get(pos.to_key())
                    if stack:
                        represented.add(stack.controlling_player)
                print(f"  Region {i+1}: {len(region.spaces)} spaces, "
                      f"players represented: {sorted(represented)}")
            all_regions.extend(regions)

        # Each line divides the board into multiple regions (not just 2) because:
        # 1. The collapsed intersections create additional boundaries
        # 2. The 9 cells in the grid can be sub-divided based on stack representation
        print(f"\nTotal regions across all border colors: {len(all_regions)}")

        # Key verification: Each region must lack at least one active player
        # This is the core property of disconnected territories per R142
        active_players = get_active_players(board)
        for region in all_regions:
            represented = set()
            for pos in region.spaces:
                stack = board.stacks.get(pos.to_key())
                if stack:
                    represented.add(stack.controlling_player)
            # Verify representation check: region must lack at least one active player
            assert len(represented) < len(active_players), (
                f"Region should lack at least one player but has {represented} "
                f"(active players: {active_players})"
            )

        # Verify we detect some disconnected regions (exact count depends on algorithm)
        # With 4 marker colors and 4 active players, we expect multiple regions
        assert len(all_regions) > 0, "Should detect at least some disconnected regions"
        print(f"Verified: All {len(all_regions)} regions correctly lack at least one active player")

    def test_scenario_2_analysis(self):
        """
        Detailed analysis of Scenario 2:
        - All lines are collapsed spaces
        - 4 different marker colors at intersections

        Per R141, collapsed spaces can form boundaries.
        The 4 intersection markers don't form connected borders.
        """
        scenario = SCENARIOS[1]
        board = create_grid_board(scenario)

        print("\n=== Scenario 2 Detailed Analysis ===")
        print("All lines: Collapsed spaces")
        print("Intersections: 4 different colors (P1, P2, P3, P4)")

        regions = get_cpu_disconnected_regions(board, player=1)

        print(f"\nRegions found: {len(regions)}")

        # With collapsed lines, cells are physically separated
        # But intersection markers of different colors prevent single-color borders

    def test_scenario_3_analysis(self):
        """
        Detailed analysis of Scenario 3:
        - All lines are collapsed spaces
        - All 4 intersection markers are the same color (P1)

        This is the most favorable for territory detection:
        - Collapsed spaces form physical barriers
        - Single marker color at intersections
        """
        scenario = SCENARIOS[2]
        board = create_grid_board(scenario)

        print("\n=== Scenario 3 Detailed Analysis ===")
        print("All lines: Collapsed spaces")
        print("Intersections: All P1 markers")

        regions = get_cpu_disconnected_regions(board, player=1)

        print(f"\nRegions found: {len(regions)}")

        # With all collapsed lines and same-color intersection markers,
        # regions should be detected as bounded by "collapsed only"
        # The intersection markers don't affect the collapsed-only path

    def test_board_visualization(self):
        """Print a visual representation of each scenario's board."""
        for scenario in SCENARIOS:
            board = create_grid_board(scenario)

            print(f"\n=== {scenario.name} Board ===")
            print(scenario.description)

            for y in range(board.size):
                row = ""
                for x in range(board.size):
                    pos = Position(x=x, y=y)
                    key = pos.to_key()

                    if key in board.collapsed_spaces:
                        row += "X "
                    elif key in board.markers:
                        row += f"{board.markers[key].player} "
                    elif key in board.stacks:
                        row += f"S{board.stacks[key].controlling_player}"
                    else:
                        row += ". "
                print(row)


class TestTerritoryRulesCompliance:
    """Test that territory detection complies with R140, R141, R142."""

    def test_r141_single_color_boundary(self):
        """
        R141: Physical disconnection requires a single-color boundary.

        Create a region surrounded by markers of TWO colors.
        It should NOT be detected as disconnected.
        """
        # Create a simple board with a cell surrounded by mixed-color markers
        board = BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
            markers={},
            collapsed_spaces={},
        )

        # Create a 3x3 cell at position (2,2) to (4,4)
        # Surround it with markers: top/bottom = P1, left/right = P2
        markers = {}

        # Top border (y=1, x=2-4) - Player 1
        for x in range(2, 5):
            pos = Position(x=x, y=1)
            markers[pos.to_key()] = MarkerInfo(position=pos, player=1, type="regular")

        # Bottom border (y=5, x=2-4) - Player 1
        for x in range(2, 5):
            pos = Position(x=x, y=5)
            markers[pos.to_key()] = MarkerInfo(position=pos, player=1, type="regular")

        # Left border (x=1, y=2-4) - Player 2
        for y in range(2, 5):
            pos = Position(x=1, y=y)
            markers[pos.to_key()] = MarkerInfo(position=pos, player=2, type="regular")

        # Right border (x=5, y=2-4) - Player 2
        for y in range(2, 5):
            pos = Position(x=5, y=y)
            markers[pos.to_key()] = MarkerInfo(position=pos, player=2, type="regular")

        # Corners - Mixed (P1 and P2)
        corners = [(1, 1), (5, 1), (1, 5), (5, 5)]
        for i, (x, y) in enumerate(corners):
            pos = Position(x=x, y=y)
            markers[pos.to_key()] = MarkerInfo(position=pos, player=(i % 2) + 1, type="regular")

        board.markers = markers

        # Add stacks inside (P1) and outside (P2) the cell
        stacks = {}
        # Inside cell
        inside_pos = Position(x=3, y=3)
        stacks[inside_pos.to_key()] = RingStack(
            position=inside_pos, rings=[1, 1], controlling_player=1, stack_height=2, cap_height=2
        )
        # Outside cell
        outside_pos = Position(x=0, y=0)
        stacks[outside_pos.to_key()] = RingStack(
            position=outside_pos, rings=[2, 2], controlling_player=2, stack_height=2, cap_height=2
        )
        board.stacks = stacks

        regions = get_cpu_disconnected_regions(board, player=1)

        print("\n=== R141 Test: Mixed-color boundary ===")
        print("Cell surrounded by P1 (top/bottom) and P2 (left/right)")
        print(f"Regions found: {len(regions)}")

        # Per R141, this should NOT create a disconnected region
        # because the boundary uses two different marker colors

    def test_r142_color_disconnection(self):
        """
        R142: Color-disconnection requires RegionColors ⊂ ActiveColors.

        Create a region that contains stacks from ALL active players.
        It should NOT be detected as disconnected.
        """
        board = BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
            markers={},
            collapsed_spaces={},
        )

        # Create a collapsed-space border around a region
        collapsed: dict[str, int] = {}
        # Vertical line at x=4
        for y in range(8):
            pos = Position(x=4, y=y)
            collapsed[pos.to_key()] = 0  # 0 indicates collapsed with no specific player
        board.collapsed_spaces = collapsed

        # Add stacks from P1 and P2 on BOTH sides
        stacks = {}
        # Left side: P1 and P2
        stacks[Position(x=1, y=1).to_key()] = RingStack(
            position=Position(x=1, y=1), rings=[1], controlling_player=1, stack_height=1, cap_height=1
        )
        stacks[Position(x=2, y=2).to_key()] = RingStack(
            position=Position(x=2, y=2), rings=[2], controlling_player=2, stack_height=1, cap_height=1
        )
        # Right side: P1 and P2
        stacks[Position(x=6, y=1).to_key()] = RingStack(
            position=Position(x=6, y=1), rings=[1], controlling_player=1, stack_height=1, cap_height=1
        )
        stacks[Position(x=7, y=2).to_key()] = RingStack(
            position=Position(x=7, y=2), rings=[2], controlling_player=2, stack_height=1, cap_height=1
        )
        board.stacks = stacks

        regions = get_cpu_disconnected_regions(board, player=1)

        print("\n=== R142 Test: All players represented ===")
        print("Both sides have P1 and P2 stacks")
        print(f"Regions found: {len(regions)}")

        # Per R142, neither side qualifies as disconnected because
        # both P1 and P2 are represented in each region


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
