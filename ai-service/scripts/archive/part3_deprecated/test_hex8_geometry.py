#!/usr/bin/env python
"""Test hex8 board geometry and territory detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.territory_cache import BoardGeometryCache
from app.models import BoardType

print("=== HEX8 Board Geometry Test ===\n")

# Test hex8 geometry
geo = BoardGeometryCache.get("hexagonal", 9)  # hex8 size = 2*radius + 1 = 9

print(f"Board type: hexagonal (hex8)")
print(f"Size: 9")
print(f"Total positions: {geo.num_positions}")
print(f"Expected for hex8 (radius=4): {3 * 4 * (4 + 1) + 1} positions")

# Show a few positions
print(f"\nFirst 10 positions:")
for i in range(min(10, len(geo.idx_to_position))):
    pos = geo.idx_to_position[i]
    num_neighbors = geo.num_neighbors[i]
    print(f"  {i}: {pos} ({num_neighbors} neighbors)")

# Check if there are any positions at (0,0) without z
print(f"\nChecking for key format issues:")
has_2d_keys = any("," in key and key.count(",") == 1 for key in geo.idx_to_position)
has_3d_keys = any("," in key and key.count(",") == 2 for key in geo.idx_to_position)
print(f"  Has 2D keys (x,y): {has_2d_keys}")
print(f"  Has 3D keys (x,y,z): {has_3d_keys}")

# Test with actual board state
print("\n=== Testing Territory Detection ===\n")

from app.board_manager import BoardManager
from app.models import BoardState, Position, RingStack
from app.training.initial_state import create_initial_state

game_state = create_initial_state(BoardType.HEX8, num_players=2)
board = game_state.board

print(f"Initial board:")
print(f"  Board type: {board.type}")
print(f"  Size: {board.size}")
print(f"  Stacks: {len(board.stacks)}")
print(f"  Markers: {len(board.markers)}")
print(f"  Collapsed: {len(board.collapsed_spaces)}")

# Check stack keys
print(f"\nStack keys (first 5):")
for i, key in enumerate(list(board.stacks.keys())[:5]):
    stack = board.stacks[key]
    print(f"  {key}: controlling_player={stack.controlling_player}, pos={stack.position}")

# Try to find disconnected regions
print(f"\n=== Finding Disconnected Regions ===")
import time
start = time.time()
regions = BoardManager.find_disconnected_regions(board, player_number=1)
elapsed = time.time() - start

print(f"Found {len(regions)} regions in {elapsed:.3f}s")

for i, region in enumerate(regions[:3]):
    print(f"\nRegion {i}:")
    print(f"  Spaces: {len(region.spaces)}")
    print(f"  Controlling player: {region.controlling_player}")
    print(f"  Is disconnected: {region.is_disconnected}")
    if region.spaces:
        print(f"  First space: {region.spaces[0]}")
        print(f"  First space key: {region.spaces[0].to_key()}")

# Now try with a board that has stacks from multiple players
print(f"\n=== Testing with 2 players on board ===")

# Add a stack for player 2
p2_stack = RingStack(
    position=Position(x=2, y=0, z=-2),
    rings=[2],
    controlling_player=2,
    stack_height=1,
    cap_height=1,
)
board.stacks[p2_stack.position.to_key()] = p2_stack

print(f"Board now has {len(board.stacks)} stacks")

start = time.time()
regions = BoardManager.find_disconnected_regions(board, player_number=1)
elapsed = time.time() - start

print(f"Found {len(regions)} regions in {elapsed:.3f}s")

if len(regions) > 5:
    print("\nWARNING: Too many regions! This might indicate a bug.")

for i, region in enumerate(regions[:5]):
    print(f"\nRegion {i}:")
    print(f"  Spaces: {len(region.spaces)}")
    print(f"  Controlling player: {region.controlling_player}")

print("\nTest complete")
