import { GameState, OvertakingCaptureAction, ContinueChainAction } from '../types';
import { positionToString } from '../../types/game';
import { calculateCapHeight, getPathPositions } from '../core';

export function mutateCapture(
  state: GameState,
  action: OvertakingCaptureAction | ContinueChainAction
): GameState {
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      markers: new Map(state.board.markers),
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    lastMoveAt: Date;
  };

  const fromKey = positionToString(action.from);
  const targetKey = positionToString(action.captureTarget);
  const toKey = positionToString(action.to);

  const attacker = newState.board.stacks.get(fromKey);
  const target = newState.board.stacks.get(targetKey);

  if (!attacker || !target) {
    throw new Error('CaptureMutator: Missing attacker or target stack');
  }

  // 1. Remove attacker from origin
  newState.board.stacks.delete(fromKey);

  // 2. Place marker at origin
  newState.board.markers.set(fromKey, {
    player: action.playerId,
    position: action.from,
    type: 'regular',
  });

  // 2.5 Process markers along the path (flip opponent's, collapse own)
  // Path includes from -> target -> landing
  // We need to process segments: from->target and target->landing
  // Excluding endpoints for each segment, but including target?
  // Rule 4.1.6: "Markers on path: Process as in non-capture movement"
  // Path is from -> target -> landing.
  // Intermediate cells are between from and landing.
  // This includes target if we consider the full path.
  // But target has a stack, so it can't have a marker.
  // So we just need to process cells between from->target and target->landing.

  const path1 = getPathPositions(action.from, action.captureTarget);
  const path2 = getPathPositions(action.captureTarget, action.to);

  // Combine paths, removing duplicates (target is in both)
  // Actually getPathPositions includes start and end.
  // path1: [from, ..., target]
  // path2: [target, ..., to]
  // We want intermediate cells: path1.slice(1, -1) and path2.slice(1, -1).

  const intermediatePositions = [...path1.slice(1, -1), ...path2.slice(1, -1)];

  for (const pos of intermediatePositions) {
    const key = positionToString(pos);
    const marker = newState.board.markers.get(key);

    if (marker) {
      if (marker.player !== action.playerId) {
        // Flip opponent marker
        newState.board.markers.set(key, {
          ...marker,
          player: action.playerId,
        });
      } else {
        // Collapse own marker
        newState.board.markers.delete(key);
        newState.board.collapsedSpaces.set(key, action.playerId);
      }
    }
  }

  // 3. Remove target stack
  newState.board.stacks.delete(targetKey);

  // 4. Create new stack at landing
  // Rule 4.2.3: "Pop the top ring from the target stack and append it to the bottom of the attacking stack’s rings array."
  // rings array is [top, ..., bottom] (consistent with calculateCapHeight).
  // So we take target.rings[0] (top) and append it to the end of newRings (bottom).

  const capturedRing = target.rings[0];
  const newRings = [...attacker.rings, capturedRing];

  newState.board.stacks.set(toKey, {
    position: action.to,
    rings: newRings,
    stackHeight: newRings.length,
    capHeight: calculateCapHeight(newRings),
    controllingPlayer: attacker.controllingPlayer, // Attacker remains on top (index 0)
  });

  // 4.5 Update target stack (if rings remain)
  if (target.rings.length > 1) {
    const remainingRings = target.rings.slice(1);
    newState.board.stacks.set(targetKey, {
      ...target,
      rings: remainingRings,
      stackHeight: remainingRings.length,
      capHeight: calculateCapHeight(remainingRings),
      controllingPlayer: remainingRings[0], // New top ring
    });
  }

  // 5. Handle landing on own marker (if applicable)
  // Rule 10.2: "If the landing space is occupied by your own marker... remove the marker... and remove the bottom-most ring from the NEW combined stack."
  const landingMarker = newState.board.markers.get(toKey);
  if (landingMarker && landingMarker.player === action.playerId) {
    newState.board.markers.delete(toKey);

    const bottomRingOwner = newRings[newRings.length - 1];
    newRings.pop(); // Remove bottom ring (last element)

    // Update elimination counts
    // Note: totalRingsEliminated is readonly in GameState interface but we cast it above if needed.
    // However, we didn't cast totalRingsEliminated in the initial assignment.
    // Let's fix the cast or use a helper.
    // For now, we'll assume the cast allows it or we'll fix it if TS complains.
    // Actually, we only cast lastMoveAt. Let's update the cast.
    (newState as any).totalRingsEliminated++;
    newState.board.eliminatedRings[bottomRingOwner] =
      (newState.board.eliminatedRings[bottomRingOwner] || 0) + 1;

    const player = newState.players.find((p) => p.playerNumber === bottomRingOwner);
    if (player) {
      player.eliminatedRings++;
    }

    // Update the stack on board with the reduced rings
    if (newRings.length > 0) {
      const stack = newState.board.stacks.get(toKey)!;
      stack.rings = newRings;
      stack.stackHeight = newRings.length;
      stack.capHeight = calculateCapHeight(newRings);
      // controllingPlayer remains same
    } else {
      // Stack eliminated completely? Rare but possible if 1+1=2 and then eliminated?
      // Actually capture requires capHeight >= target capHeight. Min capHeight is 1.
      // So min stack size is 1+1=2. Removing 1 leaves 1. So stack never disappears here.
    }
  }

  // 6. Update timestamps
  newState.lastMoveAt = new Date();

  return newState;
}
