import { GameState, MoveStackAction } from '../types';
import { positionToString } from '../../types/game';
import { calculateCapHeight } from '../core';

export function mutateMovement(state: GameState, action: MoveStackAction): GameState {
  // Deep copy state for immutability
  // In a real production app with high performance needs, we'd use Immer or similar.
  // For now, manual copying of relevant parts is sufficient and explicit.
  // We cast to Mutable<GameState> internally to allow updates before returning as Readonly GameState
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
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };

  const fromKey = positionToString(action.from);
  const toKey = positionToString(action.to);
  const stack = newState.board.stacks.get(fromKey);

  if (!stack) {
    throw new Error('MovementMutator: No stack at origin');
  }

  // 1. Remove stack from origin
  newState.board.stacks.delete(fromKey);

  // 2. Place marker at origin
  newState.board.markers.set(fromKey, {
    player: action.playerId,
    position: action.from,
    type: 'regular',
  });

  // 3. Handle landing
  const landingMarker = newState.board.markers.get(toKey);

  if (landingMarker) {
    // Landing on any marker (own or opponent) per RR-CANON-R091/R092:
    // - Remove the marker (do not collapse)
    newState.board.markers.delete(toKey);

    // - Eliminate the TOP ring of the moving stack's cap
    // TOP ring is rings[0] per actual codebase convention (consistent with calculateCapHeight)
    const topRingOwner = stack.rings[0];
    const newRings = stack.rings.slice(1); // Remove first element (the top)

    // Update elimination counts
    newState.totalRingsEliminated++;
    newState.board.eliminatedRings[topRingOwner] =
      (newState.board.eliminatedRings[topRingOwner] || 0) + 1;

    const player = newState.players.find((p) => p.playerNumber === topRingOwner);
    if (player) {
      player.eliminatedRings++;
    }

    // If stack becomes empty (was height 1), it's gone. Otherwise place it.
    if (newRings.length > 0) {
      newState.board.stacks.set(toKey, {
        position: action.to,
        rings: newRings,
        stackHeight: newRings.length,
        capHeight: calculateCapHeight(newRings),
        controllingPlayer: newRings[0], // New top ring is the controller
      });
    }
  } else {
    // Landing on empty space
    newState.board.stacks.set(toKey, {
      ...stack,
      position: action.to,
    });
  }

  // 4. Update timestamps
  newState.lastMoveAt = new Date();

  return newState;
}
