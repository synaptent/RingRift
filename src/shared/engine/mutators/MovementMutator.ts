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
    // Landing on a marker
    if (landingMarker.player === action.playerId) {
      // Landing on OWN marker:
      // - Remove the marker
      newState.board.markers.delete(toKey);

      // - Eliminate the bottom ring of the moving stack (which is the top ring in the array representation? No, rings are bottom-to-top)
      // Wait, let's check RingStack definition in types/game.ts: "rings: number[]; // Array of player numbers, bottom to top"
      // So index 0 is bottom, index length-1 is top.

      // Rule 8.2: "If you land on your own marker... remove the marker... and remove the bottom-most ring from your stack."
      const bottomRingOwner = stack.rings[0];
      const newRings = stack.rings.slice(1);

      // Update elimination counts
      newState.totalRingsEliminated++;
      newState.board.eliminatedRings[bottomRingOwner] =
        (newState.board.eliminatedRings[bottomRingOwner] || 0) + 1;

      const player = newState.players.find((p) => p.playerNumber === bottomRingOwner);
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
          controllingPlayer: newRings[newRings.length - 1],
        });
      }
    } else {
      // Landing on OPPONENT marker:
      // This is generally not allowed for simple moves, but if valid (e.g. future rule change or specific context),
      // the marker would flip. However, MovementValidator currently blocks this.
      // We'll assume validator passed, so this branch shouldn't be reached for standard moves unless rules change.
      // For robustness, we'll treat it as a normal move (replacing marker? No, that's capture).
      // Let's throw for now to ensure we don't silently do the wrong thing.
      throw new Error(
        'MovementMutator: Landing on opponent marker is not supported in simple movement'
      );
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
