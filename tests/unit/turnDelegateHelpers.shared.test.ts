import {
  type BoardType,
  type Player,
  type TimeControl,
  type GameState,
  type Position,
  type RingStack,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { hasAnyPlacementForPlayer } from '../../src/shared/engine/turnDelegateHelpers';

describe('turnDelegateHelpers – hasAnyPlacementForPlayer', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createTestState(id: string): GameState {
    const state = createInitialGameState(
      id,
      boardType,
      basePlayers,
      timeControl
    ) as unknown as GameState;

    state.currentPhase = 'ring_placement';
    state.currentPlayer = 1;
    state.gameStatus = 'active';
    return state;
  }

  function addStack(
    state: GameState,
    position: Position,
    rings: number[],
    controllingPlayer: number
  ): void {
    const key = positionToString(position);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer,
    };
    state.board.stacks.set(key, stack);
  }

  it('returns true for a fresh game with rings in hand', () => {
    const state = createTestState('fresh-game');

    expect(hasAnyPlacementForPlayer(state, 1)).toBe(true);
    expect(hasAnyPlacementForPlayer(state, 2)).toBe(true);
  });

  it('returns false when player has no rings in hand', () => {
    const state = createTestState('no-rings');
    state.players = state.players.map((p) =>
      p.playerNumber === 1 ? { ...p, ringsInHand: 0 } : p
    );

    expect(hasAnyPlacementForPlayer(state, 1)).toBe(false);
    // Other players remain unaffected
    expect(hasAnyPlacementForPlayer(state, 2)).toBe(true);
  });

  it('returns false when ringsPerPlayer cap is fully reached on the board', () => {
    const state = createTestState('cap-reached');

    // Place all rings for player 1 on a single stack so that
    // ringsOnBoard === ringsPerPlayer while ringsInHand remains > 0.
    const pos: Position = { x: 0, y: 0 };
    addStack(
      state,
      pos,
      Array(18).fill(1),
      1
    );
    // Leave ringsInHand unchanged to simulate an over-cap state; the helper
    // should still respect the ringsPerPlayer cap and report no placements.
    state.players = state.players.map((p) =>
      p.playerNumber === 1 ? { ...p, ringsInHand: 5 } : p
    );

    expect(hasAnyPlacementForPlayer(state, 1)).toBe(false);
  });
});

