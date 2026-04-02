/**
 * CaptureMutator branch coverage tests
 * Tests for src/shared/engine/mutators/CaptureMutator.ts
 */

import { mutateCapture } from '../../../src/shared/engine/mutators/CaptureMutator';
import type {
  GameState,
  OvertakingCaptureAction,
  ContinueChainAction,
} from '../../../src/shared/engine/types';
import type { BoardType, Position, RingStack } from '../../../src/shared/types/game';
import { inferTotalRingsInPlay } from '../../utils/fixtures';

function posStr(x: number, y: number): string {
  return `${x},${y}`;
}

function createMinimalState(
  overrides: Partial<{
    currentPhase: string;
    currentPlayer: number;
    boardType: BoardType;
    boardSize: number;
    stacks: Map<string, RingStack>;
    markers: Map<string, { player: number; position: Position; type: string }>;
    collapsedSpaces: Map<string, number>;
    players: Array<{
      playerNumber: number;
      ringsInHand: number;
      eliminated: boolean;
      eliminatedRings: number;
    }>;
  }>
): GameState {
  const boardType = overrides.boardType ?? 'square8';
  const boardSize = overrides.boardSize ?? 8;
  const playerCount = overrides.players?.length ?? 2;

  const players =
    overrides.players ??
    Array.from({ length: playerCount }, (_, i) => ({
      playerNumber: i + 1,
      ringsInHand: 10,
      eliminated: false,
      eliminatedRings: 0,
      score: 0,
      reserveStacks: 0,
      reserveRings: 0,
      territorySpaces: 0,
    }));
  const board = {
    type: boardType,
    size: boardSize,
    stacks: overrides.stacks ?? new Map(),
    markers: overrides.markers ?? new Map(),
    collapsedSpaces: overrides.collapsedSpaces ?? new Map(),
    rings: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    geometry: { type: boardType, size: boardSize },
  };

  return {
    board,
    currentPhase: overrides.currentPhase ?? 'movement',
    currentPlayer: overrides.currentPlayer ?? 1,
    players,
    turnNumber: 1,
    gameStatus: 'active',
    moveHistory: [],
    pendingDecision: null,
    victoryCondition: null,
    totalRingsInPlay: inferTotalRingsInPlay(players, board),
    totalRingsEliminated: 0,
  } as unknown as GameState;
}

describe('CaptureMutator', () => {
  describe('mutateCapture shim delegation', () => {
    it('should delegate to CaptureAggregate and return transformed state', () => {
      // Set up a valid capture scenario
      const stacks = new Map<string, RingStack>([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            rings: [1, 1],
            stackHeight: 2,
            capHeight: 2,
            controllingPlayer: 1,
          },
        ],
        [
          '2,0',
          {
            position: { x: 2, y: 0 },
            rings: [2],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        stacks,
        players: [
          { playerNumber: 1, ringsInHand: 5, eliminated: false, eliminatedRings: 0 },
          { playerNumber: 2, ringsInHand: 5, eliminated: false, eliminatedRings: 0 },
        ],
      });

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: { x: 0, y: 0 },
        captureTarget: { x: 2, y: 0 },
        to: { x: 4, y: 0 },
      };

      const result = mutateCapture(state, action);

      // Verify capture was applied
      expect(result.board.stacks.has('0,0')).toBe(false); // Attacker moved
      expect(result.board.stacks.has('2,0')).toBe(false); // Target captured
      expect(result.board.stacks.has('4,0')).toBe(true); // Landing position
      expect(result.board.markers.has('0,0')).toBe(true); // Marker at origin
    });

    it('should handle ContinueChainAction type', () => {
      const stacks = new Map<string, RingStack>([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            rings: [1, 1, 1],
            stackHeight: 3,
            capHeight: 3,
            controllingPlayer: 1,
          },
        ],
        [
          '2,0',
          {
            position: { x: 2, y: 0 },
            rings: [2],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPhase: 'chain_capture',
        currentPlayer: 1,
        stacks,
      });

      const action: ContinueChainAction = {
        type: 'CONTINUE_CHAIN',
        playerId: 1,
        from: { x: 0, y: 0 },
        captureTarget: { x: 2, y: 0 },
        to: { x: 4, y: 0 },
      };

      const result = mutateCapture(state, action);

      // Verify chain capture was applied
      expect(result.board.stacks.has('4,0')).toBe(true);
      expect(result.board.markers.has('0,0')).toBe(true);
    });
  });
});
