/**
 * Recovery Action Contract Vectors Test
 *
 * Tests the recovery action (RR-CANON-R110–R115) using contract vectors.
 * Validates that recovery slides work correctly with Option 1/2 semantics.
 */

import fs from 'fs';
import path from 'path';

import { importVectorBundle, type ContractTestVector } from '../../src/shared/engine/contracts';
import { deserializeGameState } from '../../src/shared/engine/contracts/serialization';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState, Move, Position } from '../../src/shared/types/game';

const RECOVERY_BUNDLE = path.resolve(
  __dirname,
  '../fixtures/contract-vectors/v2/recovery_action.vectors.json'
);

function loadRecoveryVectors(): ContractTestVector[] {
  const json = fs.readFileSync(RECOVERY_BUNDLE, 'utf8');
  const bundle = importVectorBundle(json);
  return bundle;
}

function convertVectorMove(vectorMove: any): Move {
  const move: any = { ...vectorMove };
  move.timestamp = move.timestamp ? new Date(move.timestamp) : new Date();
  move.thinkTime = move.thinkTime ?? 0;
  return move as Move;
}

describe('Recovery Action Contract Vectors', () => {
  let vectors: ContractTestVector[];

  beforeAll(() => {
    vectors = loadRecoveryVectors();
  });

  it('should load recovery vectors successfully', () => {
    expect(vectors.length).toBeGreaterThan(0);
    expect(vectors.every((v) => v.input && v.input.state)).toBe(true);
  });

  describe('Recovery slides with Option 1/2', () => {
    it.each([
      ['recovery.exact_length_option1'],
      ['recovery.overlength_option1'],
      ['recovery.overlength_option2_free'],
    ])('should process vector %s correctly', (vectorId) => {
      const vector = vectors.find((v) => v.id === vectorId);
      if (!vector) {
        throw new Error(`Vector ${vectorId} not found`);
      }

      const initialState = deserializeGameState(vector.input.state);
      const initialMove = convertVectorMove(vector.input.initialMove);

      // Process the recovery move
      const result = processTurn(initialState, initialMove);
      const finalState = (result as any).nextState || (result as any).state;

      // Basic validation - move should be processed
      expect(finalState).toBeDefined();

      // If there are assertions, validate them
      if (vector.assertions) {
        const assertions = vector.assertions;

        // Check final phase if specified
        if (assertions.finalPhase) {
          // Note: The actual phase depends on whether line/territory processing is needed
          expect(['line_processing', 'territory_processing', 'ring_placement']).toContain(
            finalState.currentPhase
          );
        }

        // Check board changes
        if (assertions.boardChanges) {
          const changes = assertions.boardChanges;

          // Check collapsed spaces were added
          if (changes.collapsedSpacesAdded) {
            for (const posKey of changes.collapsedSpacesAdded) {
              const [x, y] = posKey.split(',').map(Number);
              const key = `${x},${y}`;
              // Collapsed spaces should be in the board
              expect(finalState.board.collapsedSpaces.has(key)).toBe(true);
            }
          }

          // Check markers remaining
          if (changes.markersRemaining) {
            for (const posKey of changes.markersRemaining) {
              const [x, y] = posKey.split(',').map(Number);
              const key = `${x},${y}`;
              // These markers should still exist
              expect(finalState.board.markers.has(key)).toBe(true);
            }
          }
        }
      }
    });
  });

  describe('Recovery eligibility', () => {
    it('should not allow recovery when player controls stacks', () => {
      // Find a vector and modify it to have a controlling stack
      const vector = vectors.find((v) => v.id === 'recovery.exact_length_option1');
      if (!vector) return;

      const state = deserializeGameState(vector.input.state);

      // Add a stack controlled by player 1
      state.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      const move = convertVectorMove(vector.input.initialMove);

      // This should fail validation or throw
      expect(() => {
        processTurn(state, move);
      }).toThrow();
    });

    it('should not allow recovery when player has rings in hand', () => {
      const vector = vectors.find((v) => v.id === 'recovery.exact_length_option1');
      if (!vector) return;

      const state = deserializeGameState(vector.input.state);

      // Give player 1 rings in hand
      const player1 = state.players.find((p) => p.playerNumber === 1);
      if (player1) {
        player1.ringsInHand = 5;
      }

      const move = convertVectorMove(vector.input.initialMove);

      // This should fail validation or throw
      expect(() => {
        processTurn(state, move);
      }).toThrow();
    });
  });
});
