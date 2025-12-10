/**
 * Cross-Language FSM Fixture Tests
 *
 * These tests load FSM transition vectors from JSON fixtures and validate
 * that the TypeScript FSM produces the expected results. The same fixtures
 * can be loaded by Python tests to ensure cross-language parity.
 *
 * Fixture location: tests/fixtures/fsm-parity/v1/fsm_transitions.vectors.json
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  transition,
  type TurnEvent,
  type TurnState,
  type GameContext,
  type RingPlacementState,
  type MovementState,
  type CaptureState,
  type ChainCaptureState,
  type LineProcessingState,
  type TerritoryProcessingState,
  type ForcedEliminationState,
  type TurnEndState,
} from '../../../src/shared/engine/fsm';

interface Position {
  x: number;
  y: number;
  z?: number;
}

interface FixtureEvent {
  type: string;
  to?: Position;
  from?: Position;
  target?: Position;
  player?: number;
  lineIndex?: number;
  regionIndex?: number;
}

interface StateContext {
  ringsInHand?: number;
  canPlace?: boolean;
  validPositions?: Position[];
  canMove?: boolean;
  hasCapturesAvailable?: boolean;
  pendingCaptures?: Array<{ target: Position }>;
  availableContinuations?: Array<{ target: Position }>;
  detectedLines?: Array<{ positions: Position[] }>;
  currentLineIndex?: number;
  hasDisconnectedRegions?: boolean;
  disconnectedRegions?: Array<{
    positions: Position[];
    eliminationsRequired: number;
  }>;
  currentRegionIndex?: number;
  ringsOverLimit?: number;
  eliminationsDone?: number;
  completedPlayer?: number;
  nextPlayer?: number;
}

interface FixtureInput {
  currentPhase: string;
  currentPlayer: number;
  numPlayers: number;
  event: FixtureEvent;
  stateContext: StateContext;
}

interface FixtureExpectedOutput {
  ok: boolean;
  nextPhase?: string;
  nextPlayer?: number;
  actions?: string[];
  reason?: string;
  winner?: number | null;
  newEliminationsDone?: number;
  errorCode?: string;
  errorCurrentPhase?: string;
  errorEventType?: string;
}

interface FixtureVector {
  id: string;
  category: string;
  description: string;
  input: FixtureInput;
  expectedOutput: FixtureExpectedOutput;
}

interface FixtureFile {
  version: string;
  generated: string;
  description: string;
  count: number;
  categories: string[];
  vectors: FixtureVector[];
}

// Load fixtures
const fixturesPath = path.join(
  __dirname,
  '../../../tests/fixtures/fsm-parity/v1/fsm_transitions.vectors.json'
);
const fixtureData: FixtureFile = JSON.parse(fs.readFileSync(fixturesPath, 'utf-8'));

// Build FSM state from fixture input
function buildState(input: FixtureInput): TurnState {
  const ctx = input.stateContext;

  switch (input.currentPhase) {
    case 'ring_placement':
      return {
        phase: 'ring_placement',
        player: input.currentPlayer,
        ringsInHand: ctx.ringsInHand ?? 0,
        canPlace: ctx.canPlace ?? true,
        validPositions: ctx.validPositions ?? [],
      } as RingPlacementState;

    case 'movement':
      return {
        phase: 'movement',
        player: input.currentPlayer,
        canMove: ctx.canMove ?? true,
        placedRingAt: null,
      } as MovementState;

    case 'capture':
      return {
        phase: 'capture',
        player: input.currentPlayer,
        pendingCaptures: (ctx.pendingCaptures ?? []).map((c) => ({
          target: c.target,
          capturingPlayer: input.currentPlayer,
          isChainCapture: false,
        })),
        chainInProgress: false,
        capturesMade: 0,
      } as CaptureState;

    case 'chain_capture':
      return {
        phase: 'chain_capture',
        player: input.currentPlayer,
        attackerPosition: { x: 0, y: 0 },
        capturedTargets: [],
        availableContinuations: (ctx.availableContinuations ?? []).map((c) => ({
          target: c.target,
          capturingPlayer: input.currentPlayer,
          isChainCapture: true,
        })),
        segmentCount: 1,
        isFirstSegment: false,
      } as ChainCaptureState;

    case 'line_processing':
      return {
        phase: 'line_processing',
        player: input.currentPlayer,
        detectedLines: (ctx.detectedLines ?? []).map((l) => ({
          positions: l.positions,
          player: input.currentPlayer,
          requiresChoice: false,
        })),
        currentLineIndex: ctx.currentLineIndex ?? 0,
        awaitingReward: false,
      } as LineProcessingState;

    case 'territory_processing':
      return {
        phase: 'territory_processing',
        player: input.currentPlayer,
        disconnectedRegions: (ctx.disconnectedRegions ?? []).map((r) => ({
          positions: r.positions,
          controllingPlayer: input.currentPlayer,
          eliminationsRequired: r.eliminationsRequired,
        })),
        currentRegionIndex: ctx.currentRegionIndex ?? 0,
        eliminationsPending: [],
      } as TerritoryProcessingState;

    case 'forced_elimination':
      return {
        phase: 'forced_elimination',
        player: input.currentPlayer,
        ringsOverLimit: ctx.ringsOverLimit ?? 1,
        eliminationsDone: ctx.eliminationsDone ?? 0,
      } as ForcedEliminationState;

    case 'turn_end':
      return {
        phase: 'turn_end',
        completedPlayer: ctx.completedPlayer ?? input.currentPlayer,
        nextPlayer: ctx.nextPlayer ?? (input.currentPlayer % input.numPlayers) + 1,
      } as TurnEndState;

    default:
      throw new Error(`Unknown phase: ${input.currentPhase}`);
  }
}

// Build FSM event from fixture input
function buildEvent(input: FixtureInput): TurnEvent {
  const evt = input.event;

  switch (evt.type) {
    case 'PLACE_RING':
      return { type: 'PLACE_RING', to: evt.to! };
    case 'SKIP_PLACEMENT':
      return { type: 'SKIP_PLACEMENT' };
    case 'NO_PLACEMENT_ACTION':
      return { type: 'NO_PLACEMENT_ACTION' };
    case 'MOVE_STACK':
      return { type: 'MOVE_STACK', from: evt.from!, to: evt.to! };
    case 'NO_MOVEMENT_ACTION':
      return { type: 'NO_MOVEMENT_ACTION' };
    case 'CAPTURE':
      return { type: 'CAPTURE', target: evt.target! };
    case 'END_CHAIN':
      return { type: 'END_CHAIN' };
    case 'CONTINUE_CHAIN':
      return { type: 'CONTINUE_CHAIN', target: evt.target! };
    case 'NO_LINE_ACTION':
      return { type: 'NO_LINE_ACTION' };
    case 'PROCESS_LINE':
      return { type: 'PROCESS_LINE', lineIndex: evt.lineIndex ?? 0 };
    case 'NO_TERRITORY_ACTION':
      return { type: 'NO_TERRITORY_ACTION' };
    case 'PROCESS_REGION':
      return { type: 'PROCESS_REGION', regionIndex: evt.regionIndex ?? 0 };
    case 'FORCED_ELIMINATE':
      return { type: 'FORCED_ELIMINATE', target: evt.target! };
    case '_ADVANCE_TURN':
      return { type: '_ADVANCE_TURN' };
    case 'RESIGN':
      return { type: 'RESIGN', player: evt.player ?? input.currentPlayer };
    case 'TIMEOUT':
      return { type: 'TIMEOUT', player: evt.player ?? input.currentPlayer };
    default:
      throw new Error(`Unknown event type: ${evt.type}`);
  }
}

// Build game context from fixture input
function buildContext(input: FixtureInput): GameContext {
  return {
    boardType: 'square8',
    numPlayers: input.numPlayers,
    ringsPerPlayer: 18,
    lineLength: 3,
  };
}

describe('Cross-Language FSM Fixtures', () => {
  // Group tests by category for readability
  const vectorsByCategory = new Map<string, FixtureVector[]>();
  for (const vector of fixtureData.vectors) {
    const category = vector.category;
    if (!vectorsByCategory.has(category)) {
      vectorsByCategory.set(category, []);
    }
    vectorsByCategory.get(category)!.push(vector);
  }

  for (const [category, vectors] of vectorsByCategory) {
    describe(`${category} transitions`, () => {
      for (const vector of vectors) {
        it(`[${vector.id}] ${vector.description}`, () => {
          const state = buildState(vector.input);
          const event = buildEvent(vector.input);
          const context = buildContext(vector.input);

          const result = transition(state, event, context);

          if (vector.expectedOutput.ok) {
            expect(result.ok).toBe(true);
            if (result.ok) {
              // Check phase transition
              if (vector.expectedOutput.nextPhase) {
                expect(result.state.phase).toBe(vector.expectedOutput.nextPhase);
              }

              // Check player (for phases that track player)
              if (vector.expectedOutput.nextPlayer !== undefined) {
                if (
                  result.state.phase === 'ring_placement' ||
                  result.state.phase === 'movement' ||
                  result.state.phase === 'capture' ||
                  result.state.phase === 'chain_capture' ||
                  result.state.phase === 'line_processing' ||
                  result.state.phase === 'territory_processing' ||
                  result.state.phase === 'forced_elimination'
                ) {
                  expect((result.state as any).player).toBe(vector.expectedOutput.nextPlayer);
                } else if (result.state.phase === 'turn_end') {
                  expect((result.state as TurnEndState).nextPlayer).toBe(
                    vector.expectedOutput.nextPlayer
                  );
                }
              }

              // Check game_over specific fields
              if (result.state.phase === 'game_over') {
                if (vector.expectedOutput.reason) {
                  expect(result.state.reason).toBe(vector.expectedOutput.reason);
                }
                if (vector.expectedOutput.winner !== undefined) {
                  expect(result.state.winner).toBe(vector.expectedOutput.winner);
                }
              }

              // Check forced_elimination counter
              if (
                result.state.phase === 'forced_elimination' &&
                vector.expectedOutput.newEliminationsDone !== undefined
              ) {
                expect((result.state as ForcedEliminationState).eliminationsDone).toBe(
                  vector.expectedOutput.newEliminationsDone
                );
              }

              // Check actions (if specified)
              if (vector.expectedOutput.actions) {
                const actualActionTypes = result.actions.map((a) => a.type);
                for (const expectedAction of vector.expectedOutput.actions) {
                  expect(actualActionTypes).toContain(expectedAction);
                }
              }
            }
          } else {
            // Expected error
            expect(result.ok).toBe(false);
            if (!result.ok) {
              if (vector.expectedOutput.errorCode) {
                expect(result.error.code).toBe(vector.expectedOutput.errorCode);
              }
              if (vector.expectedOutput.errorCurrentPhase) {
                expect(result.error.currentPhase).toBe(vector.expectedOutput.errorCurrentPhase);
              }
              if (vector.expectedOutput.errorEventType) {
                expect(result.error.eventType).toBe(vector.expectedOutput.errorEventType);
              }
            }
          }
        });
      }
    });
  }

  it('should have loaded all fixture vectors', () => {
    expect(fixtureData.vectors.length).toBe(fixtureData.count);
    expect(fixtureData.vectors.length).toBeGreaterThan(0);
  });
});
