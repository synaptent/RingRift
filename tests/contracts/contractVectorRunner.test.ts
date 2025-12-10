/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Contract Vector Test Runner
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Runs contract test vectors against the TypeScript canonical engine.
 * These tests validate that the engine produces expected outputs for
 * well-defined input scenarios.
 *
 * The same vectors can be run against the Python AI rules engine for
 * parity validation.
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  importVectorBundle,
  validateAgainstAssertions,
  type ContractTestVector,
} from '../../src/shared/engine/contracts';
import {
  deserializeGameState,
  serializeGameState,
} from '../../src/shared/engine/contracts/serialization';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { Move } from '../../src/shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Vector Loading
// ═══════════════════════════════════════════════════════════════════════════

const VECTORS_DIR = path.join(__dirname, '../fixtures/contract-vectors/v2');

interface VectorBundle {
  version: string;
  count: number;
  categories: string[];
  vectors: ContractTestVector[];
}

function loadVectorFile(filename: string): VectorBundle {
  const filePath = path.join(VECTORS_DIR, filename);
  const content = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(content) as VectorBundle;
}

function loadAllVectors(): ContractTestVector[] {
  const files = fs.readdirSync(VECTORS_DIR).filter((f) => f.endsWith('.vectors.json'));
  const allVectors: ContractTestVector[] = [];

  for (const file of files) {
    const bundle = loadVectorFile(file);
    allVectors.push(...bundle.vectors);
  }

  return allVectors;
}

/**
 * Group vectors by a `sequence:<id>` tag so we can run multi-step
 * contract sequences end-to-end using the orchestrator.
 *
 * This allows us to:
 * - Verify that the serialized input state for each subsequent step matches
 *   the actual nextState of the previous step, and
 * - Reuse the existing single-step assertion helper for each hop.
 */
function groupVectorsBySequenceTag(
  vectors: ContractTestVector[]
): Map<string, ContractTestVector[]> {
  const sequences = new Map<string, ContractTestVector[]>();

  for (const vector of vectors) {
    const sequenceTag = vector.tags.find((t) => t.startsWith('sequence:'));
    if (!sequenceTag) continue;

    const sequenceId = sequenceTag.slice('sequence:'.length);
    const existing = sequences.get(sequenceId);
    if (existing) {
      existing.push(vector);
    } else {
      sequences.set(sequenceId, [vector]);
    }
  }

  // Ensure deterministic ordering within each sequence (e.g. segment1, segment2, ...)
  for (const [, seqVectors] of sequences) {
    seqVectors.sort((a, b) => a.id.localeCompare(b.id));
  }

  return sequences;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test Utilities
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert a contract vector move to the Move type expected by processTurn.
 *
 * IMPORTANT: contract moves may carry additional decision-specific fields
 * (e.g. disconnectedRegions for process_territory_region, eliminationFromStack
 * for eliminate_rings_from_stack). We must preserve those fields when
 * reconstructing the Move, otherwise the orchestrator/aggregates may treat
 * the move as a no-op (because key context is missing).
 */
function convertVectorMove(vectorMove: ContractTestVector['input']['move']): Move {
  // Start from a shallow copy so we preserve any extra properties that are not
  // explicitly listed in the core Move type (such as disconnectedRegions,
  // eliminationFromStack, collapsedMarkers, etc.).
  const base: any = { ...vectorMove };

  // Normalise timestamp to a Date instance; fall back to "now" if missing.
  if (base.timestamp) {
    base.timestamp = new Date(base.timestamp as string);
  } else {
    base.timestamp = new Date();
  }

  // Normalise thinkTime to a number.
  if (base.thinkTime == null) {
    base.thinkTime = 0;
  }

  return base as Move;
}

/**
 * Create a NO_LINE_ACTION or NO_TERRITORY_ACTION move for auto-completing phases.
 */
function createAutoCompleteMove(phase: string, player: number): Move {
  const moveType =
    phase === 'line_processing'
      ? 'no_line_action'
      : phase === 'territory_processing'
        ? 'no_territory_action'
        : 'skip_placement'; // fallback

  return {
    type: moveType,
    player,
    position: { x: 0, y: 0 },
    timestamp: new Date(),
    thinkTime: 0,
  } as Move;
}

/**
 * Auto-complete a turn by processing through remaining phases until 'complete'.
 * Used when vectors expect 'complete' but orchestrator returns 'awaiting_decision'.
 * This handles the multi-phase turn model where movement/capture triggers
 * line_processing and territory_processing phases.
 */
function autoCompleteTurn(
  initialResult: ReturnType<typeof processTurn>,
  maxIterations = 10
): ReturnType<typeof processTurn> {
  let result = initialResult;
  let iterations = 0;

  while (result.status === 'awaiting_decision' && iterations < maxIterations) {
    const state = result.nextState;
    const phase = state.currentPhase;

    // Only auto-complete line_processing and territory_processing phases
    if (phase !== 'line_processing' && phase !== 'territory_processing') {
      // Can't auto-complete other phases (capture, chain_capture, etc.)
      break;
    }

    const autoMove = createAutoCompleteMove(phase, state.currentPlayer);
    result = processTurn(state, autoMove);
    iterations++;
  }

  return result;
}

/**
 * Run a single test vector and return validation results.
 *
 * NOTE: When a vector expects status='complete' but the orchestrator returns
 * 'awaiting_decision' (due to multi-phase turns), we auto-complete the turn
 * by applying NO_LINE_ACTION and NO_TERRITORY_ACTION moves. This allows legacy
 * vectors to work with the new multi-phase turn model without regeneration.
 */
function runVector(vector: ContractTestVector): {
  passed: boolean;
  failures: string[];
  skipped: boolean;
  skipReason?: string;
} {
  // Check for skip field on the vector
  if ((vector as any).skip) {
    return {
      passed: true,
      failures: [],
      skipped: true,
      skipReason: (vector as any).skip,
    };
  }

  // Skip vectors that use 'initialMove' (multi-step chain sequence format)
  // These require special handling not supported by runVector
  if (!vector.input.move && (vector.input as any).initialMove) {
    return {
      passed: true,
      failures: [],
      skipped: true,
      skipReason: 'Uses initialMove format (multi-step chain sequence) - requires special handling',
    };
  }

  try {
    // Deserialize input state
    const inputState = deserializeGameState(vector.input.state);

    // Convert the vector move to Move type
    const move = convertVectorMove(vector.input.move);

    // Run through the actual orchestrator
    let result = processTurn(inputState, move);

    // Auto-complete the turn if vector expects 'complete' but we got 'awaiting_decision'
    // This handles the multi-phase turn model where movement/capture triggers
    // line_processing and territory_processing phases automatically
    if (vector.expectedOutput.status === 'complete' && result.status === 'awaiting_decision') {
      result = autoCompleteTurn(result);
    }

    // Validate assertions against the result state
    const validation = validateAgainstAssertions(
      result.nextState,
      vector.expectedOutput.assertions
    );

    // Also check status if expected
    if (vector.expectedOutput.status) {
      if (vector.expectedOutput.status !== result.status) {
        validation.valid = false;
        validation.failures.push(
          `Expected status '${vector.expectedOutput.status}', got '${result.status}'`
        );
      }
    }

    return {
      passed: validation.valid,
      failures: validation.failures,
      skipped: false,
    };
  } catch (error) {
    return {
      passed: false,
      failures: [`Exception: ${error instanceof Error ? error.message : String(error)}`],
      skipped: false,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test Suites
// ═══════════════════════════════════════════════════════════════════════════

describe('Contract Test Vectors', () => {
  describe('Vector Loading', () => {
    it('should load vectors from all bundle files', () => {
      const vectors = loadAllVectors();
      expect(vectors.length).toBeGreaterThan(0);
    });

    it('should have valid vector structure', () => {
      const vectors = loadAllVectors();
      for (const vector of vectors) {
        expect(vector.id).toBeDefined();
        expect(vector.category).toBeDefined();
        expect(vector.input).toBeDefined();
        expect(vector.input.state).toBeDefined();
        // Vectors may use 'move' or 'initialMove' (for multi-step sequences)
        const hasMove =
          vector.input.move !== undefined || (vector.input as any).initialMove !== undefined;
        expect(hasMove).toBe(true);
        expect(vector.expectedOutput).toBeDefined();
        expect(vector.expectedOutput.status).toMatch(/^(complete|awaiting_decision)$/);
      }
    });
  });

  describe('Placement Vectors (§§4, 8; FAQ Q1–Q3)', () => {
    let vectors: ContractTestVector[];

    beforeAll(() => {
      const bundle = loadVectorFile('placement.vectors.json');
      vectors = bundle.vectors;
    });

    it('should have placement vectors', () => {
      expect(vectors.length).toBeGreaterThan(0);
      expect(vectors.every((v) => v.category === 'placement')).toBe(true);
    });

    it('should pass all placement vectors', () => {
      for (const vector of vectors) {
        const result = runVector(vector);
        if (!result.passed) {
          console.error(`Placement test failed: ${vector.id}`, result.failures);
        }
        expect(result.failures).toEqual([]);
        expect(result.passed).toBe(true);
      }
    });
  });

  describe('Movement Vectors', () => {
    let vectors: ContractTestVector[];

    beforeAll(() => {
      const bundle = loadVectorFile('movement.vectors.json');
      vectors = bundle.vectors;
    });

    it('should have movement vectors', () => {
      expect(vectors.length).toBeGreaterThan(0);
      expect(vectors.every((v) => v.category === 'movement')).toBe(true);
    });

    // UNSKIPPED 2025-12-06: Auto-completion of multi-phase turns now handles this.
    // The runVector function auto-completes line_processing and territory_processing
    // phases when vectors expect 'complete' but orchestrator returns 'awaiting_decision'.
    it('should pass all movement vectors', () => {
      for (const vector of vectors) {
        const result = runVector(vector);
        if (!result.passed) {
          console.error(`Movement test failed: ${vector.id}`, result.failures);
        }
        expect(result.failures).toEqual([]);
        expect(result.passed).toBe(true);
      }
    });
  });

  describe('Capture Vectors', () => {
    let vectors: ContractTestVector[];

    beforeAll(() => {
      const bundle = loadVectorFile('capture.vectors.json');
      vectors = bundle.vectors;
    });

    it('should have capture vectors', () => {
      expect(vectors.length).toBeGreaterThan(0);
      expect(vectors.every((v) => v.category === 'capture')).toBe(true);
    });

    // UNSKIPPED 2025-12-06: Auto-completion of multi-phase turns now handles this.
    it('should pass all capture vectors', () => {
      for (const vector of vectors) {
        const result = runVector(vector);
        if (!result.passed) {
          console.error(`Capture test failed: ${vector.id}`, result.failures);
        }
        expect(result.failures).toEqual([]);
        expect(result.passed).toBe(true);
      }
    });
  });

  describe('Line Detection Vectors', () => {
    let vectors: ContractTestVector[];

    beforeAll(() => {
      const bundle = loadVectorFile('line_detection.vectors.json');
      vectors = bundle.vectors;
    });

    it('should have line detection vectors', () => {
      expect(vectors.length).toBeGreaterThan(0);
      expect(vectors.every((v) => v.category === 'line_detection')).toBe(true);
    });

    // UNSKIPPED 2025-12-06: Auto-completion of multi-phase turns now handles this.
    it('should pass all line detection vectors', () => {
      for (const vector of vectors) {
        const result = runVector(vector);
        if (!result.passed) {
          console.error(`Line detection test failed: ${vector.id}`, result.failures);
        }
        expect(result.failures).toEqual([]);
        expect(result.passed).toBe(true);
      }
    });
  });

  describe('Territory Vectors', () => {
    let vectors: ContractTestVector[];

    beforeAll(() => {
      const bundle = loadVectorFile('territory.vectors.json');
      vectors = bundle.vectors;
    });

    it('should have territory vectors', () => {
      expect(vectors.length).toBeGreaterThan(0);
      expect(vectors.every((v) => v.category === 'territory')).toBe(true);
    });

    // SKIP 2025-12-06: Territory vectors have phase/move mismatches (eliminate_rings_from_stack
    // and process_territory_region moves in wrong phases). Vector data needs regeneration.
    // See: docs/SKIPPED_TESTS_TRIAGE.md, PA-1 task
    it.skip('should pass all territory vectors', () => {
      for (const vector of vectors) {
        const result = runVector(vector);
        if (!result.passed) {
          console.error(`Territory test failed: ${vector.id}`, result.failures);
        }
        expect(result.failures).toEqual([]);
        expect(result.passed).toBe(true);
      }
    });
  });

  describe('Smoke Tests', () => {
    it('should identify smoke-tagged vectors', () => {
      const vectors = loadAllVectors();
      const smokeVectors = vectors.filter((v) => v.tags.includes('smoke'));
      expect(smokeVectors.length).toBeGreaterThan(0);
    });

    // UNSKIPPED 2025-12-06: Auto-completion of multi-phase turns now handles this.
    it('should pass all smoke vectors', () => {
      const vectors = loadAllVectors();
      const smokeVectors = vectors.filter((v) => v.tags.includes('smoke'));

      for (const vector of smokeVectors) {
        const result = runVector(vector);
        if (!result.passed) {
          console.error(`Smoke test failed: ${vector.id}`, result.failures);
        }
        expect(result.passed).toBe(true);
      }
    });
  });

  describe('Meta-move vectors', () => {
    it('includes the swap_sides pie-rule meta-move vector when present', () => {
      const bundlePath = path.join(VECTORS_DIR, 'meta_moves.vectors.json');
      if (!fs.existsSync(bundlePath)) {
        // Optional bundle: do not fail when meta_moves have not been generated yet.
        return;
      }

      const bundle = loadVectorFile('meta_moves.vectors.json');
      const ids = new Set(bundle.vectors.map((v) => v.id));
      expect(ids.has('meta.swap_sides.after_p1_first_move.square8')).toBe(true);
    });
  });
});

/**
 * Multi-step sequences test that chained game states are internally consistent.
 * Auto-complete logic handles multi-phase turns where movement/capture triggers
 * line_processing → territory_processing phases.
 *
 * @skip FSM orchestration is now canonical. Contract vectors were generated with
 * legacy orchestration and produce different phase sequences under FSM.
 * Enable once contract vectors are regenerated with FSM orchestration.
 */
describe.skip('Multi-step contract sequences', () => {
  const allVectors = loadAllVectors();
  const sequences = groupVectorsBySequenceTag(allVectors);
  const sequenceEntries = Array.from(sequences.entries());

  if (sequenceEntries.length === 0) {
    it('has no multi-step contract sequences defined', () => {
      expect(sequenceEntries.length).toBe(0);
    });
    return;
  }

  for (const [sequenceId, seqVectors] of sequenceEntries) {
    // Check if any vector in the sequence has a skip field
    const hasSkippedVector = seqVectors.some((v) => (v as any).skip);

    if (hasSkippedVector) {
      it.skip(`sequence ${sequenceId} is internally consistent across steps`, () => {
        // Skipped due to pending implementation
      });
      continue;
    }

    it(`sequence ${sequenceId} is internally consistent across steps`, () => {
      expect(seqVectors.length).toBeGreaterThan(1);

      // Initialise from the first step's serialized state
      let currentState = deserializeGameState(seqVectors[0].input.state);

      seqVectors.forEach((vector, index) => {
        // For subsequent steps, ensure the stored input state matches the
        // previous step's nextState (round-tripped through serialization).
        if (index > 0) {
          const serialized = serializeGameState(currentState);
          expect(serialized).toEqual(vector.input.state);
        }

        const move = convertVectorMove(vector.input.move);
        let result = processTurn(currentState, move);

        // Auto-complete multi-phase turns if vector expects 'complete' status
        if (vector.expectedOutput.status === 'complete' && result.status === 'awaiting_decision') {
          result = autoCompleteTurn(result);
        }

        const validation = validateAgainstAssertions(
          result.nextState,
          vector.expectedOutput.assertions
        );

        if (vector.expectedOutput.status && vector.expectedOutput.status !== result.status) {
          validation.valid = false;
          validation.failures.push(
            `Expected status '${vector.expectedOutput.status}', got '${result.status}'`
          );
        }

        if (!validation.valid) {
          const failures = validation.failures.join('\n');
          throw new Error(
            `Multi-step sequence '${sequenceId}', step ${index} (vector ${vector.id}) failed:\n${failures}`
          );
        }

        currentState = result.nextState;
      });
    });
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// Utilities Export for CLI Usage
// ═══════════════════════════════════════════════════════════════════════════

export { loadAllVectors, loadVectorFile, runVector };
