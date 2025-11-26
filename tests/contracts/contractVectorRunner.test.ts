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
import { deserializeGameState } from '../../src/shared/engine/contracts/serialization';
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

// ═══════════════════════════════════════════════════════════════════════════
// Test Utilities
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert a contract vector move to the Move type expected by processTurn.
 */
function convertVectorMove(vectorMove: ContractTestVector['input']['move']): Move {
  return {
    id: vectorMove.id,
    type: vectorMove.type as Move['type'],
    player: vectorMove.player,
    from: vectorMove.from,
    to: vectorMove.to,
    captureTarget: vectorMove.captureTarget,
    placedOnStack: vectorMove.placedOnStack,
    placementCount: vectorMove.placementCount,
    formedLines: vectorMove.formedLines,
    timestamp: new Date(vectorMove.timestamp),
    thinkTime: vectorMove.thinkTime ?? 0,
    moveNumber: vectorMove.moveNumber,
  } as Move;
}

/**
 * Run a single test vector and return validation results.
 */
function runVector(vector: ContractTestVector): {
  passed: boolean;
  failures: string[];
  skipped: boolean;
  skipReason?: string;
} {
  try {
    // Deserialize input state
    const inputState = deserializeGameState(vector.input.state);

    // Convert the vector move to Move type
    const move = convertVectorMove(vector.input.move);

    // Run through the actual orchestrator
    const result = processTurn(inputState, move);

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
        expect(vector.input.move).toBeDefined();
        expect(vector.expectedOutput).toBeDefined();
        expect(vector.expectedOutput.status).toMatch(/^(complete|awaiting_decision)$/);
      }
    });
  });

  describe('Placement Vectors', () => {
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

    it('should pass all territory vectors', () => {
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
});

// ═══════════════════════════════════════════════════════════════════════════
// Utilities Export for CLI Usage
// ═══════════════════════════════════════════════════════════════════════════

export { loadAllVectors, loadVectorFile, runVector };
