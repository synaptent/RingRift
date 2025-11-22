import { readdirSync, readFileSync } from 'fs';
import { join } from 'path';

import { GameEngine } from '../../src/shared/engine/GameEngine';
import { computeProgressSnapshot, hashGameState } from '../../src/shared/engine/core';
import { moveToGameAction } from '../../src/shared/engine/moveActionAdapter';
import type { Move, GameState as SharedGameState, BoardType } from '../../src/shared/types/game';

// Helper to hydrate JSON-deserialized state back into Maps/Sets
function hydrateGameState(state: any): SharedGameState {
  const hydrated = { ...state };

  if (hydrated.board) {
    hydrated.board = { ...hydrated.board };

    // Hydrate stacks
    if (Array.isArray(hydrated.board.stacks)) {
      hydrated.board.stacks = new Map(hydrated.board.stacks);
    } else if (typeof hydrated.board.stacks === 'object' && hydrated.board.stacks !== null) {
      hydrated.board.stacks = new Map(Object.entries(hydrated.board.stacks));
    }

    // Hydrate markers
    if (Array.isArray(hydrated.board.markers)) {
      hydrated.board.markers = new Map(hydrated.board.markers);
    } else if (typeof hydrated.board.markers === 'object' && hydrated.board.markers !== null) {
      hydrated.board.markers = new Map(Object.entries(hydrated.board.markers));
    }

    // Hydrate collapsedSpaces
    // Note: In some versions this might be a Set or a Map.
    // The codebase seems to use Map<string, boolean> or Set<string>.
    // initialState.ts uses new Map().
    if (Array.isArray(hydrated.board.collapsedSpaces)) {
      // If it's an array of strings, it might be a Set serialization or Map entries
      if (
        hydrated.board.collapsedSpaces.length > 0 &&
        typeof hydrated.board.collapsedSpaces[0] === 'string'
      ) {
        // Assume it's a list of keys for a Set, but we need a Map based on initialState.ts
        // Actually, let's check if it's [[key, val]] or [key, key]
        hydrated.board.collapsedSpaces = new Map(
          hydrated.board.collapsedSpaces.map((k: string) => [k, true])
        );
      } else {
        hydrated.board.collapsedSpaces = new Map(hydrated.board.collapsedSpaces);
      }
    } else if (
      typeof hydrated.board.collapsedSpaces === 'object' &&
      hydrated.board.collapsedSpaces !== null
    ) {
      hydrated.board.collapsedSpaces = new Map(Object.entries(hydrated.board.collapsedSpaces));
    }

    // Hydrate territories
    if (Array.isArray(hydrated.board.territories)) {
      hydrated.board.territories = new Map(hydrated.board.territories);
    } else if (
      typeof hydrated.board.territories === 'object' &&
      hydrated.board.territories !== null
    ) {
      hydrated.board.territories = new Map(Object.entries(hydrated.board.territories));
    }
  }

  // Hydrate dates if needed
  if (typeof hydrated.createdAt === 'string') hydrated.createdAt = new Date(hydrated.createdAt);
  if (typeof hydrated.lastMoveAt === 'string') hydrated.lastMoveAt = new Date(hydrated.lastMoveAt);

  return hydrated as SharedGameState;
}

interface TraceStepExpected {
  tsValid?: boolean;
  tsStateHash?: string;
  tsS?: number;
}

interface TraceStep {
  label?: string;
  move: Move;
  expected?: TraceStepExpected;
  stateHash?: string;
  sInvariant?: number;
}

interface TraceFixture {
  version: 'v1';
  boardType: BoardType;
  initialState: SharedGameState;
  steps: TraceStep[];
}

interface LoadedTraceFixture {
  name: string;
  fixture: TraceFixture;
}

function loadTraceFixtures(): LoadedTraceFixture[] {
  const fixturesDir = join(__dirname, '..', 'fixtures', 'rules-parity', 'v1');
  let entries: string[];
  try {
    entries = readdirSync(fixturesDir);
  } catch {
    return [];
  }
  return entries
    .filter((name) => name.startsWith('trace.') && name.endsWith('.json'))
    .map((name) => {
      const fullPath = join(fixturesDir, name);
      const raw = readFileSync(fullPath, 'utf8');
      const fixture = JSON.parse(raw) as TraceFixture;
      return { name, fixture };
    });
}

describe('Trace fixtures shared-engine self-consistency', () => {
  const loaded = loadTraceFixtures();

  if (loaded.length === 0) {
    it('has no trace fixtures to validate yet', () => {
      expect(loaded.length).toBe(0);
    });
    return;
  }

  for (const { name, fixture } of loaded) {
    it(`replays ${name} through shared GameEngine`, () => {
      const { boardType, initialState } = fixture;
      expect(boardType).toBeDefined();

      const hydratedState = hydrateGameState(initialState);
      const engine = new GameEngine(hydratedState);

      for (const [index, step] of fixture.steps.entries()) {
        const before = engine.getGameState();
        const action = moveToGameAction(step.move, before as any);
        const event = engine.processAction(action);

        if (event.type === 'ERROR_OCCURRED') {
          console.error(
            `[TraceFixtures] Error processing step ${index} in ${name}:`,
            JSON.stringify(event.payload, null, 2)
          );
        }
        expect(event.type).toBe('ACTION_PROCESSED');

        const afterState = engine.getGameState();
        const hash = hashGameState(afterState as any);
        const progress = computeProgressSnapshot(afterState as any);

        const expectedHash = step.expected?.tsStateHash ?? step.stateHash;
        const expectedS = step.expected?.tsS ?? step.sInvariant;

        const stepLabel = step.label ?? `step ${index}`;
        if (expectedHash) {
          expect(hash).toBe(expectedHash);
        }
        if (typeof expectedS === 'number') {
          expect(progress.S).toBe(expectedS);
        }

        // Keep engine advanced for next step.
        void stepLabel;
      }
    });
  }
});
