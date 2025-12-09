import fs from 'fs';
import path from 'path';

import {
  importVectorBundle,
  validateAgainstAssertions,
  type ContractTestVector,
} from '../../src/shared/engine/contracts';
import { deserializeGameState } from '../../src/shared/engine/contracts/serialization';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState, Move, Position } from '../../src/shared/types/game';

type PhaseTransitionHint = {
  phaseAfter?: string;
  availableChainTarget?: Position;
  expectedLandingPosition?: Position;
};

const MULTI_PHASE_BUNDLE = path.resolve(
  __dirname,
  '../fixtures/contract-vectors/v2/multi_phase_turn.vectors.json'
);

function loadMultiPhaseVectors(): ContractTestVector[] {
  const json = fs.readFileSync(MULTI_PHASE_BUNDLE, 'utf8');
  const bundle = importVectorBundle(json);
  return bundle;
}

function positionsEqual(a?: Position, b?: Position): boolean {
  return !!a && !!b && a.x === b.x && a.y === b.y;
}

function convertVectorMove(vectorMove: any): Move {
  const move: any = { ...vectorMove };
  move.timestamp = move.timestamp ? new Date(move.timestamp) : new Date();
  move.thinkTime = move.thinkTime ?? 0;
  return move as Move;
}

function makeBookkeepingMove(decisionType: string, player: number): Move | null {
  const base = {
    player,
    timestamp: new Date(),
    thinkTime: 0,
    position: { x: 0, y: 0 },
  } as any;

  switch (decisionType) {
    case 'no_line_action_required':
      return { ...base, type: 'no_line_action' } as Move;
    case 'no_territory_action_required':
      return { ...base, type: 'no_territory_action' } as Move;
    case 'no_movement_action_required':
      return { ...base, type: 'skip_capture' } as Move;
    case 'no_placement_action_required':
      return { ...base, type: 'no_placement_action' } as Move;
    default:
      return null;
  }
}

function pickDecisionMove(
  decision: NonNullable<ReturnType<typeof processTurn>['pendingDecision']>,
  state: GameState,
  phaseHints: PhaseTransitionHint[]
): Move {
  const options = decision.options ?? [];
  if (options.length === 0) {
    const synthetic = makeBookkeepingMove(decision.type, decision.player);
    if (!synthetic) {
      throw new Error(`No options available for decision type ${decision.type}`);
    }
    return synthetic;
  }

  const hint = phaseHints.find((h) => h.phaseAfter === state.currentPhase);

  switch (decision.type) {
    case 'chain_capture': {
      if (hint?.availableChainTarget) {
        const targeted = options.find((opt: any) =>
          positionsEqual(opt.to, hint.availableChainTarget)
        );
        if (targeted) return targeted;
      }
      break;
    }
    case 'line_reward': {
      // Prefer the option that does not collapse markers when available.
      const zeroCollapse = options.find((opt: any) => (opt.collapsedMarkers ?? []).length === 0);
      if (zeroCollapse) return zeroCollapse;

      // Otherwise choose the smallest collapse set to align with FAQ/Q20-style vectors.
      const sorted = [...options].sort((a: any, b: any) => {
        const aLen = (a.collapsedMarkers ?? []).length;
        const bLen = (b.collapsedMarkers ?? []).length;
        return aLen - bLen;
      });
      return sorted[0];
    }
    case 'region_order': {
      if (hint?.availableChainTarget) {
        const matching = options.find((opt: any) =>
          (opt.disconnectedRegions ?? []).some((reg: any) =>
            reg.spaces?.some((pos: Position) => positionsEqual(pos, hint.availableChainTarget))
          )
        );
        if (matching) return matching;
      }
      const sortedBySize = [...options].sort((a: any, b: any) => {
        const aSize = (a.disconnectedRegions?.[0]?.spaces ?? []).length;
        const bSize = (b.disconnectedRegions?.[0]?.spaces ?? []).length;
        return aSize - bSize;
      });
      return sortedBySize[0];
    }
    case 'line_order': {
      // Prefer deterministic ordering by formed line key
      const sorted = [...options].sort((a: any, b: any) => {
        const aKey = JSON.stringify(a.formedLines ?? []);
        const bKey = JSON.stringify(b.formedLines ?? []);
        return aKey.localeCompare(bKey);
      });
      return sorted[0];
    }
    default:
      break;
  }

  return options[0];
}

function driveInitialMoveVector(vector: ContractTestVector) {
  const phaseHints = ((vector.input as any).phaseTransitions ?? []) as PhaseTransitionHint[];
  const territoryRegion = (vector.input as any).territoryExpectation
    ?.potentiallyDisconnectedRegion?.[0];
  if (territoryRegion) {
    phaseHints.push({ phaseAfter: 'territory_processing', availableChainTarget: territoryRegion });
  }
  const state = deserializeGameState((vector.input as any).state);
  const initialMove = convertVectorMove((vector.input as any).initialMove);

  const phases: string[] = [];
  const decisionTypes: Set<string> = new Set();

  let result = processTurn(state, initialMove);
  phases.push(...result.metadata.phasesTraversed);
  let currentState = result.nextState;

  while (result.status === 'awaiting_decision' && result.pendingDecision) {
    decisionTypes.add(result.pendingDecision.type);
    const chosen = pickDecisionMove(result.pendingDecision, currentState, phaseHints);
    result = processTurn(currentState, chosen);
    phases.push(...result.metadata.phasesTraversed);
    currentState = result.nextState;
  }

  return { result, phases, decisionTypes };
}

function resolvePendingDecisions(
  initialResult: ReturnType<typeof processTurn>,
  phaseHints: PhaseTransitionHint[] = []
) {
  let result = initialResult;
  let currentState = result.nextState;
  const phases: string[] = [...result.metadata.phasesTraversed];

  while (result.status === 'awaiting_decision' && result.pendingDecision) {
    const chosen = pickDecisionMove(result.pendingDecision, currentState, phaseHints);
    result = processTurn(currentState, chosen);
    phases.push(...result.metadata.phasesTraversed);
    currentState = result.nextState;
  }

  return { result, phases };
}

describe('Multi-phase turn contract vectors (line → territory across boards)', () => {
  const vectors = loadMultiPhaseVectors();

  it('executes initial-move multi-phase sequences through chain → line → territory for all boards', () => {
    const initialMoveVectors = vectors.filter((v: any) => v.input?.initialMove);
    expect(initialMoveVectors.length).toBeGreaterThan(0);

    for (const vector of initialMoveVectors) {
      const { result, phases, decisionTypes } = driveInitialMoveVector(vector);

      expect(result.status).toBe('complete');

      const validation = validateAgainstAssertions(
        result.nextState,
        vector.expectedOutput.assertions
      );
      if (!validation.valid) {
        throw new Error(`Vector ${vector.id} failed assertions: ${validation.failures.join('; ')}`);
      }

      const expectedPhases = (vector.input as any).expectedPhaseSequence ?? [];
      for (const phase of expectedPhases) {
        expect(phases).toContain(phase);
      }

      expect(phases).toContain('territory_processing');
    }
  });

  it('runs multi-region line → territory sequences across square8/19/hex using recorded moves', () => {
    const multiRegionSequences = vectors.filter((v) =>
      (v.tags ?? []).some((t) => t.startsWith('sequence:turn.line_then_territory.multi_region'))
    );
    const sequencesById = new Map<string, ContractTestVector[]>();

    for (const vector of multiRegionSequences) {
      const seqTag = (vector.tags ?? []).find((t) =>
        t.startsWith('sequence:turn.line_then_territory.multi_region')
      );
      if (!seqTag) continue;
      const seqId = seqTag.replace('sequence:', '');
      const arr = sequencesById.get(seqId) ?? [];
      arr.push(vector);
      sequencesById.set(seqId, arr);
    }

    expect(sequencesById.size).toBeGreaterThan(0);

    for (const [seqId, seqVectors] of sequencesById.entries()) {
      const ordered = seqVectors.sort((a, b) => a.id.localeCompare(b.id));

      ordered.forEach((vector, idx) => {
        const startState = deserializeGameState((vector.input as any).state);
        const move = convertVectorMove((vector.input as any).move);
        const initialResult = processTurn(startState, move);
        const { result } = resolvePendingDecisions(initialResult);
        expect(result.status).toBe(vector.expectedOutput.status);

        const validation = validateAgainstAssertions(
          result.nextState,
          vector.expectedOutput.assertions
        );
        if (!validation.valid) {
          throw new Error(
            `Sequence ${seqId}, step ${idx} (${vector.id}) failed assertions: ${validation.failures.join(
              '; '
            )}`
          );
        }
      });
    }
  });
});
