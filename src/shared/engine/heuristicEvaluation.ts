import type { GameState, Move, RingStack } from '../types/game';

/**
 * Minimal heuristic weight profile used by TypeScript fallback AI and sandbox.
 *
 * This is intentionally a strict subset of the Python HeuristicAI feature
 * space, focusing on cheap-to-compute, high-signal terms that are easy to
 * keep aligned across languages:
 *
 * - Stack control & distribution (number of stacks, effective heights).
 * - Simple territory advantage (players[].territorySpaces).
 * - Local vulnerability (taller enemy stacks adjacent to our stacks).
 */
export interface HeuristicWeights {
  /** Reward for controlling more stacks than the opponent. */
  stackControl: number;
  /** Reward for total effective stack height advantage. */
  stackHeight: number;
  /** Reward for territorySpaces advantage. */
  territory: number;
  /** Penalty for vulnerable stacks adjacent to taller enemy stacks. */
  vulnerability: number;
}

/**
 * v1 balanced profile, loosely aligned with the Python HeuristicAI defaults.
 * The absolute scale is less important than relative ratios; these values are
 * chosen to keep scores in a reasonable range for simple move ordering.
 */
export const HEURISTIC_WEIGHTS_V1_BALANCED: HeuristicWeights = {
  stackControl: 10.0,
  stackHeight: 5.0,
  territory: 8.0,
  vulnerability: 8.0,
};

/**
 * Lightweight persona ids for TS-side heuristic profiles.
 *
 * These mirror the Python ids in app.ai.heuristic_weights. Ladder-linked
 * ids (v1-heuristic-2/3/4/5) currently all point at the balanced profile
 * but are included so difficulty → persona mappings can stay in sync.
 */
export type HeuristicPersonaId =
  | 'heuristic_v1_balanced'
  | 'heuristic_v1_aggressive'
  | 'heuristic_v1_territorial'
  | 'heuristic_v1_defensive'
  | 'v1-heuristic-2'
  | 'v1-heuristic-3'
  | 'v1-heuristic-4'
  | 'v1-heuristic-5';

/**
 * v1 aggressive persona – slightly more willing to accept vulnerability in
 * exchange for additional stack presence and height.
 */
export const HEURISTIC_WEIGHTS_V1_AGGRESSIVE: HeuristicWeights = {
  stackControl: 12.0,
  stackHeight: 6.0,
  territory: 7.0,
  vulnerability: 6.0,
};

/**
 * v1 territorial persona – emphasises territory advantage and line-friendly
 * stack distribution a bit more than raw height.
 */
export const HEURISTIC_WEIGHTS_V1_TERRITORIAL: HeuristicWeights = {
  stackControl: 9.0,
  stackHeight: 4.0,
  territory: 10.0,
  vulnerability: 8.0,
};

/**
 * v1 defensive persona – prioritises safety and reducing vulnerability.
 */
export const HEURISTIC_WEIGHTS_V1_DEFENSIVE: HeuristicWeights = {
  stackControl: 11.0,
  stackHeight: 5.0,
  territory: 8.0,
  vulnerability: 10.0,
};

/**
 * TS-side registry mirroring the Python HEURISTIC_WEIGHT_PROFILES mapping.
 *
 * Only the subset of weights used by evaluateHeuristicState is represented
 * here, but the profile ids and relative tendencies are kept aligned with
 * the Python definitions so that cross-language fixtures can share ids.
 */
export const HEURISTIC_WEIGHT_PROFILES_TS: Record<HeuristicPersonaId, HeuristicWeights> = {
  heuristic_v1_balanced: HEURISTIC_WEIGHTS_V1_BALANCED,
  heuristic_v1_aggressive: HEURISTIC_WEIGHTS_V1_AGGRESSIVE,
  heuristic_v1_territorial: HEURISTIC_WEIGHTS_V1_TERRITORIAL,
  heuristic_v1_defensive: HEURISTIC_WEIGHTS_V1_DEFENSIVE,
  // Ladder-linked ids – currently all reference the balanced profile but
  // may be re-pointed in future without changing external difficulty
  // contracts.
  'v1-heuristic-2': HEURISTIC_WEIGHTS_V1_BALANCED,
  'v1-heuristic-3': HEURISTIC_WEIGHTS_V1_BALANCED,
  'v1-heuristic-4': HEURISTIC_WEIGHTS_V1_BALANCED,
  'v1-heuristic-5': HEURISTIC_WEIGHTS_V1_BALANCED,
};

/**
 * Resolve a TS-side heuristic weight profile from a persona id, falling
 * back to the balanced profile when unknown or omitted.
 */
export function getHeuristicWeightsTS(profileId?: string | null): HeuristicWeights {
  if (!profileId) return HEURISTIC_WEIGHTS_V1_BALANCED;
  const weights = HEURISTIC_WEIGHT_PROFILES_TS[profileId as HeuristicPersonaId];
  return weights ?? HEURISTIC_WEIGHTS_V1_BALANCED;
}

/** Simple, cheap board-evaluation heuristic for fallback AI use. */
export function evaluateHeuristicState(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights = HEURISTIC_WEIGHTS_V1_BALANCED
): number {
  // Terminal handling: mirror the Python heuristic convention where
  // finished games dominate all other considerations.
  if (state.gameStatus === 'finished') {
    // eslint-disable-line @typescript-eslint/no-unnecessary-boolean-literal-compare
    if (state.winner === playerNumber) return 100_000;
    if (state.winner !== undefined && state.winner !== null) return -100_000;
    return 0;
  }

  let score = 0;

  // --- Stack control & height ---
  let myStacks = 0;
  let oppStacks = 0;
  let myHeight = 0;
  let oppHeight = 0;

  state.board.stacks.forEach((stack: RingStack) => {
    const hRaw = stack.stackHeight || stack.rings.length || 0;
    // Diminishing returns for very tall stacks to discourage all-in towers.
    const hEff = hRaw <= 5 ? hRaw : 5 + (hRaw - 5) * 0.1;

    if (stack.controllingPlayer === playerNumber) {
      myStacks += 1;
      myHeight += hEff;
    } else {
      oppStacks += 1;
      oppHeight += hEff;
    }
  });

  if (myStacks === 0) {
    score -= 50; // Catastrophic: no stacks on board.
  } else if (myStacks === 1) {
    score -= 10; // Single-stack positions are fragile.
  } else {
    score += myStacks * 2; // Diversification bonus.
  }

  score += (myStacks - oppStacks) * weights.stackControl;
  score += (myHeight - oppHeight) * weights.stackHeight;

  // --- Simple territory advantage ---
  const me = state.players.find((p) => p.playerNumber === playerNumber);
  if (me) {
    let oppTerritoryMax = 0;
    for (const p of state.players) {
      if (p.playerNumber === playerNumber) continue;
      if (p.territorySpaces > oppTerritoryMax) {
        oppTerritoryMax = p.territorySpaces;
      }
    }
    const terrDelta = me.territorySpaces - oppTerritoryMax;
    score += terrDelta * weights.territory;
  }

  // --- Local vulnerability (adjacent only; no full line-of-sight) ---
  // For each of our stacks, look at immediate neighbours; if an enemy stack
  // is strictly taller, penalise the height difference.
  const directions = [
    { dx: 1, dy: 0 },
    { dx: -1, dy: 0 },
    { dx: 0, dy: 1 },
    { dx: 0, dy: -1 },
    { dx: 1, dy: 1 },
    { dx: -1, dy: -1 },
  ];

  let vulnerabilityRaw = 0;

  state.board.stacks.forEach((stack: RingStack) => {
    if (stack.controllingPlayer !== playerNumber) return;

    const baseH = stack.stackHeight || stack.rings.length || 0;
    const { position } = stack;

    for (const dir of directions) {
      const nx = position.x + dir.dx;
      const ny = position.y + dir.dy;
      const nKey = `${nx},${ny}${position.z !== undefined ? `,${position.z}` : ''}`;
      const neighbour = state.board.stacks.get(nKey as any);
      if (!neighbour) continue;
      if (neighbour.controllingPlayer === playerNumber) continue;

      const nh = neighbour.stackHeight || neighbour.rings.length || 0;
      if (nh > baseH) {
        vulnerabilityRaw += nh - baseH;
      }
    }
  });

  if (vulnerabilityRaw !== 0) {
    score -= vulnerabilityRaw * weights.vulnerability;
  }

  return score;
}

/**
 * Parameters for move-scoring once shared mutators are available.
 *
 * This is a forward-looking API only – callers should not rely on it yet.
 */
export interface MoveScoringParams {
  /** Game state before applying the move. */
  before: GameState;
  /** Canonical move to evaluate. */
  move: Move;
  /** Player for whom the score is being computed. */
  playerNumber: number;
  /** Optional persona weights; defaults to the balanced profile. */
  weights?: HeuristicWeights;
}

/**
 * Design-time stub for future move-based heuristic scoring.
 *
 * Once the shared mutators in movementApplication.ts and placementHelpers.ts
 * are fully implemented, this helper will:
 *
 * - apply `params.move` to a cloned `before` state via the shared mutators;
 * - evaluate both before/after states with {@link evaluateHeuristicState}; and
 * - return the delta (after - before) as a move score.
 *
 * For now it intentionally throws so it cannot be used in production code
 * paths; tests and hosts should treat it as a placeholder API surface only.
 */
export function scoreMove(_params: MoveScoringParams): number {
  throw new Error(
    'TODO(heuristic-move-scoring): scoreMove is a design-time stub. ' +
      'It will be wired to shared movement/placement mutators once they are implemented.'
  );
}
