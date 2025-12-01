import type {
  GameState,
  Move,
  RingStack,
  Position,
  BoardType,
  MarkerInfo,
  BoardState,
} from '../types/game';
import { getMovementDirectionsForBoardType, Direction } from './core';

/**
 * Complete heuristic weight profile matching Python HeuristicAI.
 *
 * This interface contains all 18 weight constants from the Python
 * implementation in ai-service/app/ai/heuristic_weights.py:
 *
 * - Stack control & height (WEIGHT_STACK_CONTROL, WEIGHT_STACK_HEIGHT)
 * - Territory (WEIGHT_TERRITORY)
 * - Rings in hand (WEIGHT_RINGS_IN_HAND)
 * - Center control (WEIGHT_CENTER_CONTROL)
 * - Adjacency (WEIGHT_ADJACENCY) - influence/proximity bonus
 * - Opponent threat (WEIGHT_OPPONENT_THREAT)
 * - Mobility (WEIGHT_MOBILITY)
 * - Eliminated rings (WEIGHT_ELIMINATED_RINGS)
 * - Line potential (WEIGHT_LINE_POTENTIAL)
 * - Victory proximity (WEIGHT_VICTORY_PROXIMITY)
 * - Marker count (WEIGHT_MARKER_COUNT)
 * - Vulnerability (WEIGHT_VULNERABILITY) - line-of-sight taller enemy stacks
 * - Overtake potential (WEIGHT_OVERTAKE_POTENTIAL)
 * - Territory closure (WEIGHT_TERRITORY_CLOSURE)
 * - Line connectivity (WEIGHT_LINE_CONNECTIVITY)
 * - Territory safety (WEIGHT_TERRITORY_SAFETY)
 * - Stack mobility (WEIGHT_STACK_MOBILITY)
 */
export interface HeuristicWeights {
  /** Reward for controlling more stacks than the opponent. */
  stackControl: number;
  /** Reward for total effective stack height advantage. */
  stackHeight: number;
  /** Reward for territorySpaces advantage. */
  territory: number;
  /** Reward for rings remaining in hand (more placement options). */
  ringsInHand: number;
  /** Reward for controlling center positions. */
  centerControl: number;
  /** Reward for board influence/adjacency. */
  adjacency: number;
  /** Penalty for opponent threats (adjacent taller stacks). */
  opponentThreat: number;
  /** Reward for mobility (available moves). */
  mobility: number;
  /** Reward for eliminated rings (towards victory). */
  eliminatedRings: number;
  /** Reward for line potential (2/3/4 markers in a row). */
  linePotential: number;
  /** Reward for proximity to victory conditions. */
  victoryProximity: number;
  /** Reward for total marker count on board. */
  markerCount: number;
  /** Penalty for vulnerable stacks (line-of-sight taller enemy stacks). */
  vulnerability: number;
  /** Reward for overtake potential (line-of-sight shorter enemy stacks). */
  overtakePotential: number;
  /** Reward for territory closure potential (marker clustering). */
  territoryClosure: number;
  /** Reward for line connectivity between markers. */
  lineConnectivity: number;
  /** Penalty for opponent stacks near our markers. */
  territorySafety: number;
  /** Reward for stack mobility (per-stack movement freedom). */
  stackMobility: number;
  /** Penalty for opponents being much closer to victory than us. */
  opponentVictoryThreat: number;
  /** Penalty for having many stacks but few real actions (forced elimination risk). */
  forcedEliminationRisk: number;
  /** Bonus/penalty for last-player-standing action advantage. */
  lpsActionAdvantage: number;
  /** Penalty when a single opponent is far ahead of other opponents. */
  multiLeaderThreat: number;
}

/**
 * v1 balanced profile, aligned with Python BASE_V1_BALANCED_WEIGHTS.
 */
export const HEURISTIC_WEIGHTS_V1_BALANCED: HeuristicWeights = {
  stackControl: 10.0,
  stackHeight: 5.0,
  territory: 8.0,
  // Softer emphasis on rings in hand; progress comes from on-board play.
  ringsInHand: 1.0,
  centerControl: 4.0,
  // Adjacency/influence is currently disabled on both TS and Python sides.
  // The weight remains in the schema for future influence heuristics but
  // no evaluation term consumes it.
  adjacency: 0.0,
  opponentThreat: 6.0,
  mobility: 4.0,
  eliminatedRings: 12.0,
  linePotential: 7.0,
  victoryProximity: 20.0,
  // De-emphasise raw marker density; structure comes from territory/lines.
  markerCount: 0.5,
  vulnerability: 8.0,
  overtakePotential: 8.0,
  // Reduce double-counting of structure; markers + connectivity already help.
  territoryClosure: 7.0,
  lineConnectivity: 4.0,
  territorySafety: 5.0,
  stackMobility: 4.0,
  opponentVictoryThreat: 6.0,
  forcedEliminationRisk: 4.0,
  lpsActionAdvantage: 2.0,
  multiLeaderThreat: 2.0,
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
 * v1 aggressive persona – emphasises elimination and overtake potential,
 * and slightly downweights safety concerns.
 *
 * Persona deltas must match ai-service/app/ai/heuristic_weights.py;
 * update both files together.
 */
export const HEURISTIC_WEIGHTS_V1_AGGRESSIVE: HeuristicWeights = {
  ...HEURISTIC_WEIGHTS_V1_BALANCED,
  eliminatedRings: HEURISTIC_WEIGHTS_V1_BALANCED.eliminatedRings * 1.25,
  overtakePotential: HEURISTIC_WEIGHTS_V1_BALANCED.overtakePotential * 1.25,
  victoryProximity: HEURISTIC_WEIGHTS_V1_BALANCED.victoryProximity * 1.15,
  linePotential: HEURISTIC_WEIGHTS_V1_BALANCED.linePotential * 1.1,
  vulnerability: HEURISTIC_WEIGHTS_V1_BALANCED.vulnerability * 0.85,
  territorySafety: HEURISTIC_WEIGHTS_V1_BALANCED.territorySafety * 0.85,
  opponentVictoryThreat: HEURISTIC_WEIGHTS_V1_BALANCED.opponentVictoryThreat * 1.1,
  forcedEliminationRisk: HEURISTIC_WEIGHTS_V1_BALANCED.forcedEliminationRisk * 0.85,
  lpsActionAdvantage: HEURISTIC_WEIGHTS_V1_BALANCED.lpsActionAdvantage * 1.0,
  multiLeaderThreat: HEURISTIC_WEIGHTS_V1_BALANCED.multiLeaderThreat * 0.8,
};

/**
 * v1 territorial persona – emphasises territory, marker structure, and
 * safety over pure elimination.
 *
 * Persona deltas must match ai-service/app/ai/heuristic_weights.py;
 * update both files together.
 */
export const HEURISTIC_WEIGHTS_V1_TERRITORIAL: HeuristicWeights = {
  ...HEURISTIC_WEIGHTS_V1_BALANCED,
  territory: HEURISTIC_WEIGHTS_V1_BALANCED.territory * 1.25,
  territoryClosure: HEURISTIC_WEIGHTS_V1_BALANCED.territoryClosure * 1.25,
  territorySafety: HEURISTIC_WEIGHTS_V1_BALANCED.territorySafety * 1.2,
  linePotential: HEURISTIC_WEIGHTS_V1_BALANCED.linePotential * 1.15,
  lineConnectivity: HEURISTIC_WEIGHTS_V1_BALANCED.lineConnectivity * 1.15,
  markerCount: HEURISTIC_WEIGHTS_V1_BALANCED.markerCount * 1.1,
  eliminatedRings: HEURISTIC_WEIGHTS_V1_BALANCED.eliminatedRings * 0.9,
  opponentVictoryThreat: HEURISTIC_WEIGHTS_V1_BALANCED.opponentVictoryThreat * 1.15,
  forcedEliminationRisk: HEURISTIC_WEIGHTS_V1_BALANCED.forcedEliminationRisk * 1.0,
  lpsActionAdvantage: HEURISTIC_WEIGHTS_V1_BALANCED.lpsActionAdvantage * 1.1,
  multiLeaderThreat: HEURISTIC_WEIGHTS_V1_BALANCED.multiLeaderThreat * 1.25,
};

/**
 * v1 defensive persona – prioritises stack safety, mobility and
 * vulnerability awareness, and tones down elimination eagerness.
 *
 * Persona deltas must match ai-service/app/ai/heuristic_weights.py;
 * update both files together.
 */
export const HEURISTIC_WEIGHTS_V1_DEFENSIVE: HeuristicWeights = {
  ...HEURISTIC_WEIGHTS_V1_BALANCED,
  stackControl: HEURISTIC_WEIGHTS_V1_BALANCED.stackControl * 1.15,
  stackMobility: HEURISTIC_WEIGHTS_V1_BALANCED.stackMobility * 1.2,
  mobility: HEURISTIC_WEIGHTS_V1_BALANCED.mobility * 1.1,
  vulnerability: HEURISTIC_WEIGHTS_V1_BALANCED.vulnerability * 1.25,
  territorySafety: HEURISTIC_WEIGHTS_V1_BALANCED.territorySafety * 1.15,
  overtakePotential: HEURISTIC_WEIGHTS_V1_BALANCED.overtakePotential * 0.9,
  eliminatedRings: HEURISTIC_WEIGHTS_V1_BALANCED.eliminatedRings * 0.9,
  opponentVictoryThreat: HEURISTIC_WEIGHTS_V1_BALANCED.opponentVictoryThreat * 1.05,
  forcedEliminationRisk: HEURISTIC_WEIGHTS_V1_BALANCED.forcedEliminationRisk * 1.3,
  lpsActionAdvantage: HEURISTIC_WEIGHTS_V1_BALANCED.lpsActionAdvantage * 1.2,
  multiLeaderThreat: HEURISTIC_WEIGHTS_V1_BALANCED.multiLeaderThreat * 1.1,
};

/**
 * TS-side registry mirroring the Python HEURISTIC_WEIGHT_PROFILES mapping.
 */
export const HEURISTIC_WEIGHT_PROFILES_TS: Record<HeuristicPersonaId, HeuristicWeights> = {
  heuristic_v1_balanced: HEURISTIC_WEIGHTS_V1_BALANCED,
  heuristic_v1_aggressive: HEURISTIC_WEIGHTS_V1_AGGRESSIVE,
  heuristic_v1_territorial: HEURISTIC_WEIGHTS_V1_TERRITORIAL,
  heuristic_v1_defensive: HEURISTIC_WEIGHTS_V1_DEFENSIVE,
  // Ladder-linked ids – all reference balanced for now
  'v1-heuristic-2': HEURISTIC_WEIGHTS_V1_BALANCED,
  'v1-heuristic-3': HEURISTIC_WEIGHTS_V1_BALANCED,
  'v1-heuristic-4': HEURISTIC_WEIGHTS_V1_BALANCED,
  'v1-heuristic-5': HEURISTIC_WEIGHTS_V1_BALANCED,
};

/**
 * Mapping from Python UPPER_SNAKE_CASE weight keys to TypeScript camelCase keys.
 * Used for JSON weight synchronization.
 */
const PYTHON_TO_TS_KEY_MAP: Record<string, keyof HeuristicWeights> = {
  WEIGHT_STACK_CONTROL: 'stackControl',
  WEIGHT_STACK_HEIGHT: 'stackHeight',
  WEIGHT_TERRITORY: 'territory',
  WEIGHT_RINGS_IN_HAND: 'ringsInHand',
  WEIGHT_CENTER_CONTROL: 'centerControl',
  WEIGHT_ADJACENCY: 'adjacency',
  WEIGHT_OPPONENT_THREAT: 'opponentThreat',
  WEIGHT_MOBILITY: 'mobility',
  WEIGHT_ELIMINATED_RINGS: 'eliminatedRings',
  WEIGHT_LINE_POTENTIAL: 'linePotential',
  WEIGHT_VICTORY_PROXIMITY: 'victoryProximity',
  WEIGHT_MARKER_COUNT: 'markerCount',
  WEIGHT_VULNERABILITY: 'vulnerability',
  WEIGHT_OVERTAKE_POTENTIAL: 'overtakePotential',
  WEIGHT_TERRITORY_CLOSURE: 'territoryClosure',
  WEIGHT_LINE_CONNECTIVITY: 'lineConnectivity',
  WEIGHT_TERRITORY_SAFETY: 'territorySafety',
  WEIGHT_STACK_MOBILITY: 'stackMobility',
  WEIGHT_OPPONENT_VICTORY_THREAT: 'opponentVictoryThreat',
  WEIGHT_FORCED_ELIMINATION_RISK: 'forcedEliminationRisk',
  WEIGHT_LPS_ACTION_ADVANTAGE: 'lpsActionAdvantage',
  WEIGHT_MULTI_LEADER_THREAT: 'multiLeaderThreat',
};

/**
 * Load heuristic weights from a JSON object (e.g., exported from Python).
 * Supports both Python UPPER_SNAKE_CASE keys and TypeScript camelCase keys.
 * Unknown keys are ignored. Missing keys fall back to balanced profile defaults.
 */
export function loadHeuristicWeightsFromJSON(json: Record<string, number>): HeuristicWeights {
  const result = { ...HEURISTIC_WEIGHTS_V1_BALANCED };

  for (const [key, value] of Object.entries(json)) {
    // Try TypeScript key directly
    if (key in result) {
      (result as Record<string, number>)[key] = value;
      continue;
    }
    // Try Python key mapping
    const tsKey = PYTHON_TO_TS_KEY_MAP[key];
    if (tsKey) {
      result[tsKey] = value;
    }
  }

  return result;
}

/**
 * Export heuristic weights to a JSON object using Python UPPER_SNAKE_CASE keys.
 * This allows TypeScript-tuned weights to be used by the Python AI service.
 */
export function exportHeuristicWeightsToJSON(weights: HeuristicWeights): Record<string, number> {
  const result: Record<string, number> = {};

  for (const [pythonKey, tsKey] of Object.entries(PYTHON_TO_TS_KEY_MAP)) {
    result[pythonKey] = weights[tsKey];
  }

  return result;
}

/**
 * Resolve a TS-side heuristic weight profile from a persona id, falling
 * back to the balanced profile when unknown or omitted.
 */
export function getHeuristicWeightsTS(profileId?: string | null): HeuristicWeights {
  if (!profileId) return HEURISTIC_WEIGHTS_V1_BALANCED;
  const weights = HEURISTIC_WEIGHT_PROFILES_TS[profileId as HeuristicPersonaId];
  return weights ?? HEURISTIC_WEIGHTS_V1_BALANCED;
}

// ============================================================================
// Geometry Helpers
// ============================================================================

/**
 * Get a position key string for map lookups.
 */
function positionKey(pos: Position): string {
  if (pos.z !== undefined) {
    return `${pos.x},${pos.y},${pos.z}`;
  }
  return `${pos.x},${pos.y}`;
}

/**
 * Add a direction to a position.
 */
function addDirection(pos: Position, dir: Direction, steps: number = 1): Position {
  const result: Position = {
    x: pos.x + dir.x * steps,
    y: pos.y + dir.y * steps,
  };
  if (pos.z !== undefined && dir.z !== undefined) {
    result.z = pos.z + dir.z * steps;
  }
  return result;
}

/**
 * Get center positions for a board (simplified heuristic).
 * Returns position keys for positions considered "center" of the board.
 */
function getCenterPositions(boardType: BoardType, size: number): Set<string> {
  const center = new Set<string>();
  const mid = Math.floor(size / 2);

  if (boardType === 'hexagonal') {
    // For hexagonal boards, center is around (0, 0, 0) in cube coords
    // and a few adjacent cells
    center.add('0,0,0');
    const hexDirs = getMovementDirectionsForBoardType('hexagonal');
    for (const dir of hexDirs) {
      const pos: Position = { x: dir.x, y: dir.y };
      if (dir.z !== undefined) {
        pos.z = dir.z;
      }
      center.add(positionKey(pos));
    }
  } else {
    // For square boards, center is the middle 3x3 region
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        center.add(`${mid + dx},${mid + dy}`);
      }
    }
  }

  return center;
}

/**
 * Check if a position is within bounds (simplified check).
 */
function isWithinBounds(pos: Position, boardType: BoardType, size: number): boolean {
  if (boardType === 'hexagonal') {
    // For hexagonal boards with cube coords, check sum and range
    const z = pos.z ?? 0;
    if (pos.x + pos.y + z !== 0) return false;
    const radius = Math.floor(size / 2);
    return Math.abs(pos.x) <= radius && Math.abs(pos.y) <= radius && Math.abs(z) <= radius;
  } else {
    // Square board bounds check
    return pos.x >= 0 && pos.x < size && pos.y >= 0 && pos.y < size;
  }
}

/**
 * Get adjacent positions around a position.
 */
function getAdjacentPositions(pos: Position, boardType: BoardType, size: number): Position[] {
  const directions = getMovementDirectionsForBoardType(boardType);
  const adjacent: Position[] = [];

  for (const dir of directions) {
    const newPos = addDirection(pos, dir, 1);
    if (isWithinBounds(newPos, boardType, size)) {
      adjacent.push(newPos);
    }
  }

  return adjacent;
}

/**
 * Iterate over all logical board coordinates for a given board.
 * Mirrors the helper used in victory logic / aggregates.
 */
function forEachBoardPosition(board: BoardState, fn: (pos: Position) => void): void {
  const size = board.size;

  if (board.type === 'hexagonal') {
    const radius = size - 1;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        fn({ x: q, y: r, z: s });
      }
    }
  } else {
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        fn({ x, y });
      }
    }
  }
}

/**
 * Get visible stacks along lines of sight from a position.
 * Returns the first stack encountered in each direction.
 */
function getVisibleStacks(pos: Position, state: GameState, maxDistance: number = 20): RingStack[] {
  const visible: RingStack[] = [];
  const directions = getMovementDirectionsForBoardType(state.boardType);
  const boardSize = state.board.size;

  for (const dir of directions) {
    let currPos = { ...pos };

    for (let step = 1; step <= maxDistance; step++) {
      currPos = addDirection(pos, dir, step);

      if (!isWithinBounds(currPos, state.boardType, boardSize)) {
        break;
      }

      const key = positionKey(currPos);

      // Check for collapsed spaces (blocked)
      if (state.board.collapsedSpaces.has(key)) {
        break;
      }

      // Check for stacks
      const stack = state.board.stacks.get(key);
      if (stack) {
        visible.push(stack);
        break; // First visible stack in this direction
      }
    }
  }

  return visible;
}

// ============================================================================
// Evaluation Functions
// ============================================================================

/**
 * Evaluate stack control and height (matches Python _evaluate_stack_control).
 */
function evaluateStackControl(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;
  let myStacks = 0;
  let oppStacks = 0;
  let myHeight = 0;
  let oppHeight = 0;

  state.board.stacks.forEach((stack: RingStack) => {
    const hRaw = stack.stackHeight || stack.rings.length || 0;
    // Diminishing returns for height > 5 (Python: h if h <= 5 else 5 + (h - 5) * 0.1)
    const hEff = hRaw <= 5 ? hRaw : 5 + (hRaw - 5) * 0.1;

    if (stack.controllingPlayer === playerNumber) {
      myStacks += 1;
      myHeight += hEff;
    } else {
      oppStacks += 1;
      oppHeight += hEff;
    }
  });

  // Risk diversification penalty/bonus (Python logic)
  if (myStacks === 0) {
    score -= 50.0; // Huge penalty for no stacks
  } else if (myStacks === 1) {
    score -= 10.0; // Penalty for single stack (vulnerable)
  } else {
    score += myStacks * 2.0; // Diversification bonus
  }

  score += (myStacks - oppStacks) * weights.stackControl;
  score += (myHeight - oppHeight) * weights.stackHeight;

  return score;
}

/**
 * Evaluate territory control (matches Python _evaluate_territory).
 */
function evaluateTerritory(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const me = state.players.find((p) => p.playerNumber === playerNumber);
  if (!me) return 0;

  const myTerritory = me.territorySpaces;

  // Compare with opponents (max opponent territory)
  let oppTerritoryMax = 0;
  for (const p of state.players) {
    if (p.playerNumber !== playerNumber) {
      oppTerritoryMax = Math.max(oppTerritoryMax, p.territorySpaces);
    }
  }

  return (myTerritory - oppTerritoryMax) * weights.territory;
}

/**
 * Evaluate rings in hand (matches Python _evaluate_rings_in_hand).
 */
function evaluateRingsInHand(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const me = state.players.find((p) => p.playerNumber === playerNumber);
  if (!me) return 0;

  return me.ringsInHand * weights.ringsInHand;
}

/**
 * Evaluate center control (matches Python _evaluate_center_control).
 */
function evaluateCenterControl(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;
  const centerPositions = getCenterPositions(state.boardType, state.board.size);

  for (const posKey of centerPositions) {
    const stack = state.board.stacks.get(posKey);
    if (stack) {
      if (stack.controllingPlayer === playerNumber) {
        score += weights.centerControl;
      } else {
        score -= weights.centerControl * 0.5;
      }
    }
  }

  return score;
}

/**
 /**
  * Evaluate opponent threats (matches Python _evaluate_opponent_threats).
  */
function evaluateOpponentThreats(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;
  const boardSize = state.board.size;

  state.board.stacks.forEach((stack: RingStack) => {
    if (stack.controllingPlayer !== playerNumber) return;

    const adjacent = getAdjacentPositions(stack.position, state.boardType, boardSize);
    for (const adjPos of adjacent) {
      const adjStack = state.board.stacks.get(positionKey(adjPos));
      if (adjStack && adjStack.controllingPlayer !== playerNumber) {
        // Opponent stack adjacent to ours is a threat.
        // Capture power is based on cap height per compact rules §10.1,
        // so we compare using cap height here rather than total stack height.
        const myCap = stack.capHeight;
        const adjCap = adjStack.capHeight;
        const threatLevel = adjCap - myCap;
        score -= threatLevel * weights.opponentThreat * 0.5;
      }
    }
  });

  return score;
}
/**
 * Evaluate mobility using pseudo-mobility (matches Python _evaluate_mobility).
 */
function evaluateMobility(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const boardSize = state.board.size;

  // My pseudo-mobility
  let myMobility = 0;
  let oppMobility = 0;

  state.board.stacks.forEach((stack: RingStack) => {
    const isMyStack = stack.controllingPlayer === playerNumber;
    const stackCap = stack.capHeight;

    const adjacent = getAdjacentPositions(stack.position, state.boardType, boardSize);
    for (const adjPos of adjacent) {
      const adjKey = positionKey(adjPos);

      // Skip collapsed spaces
      if (state.board.collapsedSpaces.has(adjKey)) continue;

      const adjStack = state.board.stacks.get(adjKey);
      if (adjStack) {
        // Capture power is based on cap height per compact rules §10.1,
        // so we compare cap heights here instead of total heights.
        const adjCap = adjStack.capHeight;
        const isAdjMine = adjStack.controllingPlayer === playerNumber;

        if (isMyStack && !isAdjMine && stackCap >= adjCap) {
          myMobility += 1;
        } else if (!isMyStack && isAdjMine && stackCap >= adjCap) {
          oppMobility += 1;
        }
      } else {
        // Empty space - can move there
        if (isMyStack) {
          myMobility += 1;
        } else {
          oppMobility += 1;
        }
      }
    }
  });

  return (myMobility - oppMobility) * weights.mobility;
}

/**
 * Evaluate eliminated rings (matches Python _evaluate_eliminated_rings).
 */
function evaluateEliminatedRings(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const me = state.players.find((p) => p.playerNumber === playerNumber);
  if (!me) return 0;

  return me.eliminatedRings * weights.eliminatedRings;
}

/**
 * Evaluate line potential (matches Python _evaluate_line_potential).
 * Checks for 2/3/4 markers in a row.
 */
function evaluateLinePotential(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;
  const directions = getMovementDirectionsForBoardType(state.boardType);

  // Get all my markers
  const myMarkers: MarkerInfo[] = [];
  state.board.markers.forEach((marker: MarkerInfo) => {
    if (marker.player === playerNumber) {
      myMarkers.push(marker);
    }
  });

  for (const marker of myMarkers) {
    const startPos = marker.position;

    for (const direction of directions) {
      // Check length 2
      const pos2 = addDirection(startPos, direction, 1);
      const key2 = positionKey(pos2);
      const marker2 = state.board.markers.get(key2);

      if (marker2 && marker2.player === playerNumber) {
        score += 1.0; // 2 in a row

        // Check length 3
        const pos3 = addDirection(startPos, direction, 2);
        const key3 = positionKey(pos3);
        const marker3 = state.board.markers.get(key3);

        if (marker3 && marker3.player === playerNumber) {
          score += 2.0; // 3 in a row (cumulative with 2)

          // Check length 4 (almost a line)
          const pos4 = addDirection(startPos, direction, 3);
          const key4 = positionKey(pos4);
          const marker4 = state.board.markers.get(key4);

          if (marker4 && marker4.player === playerNumber) {
            score += 5.0; // 4 in a row
          }
        }
      }
    }
  }

  return score * weights.linePotential;
}

/**
 /**
  * Unweighted victory proximity base score for an arbitrary player.
  * Mirrors the Python _victory_proximity_base_for_player helper.
  */
function victoryProximityBaseForPlayer(state: GameState, playerNumber: number): number {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player) return 0;

  const ringsNeeded = state.victoryThreshold - player.eliminatedRings;
  const territoryNeeded = state.territoryVictoryThreshold - player.territorySpaces;

  if (ringsNeeded <= 0 || territoryNeeded <= 0) {
    return 1000.0;
  }

  let score = 0;
  score += (1.0 / Math.max(1, ringsNeeded)) * 50.0;
  score += (1.0 / Math.max(1, territoryNeeded)) * 50.0;
  return score;
}

/**
 * Evaluate victory proximity (matches Python _evaluate_victory_proximity).
 */
function evaluateVictoryProximity(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const base = victoryProximityBaseForPlayer(state, playerNumber);
  return base * weights.victoryProximity;
}

/**
 * Evaluate opponent victory threat – how much closer the leading opponent
 * is to victory than we are. Mirrors Python _evaluate_opponent_victory_threat.
 */
function evaluateOpponentVictoryThreat(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const selfProx = victoryProximityBaseForPlayer(state, playerNumber);

  let maxOppProx = 0;
  for (const p of state.players) {
    if (p.playerNumber === playerNumber) continue;
    const prox = victoryProximityBaseForPlayer(state, p.playerNumber);
    if (prox > maxOppProx) {
      maxOppProx = prox;
    }
  }

  const rawGap = maxOppProx - selfProx;
  const relativeThreat = Math.max(0, rawGap);

  return -relativeThreat * weights.opponentVictoryThreat;
}

/**
 * Approximate the number of "real" actions (moves + placements) available
 * to a given player.
 *
 * - Counts one move per stack that has at least one legal-looking move
 *   (empty neighbor or capturable enemy stack).
 * - Adds one additional action if the player has rings in hand and there
 *   exists at least one empty, non-collapsed space where a ring could be
 *   placed.
 */
function approxRealActionsForPlayer(state: GameState, playerNumber: number): number {
  const board = state.board;
  let approxMoves = 0;

  board.stacks.forEach((stack) => {
    if (stack.controllingPlayer !== playerNumber) return;

    const adj = getAdjacentPositions(stack.position, state.boardType, board.size);
    let stackHasAnyMove = false;

    for (const pos of adj) {
      const key = positionKey(pos);
      if (board.collapsedSpaces.has(key)) continue;

      const target = board.stacks.get(key);
      if (target) {
        if (target.controllingPlayer !== playerNumber && stack.capHeight >= target.capHeight) {
          stackHasAnyMove = true;
          break;
        }
      } else {
        stackHasAnyMove = true;
        break;
      }
    }

    if (stackHasAnyMove) {
      approxMoves += 1;
    }
  });

  let approxPlacement = 0;
  const me = state.players.find((p) => p.playerNumber === playerNumber);
  if (me && me.ringsInHand > 0) {
    let hasEmpty = false;
    forEachBoardPosition(board, (pos) => {
      if (hasEmpty) return;
      const key = positionKey(pos);
      if (!board.stacks.has(key) && !board.collapsedSpaces.has(key)) {
        hasEmpty = true;
      }
    });
    if (hasEmpty) {
      approxPlacement = 1;
    }
  }

  return approxMoves + approxPlacement;
}

/**
 * Evaluate forced-elimination risk: penalise positions where we control
 * many stacks but have very few real actions (moves or placements).
 */
function evaluateForcedEliminationRisk(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const board = state.board;
  let controlledStacks = 0;

  board.stacks.forEach((stack) => {
    if (stack.controllingPlayer === playerNumber) {
      controlledStacks++;
    }
  });

  if (controlledStacks === 0) return 0;

  const approxActions = approxRealActionsForPlayer(state, playerNumber);
  const ratio = approxActions / Math.max(1, controlledStacks);

  let riskFactor: number;
  if (ratio >= 2.0) {
    riskFactor = 0.0;
  } else if (ratio >= 1.0) {
    riskFactor = 2.0 - ratio;
  } else {
    riskFactor = 1.0 + (1.0 - ratio);
  }

  return -riskFactor * weights.forcedEliminationRisk;
}

/**
 * Last-player-standing (LPS) action advantage heuristic.
 *
 * In 3+ player games, reward being one of the few players with real
 * actions left and penalise being the only player without actions.
 */
function evaluateLpsActionAdvantage(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const players = state.players;
  if (players.length <= 2) return 0;

  const actionsByPlayer: Record<number, number> = {};
  for (const p of players) {
    actionsByPlayer[p.playerNumber] = approxRealActionsForPlayer(state, p.playerNumber);
  }

  const selfActions = actionsByPlayer[playerNumber] ?? 0;
  const selfHas = selfActions > 0;

  let oppWithAction = 0;
  for (const p of players) {
    if (p.playerNumber === playerNumber) continue;
    if ((actionsByPlayer[p.playerNumber] ?? 0) > 0) {
      oppWithAction++;
    }
  }

  let advantage: number;
  if (!selfHas) {
    advantage = -1.0;
  } else {
    const totalOpponents = players.length - 1;
    if (totalOpponents <= 0) {
      advantage = 0.0;
    } else {
      const inactiveFraction = (totalOpponents - oppWithAction) / totalOpponents;
      advantage = inactiveFraction;
    }
  }

  return advantage * weights.lpsActionAdvantage;
}

/**
 * Multi-player leader threat heuristic.
 *
 * In 3+ player games, penalise positions where a single opponent is much
 * closer to victory than the other opponents.
 */
function evaluateMultiLeaderThreat(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  const players = state.players;
  if (players.length <= 2) return 0;

  const proxByPlayer: Record<number, number> = {};
  for (const p of players) {
    proxByPlayer[p.playerNumber] = victoryProximityBaseForPlayer(state, p.playerNumber);
  }

  const oppProx: number[] = [];
  for (const p of players) {
    if (p.playerNumber === playerNumber) continue;
    oppProx.push(proxByPlayer[p.playerNumber]);
  }

  if (oppProx.length < 2) return 0;

  oppProx.sort((a, b) => b - a);
  const oppTop1 = oppProx[0];
  const oppTop2 = oppProx[1];

  const leaderGap = Math.max(0, oppTop1 - oppTop2);
  return -leaderGap * weights.multiLeaderThreat;
}
/**
 * Evaluate marker count (matches Python _evaluate_marker_count).
 */
function evaluateMarkerCount(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let myMarkers = 0;
  state.board.markers.forEach((marker: MarkerInfo) => {
    if (marker.player === playerNumber) {
      myMarkers += 1;
    }
  });

  return myMarkers * weights.markerCount;
}

/**
 * Evaluate vulnerability using line-of-sight (matches Python _evaluate_vulnerability).
 */
function evaluateVulnerability(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;

  state.board.stacks.forEach((stack: RingStack) => {
    if (stack.controllingPlayer !== playerNumber) return;

    const myCap = stack.capHeight;
    const visibleStacks = getVisibleStacks(stack.position, state);

    for (const visibleStack of visibleStacks) {
      if (visibleStack.controllingPlayer !== playerNumber) {
        const visibleCap = visibleStack.capHeight;
        // Capture power is based on cap height per compact rules §10.1,
        // so we compare using cap height here.
        if (visibleCap > myCap) {
          const diff = visibleCap - myCap;
          score -= diff * 1.0;
        }
      }
    }
  });

  return score * weights.vulnerability;
}

/**
 * Evaluate overtake potential (matches Python _evaluate_overtake_potential).
 */
function evaluateOvertakePotential(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;

  state.board.stacks.forEach((stack: RingStack) => {
    if (stack.controllingPlayer !== playerNumber) return;

    const myCap = stack.capHeight;
    const visibleStacks = getVisibleStacks(stack.position, state);

    for (const visibleStack of visibleStacks) {
      if (visibleStack.controllingPlayer !== playerNumber) {
        const visibleCap = visibleStack.capHeight;
        // Capture power is based on cap height per compact rules §10.1,
        // so we compare using cap height here.
        if (myCap > visibleCap) {
          const diff = myCap - visibleCap;
          score += diff * 1.0;
        }
      }
    }
  });

  return score * weights.overtakePotential;
}

/**
 * Evaluate territory closure potential (matches Python _evaluate_territory_closure).
 * Uses marker clustering as a proxy for closure potential.
 */
function evaluateTerritoryClosure(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  // Get my markers
  const myMarkers: MarkerInfo[] = [];
  state.board.markers.forEach((marker: MarkerInfo) => {
    if (marker.player === playerNumber) {
      myMarkers.push(marker);
    }
  });

  if (myMarkers.length === 0) return 0;

  // Calculate clustering - average distance between markers
  let totalDist = 0;
  let count = 0;

  // Sample first 10 markers for efficiency (matches Python)
  const markersToCheck = myMarkers.slice(0, 10);

  for (let i = 0; i < markersToCheck.length; i++) {
    for (let j = i + 1; j < markersToCheck.length; j++) {
      const m1 = markersToCheck[i];
      const m2 = markersToCheck[j];

      let dist = Math.abs(m1.position.x - m2.position.x) + Math.abs(m1.position.y - m2.position.y);
      if (m1.position.z !== undefined && m2.position.z !== undefined) {
        dist += Math.abs(m1.position.z - m2.position.z);
      }

      totalDist += dist;
      count += 1;
    }
  }

  if (count === 0) return 0;

  const avgDist = totalDist / count;

  // Lower average distance is better (more clustered)
  const clusteringScore = 10.0 / Math.max(1.0, avgDist);

  // Also reward total number of markers as a prerequisite for territory
  const markerCountScore = myMarkers.length * 0.5;

  return (clusteringScore + markerCountScore) * weights.territoryClosure;
}

/**
 * Evaluate line connectivity (matches Python _evaluate_line_connectivity).
 */
function evaluateLineConnectivity(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;
  const directions = getMovementDirectionsForBoardType(state.boardType);

  // Get my markers
  const myMarkers: MarkerInfo[] = [];
  state.board.markers.forEach((marker: MarkerInfo) => {
    if (marker.player === playerNumber) {
      myMarkers.push(marker);
    }
  });

  for (const marker of myMarkers) {
    const startPos = marker.position;

    for (const direction of directions) {
      // Check distance 1 and 2 in each direction
      const pos1 = addDirection(startPos, direction, 1);
      const key1 = positionKey(pos1);
      const pos2 = addDirection(startPos, direction, 2);
      const key2 = positionKey(pos2);

      const marker1 = state.board.markers.get(key1);
      const marker2 = state.board.markers.get(key2);

      const hasM1 = marker1 && marker1.player === playerNumber;
      const hasM2 = marker2 && marker2.player === playerNumber;

      if (hasM1) {
        score += 1.0; // Connected neighbor
      }
      if (hasM2 && !hasM1) {
        // Gap of 1, potential to connect
        // Check if gap is empty (not collapsed or occupied by stack)
        if (!state.board.collapsedSpaces.has(key1) && !state.board.stacks.has(key1)) {
          score += 0.5;
        }
      }
    }
  }

  return score * weights.lineConnectivity;
}

/**
 * Evaluate territory safety (matches Python _evaluate_territory_safety).
 */
function evaluateTerritorySafety(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;

  // Get my markers
  const myMarkers: MarkerInfo[] = [];
  state.board.markers.forEach((marker: MarkerInfo) => {
    if (marker.player === playerNumber) {
      myMarkers.push(marker);
    }
  });

  // Get opponent stacks
  const opponentStacks: RingStack[] = [];
  state.board.stacks.forEach((stack: RingStack) => {
    if (stack.controllingPlayer !== playerNumber) {
      opponentStacks.push(stack);
    }
  });

  if (myMarkers.length === 0 || opponentStacks.length === 0) return 0;

  for (const marker of myMarkers) {
    let minDist = Infinity;

    for (const stack of opponentStacks) {
      // Manhattan distance approximation
      let dist =
        Math.abs(marker.position.x - stack.position.x) +
        Math.abs(marker.position.y - stack.position.y);
      if (marker.position.z !== undefined && stack.position.z !== undefined) {
        dist += Math.abs(marker.position.z - stack.position.z);
      }
      minDist = Math.min(minDist, dist);
    }

    // If opponent is very close (dist 1 or 2), penalty
    if (minDist <= 2) {
      score -= 3.0 - minDist; // -2 for dist 1, -1 for dist 2
    }
  }

  return score * weights.territorySafety;
}

/**
 * Evaluate stack mobility (matches Python _evaluate_stack_mobility).
 */
function evaluateStackMobility(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights
): number {
  let score = 0;
  const boardSize = state.board.size;

  state.board.stacks.forEach((stack: RingStack) => {
    if (stack.controllingPlayer !== playerNumber) return;

    const stackCap = stack.capHeight;
    const adjacent = getAdjacentPositions(stack.position, state.boardType, boardSize);
    let validMovesFromHere = 0;

    for (const adjPos of adjacent) {
      const adjKey = positionKey(adjPos);

      // Check if blocked by collapsed space
      if (state.board.collapsedSpaces.has(adjKey)) continue;

      // Check if blocked by stack (unless capture possible)
      const adjStack = state.board.stacks.get(adjKey);
      if (adjStack) {
        if (adjStack.controllingPlayer !== playerNumber) {
          // Capture power is based on cap height per compact rules §10.1,
          // so we compare using cap height here.
          const adjCap = adjStack.capHeight;
          if (stackCap >= adjCap) {
            validMovesFromHere += 1;
          }
        }
        continue;
      }

      validMovesFromHere += 1;
    }

    // Reward stacks with more freedom
    score += validMovesFromHere;

    // Penalty for completely blocked stacks (dead weight)
    if (validMovesFromHere === 0) {
      score -= 5.0;
    }
  });

  return score * weights.stackMobility;
}

// ============================================================================
// Main Evaluation Function
// ============================================================================

/**
 * Complete heuristic evaluation matching Python HeuristicAI.evaluate_position().
 * Evaluates all factors and returns a comprehensive score.
 */
export function evaluateHeuristicState(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights = HEURISTIC_WEIGHTS_V1_BALANCED
): number {
  // Terminal handling: mirror the Python heuristic convention where
  // finished games dominate all other considerations.
  if (state.gameStatus === 'finished') {
    if (state.winner === playerNumber) return 100_000;
    if (state.winner !== undefined && state.winner !== null) return -100_000;
    return 0;
  }

  let score = 0;

  // Factor 1 & 2: Stack control and height
  score += evaluateStackControl(state, playerNumber, weights);

  // Factor 3: Territory control
  score += evaluateTerritory(state, playerNumber, weights);

  // Factor 4: Rings in hand
  score += evaluateRingsInHand(state, playerNumber, weights);

  // Factor 5: Center control
  score += evaluateCenterControl(state, playerNumber, weights);

  // Factor 6: Opponent threats
  score += evaluateOpponentThreats(state, playerNumber, weights);

  // Factor 7: Mobility
  score += evaluateMobility(state, playerNumber, weights);

  // Factor 8: Eliminated rings
  score += evaluateEliminatedRings(state, playerNumber, weights);

  // Factor 9: Line potential
  score += evaluateLinePotential(state, playerNumber, weights);

  // Factor 10: Victory proximity
  score += evaluateVictoryProximity(state, playerNumber, weights);

  // Factor 11: Opponent victory threat
  score += evaluateOpponentVictoryThreat(state, playerNumber, weights);

  // Factor 12: Marker count
  score += evaluateMarkerCount(state, playerNumber, weights);

  // Factor 13: Vulnerability (line-of-sight)
  score += evaluateVulnerability(state, playerNumber, weights);

  // Factor 14: Overtake potential
  score += evaluateOvertakePotential(state, playerNumber, weights);

  // Factor 15: Territory closure
  score += evaluateTerritoryClosure(state, playerNumber, weights);

  // Factor 16: Line connectivity
  score += evaluateLineConnectivity(state, playerNumber, weights);

  // Factor 17: Territory safety
  score += evaluateTerritorySafety(state, playerNumber, weights);

  // Factor 18: Stack mobility
  score += evaluateStackMobility(state, playerNumber, weights);

  // Factor 19: Forced-elimination risk
  score += evaluateForcedEliminationRisk(state, playerNumber, weights);

  // Factor 20: LPS action advantage
  score += evaluateLpsActionAdvantage(state, playerNumber, weights);

  // Factor 21: Multi-leader threat
  score += evaluateMultiLeaderThreat(state, playerNumber, weights);

  return score;
}
/**
 * Evaluate individual factor contributions for debugging/analysis.
 * Returns a breakdown of all factor scores.
 */
export interface FactorBreakdown {
  stackControl: number;
  territory: number;
  ringsInHand: number;
  centerControl: number;
  opponentThreats: number;
  mobility: number;
  eliminatedRings: number;
  linePotential: number;
  victoryProximity: number;
  opponentVictoryThreat: number;
  markerCount: number;
  vulnerability: number;
  overtakePotential: number;
  territoryClosure: number;
  lineConnectivity: number;
  territorySafety: number;
  stackMobility: number;
  forcedEliminationRisk: number;
  lpsActionAdvantage: number;
  multiLeaderThreat: number;
  total: number;
}

/**
 * Evaluate all factors and return individual contributions.
 * Useful for debugging and weight tuning.
 */
export function evaluateHeuristicStateWithBreakdown(
  state: GameState,
  playerNumber: number,
  weights: HeuristicWeights = HEURISTIC_WEIGHTS_V1_BALANCED
): FactorBreakdown {
  if (state.gameStatus === 'finished') {
    const terminalScore = state.winner === playerNumber ? 100_000 : -100_000;
    return {
      stackControl: terminalScore,
      territory: 0,
      ringsInHand: 0,
      centerControl: 0,
      opponentThreats: 0,
      mobility: 0,
      eliminatedRings: 0,
      linePotential: 0,
      victoryProximity: 0,
      opponentVictoryThreat: 0,
      markerCount: 0,
      vulnerability: 0,
      overtakePotential: 0,
      territoryClosure: 0,
      lineConnectivity: 0,
      territorySafety: 0,
      stackMobility: 0,
      forcedEliminationRisk: 0,
      lpsActionAdvantage: 0,
      multiLeaderThreat: 0,
      total: terminalScore,
    };
  }

  const breakdown: FactorBreakdown = {
    stackControl: evaluateStackControl(state, playerNumber, weights),
    territory: evaluateTerritory(state, playerNumber, weights),
    ringsInHand: evaluateRingsInHand(state, playerNumber, weights),
    centerControl: evaluateCenterControl(state, playerNumber, weights),
    opponentThreats: evaluateOpponentThreats(state, playerNumber, weights),
    mobility: evaluateMobility(state, playerNumber, weights),
    eliminatedRings: evaluateEliminatedRings(state, playerNumber, weights),
    linePotential: evaluateLinePotential(state, playerNumber, weights),
    victoryProximity: evaluateVictoryProximity(state, playerNumber, weights),
    opponentVictoryThreat: evaluateOpponentVictoryThreat(state, playerNumber, weights),
    markerCount: evaluateMarkerCount(state, playerNumber, weights),
    vulnerability: evaluateVulnerability(state, playerNumber, weights),
    overtakePotential: evaluateOvertakePotential(state, playerNumber, weights),
    territoryClosure: evaluateTerritoryClosure(state, playerNumber, weights),
    lineConnectivity: evaluateLineConnectivity(state, playerNumber, weights),
    territorySafety: evaluateTerritorySafety(state, playerNumber, weights),
    stackMobility: evaluateStackMobility(state, playerNumber, weights),
    forcedEliminationRisk: evaluateForcedEliminationRisk(state, playerNumber, weights),
    lpsActionAdvantage: evaluateLpsActionAdvantage(state, playerNumber, weights),
    multiLeaderThreat: evaluateMultiLeaderThreat(state, playerNumber, weights),
    total: 0,
  };

  breakdown.total = Object.entries(breakdown)
    .filter(([key]) => key !== 'total')
    .reduce((sum, [, value]) => sum + value, 0);

  return breakdown;
}

// ============================================================================
// Move Scoring API (Forward-looking)
// ============================================================================

/**
 * Parameters for move-scoring once shared mutators are available.
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
  throw new Error('scoreMove is not yet implemented – awaiting shared mutator integration');
}
