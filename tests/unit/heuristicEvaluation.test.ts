/**
 * Comprehensive tests for TypeScript heuristic evaluation.
 *
 * Tests all 17 evaluation factors independently, JSON weight loading,
 * and weight synchronization between Python and TypeScript.
 */

import {
  evaluateHeuristicState,
  evaluateHeuristicStateWithBreakdown,
  loadHeuristicWeightsFromJSON,
  exportHeuristicWeightsToJSON,
  getHeuristicWeightsTS,
  HEURISTIC_WEIGHTS_V1_BALANCED,
  HEURISTIC_WEIGHTS_V1_AGGRESSIVE,
  HEURISTIC_WEIGHTS_V1_TERRITORIAL,
  HEURISTIC_WEIGHTS_V1_DEFENSIVE,
  HEURISTIC_WEIGHT_PROFILES_TS,
  HeuristicWeights,
  FactorBreakdown,
} from '../../src/shared/engine/heuristicEvaluation';
import type {
  GameState,
  BoardState,
  Player,
  RingStack,
  MarkerInfo,
} from '../../src/shared/types/game';

// ============================================================================
// Test Helpers
// ============================================================================

function makeEmptyBoardState(type: 'square8' | 'hexagonal' = 'square8', size = 8): BoardState {
  return {
    stacks: new Map<string, RingStack>(),
    markers: new Map<string, MarkerInfo>(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size,
    type,
  } as unknown as BoardState;
}

function makePlayer(playerNumber: number, overrides: Partial<Player> = {}): Player {
  return {
    id: `p${playerNumber}`,
    username: `Player${playerNumber}`,
    type: 'ai',
    playerNumber,
    isReady: true,
    timeRemaining: 600_000,
    ringsInHand: 10,
    eliminatedRings: 0,
    territorySpaces: 0,
    ...overrides,
  } as unknown as Player;
}

function makeBaseGameState(overrides: Partial<GameState> = {}): GameState {
  const board = (overrides.board as BoardState) ?? makeEmptyBoardState();
  const players: Player[] = (overrides.players as Player[]) ?? [makePlayer(1), makePlayer(2)];

  return {
    id: 'heuristic-test',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 33,
    ...overrides,
  } as unknown as GameState;
}

function makeStack(
  x: number,
  y: number,
  controllingPlayer: number,
  height: number,
  z?: number
): RingStack {
  const position = z !== undefined ? { x, y, z } : { x, y };
  return {
    position,
    rings: Array(height).fill(controllingPlayer),
    stackHeight: height,
    capHeight: height,
    controllingPlayer,
  } as RingStack;
}

function makeMarker(x: number, y: number, player: number, z?: number): MarkerInfo {
  const position = z !== undefined ? { x, y, z } : { x, y };
  return { position, player } as MarkerInfo;
}

function posKey(x: number, y: number, z?: number): string {
  return z !== undefined ? `${x},${y},${z}` : `${x},${y}`;
}

// ============================================================================
// Factor 1 & 2: Stack Control and Stack Height
// ============================================================================

describe('evaluateStackControl', () => {
  it('rewards controlling more stacks than opponent', () => {
    const board = makeEmptyBoardState();

    // Player 1: 3 stacks
    board.stacks.set('0,0', makeStack(0, 0, 1, 1));
    board.stacks.set('1,1', makeStack(1, 1, 1, 1));
    board.stacks.set('2,2', makeStack(2, 2, 1, 1));

    // Player 2: 1 stack
    board.stacks.set('5,5', makeStack(5, 5, 2, 1));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.stackControl).toBeGreaterThan(0);
  });

  it('rewards taller stacks with diminishing returns', () => {
    const board = makeEmptyBoardState();

    // Player 1: one tall stack (height 7)
    // Diminishing returns: h <= 5 ? h : 5 + (h - 5) * 0.1
    // For h=7: 5 + (7-5) * 0.1 = 5.2
    board.stacks.set('0,0', makeStack(0, 0, 1, 7));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // Stack Control contributes both count and height
    expect(breakdown.stackControl).toBeGreaterThan(0);
  });

  it('penalizes when opponent has more stacks', () => {
    const board = makeEmptyBoardState();

    // Player 1: 1 stack
    board.stacks.set('0,0', makeStack(0, 0, 1, 1));

    // Player 2: 3 stacks
    board.stacks.set('5,5', makeStack(5, 5, 2, 1));
    board.stacks.set('6,6', makeStack(6, 6, 2, 1));
    board.stacks.set('7,7', makeStack(7, 7, 2, 1));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.stackControl).toBeLessThan(0);
  });
});

// ============================================================================
// Factor 3: Territory
// ============================================================================

describe('evaluateTerritory', () => {
  it('rewards territory advantage', () => {
    const players = [makePlayer(1, { territorySpaces: 15 }), makePlayer(2, { territorySpaces: 5 })];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.territory).toBeGreaterThan(0);
  });

  it('penalizes territory disadvantage', () => {
    const players = [makePlayer(1, { territorySpaces: 5 }), makePlayer(2, { territorySpaces: 20 })];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.territory).toBeLessThan(0);
  });

  it('returns zero for equal territory', () => {
    const players = [
      makePlayer(1, { territorySpaces: 10 }),
      makePlayer(2, { territorySpaces: 10 }),
    ];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.territory).toBe(0);
  });
});

// ============================================================================
// Factor 4: Rings in Hand
// ============================================================================

describe('evaluateRingsInHand', () => {
  it('rewards having rings in hand', () => {
    const players = [makePlayer(1, { ringsInHand: 10 }), makePlayer(2, { ringsInHand: 5 })];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // Rings in hand = my rings * weight
    expect(breakdown.ringsInHand).toBeGreaterThan(0);
  });

  it('returns zero when no rings in hand', () => {
    const players = [makePlayer(1, { ringsInHand: 0 }), makePlayer(2, { ringsInHand: 5 })];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.ringsInHand).toBe(0);
  });
});

// ============================================================================
// Factor 5: Center Control
// ============================================================================

describe('evaluateCenterControl', () => {
  it('rewards controlling center positions on square board', () => {
    const board = makeEmptyBoardState('square8', 8);

    // Center of 8x8 board is around (3,3), (3,4), (4,3), (4,4)
    board.stacks.set('3,3', makeStack(3, 3, 1, 2));
    board.stacks.set('4,4', makeStack(4, 4, 1, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.centerControl).toBeGreaterThan(0);
  });

  it('penalizes when opponent controls center', () => {
    const board = makeEmptyBoardState('square8', 8);

    // Player 2 controls center
    board.stacks.set('3,3', makeStack(3, 3, 2, 2));
    board.stacks.set('4,4', makeStack(4, 4, 2, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.centerControl).toBeLessThan(0);
  });
});

// ============================================================================
// Factor 6: Opponent Threats
// ============================================================================

describe('evaluateOpponentThreats', () => {
  it('penalizes adjacent taller enemy stacks', () => {
    const board = makeEmptyBoardState();

    // Player 1 stack at (3,3) height 1
    board.stacks.set('3,3', makeStack(3, 3, 1, 1));
    // Player 2 taller stack adjacent at (3,4) height 3
    board.stacks.set('3,4', makeStack(3, 4, 2, 3));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.opponentThreats).toBeLessThan(0);
  });

  it('returns zero when no adjacent enemy stacks', () => {
    const board = makeEmptyBoardState();

    // Player 1 stack isolated
    board.stacks.set('0,0', makeStack(0, 0, 1, 2));
    // Player 2 stack far away
    board.stacks.set('7,7', makeStack(7, 7, 2, 3));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.opponentThreats).toBe(0);
  });
});

// ============================================================================
// Factor 7: Mobility
// ============================================================================

describe('evaluateMobility', () => {
  it('rewards having more mobility', () => {
    const board = makeEmptyBoardState();

    // Player 1: stack with many open directions
    board.stacks.set('4,4', makeStack(4, 4, 1, 2));

    // Player 2: stack in corner with fewer options
    board.stacks.set('0,0', makeStack(0, 0, 2, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // Central position should have better mobility
    expect(breakdown.mobility).toBeGreaterThan(0);
  });
});

// ============================================================================
// Factor 8: Eliminated Rings
// ============================================================================

describe('evaluateEliminatedRings', () => {
  it('rewards eliminating opponent rings', () => {
    const players = [makePlayer(1, { eliminatedRings: 5 }), makePlayer(2, { eliminatedRings: 0 })];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // 5 eliminated * weight
    expect(breakdown.eliminatedRings).toBeGreaterThan(0);
  });

  it('returns zero when no rings eliminated', () => {
    const players = [makePlayer(1, { eliminatedRings: 0 }), makePlayer(2, { eliminatedRings: 0 })];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.eliminatedRings).toBe(0);
  });
});

// ============================================================================
// Factor 9: Line Potential
// ============================================================================

describe('evaluateLinePotential', () => {
  it('rewards markers aligned in a row', () => {
    const board = makeEmptyBoardState();

    // 3 markers in a row for player 1
    board.markers.set('2,2', makeMarker(2, 2, 1));
    board.markers.set('3,2', makeMarker(3, 2, 1));
    board.markers.set('4,2', makeMarker(4, 2, 1));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.linePotential).toBeGreaterThan(0);
  });

  it('returns zero when player has no markers', () => {
    const board = makeEmptyBoardState();

    // Player 2 has 4 markers in a row, but we evaluate for player 1
    board.markers.set('2,2', makeMarker(2, 2, 2));
    board.markers.set('3,2', makeMarker(3, 2, 2));
    board.markers.set('4,2', makeMarker(4, 2, 2));
    board.markers.set('5,2', makeMarker(5, 2, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // linePotential only evaluates the current player's markers
    expect(breakdown.linePotential).toBe(0);
  });
});

// ============================================================================
// Factor 10: Victory Proximity
// ============================================================================

describe('evaluateVictoryProximity', () => {
  it('rewards being close to victory threshold', () => {
    const players = [
      makePlayer(1, { eliminatedRings: 8, territorySpaces: 30 }), // Close to 10 and 33
      makePlayer(2, { eliminatedRings: 0, territorySpaces: 0 }),
    ];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.victoryProximity).toBeGreaterThan(0);
  });

  it('returns high value when at victory threshold', () => {
    const players = [
      makePlayer(1, { eliminatedRings: 10 }), // At victory threshold
      makePlayer(2, { eliminatedRings: 0 }),
    ];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // Should return 1000.0 * victoryProximity weight
    expect(breakdown.victoryProximity).toBeGreaterThan(100);
  });
});

// ============================================================================
// Additional Factor: Opponent Victory Threat
// ============================================================================

describe('evaluateOpponentVictoryThreat', () => {
  it('is zero when we are ahead of all opponents', () => {
    const players = [
      makePlayer(1, { eliminatedRings: 5, territorySpaces: 10 }),
      makePlayer(2, { eliminatedRings: 1, territorySpaces: 5 }),
      makePlayer(3, { eliminatedRings: 1, territorySpaces: 5 }),
    ];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.opponentVictoryThreat).toBe(0);
  });

  it('penalizes when an opponent is closer to victory', () => {
    const players = [
      makePlayer(1, { eliminatedRings: 0, territorySpaces: 0 }),
      makePlayer(2, { eliminatedRings: 8, territorySpaces: 30 }),
      makePlayer(3, { eliminatedRings: 0, territorySpaces: 0 }),
    ];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.opponentVictoryThreat).toBeLessThan(0);
  });
});

// ============================================================================
// Factor 11: Marker Count
// ============================================================================

describe('evaluateMarkerCount', () => {
  it('rewards having more markers', () => {
    const board = makeEmptyBoardState();

    // Player 1 has 5 markers
    for (let i = 0; i < 5; i++) {
      board.markers.set(`${i},0`, makeMarker(i, 0, 1));
    }

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.markerCount).toBeGreaterThan(0);
  });

  it('returns zero for no markers', () => {
    const state = makeBaseGameState();
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.markerCount).toBe(0);
  });
});

// ============================================================================
// Factor 12: Vulnerability (line-of-sight)
// ============================================================================

describe('evaluateVulnerability', () => {
  it('penalizes stacks with line-of-sight to taller enemy stacks', () => {
    const board = makeEmptyBoardState();

    // Player 1 short stack
    board.stacks.set('3,3', makeStack(3, 3, 1, 1));
    // Player 2 taller stack in line of sight (same row)
    board.stacks.set('6,3', makeStack(6, 3, 2, 4));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.vulnerability).toBeLessThan(0);
  });

  it('returns zero when equal heights', () => {
    const board = makeEmptyBoardState();

    // Same height stacks
    board.stacks.set('3,3', makeStack(3, 3, 1, 2));
    board.stacks.set('6,3', makeStack(6, 3, 2, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.vulnerability).toBe(0);
  });
});

// ============================================================================
// Factor 13: Overtake Potential
// ============================================================================

describe('evaluateOvertakePotential', () => {
  it('rewards stacks that can overtake shorter enemy stacks', () => {
    const board = makeEmptyBoardState();

    // Player 1 tall stack
    board.stacks.set('3,3', makeStack(3, 3, 1, 4));
    // Player 2 shorter stack in line of sight
    board.stacks.set('6,3', makeStack(6, 3, 2, 1));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.overtakePotential).toBeGreaterThan(0);
  });

  it('returns zero when stack is shorter', () => {
    const board = makeEmptyBoardState();

    // Player 1 short stack
    board.stacks.set('3,3', makeStack(3, 3, 1, 1));
    // Player 2 taller stack
    board.stacks.set('6,3', makeStack(6, 3, 2, 4));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.overtakePotential).toBe(0);
  });
});

// ============================================================================
// Factor 14: Territory Closure
// ============================================================================

describe('evaluateTerritoryClosure', () => {
  it('rewards clustered markers', () => {
    const board = makeEmptyBoardState();

    // Clustered markers for player 1
    board.markers.set('3,3', makeMarker(3, 3, 1));
    board.markers.set('3,4', makeMarker(3, 4, 1));
    board.markers.set('4,3', makeMarker(4, 3, 1));
    board.markers.set('4,4', makeMarker(4, 4, 1));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.territoryClosure).toBeGreaterThan(0);
  });

  it('returns zero for no markers', () => {
    const state = makeBaseGameState();
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.territoryClosure).toBe(0);
  });
});

// ============================================================================
// Factor 15: Line Connectivity
// ============================================================================

describe('evaluateLineConnectivity', () => {
  it('rewards connected markers', () => {
    const board = makeEmptyBoardState();

    // Connected markers in a line
    board.markers.set('3,3', makeMarker(3, 3, 1));
    board.markers.set('4,3', makeMarker(4, 3, 1));
    board.markers.set('5,3', makeMarker(5, 3, 1));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.lineConnectivity).toBeGreaterThan(0);
  });

  it('rewards markers with connection potential (gaps)', () => {
    const board = makeEmptyBoardState();

    // Markers with gap (could connect)
    board.markers.set('3,3', makeMarker(3, 3, 1));
    // Gap at 4,3
    board.markers.set('5,3', makeMarker(5, 3, 1));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.lineConnectivity).toBeGreaterThan(0);
  });
});

// ============================================================================
// Factor 16: Territory Safety
// ============================================================================

describe('evaluateTerritorySafety', () => {
  it('penalizes markers near opponent stacks', () => {
    const board = makeEmptyBoardState();

    // Player 1 marker
    board.markers.set('3,3', makeMarker(3, 3, 1));
    // Player 2 stack very close (distance 1)
    board.stacks.set('3,4', makeStack(3, 4, 2, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.territorySafety).toBeLessThan(0);
  });

  it('returns zero when no opponent stacks nearby', () => {
    const board = makeEmptyBoardState();

    // Player 1 marker
    board.markers.set('0,0', makeMarker(0, 0, 1));
    // Player 2 stack far away
    board.stacks.set('7,7', makeStack(7, 7, 2, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.territorySafety).toBe(0);
  });
});

// ============================================================================
// Factor 17: Stack Mobility
// ============================================================================

describe('evaluateStackMobility', () => {
  it('rewards stacks with more movement options', () => {
    const board = makeEmptyBoardState();

    // Player 1 stack in center (many options)
    board.stacks.set('4,4', makeStack(4, 4, 1, 2));

    // Player 2 stack in corner (fewer options)
    board.stacks.set('0,0', makeStack(0, 0, 2, 2));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.stackMobility).toBeGreaterThan(0);
  });

  it('penalizes completely blocked stacks', () => {
    const board = makeEmptyBoardState();

    // Player 1 stack
    board.stacks.set('3,3', makeStack(3, 3, 1, 1));
    // Surrounded by taller enemy stacks (blocked)
    board.stacks.set('2,3', makeStack(2, 3, 2, 3));
    board.stacks.set('4,3', makeStack(4, 3, 2, 3));
    board.stacks.set('3,2', makeStack(3, 2, 2, 3));
    board.stacks.set('3,4', makeStack(3, 4, 2, 3));
    // Also diagonals for complete blocking
    board.stacks.set('2,2', makeStack(2, 2, 2, 3));
    board.stacks.set('4,4', makeStack(4, 4, 2, 3));
    board.stacks.set('2,4', makeStack(2, 4, 2, 3));
    board.stacks.set('4,2', makeStack(4, 2, 2, 3));

    const state = makeBaseGameState({ board });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // Should be negative due to blocked stack penalty
    expect(breakdown.stackMobility).toBeLessThan(0);
  });
});

// ============================================================================
// Additional Factor: Forced-Elimination Risk
// ============================================================================

describe('evaluateForcedEliminationRisk', () => {
  it('penalizes many stacks with few real actions', () => {
    const highRiskBoard = makeEmptyBoardState();

    // Player 1: two stacks on the board
    highRiskBoard.stacks.set('3,3', makeStack(3, 3, 1, 1));
    highRiskBoard.stacks.set('4,4', makeStack(4, 4, 1, 1));

    // Collapse all other spaces so there are no moves or placements.
    for (let x = 0; x < highRiskBoard.size; x++) {
      for (let y = 0; y < highRiskBoard.size; y++) {
        const key = `${x},${y}`;
        if (key !== '3,3' && key !== '4,4') {
          highRiskBoard.collapsedSpaces.set(key, 1);
        }
      }
    }

    const highRiskState = makeBaseGameState({ board: highRiskBoard });
    const highRisk = evaluateHeuristicStateWithBreakdown(
      highRiskState,
      1,
      HEURISTIC_WEIGHTS_V1_BALANCED
    ).forcedEliminationRisk;

    const lowRiskBoard = makeEmptyBoardState();
    lowRiskBoard.stacks.set('3,3', makeStack(3, 3, 1, 1));
    lowRiskBoard.stacks.set('4,4', makeStack(4, 4, 1, 1));

    const lowRiskState = makeBaseGameState({ board: lowRiskBoard });
    const lowRisk = evaluateHeuristicStateWithBreakdown(
      lowRiskState,
      1,
      HEURISTIC_WEIGHTS_V1_BALANCED
    ).forcedEliminationRisk;

    expect(highRisk).toBeLessThan(0);
    expect(highRisk).toBeLessThan(lowRisk);
  });

  it('returns zero when player controls no stacks', () => {
    const state = makeBaseGameState();
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.forcedEliminationRisk).toBe(0);
  });
});

// ============================================================================
// Additional Factor: LPS Action Advantage
// ============================================================================

describe('evaluateLpsActionAdvantage', () => {
  it('rewards being one of the few players with actions', () => {
    const board = makeEmptyBoardState();
    board.stacks.set('3,3', makeStack(3, 3, 1, 1));

    const players = [
      makePlayer(1, { ringsInHand: 1 }),
      makePlayer(2, { ringsInHand: 0 }),
      makePlayer(3, { ringsInHand: 0 }),
    ];
    const state = makeBaseGameState({ board, players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.lpsActionAdvantage).toBeGreaterThan(0);
  });

  it('penalizes having no actions when opponents do', () => {
    const board = makeEmptyBoardState();
    // Only player 2 has a mobile stack.
    board.stacks.set('3,3', makeStack(3, 3, 2, 1));

    const players = [
      makePlayer(1, { ringsInHand: 0 }),
      makePlayer(2, { ringsInHand: 1 }),
      makePlayer(3, { ringsInHand: 0 }),
    ];
    const state = makeBaseGameState({ board, players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.lpsActionAdvantage).toBeLessThan(0);
  });
});

// ============================================================================
// Additional Factor: Multi-Leader Threat
// ============================================================================

describe('evaluateMultiLeaderThreat', () => {
  it('returns zero for two-player games', () => {
    const players = [makePlayer(1), makePlayer(2)];
    const state = makeBaseGameState({ players });
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.multiLeaderThreat).toBe(0);
  });

  it('penalizes when one opponent is far ahead of others', () => {
    const baseThresholds = {
      victoryThreshold: 10,
      territoryVictoryThreshold: 100,
    };

    const smallGapPlayers = [
      makePlayer(1, { eliminatedRings: 0, territorySpaces: 0 }),
      makePlayer(2, { eliminatedRings: 5, territorySpaces: 0 }),
      makePlayer(3, { eliminatedRings: 4, territorySpaces: 0 }),
    ];
    const smallGapState = makeBaseGameState({
      players: smallGapPlayers,
      ...baseThresholds,
    });
    const smallGap = evaluateHeuristicStateWithBreakdown(
      smallGapState,
      1,
      HEURISTIC_WEIGHTS_V1_BALANCED
    ).multiLeaderThreat;

    const largeGapPlayers = [
      makePlayer(1, { eliminatedRings: 0, territorySpaces: 0 }),
      makePlayer(2, { eliminatedRings: 9, territorySpaces: 0 }),
      makePlayer(3, { eliminatedRings: 1, territorySpaces: 0 }),
    ];
    const largeGapState = makeBaseGameState({
      players: largeGapPlayers,
      ...baseThresholds,
    });
    const largeGap = evaluateHeuristicStateWithBreakdown(
      largeGapState,
      1,
      HEURISTIC_WEIGHTS_V1_BALANCED
    ).multiLeaderThreat;

    expect(smallGap).toBeLessThanOrEqual(0);
    expect(largeGap).toBeLessThan(smallGap);
  });
});

// ============================================================================
// JSON Weight Loading
// ============================================================================

describe('loadHeuristicWeightsFromJSON', () => {
  it('loads weights from TypeScript camelCase keys', () => {
    const json = {
      stackControl: 15.0,
      stackHeight: 7.0,
      territory: 10.0,
    };

    const weights = loadHeuristicWeightsFromJSON(json);

    expect(weights.stackControl).toBe(15.0);
    expect(weights.stackHeight).toBe(7.0);
    expect(weights.territory).toBe(10.0);
    // Other values should fall back to balanced defaults
    expect(weights.ringsInHand).toBe(HEURISTIC_WEIGHTS_V1_BALANCED.ringsInHand);
  });

  it('loads weights from Python UPPER_SNAKE_CASE keys', () => {
    const json = {
      WEIGHT_STACK_CONTROL: 20.0,
      WEIGHT_TERRITORY: 15.0,
      WEIGHT_MOBILITY: 8.0,
    };

    const weights = loadHeuristicWeightsFromJSON(json);

    expect(weights.stackControl).toBe(20.0);
    expect(weights.territory).toBe(15.0);
    expect(weights.mobility).toBe(8.0);
  });

  it('handles mixed Python and TypeScript keys', () => {
    const json = {
      WEIGHT_STACK_CONTROL: 25.0,
      territory: 12.0,
      WEIGHT_VULNERABILITY: 10.0,
    };

    const weights = loadHeuristicWeightsFromJSON(json);

    expect(weights.stackControl).toBe(25.0);
    expect(weights.territory).toBe(12.0);
    expect(weights.vulnerability).toBe(10.0);
  });

  it('ignores unknown keys', () => {
    const json = {
      stackControl: 15.0,
      unknownKey: 999.0,
      NONEXISTENT_WEIGHT: 123.0,
    };

    const weights = loadHeuristicWeightsFromJSON(json);

    expect(weights.stackControl).toBe(15.0);
    expect((weights as unknown as Record<string, number>)['unknownKey']).toBeUndefined();
  });

  it('loads complete Python export format', () => {
    // Simulates output from ai-service/scripts/export_heuristic_weights.py
    const pythonExport = {
      WEIGHT_STACK_CONTROL: 10.0,
      WEIGHT_STACK_HEIGHT: 5.0,
      WEIGHT_TERRITORY: 8.0,
      WEIGHT_RINGS_IN_HAND: 3.0,
      WEIGHT_CENTER_CONTROL: 4.0,
      WEIGHT_ADJACENCY: 2.0,
      WEIGHT_OPPONENT_THREAT: 6.0,
      WEIGHT_MOBILITY: 4.0,
      WEIGHT_ELIMINATED_RINGS: 12.0,
      WEIGHT_LINE_POTENTIAL: 7.0,
      WEIGHT_VICTORY_PROXIMITY: 20.0,
      WEIGHT_MARKER_COUNT: 1.5,
      WEIGHT_VULNERABILITY: 8.0,
      WEIGHT_OVERTAKE_POTENTIAL: 8.0,
      WEIGHT_TERRITORY_CLOSURE: 10.0,
      WEIGHT_LINE_CONNECTIVITY: 6.0,
      WEIGHT_TERRITORY_SAFETY: 5.0,
      WEIGHT_STACK_MOBILITY: 4.0,
    };

    const weights = loadHeuristicWeightsFromJSON(pythonExport);

    expect(weights.stackControl).toBe(10.0);
    expect(weights.stackHeight).toBe(5.0);
    expect(weights.territory).toBe(8.0);
    expect(weights.ringsInHand).toBe(3.0);
    expect(weights.centerControl).toBe(4.0);
    expect(weights.adjacency).toBe(2.0);
    expect(weights.opponentThreat).toBe(6.0);
    expect(weights.mobility).toBe(4.0);
    expect(weights.eliminatedRings).toBe(12.0);
    expect(weights.linePotential).toBe(7.0);
    expect(weights.victoryProximity).toBe(20.0);
    expect(weights.markerCount).toBe(1.5);
    expect(weights.vulnerability).toBe(8.0);
    expect(weights.overtakePotential).toBe(8.0);
    expect(weights.territoryClosure).toBe(10.0);
    expect(weights.lineConnectivity).toBe(6.0);
    expect(weights.territorySafety).toBe(5.0);
    expect(weights.stackMobility).toBe(4.0);
  });
});

describe('exportHeuristicWeightsToJSON', () => {
  it('exports weights to Python UPPER_SNAKE_CASE format', () => {
    const json = exportHeuristicWeightsToJSON(HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(json.WEIGHT_STACK_CONTROL).toBe(10.0);
    expect(json.WEIGHT_STACK_HEIGHT).toBe(5.0);
    expect(json.WEIGHT_TERRITORY).toBe(8.0);
    expect(json.WEIGHT_MOBILITY).toBe(4.0);
  });

  it('round-trips through load -> export', () => {
    const original = HEURISTIC_WEIGHTS_V1_BALANCED;
    const exported = exportHeuristicWeightsToJSON(original);
    const reloaded = loadHeuristicWeightsFromJSON(exported);

    expect(reloaded.stackControl).toBe(original.stackControl);
    expect(reloaded.stackHeight).toBe(original.stackHeight);
    expect(reloaded.territory).toBe(original.territory);
    expect(reloaded.vulnerability).toBe(original.vulnerability);
    expect(reloaded.stackMobility).toBe(original.stackMobility);
  });

  it('exports all 22 weight constants', () => {
    const json = exportHeuristicWeightsToJSON(HEURISTIC_WEIGHTS_V1_BALANCED);
    const keys = Object.keys(json);

    expect(keys.length).toBe(22);
    expect(keys).toContain('WEIGHT_STACK_CONTROL');
    expect(keys).toContain('WEIGHT_STACK_HEIGHT');
    expect(keys).toContain('WEIGHT_TERRITORY');
    expect(keys).toContain('WEIGHT_RINGS_IN_HAND');
    expect(keys).toContain('WEIGHT_CENTER_CONTROL');
    expect(keys).toContain('WEIGHT_ADJACENCY');
    expect(keys).toContain('WEIGHT_OPPONENT_THREAT');
    expect(keys).toContain('WEIGHT_MOBILITY');
    expect(keys).toContain('WEIGHT_ELIMINATED_RINGS');
    expect(keys).toContain('WEIGHT_LINE_POTENTIAL');
    expect(keys).toContain('WEIGHT_VICTORY_PROXIMITY');
    expect(keys).toContain('WEIGHT_MARKER_COUNT');
    expect(keys).toContain('WEIGHT_VULNERABILITY');
    expect(keys).toContain('WEIGHT_OVERTAKE_POTENTIAL');
    expect(keys).toContain('WEIGHT_TERRITORY_CLOSURE');
    expect(keys).toContain('WEIGHT_LINE_CONNECTIVITY');
    expect(keys).toContain('WEIGHT_TERRITORY_SAFETY');
    expect(keys).toContain('WEIGHT_STACK_MOBILITY');
    expect(keys).toContain('WEIGHT_OPPONENT_VICTORY_THREAT');
    expect(keys).toContain('WEIGHT_FORCED_ELIMINATION_RISK');
    expect(keys).toContain('WEIGHT_LPS_ACTION_ADVANTAGE');
    expect(keys).toContain('WEIGHT_MULTI_LEADER_THREAT');
  });
});

// ============================================================================
// Weight Profile Resolution
// ============================================================================

describe('getHeuristicWeightsTS', () => {
  it('returns balanced profile for null/undefined', () => {
    expect(getHeuristicWeightsTS(null)).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    expect(getHeuristicWeightsTS(undefined)).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    expect(getHeuristicWeightsTS('')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
  });

  it('returns correct profile for known ids', () => {
    expect(getHeuristicWeightsTS('heuristic_v1_balanced')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    expect(getHeuristicWeightsTS('heuristic_v1_aggressive')).toBe(HEURISTIC_WEIGHTS_V1_AGGRESSIVE);
    expect(getHeuristicWeightsTS('heuristic_v1_territorial')).toBe(
      HEURISTIC_WEIGHTS_V1_TERRITORIAL
    );
    expect(getHeuristicWeightsTS('heuristic_v1_defensive')).toBe(HEURISTIC_WEIGHTS_V1_DEFENSIVE);
  });

  it('returns balanced profile for ladder-linked ids', () => {
    expect(getHeuristicWeightsTS('v1-heuristic-2')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    expect(getHeuristicWeightsTS('v1-heuristic-3')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    expect(getHeuristicWeightsTS('v1-heuristic-4')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    expect(getHeuristicWeightsTS('v1-heuristic-5')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
  });

  it('returns balanced profile for unknown ids', () => {
    expect(getHeuristicWeightsTS('unknown_profile')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    expect(getHeuristicWeightsTS('random_string')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
  });
});

// ============================================================================
// Overall Evaluation Consistency
// ============================================================================

describe('evaluateHeuristicState consistency', () => {
  it('is deterministic for the same state and weights', () => {
    const board = makeEmptyBoardState();
    board.stacks.set('3,3', makeStack(3, 3, 1, 2));
    board.stacks.set('5,5', makeStack(5, 5, 2, 2));
    board.markers.set('2,2', makeMarker(2, 2, 1));
    board.markers.set('6,6', makeMarker(6, 6, 2));

    const state = makeBaseGameState({ board });

    const score1 = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const score2 = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const score3 = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(score1).toBe(score2);
    expect(score2).toBe(score3);
  });

  it('produces different scores for different players', () => {
    const board = makeEmptyBoardState();
    board.stacks.set('3,3', makeStack(3, 3, 1, 3)); // Player 1 taller
    board.stacks.set('5,5', makeStack(5, 5, 2, 1));

    const state = makeBaseGameState({ board });

    const scoreP1 = evaluateHeuristicState(state, 1);
    const scoreP2 = evaluateHeuristicState(state, 2);

    expect(scoreP1).not.toBe(scoreP2);
    expect(scoreP1).toBeGreaterThan(scoreP2);
  });

  it('produces different scores for different weight profiles', () => {
    const board = makeEmptyBoardState();
    board.stacks.set('3,3', makeStack(3, 3, 1, 2));

    const state = makeBaseGameState({ board });

    const balanced = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const aggressive = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_AGGRESSIVE);
    const territorial = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_TERRITORIAL);
    const defensive = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_DEFENSIVE);

    // All should be different (different weight profiles)
    const scores = new Set([balanced, aggressive, territorial, defensive]);
    expect(scores.size).toBeGreaterThan(1);
  });

  it('breakdown total matches overall evaluation', () => {
    const board = makeEmptyBoardState();
    board.stacks.set('3,3', makeStack(3, 3, 1, 2));
    board.stacks.set('5,5', makeStack(5, 5, 2, 1));
    board.markers.set('2,2', makeMarker(2, 2, 1));

    const state = makeBaseGameState({ board });

    const overall = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const breakdown = evaluateHeuristicStateWithBreakdown(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(breakdown.total).toBeCloseTo(overall, 5);
  });

  it('handles terminal states correctly', () => {
    const board = makeEmptyBoardState();
    const state = makeBaseGameState({
      board,
      gameStatus: 'finished',
      winner: 1,
    } as Partial<GameState>);

    const score = evaluateHeuristicState(state, 1);
    expect(score).toBe(100_000);

    const scoreLoser = evaluateHeuristicState(state, 2);
    expect(scoreLoser).toBe(-100_000);
  });

  it('handles draw state correctly', () => {
    const state = makeBaseGameState({
      gameStatus: 'finished',
      winner: undefined,
    } as Partial<GameState>);

    const score = evaluateHeuristicState(state, 1);
    expect(score).toBe(0);
  });
});

// ============================================================================
// Weight Profiles Validation
// ============================================================================

describe('HEURISTIC_WEIGHT_PROFILES_TS', () => {
  it('contains all expected profiles', () => {
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('heuristic_v1_balanced');
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('heuristic_v1_aggressive');
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('heuristic_v1_territorial');
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('heuristic_v1_defensive');
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('v1-heuristic-2');
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('v1-heuristic-3');
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('v1-heuristic-4');
    expect(HEURISTIC_WEIGHT_PROFILES_TS).toHaveProperty('v1-heuristic-5');
  });

  it('all profiles have all weight keys', () => {
    const requiredKeys: (keyof HeuristicWeights)[] = [
      'stackControl',
      'stackHeight',
      'territory',
      'ringsInHand',
      'centerControl',
      'adjacency',
      'opponentThreat',
      'mobility',
      'eliminatedRings',
      'linePotential',
      'victoryProximity',
      'markerCount',
      'vulnerability',
      'overtakePotential',
      'territoryClosure',
      'lineConnectivity',
      'territorySafety',
      'stackMobility',
      'opponentVictoryThreat',
      'forcedEliminationRisk',
      'lpsActionAdvantage',
      'multiLeaderThreat',
    ];

    for (const profile of Object.values(HEURISTIC_WEIGHT_PROFILES_TS)) {
      for (const key of requiredKeys) {
        expect(profile).toHaveProperty(key);
        expect(typeof profile[key]).toBe('number');
      }
    }
  });

  it('personality profiles have different weights than balanced', () => {
    const balanced = HEURISTIC_WEIGHTS_V1_BALANCED;
    const aggressive = HEURISTIC_WEIGHTS_V1_AGGRESSIVE;
    const territorial = HEURISTIC_WEIGHTS_V1_TERRITORIAL;
    const defensive = HEURISTIC_WEIGHTS_V1_DEFENSIVE;

    // Aggressive differs in expected ways
    expect(aggressive.stackControl).toBeGreaterThan(balanced.stackControl);
    expect(aggressive.overtakePotential).toBeGreaterThan(balanced.overtakePotential);

    // Territorial differs in expected ways
    expect(territorial.territory).toBeGreaterThan(balanced.territory);
    expect(territorial.territoryClosure).toBeGreaterThan(balanced.territoryClosure);

    // Defensive differs in expected ways
    expect(defensive.vulnerability).toBeGreaterThan(balanced.vulnerability);
    expect(defensive.territorySafety).toBeGreaterThan(balanced.territorySafety);
  });
});
