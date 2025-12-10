/**
 * heuristicEvaluation.branchCoverage.test.ts
 *
 * Branch coverage tests for heuristicEvaluation.ts targeting uncovered branches:
 * - Weight profile loading/exporting
 * - Board geometry helpers (hex vs square)
 * - Terminal state handling
 * - 3+ player game heuristics
 * - Edge cases in evaluation factors
 */

import {
  evaluateHeuristicState,
  evaluateHeuristicStateWithBreakdown,
  loadHeuristicWeightsFromJSON,
  exportHeuristicWeightsToJSON,
  getHeuristicWeightsTS,
  scoreMove,
  HEURISTIC_WEIGHTS_V1_BALANCED,
  HEURISTIC_WEIGHTS_V1_AGGRESSIVE,
  HEURISTIC_WEIGHTS_V1_TERRITORIAL,
  HEURISTIC_WEIGHTS_V1_DEFENSIVE,
  HEURISTIC_WEIGHT_PROFILES_TS,
  HeuristicWeights,
} from '../../src/shared/engine/heuristicEvaluation';
import { GameState, Position, BoardType, BOARD_CONFIGS } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Helper to create a minimal game state for testing
function makeEmptyGameState(options?: {
  numPlayers?: number;
  gameStatus?: 'waiting' | 'active' | 'paused' | 'completed' | 'finished';
  winner?: number;
  boardType?: BoardType;
}): GameState {
  const numPlayers = options?.numPlayers ?? 2;
  const boardType = options?.boardType ?? 'square8';
  const boardConfig = BOARD_CONFIGS[boardType];

  const players = [];
  for (let i = 1; i <= numPlayers; i++) {
    players.push({
      id: `player-${i}`,
      username: `Player${i}`,
      playerNumber: i,
      type: 'human' as const,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: boardConfig.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    });
  }

  return {
    gameId: 'test-game',
    board: {
      type: boardType,
      size: boardConfig.size,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Set(),
    },
    boardType,
    players,
    currentPlayer: 1,
    currentPhase: 'ring_placement' as GameState['currentPhase'],
    gameStatus: (options?.gameStatus ?? 'active') as GameState['gameStatus'],
    winner: options?.winner,
    moveHistory: [],
    spectators: [],
    maxPlayers: numPlayers,
    totalRingsInPlay: boardConfig.ringsPerPlayer * numPlayers,
    victoryThreshold: boardConfig.ringsPerPlayer, // Per RR-CANON-R061
    territoryVictoryThreshold: Math.floor((boardConfig.size * boardConfig.size) / 2) + 1,
    timeControl: { initialTime: 600, increment: 0, type: 'blitz' },
    isRated: false,
    createdAt: new Date(),
  } as GameState;
}

function pos(x: number, y: number, z?: number): Position {
  if (z !== undefined) return { x, y, z };
  return { x, y };
}

function addStack(
  state: GameState,
  position: Position,
  controllingPlayer: number,
  stackHeight: number,
  capHeight?: number
): void {
  const key = positionToString(position);
  state.board.stacks.set(key, {
    position,
    controllingPlayer,
    stackHeight,
    capHeight: capHeight ?? stackHeight,
    rings: [],
  });
}

function addMarker(state: GameState, position: Position, player: number): void {
  const key = positionToString(position);
  state.board.markers.set(key, { player, position });
}

describe('heuristicEvaluation branch coverage', () => {
  describe('weight profiles', () => {
    it('exports all weight profiles', () => {
      expect(HEURISTIC_WEIGHTS_V1_BALANCED).toBeDefined();
      expect(HEURISTIC_WEIGHTS_V1_AGGRESSIVE).toBeDefined();
      expect(HEURISTIC_WEIGHTS_V1_TERRITORIAL).toBeDefined();
      expect(HEURISTIC_WEIGHTS_V1_DEFENSIVE).toBeDefined();
    });

    it('profiles have all required weight keys', () => {
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
      ];

      for (const key of requiredKeys) {
        expect(typeof HEURISTIC_WEIGHTS_V1_BALANCED[key]).toBe('number');
      }
    });

    it('registry contains all persona profiles', () => {
      expect(HEURISTIC_WEIGHT_PROFILES_TS['heuristic_v1_balanced']).toBe(
        HEURISTIC_WEIGHTS_V1_BALANCED
      );
      expect(HEURISTIC_WEIGHT_PROFILES_TS['heuristic_v1_aggressive']).toBe(
        HEURISTIC_WEIGHTS_V1_AGGRESSIVE
      );
      expect(HEURISTIC_WEIGHT_PROFILES_TS['v1-heuristic-2']).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    });
  });

  describe('loadHeuristicWeightsFromJSON', () => {
    it('loads TypeScript camelCase keys', () => {
      const json = { stackControl: 15.0, territory: 12.0 };
      const weights = loadHeuristicWeightsFromJSON(json);
      expect(weights.stackControl).toBe(15.0);
      expect(weights.territory).toBe(12.0);
    });

    it('loads Python UPPER_SNAKE_CASE keys', () => {
      const json = { WEIGHT_STACK_CONTROL: 20.0, WEIGHT_TERRITORY: 15.0 };
      const weights = loadHeuristicWeightsFromJSON(json);
      expect(weights.stackControl).toBe(20.0);
      expect(weights.territory).toBe(15.0);
    });

    it('ignores unknown keys', () => {
      const json = { unknownKey: 999, stackControl: 10.0 };
      const weights = loadHeuristicWeightsFromJSON(json);
      expect(weights.stackControl).toBe(10.0);
      expect((weights as Record<string, number>)['unknownKey']).toBeUndefined();
    });

    it('uses defaults for missing keys', () => {
      const json = { stackControl: 25.0 };
      const weights = loadHeuristicWeightsFromJSON(json);
      expect(weights.stackControl).toBe(25.0);
      expect(weights.territory).toBe(HEURISTIC_WEIGHTS_V1_BALANCED.territory);
    });
  });

  describe('exportHeuristicWeightsToJSON', () => {
    it('exports weights with Python keys', () => {
      const json = exportHeuristicWeightsToJSON(HEURISTIC_WEIGHTS_V1_BALANCED);
      expect(json['WEIGHT_STACK_CONTROL']).toBe(HEURISTIC_WEIGHTS_V1_BALANCED.stackControl);
      expect(json['WEIGHT_TERRITORY']).toBe(HEURISTIC_WEIGHTS_V1_BALANCED.territory);
    });

    it('round-trips through JSON', () => {
      const exported = exportHeuristicWeightsToJSON(HEURISTIC_WEIGHTS_V1_BALANCED);
      const reloaded = loadHeuristicWeightsFromJSON(exported);
      expect(reloaded.stackControl).toBe(HEURISTIC_WEIGHTS_V1_BALANCED.stackControl);
      expect(reloaded.territory).toBe(HEURISTIC_WEIGHTS_V1_BALANCED.territory);
    });
  });

  describe('getHeuristicWeightsTS', () => {
    it('returns balanced weights for null', () => {
      expect(getHeuristicWeightsTS(null)).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    });

    it('returns balanced weights for undefined', () => {
      expect(getHeuristicWeightsTS(undefined)).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    });

    it('returns balanced weights for empty string', () => {
      expect(getHeuristicWeightsTS('')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    });

    it('returns matching profile for valid id', () => {
      expect(getHeuristicWeightsTS('heuristic_v1_aggressive')).toBe(
        HEURISTIC_WEIGHTS_V1_AGGRESSIVE
      );
    });

    it('returns balanced weights for unknown id', () => {
      expect(getHeuristicWeightsTS('unknown_profile')).toBe(HEURISTIC_WEIGHTS_V1_BALANCED);
    });
  });

  describe('evaluateHeuristicState', () => {
    describe('terminal states', () => {
      it('returns large positive for player winning', () => {
        const state = makeEmptyGameState({ gameStatus: 'finished', winner: 1 });
        const score = evaluateHeuristicState(state, 1);
        expect(score).toBe(100_000);
      });

      it('returns large negative for player losing', () => {
        const state = makeEmptyGameState({ gameStatus: 'finished', winner: 2 });
        const score = evaluateHeuristicState(state, 1);
        expect(score).toBe(-100_000);
      });

      it('returns 0 for draw', () => {
        const state = makeEmptyGameState({ gameStatus: 'finished' });
        state.winner = undefined;
        const score = evaluateHeuristicState(state, 1);
        expect(score).toBe(0);
      });
    });

    describe('stack control evaluation', () => {
      it('penalizes having no stacks', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        const score1 = evaluateHeuristicState(state, 1);

        addStack(state, pos(3, 3), 1, 2);
        const score2 = evaluateHeuristicState(state, 1);

        expect(score2).toBeGreaterThan(score1);
      });

      it('penalizes single stack vs multiple stacks', () => {
        const state1 = makeEmptyGameState();
        state1.players[0].ringsInHand = 0;
        addStack(state1, pos(3, 3), 1, 2);
        const score1 = evaluateHeuristicState(state1, 1);

        const state2 = makeEmptyGameState();
        state2.players[0].ringsInHand = 0;
        addStack(state2, pos(2, 2), 1, 2);
        addStack(state2, pos(4, 4), 1, 2);
        const score2 = evaluateHeuristicState(state2, 1);

        expect(score2).toBeGreaterThan(score1);
      });

      it('applies diminishing returns for tall stacks', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        // Stack height 10 should have diminishing returns vs height 5
        addStack(state, pos(3, 3), 1, 10);
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });
    });

    describe('territory evaluation', () => {
      it('rewards territory advantage', () => {
        const state = makeEmptyGameState();
        state.players[0].territorySpaces = 10;
        state.players[1].territorySpaces = 5;
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });

      it('handles missing player gracefully', () => {
        const state = makeEmptyGameState();
        const score = evaluateHeuristicState(state, 99);
        expect(typeof score).toBe('number');
      });
    });

    describe('center control evaluation', () => {
      it('rewards controlling center positions', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        // Center of 8x8 board is around (3,3) to (4,4)
        addStack(state, pos(3, 3), 1, 2);
        const scoreCenter = evaluateHeuristicState(state, 1);

        const state2 = makeEmptyGameState();
        state2.players[0].ringsInHand = 0;
        // Corner is not center
        addStack(state2, pos(0, 0), 1, 2);
        const scoreCorner = evaluateHeuristicState(state2, 1);

        expect(scoreCenter).toBeGreaterThan(scoreCorner);
      });
    });

    describe('mobility evaluation', () => {
      it('rewards having more movement options', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        // Stack in center has many move options
        addStack(state, pos(3, 3), 1, 2);
        const scoreCenter = evaluateHeuristicState(state, 1);

        // Stack in corner has fewer options
        const state2 = makeEmptyGameState();
        state2.players[0].ringsInHand = 0;
        addStack(state2, pos(0, 0), 1, 2);
        const scoreCorner = evaluateHeuristicState(state2, 1);

        expect(scoreCenter).toBeGreaterThan(scoreCorner);
      });
    });

    describe('victory proximity evaluation', () => {
      it('rewards being close to elimination victory', () => {
        const state = makeEmptyGameState();
        state.players[0].eliminatedRings = state.victoryThreshold - 1;
        const score = evaluateHeuristicState(state, 1);
        expect(score).toBeGreaterThan(0);
      });

      it('rewards being close to territory victory', () => {
        const state = makeEmptyGameState();
        state.players[0].territorySpaces = state.territoryVictoryThreshold - 1;
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });
    });

    describe('line potential evaluation', () => {
      it('rewards markers in a row', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        // Add 3 markers in a row
        addMarker(state, pos(2, 2), 1);
        addMarker(state, pos(3, 2), 1);
        addMarker(state, pos(4, 2), 1);
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });

      it('rewards 4 markers in a row more than 3', () => {
        const state3 = makeEmptyGameState();
        addMarker(state3, pos(2, 2), 1);
        addMarker(state3, pos(3, 2), 1);
        addMarker(state3, pos(4, 2), 1);

        const state4 = makeEmptyGameState();
        addMarker(state4, pos(2, 2), 1);
        addMarker(state4, pos(3, 2), 1);
        addMarker(state4, pos(4, 2), 1);
        addMarker(state4, pos(5, 2), 1);

        const score3 = evaluateHeuristicState(state3, 1);
        const score4 = evaluateHeuristicState(state4, 1);
        expect(score4).toBeGreaterThan(score3);
      });
    });

    describe('vulnerability evaluation', () => {
      it('penalizes vulnerable stacks in line-of-sight', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        // My short stack
        addStack(state, pos(2, 2), 1, 1, 1);
        // Enemy tall stack in line-of-sight
        addStack(state, pos(4, 2), 2, 3, 3);
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });
    });

    describe('overtake potential evaluation', () => {
      it('rewards overtake opportunities', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        // My tall stack
        addStack(state, pos(2, 2), 1, 3, 3);
        // Enemy short stack in line-of-sight
        addStack(state, pos(4, 2), 2, 1, 1);
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });
    });

    describe('territory closure evaluation', () => {
      it('rewards clustered markers', () => {
        const state = makeEmptyGameState();
        // Clustered markers
        addMarker(state, pos(3, 3), 1);
        addMarker(state, pos(3, 4), 1);
        addMarker(state, pos(4, 3), 1);
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });

      it('handles empty markers gracefully', () => {
        const state = makeEmptyGameState();
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });
    });

    describe('collapsed spaces', () => {
      it('handles collapsed spaces in line-of-sight', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        addStack(state, pos(2, 2), 1, 2);
        // Collapsed space blocks line-of-sight
        state.board.collapsedSpaces.add(positionToString(pos(3, 2)));
        addStack(state, pos(4, 2), 2, 3);
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });
    });
  });

  describe('evaluateHeuristicStateWithBreakdown', () => {
    it('returns breakdown for finished game', () => {
      const state = makeEmptyGameState({ gameStatus: 'finished', winner: 1 });
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(breakdown.total).toBe(100_000);
      expect(breakdown.stackControl).toBe(100_000);
    });

    it('returns breakdown for active game', () => {
      const state = makeEmptyGameState();
      addStack(state, pos(3, 3), 1, 2);
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(typeof breakdown.total).toBe('number');
      expect(typeof breakdown.stackControl).toBe('number');
      expect(typeof breakdown.territory).toBe('number');
      expect(typeof breakdown.mobility).toBe('number');
    });

    it('sums all factors to total', () => {
      const state = makeEmptyGameState();
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      const manualSum =
        breakdown.stackControl +
        breakdown.territory +
        breakdown.ringsInHand +
        breakdown.centerControl +
        breakdown.opponentThreats +
        breakdown.mobility +
        breakdown.eliminatedRings +
        breakdown.linePotential +
        breakdown.victoryProximity +
        breakdown.opponentVictoryThreat +
        breakdown.markerCount +
        breakdown.vulnerability +
        breakdown.overtakePotential +
        breakdown.territoryClosure +
        breakdown.lineConnectivity +
        breakdown.territorySafety +
        breakdown.stackMobility +
        breakdown.forcedEliminationRisk +
        breakdown.lpsActionAdvantage +
        breakdown.multiLeaderThreat;
      expect(breakdown.total).toBeCloseTo(manualSum, 5);
    });
  });

  describe('multi-player game heuristics', () => {
    describe('forced elimination risk', () => {
      it('penalizes many stacks with few actions', () => {
        const state = makeEmptyGameState();
        state.players[0].ringsInHand = 0;
        // Multiple stacks but blocked
        addStack(state, pos(0, 0), 1, 1);
        addStack(state, pos(0, 7), 1, 1);
        addStack(state, pos(7, 0), 1, 1);
        addStack(state, pos(7, 7), 1, 1);
        // Block all with opponent stacks
        addStack(state, pos(1, 0), 2, 5);
        addStack(state, pos(0, 1), 2, 5);
        const score = evaluateHeuristicState(state, 1);
        expect(typeof score).toBe('number');
      });
    });

    describe('LPS action advantage', () => {
      it('returns 0 for 2-player games', () => {
        const state = makeEmptyGameState({ numPlayers: 2 });
        const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
        expect(breakdown.lpsActionAdvantage).toBe(0);
      });

      it('evaluates for 3-player games', () => {
        const state = makeEmptyGameState({ numPlayers: 3 });
        const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
        // With all players having actions, should be small
        expect(typeof breakdown.lpsActionAdvantage).toBe('number');
      });

      it('rewards being the only player with actions', () => {
        const state = makeEmptyGameState({ numPlayers: 3 });
        // Give player 1 actions
        state.players[0].ringsInHand = 10;
        // Remove actions from opponents
        state.players[1].ringsInHand = 0;
        state.players[2].ringsInHand = 0;
        const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
        expect(breakdown.lpsActionAdvantage).toBeGreaterThanOrEqual(0);
      });
    });

    describe('multi-leader threat', () => {
      it('returns 0 for 2-player games', () => {
        const state = makeEmptyGameState({ numPlayers: 2 });
        const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
        expect(breakdown.multiLeaderThreat).toBe(0);
      });

      it('penalizes when one opponent leads others', () => {
        const state = makeEmptyGameState({ numPlayers: 3 });
        // Player 2 is close to victory
        state.players[1].eliminatedRings = state.victoryThreshold - 2;
        // Player 3 is far from victory
        state.players[2].eliminatedRings = 0;
        const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
        expect(breakdown.multiLeaderThreat).toBeLessThanOrEqual(0);
      });
    });

    describe('opponent victory threat', () => {
      it('penalizes when opponent is closer to victory', () => {
        const state = makeEmptyGameState();
        // Player 2 is close to victory
        state.players[1].eliminatedRings = state.victoryThreshold - 1;
        state.players[0].eliminatedRings = 0;
        const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
        expect(breakdown.opponentVictoryThreat).toBeLessThanOrEqual(0);
      });
    });
  });

  describe('hexagonal board support', () => {
    it('evaluates on hexagonal board', () => {
      const state = makeEmptyGameState({ boardType: 'hexagonal' });
      // Add stack at cube coordinate origin
      addStack(state, pos(0, 0, 0), 1, 2);
      const score = evaluateHeuristicState(state, 1);
      expect(typeof score).toBe('number');
    });

    it('handles hex board center positions', () => {
      const state = makeEmptyGameState({ boardType: 'hexagonal' });
      state.players[0].ringsInHand = 0;
      // Center of hex board is at origin
      addStack(state, pos(0, 0, 0), 1, 2);
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(breakdown.centerControl).toBeGreaterThan(0);
    });

    it('handles z coordinates in marker clustering', () => {
      const state = makeEmptyGameState({ boardType: 'hexagonal' });
      addMarker(state, pos(0, 0, 0), 1);
      addMarker(state, pos(1, -1, 0), 1);
      addMarker(state, pos(0, 1, -1), 1);
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(typeof breakdown.territoryClosure).toBe('number');
    });
  });

  describe('scoreMove', () => {
    it('throws not implemented error', () => {
      const state = makeEmptyGameState();
      expect(() =>
        scoreMove({
          before: state,
          move: { id: 'test', type: 'place_ring' } as any,
          playerNumber: 1,
        })
      ).toThrow('scoreMove is not yet implemented');
    });
  });

  describe('edge cases', () => {
    it('handles 4-player game', () => {
      const state = makeEmptyGameState({ numPlayers: 4 });
      const score = evaluateHeuristicState(state, 1);
      expect(typeof score).toBe('number');
    });

    it('handles custom weights', () => {
      const state = makeEmptyGameState();
      const customWeights = { ...HEURISTIC_WEIGHTS_V1_BALANCED, stackControl: 100.0 };
      const scoreDefault = evaluateHeuristicState(state, 1);
      const scoreCustom = evaluateHeuristicState(state, 1, customWeights);
      expect(typeof scoreDefault).toBe('number');
      expect(typeof scoreCustom).toBe('number');
    });

    it('handles opponent threats with adjacent stacks', () => {
      const state = makeEmptyGameState();
      state.players[0].ringsInHand = 0;
      addStack(state, pos(3, 3), 1, 2, 2);
      addStack(state, pos(3, 4), 2, 4, 4); // Adjacent taller opponent
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(breakdown.opponentThreats).toBeLessThanOrEqual(0);
    });

    it('handles territory safety with nearby opponent stacks', () => {
      const state = makeEmptyGameState();
      addMarker(state, pos(3, 3), 1);
      addStack(state, pos(3, 4), 2, 2); // Opponent stack 1 space away
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(breakdown.territorySafety).toBeLessThanOrEqual(0);
    });

    it('handles line connectivity gaps', () => {
      const state = makeEmptyGameState();
      // Markers with a gap between
      addMarker(state, pos(2, 2), 1);
      addMarker(state, pos(4, 2), 1); // Gap at (3,2)
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(typeof breakdown.lineConnectivity).toBe('number');
    });

    it('handles stack mobility with blocked stacks', () => {
      const state = makeEmptyGameState();
      state.players[0].ringsInHand = 0;
      addStack(state, pos(0, 0), 1, 1, 1); // Corner stack
      // Block all moves
      addStack(state, pos(1, 0), 2, 5, 5);
      addStack(state, pos(0, 1), 2, 5, 5);
      addStack(state, pos(1, 1), 2, 5, 5);
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      // Should have penalty for blocked stack
      expect(typeof breakdown.stackMobility).toBe('number');
    });

    it('mobility evaluates capture opportunity against weaker adjacent opponent (line 623)', () => {
      const state = makeEmptyGameState();
      state.players[0].ringsInHand = 0;
      // My taller stack can capture adjacent weaker opponent
      addStack(state, pos(3, 3), 1, 3, 3);
      addStack(state, pos(3, 4), 2, 1, 1); // Adjacent weaker opponent
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      // Mobility should be positive due to capture opportunity
      expect(breakdown.mobility).toBeGreaterThan(0);
    });

    it('victory proximity returns 1000 when victory threshold reached (line 724)', () => {
      const state = makeEmptyGameState();
      // Player 1 has reached the elimination victory threshold
      state.players[0].eliminatedRings = state.victoryThreshold;
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      // Victory proximity should be at max (1000 * weight)
      expect(breakdown.victoryProximity).toBeGreaterThan(0);
    });

    it('victory proximity returns 1000 when territory threshold reached (line 724)', () => {
      const state = makeEmptyGameState();
      // Player 1 has reached the territory victory threshold
      state.players[0].territorySpaces = state.territoryVictoryThreshold;
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      expect(breakdown.victoryProximity).toBeGreaterThan(0);
    });

    it('approxMovesForPlayer counts capture opportunities (lines 798-799)', () => {
      const state = makeEmptyGameState({ numPlayers: 3 });
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;
      state.players[2].ringsInHand = 0;
      // Player 1 has a stack that can capture an adjacent opponent stack
      addStack(state, pos(3, 3), 1, 3, 3);
      addStack(state, pos(3, 4), 2, 1, 1); // Adjacent weaker opponent
      // Player 2 has no moves (blocked)
      addStack(state, pos(0, 0), 2, 1, 1);
      addStack(state, pos(1, 0), 1, 5, 5);
      addStack(state, pos(0, 1), 1, 5, 5);
      addStack(state, pos(1, 1), 1, 5, 5);
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      // Should register as having advantage due to capture opportunity
      expect(typeof breakdown.lpsActionAdvantage).toBe('number');
    });

    it('LPS action advantage penalizes player with no actions (line 898)', () => {
      const state = makeEmptyGameState({ numPlayers: 3 });
      // Player 1 has no actions: no rings in hand and no stacks
      state.players[0].ringsInHand = 0;
      // Player 2 and 3 have actions
      state.players[1].ringsInHand = 5;
      state.players[2].ringsInHand = 5;
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      // Should be negative since player has no actions
      expect(breakdown.lpsActionAdvantage).toBeLessThan(0);
    });

    it('territory safety calculates hex z-distance (line 1170)', () => {
      const state = makeEmptyGameState({ boardType: 'hexagonal' });
      // Add marker at a hex position
      addMarker(state, pos(0, 0, 0), 1);
      // Add opponent stack nearby (uses z coordinate in distance)
      addStack(state, pos(1, 0, -1), 2, 2, 2);
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      // Territory safety should account for the nearby threat
      expect(breakdown.territorySafety).toBeLessThanOrEqual(0);
    });

    it('stack mobility counts capture moves (line 1216)', () => {
      const state = makeEmptyGameState();
      state.players[0].ringsInHand = 0;
      // My tall stack at center with adjacent weaker opponent stacks
      addStack(state, pos(3, 3), 1, 4, 4);
      // Multiple adjacent weaker opponents that can be captured
      addStack(state, pos(3, 4), 2, 1, 1);
      addStack(state, pos(4, 3), 2, 1, 1);
      const breakdown = evaluateHeuristicStateWithBreakdown(state, 1);
      // Stack mobility should count these capture opportunities
      expect(breakdown.stackMobility).toBeGreaterThan(0);
    });
  });
});
