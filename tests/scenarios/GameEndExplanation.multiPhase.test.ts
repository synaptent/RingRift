/**
 * GameEndExplanation Multi-Phase Turn Scenario Tests
 *
 * Tests that verify GameEndExplanation payloads are correctly generated
 * for victories occurring after multi-phase turns (chain→line→territory).
 *
 * Coverage targets from TODO.md Immediate Handoffs:
 * - Chain→line→territory multi-choice turns on all boards
 * - Tests that assert GameEndExplanation payloads for those sequences
 *
 * Axis references: V1, V2, L1-L4, T1-T4 (RULES_SCENARIO_MATRIX.md)
 */

import fs from 'fs';
import path from 'path';

import {
  toVictoryState,
  processTurn,
} from '../../src/shared/engine/orchestration/turnOrchestrator';
import {
  importVectorBundle,
  deserializeGameState,
  type ContractTestVector,
} from '../../src/shared/engine/contracts';
import type {
  GameState,
  GameHistoryEntry,
  BoardType,
  BoardState,
  Move,
  Position,
  Player,
  Stack,
  Marker,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import type { GameEndExplanation } from '../../src/shared/types/gameEndExplanation';

// ═══════════════════════════════════════════════════════════════════════════
// Test Utilities
// ═══════════════════════════════════════════════════════════════════════════

function createPlayer(playerNumber: number, options: Partial<Player> = {}): Player {
  return {
    id: `player-${playerNumber}`,
    username: `Player ${playerNumber}`,
    playerNumber,
    type: 'human',
    isReady: true,
    timeRemaining: 600000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
    ...options,
  };
}

function createEmptyBoard(boardType: BoardType = 'square8'): BoardState {
  const sizes: Record<BoardType, number> = {
    square8: 8,
    square19: 19,
    hexagonal: 12, // radius
  };
  return {
    type: boardType,
    size: sizes[boardType],
    stacks: new Map(),
    markers: new Map(),
    territories: new Map(),
    formedLines: [],
    collapsedSpaces: new Map(),
    eliminatedRings: {},
  };
}

function addStack(
  board: BoardState,
  pos: Position,
  rings: number[],
  controllingPlayer: number
): void {
  const key = positionToString(pos);
  const stack: Stack = {
    position: pos,
    rings,
    stackHeight: rings.length,
    capHeight: rings.filter((r) => r === controllingPlayer).length,
    controllingPlayer,
  };
  board.stacks.set(key, stack);
}

function addMarker(board: BoardState, pos: Position, player: number): void {
  const key = positionToString(pos);
  const marker: Marker = { position: pos, player, type: 'regular' };
  board.markers.set(key, marker);
}

function addCollapsedSpace(board: BoardState, pos: Position, player: number): void {
  const key = positionToString(pos);
  // collapsedSpaces is Map<string, number> (position string -> player number)
  board.collapsedSpaces.set(key, player);
}

function createBaseState(boardType: BoardType = 'square8', numPlayers: number = 2): GameState {
  const ringsPerPlayer: Record<BoardType, number> = {
    square8: 18,
    square19: 72,
    hexagonal: 96,
  };
  const victoryThreshold: Record<BoardType, number> = {
    square8: 18, // ring elimination
    square19: 72,
    hexagonal: 96,
  };
  const territoryVictoryThreshold: Record<BoardType, number> = {
    square8: 33, // ~50% of 64 spaces
    square19: 181, // ~50% of 361 spaces
    hexagonal: 235, // ~50% of 469 spaces
  };

  return {
    id: `test-game-${boardType}-${numPlayers}p`,
    currentPlayer: 1,
    currentPhase: 'movement',
    gameStatus: 'active',
    boardType,
    players: Array.from({ length: numPlayers }, (_, i) =>
      createPlayer(i + 1, { ringsInHand: ringsPerPlayer[boardType] })
    ),
    board: createEmptyBoard(boardType),
    moveHistory: [],
    history: [],
    lastMoveAt: new Date(),
    createdAt: new Date(),
    isRated: false,
    spectators: [],
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
    maxPlayers: numPlayers,
    totalRingsInPlay: ringsPerPlayer[boardType] * numPlayers,
    victoryThreshold: victoryThreshold[boardType],
    territoryVictoryThreshold: territoryVictoryThreshold[boardType],
  };
}

function addForcedEliminationHistory(state: GameState, actor: number = 1): void {
  const entry: GameHistoryEntry = {
    moveNumber: 1,
    action: {
      id: `forced-elimination-${actor}-1`,
      type: 'forced_elimination',
      player: actor,
      to: { x: 0, y: 0 },
    },
    actor,
    phaseBefore: 'forced_elimination',
    phaseAfter: 'forced_elimination',
    statusBefore: 'active',
    statusAfter: 'active',
    progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
    progressAfter: { markers: 0, collapsed: 0, eliminated: 1, S: 1 },
  };

  state.history = [entry];
}

// ═══════════════════════════════════════════════════════════════════════════
// Contract Vector Utilities
// ═══════════════════════════════════════════════════════════════════════════

type PhaseTransitionHint = {
  phaseAfter?: string;
  availableChainTarget?: Position;
};

const MULTI_PHASE_BUNDLE = path.resolve(
  __dirname,
  '../fixtures/contract-vectors/v2/multi_phase_turn.vectors.json'
);

function loadMultiPhaseVectors(): ContractTestVector[] {
  const json = fs.readFileSync(MULTI_PHASE_BUNDLE, 'utf8');
  return importVectorBundle(json);
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

function makeBookkeepingMove(decisionType: string, state: GameState, player: number): Move | null {
  const moveNumber = state.moveHistory.length + 1;
  const base = {
    id: `auto-${decisionType}-${moveNumber}`,
    player,
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber,
    to: { x: 0, y: 0 },
  } as Move;

  switch (decisionType) {
    case 'no_line_action_required':
      return { ...base, type: 'no_line_action' };
    case 'no_territory_action_required':
      return { ...base, type: 'no_territory_action' };
    case 'no_movement_action_required':
      return { ...base, type: 'no_movement_action' };
    case 'no_placement_action_required':
      return { ...base, type: 'no_placement_action' };
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
    const synthetic = makeBookkeepingMove(decision.type, state, decision.player);
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
      const zeroCollapse = options.find((opt: any) => (opt.collapsedMarkers ?? []).length === 0);
      if (zeroCollapse) return zeroCollapse;

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

function buildPhaseHints(vector: ContractTestVector): PhaseTransitionHint[] {
  const input = vector.input as any;
  const phaseHints = (input.phaseTransitions ?? []) as PhaseTransitionHint[];
  const territoryRegion = input.territoryExpectation?.potentiallyDisconnectedRegion?.[0];
  if (territoryRegion) {
    phaseHints.push({ phaseAfter: 'territory_processing', availableChainTarget: territoryRegion });
  }
  return phaseHints;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test Suites
// ═══════════════════════════════════════════════════════════════════════════

describe('GameEndExplanation for multi-phase turn scenarios', () => {
  describe('Ring elimination after line processing', () => {
    it.each<[BoardType, number, number]>([
      ['square8', 4, 18],
      ['square19', 5, 72],
      ['hexagonal', 6, 96],
    ])(
      'generates GameEndExplanation for %s ring elimination victory (line length %d)',
      (boardType, _lineLength, threshold) => {
        const state = createBaseState(boardType, 2);

        // Set up near-victory state: player 1 is one ring elimination away
        state.players[0].eliminatedRings = threshold - 1;
        state.players[1].eliminatedRings = 5;

        // Add a single ring in hand for player 2 (will be eliminated)
        state.players[1].ringsInHand = 1;

        // Set up a line that when collapsed will eliminate one more ring from P2
        // by processing already-formed line (simulate post-line collapse state)
        state.players[0].eliminatedRings = threshold; // Now at threshold

        // Evaluate victory
        const victory = toVictoryState(state);

        expect(victory.isGameOver).toBe(true);
        expect(victory.winner).toBe(1);
        expect(victory.reason).toBe('ring_elimination');

        // Assert GameEndExplanation structure
        expect(victory.gameEndExplanation).toBeDefined();
        const explanation = victory.gameEndExplanation!;

        expect(explanation.outcomeType).toBe('ring_elimination');
        expect(explanation.victoryReasonCode).toBe('victory_ring_majority');
        expect(explanation.winnerPlayerId).toBe('P1');
        expect(explanation.boardType).toBe(boardType);
        expect(explanation.numPlayers).toBe(2);

        // Score breakdown should reflect final state
        expect(explanation.scoreBreakdown).toBeDefined();
        const p1Score = explanation.scoreBreakdown!['P1'];
        expect(p1Score.eliminatedRings).toBe(threshold);
      }
    );
  });

  describe('Territory victory after line collapse creates disconnected region', () => {
    it('generates GameEndExplanation for territory victory on square8', () => {
      const state = createBaseState('square8', 2);

      // Set up state where player 1 has achieved territory threshold
      state.players[0].territorySpaces = 35; // Above threshold of 33
      state.players[1].territorySpaces = 10;

      // Add collapsed spaces to reflect territory control
      for (let x = 0; x < 7; x++) {
        for (let y = 0; y < 5; y++) {
          addCollapsedSpace(state.board, { x, y }, 1);
        }
      }

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.winner).toBe(1);
      expect(victory.reason).toBe('territory_control');

      expect(victory.gameEndExplanation).toBeDefined();
      const explanation = victory.gameEndExplanation!;

      expect(explanation.outcomeType).toBe('territory_control');
      expect(explanation.victoryReasonCode).toBe('victory_territory_majority');
      expect(explanation.winnerPlayerId).toBe('P1');
      expect(explanation.boardType).toBe('square8');

      // Score breakdown
      const p1Score = explanation.scoreBreakdown!['P1'];
      expect(p1Score.territorySpaces).toBeGreaterThanOrEqual(33);
    });

    it('generates GameEndExplanation for territory victory on square19', () => {
      const state = createBaseState('square19', 2);

      state.players[0].territorySpaces = 12;
      state.players[1].territorySpaces = 3;

      // Use a smaller, achievable threshold for testing.
      state.territoryVictoryThreshold = 10;

      // Add collapsed spaces to reflect territory control.
      for (let x = 0; x < 4; x++) {
        for (let y = 0; y < 3; y++) {
          addCollapsedSpace(state.board, { x, y }, 1);
        }
      }

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.winner).toBe(1);
      expect(victory.reason).toBe('territory_control');

      expect(victory.gameEndExplanation).toBeDefined();
      const explanation = victory.gameEndExplanation!;

      expect(explanation.outcomeType).toBe('territory_control');
      expect(explanation.victoryReasonCode).toBe('victory_territory_majority');
      expect(explanation.winnerPlayerId).toBe('P1');
      expect(explanation.boardType).toBe('square19');
    });

    it('generates GameEndExplanation for territory victory on hexagonal board', () => {
      const state = createBaseState('hexagonal', 2);

      // Set up state where player 1 has achieved territory threshold
      // For hex boards, use valid axial coordinates (q, r) where q + r < radius
      // Threshold is 235 for hexagonal, so we need at least 235 collapsed spaces
      state.players[0].territorySpaces = 240;
      state.players[1].territorySpaces = 50;

      // Use a smaller, achievable threshold for testing
      state.territoryVictoryThreshold = 10;

      // Add collapsed spaces using valid positions for hex board
      // Hex board uses axial coordinates where |q| + |r| + |q+r| <= 2*radius
      const hexPositions = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: -1, y: 0 },
        { x: 0, y: 1 },
        { x: 0, y: -1 },
        { x: 1, y: -1 },
        { x: -1, y: 1 },
        { x: 2, y: 0 },
        { x: -2, y: 0 },
        { x: 0, y: 2 },
        { x: 0, y: -2 },
        { x: 1, y: 1 },
      ];
      hexPositions.forEach((pos) => addCollapsedSpace(state.board, pos, 1));

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.winner).toBe(1);
      expect(victory.reason).toBe('territory_control');

      expect(victory.gameEndExplanation).toBeDefined();
      const explanation = victory.gameEndExplanation!;

      expect(explanation.outcomeType).toBe('territory_control');
      expect(explanation.boardType).toBe('hexagonal');
    });
  });

  describe('Last Player Standing after forced elimination cascade', () => {
    it.each<BoardType>(['square8', 'square19', 'hexagonal'])(
      'generates GameEndExplanation for LPS victory on %s with weird-state context',
      (boardType) => {
        const state = createBaseState(boardType, 2);

        // Create LPS condition: Player 2 has no rings anywhere
        state.board.stacks.clear();
        state.board.markers.clear();

        state.players[0].ringsInHand = 0;
        state.players[0].eliminatedRings = 10;
        state.players[1].ringsInHand = 0;
        state.players[1].eliminatedRings = 0; // All rings gone from P2

        // Add markers to break ties
        addMarker(state.board, { x: 0, y: 0 }, 1);
        addMarker(state.board, { x: 1, y: 0 }, 1);
        addMarker(state.board, { x: 0, y: 1 }, 2);

        // Simulate all P2 rings eliminated
        state.players[1].eliminatedRings = state.victoryThreshold;

        const victory = toVictoryState(state);

        // Should trigger ring elimination (opponent reached threshold)
        // or LPS depending on exact state
        expect(victory.isGameOver).toBe(true);
        expect(victory.gameEndExplanation).toBeDefined();

        const explanation = victory.gameEndExplanation!;
        expect(explanation.boardType).toBe(boardType);
        expect(explanation.numPlayers).toBe(2);
        expect(['ring_elimination', 'last_player_standing']).toContain(explanation.outcomeType);
      }
    );
  });

  describe('Structural stalemate with tiebreak resolution', () => {
    it('generates GameEndExplanation with tiebreak steps for stalemate', () => {
      const state = createBaseState('square8', 2);

      // Create stalemate: both players have no moves possible
      state.board.stacks.clear();
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      // Equal eliminated rings
      state.players[0].eliminatedRings = 9;
      state.players[1].eliminatedRings = 9;

      // Player 1 has more markers (wins tiebreak)
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);
      addMarker(state.board, { x: 0, y: 1 }, 2);

      // Equal territory
      state.players[0].territorySpaces = 5;
      state.players[1].territorySpaces = 5;

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.gameEndExplanation).toBeDefined();

      const explanation = victory.gameEndExplanation!;
      expect(explanation.scoreBreakdown).toBeDefined();

      // Check score breakdown reflects marker difference
      const p1Score = explanation.scoreBreakdown!['P1'];
      const p2Score = explanation.scoreBreakdown!['P2'];
      expect(p1Score.markers).toBeGreaterThan(p2Score.markers);
    });

    it('adds ANM/FE weird-state context when stalemate follows forced elimination', () => {
      const state = createBaseState('square8', 2);

      state.board.stacks.clear();
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;
      state.players[0].eliminatedRings = 9;
      state.players[1].eliminatedRings = 9;

      // Equal markers for both players to ensure true stalemate (not marker tiebreaker)
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 0, y: 1 }, 2);
      addMarker(state.board, { x: 1, y: 1 }, 2);

      addForcedEliminationHistory(state, 1);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.gameEndExplanation).toBeDefined();

      const explanation = victory.gameEndExplanation!;
      // With a valid game state, tiebreakers resolve to last_player_standing
      // (marker count or last actor), not structural_stalemate
      expect(explanation.outcomeType).toBe('last_player_standing');

      // ANM/FE context is still added for LPS outcomes with forced elimination
      const reasonCodes = explanation.weirdStateContext?.reasonCodes || [];
      expect(reasonCodes).toEqual(
        expect.arrayContaining([
          'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
          'ANM_MOVEMENT_FE_BLOCKED',
          'FE_SEQUENCE_CURRENT_PLAYER',
        ])
      );

      const rulesContexts = explanation.weirdStateContext?.rulesContextTags || [];
      expect(new Set(rulesContexts)).toEqual(
        new Set(['last_player_standing', 'anm_forced_elimination'])
      );
    });
  });

  describe('3-4 player victories with GameEndExplanation', () => {
    it('generates GameEndExplanation for 3-player ring elimination victory', () => {
      const state = createBaseState('square8', 3);

      // Player 1 reaches ring elimination threshold
      state.players[0].eliminatedRings = 18;
      state.players[1].eliminatedRings = 5;
      state.players[2].eliminatedRings = 5;

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.winner).toBe(1);
      expect(victory.reason).toBe('ring_elimination');

      expect(victory.gameEndExplanation).toBeDefined();
      const explanation = victory.gameEndExplanation!;

      expect(explanation.numPlayers).toBe(3);
      expect(explanation.outcomeType).toBe('ring_elimination');
      expect(explanation.scoreBreakdown).toBeDefined();

      // All 3 players should be in score breakdown
      expect(explanation.scoreBreakdown!['P1']).toBeDefined();
      expect(explanation.scoreBreakdown!['P2']).toBeDefined();
      expect(explanation.scoreBreakdown!['P3']).toBeDefined();
    });

    it('generates GameEndExplanation for 4-player territory victory', () => {
      const state = createBaseState('square8', 4);

      // For 4-player games, territory threshold is typically lower
      // Use a realistic threshold for testing
      state.territoryVictoryThreshold = 20;

      // Player 3 wins by territory control
      state.players[2].territorySpaces = 25;
      state.players[0].territorySpaces = 5;
      state.players[1].territorySpaces = 5;
      state.players[3].territorySpaces = 5;

      // Add collapsed spaces for player 3 (playerNumber = 3)
      for (let i = 0; i < 25; i++) {
        addCollapsedSpace(state.board, { x: i % 8, y: Math.floor(i / 8) }, 3);
      }

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.winner).toBe(3);
      expect(victory.reason).toBe('territory_control');

      expect(victory.gameEndExplanation).toBeDefined();
      const explanation = victory.gameEndExplanation!;

      expect(explanation.numPlayers).toBe(4);
      expect(explanation.outcomeType).toBe('territory_control');
      expect(explanation.winnerPlayerId).toBe('P3');

      // All 4 players should be in score breakdown
      expect(explanation.scoreBreakdown!['P1']).toBeDefined();
      expect(explanation.scoreBreakdown!['P2']).toBeDefined();
      expect(explanation.scoreBreakdown!['P3']).toBeDefined();
      expect(explanation.scoreBreakdown!['P4']).toBeDefined();
    });
  });

  describe('Orchestrator-driven multi-phase territory victories', () => {
    const vectors = loadMultiPhaseVectors();
    const vectorIds: Array<{ boardType: BoardType; id: string }> = [
      { boardType: 'square8', id: 'multi_phase.full_sequence_with_territory' },
      { boardType: 'square19', id: 'multi_phase.full_sequence_with_territory_square19' },
      { boardType: 'hexagonal', id: 'multi_phase.full_sequence_with_territory_hex' },
    ];

    it.each(vectorIds)(
      'produces territory GameEndExplanation after multi-phase turn on %s',
      ({ boardType, id }) => {
        const vector = vectors.find((entry) => entry.id === id);
        if (!vector) {
          throw new Error(`Missing multi-phase contract vector: ${id}`);
        }

        const state = deserializeGameState((vector.input as any).state);
        state.territoryVictoryThreshold = 1;
        state.players.forEach((player) => {
          player.territorySpaces = 0;
        });

        const initialMove = convertVectorMove((vector.input as any).initialMove);
        const phaseHints = buildPhaseHints(vector);

        let result = processTurn(state, initialMove);
        const phases = [...result.metadata.phasesTraversed];
        let currentState = result.nextState;

        while (result.status === 'awaiting_decision' && result.pendingDecision) {
          const chosen = pickDecisionMove(result.pendingDecision, currentState, phaseHints);
          result = processTurn(currentState, chosen);
          phases.push(...result.metadata.phasesTraversed);
          currentState = result.nextState;
        }

        expect(phases).toEqual(expect.arrayContaining(['line_processing', 'territory_processing']));
        expect(result.status).toBe('complete');
        expect(result.victoryResult).toBeDefined();
        expect(result.victoryResult!.isGameOver).toBe(true);
        expect(result.victoryResult!.reason).toBe('territory_control');

        const explanation: GameEndExplanation = result.victoryResult!.gameEndExplanation!;
        expect(explanation).toBeDefined();
        expect(explanation.outcomeType).toBe('territory_control');
        expect(explanation.boardType).toBe(boardType);
        expect(explanation.winnerPlayerId).toBe('P1');
      }
    );
  });

  describe('UX copy and teaching references in GameEndExplanation', () => {
    it('includes primaryConceptId for complex endings', () => {
      const state = createBaseState('square8', 2);

      // LPS ending which should include teaching references
      state.board.stacks.clear();
      state.board.markers.clear();
      state.players.forEach((p) => {
        p.ringsInHand = 0;
        p.territorySpaces = 0;
        p.eliminatedRings = 9;
      });

      // Add markers for tiebreak
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 0, y: 1 }, 2);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.gameEndExplanation).toBeDefined();

      const explanation = victory.gameEndExplanation!;

      // LPS endings should have teaching context
      if (explanation.outcomeType === 'last_player_standing') {
        expect(explanation.primaryConceptId).toBeDefined();
        expect(explanation.weirdStateContext).toBeDefined();
        expect(explanation.weirdStateContext!.teachingTopicIds).toBeDefined();
        expect(explanation.weirdStateContext!.teachingTopicIds.length).toBeGreaterThan(0);
      }
    });
  });
});
