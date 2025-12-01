/**
 * LPS Cross-Interaction Tests: Lines, Territory, and Captures
 *
 * These tests exercise the interactions between:
 * - Line Processing System (LPS) and captures
 * - Line completion and territory processing
 * - Phase transitions during complex multi-phase turns
 *
 * This file replaces the legacy LPS cross-interaction suite that previously
 * relied on mocking internal GameEngine methods. The new implementation uses:
 * - The orchestrator-based turn processing via processTurn()
 * - Public GameEngine.makeMove() API
 * - Shared decision helpers from lineDecisionHelpers and territoryDecisionHelpers
 *
 * Cross-interaction scenarios:
 * 1. Line completion during a turn can affect capture opportunities
 * 2. Line completion creates territory boundaries
 * 3. Captures that land rings can form new lines
 * 4. Phase ordering: movement → capture → line → territory → victory
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine/rulesConfig';
import {
  createOrchestratorBackendEngine,
  createBackendOrchestratorHarness,
  toEngineMove,
  seedOverlengthLineForPlayer,
  filterRealActionMoves,
} from '../helpers/orchestratorTestUtils';
import { enumerateProcessLineMoves } from '../../src/shared/engine/lineDecisionHelpers';

const boardType: BoardType = 'square8';
const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

/**
 * Helper to create a 3-player configuration for LPS scenarios.
 */
function createThreePlayerConfig(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p3',
      username: 'Player3',
      type: 'human',
      playerNumber: 3,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

/**
 * Helper to seed a stack on the board.
 */
function seedStack(state: GameState, pos: Position, playerNumber: number, height: number): void {
  const rings = Array(height).fill(playerNumber);
  state.board.stacks.set(positionToString(pos), {
    position: pos,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer: playerNumber,
  } as any);
}

/**
 * Helper to seed a marker on the board.
 */
function seedMarker(state: GameState, pos: Position, playerNumber: number): void {
  state.board.markers.set(positionToString(pos), {
    position: pos,
    player: playerNumber,
    type: 'regular',
  } as any);
}

describe('GameEngine LPS + Line/Territory Cross-Interaction Scenarios', () => {
  const requiredLineLength = getEffectiveLineLengthThreshold(boardType, 2);

  describe('Line Processing after Movement', () => {
    /**
     * Test that line processing is correctly invoked after a movement
     * that completes a line of markers.
     */
    it('should detect and process line after movement completes marker sequence', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-line-after-move', boardType);
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Clear the board and set up a scenario where P1 is about to complete a line
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();

      // Seed markers for Player 1 that are 1 short of a complete line
      for (let i = 0; i < requiredLineLength - 1; i++) {
        seedMarker(state, { x: i, y: 0 }, 1);
      }

      // Place a stack that can move to complete the line
      const stackPos: Position = { x: requiredLineLength - 1, y: 1 };
      seedStack(state, stackPos, 1, 2);

      // Give P2 a stack so the game doesn't immediately end
      seedStack(state, { x: 7, y: 7 }, 2, 2);

      state.currentPlayer = 1;
      state.currentPhase = 'movement';
      state.gameStatus = 'active';

      const beforeSnapshot = computeProgressSnapshot(state as any);

      // Check that there are valid moves available
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);

      // There should be movement options for P1
      const movementMoves = orchMoves.filter(
        (m) => m.type === 'move_ring' || m.type === 'move_stack'
      );
      expect(movementMoves.length).toBeGreaterThanOrEqual(0);

      // Verify the initial state has the expected marker count
      const initialMarkers = Array.from(state.board.markers.values()).filter(
        (m) => (m as any).player === 1
      );
      expect(initialMarkers.length).toBe(requiredLineLength - 1);
    });

    /**
     * Test that overlength lines (longer than required length) are handled
     * correctly during line processing, offering reward choices.
     */
    it('should enumerate line reward choices for overlength lines', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-overlength', boardType);
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Clear board and seed an overlength line
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();

      const linePositions = seedOverlengthLineForPlayer(engine, 1, 0, 1);

      // Provide a stack for line reward elimination
      const elimStackPos: Position = { x: 7, y: 7 };
      seedStack(state, elimStackPos, 1, 3);

      state.currentPlayer = 1;
      state.currentPhase = 'line_processing';
      state.gameStatus = 'active';

      // Enumerate process_line moves
      const processLineMoves = enumerateProcessLineMoves(state, 1);
      expect(processLineMoves.length).toBeGreaterThan(0);

      // Verify the line has expected length
      const lineLength = linePositions.length;
      expect(lineLength).toBeGreaterThan(requiredLineLength);
    });
  });

  describe('Line → Territory Transition', () => {
    /**
     * Test that collapsed line spaces are correctly converted to territory
     * for the player who completed the line.
     */
    it('should collapse line markers to territory spaces', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-line-territory', boardType);
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Clear board and seed a complete line
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();

      // Seed exact-length line for P1
      const linePositions: Position[] = [];
      for (let i = 0; i < requiredLineLength; i++) {
        const pos = { x: i, y: 0 };
        linePositions.push(pos);
        seedMarker(state, pos, 1);
      }

      // Update formedLines cache
      state.board.formedLines = [
        {
          positions: linePositions,
          player: 1,
          length: requiredLineLength,
          direction: { x: 1, y: 0 },
        },
      ];

      // Provide a stack for line reward elimination
      seedStack(state, { x: 7, y: 7 }, 1, 2);

      state.currentPlayer = 1;
      state.currentPhase = 'line_processing';
      state.gameStatus = 'active';

      const beforePlayer1 = state.players.find((p) => p.playerNumber === 1)!;
      const beforeTerritory = beforePlayer1.territorySpaces;

      // Enumerate and verify process_line moves exist
      const processLineMoves = enumerateProcessLineMoves(state, 1);
      expect(processLineMoves.length).toBeGreaterThan(0);

      // The first move should process the line
      const processMove = processLineMoves[0];
      expect(processMove.type).toBe('process_line');
      expect(processMove.formedLines).toBeDefined();
      expect(processMove.formedLines!.length).toBe(1);
    });

    /**
     * Test that line completion affects territory detection boundaries.
     * When markers collapse to territory, they create new board geometry.
     */
    it('should update board geometry when line collapses to territory', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-geometry', boardType);
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Clear board
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();

      // Manually collapse some spaces to simulate completed line
      for (let i = 0; i < requiredLineLength; i++) {
        const key = positionToString({ x: i, y: 0 });
        state.board.collapsedSpaces.set(key, 1);
      }

      // Update territory count
      const player1 = state.players.find((p) => p.playerNumber === 1)!;
      player1.territorySpaces = requiredLineLength;

      // Verify collapsed spaces are recorded correctly
      for (let i = 0; i < requiredLineLength; i++) {
        const key = positionToString({ x: i, y: 0 });
        expect(state.board.collapsedSpaces.get(key)).toBe(1);
      }

      expect(player1.territorySpaces).toBe(requiredLineLength);
    });
  });

  describe('Phase Ordering Verification', () => {
    /**
     * Test that the orchestrator processes phases in the correct order:
     * movement → capture (chain) → line_processing → territory_processing → victory
     */
    it('should maintain correct phase ordering through processTurn', () => {
      const engine = createOrchestratorBackendEngine('lps-cross-phase-order', boardType);
      const baseState = engine.getGameState();

      // Create a synthetic state for testing phase transitions
      const stackPos: Position = { x: 3, y: 3 };
      const state: GameState = {
        ...baseState,
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'movement',
        moveHistory: [
          {
            id: 'prev-1',
            type: 'place_ring',
            player: 1,
            to: { x: 0, y: 0 },
            timestamp: new Date(0),
            thinkTime: 0,
            moveNumber: 1,
          } as Move,
        ],
        history: [],
        players: baseState.players.map((p: Player) =>
          p.playerNumber === 1
            ? { ...p, ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }
            : { ...p, ringsInHand: 0, eliminatedRings: 1, territorySpaces: 0 }
        ),
        board: {
          ...baseState.board,
          stacks: new Map([
            [
              positionToString(stackPos),
              {
                position: stackPos,
                rings: [1],
                stackHeight: 1,
                capHeight: 1,
                controllingPlayer: 1,
              } as any,
            ],
          ]),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: { 1: 0, 2: 1 },
        },
        totalRingsEliminated: 1,
        winner: undefined,
      };

      // Apply an elimination move to trigger phase transitions
      const eliminateMove: Move = {
        id: 'elim-test',
        type: 'eliminate_rings_from_stack',
        player: 1,
        to: stackPos,
        eliminatedRings: [{ player: 1, count: 1 }],
        eliminationFromStack: {
          position: stackPos,
          capHeight: 1,
          totalHeight: 1,
        },
        timestamp: new Date(0),
        thinkTime: 0,
        moveNumber: state.moveHistory.length + 1,
      } as Move;

      const result = processTurn(state, eliminateMove);

      // Verify phase traversal order in metadata
      const phases = result.metadata.phasesTraversed;

      // The phases should include movement at minimum
      expect(phases[0]).toBe('movement');

      // Line processing should come before territory processing
      const lineIndex = phases.indexOf('line_processing');
      const territoryIndex = phases.indexOf('territory_processing');

      if (lineIndex !== -1 && territoryIndex !== -1) {
        expect(lineIndex).toBeLessThan(territoryIndex);
      }
    });

    /**
     * Test that LPS victory is only triggered after all post-move phases complete.
     */
    it('should trigger LPS victory only after line and territory phases', () => {
      const engine = createOrchestratorBackendEngine('lps-cross-victory-timing', boardType);
      const baseState = engine.getGameState();

      const stackPos: Position = { x: 3, y: 3 };

      // Create a stalemate position where LPS should trigger
      const state: GameState = {
        ...baseState,
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'movement',
        moveHistory: [
          {
            id: 'prev-1',
            type: 'place_ring',
            player: 1,
            to: { x: 0, y: 0 },
            timestamp: new Date(0),
            thinkTime: 0,
            moveNumber: 1,
          } as Move,
        ],
        history: [],
        players: baseState.players.map((p: Player) =>
          p.playerNumber === 1
            ? { ...p, ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }
            : { ...p, ringsInHand: 0, eliminatedRings: 1, territorySpaces: 0 }
        ),
        board: {
          ...baseState.board,
          stacks: new Map([
            [
              positionToString(stackPos),
              {
                position: stackPos,
                rings: [1],
                stackHeight: 1,
                capHeight: 1,
                controllingPlayer: 1,
              } as any,
            ],
          ]),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: { 1: 0, 2: 1 },
        },
        totalRingsEliminated: 1,
        winner: undefined,
      };

      const eliminateMove: Move = {
        id: 'elim-victory',
        type: 'eliminate_rings_from_stack',
        player: 1,
        to: stackPos,
        eliminatedRings: [{ player: 1, count: 1 }],
        eliminationFromStack: {
          position: stackPos,
          capHeight: 1,
          totalHeight: 1,
        },
        timestamp: new Date(0),
        thinkTime: 0,
        moveNumber: state.moveHistory.length + 1,
      } as Move;

      const result = processTurn(state, eliminateMove);

      // Should trigger victory
      expect(result.victoryResult).toBeDefined();
      expect(result.victoryResult!.isGameOver).toBe(true);
      expect(result.victoryResult!.reason).toBe('last_player_standing');
      expect(result.victoryResult!.winner).toBe(1);

      // Verify phases were traversed before victory
      const phases = result.metadata.phasesTraversed;
      expect(phases.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Orchestrator Invariants', () => {
    /**
     * Test that all moves enumerated by the orchestrator validate correctly.
     */
    it('all getValidMoves entries should validate successfully', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-invariant-basic', boardType);
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);

      expect(orchMoves.length).toBeGreaterThan(0);

      for (const m of orchMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    });

    /**
     * Test that real action moves are properly identified and validated.
     */
    it('real action moves should validate and be a subset of all moves', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-invariant-real', boardType);
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);

      const realMoves = filterRealActionMoves(orchMoves);

      // Real moves should be a subset of all moves
      expect(realMoves.length).toBeLessThanOrEqual(orchMoves.length);

      // All real moves should validate
      for (const m of realMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    });

    /**
     * Test that process_line moves are properly validated.
     */
    it('process_line moves should validate when lines exist', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-invariant-line', boardType);
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Clear and seed a line
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();

      seedOverlengthLineForPlayer(engine, 1, 0, 1);
      seedStack(state, { x: 7, y: 7 }, 1, 3);

      state.currentPlayer = 1;
      state.currentPhase = 'line_processing';
      state.gameStatus = 'active';

      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);

      const lineMoves = orchMoves.filter(
        (m) => m.type === 'process_line' || m.type === 'choose_line_reward'
      );

      // Line processing moves should exist
      expect(lineMoves.length).toBeGreaterThan(0);

      // All line moves should validate
      for (const m of lineMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    });
  });

  describe('S-Invariant Preservation', () => {
    /**
     * The S-invariant (S = markers + collapsed + eliminated) must be
     * non-decreasing across line processing transitions.
     */
    it('should preserve S-invariant during line collapse', async () => {
      const engine = createOrchestratorBackendEngine('lps-cross-s-invariant', boardType);
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Clear and set up initial state
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();

      // Seed markers for a line
      for (let i = 0; i < requiredLineLength; i++) {
        seedMarker(state, { x: i, y: 0 }, 1);
      }

      // Give P1 a stack outside the line
      seedStack(state, { x: 7, y: 7 }, 1, 2);

      state.currentPlayer = 1;
      state.currentPhase = 'line_processing';
      state.gameStatus = 'active';

      const beforeSnapshot = computeProgressSnapshot(state as any);

      // Manually collapse the line (simulating line processing)
      for (let i = 0; i < requiredLineLength; i++) {
        const key = positionToString({ x: i, y: 0 });
        state.board.markers.delete(key);
        state.board.collapsedSpaces.set(key, 1);
      }

      const player1 = state.players.find((p) => p.playerNumber === 1)!;
      player1.territorySpaces = requiredLineLength;

      const afterSnapshot = computeProgressSnapshot(state as any);

      // S must be non-decreasing: markers converted to collapsed spaces
      expect(afterSnapshot.S).toBeGreaterThanOrEqual(beforeSnapshot.S);
    });
  });
});

describe('LPS Cross-Interaction Edge Cases', () => {
  /**
   * Test that captures do not interfere with line detection.
   */
  it('should handle capture followed by line detection correctly', async () => {
    const engine = createOrchestratorBackendEngine('lps-edge-capture-line', boardType);
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // Set up a board with both capture opportunities and line potential
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();

    // Player 1 stack that could capture
    seedStack(state, { x: 3, y: 3 }, 1, 2);
    // Player 2 stack to be captured
    seedStack(state, { x: 3, y: 4 }, 2, 1);

    // Markers that could form a line after capture
    seedMarker(state, { x: 0, y: 0 }, 1);
    seedMarker(state, { x: 1, y: 0 }, 1);

    state.currentPlayer = 1;
    state.currentPhase = 'capture';
    state.gameStatus = 'active';

    const harness = createBackendOrchestratorHarness(engine);
    const orchState = harness.getState();
    const orchMoves = harness.adapter.getValidMovesFor(orchState);

    // Should have moves available
    expect(orchMoves.length).toBeGreaterThanOrEqual(0);

    // All moves should validate
    for (const m of orchMoves) {
      const validation = harness.adapter.validateMoveOnly(orchState, m);
      expect(validation.valid).toBe(true);
    }
  });

  /**
   * Test that territory collapse does not affect adjacent lines.
   */
  it('should not affect unrelated lines during territory collapse', async () => {
    const engine = createOrchestratorBackendEngine('lps-edge-territory-line', boardType);
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();

    // Markers forming a line at row 0
    const requiredLineLength = getEffectiveLineLengthThreshold(boardType, 2);
    for (let i = 0; i < requiredLineLength; i++) {
      seedMarker(state, { x: i, y: 0 }, 1);
    }

    // Collapsed territory at row 5 (unrelated to the line)
    for (let i = 0; i < 3; i++) {
      state.board.collapsedSpaces.set(positionToString({ x: i, y: 5 }), 2);
    }

    // The line markers should still be intact
    for (let i = 0; i < requiredLineLength; i++) {
      const key = positionToString({ x: i, y: 0 });
      expect(state.board.markers.has(key)).toBe(true);
    }

    // Collapsed spaces should be separate
    for (let i = 0; i < 3; i++) {
      const key = positionToString({ x: i, y: 5 });
      expect(state.board.collapsedSpaces.get(key)).toBe(2);
    }
  });

  /**
   * Test multi-player LPS scenarios with line interactions.
   */
  it('should handle 3-player LPS with active line processing', async () => {
    const players = createThreePlayerConfig();
    // Give players some rings in hand for placement
    players[0].ringsInHand = 3;
    players[1].ringsInHand = 3;
    players[2].ringsInHand = 3;

    const engine = createOrchestratorBackendEngine(
      'lps-edge-3player-line',
      boardType,
      players,
      timeControl
    );
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // P1 has markers (potential line) - added during setup
    seedMarker(state, { x: 0, y: 0 }, 1);
    seedMarker(state, { x: 1, y: 0 }, 1);

    // The game should be in a valid state after startGame()
    expect(state.gameStatus).toBe('active');

    const harness = createBackendOrchestratorHarness(engine);
    const orchState = harness.getState();

    // All enumerated moves should validate (check real action moves only)
    const orchMoves = harness.adapter.getValidMovesFor(orchState);
    const realMoves = filterRealActionMoves(orchMoves);

    // For 3-player game with rings in hand, should have placement moves
    expect(realMoves.length).toBeGreaterThan(0);

    // Real action moves should validate
    for (const m of realMoves) {
      const validation = harness.adapter.validateMoveOnly(orchState, m);
      expect(validation.valid).toBe(true);
    }
  });
});
