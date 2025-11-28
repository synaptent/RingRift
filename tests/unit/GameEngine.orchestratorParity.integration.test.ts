/**
 * GameEngine orchestrator vs legacy parity (smoke)
 *
 * These tests run a short, deterministic sequence of canonical Moves through
 * two GameEngine instances:
 *   - one using the legacy internal turn pipeline, and
 *   - one delegating to TurnEngineAdapter (shared orchestrator).
 *
 * For the chosen scenarios, the final GameState hashes and high-level fields
 * should match, providing a fast integration-level signal that the adapter
 * path stays in lockstep with the legacy behaviour.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import type {
  BoardType,
  GameState,
  Move,
  Player,
  TimeControl,
  Position,
  RingStack,
} from '../../src/shared/types/game';
import { BOARD_CONFIGS } from '../../src/shared/types/game';
import { BoardManager } from '../../src/server/game/BoardManager';
import { hashGameState, calculateCapHeight } from '../../src/shared/engine/core';

const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'ai',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: BOARD_CONFIGS.square8.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'ai',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: BOARD_CONFIGS.square8.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

function createEnginePair(
  gameId: string,
  boardType: BoardType = 'square8'
): { legacy: GameEngine; orchestrator: GameEngine } {
  const players = createPlayers();

  // Legacy path – explicitly disable orchestrator adapter
  const legacy = new GameEngine(gameId + '-legacy', boardType, players, timeControl, true);
  legacy.disableOrchestratorAdapter();
  legacy.enableMoveDrivenDecisionPhases();
  const startedLegacy = legacy.startGame();
  if (!startedLegacy) {
    throw new Error('Failed to start legacy GameEngine for orchestrator parity test');
  }

  // Orchestrator path – explicitly enable adapter
  const orchestrator = new GameEngine(gameId + '-orch', boardType, players, timeControl, true);
  orchestrator.enableOrchestratorAdapter();
  const startedOrch = orchestrator.startGame();
  if (!startedOrch) {
    throw new Error('Failed to start orchestrator GameEngine for orchestrator parity test');
  }

  return { legacy, orchestrator };
}

function snapshot(state: GameState) {
  return {
    hash: hashGameState(state),
    currentPhase: state.currentPhase,
    currentPlayer: state.currentPlayer,
    gameStatus: state.gameStatus,
    moveCount: state.moveHistory.length,
  };
}

function canonicalEngineMove(move: Move): Omit<Move, 'id' | 'timestamp' | 'moveNumber'> {
  // Strip non-semantic fields; keep everything else so both engines see
  // identical canonical payloads.
  const { id, timestamp, moveNumber, ...rest } = move as any;
  return {
    ...rest,
    thinkTime: 0,
  };
}

describe('GameEngine orchestrator vs legacy parity (short sequences)', () => {
  it('keeps states in lockstep for a short ring_placement + movement sequence on square8', async () => {
    const { legacy, orchestrator } = createEnginePair('orch-parity-basic', 'square8');

    // Sanity: initial snapshots must match
    const initialLegacy = legacy.getGameState();
    const initialOrch = orchestrator.getGameState();
    expect(snapshot(initialOrch)).toEqual(snapshot(initialLegacy));

    const maxSteps = 6;

    for (let i = 0; i < maxSteps; i++) {
      const legacyStateBefore = legacy.getGameState();
      const orchStateBefore = orchestrator.getGameState();

      // High-level fields must be aligned before each step
      expect(snapshot(orchStateBefore)).toEqual(snapshot(legacyStateBefore));

      if (legacyStateBefore.gameStatus !== 'active') {
        break;
      }

      const currentPlayer = legacyStateBefore.currentPlayer;
      const candidates = legacy.getValidMoves(currentPlayer);
      expect(candidates.length).toBeGreaterThan(0);

      // Deterministic choice: sort by type and coordinates and pick first
      const sorted = [...candidates].sort((a, b) => {
        if (a.type !== b.type) {
          return a.type < b.type ? -1 : 1;
        }
        const ax = a.to?.x ?? 0;
        const ay = a.to?.y ?? 0;
        const bx = b.to?.x ?? 0;
        const by = b.to?.y ?? 0;
        if (ay !== by) return ay - by;
        if (ax !== bx) return ax - bx;
        return 0;
      });

      const chosen = sorted[0];
      const engineMove = canonicalEngineMove(chosen);

      const legacyResult = await legacy.makeMove(engineMove);
      expect(legacyResult.success).toBe(true);

      const orchResult = await orchestrator.makeMove(engineMove);
      expect(orchResult.success).toBe(legacyResult.success);

      const legacyAfter = legacy.getGameState();
      const orchAfter = orchestrator.getGameState();

      expect(snapshot(orchAfter)).toEqual(snapshot(legacyAfter));

      if (legacyAfter.gameStatus !== 'active') {
        break;
      }
    }
  });

  it('keeps states in lockstep for a single overtaking capture with markers on square8', async () => {
    const { legacy, orchestrator } = createEnginePair('orch-parity-capture', 'square8');

    // Both engines start from their own initial states; now seed an identical
    // capture fixture into each internal board/state.
    const boardType: BoardType = 'square8';

    function seedCaptureFixture(engine: GameEngine) {
      const anyEngine = engine as any;
      const bm: BoardManager = anyEngine.boardManager;
      const internalBoard = anyEngine.gameState.board;

      const from: Position = { x: 2, y: 1 };
      const target: Position = { x: 2, y: 3 };
      const landing: Position = { x: 2, y: 5 };
      const movingPlayer = 1;
      const targetPlayer = 2;

      // Clear any existing geometry
      internalBoard.stacks.clear();
      internalBoard.markers.clear();
      internalBoard.collapsedSpaces.clear();
      internalBoard.territories.clear();
      internalBoard.formedLines.length = 0;
      internalBoard.eliminatedRings = {};

      // Attacker stack at from (P1), height 2
      const attackerRings = [movingPlayer, movingPlayer];
      const attackerStack: RingStack = {
        position: from,
        rings: attackerRings,
        stackHeight: attackerRings.length,
        capHeight: calculateCapHeight(attackerRings),
        controllingPlayer: movingPlayer,
      };
      bm.setStack(from, attackerStack, internalBoard);

      // Target stack at target (P2), height 2
      const targetRings = [targetPlayer, targetPlayer];
      const targetStack: RingStack = {
        position: target,
        rings: targetRings,
        stackHeight: targetRings.length,
        capHeight: calculateCapHeight(targetRings),
        controllingPlayer: targetPlayer,
      };
      bm.setStack(target, targetStack, internalBoard);

      // Markers along the capture path:
      // Own marker at (2,2) – should collapse to territory.
      bm.setMarker({ x: 2, y: 2 }, movingPlayer, internalBoard);
      // Opponent marker at (2,4) – should be flipped/removed appropriately.
      bm.setMarker({ x: 2, y: 4 }, targetPlayer, internalBoard);
      // Own marker at landing – landing on own marker triggers top-ring elimination.
      bm.setMarker(landing, movingPlayer, internalBoard);

      anyEngine.gameState.currentPlayer = movingPlayer;
      anyEngine.gameState.currentPhase = 'movement';

      return { from, target, landing, movingPlayer };
    }

    const legacyFixture = seedCaptureFixture(legacy);
    const orchFixture = seedCaptureFixture(orchestrator);

    // Sanity: fixtures must agree
    expect(legacyFixture).toEqual(orchFixture);

    const { from, target, landing, movingPlayer } = legacyFixture;

    const captureMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'overtaking_capture',
      player: movingPlayer,
      from,
      captureTarget: target,
      to: landing,
      thinkTime: 0,
    };

    const legacyResult = await legacy.makeMove(captureMove);
    expect(legacyResult.success).toBe(true);

    const orchResult = await orchestrator.makeMove(captureMove);
    expect(orchResult.success).toBe(legacyResult.success);

    const legacyAfter = legacy.getGameState();
    const orchAfter = orchestrator.getGameState();

    expect(snapshot(orchAfter)).toEqual(snapshot(legacyAfter));
  });
});
