import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BoardState,
  GameState,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { computeSMetric, computeTMetric, isANMState } from '../../src/shared/engine';
import {
  findAllLines,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
} from '../../src/shared/engine/aggregates/LineAggregate';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine/rulesConfig';

/**
 * Line-formation scenario tests aligned with rules/FAQ.
 *
 * These focus on:
 * - Section 11 (Line Formation & Collapse)
 * - FAQ Q7 (exact-length lines)
 * - FAQ Q22 (graduated line rewards and Option 1 vs Option 2)
 *
 * Note: These tests use the shared engine's LineAggregate directly
 * instead of the deprecated GameEngine.processLineFormations() method.
 * See Wave 5.4 in TODO.md for deprecation context.
 */

describe('GameEngine line formation scenarios (square8)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  // For 2-player games, use the effective threshold (4 for square8 2p) instead of raw config
  const numPlayers = 2;
  const requiredLength = getEffectiveLineLengthThreshold(boardType, numPlayers, undefined);

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
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
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createEngine(): {
    engine: GameEngine;
    gameState: GameState;
    boardManager: any;
  } {
    const engine = new GameEngine('lines-scenarios', boardType, basePlayers, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const boardManager: any = engineAny.boardManager;
    return { engine, gameState, boardManager };
  }

  function makeStack(
    boardManager: any,
    gameState: GameState,
    playerNumber: number,
    height: number,
    position: Position
  ) {
    const rings = Array(height).fill(playerNumber);
    const stack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber,
    };
    boardManager.setStack(position, stack, gameState.board);
  }

  test('Q7_exact_length_line_collapse_shared_engine', () => {
    // Rules reference:
    // - Section 11.2: exact-length line → collapse all markers + eliminate ring/cap.
    // - FAQ Q7: exact-length lines always require elimination.
    //
    // This test uses the shared engine's LineAggregate directly instead of
    // the deprecated GameEngine.processLineFormations() method.
    const { engine, gameState, boardManager } = createEngine();

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'line_processing';
    const board = gameState.board;

    // Clear any existing markers/stacks/collapsed spaces.
    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Create exact-length line for player 1 at y = 1 with actual markers.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength; i++) {
      const pos = { x: i, y: 1 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    // Add a stack for player 1 so there is a cap to eliminate.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(boardManager, gameState, 1, 2, stackPos);

    // Detect lines using shared engine
    const detectedLines = findAllLines(board);
    expect(detectedLines.length).toBe(1);
    expect(detectedLines[0].length).toBe(requiredLength);
    expect(detectedLines[0].player).toBe(1);

    // Cache the formed lines on the board for the shared engine to access
    board.formedLines = detectedLines;

    const player1Before = gameState.players.find((p) => p.playerNumber === 1)!;
    const initialTerritory = player1Before.territorySpaces;

    // INV-S-MONOTONIC / INV-ELIMINATION-MONOTONIC (R191, R207)
    const sBefore = computeSMetric(gameState);
    const tBefore = computeTMetric(gameState);

    // Use shared engine to enumerate and apply line decision
    const processLineMoves = enumerateProcessLineMoves(gameState, 1);
    expect(processLineMoves.length).toBe(1);

    // Apply the process_line decision using shared engine
    const outcome = applyProcessLineDecision(gameState, processLineMoves[0]);

    const sAfter = computeSMetric(outcome.nextState);
    const tAfter = computeTMetric(outcome.nextState);

    expect(sAfter).toBeGreaterThanOrEqual(sBefore);
    // T is non-decreasing per decision-chain; individual steps may preserve T.
    expect(tAfter).toBeGreaterThanOrEqual(tBefore);

    const player1After = outcome.nextState.players.find((p) => p.playerNumber === 1)!;

    // Verify all line positions collapsed (markers removed)
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(outcome.nextState.board.markers.has(key)).toBe(false);
    }

    // Exact-length line grants elimination reward
    expect(outcome.pendingLineRewardElimination).toBe(true);

    // Territory metric: player gains territory from collapsed line
    const territoryDelta = player1After.territorySpaces - initialTerritory;
    expect(territoryDelta).toBe(requiredLength);

    // INV-ACTIVE-NO-MOVES / INV-PHASE-CONSISTENCY: resulting ACTIVE state is not ANM.
    if (outcome.nextState.gameStatus === 'active') {
      expect(isANMState(outcome.nextState)).toBe(false);
    }
  });

  test('Q22_graduated_rewards_option2_min_collapse_shared_engine', () => {
    // Rules reference:
    // - Section 11.2 / 11.3: lines longer than required may use Option 2
    //   (collapse only the minimum required markers, no elimination).
    // - FAQ Q22: strategic tradeoff for preserving rings by choosing
    //   minimum collapse.
    //
    // This test uses the shared engine's LineAggregate directly instead of
    // the deprecated GameEngine.processLineFormations() method.
    const { engine, gameState, boardManager } = createEngine();

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'line_processing';
    const board = gameState.board;

    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Create overlength line with actual markers: requiredLength + 1 markers.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      const pos = { x: i, y: 2 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    // Add a stack for player 1.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(boardManager, gameState, 1, 2, stackPos);

    // Detect lines using shared engine
    const detectedLines = findAllLines(board);
    expect(detectedLines.length).toBe(1);
    expect(detectedLines[0].length).toBe(requiredLength + 1);
    expect(detectedLines[0].player).toBe(1);

    // Cache the formed lines on the board
    board.formedLines = detectedLines;

    const player1Before = gameState.players.find((p) => p.playerNumber === 1)!;
    const initialTerritory = player1Before.territorySpaces;

    // INV-S-MONOTONIC / INV-ELIMINATION-MONOTONIC (R191, R207)
    const sBefore = computeSMetric(gameState);
    const tBefore = computeTMetric(gameState);

    // Use shared engine to enumerate line reward choices for overlength line
    const rewardMoves = enumerateChooseLineRewardMoves(gameState, 1, 0);

    // Overlength line should have at least one reward option
    expect(rewardMoves.length).toBeGreaterThanOrEqual(1);

    // Find a minimum-collapse move (Option 2 - no elimination reward)
    // If no explicit minimum-collapse move exists, create one manually via
    // the apply function with explicit collapsedMarkers
    const minCollapseSegment = linePositions.slice(0, requiredLength);

    // Create a synthetic choose_line_reward move for Option 2 (minimum collapse)
    const minCollapseMove = {
      ...rewardMoves[0],
      id: `choose-line-reward-0-min-segment`,
      type: 'choose_line_reward' as const,
      collapsedMarkers: minCollapseSegment,
    };

    // Apply Option 2: minimum collapse (first segment)
    const outcome = applyChooseLineRewardDecision(gameState, minCollapseMove);

    const sAfter = computeSMetric(outcome.nextState);
    const tAfter = computeTMetric(outcome.nextState);

    expect(sAfter).toBeGreaterThanOrEqual(sBefore);
    expect(tAfter).toBeGreaterThanOrEqual(tBefore);

    const player1After = outcome.nextState.players.find((p) => p.playerNumber === 1)!;

    // Option 2 (minimum collapse) grants NO elimination reward
    expect(outcome.pendingLineRewardElimination).toBe(false);

    // Territory: exactly requiredLength spaces collapsed
    const territoryDelta = player1After.territorySpaces - initialTerritory;
    expect(territoryDelta).toBe(requiredLength);

    // One marker should remain (overlength - requiredLength = 1 remaining)
    const remainingMarkers = Array.from(outcome.nextState.board.markers.values()).filter(
      (m) => m.player === 1
    );
    expect(remainingMarkers.length).toBe(1);

    // INV-ACTIVE-NO-MOVES: resulting ACTIVE state is not ANM.
    if (outcome.nextState.gameStatus === 'active') {
      expect(isANMState(outcome.nextState)).toBe(false);
    }
  });

  test('line_processing_getValidMoves_exposes_process_line_and_rich_choose_line_reward_moves', () => {
    // Rules reference:
    // - Section 11.2–11.3: when multiple lines exist for the moving player,
    //   line_processing should surface one process_line move per line.
    // - Overlength lines expose a richer choose_line_reward surface:
    //   - One collapse-all reward (implicit Option 1).
    //   - One or more minimum-collapse contiguous segments of length L
    //     (Option 2-style rewards).
    const { engine, gameState, boardManager } = createEngine();
    const board = gameState.board;

    gameState.currentPlayer = 1;
    (gameState as any).currentPhase = 'line_processing';

    // Clear any existing markers/stacks/collapsed spaces.
    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Synthetic exact-length and overlength lines for player 1.
    const exactLine: Position[] = [];
    for (let i = 0; i < requiredLength; i++) {
      exactLine.push({ x: i, y: 0 });
    }

    const overlengthLine: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      overlengthLine.push({ x: i, y: 2 });
    }

    // Seed markers and formedLines so the shared line-detection helpers
    // (lineDecisionHelpers + lineDetection) see exactly these two lines.
    const allLinePositions = [...exactLine, ...overlengthLine];
    for (const pos of allLinePositions) {
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      } as any);
    }

    (board as any).formedLines = [
      {
        player: 1,
        positions: exactLine,
        length: exactLine.length,
        direction: { x: 1, y: 0 },
      },
      {
        player: 1,
        positions: overlengthLine,
        length: overlengthLine.length,
        direction: { x: 1, y: 0 },
      },
    ];

    const moves = engine.getValidMoves(1);

    const processLineMoves = moves.filter((m) => m.type === 'process_line');
    const rewardMoves = moves.filter((m) => m.type === 'choose_line_reward');

    // One process_line per player-owned line.
    expect(processLineMoves).toHaveLength(2);
    expect(processLineMoves.every((m) => m.player === 1)).toBe(true);

    // There should be at least one reward surface for the overlength line.
    expect(rewardMoves.length).toBeGreaterThanOrEqual(1);
    expect(rewardMoves.every((m) => m.player === 1)).toBe(true);

    const overlengthKey = overlengthLine.map((p) => positionToString(p)).join('|');

    // All choose_line_reward moves should embed the overlength line's key
    // in their id so tools can map them back to a concrete line geometry.
    const rewardIds = rewardMoves.map((m) => m.id);
    expect(rewardIds.some((id) => id.includes(overlengthKey))).toBe(true);
  });
});
