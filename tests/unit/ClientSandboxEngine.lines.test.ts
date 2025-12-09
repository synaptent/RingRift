import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardState,
  BoardType,
  GameState,
  Move,
  Position,
  RingStack,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString,
  BOARD_CONFIGS,
} from '../../src/shared/types/game';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine';
import { isFSMOrchestratorActive } from '../../src/shared/utils/envFlags';

/**
 * Sandbox line detection + reward tests.
 *
 * These exercise ClientSandboxEngine's line-processing behaviour, which
 * mirrors backend defaults when no interaction manager is wired:
 * - Exact-length line: collapse all markers in the line and eliminate a cap.
 * - Longer-than-required line: collapse only the minimum required markers,
 *   with no elimination.
 */

describe('ClientSandboxEngine line processing', () => {
  // TODO: FSM validation is stricter - rejects choose_line_reward without
  // proper pending decision state. These tests manually inject board state
  // without setting up proper line detection flow.
  if (isFSMOrchestratorActive()) {
    it.skip('Skipping - FSM rejects line rewards without pending decision state', () => {});
    return;
  }

  const boardType: BoardType = 'square8';
  // Effective line threshold is always 3 on square8 (per RR-CANON-R120),
  // regardless of player count.
  const requiredLength = getEffectiveLineLengthThreshold(boardType, 2);

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // For these tests we never actually trigger PlayerChoices, but we
      // provide a trivial handler to satisfy the constructor.
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function makeStack(playerNumber: number, height: number, position: Position, board: BoardState) {
    const rings = Array(height).fill(playerNumber);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber,
    };
    board.stacks.set(positionToString(position), stack);
  }

  test('Q7_exact_length_line_collapse_sandbox', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    const board = state.board;

    // Clear any existing markers/stacks.
    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Place an exact-length horizontal line of markers for player 1 at y=1.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength; i++) {
      const pos: Position = { x: i, y: 1 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    // Add a stack for player 1 so there is a cap to eliminate.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(1, 2, stackPos, board);

    const initialTotalEliminated = state.totalRingsEliminated;
    const initialTerritory = state.players.find((p) => p.playerNumber === 1)!.territorySpaces;

    // Invoke line processing directly.
    engineAny.processLinesForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find((p) => p.playerNumber === 1)!;

    // All marker positions in the line should now be collapsed spaces for p1,
    // with no markers or stacks remaining at those positions.
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(finalBoard.markers.has(key)).toBe(false);
      expect(finalBoard.stacks.has(key)).toBe(false);
    }

    // The stack used for elimination should have been removed, and player1's
    // eliminatedRings / totalRingsEliminated increased by at least 1.
    expect(finalBoard.stacks.get(positionToString(stackPos))).toBeUndefined();
    expect(player1.eliminatedRings).toBeGreaterThan(0);
    expect(finalState.totalRingsEliminated).toBeGreaterThan(initialTotalEliminated);

    // Territory spaces should have increased by exactly the line length.
    expect(player1.territorySpaces).toBe(initialTerritory + requiredLength);

    // Sandbox-only visual cue: recent line highlights should match the
    // collapsed marker positions for this exact-length line.
    const recentHighlights = engine.consumeRecentLineHighlights();
    const recentKeys = new Set(recentHighlights.map((p) => positionToString(p)));
    const expectedKeys = new Set(linePositions.map((p) => positionToString(p)));
    expect(recentKeys).toEqual(expectedKeys);
  });

  test('longer-than-required line collapses minimum markers without elimination', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    const board = state.board;

    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Place a line longer than required: requiredLength + 1 markers.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      const pos: Position = { x: i, y: 2 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    const detectedLines = engineAny.findAllLines(board) as any[];
    expect(detectedLines.length).toBeGreaterThanOrEqual(1);

    // Add a stack for player 1; since this is a longer line we expect no
    // elimination, so the stack should remain unchanged.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(1, 2, stackPos, board);

    const initialPlayer1 = state.players.find((p) => p.playerNumber === 1)!;
    const initialEliminated = initialPlayer1.eliminatedRings;
    const initialTotalEliminated = state.totalRingsEliminated;
    const initialTerritory = initialPlayer1.territorySpaces;

    engineAny.processLinesForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find((p) => p.playerNumber === 1)!;

    // Exactly requiredLength markers should be collapsed; the remaining
    // marker should still exist and not be collapsed.
    const collapsedKeys = new Set<string>();
    for (const [key, owner] of finalBoard.collapsedSpaces) {
      if (owner === 1) collapsedKeys.add(key);
    }

    expect(collapsedKeys.size).toBe(requiredLength);

    const remainingPos = linePositions[requiredLength];
    const remainingKey = positionToString(remainingPos);
    expect(finalBoard.markers.has(remainingKey)).toBe(true);
    expect(finalBoard.collapsedSpaces.has(remainingKey)).toBe(false);

    // No elimination should have occurred.
    expect(player1.eliminatedRings).toBe(initialEliminated);
    expect(finalState.totalRingsEliminated).toBe(initialTotalEliminated);

    // Territory spaces should have increased by exactly requiredLength.
    expect(player1.territorySpaces).toBe(initialTerritory + requiredLength);

    // Stack should still exist at stackPos (no forced elimination in this test).
    expect(finalBoard.stacks.get(positionToString(stackPos))).toBeDefined();
  });

  test('enumerates canonical line-processing decision Moves for current player', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    const board = state.board;

    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Exact-length horizontal line at y=0.
    const exactLinePositions: Position[] = [];
    for (let i = 0; i < requiredLength; i++) {
      const pos: Position = { x: i, y: 0 };
      exactLinePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    // Longer-than-required horizontal line at y=3 (requiredLength + 1 markers).
    const longLinePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      const pos: Position = { x: i, y: 3 };
      longLinePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    const moves: any[] = engineAny.getValidLineProcessingMovesForCurrentPlayer();
    const processLineMoves = moves.filter((m) => m.type === 'process_line');
    const chooseRewardMoves = moves.filter((m) => m.type === 'choose_line_reward');

    // One process_line per line, plus a richer reward surface for the
    // overlength line (collapse-all + all minimum segments).
    expect(processLineMoves.length).toBe(2);
    expect(chooseRewardMoves.length).toBeGreaterThanOrEqual(2);

    const keyFrom = (positions: Position[]) =>
      positions
        .map((p) => positionToString(p))
        .sort()
        .join('|');

    const exactKey = keyFrom(exactLinePositions);
    const longKey = keyFrom(longLinePositions);

    const processExact = processLineMoves.find(
      (m) => m.formedLines && m.formedLines[0] && keyFrom(m.formedLines[0].positions) === exactKey
    );
    const processLong = processLineMoves.find(
      (m) => m.formedLines && m.formedLines[0] && keyFrom(m.formedLines[0].positions) === longKey
    );

    expect(processExact).toBeDefined();
    expect(processLong).toBeDefined();

    const chooseForExact = chooseRewardMoves.filter(
      (m) => m.formedLines && m.formedLines[0] && keyFrom(m.formedLines[0].positions) === exactKey
    );
    const chooseForLong = chooseRewardMoves.filter(
      (m) => m.formedLines && m.formedLines[0] && keyFrom(m.formedLines[0].positions) === longKey
    );

    // Exact-length line may be expressed as a single collapse-all reward.
    // Under the shared helper semantics we expect exactly one such Move.
    expect(chooseForExact.length).toBe(1);
    expect(chooseForExact[0].collapsedMarkers).toBeUndefined();

    // Overlength line should expose one collapse-all reward and one or more
    // minimum-collapse contiguous segments of length L.
    expect(chooseForLong.length).toBeGreaterThanOrEqual(2);

    const collapseAllForLong = chooseForLong.filter((m) => {
      const collapsed = m.collapsedMarkers ?? [];
      return collapsed.length === 0 || collapsed.length >= longLinePositions.length;
    });
    const minCollapseForLong = chooseForLong.filter(
      (m) => m.collapsedMarkers && m.collapsedMarkers.length === requiredLength
    );

    expect(collapseAllForLong.length).toBeGreaterThanOrEqual(1);
    expect(minCollapseForLong.length).toBeGreaterThanOrEqual(1);
  });

  test('canonical choose_line_reward Move collapses entire overlength line (orchestrator flow)', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    state.currentPhase = 'line_processing'; // Required for choose_line_reward moves
    const board = state.board;

    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Build an overlength horizontal line at y=4.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 2; i++) {
      const pos: Position = { x: i, y: 4 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    // Single stack for player 1 that would be used for elimination.
    const stackPos: Position = { x: 7, y: 7 };
    const stackHeight = 3;
    makeStack(1, stackHeight, stackPos, board);

    const playerBefore = state.players.find((p) => p.playerNumber === 1)!;
    const initialTerritory = playerBefore.territorySpaces;

    // Construct a canonical choose_line_reward Move matching this line.
    const lineKey = linePositions.map((p) => positionToString(p)).join('|');
    const move: Move = {
      id: `choose-line-reward-0-${lineKey}`,
      type: 'choose_line_reward',
      player: 1,
      formedLines: [
        {
          positions: linePositions,
          player: 1,
          length: linePositions.length,
          direction: { x: 1, y: 0 },
        } as any,
      ],
      // Decision moves are phase-driven; `to` is unused but required.
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    await engine.applyCanonicalMove(move);

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const playerAfter = finalState.players.find((p) => p.playerNumber === 1)!;

    // All marker positions in the line should now be collapsed spaces for player 1,
    // with no markers or stacks remaining at those positions.
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(finalBoard.markers.has(key)).toBe(false);
      expect(finalBoard.stacks.has(key)).toBe(false);
    }

    // Under orchestrator-driven flow, elimination requires an explicit
    // eliminate_rings_from_stack move. The choose_line_reward move only
    // handles line collapse. Elimination is NOT automatic - it would be
    // surfaced as a pending decision for explicit handling.
    // The stack should still exist at this point (elimination not yet applied).
    const stackAfter = finalBoard.stacks.get(positionToString(stackPos));
    expect(stackAfter).toBeDefined();
    if (stackAfter) {
      expect(stackAfter.stackHeight).toBe(stackHeight);
    }

    // Territory spaces should have increased by at least the line length.
    expect(playerAfter.territorySpaces).toBeGreaterThanOrEqual(
      initialTerritory + linePositions.length
    );
  });

  test('2p-8x8: 2-in-a-row does NOT trigger line processing (below threshold)', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    const board = state.board;

    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Place a 2-length horizontal line of markers for player 1 at y=1.
    // On 2p-8x8, the threshold is 3-in-a-row (per RR-CANON-R120), so 2-in-a-row
    // should NOT be enough to trigger line processing.
    const linePositions: Position[] = [];
    for (let i = 0; i < 2; i++) {
      const pos: Position = { x: i, y: 1 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular',
      });
    }

    // Add a stack for player 1 so there is a cap to eliminate if it were processed.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(1, 2, stackPos, board);

    const initialTotalEliminated = state.totalRingsEliminated;
    const initialTerritory = state.players.find((p) => p.playerNumber === 1)!.territorySpaces;

    // Invoke line processing directly.
    engineAny.processLinesForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find((p) => p.playerNumber === 1)!;

    // Markers should remain as markers.
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(finalBoard.markers.has(key)).toBe(true);
      expect(finalBoard.collapsedSpaces.has(key)).toBe(false);
    }

    // No elimination should have occurred.
    expect(player1.eliminatedRings).toBe(0);
    expect(finalState.totalRingsEliminated).toBe(initialTotalEliminated);

    // Territory spaces should be unchanged.
    expect(player1.territorySpaces).toBe(initialTerritory);
  });
});
