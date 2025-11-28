import { BoardType, GameState, Move, positionToString } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { findAllLines } from '../../src/shared/engine/lineDetection';
import { enumerateProcessLineMoves } from '../../src/shared/engine/lineDecisionHelpers';

/**
 * Focused invariant test for the known seed-5 tail mismatch.
 *
 * We replay the sandbox AI trace for square8 / 2p / seed=5 up to
 * move index 62 (0-based, moveNumber 63, a `move_stack` by P2) into
 * both backend and sandbox engines. At that checkpoint we assert that:
 *
 * 1. Backend and sandbox agree on marker-line geometry when inspected
 *    via the shared `findAllLines` helper.
 * 2. For each host snapshot, `enumerateProcessLineMoves` using
 *    `detectionMode: 'use_board_cache'` and `detectionMode: 'detect_now'`
 *    agree on which lines belong to the current player.
 * 3. The sets of lines exposed to the current player by
 *    `enumerateProcessLineMoves(..., detectionMode: 'detect_now')` are
 *    identical for backend and sandbox snapshots.
 * 4. We capture whether there is at least one canonical line for the
 *    current player at this snapshot. This tells us whether a
 *    `line_processing` phase is *semantically required* at move 63,
 *    which in turn identifies whether backend or sandbox is mis-
 *    scheduling the phase in the full parity harness.
 *
 * This isolates whether any residual seed-5 mismatch at move 63 is due
 * purely to scheduling/phase flow or to an inconsistency in how the
 * hosts populate and consume line caches at that position.
 */

// Skip with orchestrator adapter - this test relies on specific sandbox trace behavior
// which may produce fewer moves with the orchestrator adapter's different processing.
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Seed5 move63 line-detection snapshot invariants (square8 / 2p / seed=5)',
  () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;
    const seed = 5;
    const MAX_STEPS = 64;

    function canonicalLineKeys(
      lines: { positions: { x: number; y: number; z?: number }[] }[]
    ): string[] {
      return lines
        .map((line) =>
          line.positions
            .map((p) => positionToString(p))
            .sort()
            .join('|')
        )
        .sort();
    }

    function canonicalMoveLineKeys(moves: Move[]): string[] {
      const keys = new Set<string>();
      for (const move of moves) {
        if (!move.formedLines || move.formedLines.length === 0) {
          continue;
        }
        const line = move.formedLines[0];
        const key = line.positions
          .map((p) => positionToString(p))
          .sort()
          .join('|');
        keys.add(key);
      }
      return Array.from(keys).sort();
    }

    function createSandboxEngineFromInitial(initial: GameState): ClientSandboxEngine {
      const config: SandboxConfig = {
        boardType: initial.boardType,
        numPlayers: initial.players.length,
        playerKinds: initial.players
          .slice()
          .sort((a, b) => a.playerNumber - b.playerNumber)
          .map((p) => p.type as 'human' | 'ai'),
      };

      const handler: SandboxInteractionHandler = {
        async requestChoice(choice: any) {
          const options = ((choice as any).options as any[]) ?? [];
          const selectedOption = options.length > 0 ? options[0] : undefined;

          return {
            choiceId: (choice as any).id,
            playerNumber: (choice as any).playerNumber,
            choiceType: (choice as any).type,
            selectedOption,
          } as any;
        },
      };

      const engine = new ClientSandboxEngine({
        config,
        interactionHandler: handler,
        traceMode: true,
      });
      const engineAny: any = engine;
      engineAny.gameState = initial;
      return engine;
    }

    test('backend and sandbox agree on line geometry and enumeration at move index 62', async () => {
      const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
      expect(trace.entries.length).toBeGreaterThan(62);

      const moves: Move[] = trace.entries.map((e) => e.action as Move);

      const backendEngine = createBackendEngineFromInitialState(trace.initialState);
      const sandboxEngine = createSandboxEngineFromInitial(trace.initialState);

      // Replay moves[0..62] inclusive into both engines.
      for (let i = 0; i <= 62; i++) {
        const move = moves[i];

        // Backend: map sandbox move to a canonical backend move and apply.
        const backendStateBefore = backendEngine.getGameState();
        const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
        const matching = findMatchingBackendMove(move, backendValidMoves);

        expect(matching).toBeDefined();

        const { id, timestamp, moveNumber, ...payload } = matching as any;
        const backendResult = await backendEngine.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );
        expect(backendResult.success).toBe(true);

        // Sandbox: apply the canonical move directly.
        await sandboxEngine.applyCanonicalMove(move);
      }

      const backendState = backendEngine.getGameState();
      const sandboxState = sandboxEngine.getGameState();

      // 1) Shared line geometry via findAllLines should match for both boards.
      const backendLines = findAllLines(backendState.board);
      const sandboxLines = findAllLines(sandboxState.board);
      const backendLineKeys = canonicalLineKeys(backendLines);
      const sandboxLineKeys = canonicalLineKeys(sandboxLines);

      expect(backendLineKeys).toEqual(sandboxLineKeys);

      // 1b) Assert that there are no canonical lines at this snapshot.
      // This locks in the seed-5 diagnostics finding: after move index 62
      // there is no line work for the current player on either host, so any
      // line_processing phase here would be spurious.
      expect(backendLineKeys.length).toBe(0);

      // 2) For each host snapshot, enumerateProcessLineMoves should be
      // consistent between cache-based and detect_now detection modes.
      const backendPlayer = backendState.currentPlayer;
      const sandboxPlayer = sandboxState.currentPlayer;

      const backendMovesFromCache = enumerateProcessLineMoves(backendState, backendPlayer, {
        detectionMode: 'use_board_cache',
      });
      const backendMovesDetectNow = enumerateProcessLineMoves(backendState, backendPlayer, {
        detectionMode: 'detect_now',
      });

      const sandboxMovesFromCache = enumerateProcessLineMoves(sandboxState, sandboxPlayer, {
        detectionMode: 'use_board_cache',
      });
      const sandboxMovesDetectNow = enumerateProcessLineMoves(sandboxState, sandboxPlayer, {
        detectionMode: 'detect_now',
      });

      const backendCacheKeys = canonicalMoveLineKeys(backendMovesFromCache);
      const backendDetectNowKeys = canonicalMoveLineKeys(backendMovesDetectNow);
      const sandboxCacheKeys = canonicalMoveLineKeys(sandboxMovesFromCache);
      const sandboxDetectNowKeys = canonicalMoveLineKeys(sandboxMovesDetectNow);

      expect(backendDetectNowKeys).toEqual(backendCacheKeys);
      expect(sandboxDetectNowKeys).toEqual(sandboxCacheKeys);

      // 2b) Assert that there are no process_line decisions available for
      // the current player at this snapshot under detect_now enumeration.
      // Combined with the geometry assertion above, this locks in the fact
      // that both hosts see zero line decisions (no work) here.
      expect(backendMovesDetectNow.length).toBe(0);
      expect(sandboxMovesDetectNow.length).toBe(0);

      // 3) The lines exposed to the current player by detect_now enumeration
      // should be identical for backend and sandbox snapshots.
      expect(backendDetectNowKeys).toEqual(sandboxDetectNowKeys);
    });
  }
);
