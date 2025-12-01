import {
  BoardType,
  GameState,
  Move,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { GameEngine } from '../../src/server/game/GameEngine';
import { summarizeBoard, computeProgressSnapshot } from '../../src/shared/engine/core';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { pos } from '../utils/fixtures';

/**
 * Focused parity/debug tests for the known seed-17 geometry / capture
 * divergences on square8:
 *   - moveNumber 16: sandbox emits an overtaking_capture C c1×d2→f4 but the
 *     backend only exposes non-capture moves from c1.
 *   - moveNumber 33: sandbox is in chain_capture for player 2 and emits
 *     continue_capture_segment h4×f4→c4 while the backend has already
 *     advanced to player 1's ring_placement phase.
 *
 * These tests reconstruct the exact pre-step sandbox + backend states for
 * each move directly from the canonical sandbox AI trace, assert geometric
 * parity at those pre-step states, and then compare phase/actor and
 * getValidMoves behaviour for the specific canonical move. They are
 * intentionally analogous to:
 *   - Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts
 *   - Seed14Move35LineParity.test.ts
 */

/**
 * TODO-SEED17-CAPTURE-PARITY: These focused parity tests reconstruct
 * exact pre-step states for seed-17 move 16 and 33 divergences between
 * backend and sandbox. The tests require deep trace infrastructure and
 * compare capture/chain-capture behavior at specific move numbers.
 *
 * To keep core CI fast, this suite is opt-in via the
 * RINGRIFT_ENABLE_SEED17_PARITY env var. When unset/false it is skipped;
 * when set to '1' or 'true' it runs as a normal describe().
 */
const SEED17_PARITY_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_ENABLE_SEED17_PARITY ?? '');

const maybeDescribe = SEED17_PARITY_ENABLED ? describe : describe.skip;

maybeDescribe('Seed17 early capture parity: GameEngine vs ClientSandboxEngine', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 17;
  const MAX_STEPS = 80;

  function createDeterministicSandboxHandler(): SandboxInteractionHandler {
    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice>(choice: TChoice): Promise<any> {
        const anyChoice: any = choice;

        if (anyChoice.type === 'capture_direction') {
          const options = anyChoice.options || [];
          if (options.length === 0) {
            throw new Error('SandboxInteractionHandler: no options for capture_direction');
          }

          // Deterministically pick the option with the smallest landing x,y
          // to keep simulations reproducible given a fixed Math.random.
          let selected = options[0];
          for (const opt of options) {
            if (
              opt.landingPosition.x < selected.landingPosition.x ||
              (opt.landingPosition.x === selected.landingPosition.x &&
                opt.landingPosition.y < selected.landingPosition.y)
            ) {
              selected = opt;
            }
          }

          return {
            choiceId: anyChoice.id,
            playerNumber: anyChoice.playerNumber,
            choiceType: anyChoice.type,
            selectedOption: selected,
          } as any;
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as any;
      },
    };

    return handler;
  }

  function createSandboxEngineFromInitialState(initial: GameState): ClientSandboxEngine {
    const boardType = initial.boardType;
    const numPlayers = initial.players.length;

    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: initial.players
        .slice()
        .sort((a, b) => a.playerNumber - b.playerNumber)
        .map((p) => p.type as 'human' | 'ai'),
    };

    const handler = createDeterministicSandboxHandler();
    return new ClientSandboxEngine({ config, interactionHandler: handler, traceMode: true });
  }

  /**
   * Strict backend move matcher for these focused tests.
   *
   * Unlike the generic movesLooselyMatch/findMatchingBackendMove helpers used
   * by the trace harness, this helper does NOT relax capture vs movement
   * typing or placementCount. For move-16/move-33 debugging we want an early
   * failure if the backend ever mis-classifies an overtaking capture as a
   * simple move, or diverges on multi-ring placements.
   */
  function strictFindMatchingBackendMove(reference: Move, candidates: Move[]): Move | null {
    for (const candidate of candidates) {
      if (candidate.player !== reference.player) continue;

      // Treat simple non-capture stack movements equivalently whether they
      // are labelled move_ring or move_stack, but require exact MoveType
      // match for captures and decision moves.
      const isSimpleMovementPair =
        (reference.type === 'move_ring' && candidate.type === 'move_stack') ||
        (reference.type === 'move_stack' && candidate.type === 'move_ring') ||
        (reference.type === 'move_ring' && candidate.type === 'move_ring') ||
        (reference.type === 'move_stack' && candidate.type === 'move_stack');

      if (isSimpleMovementPair) {
        const fromOk =
          (!reference.from && !candidate.from) ||
          (reference.from &&
            candidate.from &&
            reference.from.x === candidate.from.x &&
            reference.from.y === candidate.from.y);
        const toOk =
          (!reference.to && !candidate.to) ||
          (reference.to &&
            candidate.to &&
            reference.to.x === candidate.to.x &&
            reference.to.y === candidate.to.y);
        if (fromOk && toOk) {
          return candidate;
        }
        continue;
      }

      // For everything else we insist on exact MoveType equality.
      if (candidate.type !== reference.type) continue;

      if (reference.type === 'place_ring') {
        const refCount = reference.placementCount ?? 1;
        const candCount = candidate.placementCount ?? 1;
        if (
          candidate.to &&
          reference.to &&
          candidate.to.x === reference.to.x &&
          candidate.to.y === reference.to.y &&
          refCount === candCount
        ) {
          return candidate;
        }
        continue;
      }

      if (
        (reference.from || candidate.from) &&
        (!reference.from ||
          !candidate.from ||
          reference.from.x !== candidate.from.x ||
          reference.from.y !== candidate.from.y)
      ) {
        continue;
      }

      if (
        (reference.to || candidate.to) &&
        (!reference.to ||
          !candidate.to ||
          reference.to.x !== candidate.to.x ||
          reference.to.y !== candidate.to.y)
      ) {
        continue;
      }

      if (reference.captureTarget || candidate.captureTarget) {
        if (
          !reference.captureTarget ||
          !candidate.captureTarget ||
          reference.captureTarget.x !== candidate.captureTarget.x ||
          reference.captureTarget.y !== candidate.captureTarget.y
        ) {
          continue;
        }
      }

      return candidate;
    }

    return null;
  }

  async function reconstructPreStepStates(targetMoveNumber: number): Promise<{
    trace: { entries: { action: Move }[] };
    targetIndex: number;
    targetMove: Move;
    sandboxEngine: ClientSandboxEngine;
    backendEngine: GameEngine;
    sandboxBefore: GameState;
    backendBefore: GameState;
  }> {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);

    const targetIndex = trace.entries.findIndex(
      (e) => (e.action as Move).moveNumber === targetMoveNumber
    );
    expect(targetIndex).toBeGreaterThanOrEqual(0);

    const targetMove = trace.entries[targetIndex].action as Move;

    // --- Reconstruct sandbox state immediately BEFORE targetMoveNumber ---
    const sandboxEngine = createSandboxEngineFromInitialState(trace.initialState as GameState);

    for (let i = 0; i < targetIndex; i++) {
      const move = trace.entries[i].action as Move;
      await sandboxEngine.applyCanonicalMove(move);
    }

    const sandboxBefore = sandboxEngine.getGameState();

    // --- Reconstruct backend state immediately BEFORE targetMoveNumber ---
    const backendEngine = createBackendEngineFromInitialState(trace.initialState as GameState);

    for (let i = 0; i < targetIndex; i++) {
      const move = trace.entries[i].action as Move;

      const backendStateBefore = backendEngine.getGameState();
      const backendMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
      const matching = strictFindMatchingBackendMove(move, backendMoves as Move[]);

      expect(matching).toBeDefined();

      const { id, timestamp, moveNumber, ...payload } = matching as Move;
      const result = await backendEngine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );

      expect(result.success).toBe(true);
    }

    const backendBefore = backendEngine.getGameState();

    return {
      trace: trace as any,
      targetIndex,
      targetMove,
      sandboxEngine,
      backendEngine,
      sandboxBefore,
      backendBefore,
    };
  }

  test('move 16: overtaking_capture C c1×d2→f4 is geometrically available on both engines', async () => {
    const TARGET_MOVE_NUMBER = 16;

    const { targetMove, sandboxEngine, backendEngine, sandboxBefore, backendBefore } =
      await reconstructPreStepStates(TARGET_MOVE_NUMBER);

    // Sanity-check that the sandbox canonical move matches the expected geometry.
    expect(targetMove.type).toBe('overtaking_capture');
    expect(targetMove.player).toBe(2);
    expect(targetMove.from).toEqual(pos(2, 0)); // c1
    expect(targetMove.captureTarget).toEqual(pos(3, 1)); // d2
    expect(targetMove.to).toEqual(pos(5, 3)); // f4

    // 1) Pre-step geometry and S-invariant parity should hold.
    expect(summarizeBoard(backendBefore.board)).toEqual(summarizeBoard(sandboxBefore.board));

    const backendSnapBefore = computeProgressSnapshot(backendBefore);
    const sandboxSnapBefore = computeProgressSnapshot(sandboxBefore);
    expect(backendSnapBefore).toEqual(sandboxSnapBefore);

    // 2) Phase and actor should agree.
    expect(backendBefore.currentPlayer).toBe(sandboxBefore.currentPlayer);
    expect(backendBefore.currentPhase).toBe(sandboxBefore.currentPhase);

    // 3) From this common pre-step state, the backend should expose a
    //    matching overtaking_capture from c1 over d2 to f4 in getValidMoves.
    const backendMoves = backendEngine.getValidMoves(backendBefore.currentPlayer);

    const matchingCapture = backendMoves.find((m) => {
      if (m.type !== 'overtaking_capture') return false;
      if (!m.from || !m.captureTarget) return false;
      return (
        m.player === targetMove.player &&
        m.from.x === targetMove.from!.x &&
        m.from.y === targetMove.from!.y &&
        m.captureTarget.x === targetMove.captureTarget!.x &&
        m.captureTarget.y === targetMove.captureTarget!.y &&
        m.to.x === targetMove.to!.x &&
        m.to.y === targetMove.to!.y
      );
    });

    expect(matchingCapture).toBeDefined();
  });

  test('move 33: chain_capture continuation from h4×f4→c4 keeps both engines in chain_capture for P2', async () => {
    const TARGET_MOVE_NUMBER = 33;

    const { targetMove, sandboxEngine, backendEngine, sandboxBefore, backendBefore } =
      await reconstructPreStepStates(TARGET_MOVE_NUMBER);

    // Sanity-check that the sandbox canonical move matches the expected geometry.
    expect(targetMove.type).toBe('continue_capture_segment');
    expect(targetMove.player).toBe(2);
    expect(targetMove.from).toEqual(pos(7, 3)); // h4
    expect(targetMove.captureTarget).toEqual(pos(5, 3)); // f4
    expect(targetMove.to).toEqual(pos(2, 3)); // c4

    // 1) Pre-step geometry and S-invariant parity should hold.
    expect(summarizeBoard(backendBefore.board)).toEqual(summarizeBoard(sandboxBefore.board));

    const backendSnapBefore = computeProgressSnapshot(backendBefore);
    const sandboxSnapBefore = computeProgressSnapshot(sandboxBefore);
    expect(backendSnapBefore).toEqual(sandboxSnapBefore);

    // 2) Both engines should agree that we are in chain_capture phase for player 2
    //    immediately before this canonical continuation.
    expect(sandboxBefore.currentPlayer).toBe(2);
    expect(sandboxBefore.currentPhase).toBe('chain_capture');

    expect(backendBefore.currentPlayer).toBe(2);
    expect(backendBefore.currentPhase).toBe('chain_capture');

    // 3) From this common pre-step state, the backend should expose a
    //    matching continue_capture_segment from h4 over f4 to c4 in getValidMoves.
    const backendMoves = backendEngine.getValidMoves(backendBefore.currentPlayer);

    const matchingContinuation = backendMoves.find((m) => {
      if (m.type !== 'continue_capture_segment') return false;
      if (!m.from || !m.captureTarget) return false;
      return (
        m.player === targetMove.player &&
        m.from.x === targetMove.from!.x &&
        m.from.y === targetMove.from!.y &&
        m.captureTarget.x === targetMove.captureTarget!.x &&
        m.captureTarget.y === targetMove.captureTarget!.y &&
        m.to.x === targetMove.to!.x &&
        m.to.y === targetMove.to!.y
      );
    });

    expect(matchingContinuation).toBeDefined();

    // 4) After applying the canonical continuation on both engines, they
    //    should agree on the exit from chain_capture: same player, same
    //    next phase, and no engine left stuck in chain_capture.
    if (!matchingContinuation) {
      return;
    }

    const { id, timestamp, moveNumber, ...payload } = matchingContinuation as Move;

    const backendResult = await backendEngine.makeMove(
      payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
    );
    expect(backendResult.success).toBe(true);

    await sandboxEngine.applyCanonicalMove(targetMove);

    const backendAfter = backendEngine.getGameState();
    const sandboxAfter = sandboxEngine.getGameState();

    expect(backendAfter.currentPlayer).toBe(sandboxAfter.currentPlayer);
    expect(backendAfter.currentPhase).toBe(sandboxAfter.currentPhase);

    // Neither engine should remain in chain_capture after this
    // continuation; exit semantics must match.
    expect(backendAfter.currentPhase).not.toBe('chain_capture');
    expect(sandboxAfter.currentPhase).not.toBe('chain_capture');
  });
});
