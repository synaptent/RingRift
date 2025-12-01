import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { hashGameState, summarizeBoard } from '../../src/shared/engine/core';
import { pos } from '../utils/fixtures';

/**
 * Helper/debug test: locate the FIRST move index in the seed-17 trace where
 * backend and sandbox diverge when we replay the sandbox canonical moves
 * into a fresh backend GameEngine using a **strict** backend move matcher.
 *
 * This mirrors the seed-5/seed-14 helpers but intentionally does **not** use
 * the relaxed capture-vs-move or placementCount equivalence from
 * movesLooselyMatch. For seed-17 we want an early failure whenever the
 * backend:
 *   - classifies a sandbox overtaking_capture as a simple move_stack, or
 *   - disagrees on placementCount for multi-ring placements, or
 *   - exposes a different from/to/captureTarget geometry even when types
 *     are the same.
 *
 * The goal is to pinpoint the *earliest* move where strict replay fails or
 * where phase/hash orchestration diverges, so we can then reconstruct that
 * pre-step state in an isolated parity test (similar to Seed17Move52Parity)
 * and patch backend semantics accordingly.
 */

/**
 * TODO-SEED17-STRICT-DIVERGENCE: This diagnostic test uses a strict
 * matcher to locate the first move index where backend and sandbox diverge
 * for seed 17. It requires complex trace infrastructure and is a
 * diagnostic helper, not a regression test.
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

maybeDescribe('Trace parity first-divergence helper: square8 / 2p / seed=17', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 17;
  const MAX_STEPS = 80;

  /**
   * Strict backend move matcher for this helper, copied from
   * Seed17Move16And33Parity with only minimal type/dep changes. See that
   * file for detailed commentary.
   */
  function strictFindMatchingBackendMove(reference: Move, candidates: Move[]): Move | null {
    for (const candidate of candidates) {
      if (candidate.player !== reference.player) continue;

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

      // For line-processing decisions, require that the underlying line
      // geometry (and, when present, collapsedMarkers for reward choices)
      // matches across engines so that we always replay the same logical
      // line into the backend that the sandbox trace recorded.
      if (
        reference.type === 'process_line' &&
        candidate.type === 'process_line' &&
        reference.formedLines &&
        reference.formedLines[0] &&
        candidate.formedLines &&
        candidate.formedLines[0]
      ) {
        const refLine = reference.formedLines[0];
        const candLine = candidate.formedLines[0];

        const refPositions = refLine.positions ?? [];
        const candPositions = candLine.positions ?? [];

        if (refPositions.length === candPositions.length) {
          const refKeys = new Set(refPositions.map((p) => `${p.x},${p.y}`));
          const candKeys = new Set(candPositions.map((p) => `${p.x},${p.y}`));

          if (refKeys.size === candKeys.size) {
            let allMatch = true;
            for (const key of refKeys) {
              if (!candKeys.has(key)) {
                allMatch = false;
                break;
              }
            }
            if (allMatch) {
              return candidate;
            }
          }
        }

        continue;
      }

      if (
        reference.type === 'choose_line_reward' &&
        candidate.type === 'choose_line_reward' &&
        reference.formedLines &&
        reference.formedLines[0] &&
        candidate.formedLines &&
        candidate.formedLines[0]
      ) {
        const refLine = reference.formedLines[0];
        const candLine = candidate.formedLines[0];

        const refPositions = refLine.positions ?? [];
        const candPositions = candLine.positions ?? [];

        const sameLine =
          refPositions.length === candPositions.length &&
          (() => {
            const refKeys = new Set(refPositions.map((p) => `${p.x},${p.y}`));
            const candKeys = new Set(candPositions.map((p) => `${p.x},${p.y}`));
            if (refKeys.size !== candKeys.size) return false;
            for (const key of refKeys) {
              if (!candKeys.has(key)) return false;
            }
            return true;
          })();

        if (!sameLine) {
          continue;
        }

        const refCollapsed = reference.collapsedMarkers ?? [];
        const candCollapsed = candidate.collapsedMarkers ?? [];

        if (refCollapsed.length || candCollapsed.length) {
          if (refCollapsed.length !== candCollapsed.length) {
            continue;
          }
          const refCKeys = new Set(refCollapsed.map((p) => `${p.x},${p.y}`));
          const candCKeys = new Set(candCollapsed.map((p) => `${p.x},${p.y}`));
          if (refCKeys.size !== candCKeys.size) {
            continue;
          }
          let allCollapsedMatch = true;
          for (const key of refCKeys) {
            if (!candCKeys.has(key)) {
              allCollapsedMatch = false;
              break;
            }
          }
          if (!allCollapsedMatch) {
            continue;
          }
        }

        return candidate;
      }

      // For territory-processing decisions, require that the disconnected
      // region being processed matches exactly (up to set equality of
      // spaces). This keeps the strict helper aligned with the looser
      // move-matcher used by the main trace harness.
      if (reference.type === 'process_territory_region') {
        if (candidate.type !== 'process_territory_region') {
          continue;
        }

        const refRegion = reference.disconnectedRegions && reference.disconnectedRegions[0];
        const candRegion = candidate.disconnectedRegions && candidate.disconnectedRegions[0];

        const refSpaces = refRegion?.spaces ?? [];
        const candSpaces = candRegion?.spaces ?? [];

        if (refSpaces.length && candSpaces.length) {
          if (refSpaces.length !== candSpaces.length) {
            continue;
          }

          const refKeys = new Set(refSpaces.map((p) => `${p.x},${p.y}`));
          const candKeys = new Set(candSpaces.map((p) => `${p.x},${p.y}`));

          if (refKeys.size !== candKeys.size) {
            continue;
          }

          let allMatch = true;
          for (const key of refKeys) {
            if (!candKeys.has(key)) {
              allMatch = false;
              break;
            }
          }
          if (!allMatch) {
            continue;
          }

          return candidate;
        }
        // If either side lacks region metadata, fall through to the
        // generic positional checks below.
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

  test('log first backend vs sandbox hash/phase divergence for seed 17 (strict matcher)', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    const engine = createBackendEngineFromInitialState(trace.initialState as GameState);

    let firstMismatchIndex = -1;

    for (let i = 0; i < trace.entries.length; i++) {
      const entry = trace.entries[i];
      const move = entry.action as Move;

      // Always advance the backend through automatic bookkeeping phases so we
      // only compare interactive phases (placement/movement/capture).
      await engine.stepAutomaticPhasesForTesting();

      const backendBefore = engine.getGameState();
      const backendMoves = engine.getValidMoves(backendBefore.currentPlayer);
      const matching = strictFindMatchingBackendMove(move, backendMoves as Move[]);

      if (!matching) {
        firstMismatchIndex = i;
        // eslint-disable-next-line no-console
        console.log('FIRST MOVE MATCH FAILURE (strict) at index', i, 'moveNumber', move.moveNumber);
        // eslint-disable-next-line no-console
        console.log('Sandbox Move:', JSON.stringify(move, null, 2));
        // eslint-disable-next-line no-console
        console.log(
          'Backend State Summary (before move):',
          JSON.stringify(summarizeBoard(backendBefore.board), null, 2)
        );
        // eslint-disable-next-line no-console
        console.log('Backend State Hash (before move):', hashGameState(backendBefore));
        // eslint-disable-next-line no-console
        console.log('Backend currentPlayer/currentPhase BEFORE move:', {
          currentPlayer: backendBefore.currentPlayer,
          currentPhase: backendBefore.currentPhase,
          gameStatus: backendBefore.gameStatus,
        });
        // eslint-disable-next-line no-console
        console.log('Sandbox phase/status BEFORE move (from trace entry):', {
          actor: entry.actor,
          phaseBefore: entry.phaseBefore,
          statusBefore: entry.statusBefore,
        });
        break;
      }

      const { id, timestamp, moveNumber, ...payload } = matching as Move;
      const result = await engine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      if (!result.success) {
        firstMismatchIndex = i;
        // eslint-disable-next-line no-console
        console.log(
          'BACKEND makeMove failure (strict) at index',
          i,
          'moveNumber',
          move.moveNumber,
          'error:',
          result.error
        );
        break;
      }

      const backendAfter = engine.getGameState();
      const backendHashAfter = hashGameState(backendAfter);
      const sandboxHashAfter = entry.stateHashAfter;

      const backendPhaseAfter = backendAfter.currentPhase;
      const backendPlayerAfter = backendAfter.currentPlayer;
      const sandboxPhaseAfter = entry.phaseAfter;
      const sandboxActor = entry.actor;

      if (sandboxPhaseAfter && backendPhaseAfter !== sandboxPhaseAfter) {
        firstMismatchIndex = i;
        // eslint-disable-next-line no-console
        console.log('FIRST PHASE DIVERGENCE (strict) at index', i, 'moveNumber', move.moveNumber);
        // eslint-disable-next-line no-console
        console.log('Sandbox phases/status (from trace entry):', {
          actor: sandboxActor,
          phaseBefore: entry.phaseBefore,
          phaseAfter: entry.phaseAfter,
          statusBefore: entry.statusBefore,
          statusAfter: entry.statusAfter,
        });
        // eslint-disable-next-line no-console
        console.log('Backend phases/status AFTER move (from live state):', {
          currentPlayer: backendPlayerAfter,
          currentPhase: backendPhaseAfter,
          gameStatus: backendAfter.gameStatus,
        });

        const backendHistoryLast = backendAfter.history[backendAfter.history.length - 1];
        // eslint-disable-next-line no-console
        console.log(
          'Backend last history entry (if any):',
          backendHistoryLast && {
            actor: backendHistoryLast.actor,
            phaseBefore: backendHistoryLast.phaseBefore,
            phaseAfter: backendHistoryLast.phaseAfter,
            statusBefore: backendHistoryLast.statusBefore,
            statusAfter: backendHistoryLast.statusAfter,
          }
        );

        // eslint-disable-next-line no-console
        console.log(
          'Sandbox State Summary AFTER move:',
          JSON.stringify(entry.boardAfterSummary, null, 2)
        );
        // eslint-disable-next-line no-console
        console.log(
          'Backend State Summary AFTER move:',
          JSON.stringify(summarizeBoard(backendAfter.board), null, 2)
        );
        break;
      }

      if (sandboxHashAfter && backendHashAfter && backendHashAfter !== sandboxHashAfter) {
        firstMismatchIndex = i;
        // eslint-disable-next-line no-console
        console.log('FIRST HASH DIVERGENCE (strict) at index', i, 'moveNumber', move.moveNumber);
        // eslint-disable-next-line no-console
        console.log('Sandbox Move at divergence:', JSON.stringify(move, null, 2));
        // eslint-disable-next-line no-console
        console.log(
          'Sandbox State Summary BEFORE move (from trace entry):',
          JSON.stringify(entry.boardBeforeSummary, null, 2)
        );
        // eslint-disable-next-line no-console
        console.log('Sandbox State Hash (before move):', entry.stateHashBefore);
        // eslint-disable-next-line no-console
        console.log(
          'Sandbox State Summary AFTER move:',
          JSON.stringify(entry.boardAfterSummary, null, 2)
        );
        // eslint-disable-next-line no-console
        console.log('Sandbox State Hash (after move):', sandboxHashAfter);
        // eslint-disable-next-line no-console
        console.log(
          'Backend State Summary AFTER move:',
          JSON.stringify(summarizeBoard(backendAfter.board), null, 2)
        );
        // eslint-disable-next-line no-console
        console.log('Backend State Hash (after move):', backendHashAfter);
        // eslint-disable-next-line no-console
        console.log('Backend currentPlayer/currentPhase AFTER move:', {
          currentPlayer: backendPlayerAfter,
          currentPhase: backendPhaseAfter,
          gameStatus: backendAfter.gameStatus,
        });
        // eslint-disable-next-line no-console
        console.log('Sandbox phase/status AFTER move (from trace entry):', {
          actor: sandboxActor,
          phaseAfter: entry.phaseAfter,
          statusAfter: entry.statusAfter,
        });
        break;
      }
    }

    if (firstMismatchIndex === -1) {
      // eslint-disable-next-line no-console
      console.log('No strict hash/phase divergence found for seed 17 up to maxSteps', MAX_STEPS);
    } else {
      throw new Error(
        `Backend vs Sandbox trace parity divergence for seed 17 (strict) at index ${firstMismatchIndex}. ` +
          'See earlier console diagnostics for phase/hash mismatch details.'
      );
    }

    expect(trace.entries.length).toBeGreaterThan(0);
  });
});
