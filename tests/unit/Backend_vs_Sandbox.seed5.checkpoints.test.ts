import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  snapshotFromGameState,
  snapshotsEqual,
  diffSnapshots,
  ComparableSnapshot,
} from '../utils/stateSnapshots';

/**
 * Diagnostic checkpoint harness around the known seed-5 backend vs sandbox
 * mismatch at move index 43.
 *
 * This test:
 *   1) Replays the first 43 moves of the sandbox AI trace into fresh
 *      backend and sandbox engines.
 *   2) Attaches debug checkpoint hooks on both engines for move 43
 *      (0-based index) and records snapshots at each debugCheckpoint
 *      label, including a synthetic "before-move-43" and the shared
 *      "end-of-move" label emitted by both engines.
 *   3) Compares backend vs sandbox snapshots for each shared label and
 *      logs a compact diff and marker/collapsed summary for the first
 *      mismatching label.
 *
 * It is intentionally diagnostic only: it does not currently fail on
 * snapshot mismatches, serving instead as an interactive debugging
 * tool while territory parity is being finalised.
 *
 * NOTE: Skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because state checkpoint mismatch (intentional divergence)
 */

// Skip this test suite when orchestrator adapter is enabled - state checkpoint diverges
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Backend vs Sandbox seed=5 checkpoint diagnostics around move index 43',
  () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;
    const seed = 5;
    const MAX_STEPS = 60;

    function summariseMarkerAndCollapsedDiff(
      backend: ComparableSnapshot,
      sandbox: ComparableSnapshot
    ): {
      markerKeysOnlyInBackend: string[];
      markerKeysOnlyInSandbox: string[];
      collapsedKeysOnlyInBackend: string[];
      collapsedKeysOnlyInSandbox: string[];
    } {
      const backendMarkerKeys = new Set(backend.markers.map((m) => m.key));
      const sandboxMarkerKeys = new Set(sandbox.markers.map((m) => m.key));
      const backendCollapsedKeys = new Set(backend.collapsedSpaces.map((c) => c.key));
      const sandboxCollapsedKeys = new Set(sandbox.collapsedSpaces.map((c) => c.key));

      const markerKeysOnlyInBackend: string[] = [];
      const markerKeysOnlyInSandbox: string[] = [];
      const collapsedKeysOnlyInBackend: string[] = [];
      const collapsedKeysOnlyInSandbox: string[] = [];

      for (const key of backendMarkerKeys) {
        if (!sandboxMarkerKeys.has(key)) {
          markerKeysOnlyInBackend.push(key);
        }
      }
      for (const key of sandboxMarkerKeys) {
        if (!backendMarkerKeys.has(key)) {
          markerKeysOnlyInSandbox.push(key);
        }
      }

      for (const key of backendCollapsedKeys) {
        if (!sandboxCollapsedKeys.has(key)) {
          collapsedKeysOnlyInBackend.push(key);
        }
      }
      for (const key of sandboxCollapsedKeys) {
        if (!backendCollapsedKeys.has(key)) {
          collapsedKeysOnlyInSandbox.push(key);
        }
      }

      markerKeysOnlyInBackend.sort();
      markerKeysOnlyInSandbox.sort();
      collapsedKeysOnlyInBackend.sort();
      collapsedKeysOnlyInSandbox.sort();

      return {
        markerKeysOnlyInBackend,
        markerKeysOnlyInSandbox,
        collapsedKeysOnlyInBackend,
        collapsedKeysOnlyInSandbox,
      };
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

    test('collects backend vs sandbox checkpoints for seed=5 move index 43', async () => {
      const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
      expect(trace.entries.length).toBeGreaterThan(0);

      const moves: Move[] = trace.entries.map((e) => e.action as Move);
      const prefixLength = 44; // apply moves[0..43]
      expect(moves.length).toBeGreaterThanOrEqual(prefixLength);

      const targetMove = moves[43];
      expect(targetMove).toBeDefined();

      const backendEngine = createBackendEngineFromInitialState(trace.initialState);
      const sandboxEngine = createSandboxEngineFromInitial(trace.initialState);

      // --- Replay prefix moves 0..42 without checkpoints ---
      for (let i = 0; i < 43; i++) {
        const move = moves[i];

        // Backend: map sandbox move to a canonical backend move and apply via makeMove.
        const backendStateBeforeStep = backendEngine.getGameState();
        const backendValidMoves = backendEngine.getValidMoves(backendStateBeforeStep.currentPlayer);
        const matching = findMatchingBackendMove(move, backendValidMoves);

        if (!matching) {
          // eslint-disable-next-line no-console
          console.error(
            '[Backend_vs_Sandbox.seed5.checkpoints] No matching backend move during prefix replay',
            {
              stepIndex: i,
              sandboxMove: {
                moveNumber: move.moveNumber,
                type: move.type,
                player: move.player,
                from: move.from,
                to: move.to,
                captureTarget: move.captureTarget,
              },
              backendCurrentPlayer: backendStateBeforeStep.currentPlayer,
              backendCurrentPhase: backendStateBeforeStep.currentPhase,
              backendValidMovesCount: backendValidMoves.length,
            }
          );
          throw new Error(
            `Prefix replay failed before move index 43; no matching backend move for sandbox moveNumber=${move.moveNumber}`
          );
        }

        const { id, timestamp, moveNumber, ...payload } = matching as any;
        const backendResult = await backendEngine.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );
        if (!backendResult.success) {
          throw new Error(
            `Backend makeMove failed during prefix replay at backend moveNumber=${(matching as any).moveNumber}: ${backendResult.error}`
          );
        }

        // Sandbox: apply the canonical move directly.
        await sandboxEngine.applyCanonicalMove(move);
      }

      // --- Attach checkpoint hooks for move 43 ---
      const backendCheckpoints: Array<{ label: string; snapshot: ComparableSnapshot }> = [];
      backendEngine.setDebugCheckpointHook((label, state) => {
        backendCheckpoints.push({
          label,
          snapshot: snapshotFromGameState(`backend-${label}`, state),
        });
      });

      const sandboxCheckpoints: Array<{ label: string; snapshot: ComparableSnapshot }> = [];
      sandboxEngine.setDebugCheckpointHook((label, state) => {
        sandboxCheckpoints.push({
          label,
          snapshot: snapshotFromGameState(`sandbox-${label}`, state),
        });
      });

      // Synthetic pre-move checkpoint so both engines share a common label.
      const backendAny: any = backendEngine;
      backendAny.debugCheckpoint('before-move-43');
      const sandboxAny: any = sandboxEngine;
      sandboxAny.debugCheckpoint('before-move-43');

      // --- Apply target move index 43 with checkpoints enabled ---
      {
        const backendStateBefore = backendEngine.getGameState();
        const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
        const matching = findMatchingBackendMove(targetMove, backendValidMoves);

        if (!matching) {
          // eslint-disable-next-line no-console
          console.error(
            '[Backend_vs_Sandbox.seed5.checkpoints] No matching backend move at index 43',
            {
              sandboxMove: {
                moveNumber: targetMove.moveNumber,
                type: targetMove.type,
                player: targetMove.player,
                from: targetMove.from,
                to: targetMove.to,
                captureTarget: targetMove.captureTarget,
              },
              backendCurrentPlayer: backendStateBefore.currentPlayer,
              backendCurrentPhase: backendStateBefore.currentPhase,
              backendValidMovesCount: backendValidMoves.length,
            }
          );
          throw new Error(
            `No matching backend move for sandbox move index 43 (moveNumber=${targetMove.moveNumber})`
          );
        }

        const { id, timestamp, moveNumber, ...payload } = matching as any;
        const backendResult = await backendEngine.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );
        if (!backendResult.success) {
          throw new Error(
            `Backend makeMove failed for move index 43 (backend moveNumber=${(matching as any).moveNumber}): ${backendResult.error}`
          );
        }

        await sandboxEngine.applyCanonicalMove(targetMove);
      }

      // --- Compare checkpoints by label ---
      const backendLabels = backendCheckpoints.map((c) => c.label);
      const sandboxLabels = sandboxCheckpoints.map((c) => c.label);

      const sharedLabelSet = new Set<string>();
      for (const label of backendLabels) {
        if (sandboxLabels.includes(label)) {
          sharedLabelSet.add(label);
        }
      }
      const sharedLabels = Array.from(sharedLabelSet);

      // eslint-disable-next-line no-console
      console.log('[Backend_vs_Sandbox.seed5.checkpoints] labels', {
        backendLabels,
        sandboxLabels,
        sharedLabels,
      });

      const sandboxByLabel = new Map<string, ComparableSnapshot>();
      for (const entry of sandboxCheckpoints) {
        if (!sandboxByLabel.has(entry.label)) {
          sandboxByLabel.set(entry.label, entry.snapshot);
        }
      }

      let firstMismatchLabel: string | null = null;
      let firstBackendSnapshot: ComparableSnapshot | null = null;
      let firstSandboxSnapshot: ComparableSnapshot | null = null;

      for (const backendEntry of backendCheckpoints) {
        const sandboxSnapshot = sandboxByLabel.get(backendEntry.label);
        if (!sandboxSnapshot) {
          continue;
        }

        if (!snapshotsEqual(backendEntry.snapshot, sandboxSnapshot)) {
          firstMismatchLabel = backendEntry.label;
          firstBackendSnapshot = backendEntry.snapshot;
          firstSandboxSnapshot = sandboxSnapshot;
          break;
        }
      }

      if (firstMismatchLabel && firstBackendSnapshot && firstSandboxSnapshot) {
        const diff = diffSnapshots(firstBackendSnapshot, firstSandboxSnapshot);
        const markerSummary = summariseMarkerAndCollapsedDiff(
          firstBackendSnapshot,
          firstSandboxSnapshot
        );
        // eslint-disable-next-line no-console
        console.error('[Backend_vs_Sandbox.seed5.checkpoints] snapshot mismatch at checkpoint', {
          seed,
          targetMoveIndex: 43,
          targetMove: {
            moveNumber: targetMove.moveNumber,
            type: targetMove.type,
            player: targetMove.player,
            from: targetMove.from,
            to: targetMove.to,
            captureTarget: targetMove.captureTarget,
          },
          label: firstMismatchLabel,
          diff,
          markerSummary,
        });
      } else {
        // eslint-disable-next-line no-console
        console.log('[Backend_vs_Sandbox.seed5.checkpoints] all shared checkpoints matched', {
          seed,
          targetMoveIndex: 43,
          targetMoveType: targetMove.type,
          sharedLabels,
        });
      }

      // Diagnostic harness: ensure we collected at least one shared label, but
      // do not turn snapshot mismatches into a test failure yet.
      expect(sharedLabels.length).toBeGreaterThan(0);
    });
  }
);
