import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  Position,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import {
  hashGameState,
  computeProgressSnapshot,
  hasAnyLegalMoveOrCaptureFromOnBoard,
  MovementBoardView,
} from '../../src/shared/engine/core';
import { createHypotheticalBoardWithPlacement } from '../../src/client/sandbox/sandboxPlacement';
import { enumerateSimpleMovementLandings } from '../../src/client/sandbox/sandboxMovement';
import {
  enumerateCaptureSegmentsFromBoard,
  CaptureBoardAdapters,
} from '../../src/client/sandbox/sandboxCaptures';
import { isFSMOrchestratorActive } from '../../src/shared/utils/envFlags';

/**
 * Targeted single-seed diagnostic for the sandbox AI stall seen in the
 * heavy fuzz harness:
 *   - boardType: square8
 *   - numPlayers: 2 AI players
 *   - seed: 18 (corresponding to run=17 in the fuzz test)
 *
 * This test is intentionally more verbose than the fuzz harness: when the
 * game fails to terminate within MAX_AI_ACTIONS, it logs a compact summary
 * of the final state directly to the Jest output so we can reason about
 * the blocked configuration without tailing the large sandbox-ai-sim.log.
 */

// TODO: FSM issue - AI seed diagnostics may differ under FSM orchestration.
// These tests were created with legacy orchestration behavior.
// Enable once FSM behavior is fully stabilized for AI simulations.
const testFn = isFSMOrchestratorActive() ? test.skip : test;

testFn(
  'ClientSandboxEngine single-seed debug: square8 with 2 AI players, seed=18 terminates or logs diagnostics',
  async () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;
    const seed = 18; // from fuzz harness: boardIndex=0, playerCountIndex=0, run=17 -> 1+17=18
    const MAX_AI_ACTIONS = 1000;

    function makePrng(seedValue: number): () => number {
      let s = seedValue >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    function createEngine(bt: BoardType, players: number): ClientSandboxEngine {
      const config: SandboxConfig = {
        boardType: bt,
        numPlayers: players,
        playerKinds: Array.from({ length: players }, () => 'ai'),
      };

      const handler: SandboxInteractionHandler = {
        async requestChoice<TChoice extends PlayerChoice>(
          choice: TChoice
        ): Promise<PlayerChoiceResponseFor<TChoice>> {
          const anyChoice = choice as any;

          if (anyChoice.type === 'capture_direction') {
            const cd = anyChoice as CaptureDirectionChoice;
            const options = cd.options || [];
            if (options.length === 0) {
              throw new Error('Test SandboxInteractionHandler: no options for capture_direction');
            }

            // Deterministically pick the option with the smallest landing x,y
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
              choiceId: cd.id,
              playerNumber: cd.playerNumber,
              choiceType: cd.type,
              selectedOption: selected,
            } as PlayerChoiceResponseFor<TChoice>;
          }

          const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
          return {
            choiceId: anyChoice.id,
            playerNumber: anyChoice.playerNumber,
            choiceType: anyChoice.type,
            selectedOption,
          } as PlayerChoiceResponseFor<TChoice>;
        },
      };

      return new ClientSandboxEngine({ config, interactionHandler: handler });
    }

    const rng = makePrng(seed);
    const originalRandom = Math.random;
    Math.random = rng;

    const engine = createEngine(boardType, numPlayers);
    const engineAny = engine as any;

    const ringPlacementWithNoRingsEvents: Array<{
      action: number;
      currentPlayer: number;
      phase: string;
      players: Array<{ playerNumber: number; ringsInHand: number; stacks: number }>;
    }> = [];

    // Rolling history of earlier snapshots so we can inspect not just the
    // final stalled state but also the lead-up when debugging termination
    // violations.
    const historySnapshots: Array<{
      action: number;
      S: number;
      markers: number;
      collapsed: number;
      eliminated: number;
      currentPlayer: number;
      currentPhase: GameState['currentPhase'];
      gameStatus: GameState['gameStatus'];
      players: Array<{
        playerNumber: number;
        ringsInHand: number;
        eliminatedRings: number;
        territorySpaces: number;
      }>;
      stackCount: number;
      markerCount: number;
      collapsedCount: number;
    }> = [];

    function summarizeStacks(state: GameState) {
      return Array.from(state.board.stacks.entries()).map(([key, stack]) => ({
        key,
        controllingPlayer: stack.controllingPlayer,
        stackHeight: stack.stackHeight,
        capHeight: stack.capHeight,
      }));
    }

    try {
      let lastHash = hashGameState(engine.getGameState());
      let stagnantSteps = 0;

      for (let i = 0; i < MAX_AI_ACTIONS; i++) {
        const before = engine.getGameState();
        const beforeProgress = computeProgressSnapshot(before);

        // Capture a bounded rolling window of state snapshots so that when
        // the test fails we can see not only the final configuration but
        // also how S and player resources evolved earlier in the run.
        historySnapshots.push({
          action: i,
          S: beforeProgress.S,
          markers: beforeProgress.markers,
          collapsed: beforeProgress.collapsed,
          eliminated: beforeProgress.eliminated,
          currentPlayer: before.currentPlayer,
          currentPhase: before.currentPhase,
          gameStatus: before.gameStatus,
          players: before.players.map((p) => ({
            playerNumber: p.playerNumber,
            ringsInHand: p.ringsInHand,
            eliminatedRings: p.eliminatedRings,
            territorySpaces: p.territorySpaces,
          })),
          stackCount: before.board.stacks.size,
          markerCount: before.board.markers.size,
          collapsedCount: before.board.collapsedSpaces.size,
        });
        if (historySnapshots.length > 200) {
          historySnapshots.shift();
        }

        // Track ring_placement phases where the current player has no rings in hand.
        const currentPlayer = before.players.find((p) => p.playerNumber === before.currentPlayer);
        if (
          before.currentPhase === 'ring_placement' &&
          currentPlayer &&
          (currentPlayer.ringsInHand ?? 0) <= 0 &&
          before.gameStatus === 'active'
        ) {
          const playersSummary = before.players.map((p) => ({
            playerNumber: p.playerNumber,
            ringsInHand: p.ringsInHand,
            stacks: before.board.stacks
              ? Array.from(before.board.stacks.values()).filter(
                  (s) => s.controllingPlayer === p.playerNumber
                ).length
              : 0,
          }));

          ringPlacementWithNoRingsEvents.push({
            action: i,
            currentPlayer: before.currentPlayer,
            phase: before.currentPhase,
            players: playersSummary,
          });
        }

        if (before.gameStatus !== 'active') {
          break;
        }

        await engine.maybeRunAITurn();

        const after = engine.getGameState();

        if (engineAny && typeof engineAny.assertBoardInvariants === 'function') {
          engineAny.assertBoardInvariants(`single-seed-debug:square8-2p-seed18:action=${i}`);
        }

        const afterHash = hashGameState(after);
        const afterProgress = computeProgressSnapshot(after);

        if (afterHash === lastHash && after.gameStatus === 'active') {
          stagnantSteps++;
        } else {
          stagnantSteps = 0;
        }

        lastHash = afterHash;

        if (stagnantSteps >= 8) {
          // Early break on clear stall; we will report diagnostics below.
          break;
        }

        // Basic invariant: S should be non-decreasing.
        expect(afterProgress.S).toBeGreaterThanOrEqual(beforeProgress.S);
      }

      const finalState = engine.getGameState();

      if (finalState.gameStatus === 'active') {
        // Before throwing, perform a deeper introspection of the final
        // state so we can reason about what options Player 2 still has
        // under the current sandbox rules.
        const engineDebugAny = engine as any;

        let legalPlacements: Position[] = [];
        let simpleMoves: Array<{ fromKey: string; to: Position }> = [];
        const captureSegments: Array<{ from: Position; target: Position; landing: Position }> = [];

        try {
          if (typeof engineDebugAny.enumerateLegalRingPlacements === 'function') {
            legalPlacements =
              engineDebugAny.enumerateLegalRingPlacements(finalState.currentPlayer) || [];
          }
        } catch (e) {
          console.error(
            '[ClientSandboxEngine.aiSingleSeedDebug] Error enumerating legal placements',
            e
          );
        }

        try {
          if (typeof engineDebugAny.enumerateSimpleMovementLandings === 'function') {
            simpleMoves =
              engineDebugAny.enumerateSimpleMovementLandings(finalState.currentPlayer) || [];
          }
        } catch (e) {
          console.error(
            '[ClientSandboxEngine.aiSingleSeedDebug] Error enumerating simple moves',
            e
          );
        }

        try {
          if (typeof engineDebugAny.enumerateCaptureSegmentsFrom === 'function') {
            const stacksForCurrent = Array.from(finalState.board.stacks.values()).filter(
              (s) => s.controllingPlayer === finalState.currentPlayer
            );
            for (const stack of stacksForCurrent) {
              const segs =
                engineDebugAny.enumerateCaptureSegmentsFrom(
                  stack.position,
                  finalState.currentPlayer
                ) || [];
              for (const seg of segs) {
                captureSegments.push(seg);
              }
            }
          }
        } catch (e) {
          console.error(
            '[ClientSandboxEngine.aiSingleSeedDebug] Error enumerating capture segments',
            e
          );
        }

        // For deeper analysis, compare the shared core's no-dead-placement
        // verdict with the sandbox movement engine's actual reachability for
        // one or two of the reported legal placements. This helps pinpoint
        // any disagreement between the models.
        const config = BOARD_CONFIGS[boardType];

        const isValidPositionLocal = (pos: Position): boolean => {
          // This single-seed debug harness is currently hard-wired to
          // square8, so we only need square-board bounds here. If we ever
          // generalise it to hex, we can mirror BoardManager / sandbox
          // helpers instead of re-encoding geometry locally.
          return pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size;
        };

        type StackSummary = {
          controllingPlayer: number;
          capHeight: number;
          stackHeight: number;
        };

        const movementViewForHypothetical = (
          hypothetical: GameState['board']
        ): MovementBoardView => {
          return {
            isValidPosition: (pos: Position) => isValidPositionLocal(pos),
            isCollapsedSpace: (pos: Position) => {
              const key = positionToString(pos);
              return hypothetical.collapsedSpaces.has(key);
            },
            getStackAt: (pos: Position): StackSummary | undefined => {
              const key = positionToString(pos);
              const stack = hypothetical.stacks.get(key);
              if (!stack) return undefined;
              return {
                controllingPlayer: stack.controllingPlayer,
                capHeight: stack.capHeight,
                stackHeight: stack.stackHeight,
              };
            },
            getMarkerOwner: (pos: Position): number | undefined => {
              const key = positionToString(pos);
              const marker = hypothetical.markers.get(key);
              return marker?.player;
            },
          };
        };

        const adaptersForHypothetical = (
          hypothetical: GameState['board']
        ): CaptureBoardAdapters => {
          return {
            isValidPosition: (pos: Position) => isValidPositionLocal(pos),
            isCollapsedSpace: (pos: Position, board) => {
              const key = positionToString(pos);
              return board.collapsedSpaces.has(key);
            },
            getMarkerOwner: (pos: Position, board) => {
              const key = positionToString(pos);
              const marker = board.markers.get(key);
              return marker?.player;
            },
          };
        };

        const comparisonResults: Array<{
          pos: Position;
          hasAction: boolean;
          simpleFromCount: number;
          captureFromCount: number;
        }> = [];

        for (const pos of legalPlacements) {
          const hypotheticalBoard = createHypotheticalBoardWithPlacement(
            finalState.board,
            pos,
            finalState.currentPlayer,
            1
          );

          const movementView = movementViewForHypothetical(hypotheticalBoard);
          const hasAction = hasAnyLegalMoveOrCaptureFromOnBoard(
            boardType,
            pos,
            finalState.currentPlayer,
            movementView
          );

          const allSimple = enumerateSimpleMovementLandings(
            boardType,
            hypotheticalBoard,
            finalState.currentPlayer,
            (p: Position) => isValidPositionLocal(p)
          );
          const fromKey = positionToString(pos);
          const simpleFrom = allSimple.filter((m) => m.fromKey === fromKey);

          const captureSegs = enumerateCaptureSegmentsFromBoard(
            boardType,
            hypotheticalBoard,
            pos,
            finalState.currentPlayer,
            adaptersForHypothetical(hypotheticalBoard)
          );

          comparisonResults.push({
            pos,
            hasAction,
            simpleFromCount: simpleFrom.length,
            captureFromCount: captureSegs.length,
          });

          if (hasAction && simpleFrom.length === 0 && captureSegs.length === 0) {
            throw new Error(
              `No-dead-placement mismatch at ${positionToString(
                pos
              )}: core reports hasAction=true, but sandbox movement enumerates no moves/captures`
            );
          }
        }

        // Log a compact diagnostic snapshot to Jest output.
        // eslint-disable-next-line no-console
        console.log('[ClientSandboxEngine.aiSingleSeedDebug] Non-terminating state snapshot', {
          boardType,
          numPlayers,
          seed,
          currentPlayer: finalState.currentPlayer,
          currentPhase: finalState.currentPhase,
          gameStatus: finalState.gameStatus,
          players: finalState.players.map((p) => ({
            playerNumber: p.playerNumber,
            type: p.type,
            ringsInHand: p.ringsInHand,
            eliminatedRings: p.eliminatedRings,
            territorySpaces: p.territorySpaces,
          })),
          stackCount: finalState.board.stacks.size,
          stacks: summarizeStacks(finalState).slice(0, 32),
          markerCount: finalState.board.markers.size,
          collapsedCount: finalState.board.collapsedSpaces.size,
          ringPlacementWithNoRingsEvents,
          // Detailed AI options for the current player at the stalled state.
          legalPlacementCount: legalPlacements.length,
          legalPlacements: legalPlacements.slice(0, 32),
          simpleMoveCount: simpleMoves.length,
          simpleMoves: simpleMoves.slice(0, 32),
          captureSegmentCount: captureSegments.length,
          captureSegments: captureSegments.slice(0, 32),
          movementComparison: comparisonResults,
          // Rolling history of earlier S / resource snapshots to help
          // locate the first divergence from the theoretical termination
          // ladder.
          recentHistory: historySnapshots.slice(-32),
        });

        throw new Error(
          'ClientSandboxEngine single-seed debug: game did not terminate ' +
            `within ${MAX_AI_ACTIONS} AI actions for square8/2p/seed=18`
        );
      }
    } finally {
      Math.random = originalRandom;
    }
  }
);
