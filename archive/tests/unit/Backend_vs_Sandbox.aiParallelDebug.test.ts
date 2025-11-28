/**
 * Archived diagnostic suite: Backend vs Sandbox parallel AI-style simulations.
 *
 * This file was moved from tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts
 * and is retained for historical/debugging reference only. It is not part of
 * the canonical rules or CI gating suites.
 */
import { GameEngine } from '../../../src/server/game/GameEngine';
import {
  BoardType,
  BOARD_CONFIGS,
  GameState,
  Move,
  Player,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../../src/shared/types/game';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../../src/client/sandbox/ClientSandboxEngine';

/**
 * Backend vs Sandbox parallel AI-style simulations.
 *
 * This is a diagnostic harness intended for debugging divergences between the
 * server-side GameEngine and the client-side ClientSandboxEngine. It drives
 * both engines from the same initial configuration and seeded PRNG, stepping
 * them forward in lockstep and comparing coarse-grained game state summaries
 * after each pair of AI actions.
 *
 * Notes / limitations:
 * - The backend side chooses a random legal move from GameEngine.getValidMoves
 *   for the current player (as in GameEngine.aiSimulation.test.ts).
 * - The sandbox side uses ClientSandboxEngine.maybeRunAITurn() to perform
 *   whatever action its internal AI policy selects for the current player.
 * - Because the AI policies differ, we do NOT expect the exact same moves; the
 *   goal is to detect structural divergences (e.g. backend reaches a terminal
 *   state but sandbox remains active, or invariant counts diverge badly).
 * - This test is intentionally skipped by default; enable it locally when
 *   investigating specific seeds/scenarios.
 */

describe.skip('Backend vs Sandbox parallel AI debug harness (square8/2p focus)', () => {
  const boardTypes: BoardType[] = ['square8'];
  const playerCounts: number[] = [2, 3];

  const RUNS_PER_SCENARIO = 10;
  const MAX_STEPS = 2000;

  function makePrng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  }

  function createBackendEngine(boardType: BoardType, numPlayers: number): GameEngine {
    const timeControl = { initialTime: 600, increment: 0, type: 'blitz' as const };
    const boardConfig = BOARD_CONFIGS[boardType];

    const players: Player[] = Array.from({ length: numPlayers }, (_, idx) => {
      const playerNumber = idx + 1;
      return {
        id: `p${playerNumber}`,
        username: `Player${playerNumber}`,
        type: 'ai',
        playerNumber,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: boardConfig.ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      } as Player;
    });

    const engine = new GameEngine(
      'backend-vs-sandbox-debug',
      boardType,
      players,
      timeControl,
      false
    );
    const started = engine.startGame();
    if (!started) {
      throw new Error('Failed to start GameEngine for backend vs sandbox debug harness');
    }
    return engine;
  }

  function createSandboxEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'ai'),
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
            throw new Error('SandboxInteractionHandler: no options for capture_direction');
          }

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
          };
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        };
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler, traceMode: true });
  }

  function summarizeState(state: GameState): { status: GameState['gameStatus']; hash: string } {
    return {
      status: state.gameStatus,
      hash: JSON.stringify({
        boardType: state.boardType,
        currentPlayer: state.currentPlayer,
        stacks: Array.from(state.board.stacks.entries()).length,
        markers: Array.from(state.board.markers.entries()).length,
        collapsed: Array.from(state.board.collapsedSpaces.entries()).length,
      }),
    };
  }

  boardTypes.forEach((boardType) => {
    playerCounts.forEach((numPlayers) => {
      test(`parallel AI debug for ${boardType} / ${numPlayers}p`, async () => {
        const prng = makePrng(42);

        for (let run = 0; run < RUNS_PER_SCENARIO; run += 1) {
          const backend = createBackendEngine(boardType, numPlayers);
          const sandbox = createSandboxEngine(boardType, numPlayers);

          let steps = 0;
          while (steps < MAX_STEPS) {
            const backendState = backend.getGameState();
            const sandboxState = sandbox.getGameState();

            const backendSummary = summarizeState(backendState);
            const sandboxSummary = summarizeState(sandboxState);

            if (backendSummary.status !== 'active' || sandboxSummary.status !== 'active') {
              break;
            }

            const backendMoves = backend.getValidMoves();
            if (backendMoves.length === 0) {
              break;
            }

            const moveIndex = Math.floor(prng() * backendMoves.length);
            const backendMove: Move = backendMoves[moveIndex];
            backend.applyCanonicalMove(backendMove);

            await sandbox.maybeRunAITurn();

            steps += 1;
          }
        }
      });
    });
  });
});
