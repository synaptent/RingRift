import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BOARD_CONFIGS,
  GameState,
  Move,
  Player,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice
} from '../../src/shared/types/game';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler
} from '../../src/client/sandbox/ClientSandboxEngine';

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

describe('Backend vs Sandbox parallel AI debug harness (square8/2p focus)', () => {
  // Focus on the scenarios that currently exhibit sandbox stalls in
  // ClientSandboxEngine.aiSimulation.test.ts. We can widen this matrix
  // later once we have a clear picture for these seeds.
  const boardTypes: BoardType[] = ['square8'];
  const playerCounts: number[] = [2, 3];

  // Use enough runs to cover the interesting seeds from the sandbox
  // AI simulation logs (e.g. seed=14 => run=13 for 2p; seed=1010 => run=9 for 3p),
  // but keep this relatively light so it remains practical to run locally.
  const RUNS_PER_SCENARIO = 10;
  const MAX_STEPS = 2000; // paired backend+sandbox AI actions, well below the main 10k caps

  /** Tiny deterministic PRNG (same LCG as other AI simulation tests). */
  function makePrng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
      // LCG parameters from Numerical Recipes
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
        territorySpaces: 0
      } as Player;
    });

    const engine = new GameEngine('backend-vs-sandbox-debug', boardType, players, timeControl, false);
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
      playerKinds: Array.from({ length: numPlayers }, () => 'ai')
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
            choiceId: cd.id,
            playerNumber: cd.playerNumber,
            choiceType: cd.type,
            selectedOption: selected
          } as PlayerChoiceResponseFor<TChoice>;
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption
        } as PlayerChoiceResponseFor<TChoice>;
      }
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  /**
   * Choose a random legal move for the backend engine using getValidMoves,
   * driven by the provided PRNG. Mirrors GameEngine.aiSimulation.test.ts.
   */
  function chooseRandomBackendMove(engine: GameEngine, state: GameState, rng: () => number): Move | null {
    const currentPlayer = state.currentPlayer;
    const moves = engine.getValidMoves(currentPlayer);

    if (!moves.length) {
      return null;
    }

    const idx = Math.floor(rng() * moves.length);
    return moves[Math.min(idx, moves.length - 1)];
  }

  interface SummaryPlayer {
    ringsInHand: number;
    eliminatedRings: number;
    territorySpaces: number;
  }

  interface StateSummary {
    gameStatus: GameState['gameStatus'];
    currentPlayer: number;
    currentPhase: GameState['currentPhase'];
    players: SummaryPlayer[];
    stacks: number;
    markers: number;
    collapsed: number;
    totalRingsEliminated: number;
  }

  function summariseState(state: GameState): StateSummary {
    const stacks = state.board.stacks.size;
    const markers = state.board.markers.size;
    const collapsed = state.board.collapsedSpaces.size;

    const players: SummaryPlayer[] = state.players
      .slice()
      .sort((a, b) => a.playerNumber - b.playerNumber)
      .map(p => ({
        ringsInHand: p.ringsInHand,
        eliminatedRings: p.eliminatedRings,
        territorySpaces: p.territorySpaces
      }));

    return {
      gameStatus: state.gameStatus,
      currentPlayer: state.currentPlayer,
      currentPhase: state.currentPhase,
      players,
      stacks,
      markers,
      collapsed,
      totalRingsEliminated: state.totalRingsEliminated ?? 0
    };
  }

  function isStructurallyTerminal(state: GameState): boolean {
    const noStacks = state.board.stacks.size === 0;
    const anyRingsInHand = state.players.some(p => p.ringsInHand > 0);
    return noStacks && !anyRingsInHand;
  }

  function getLocalContext(state: GameState, move: Move | null): any {
    if (!move) {
      return { positions: [] };
    }

    const positions: {
      label: string;
      key: string;
      stack?: { controllingPlayer: number; stackHeight: number; capHeight: number };
      marker?: { player: number };
      collapsedOwner?: number;
    }[] = [];

    const addPosition = (label: string, pos: { x: number; y: number; z?: number } | undefined) => {
      if (!pos) return;
      const key = pos.z !== undefined ? `${pos.x},${pos.y},${pos.z}` : `${pos.x},${pos.y}`;
      const stack = state.board.stacks.get(key);
      const marker = state.board.markers.get(key);
      const collapsedOwner = state.board.collapsedSpaces.get(key);

      positions.push({
        label,
        key,
        stack:
          stack && {
            controllingPlayer: stack.controllingPlayer,
            stackHeight: stack.stackHeight,
            capHeight: stack.capHeight
          },
        marker: marker && { player: marker.player },
        collapsedOwner
      });
    };

    addPosition('from', move.from);
    addPosition('to', move.to);
    addPosition('captureTarget', move.captureTarget);

    return { positions };
  }

  for (const boardType of boardTypes) {
    for (const numPlayers of playerCounts) {
      const scenarioLabel = `${boardType} with ${numPlayers} AI players`;

      test(`${scenarioLabel}: backend vs sandbox stay in sync on coarse invariants for a sample of seeds`, async () => {
        const boardIndex = boardTypes.indexOf(boardType);
        const playerCountIndex = playerCounts.indexOf(numPlayers);

        for (let run = 0; run < RUNS_PER_SCENARIO; run++) {
          const seed = 1 + run + playerCountIndex * 1000 + boardIndex * 100000;
          const rng = makePrng(seed);

          const backend = createBackendEngine(boardType, numPlayers);
          const sandbox = createSandboxEngine(boardType, numPlayers);

          // Drive both engines forward with paired AI actions.
          for (let step = 0; step < MAX_STEPS; step++) {
            const backendBefore = backend.getGameState();
            const sandboxBefore = sandbox.getGameState();

            const backendSummaryBefore = summariseState(backendBefore);
            const sandboxSummaryBefore = summariseState(sandboxBefore);

            // If both engines consider the game non-active, stop early.
            if (
              backendSummaryBefore.gameStatus !== 'active' &&
              sandboxSummaryBefore.gameStatus !== 'active'
            ) {
              break;
            }

            // Auto-advance the backend through non-interactive phases
            // before asking it for a player move. During line_processing
            // and territory_processing there are no legal moves to choose
            // from; they are internal bookkeeping phases.
            if (
              backendBefore.currentPhase === 'line_processing' ||
              backendBefore.currentPhase === 'territory_processing'
            ) {
              backend.stepAutomaticPhasesForTesting();
              const afterAuto = backend.getGameState();
              if (afterAuto.gameStatus !== 'active') {
                break;
              }
              // Re-evaluate from the new state on the next loop.
              continue;
            }

            // If one engine has structurally terminal board state but still
            // reports gameStatus === 'active', surface this immediately.
            const backendStructTerminal = isStructurallyTerminal(backendBefore);
            const sandboxStructTerminal = isStructurallyTerminal(sandboxBefore);

            if (backendStructTerminal && backendSummaryBefore.gameStatus === 'active') {
              throw new Error(
                `Backend structurally terminal but still active: scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}`
              );
            }

            if (sandboxStructTerminal && sandboxSummaryBefore.gameStatus === 'active') {
              throw new Error(
                `Sandbox structurally terminal but still active: scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}`
              );
            }

            // Backend AI step: choose a single canonical move for this step.
            let sharedMove: Move | null = null;
            if (backendSummaryBefore.gameStatus === 'active') {
              sharedMove = chooseRandomBackendMove(backend, backendBefore, rng);
              if (!sharedMove) {
                throw new Error(
                  `Backend has no legal moves for active game: scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}, ` +
                    `currentPlayer=${backendBefore.currentPlayer}, phase=${backendBefore.currentPhase}`
                );
              }

              const { id, timestamp, moveNumber, ...payload } = sharedMove;
              const result = await backend.makeMove(
                payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
              );
              if (!result.success) {
                throw new Error(
                  `Backend makeMove failed in debug harness: scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}, ` +
                    `error=${result.error}`
                );
              }
            }

            // Sandbox canonical replay step: apply the SAME backend-chosen
            // move into the sandbox using its test-only applyCanonicalMove
            // helper. This bypasses sandbox AI heuristics so that both
            // engines see the identical move sequence.
            if (sandboxSummaryBefore.gameStatus === 'active' && sharedMove) {
              await sandbox.applyCanonicalMove(sharedMove);
            }

            const backendAfter = backend.getGameState();
            const sandboxAfter = sandbox.getGameState();

            const backendSummaryAfter = summariseState(backendAfter);
            const sandboxSummaryAfter = summariseState(sandboxAfter);

            // If one engine has ended but the other remains active in a
            // structurally terminal position, treat this as a divergence.
            const backendStructAfter = isStructurallyTerminal(backendAfter);
            const sandboxStructAfter = isStructurallyTerminal(sandboxAfter);

            if (
              backendSummaryAfter.gameStatus !== sandboxSummaryAfter.gameStatus ||
              backendStructAfter !== sandboxStructAfter
            ) {
              // eslint-disable-next-line no-console
              console.error('[Backend_vs_Sandbox.aiParallelDebug] Divergence detected', {
                scenario: scenarioLabel,
                run,
                seed,
                step,
                sharedMove,
                backendSummaryBefore,
                sandboxSummaryBefore,
                backendSummaryAfter,
                sandboxSummaryAfter,
                backendStructAfter,
                sandboxStructAfter,
                backendLocalContext: getLocalContext(backendAfter, sharedMove),
                sandboxLocalContext: getLocalContext(sandboxAfter, sharedMove)
              });

              throw new Error(
                `Backend vs sandbox divergence at step=${step}, scenario=${scenarioLabel}, run=${run}, seed=${seed}`
              );
            }
          }
        }
      });
    }
  }
});
