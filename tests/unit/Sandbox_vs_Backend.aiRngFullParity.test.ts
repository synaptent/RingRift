import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BOARD_CONFIGS,
  GameState,
  Move,
  Player,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import { hashGameState } from '../../src/shared/engine/core';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { LocalAIRng } from '../../src/shared/engine/localAIMoveSelection';
import { AIEngine } from '../../src/server/game/ai/AIEngine';

/**
 * Shared-RNG full AI parity harness (diagnostic / experimental).
 *
 * This suite is intentionally focused and kept *minimal* so it can run as a
 * CI-safe smoke test. Earlier documentation (see
 * tests/TEST_SUITE_PARITY_PLAN.md §2.1, row
 * Sandbox_vs_Backend.aiRngFullParity) described this harness as
 * `describe.skip` / diagnostic-only; that referred to an older, much heavier
 * version. The current implementation runs a short, deterministic sequence
 * over a single curated seed.
 *
 * The existing suites already cover:
 *   - Canonical move + history parity via trace tests;
 *   - Heuristic coverage (sandbox AI moves being legal and not
 *     under-covered) via Sandbox_vs_Backend.aiHeuristicCoverage.test.
 *
 * Here we:
 *   - Use identical board/player configuration on sandbox and backend;
 *   - For a handful of seeds and early steps, drive
 *       - Sandbox via ClientSandboxEngine.maybeRunAITurn(rngSandbox);
 *       - Backend via GameEngine.getValidMoves +
 *         AIEngine.chooseLocalMoveFromCandidates(..., rngBackend);
 *   - Compare the resulting canonical moves using loose-matching semantics
 *     specific to this harness.
 *
 * IMPORTANT: sandbox and backend may consume RNG in different patterns. To
 * avoid unintended coupling from call-count differences, we use two
 * *identically seeded* RNG instances (rngSandbox, rngBackend) rather than a
 * single shared function. This keeps the random streams consistent in
 * distribution without requiring identical call counts.
 *
 * NOTE: Skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because full RNG parity mismatch (intentional divergence)
 */

// Skip this test suite when orchestrator adapter is enabled - full RNG parity mismatch
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Sandbox vs Backend AI full RNG-aligned parity (minimal smoke test)',
  () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;

    // Keep this suite very light: a small number of early steps and a single
    // representative seed, so it can run in CI without adding noticeable cost.
    const MAX_STEPS = 8;
    const SEEDS = [5];

    /** Tiny deterministic PRNG (same LCG as other AI tests). */
    function makePrng(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    function createBackendEngine(bt: BoardType, playersCount: number): GameEngine {
      const timeControl = { initialTime: 600, increment: 0, type: 'blitz' as const };
      const boardConfig = BOARD_CONFIGS[bt];

      const players: Player[] = Array.from({ length: playersCount }, (_, idx) => {
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
        'sandbox-vs-backend-ai-rng-full-parity',
        bt,
        players,
        timeControl,
        false
      );
      engine.enableMoveDrivenDecisionPhases();
      const started = engine.startGame();
      if (!started) {
        throw new Error('Failed to start GameEngine for RNG full-parity harness');
      }
      return engine;
    }

    function createSandboxEngine(bt: BoardType, playersCount: number): ClientSandboxEngine {
      const config: SandboxConfig = {
        boardType: bt,
        numPlayers: playersCount,
        playerKinds: Array.from({ length: playersCount }, () => 'ai'),
      };

      const handler: SandboxInteractionHandler = {
        async requestChoice<TChoice>(choice: TChoice): Promise<any> {
          const anyChoice = choice as any;

          if (anyChoice.type === 'capture_direction') {
            const options = anyChoice.options || [];
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

    interface SummaryLite {
      gameStatus: GameState['gameStatus'];
      currentPlayer: number;
      currentPhase: GameState['currentPhase'];
      stacks: number;
      markers: number;
      collapsed: number;
      totalRingsEliminated: number;
    }

    function summariseStateLite(state: GameState): SummaryLite {
      return {
        gameStatus: state.gameStatus,
        currentPlayer: state.currentPlayer,
        currentPhase: state.currentPhase,
        stacks: state.board.stacks.size,
        markers: state.board.markers.size,
        collapsed: state.board.collapsedSpaces.size,
        totalRingsEliminated: state.totalRingsEliminated ?? 0,
      };
    }

    function statesStructurallyAligned(a: SummaryLite, b: SummaryLite): boolean {
      return (
        a.gameStatus === b.gameStatus &&
        a.currentPlayer === b.currentPlayer &&
        a.currentPhase === b.currentPhase &&
        a.stacks === b.stacks &&
        a.markers === b.markers &&
        a.collapsed === b.collapsed &&
        a.totalRingsEliminated === b.totalRingsEliminated
      );
    }

    function positionsEqual(a?: Position, b?: Position): boolean {
      if (!a && !b) return true;
      if (!a || !b) return false;
      return a.x === b.x && a.y === b.y && (a.z ?? 0) === (b.z ?? 0);
    }

    function movesLooselyMatch(a: Move, b: Move): boolean {
      if (a.player !== b.player) return false;

      const isSimpleMovementPair =
        (a.type === 'move_ring' && b.type === 'move_stack') ||
        (a.type === 'move_stack' && b.type === 'move_ring') ||
        (a.type === 'move_ring' && b.type === 'move_ring') ||
        (a.type === 'move_stack' && b.type === 'move_stack');

      if (isSimpleMovementPair) {
        return positionsEqual(a.from, b.from) && positionsEqual(a.to, b.to);
      }

      if (a.type === 'overtaking_capture' && b.type === 'overtaking_capture') {
        const sameOrigin = positionsEqual(a.from, b.from) && !!a.captureTarget && !!b.captureTarget;
        const sameTarget =
          sameOrigin && positionsEqual(a.captureTarget as Position, b.captureTarget as Position);
        const sameLanding = sameTarget && positionsEqual(a.to as Position, b.to as Position);
        return sameLanding;
      }

      if (a.type !== b.type) return false;

      if (a.type === 'place_ring') {
        return positionsEqual(a.to, b.to);
      }

      if (a.type === 'skip_placement') {
        return true;
      }

      return false;
    }

    async function resolveBackendChainIfPresent(backend: GameEngine): Promise<void> {
      const MAX_CHAIN_STEPS = 32;
      let steps = 0;

      for (;;) {
        const state = backend.getGameState();

        if (state.currentPhase !== 'chain_capture' || state.gameStatus !== 'active') {
          break;
        }

        steps++;
        if (steps > MAX_CHAIN_STEPS) {
          throw new Error('resolveBackendChainIfPresent: exceeded maximum chain-capture steps');
        }

        const currentPlayer = state.currentPlayer;
        const moves = backend.getValidMoves(currentPlayer);
        const chainMoves = moves.filter((m) => m.type === 'continue_capture_segment');

        if (chainMoves.length === 0) {
          break;
        }

        const next = chainMoves.reduce((best, current) => {
          if (!best.to || !current.to) return best;

          const bx = best.to.x;
          const by = best.to.y;
          const bz = best.to.z !== undefined ? best.to.z : 0;
          const cx = current.to.x;
          const cy = current.to.y;
          const cz = current.to.z !== undefined ? current.to.z : 0;

          if (cx < bx) return current;
          if (cx > bx) return best;
          if (cy < by) return current;
          if (cy > by) return best;
          if (cz < bz) return current;
          if (cz > bz) return best;
          return best;
        }, chainMoves[0]);

        const { id, timestamp, moveNumber, ...payload } = next as any;
        const result = await backend.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );

        if (!result.success) {
          throw new Error(
            `resolveBackendChainIfPresent: backend.makeMove failed during chain resolution: ${result.error}`
          );
        }
      }
    }

    function describeMoveForLog(move: Move): string {
      const parts: string[] = [];
      parts.push(`type=${move.type}`);
      parts.push(`player=${move.player}`);
      if (move.from) parts.push(`from=${positionToString(move.from)}`);
      if (move.to) parts.push(`to=${positionToString(move.to)}`);
      return parts.join(',');
    }

    for (const seed of SEEDS) {
      test(`square8 / 2 AI players / seed=${seed}: sandbox vs backend choose RNG-aligned moves on early steps (diagnostic)`, async () => {
        const backend = createBackendEngine(boardType, numPlayers);
        const sandbox = createSandboxEngine(boardType, numPlayers);
        const aiEngine = new AIEngine();

        const rngSandbox: LocalAIRng = makePrng(seed);
        const rngBackend: LocalAIRng = makePrng(seed);

        const originalRandom = Math.random;
        // For the minimal full-parity smoke test we allow other parts of the
        // engine (e.g. UUID generation) to use Math.random. Policy selection
        // itself is already covered by Sandbox_vs_Backend.aiRngParity.test.ts,
        // which enforces that injected RNG is used instead of Math.random.
        try {
          for (let step = 0; step < MAX_STEPS; step++) {
            backend.stepAutomaticPhasesForTesting();
            await resolveBackendChainIfPresent(backend);

            const backendBefore = backend.getGameState();
            const sandboxBefore = sandbox.getGameState();

            if (backendBefore.gameStatus !== 'active' || sandboxBefore.gameStatus !== 'active') {
              break;
            }

            if (backendBefore.currentPlayer !== sandboxBefore.currentPlayer) {
              throw new Error(
                `Pre-step desync in RNG full-parity harness: seed=${seed}, step=${step}, ` +
                  `backendCurrent=${backendBefore.currentPlayer}, sandboxCurrent=${sandboxBefore.currentPlayer}, ` +
                  `backendPhase=${backendBefore.currentPhase}, sandboxPhase=${sandboxBefore.currentPhase}`
              );
            }

            const backendSummaryBefore = summariseStateLite(backendBefore);
            const sandboxSummaryBefore = summariseStateLite(sandboxBefore);

            if (!statesStructurallyAligned(backendSummaryBefore, sandboxSummaryBefore)) {
              break;
            }

            const currentPlayer = sandboxBefore.currentPlayer;
            const backendMoves = backend.getValidMoves(currentPlayer);

            if (backendMoves.length === 0) {
              break;
            }

            const backendPolicyMove = aiEngine.chooseLocalMoveFromCandidates(
              currentPlayer,
              backendBefore,
              backendMoves,
              rngBackend
            );

            if (!backendPolicyMove) {
              throw new Error(
                `Backend local AI produced no move despite non-empty getValidMoves; seed=${seed}, step=${step}, player=${currentPlayer}`
              );
            }

            const sandboxBeforeHash = hashGameState(sandboxBefore);

            await sandbox.maybeRunAITurn(rngSandbox);

            const sandboxAfter = sandbox.getGameState();
            const sandboxAfterHash = hashGameState(sandboxAfter);
            const sandboxMove = sandbox.getLastAIMoveForTesting();

            if (!sandboxMove) {
              if (sandboxBeforeHash === sandboxAfterHash && sandboxAfter.gameStatus === 'active') {
                throw new Error(
                  `Sandbox AI produced no move in RNG full-parity harness; seed=${seed}, step=${step}, player=${currentPlayer}`
                );
              }
              break;
            }

            if (!movesLooselyMatch(sandboxMove, backendPolicyMove)) {
              throw new Error(
                `Sandbox vs backend local AI chose different moves under RNG alignment; ` +
                  `seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                  `sandboxMove=${describeMoveForLog(sandboxMove)}, ` +
                  `backendPolicyMove=${describeMoveForLog(backendPolicyMove)}`
              );
            }

            const { id, timestamp, moveNumber, ...payload } = backendPolicyMove as any;
            const result = await backend.makeMove(
              payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
            );

            if (!result.success) {
              throw new Error(
                `Backend makeMove failed when applying backend policy move in RNG full-parity harness; ` +
                  `seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                  `move=${describeMoveForLog(backendPolicyMove)}, error=${result.error}`
              );
            }
          }
        } finally {
          Math.random = originalRandom;
        }
      });
    }
  }
);
