import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  RingStack,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString,
} from '../../src/shared/types/game';
import { SandboxOrchestratorAdapter } from '../../src/client/sandbox/SandboxOrchestratorAdapter';

/**
 * Sandbox chain capture parity tests.
 *
 * These tests mirror core chain-capture scenarios from the backend GameEngine
 * tests and run against ClientSandboxEngine with the orchestrator adapter
 * enabled. They verify consistent overtaking behaviour (top-ring-only),
 * marker handling, and mandatory chain continuation via the unified Move model.
 *
 * Tests use applyCanonicalMove() and the orchestrator's getValidMoves() to
 * drive chain-capture sequences, ensuring parity with backend GameEngine
 * behaviour.
 */

describe('ClientSandboxEngine chain capture parity', () => {
  const boardType: BoardType = 'square8';

  function createEngine(numPlayers: number = 3): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array(numPlayers).fill('human') as ('human' | 'ai')[],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        // For capture_direction choices, select a deterministic option whose
        // landingPosition is lexicographically smallest (x, then y). This
        // mirrors the backend GameEngine chain‑capture tests and keeps
        // sandbox parity scenarios reproducible.
        if (anyChoice.type === 'capture_direction') {
          const cdChoice = anyChoice as CaptureDirectionChoice;
          const options = cdChoice.options || [];

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
            choiceId: cdChoice.id,
            playerNumber: cdChoice.playerNumber,
            choiceType: cdChoice.type,
            selectedOption: selected,
          } as PlayerChoiceResponseFor<TChoice>;
        }

        // Fallback for other choice types: pick the first option if present.
        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;

        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
    // Enable the orchestrator adapter for canonical Move-based processing
    engine.enableOrchestratorAdapter();
    return engine;
  }

  function getAdapter(engine: ClientSandboxEngine): SandboxOrchestratorAdapter {
    const engineAny = engine as any;
    return engineAny.getOrchestratorAdapter() as SandboxOrchestratorAdapter;
  }

  function setupBoard(
    engine: ClientSandboxEngine,
    stacks: { pos: Position; player: number; height: number }[],
    currentPlayer: number = 1,
    phase: GameState['currentPhase'] = 'movement'
  ): void {
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState;
    const board = state.board;

    // Clear existing stacks
    board.stacks.clear();

    for (const s of stacks) {
      const rings = Array(s.height).fill(s.player);
      const stack: RingStack = {
        position: s.pos,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: s.player,
      };
      board.stacks.set(positionToString(s.pos), stack);
    }

    state.currentPlayer = currentPlayer;
    state.currentPhase = phase;
    state.gameStatus = 'active';
  }

  /**
   * Resolve any active chain by repeatedly applying continue_capture_segment
   * moves from the orchestrator adapter until no more chain moves are available.
   */
  async function resolveChainIfPresent(engine: ClientSandboxEngine): Promise<void> {
    const adapter = getAdapter(engine);
    const engineAny = engine as any;

    const MAX_STEPS = 16;
    let steps = 0;

    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      steps++;
      if (steps > MAX_STEPS) {
        throw new Error('resolveChainIfPresent: exceeded maximum chain-capture steps');
      }

      const moves = adapter.getValidMoves();
      const chainMoves = moves.filter((m: Move) => m.type === 'continue_capture_segment');

      if (chainMoves.length === 0) {
        // No more chain moves - chain should terminate
        break;
      }

      // Select the first chain move (deterministic)
      const next = chainMoves[0];
      await engine.applyCanonicalMove(next);
    }
  }

  test('two-step chain capture mirrors backend behaviour via orchestrator adapter', async () => {
    // Q10 / Rust test_start_chain_capture + test_complete_chain_capture baseline.
    // Scenario:
    // - Player 1 (Red) at (2,2) height 2
    // - Player 2 (Blue) at (2,3) height 1
    // - Player 3 (Green) at (2,5) height 1
    // Expected:
    // Red captures Blue and then Green in a chain, finishing at (2,7)
    // with height 4, and original target positions are empty.

    const engine = createEngine(3);
    const adapter = getAdapter(engine);

    const redPos: Position = { x: 2, y: 2 };
    const bluePos: Position = { x: 2, y: 3 };
    const greenPos: Position = { x: 2, y: 5 };

    setupBoard(
      engine,
      [
        { pos: redPos, player: 1, height: 2 },
        { pos: bluePos, player: 2, height: 1 },
        { pos: greenPos, player: 3, height: 1 },
      ],
      1,
      'movement'
    );

    // Get valid moves - should include overtaking_capture options
    const moves = adapter.getValidMoves();
    const captureMoves = moves.filter((m: Move) => m.type === 'overtaking_capture');

    // Find the capture that targets Blue at (2,3) with landing at (2,4)
    const initialCapture = captureMoves.find(
      (m: Move) =>
        m.from?.x === 2 &&
        m.from?.y === 2 &&
        m.captureTarget?.x === 2 &&
        m.captureTarget?.y === 3 &&
        m.to?.x === 2 &&
        m.to?.y === 4
    );

    expect(initialCapture).toBeDefined();
    if (!initialCapture) return;

    // Apply the initial capture
    await engine.applyCanonicalMove(initialCapture);

    // Engine should now be in chain_capture phase
    const stateAfterFirst = engine.getGameState();
    expect(stateAfterFirst.currentPhase).toBe('chain_capture');

    // Resolve the rest of the chain
    await resolveChainIfPresent(engine);

    // Verify final state
    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    const stackAtRed = finalBoard.stacks.get('2,2');
    const stackAtBlue = finalBoard.stacks.get('2,3');
    const stackAtGreen = finalBoard.stacks.get('2,5');
    const stackAtFinal = finalBoard.stacks.get('2,7');

    expect(stackAtRed).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtGreen).toBeUndefined();
    expect(stackAtFinal).toBeDefined();
    expect(stackAtFinal!.stackHeight).toBe(4);
    expect(stackAtFinal!.controllingPlayer).toBe(1);

    // Phase should have advanced past chain_capture
    expect(finalState.currentPhase).not.toBe('chain_capture');
  });

  test('orthogonal multi-branch chain capture exposes multiple continue_capture_segment options', async () => {
    // Mirrors the Rust `test_chain_capture_player_choice_simulation` and the
    // backend GameEngine chain‑capture choice integration tests:
    //
    // - Red at (3,3) h2 (attacker)
    // - Blue at (3,4) h1 (initial target)
    // - Green at (4,5) h1
    // - Yellow at (2,5) h1
    //
    // After Red captures Blue and lands at (3,5), there are multiple legal
    // follow‑up capture directions exposed via continue_capture_segment moves.

    const engine = createEngine(4);
    const adapter = getAdapter(engine);

    const redPos: Position = { x: 3, y: 3 };
    const bluePos: Position = { x: 3, y: 4 };
    const greenPos: Position = { x: 4, y: 5 };
    const yellowPos: Position = { x: 2, y: 5 };

    setupBoard(
      engine,
      [
        { pos: redPos, player: 1, height: 2 },
        { pos: bluePos, player: 2, height: 1 },
        { pos: greenPos, player: 3, height: 1 },
        { pos: yellowPos, player: 4, height: 1 },
      ],
      1,
      'movement'
    );

    // Get the initial capture move: Red at (3,3) captures Blue at (3,4), lands at (3,5)
    const moves = adapter.getValidMoves();
    const captureMoves = moves.filter((m: Move) => m.type === 'overtaking_capture');

    const initialCapture = captureMoves.find(
      (m: Move) =>
        m.from?.x === 3 &&
        m.from?.y === 3 &&
        m.captureTarget?.x === 3 &&
        m.captureTarget?.y === 4 &&
        m.to?.x === 3 &&
        m.to?.y === 5
    );

    expect(initialCapture).toBeDefined();
    if (!initialCapture) return;

    // Apply the initial capture
    await engine.applyCanonicalMove(initialCapture);

    // Should be in chain_capture phase
    const stateAfterFirst = engine.getGameState();
    expect(stateAfterFirst.currentPhase).toBe('chain_capture');

    // Get chain continuation moves
    const chainMoves = adapter.getValidMoves();
    const continuations = chainMoves.filter((m: Move) => m.type === 'continue_capture_segment');

    // Should have multiple continuation options (towards Green and Yellow)
    expect(continuations.length).toBeGreaterThan(1);

    // Verify the expected target options exist
    const targetKeys = continuations.map(
      (m: Move) => `${m.captureTarget?.x},${m.captureTarget?.y}`
    );

    // Should be able to capture Green at (4,5) or Yellow at (2,5)
    expect(targetKeys).toContain('4,5');
    expect(targetKeys).toContain('2,5');

    // Pick lexicographically smallest landing (deterministic selection)
    const selected = continuations.reduce((prev: Move, cur: Move) =>
      (cur.to?.x ?? 0) < (prev.to?.x ?? 0) ||
      ((cur.to?.x ?? 0) === (prev.to?.x ?? 0) && (cur.to?.y ?? 0) < (prev.to?.y ?? 0))
        ? cur
        : prev
    );

    await engine.applyCanonicalMove(selected);

    // Resolve any remaining chain
    await resolveChainIfPresent(engine);

    // Verify final state
    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    // Original attacker and first target should be gone
    expect(finalBoard.stacks.get('3,3')).toBeUndefined();
    expect(finalBoard.stacks.get('3,4')).toBeUndefined();
    expect(finalBoard.stacks.get('3,5')).toBeUndefined();

    // There should be exactly one Red-controlled stack on the board
    const redStacks = Array.from(finalBoard.stacks.values()).filter(
      (s) => s.controllingPlayer === 1
    );
    expect(redStacks.length).toBe(1);
    expect(redStacks[0].stackHeight).toBeGreaterThanOrEqual(3);

    // Phase should have advanced past chain_capture
    expect(finalState.currentPhase).not.toBe('chain_capture');
  });

  test('chain capture with no follow-up targets terminates correctly', async () => {
    // Single capture with no further targets - should complete immediately
    //
    // - Red at (2,2) h2
    // - Blue at (2,3) h1
    // - No other stacks in capture range

    const engine = createEngine(2);
    const adapter = getAdapter(engine);

    setupBoard(
      engine,
      [
        { pos: { x: 2, y: 2 }, player: 1, height: 2 },
        { pos: { x: 2, y: 3 }, player: 2, height: 1 },
      ],
      1,
      'movement'
    );

    const moves = adapter.getValidMoves();
    const captureMoves = moves.filter((m: Move) => m.type === 'overtaking_capture');

    const capture = captureMoves.find(
      (m: Move) =>
        m.from?.x === 2 &&
        m.from?.y === 2 &&
        m.captureTarget?.x === 2 &&
        m.captureTarget?.y === 3
    );

    expect(capture).toBeDefined();
    if (!capture) return;

    await engine.applyCanonicalMove(capture);

    // Resolve chain if any (should be none)
    await resolveChainIfPresent(engine);

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    // Only one stack should remain - the capturing stack
    expect(finalBoard.stacks.size).toBe(1);
    const redStack = Array.from(finalBoard.stacks.values())[0];
    expect(redStack.controllingPlayer).toBe(1);
    expect(redStack.stackHeight).toBe(3); // 2 original + 1 captured

    // Phase should not be chain_capture
    expect(finalState.currentPhase).not.toBe('chain_capture');
  });
});
