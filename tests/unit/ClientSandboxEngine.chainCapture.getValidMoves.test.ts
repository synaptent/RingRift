import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Move,
  Position,
  RingStack,
  PlayerChoice,
  PlayerChoiceResponseFor,
  positionToString,
} from '../../src/shared/types/game';

/**
 * Targeted parity test for ClientSandboxEngine.getValidMoves during the
 * chain_capture phase when running in legacy mode (orchestrator adapter
 * disabled). This ensures the sandbox host exposes
 * continue_capture_segment candidates derived from the same shared
 * capture aggregate as the backend GameEngine.
 */

describe('ClientSandboxEngine.getValidMoves – chain_capture (legacy sandbox path)', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        // For this test we do not expect interactive PlayerChoices to fire;
        // if they do, pick the first option deterministically.
        const anyChoice = choice as any;
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
    // Now uses orchestrator-backed engine which delegates to shared capture
    // aggregate for chain_capture processing.
    return engine;
  }

  function setupBoard(
    engine: ClientSandboxEngine,
    stacks: { pos: Position; player: number; height: number }[]
  ): void {
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState;
    const board = state.board;

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

    state.currentPlayer = 1;
    state.currentPhase = 'movement';
    state.gameStatus = 'active';
  }

  it('exposes continue_capture_segment moves from current chain position', async () => {
    const engine = createEngine();

    const attacker: Position = { x: 2, y: 2 };
    const firstTarget: Position = { x: 2, y: 3 };
    const secondTarget: Position = { x: 2, y: 5 };

    // Simple vertical chain: attacker at (2,2) height 2, targets at (2,3)
    // and (2,5) height 1 each, with empty landing cells beyond.
    setupBoard(engine, [
      { pos: attacker, player: 1, height: 2 },
      { pos: firstTarget, player: 2, height: 1 },
      { pos: secondTarget, player: 3, height: 1 },
    ]);

    const engineAny = engine as any;

    // Use the same capture-segment enumerator as the sandbox engine to
    // build the initial overtaking_capture Move.
    const segments = (
      engineAny.enumerateCaptureSegmentsFrom as (
        from: Position,
        playerNumber: number
      ) => Array<{ from: Position; target: Position; landing: Position }>
    )(attacker, 1);

    expect(segments.length).toBeGreaterThan(0);

    const initialSeg =
      segments.find(
        (seg) =>
          positionToString(seg.target) === positionToString(firstTarget) &&
          positionToString(seg.landing) === '2,4'
      ) ?? segments[0];

    const initialMove: Move = {
      id: '',
      type: 'overtaking_capture',
      player: 1,
      from: initialSeg.from,
      captureTarget: initialSeg.target,
      to: initialSeg.landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    await engine.applyCanonicalMove(initialMove);

    const after = engine.getGameState();
    expect(after.currentPhase).toBe('chain_capture');
    expect(after.currentPlayer).toBe(1);

    const moves = engine.getValidMoves(1);
    const chainMoves = moves.filter((m) => m.type === 'continue_capture_segment');

    expect(chainMoves.length).toBeGreaterThan(0);

    // All continuation moves should originate from the initial landing position
    for (const m of chainMoves) {
      expect(m.from).toBeDefined();
      expect(m.captureTarget).toBeDefined();
      expect(m.to).toBeDefined();
      expect(positionToString(m.from as Position)).toBe(positionToString(initialSeg.landing));
    }
  });
});
