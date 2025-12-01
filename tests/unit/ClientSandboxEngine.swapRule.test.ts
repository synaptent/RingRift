import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Move,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  BOARD_CONFIGS,
} from '../../src/shared/types/game';

/**
 * Sandbox swap rule (pie rule) tests.
 *
 * Verifies that the client-side engine correctly offers and applies the
 * swap_sides meta-move for 2-player games, mirroring backend semantics.
 */
describe('ClientSandboxEngine swap rule (pie rule)', () => {
  const boardType: BoardType = 'square8';

  function createEngine(numPlayers: number = 2): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array(numPlayers).fill('human'),
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption: (choice as any).options[0],
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('offers swap_sides exactly once after P1 first turn in active 2p games', async () => {
    const engine = createEngine(2);
    const engineAny = engine as any;

    // Manually construct the state to simulate P1 having moved.
    // This avoids fragility with the full turn engine in unit tests.
    const state: GameState = engine.getGameState();

    // 1. P1 places a ring.
    const placeMove: Move = {
      id: 'place-1',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      placementCount: 1,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };
    state.moveHistory.push(placeMove);

    // 2. P1 moves the ring.
    const moveMove: Move = {
      id: 'move-1',
      type: 'move_stack',
      player: 1,
      from: { x: 3, y: 3 },
      to: { x: 3, y: 4 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 2,
    };
    state.moveHistory.push(moveMove);

    // 3. Set current player to P2.
    state.currentPlayer = 2;
    state.currentPhase = 'ring_placement'; // P2 starts with placement

    // Inject state back into engine (using internal setter if available, or just modifying the reference if it's shared)
    // ClientSandboxEngine.gameState is private, but we can use the test-only setGameState hook if exposed,
    // or rely on the fact that getGameState returns a clone but we can't set it back easily without a setter.
    // However, createEngine initializes it.
    // Let's use the `initFromSerializedState` or similar if available, or just cast to any.
    engineAny.gameState = state;

    // Now check swap availability.
    expect(engine.canCurrentPlayerSwapSides()).toBe(true);

    // Apply swap.
    const applied = engine.applySwapSidesForCurrentPlayer();
    expect(applied).toBe(true);

    // Verify state after swap.
    const after = engine.getGameState();
    expect(after.currentPlayer).toBe(2); // Still P2's turn

    // Verify swap move in history.
    const lastMove = after.moveHistory[after.moveHistory.length - 1];
    expect(lastMove.type).toBe('swap_sides');
    expect(lastMove.player).toBe(2);

    // Swap should no longer be available.
    expect(engine.canCurrentPlayerSwapSides()).toBe(false);
  });

  test('does not offer swap_sides in 3p games', async () => {
    const engine = createEngine(3);
    const engineAny = engine as any;

    const state: GameState = engine.getGameState();

    // Simulate P1 moves
    state.moveHistory.push({
      id: 'place-1',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      placementCount: 1,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move);

    state.currentPlayer = 2;
    engineAny.gameState = state;

    expect(engine.canCurrentPlayerSwapSides()).toBe(false);
  });
});
