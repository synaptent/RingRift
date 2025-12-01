import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoiceResponseFor,
  PlayerChoice,
  Position,
} from '../../src/shared/types/game';

class NoopSandboxInteractionHandler implements SandboxInteractionHandler {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const anyChoice = choice as any;
    const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
    return {
      choiceId: anyChoice.id,
      playerNumber: anyChoice.playerNumber,
      choiceType: anyChoice.type,
      selectedOption,
    } as PlayerChoiceResponseFor<TChoice>;
  }
}

describe('ClientSandboxEngine pie rule (swap_sides meta-move)', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): {
    engine: ClientSandboxEngine;
    state: GameState;
  } {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler: new NoopSandboxInteractionHandler(),
    });
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // Mark game as active for testing pie-rule semantics.
    state.gameStatus = 'active';

    return { engine, state };
  }

  it('does not allow swap_sides before Player 1 has completed a turn', () => {
    const { engine } = createEngine();

    expect(engine.canCurrentPlayerSwapSides()).toBe(false);
  });

  it('allows and applies swap_sides exactly once after Player 1 first turn', () => {
    const { engine, state } = createEngine();
    const engineAny: any = engine;

    // Simulate Player 1 completing a first turn by pushing a dummy move
    // into moveHistory and rotating to Player 2.
    const dummyMove: any = {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 } as Position,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };
    state.moveHistory.push(dummyMove);
    state.currentPlayer = 2;

    // Give distinct usernames/ids so we can verify the swap.
    state.players[0].username = 'Alice';
    state.players[1].username = 'Bob';
    state.players[0].id = 'user-alice';
    state.players[1].id = 'user-bob';

    expect(engine.canCurrentPlayerSwapSides()).toBe(true);

    const applied = engine.applySwapSidesForCurrentPlayer();
    expect(applied).toBe(true);

    const after = engine.getGameState();
    expect(after.currentPlayer).toBe(2);

    const p1After = after.players.find((p) => p.playerNumber === 1)!;
    const p2After = after.players.find((p) => p.playerNumber === 2)!;

    // Identities/usernames should have swapped seats.
    expect(p1After.username).toBe('Bob');
    expect(p1After.id).toBe('user-bob');
    expect(p2After.username).toBe('Alice');
    expect(p2After.id).toBe('user-alice');

    // History should contain a swap_sides entry at the end.
    const lastMove = after.moveHistory[after.moveHistory.length - 1];
    expect(lastMove.type).toBe('swap_sides');
    expect(lastMove.player).toBe(2);

    // Pie rule must not be offered again.
    expect(engine.canCurrentPlayerSwapSides()).toBe(false);
  });
});

