import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameResult,
  GameState,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import { addStack, pos } from '../utils/fixtures';

/**
 * ClientSandboxEngine victory-condition tests.
 *
 * These mirror the ring-elimination and territory-control checks
 * performed by GameEngine but exercise the client-local sandbox
 * harness instead.
 */

describe('ClientSandboxEngine victory conditions (square19)', () => {
  const boardType: BoardType = 'square19';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('ring-elimination victory when a player reaches victoryThreshold', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Sanity: victoryResult starts null and game is active.
    expect(engine.getVictoryResult()).toBeNull();
    expect(state.gameStatus).toBe('active');

    // Force player 1 to be just below their ring-elimination threshold,
    // then apply one more elimination via totalRingsEliminated.
    const p1 = state.players.find((p) => p.playerNumber === 1)!;
    const threshold = state.victoryThreshold;
    p1.eliminatedRings = threshold - 1;

    // Simulate an additional elimination attributed to player 1.
    state.board.eliminatedRings[1] = threshold;
    p1.eliminatedRings = threshold;

    // Run the victory check directly.
    engineAny.checkAndApplyVictory();

    const result: GameResult | null = engine.getVictoryResult();
    const finalState = engine.getGameState();

    expect(result).not.toBeNull();
    expect(result!.reason).toBe('ring_elimination');
    expect(result!.winner).toBe(1);
    expect(finalState.gameStatus).toBe('completed');
    expect(finalState.winner).toBe(1);
  });

  test('territory-control victory when a player reaches territoryVictoryThreshold', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Reset any existing victory state.
    engineAny.victoryResult = null;
    state.gameStatus = 'active';
    state.winner = undefined as any;

    const p1 = state.players.find((p) => p.playerNumber === 1)!;
    const territoryThreshold = state.territoryVictoryThreshold;

    // Directly set player 1's territory to threshold and re-run checks.
    p1.territorySpaces = territoryThreshold;

    engineAny.checkAndApplyVictory();

    const result: GameResult | null = engine.getVictoryResult();
    const finalState = engine.getGameState();

    expect(result).not.toBeNull();
    expect(result!.reason).toBe('territory_control');
    expect(result!.winner).toBe(1);
    expect(finalState.gameStatus).toBe('completed');
    expect(finalState.winner).toBe(1);
  });

  test('stalemate tiebreaker uses markers before last-actor in sandbox', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Structural terminality: no stacks, no rings in hand.
    state.board.stacks.clear();
    state.board.markers.clear();

    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 2;
    });

    // Player 1: two markers; Player 2: one marker.
    (state.board.markers as any).set('0,0', {
      player: 1,
      position: { x: 0, y: 0 },
      type: 'regular',
    });
    (state.board.markers as any).set('1,0', {
      player: 1,
      position: { x: 1, y: 0 },
      type: 'regular',
    });
    (state.board.markers as any).set('0,1', {
      player: 2,
      position: { x: 0, y: 1 },
      type: 'regular',
    });

    // Ensure victoryResult is clear.
    engineAny.victoryResult = null;

    engineAny.checkAndApplyVictory();

    const result: GameResult | null = engine.getVictoryResult();
    const finalState = engine.getGameState();

    expect(result).not.toBeNull();
    expect(result!.reason).toBe('last_player_standing');
    expect(result!.winner).toBe(1);
    expect(finalState.gameStatus).toBe('completed');
    expect(finalState.winner).toBe(1);
  });

  test('stalemate final rung uses last-actor in sandbox when fully tied', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Structural terminality: no stacks, no rings in hand, no markers.
    state.board.stacks.clear();
    state.board.markers.clear();

    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    // With players [1,2] and currentPlayer = 1, the sandbox
    // getLastActorFromState fallback will treat Player 2 as the
    // last actor in absence of history.
    state.currentPlayer = 1;

    engineAny.victoryResult = null;

    engineAny.checkAndApplyVictory();

    const result: GameResult | null = engine.getVictoryResult();
    const finalState = engine.getGameState();

    expect(result).not.toBeNull();
    expect(result!.reason).toBe('last_player_standing');
    expect(result!.winner).toBe(2);
    expect(finalState.gameStatus).toBe('completed');
    expect(finalState.winner).toBe(2);
  });
});
