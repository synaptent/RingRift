import { GameEngine } from '../../src/server/game/GameEngine';
import { GameState, Player, TimeControl, BOARD_CONFIGS } from '../../src/shared/types/game';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';

// Mock the AI service client to simulate downtime/failure
jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: () => ({
    getAIMove: jest.fn().mockRejectedValue(new Error('Service unavailable')),
    getLineRewardChoice: jest.fn().mockRejectedValue(new Error('Service unavailable')),
    getRingEliminationChoice: jest.fn().mockRejectedValue(new Error('Service unavailable')),
    getRegionOrderChoice: jest.fn().mockRejectedValue(new Error('Service unavailable')),
  }),
}));

describe('Full Game Flow Integration (AI Fallback)', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const MAX_MOVES = 500;

  it('completes a full game using local AI fallback when service is down', async () => {
    // Setup 2 AI players
    const players: Player[] = [
      {
        id: 'p1',
        username: 'AI-1',
        playerNumber: 1,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'AI-2',
        playerNumber: 2,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    // Configure AI engine for these players
    globalAIEngine.createAI(1, 5);
    globalAIEngine.createAI(2, 5);

    const engine = new GameEngine('integration-test', 'square8', players, timeControl);
    engine.startGame();

    let moves = 0;
    while (engine.getGameState().gameStatus === 'active' && moves < MAX_MOVES) {
      const state = engine.getGameState();

      // If it's an interactive phase, the AI engine should generate a move
      if (
        state.currentPhase === 'ring_placement' ||
        state.currentPhase === 'movement' ||
        state.currentPhase === 'capture'
      ) {
        // Simulate the WebSocketServer's role: ask AI for a move
        const move = await globalAIEngine.getAIMove(state.currentPlayer, state);

        if (move) {
          const result = await engine.makeMove(move);
          if (!result.success) {
            console.error('Move failed:', result, move);
          }
          expect(result.success).toBe(true);
        } else {
          // If no move returned (e.g. no valid moves), the engine might be stuck
          // or waiting for a timeout. In this test, we expect valid moves.
          // However, if the game is actually over but status hasn't updated yet, break.
          break;
        }
      } else {
        // Automatic phases
        engine.stepAutomaticPhasesForTesting();
      }

      moves++;
    }

    const finalState = engine.getGameState();
    console.log(`Game ended after ${moves} moves. Status: ${finalState.gameStatus}`);

    // Assert game finished naturally
    expect(finalState.gameStatus).not.toBe('active');
    expect(['completed', 'finished']).toContain(finalState.gameStatus);
  }, 30000); // Increase timeout for full game simulation
});
