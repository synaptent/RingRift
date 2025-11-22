import { GameEngine } from './GameEngine';
import { RulesBackendFacade } from './RulesBackendFacade';
import { PythonRulesClient } from '../services/PythonRulesClient';
import { logger } from '../utils/logger';

async function runTest() {
  logger.info('Starting Python Rules Integration Test');

  // 1. Setup
  const boardType = 'square8';
  const players = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human' as const,
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human' as const,
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
  const timeControl = { type: 'blitz' as const, initialTime: 600, increment: 0 };

  const gameEngine = new GameEngine('test-game-id', boardType, players, timeControl);
  // Mock Python client to avoid connection errors during test
  const pythonClient = new PythonRulesClient();
  // @ts-expect-error: mock evaluateMove signature for integration test harness
  pythonClient.evaluateMove = async (state, move) => {
    logger.info('MOCK Python evaluateMove called');
    return {
      valid: true,
      nextState: state, // In a real scenario, this would be the updated state
      stateHash: 'mock-hash',
      sInvariant: 0,
      gameStatus: 'active',
    };
  };

  const rulesFacade = new RulesBackendFacade(gameEngine, pythonClient);

  // 2. Start Game
  gameEngine.startGame();
  logger.info('Game started');

  // 3. Test Move: Place Ring
  // We'll try to place a ring for Player 1.
  // In 'python' mode, this should go to Python for validation.
  const move = {
    type: 'place_ring' as const,
    player: 1,
    to: { x: 3, y: 3 },
    placementCount: 1,
    thinkTime: 0,
  };

  logger.info('Attempting move:', move);

  try {
    // We need to set the env var for this process to simulate the mode
    process.env.RINGRIFT_RULES_MODE = 'python';

    const result = await rulesFacade.applyMove(move);

    if (result.success) {
      logger.info('Move successful!', {
        gameStateHash: result.gameState ? 'present' : 'missing',
        moveHistoryLength: result.gameState?.moveHistory.length,
      });
    } else {
      logger.error('Move failed:', result.error);
    }
  } catch (error) {
    logger.error('Test failed with exception:', error);
  }
}

runTest().catch(console.error);
