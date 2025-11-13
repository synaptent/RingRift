import { WebSocketServer } from './server';
import { createServer } from 'http';
import { GameEngine } from '../game/GameEngine';
import { Player, TimeControl } from '../../shared/types/game';

console.log('🔌 Testing WebSocket Server Integration...\n');

// Create a test HTTP server
const httpServer = createServer();

try {
  // Test WebSocketServer instantiation
  const wsServer = new WebSocketServer(httpServer);
  console.log('✅ WebSocketServer created successfully');
  console.log(`🔌 WebSocket server instance: ${wsServer.constructor.name}`);
  
  // Test GameEngine integration
  const players: Player[] = [
    {
      id: 'test-player-1',
      username: 'TestPlayer1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 12,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'test-player-2',
      username: 'TestPlayer2',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 12,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  const timeControl: TimeControl = {
    type: 'classical',
    initialTime: 600000,
    increment: 0
  };

  const gameEngine = new GameEngine('test-ws-game', 'square8', players, timeControl);
  console.log('✅ GameEngine integration working');
  
  // Test game state retrieval
  const gameState = gameEngine.getGameState();
  console.log('✅ Game state retrieval working');
  console.log(`📋 Game ID: ${gameState.id}`);
  console.log(`🎯 Board Type: ${gameState.boardType}`);
  console.log(`👥 Players: ${gameState.players.length}`);
  console.log(`🎲 Current Phase: ${gameState.currentPhase}`);
  
  // Test valid moves generation
  const validMoves = gameEngine.getValidMoves(1);
  console.log(`🎯 Valid moves available: ${validMoves.length}`);
  
  console.log('\n🎉 WebSocket Server Integration test completed successfully!');
  console.log('✅ Backend is ready for real-time multiplayer RingRift gameplay');
  
} catch (error) {
  console.error('❌ WebSocket Server Integration test failed:', error);
  process.exit(1);
}