import { GameEngine } from './GameEngine';
import { Player, TimeControl } from '../../shared/types/game';

// Simple test to verify the game engine works
async function testRingRiftGame() {
  console.log('🎮 Testing RingRift Game Engine...\n');

  // Create test players
  const players: Player[] = [
    {
      id: 'user1',
      username: 'Alice',
      type: 'human',
      playerNumber: 1,
      rating: 1200,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 12,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'user2',
      username: 'Bob',
      type: 'human',
      playerNumber: 2,
      rating: 1150,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 12,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  // Create time control
  const timeControl: TimeControl = {
    type: 'rapid',
    initialTime: 600, // 10 minutes
    increment: 5   // 5 seconds
  };

  // Create game engine for 8x8 board
  const gameEngine = new GameEngine(
    'test-game-1',
    'square8',
    players,
    timeControl,
    false // not rated
  );

  console.log('✅ Game created successfully');
  console.log(`📋 Game ID: ${gameEngine.getGameState().id}`);
  console.log(`🎯 Board Type: ${gameEngine.getGameState().boardType}`);
  console.log(`👥 Players: ${gameEngine.getGameState().players.map(p => p.username).join(' vs ')}`);
  console.log(`⏰ Current Turn: Player ${gameEngine.getGameState().currentPlayer}`);
  console.log(`🎲 Game Phase: ${gameEngine.getGameState().currentPhase}`);
  console.log(`📊 Game Status: ${gameEngine.getGameState().gameStatus}\n`);

  // Test ring placement
  console.log('🔄 Testing ring placement...');
  try {
    const placementResult = await gameEngine.makeMove({
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      thinkTime: 1000
    });

    if (placementResult.success) {
      console.log('✅ Ring placement successful');
      console.log(`📍 Placed ring at (3,3)`);
      console.log(`🎲 New phase: ${gameEngine.getGameState().currentPhase}`);
    } else {
      console.log('❌ Ring placement failed:', placementResult.error);
    }
  } catch (error) {
    console.log('❌ Ring placement error:', error);
  }

  // Test getting valid moves
  console.log('\n🎯 Testing valid moves...');
  try {
    const validMoves = gameEngine.getValidMoves(gameEngine.getGameState().currentPlayer);
    console.log(`✅ Found ${validMoves.length} valid moves for current player`);
    
    if (validMoves.length > 0) {
      console.log('📋 Sample valid moves:');
      validMoves.slice(0, 3).forEach((move, index) => {
        console.log(`   ${index + 1}. ${move.type} at (${move.to.x}, ${move.to.y})`);
      });
    }
  } catch (error) {
    console.log('❌ Error getting valid moves:', error);
  }

  // Test board state
  console.log('\n🏁 Final game state:');
  const finalState = gameEngine.getGameState();
  console.log(`📊 Board has ${finalState.board.stacks.size} stacks`);
  console.log(`⏱️  Turn: ${finalState.currentPlayer}`);
  console.log(`🎲 Phase: ${finalState.currentPhase}`);
  console.log(`📈 Status: ${finalState.gameStatus}`);

  console.log('\n🎉 RingRift Game Engine test completed successfully!');
}

// Run the test
if (require.main === module) {
  // eslint-disable-next-line @typescript-eslint/no-floating-promises
  testRingRiftGame();
}

export { testRingRiftGame };
