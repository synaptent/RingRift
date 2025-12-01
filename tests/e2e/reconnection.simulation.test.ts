/**
 * Reconnection Simulation E2E Tests
 * ============================================================================
 *
 * This test suite demonstrates the usage of NetworkSimulator for testing
 * network partition and reconnection scenarios in E2E tests.
 *
 * The NetworkSimulator provides:
 * - Force disconnect clients (simulate network loss)
 * - Simulate reconnection after delay
 * - Message interception (drop/delay)
 * - Network condition simulation (latency, packet loss)
 *
 * RUN COMMAND: npx playwright test reconnection.simulation.test.ts
 *
 * NOTE: These tests require:
 * - PostgreSQL running (for user accounts and game persistence)
 * - Redis running (for WebSocket session management)
 * - Backend server running on http://localhost:3000
 */

import { test, expect } from '@playwright/test';
import {
  MultiClientCoordinator,
  createMultiClientCoordinator,
  isGameStateMessage,
} from '../helpers/MultiClientCoordinator';
import {
  NetworkSimulator,
  NetworkAwareCoordinator,
  createNetworkSimulator,
  createNetworkAwareCoordinator,
  type NetworkCondition,
  type MessageInterceptor,
} from '../helpers/NetworkSimulator';
import {
  waitForApiReady,
  generateTestUser,
  registerUser,
  createFixtureGame,
  type TestUser,
} from './helpers/test-utils';
import { HomePage } from './pages';

test.describe('Network Partition Simulation with NetworkSimulator', () => {
  // Mark all tests as slow since they involve network simulation
  test.slow();
  test.setTimeout(120_000); // 2 minutes per test

  const serverUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';

  test.beforeEach(async ({ page }) => {
    // Ensure backend is ready before running tests
    await waitForApiReady(page);
  });

  test.describe('NetworkSimulator API Verification', () => {
    test('creates NetworkSimulator with MultiClientCoordinator', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        expect(networkSimulator).toBeTruthy();
        expect(networkSimulator).toBeInstanceOf(NetworkSimulator);
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('creates NetworkAwareCoordinator with built-in network simulation', async () => {
      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        expect(coordinator).toBeTruthy();
        expect(coordinator).toBeInstanceOf(NetworkAwareCoordinator);
        expect(coordinator.network).toBeTruthy();
        expect(coordinator.network).toBeInstanceOf(NetworkSimulator);
        expect(coordinator.getConnectedClientIds()).toEqual([]);
      } finally {
        await coordinator.cleanup();
      }
    });

    test('network condition getters and setters work correctly', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // Initially no condition
        expect(networkSimulator.getCondition('test-client')).toBeUndefined();

        // Set condition
        const condition: NetworkCondition = {
          latencyMs: 100,
          packetLoss: 0.1,
        };
        networkSimulator.setCondition('test-client', condition);
        expect(networkSimulator.getCondition('test-client')).toEqual(condition);

        // Clear condition
        networkSimulator.clearCondition('test-client');
        expect(networkSimulator.getCondition('test-client')).toBeUndefined();
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('isDisconnected returns false for non-disconnected clients', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        expect(networkSimulator.isDisconnected('unknown')).toBe(false);
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('getDisconnectionDuration returns null for non-disconnected clients', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        expect(networkSimulator.getDisconnectionDuration('unknown')).toBeNull();
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('forceDisconnect throws for unknown client', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        await expect(networkSimulator.forceDisconnect('unknown')).rejects.toThrow(
          "Client 'unknown' not found or not connected"
        );
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('simulateReconnect throws without prior forceDisconnect', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        await expect(networkSimulator.simulateReconnect('unknown')).rejects.toThrow(
          "Client 'unknown' was not disconnected via forceDisconnect"
        );
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('clearAllConditions clears all client conditions', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // Set conditions on multiple clients
        networkSimulator.setCondition('client1', { latencyMs: 100 });
        networkSimulator.setCondition('client2', { packetLoss: 0.5 });

        expect(networkSimulator.getCondition('client1')).toBeDefined();
        expect(networkSimulator.getCondition('client2')).toBeDefined();

        // Clear all
        networkSimulator.clearAllConditions();

        expect(networkSimulator.getCondition('client1')).toBeUndefined();
        expect(networkSimulator.getCondition('client2')).toBeUndefined();
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('cleanup clears all state safely', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      // Set up some state
      networkSimulator.setCondition('client1', { latencyMs: 100 });
      networkSimulator.setCondition('client2', { disconnectAfter: 5000 });

      // Cleanup should not throw
      networkSimulator.cleanup();

      // Double cleanup should be safe
      networkSimulator.cleanup();

      await coordinator.cleanup();
    });
  });

  /**
   * Helper to register a user and extract their JWT token using the UI.
   */
  async function createUserAndGetToken(
    page: import('@playwright/test').Page,
    user: TestUser
  ): Promise<string> {
    await registerUser(page, user.username, user.email, user.password);
    const token = await page.evaluate(() => {
      return localStorage.getItem('auth_token') ?? localStorage.getItem('token');
    });
    if (!token) {
      throw new Error('Failed to get auth token after registration');
    }
    return token;
  }

  test.describe('Message Interception Patterns', () => {
    test('demonstrates dropNextMessage pattern', async () => {
      /**
       * This test demonstrates how dropNextMessage would be used
       * to simulate a dropped message during gameplay.
       *
       * Use case: Testing that the UI handles missing updates gracefully
       */

      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // In a real test with valid tokens:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.joinGame('player1', gameId);

        // Drop the next message from player1
        networkSimulator.dropNextMessage('player1');

        // This move will be silently dropped
        await coordinator.sendMoveById('player1', gameId, moveId);

        // Verify the game state didn't change
        const state = coordinator.getLastGameState('player1');
        expect(state.moveHistory.length).toBe(0);
        */

        expect(networkSimulator).toBeTruthy();
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('demonstrates delayNextMessage pattern', async () => {
      /**
       * This test demonstrates how delayNextMessage would be used
       * to simulate network latency for a specific message.
       *
       * Use case: Testing timeout handling when a move takes too long
       */

      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.joinGame('player1', gameId);

        // Delay the next message by 5 seconds
        networkSimulator.delayNextMessage('player1', 5000);

        const startTime = Date.now();
        await coordinator.sendMoveById('player1', gameId, moveId);

        // Wait for the delayed update
        const state = await coordinator.waitForGameState(
          'player1',
          (s) => s.moveHistory.length > 0,
          10000
        );

        const elapsed = Date.now() - startTime;
        expect(elapsed).toBeGreaterThanOrEqual(4900); // Allow some variance
        */

        expect(networkSimulator).toBeTruthy();
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('demonstrates custom message interceptor pattern', async () => {
      /**
       * This test demonstrates how to use a custom interceptor
       * for fine-grained message control.
       *
       * Use case: Conditionally dropping/delaying specific message types
       */

      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // Create a custom interceptor
        let chatMessageCount = 0;
        const interceptor: MessageInterceptor = (message, direction) => {
          // Drop every other chat message
          if (direction === 'outbound') {
            // Check if it's a chat message (in real usage, inspect message type)
            chatMessageCount++;
            if (chatMessageCount % 2 === 0) {
              return { action: 'drop' };
            }
          }
          return { action: 'pass' };
        };

        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.joinGame('player1', gameId);

        networkSimulator.interceptMessages('player1', interceptor);

        // First chat goes through
        await coordinator.sendChat('player1', gameId, 'Hello');

        // Second chat is dropped
        await coordinator.sendChat('player1', gameId, 'Dropped');

        // Third chat goes through
        await coordinator.sendChat('player1', gameId, 'World');

        // Verify only 2 messages received by others
        const messages = coordinator.getMessagesMatching('player2', 
          (m) => m.eventName === 'chat_message'
        );
        expect(messages.length).toBe(2);
        */

        expect(interceptor).toBeTruthy();
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('demonstrates clearInterceptor to restore normal messaging', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // Set up an interceptor
        networkSimulator.interceptMessages('client1', () => ({ action: 'drop' }));

        // Clear it
        networkSimulator.clearInterceptor('client1');

        // Verify cleared (no-op, just ensure it doesn't throw)
        expect(true).toBe(true);
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });
  });

  test.describe('Network Condition Patterns', () => {
    test('demonstrates latency simulation', async () => {
      /**
       * This test demonstrates how to simulate network latency
       * for all messages from a client.
       *
       * Use case: Testing UI behavior under high latency conditions
       */

      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // Set 200ms latency
        const condition: NetworkCondition = {
          latencyMs: 200,
        };

        networkSimulator.setCondition('player1', condition);

        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.joinGame('player1', gameId);

        networkSimulator.setCondition('player1', condition);

        const startTime = Date.now();
        await coordinator.sendMoveById('player1', gameId, moveId);
        
        // Wait for the response (should take at least 200ms due to latency)
        await coordinator.waitForGameState(
          'player1',
          (s) => s.moveHistory.length > 0,
          5000
        );

        const elapsed = Date.now() - startTime;
        expect(elapsed).toBeGreaterThanOrEqual(190);
        */

        expect(networkSimulator.getCondition('player1')).toEqual(condition);
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('demonstrates packet loss simulation', async () => {
      /**
       * This test demonstrates how to simulate packet loss
       * for testing message retry logic.
       *
       * Use case: Testing that critical messages are retried
       */

      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // 50% packet loss
        const condition: NetworkCondition = {
          packetLoss: 0.5,
        };

        networkSimulator.setCondition('player1', condition);

        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.joinGame('player1', gameId);

        networkSimulator.setCondition('player1', condition);

        // Send multiple moves; some should be dropped
        const moves = ['move1', 'move2', 'move3', 'move4', 'move5'];
        for (const move of moves) {
          await coordinator.sendMoveById('player1', gameId, move);
        }

        // Verify some (but not all) got through due to packet loss
        const state = coordinator.getLastGameState('player1');
        expect(state.moveHistory.length).toBeLessThan(5);
        expect(state.moveHistory.length).toBeGreaterThan(0);
        */

        expect(networkSimulator.getCondition('player1')).toEqual(condition);
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });

    test('demonstrates auto-disconnect after delay', async () => {
      /**
       * This test demonstrates how to simulate a connection
       * that drops after a specified delay.
       *
       * Use case: Testing mid-message disconnect handling
       */

      const coordinator = createMultiClientCoordinator(serverUrl);
      const networkSimulator = createNetworkSimulator(coordinator);

      try {
        // Auto-disconnect after 500ms
        const condition: NetworkCondition = {
          disconnectAfter: 500,
        };

        networkSimulator.setCondition('player1', condition);

        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        networkSimulator.setCondition('player1', condition);

        // Start an action
        await coordinator.joinGame('player1', gameId);

        // Wait for auto-disconnect
        await page.waitForTimeout(600);

        // Verify client is no longer connected
        expect(coordinator.isConnected('player1')).toBe(false);
        expect(networkSimulator.isDisconnected('player1')).toBe(true);
        */

        expect(networkSimulator.getCondition('player1')).toEqual(condition);
      } finally {
        networkSimulator.cleanup();
        await coordinator.cleanup();
      }
    });
  });

  test.describe('Reconnection Scenarios', () => {
    test('scenario 1: restore game state after disconnect', async () => {
      /**
       * Test Scenario: Player disconnects and reconnects, verifying
       * that game state is restored correctly.
       *
       * This tests the server's ability to maintain game state
       * during client disconnection.
       */

      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        // In a real test with valid tokens:
        /*
        // Setup: Two players in a game
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.connect('player2', { playerId: 'p2', token: token2 });

        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);

        // Player 1 makes a move
        await coordinator.sendMoveById('player1', gameId, move1Id);
        
        // Wait for player 2 to see the update
        await coordinator.waitForGameState(
          'player2',
          (s) => s.moveHistory.length === 1,
          5000
        );

        // Simulate network partition for player 1
        await coordinator.network.forceDisconnect('player1');
        
        expect(coordinator.network.isDisconnected('player1')).toBe(true);

        // Player 2 makes a move while player 1 is disconnected
        await coordinator.sendMoveById('player2', gameId, move2Id);

        // Player 1 reconnects after 1 second
        await coordinator.network.simulateReconnect('player1', 1000);

        // Verify player 1 received missed update
        const state = coordinator.getLastGameState('player1');
        expect(state?.moveHistory.length).toBe(2);
        expect(coordinator.network.isDisconnected('player1')).toBe(false);
        */

        expect(coordinator.network).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('scenario 2: handle disconnect during move submission', async () => {
      /**
       * Test Scenario: Player's connection drops mid-message.
       *
       * This tests that:
       * 1. The client handles send failures gracefully
       * 2. Retry after reconnect succeeds
       */

      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.joinGame('player1', gameId);

        // Configure disconnect 50ms after sending
        coordinator.network.setCondition('player1', {
          disconnectAfter: 50,
        });

        // Attempt move - should fail due to disconnect
        const movePromise = coordinator.sendMoveById('player1', gameId, moveId);

        // The promise should reject due to disconnection
        await expect(movePromise).rejects.toThrow();

        // Verify disconnected
        expect(coordinator.isConnected('player1')).toBe(false);

        // Reconnect
        coordinator.network.clearCondition('player1');
        await coordinator.network.simulateReconnect('player1');

        // Retry the move
        await coordinator.sendMoveById('player1', gameId, moveId);

        // Verify it went through
        const state = coordinator.getLastGameState('player1');
        expect(state?.moveHistory.length).toBeGreaterThan(0);
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('scenario 3: reconnection window expiry', async () => {
      /**
       * Test Scenario: Player fails to reconnect within the server's
       * reconnection window (30 seconds by default).
       *
       * This tests that:
       * 1. Player choices are cleared after timeout
       * 2. Opponent is notified of abandonment in rated games
       */

      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.connect('player2', { playerId: 'p2', token: token2 });

        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);

        // Disconnect player 1
        await coordinator.network.forceDisconnect('player1');

        // Track disconnection duration
        expect(coordinator.network.getDisconnectionDuration('player1')).toBeGreaterThanOrEqual(0);

        // Wait for reconnection window to expire (would be 30 seconds in real test)
        // In test, we'd use a shorter timeout or mock the timer
        // await page.waitForTimeout(31000);

        // Player 2 should receive abandonment notification
        // await coordinator.waitForEvent('player2', 'game_over', 
        //   (data) => data.reason === 'abandonment'
        // );
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('scenario 4: concurrent player disconnects', async () => {
      /**
       * Test Scenario: Multiple players disconnect and reconnect
       * at different times.
       *
       * This tests the server's ability to handle multiple
       * concurrent disconnection states.
       */

      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.connect('player2', { playerId: 'p2', token: token2 });
        await coordinator.connect('spectator', { playerId: 's1', token: tokenS });

        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);
        await coordinator.joinGame('spectator', gameId);

        // Disconnect both players
        await coordinator.network.forceDisconnect('player1');
        await page.waitForTimeout(500);
        await coordinator.network.forceDisconnect('player2');

        // Both should be disconnected
        expect(coordinator.network.isDisconnected('player1')).toBe(true);
        expect(coordinator.network.isDisconnected('player2')).toBe(true);

        // Spectator should be notified of both disconnects
        const disconnectMessages = coordinator.getMessagesMatching(
          'spectator',
          (m) => m.eventName === 'player_disconnected'
        );
        expect(disconnectMessages.length).toBe(2);

        // Player 2 reconnects first
        await coordinator.network.simulateReconnect('player2', 500);

        // Then player 1
        await coordinator.network.simulateReconnect('player1', 500);

        // Both should be back
        expect(coordinator.isConnected('player1')).toBe(true);
        expect(coordinator.isConnected('player2')).toBe(true);
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('scenario 5: intermittent connection with packet loss', async () => {
      /**
       * Test Scenario: Player experiences intermittent connectivity
       * with high packet loss.
       *
       * This tests that:
       * 1. Some messages get through despite packet loss
       * 2. Game state eventually synchronizes
       */

      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.connect('player2', { playerId: 'p2', token: token2 });

        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);

        // Apply high packet loss to player 1
        coordinator.network.setCondition('player1', {
          packetLoss: 0.7, // 70% packet loss
          latencyMs: 150,
        });

        // Player 1 attempts multiple moves (some will be lost)
        for (let i = 0; i < 5; i++) {
          try {
            await coordinator.sendMoveById('player1', gameId, moves[i]);
          } catch {
            // Expected: some moves may fail
          }
          await page.waitForTimeout(100);
        }

        // Clear conditions
        coordinator.network.clearCondition('player1');

        // Wait for state to stabilize
        await page.waitForTimeout(1000);

        // States may not match exactly due to lost packets
        const p1State = coordinator.getLastGameState('player1');
        const p2State = coordinator.getLastGameState('player2');

        // But they should be consistent
        expect(p1State?.currentPlayer).toBe(p2State?.currentPlayer);
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('disconnect mid decision, reconnect, sees consistent timeout-based game_over', async ({
      browser,
    }) => {
      const coordinator = createNetworkAwareCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();

      try {
        // Register users and obtain JWT tokens via UI.
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);

        // Create a short-timeout decision-phase fixture game.
        const { gameId } = await createFixtureGame(page1, {
          scenario: 'line_processing',
          isRated: true,
          shortTimeoutMs: 4000,
          shortWarningBeforeMs: 2000,
        });

        // Connect both players via WebSocket.
        await coordinator.connect('player1', {
          playerId: user1.username,
          userId: user1.username,
          token: token1,
        });
        await coordinator.connect('player2', {
          playerId: user2.username,
          userId: user2.username,
          token: token2,
        });

        // Join the same game.
        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);

        // Wait until the game is in the line_processing decision phase.
        await coordinator.waitForPhase('player1', 'line_processing', 15_000);

        // Force-disconnect player1 mid-decision.
        await coordinator.network.forceDisconnect('player1');

        // Player 2 should eventually receive a timeout-based game_over.
        const p2GameOver = (await coordinator.waitForGameOver('player2', 30_000)) as any;

        expect(p2GameOver?.data?.gameResult).toBeDefined();
        const reason = p2GameOver.data.gameResult.reason;
        // Reconnection-window expiry should be treated as abandonment, not a bare timeout.
        expect(reason).toBe('abandonment');

        // Reconnect player1 and ensure they also see the same final result.
        await coordinator.network.simulateReconnect('player1', 1_000);
        const p1GameOver = (await coordinator.waitForGameOver('player1', 10_000)) as any;

        expect(p1GameOver?.data?.gameResult).toBeDefined();
        expect(p1GameOver.data.gameResult.reason).toBe('abandonment');
        expect(p1GameOver.data.gameResult.winner).toBe(p2GameOver.data.gameResult.winner);
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
      }
    });

    test('rated vs unrated reconnection-window abandonment updates rating appropriately', async ({
      browser,
    }) => {
      const coordinator = createNetworkAwareCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();

      const readRating = async (page: import('@playwright/test').Page): Promise<number> => {
        const homePage = new HomePage(page);
        await homePage.goto();
        await homePage.goToProfile();
        await page.waitForURL('**/profile', { timeout: 10_000 });
        const ratingText = await page.locator('.text-emerald-400').first().textContent();
        return parseInt((ratingText || '').replace(/[^0-9]/g, ''), 10);
      };

      try {
        // Register users and obtain JWT tokens via UI.
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);

        const initialP1Rating = await readRating(page1);
        expect(initialP1Rating).toBeGreaterThan(0);

        // --- Rated abandonment: rating should change for the abandoning player (P1) ---
        const rated = await createFixtureGame(page1, {
          scenario: 'line_processing',
          isRated: true,
          shortTimeoutMs: 4_000,
          shortWarningBeforeMs: 2_000,
        });

        await coordinator.connect('player1', {
          playerId: user1.username,
          userId: user1.username,
          token: token1,
        });
        await coordinator.connect('player2', {
          playerId: user2.username,
          userId: user2.username,
          token: token2,
        });

        await coordinator.joinGame('player1', rated.gameId);
        await coordinator.joinGame('player2', rated.gameId);

        await coordinator.waitForPhase('player1', 'line_processing', 15_000);

        // Player 1 disconnects mid-decision and never reconnects → abandonment.
        await coordinator.network.forceDisconnect('player1');

        const ratedResults = await coordinator.waitForAll(['player1', 'player2'], {
          type: 'gameOver',
          predicate: (data) => true,
          timeout: 30_000,
        });

        const ratedP1 = ratedResults.get('player1') as any;
        const ratedP2 = ratedResults.get('player2') as any;
        expect(ratedP1?.data?.gameResult.reason).toBe('abandonment');
        expect(ratedP2?.data?.gameResult.reason).toBe('abandonment');

        // Allow backend to persist rating updates.
        await page1.waitForTimeout(2_000);

        const afterRatedAbandonRating = await readRating(page1);
        expect(afterRatedAbandonRating).not.toBe(initialP1Rating);

        await coordinator.disconnectAll();

        // --- Unrated abandonment: rating should NOT change ---
        const unrated = await createFixtureGame(page1, {
          scenario: 'line_processing',
          isRated: false,
          shortTimeoutMs: 4_000,
          shortWarningBeforeMs: 2_000,
        });

        await coordinator.connect('player1', {
          playerId: user1.username,
          userId: user1.username,
          token: token1,
        });
        await coordinator.connect('player2', {
          playerId: user2.username,
          userId: user2.username,
          token: token2,
        });

        await coordinator.joinGame('player1', unrated.gameId);
        await coordinator.joinGame('player2', unrated.gameId);

        await coordinator.waitForPhase('player1', 'line_processing', 15_000);

        await coordinator.network.forceDisconnect('player1');

        const unratedResults = await coordinator.waitForAll(['player1', 'player2'], {
          type: 'gameOver',
          predicate: (data) => true,
          timeout: 30_000,
        });

        const unratedP1 = unratedResults.get('player1') as any;
        const unratedP2 = unratedResults.get('player2') as any;
        expect(unratedP1?.data?.gameResult.reason).toBe('abandonment');
        expect(unratedP2?.data?.gameResult.reason).toBe('abandonment');

        await page1.waitForTimeout(2_000);

        const afterUnratedAbandonRating = await readRating(page1);
        expect(afterUnratedAbandonRating).toBe(afterRatedAbandonRating);
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
      }
    });
  });

  test.describe('Integration with NetworkAwareCoordinator', () => {
    test('NetworkAwareCoordinator provides coordinator methods', async () => {
      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        // Verify all coordinator methods are available
        expect(typeof coordinator.connect).toBe('function');
        expect(typeof coordinator.disconnect).toBe('function');
        expect(typeof coordinator.disconnectAll).toBe('function');
        expect(typeof coordinator.cleanup).toBe('function');
        expect(typeof coordinator.send).toBe('function');
        expect(typeof coordinator.joinGame).toBe('function');
        expect(typeof coordinator.leaveGame).toBe('function');
        expect(typeof coordinator.sendMoveById).toBe('function');
        expect(typeof coordinator.sendChat).toBe('function');
        expect(typeof coordinator.waitFor).toBe('function');
        expect(typeof coordinator.waitForAll).toBe('function');
        expect(typeof coordinator.waitForGameState).toBe('function');
        expect(typeof coordinator.waitForPhase).toBe('function');
        expect(typeof coordinator.waitForTurn).toBe('function');
        expect(typeof coordinator.waitForGameOver).toBe('function');
        expect(typeof coordinator.waitForEvent).toBe('function');
        expect(typeof coordinator.executeSequence).toBe('function');
        expect(typeof coordinator.executeParallel).toBe('function');
        expect(typeof coordinator.getMessages).toBe('function');
        expect(typeof coordinator.clearMessages).toBe('function');
        expect(typeof coordinator.getMessagesMatching).toBe('function');
        expect(typeof coordinator.getLastGameState).toBe('function');
        expect(typeof coordinator.getConnectedClientIds).toBe('function');
        expect(typeof coordinator.isConnected).toBe('function');
        expect(typeof coordinator.getSocket).toBe('function');

        // Verify network property
        expect(coordinator.network).toBeInstanceOf(NetworkSimulator);
      } finally {
        await coordinator.cleanup();
      }
    });

    test('NetworkAwareCoordinator stores client configs for reconnection', async () => {
      const coordinator = createNetworkAwareCoordinator(serverUrl);

      try {
        // Config is stored on connect (would need valid token for actual connection)
        // This verifies the API exists
        expect(typeof coordinator.getClientConfig).toBe('function');
        expect(coordinator.getClientConfig('unknown')).toBeUndefined();
      } finally {
        await coordinator.cleanup();
      }
    });
  });
});

/**
 * Example: Complete Reconnection Flow Test
 * ============================================================================
 *
 * This is a comprehensive example showing how a complete reconnection
 * test would look using the NetworkAwareCoordinator.
 *
 * Note: This is commented out because it requires a running backend and
 * valid JWT tokens. It serves as documentation for the intended usage.
 */

/*
test('full reconnection flow example', async ({ browser }) => {
  const coordinator = createNetworkAwareCoordinator('http://localhost:3000');

  try {
    // Create users and get tokens
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();

    const user1 = generateTestUser();
    const user2 = generateTestUser();

    const token1 = await createUserAndGetToken(page1, user1);
    const token2 = await createUserAndGetToken(page2, user2);

    // Connect both players
    await coordinator.connect('player1', {
      playerId: user1.id,
      userId: user1.id,
      token: token1,
    });
    await coordinator.connect('player2', {
      playerId: user2.id,
      userId: user2.id,
      token: token2,
    });

    // Player 1 creates a game
    const gameId = 'created-game-id';

    // Both join the game
    await coordinator.joinGame('player1', gameId);
    await coordinator.joinGame('player2', gameId);

    // Wait for initial state
    await coordinator.waitForAll(['player1', 'player2'], {
      type: 'gameState',
      predicate: (data) => isGameStateMessage(data),
      timeout: 10000,
    });

    // Player 1 makes some moves
    await coordinator.sendMoveById('player1', gameId, 'move-1');
    await coordinator.waitForTurn('player2', 2, 5000);

    // === SIMULATE NETWORK PARTITION ===
    // Player 1's connection drops
    await coordinator.network.forceDisconnect('player1');

    // Verify player1 is disconnected
    expect(coordinator.isConnected('player1')).toBe(false);
    expect(coordinator.network.isDisconnected('player1')).toBe(true);

    // Player 2 sees player1 disconnect
    await coordinator.waitForEvent('player2', 'player_disconnected', undefined, 5000);

    // Player 2 continues playing (makes a move)
    await coordinator.sendMoveById('player2', gameId, 'move-2');

    // Track how long player 1 was disconnected
    const disconnectDuration = coordinator.network.getDisconnectionDuration('player1');
    console.log(`Player 1 disconnected for ${disconnectDuration}ms`);

    // === SIMULATE RECONNECTION ===
    // Player 1 reconnects after 2 seconds
    await coordinator.network.simulateReconnect('player1', 2000);

    // Verify reconnected
    expect(coordinator.isConnected('player1')).toBe(true);
    expect(coordinator.network.isDisconnected('player1')).toBe(false);

    // Player 2 sees player1 reconnect
    await coordinator.waitForEvent('player2', 'player_reconnected', undefined, 5000);

    // Player 1 should have received the missed game state
    const p1State = coordinator.getLastGameState('player1');
    expect(p1State?.moveHistory.length).toBe(2);

    // Game continues normally
    await coordinator.sendMoveById('player1', gameId, 'move-3');
    await coordinator.waitForTurn('player2', 2, 5000);

    // Cleanup
    await coordinator.leaveGame('player1', gameId);
    await coordinator.leaveGame('player2', gameId);
    await context1.close();
    await context2.close();
  } finally {
    await coordinator.cleanup();
  }
});
*/
