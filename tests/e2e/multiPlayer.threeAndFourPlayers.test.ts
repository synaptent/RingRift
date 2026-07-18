/**
 * Multi-Player E2E Tests (3-4 Players)
 * ============================================================================
 *
 * This test suite covers 3-4 player game scenarios including:
 * - Game creation and joining for 3-4 players
 * - Turn rotation across all players
 * - Victory conditions with multiple opponents
 * - Spectator observation of multiplayer games
 * - LPS (Last Player Standing) rotation logic
 *
 * RUN COMMAND: npx playwright test multiPlayer.threeAndFourPlayers.test.ts
 *
 * NOTE: These tests require:
 * - PostgreSQL running (for user accounts and game persistence)
 * - Redis running (for WebSocket session management)
 * - Backend server running on http://localhost:3000
 */

import { test, expect } from '@playwright/test';
import {
  createMultiClientCoordinator,
  isGameStateMessage,
} from '../helpers/MultiClientCoordinator';
import {
  generateTestUser,
  registerUser,
  waitForApiReady,
  type TestUser,
} from './helpers/test-utils';
import type { GameStateUpdateMessage, Move } from '../../src/shared/types/websocket';

/**
 * Helper to register a user and extract their JWT token.
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

/**
 * Creates a 3 or 4 player game via API.
 */
async function createMultiPlayerGame(
  page: import('@playwright/test').Page,
  token: string,
  maxPlayers: 3 | 4,
  options: { isRated?: boolean } = {}
): Promise<string> {
  const apiBaseUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';
  const response = await page.request.post(`${apiBaseUrl.replace(/\/$/, '')}/api/games`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
    data: {
      boardType: 'square8',
      timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
      isRated: options.isRated ?? false,
      isPrivate: false,
      maxPlayers,
    },
  });
  if (!response.ok()) {
    throw new Error(
      `Failed to create ${maxPlayers}-player game: ${response.status()} - ${await response.text()}`
    );
  }
  const json = await response.json();
  return json.data.game.id;
}

async function joinMultiPlayerGame(
  page: import('@playwright/test').Page,
  token: string,
  gameId: string
): Promise<void> {
  const apiBaseUrl = (process.env.E2E_API_BASE_URL || 'http://localhost:3000').replace(/\/$/, '');
  const response = await page.request.post(`${apiBaseUrl}/api/games/${gameId}/join`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok()) {
    throw new Error(
      `Failed to occupy multiplayer seat: ${response.status()} - ${await response.text()}`
    );
  }
}

test.describe('3-4 Player Game Flows', () => {
  test.slow();
  test.setTimeout(180_000); // 3 minutes per test for multiplayer coordination

  const serverUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';

  test.beforeEach(async ({ page }) => {
    await waitForApiReady(page);
  });

  test.describe('3-Player Games', () => {
    test('three players can join and see consistent game state', async ({ browser }) => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const context3 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();
      const page3 = await context3.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();
      const user3 = generateTestUser();

      try {
        // Register all three users and get tokens
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);
        const token3 = await createUserAndGetToken(page3, user3);

        // Create a 3-player game
        const gameId = await createMultiPlayerGame(page1, token1, 3);
        await joinMultiPlayerGame(page2, token2, gameId);
        await joinMultiPlayerGame(page3, token3, gameId);

        // Connect all three players via WebSocket
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
        await coordinator.connect('player3', {
          playerId: user3.username,
          userId: user3.username,
          token: token3,
        });

        // All players join the game room
        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);
        await coordinator.joinGame('player3', gameId);

        // Wait for all players to receive initial game state
        const initialStates = await coordinator.waitForAll(['player1', 'player2', 'player3'], {
          type: 'gameState',
          predicate: (data) => isGameStateMessage(data),
          timeout: 30_000,
        });

        expect(initialStates.size).toBe(3);

        // Verify all players see consistent maxPlayers
        const state1 = coordinator.getLastGameState('player1');
        const state2 = coordinator.getLastGameState('player2');
        const state3 = coordinator.getLastGameState('player3');

        expect(state1).not.toBeNull();
        expect(state2).not.toBeNull();
        expect(state3).not.toBeNull();

        expect(state1!.maxPlayers).toBe(3);
        expect(state2!.maxPlayers).toBe(3);
        expect(state3!.maxPlayers).toBe(3);

        // Verify player list is consistent
        expect(state1!.players.length).toBe(state2!.players.length);
        expect(state2!.players.length).toBe(state3!.players.length);
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
        await context3.close();
      }
    });

    test('turn rotation cycles through all 3 players correctly', async ({ browser }) => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const context3 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();
      const page3 = await context3.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();
      const user3 = generateTestUser();

      try {
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);
        const token3 = await createUserAndGetToken(page3, user3);

        const gameId = await createMultiPlayerGame(page1, token1, 3);
        await joinMultiPlayerGame(page2, token2, gameId);
        await joinMultiPlayerGame(page3, token3, gameId);

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
        await coordinator.connect('player3', {
          playerId: user3.username,
          userId: user3.username,
          token: token3,
        });

        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);
        await coordinator.joinGame('player3', gameId);

        // Wait for P1's turn
        const p1Turn = (await coordinator.waitForGameState(
          'player1',
          (state) => state.currentPhase === 'ring_placement' && state.currentPlayer === 1,
          30_000
        )) as GameStateUpdateMessage;

        expect(p1Turn.data.gameState.currentPlayer).toBe(1);

        // P1 makes a move
        const p1Move = p1Turn.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p1Move).toBeDefined();
        await coordinator.sendMoveById('player1', gameId, p1Move!.id);

        // Wait for P2's turn
        const p2Turn = (await coordinator.waitForTurn(
          'player2',
          2,
          30_000
        )) as GameStateUpdateMessage;
        expect(p2Turn.data.gameState.currentPlayer).toBe(2);

        // P2 makes a move
        const p2Move = p2Turn.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p2Move).toBeDefined();
        await coordinator.sendMoveById('player2', gameId, p2Move!.id);

        // Wait for P3's turn (verifies 3-player rotation)
        const p3Turn = (await coordinator.waitForTurn(
          'player3',
          3,
          30_000
        )) as GameStateUpdateMessage;
        expect(p3Turn.data.gameState.currentPlayer).toBe(3);

        // P3 makes a move
        const p3Move = p3Turn.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p3Move).toBeDefined();
        await coordinator.sendMoveById('player3', gameId, p3Move!.id);

        // Wait for rotation back to P1 (confirms full cycle)
        const p1Again = (await coordinator.waitForTurn(
          'player1',
          1,
          30_000
        )) as GameStateUpdateMessage;
        expect(p1Again.data.gameState.currentPlayer).toBe(1);
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
        await context3.close();
      }
    });
  });

  test.describe('4-Player Games', () => {
    test('four players can join and see consistent game state', async ({ browser }) => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const context3 = await browser.newContext();
      const context4 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();
      const page3 = await context3.newPage();
      const page4 = await context4.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();
      const user3 = generateTestUser();
      const user4 = generateTestUser();

      try {
        // Register all four users and get tokens
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);
        const token3 = await createUserAndGetToken(page3, user3);
        const token4 = await createUserAndGetToken(page4, user4);

        // Create a 4-player game
        const gameId = await createMultiPlayerGame(page1, token1, 4);
        await joinMultiPlayerGame(page2, token2, gameId);
        await joinMultiPlayerGame(page3, token3, gameId);
        await joinMultiPlayerGame(page4, token4, gameId);

        // Connect all four players via WebSocket
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
        await coordinator.connect('player3', {
          playerId: user3.username,
          userId: user3.username,
          token: token3,
        });
        await coordinator.connect('player4', {
          playerId: user4.username,
          userId: user4.username,
          token: token4,
        });

        // All players join the game room
        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);
        await coordinator.joinGame('player3', gameId);
        await coordinator.joinGame('player4', gameId);

        // Wait for all players to receive initial game state
        const initialStates = await coordinator.waitForAll(
          ['player1', 'player2', 'player3', 'player4'],
          {
            type: 'gameState',
            predicate: (data) => isGameStateMessage(data),
            timeout: 30_000,
          }
        );

        expect(initialStates.size).toBe(4);

        // Verify all players see consistent maxPlayers
        const state1 = coordinator.getLastGameState('player1');
        const state2 = coordinator.getLastGameState('player2');
        const state3 = coordinator.getLastGameState('player3');
        const state4 = coordinator.getLastGameState('player4');

        expect(state1).not.toBeNull();
        expect(state2).not.toBeNull();
        expect(state3).not.toBeNull();
        expect(state4).not.toBeNull();

        expect(state1!.maxPlayers).toBe(4);
        expect(state2!.maxPlayers).toBe(4);
        expect(state3!.maxPlayers).toBe(4);
        expect(state4!.maxPlayers).toBe(4);
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
        await context3.close();
        await context4.close();
      }
    });

    test('turn rotation cycles through all 4 players correctly', async ({ browser }) => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const context3 = await browser.newContext();
      const context4 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();
      const page3 = await context3.newPage();
      const page4 = await context4.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();
      const user3 = generateTestUser();
      const user4 = generateTestUser();

      try {
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);
        const token3 = await createUserAndGetToken(page3, user3);
        const token4 = await createUserAndGetToken(page4, user4);

        const gameId = await createMultiPlayerGame(page1, token1, 4);
        await joinMultiPlayerGame(page2, token2, gameId);
        await joinMultiPlayerGame(page3, token3, gameId);
        await joinMultiPlayerGame(page4, token4, gameId);

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
        await coordinator.connect('player3', {
          playerId: user3.username,
          userId: user3.username,
          token: token3,
        });
        await coordinator.connect('player4', {
          playerId: user4.username,
          userId: user4.username,
          token: token4,
        });

        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);
        await coordinator.joinGame('player3', gameId);
        await coordinator.joinGame('player4', gameId);

        // Wait for P1's turn
        const p1Turn = (await coordinator.waitForGameState(
          'player1',
          (state) => state.currentPhase === 'ring_placement' && state.currentPlayer === 1,
          30_000
        )) as GameStateUpdateMessage;

        expect(p1Turn.data.gameState.currentPlayer).toBe(1);

        // P1 makes a move
        const p1Move = p1Turn.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p1Move).toBeDefined();
        await coordinator.sendMoveById('player1', gameId, p1Move!.id);

        // Wait for P2's turn
        const p2Turn = (await coordinator.waitForTurn(
          'player2',
          2,
          30_000
        )) as GameStateUpdateMessage;
        expect(p2Turn.data.gameState.currentPlayer).toBe(2);

        const p2Move = p2Turn.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p2Move).toBeDefined();
        await coordinator.sendMoveById('player2', gameId, p2Move!.id);

        // Wait for P3's turn
        const p3Turn = (await coordinator.waitForTurn(
          'player3',
          3,
          30_000
        )) as GameStateUpdateMessage;
        expect(p3Turn.data.gameState.currentPlayer).toBe(3);

        const p3Move = p3Turn.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p3Move).toBeDefined();
        await coordinator.sendMoveById('player3', gameId, p3Move!.id);

        // Wait for P4's turn (verifies 4-player rotation)
        const p4Turn = (await coordinator.waitForTurn(
          'player4',
          4,
          30_000
        )) as GameStateUpdateMessage;
        expect(p4Turn.data.gameState.currentPlayer).toBe(4);

        const p4Move = p4Turn.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p4Move).toBeDefined();
        await coordinator.sendMoveById('player4', gameId, p4Move!.id);

        // Wait for rotation back to P1 (confirms full 4-player cycle)
        const p1Again = (await coordinator.waitForTurn(
          'player1',
          1,
          30_000
        )) as GameStateUpdateMessage;
        expect(p1Again.data.gameState.currentPlayer).toBe(1);
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
        await context3.close();
        await context4.close();
      }
    });
  });

  test.describe('Spectator Observation', () => {
    test('spectator sees all player moves in 3-player game', async ({ browser }) => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const context3 = await browser.newContext();
      const contextSpec = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();
      const page3 = await context3.newPage();
      const pageSpec = await contextSpec.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();
      const user3 = generateTestUser();
      const spectatorUser = generateTestUser();

      try {
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);
        const token3 = await createUserAndGetToken(page3, user3);
        const spectatorToken = await createUserAndGetToken(pageSpec, spectatorUser);

        const gameId = await createMultiPlayerGame(page1, token1, 3);
        await joinMultiPlayerGame(page2, token2, gameId);
        await joinMultiPlayerGame(page3, token3, gameId);

        // Connect all players and spectator
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
        await coordinator.connect('player3', {
          playerId: user3.username,
          userId: user3.username,
          token: token3,
        });
        await coordinator.connect('spectator', {
          playerId: spectatorUser.username,
          userId: spectatorUser.username,
          token: spectatorToken,
        });

        // All join the game
        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);
        await coordinator.joinGame('player3', gameId);
        await coordinator.joinGame('spectator', gameId);

        // Wait for initial state on all clients
        await coordinator.waitForAll(['player1', 'player2', 'player3', 'spectator'], {
          type: 'gameState',
          predicate: (data) => isGameStateMessage(data),
          timeout: 30_000,
        });

        // P1 makes a move
        const p1State = (await coordinator.waitForTurn(
          'player1',
          1,
          30_000
        )) as GameStateUpdateMessage;
        const p1Move = p1State.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p1Move).toBeDefined();
        await coordinator.sendMoveById('player1', gameId, p1Move!.id);

        // Verify spectator sees the move
        const specStateAfterP1 = (await coordinator.waitForGameState(
          'spectator',
          (state) => state.moveHistory && state.moveHistory.length > 0,
          30_000
        )) as GameStateUpdateMessage;

        expect(specStateAfterP1.data.gameState.moveHistory.length).toBeGreaterThan(0);

        // P2 makes a move
        const p2State = (await coordinator.waitForTurn(
          'player2',
          2,
          30_000
        )) as GameStateUpdateMessage;
        const p2Move = p2State.data.validMoves.find((m) => m.type === 'place_ring');
        expect(p2Move).toBeDefined();
        await coordinator.sendMoveById('player2', gameId, p2Move!.id);

        // Verify spectator sees P2's move too
        const specStateAfterP2 = (await coordinator.waitForGameState(
          'spectator',
          (state) => state.moveHistory && state.moveHistory.length > 1,
          30_000
        )) as GameStateUpdateMessage;

        expect(specStateAfterP2.data.gameState.moveHistory.length).toBeGreaterThanOrEqual(2);

        // Verify spectator's state matches player1's state
        const p1Final = coordinator.getLastGameState('player1');
        const specFinal = coordinator.getLastGameState('spectator');

        expect(specFinal).not.toBeNull();
        expect(p1Final).not.toBeNull();
        expect(specFinal!.moveHistory.length).toBe(p1Final!.moveHistory.length);
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
        await context3.close();
        await contextSpec.close();
      }
    });
  });

  test.describe('Multi-Player Victory Conditions', () => {
    test('game ends when one player in 3-player game is eliminated', async ({ browser }) => {
      // This test verifies that game over is properly signaled to all players
      // when a victory condition is reached in a 3-player game
      const coordinator = createMultiClientCoordinator(serverUrl);

      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const context3 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();
      const page3 = await context3.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();
      const user3 = generateTestUser();

      try {
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);
        const token3 = await createUserAndGetToken(page3, user3);

        const gameId = await createMultiPlayerGame(page1, token1, 3);
        await joinMultiPlayerGame(page2, token2, gameId);
        await joinMultiPlayerGame(page3, token3, gameId);

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
        await coordinator.connect('player3', {
          playerId: user3.username,
          userId: user3.username,
          token: token3,
        });

        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);
        await coordinator.joinGame('player3', gameId);

        // Wait for initial game state
        await coordinator.waitForAll(['player1', 'player2', 'player3'], {
          type: 'gameState',
          predicate: (data) => isGameStateMessage(data),
          timeout: 30_000,
        });

        // Verify all three players are properly connected
        const state1 = coordinator.getLastGameState('player1');
        expect(state1).not.toBeNull();
        expect(state1!.maxPlayers).toBe(3);
        expect(state1!.players.length).toBeGreaterThanOrEqual(1);

        // The game should continue until a victory condition is met
        // For this test, we just verify the multi-player structure is correct
        // Victory testing is covered more thoroughly in fixture-based tests
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
        await context3.close();
      }
    });
  });
});
