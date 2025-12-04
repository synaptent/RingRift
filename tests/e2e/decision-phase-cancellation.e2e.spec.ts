import { test, expect } from '@playwright/test';
import {
  generateTestUser,
  registerUser,
  waitForApiReady,
  type TestUser,
  createFixtureGame,
} from './helpers/test-utils';
import { createNetworkAwareCoordinator } from '../helpers/NetworkSimulator';

/**
 * Decision-phase cancellation under terminateUserSessions (multi-client E2E)
 * ============================================================================
 *
 * This suite drives a short-timeout decision-phase fixture game under
 * degraded network conditions (packet loss/latency) and then triggers
 * account deletion for one player via DELETE /api/users/me. The backend
 * uses WebSocketServer.terminateUserSessions for that user, and we assert:
 *
 * - The opponent receives a terminal game_over event for the fixture game.
 * - The terminated user does not receive any new game_state or
 *   player_choice_required events after terminateUserSessions, even under
 *   packet loss.
 *
 * This mirrors the unit-level guarantees in
 * tests/unit/WebSocketServer.sessionTermination.test.ts at an E2E level
 * using the NetworkAwareCoordinator + NetworkSimulator infrastructure.
 *
 * RUN COMMAND:
 *   npx playwright test decision-phase-cancellation.e2e.spec.ts
 *
 * REQUIREMENTS:
 * - PostgreSQL running
 * - Redis running
 * - Backend server running on http://localhost:3000 (or E2E_API_BASE_URL)
 */

const serverUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';

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

test.describe('Decision-phase cancellation under terminateUserSessions (P18.3-1)', () => {
  test.slow();
  test.setTimeout(120_000);

  test('line_processing decision canceled via account deletion under packet loss yields clean event behaviour', async ({
    browser,
  }) => {
    const coordinator = createNetworkAwareCoordinator(serverUrl);

    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();

    const user1 = generateTestUser();
    const user2 = generateTestUser();

    await waitForApiReady(page1);

    try {
      // Register users and obtain JWT tokens via UI.
      const token1 = await createUserAndGetToken(page1, user1);
      const token2 = await createUserAndGetToken(page2, user2);

      // Create a short-timeout decision-phase fixture game that enters
      // line_processing with an outstanding decision.
      const { gameId } = await createFixtureGame(page1, {
        scenario: 'line_processing',
        isRated: false,
        shortTimeoutMs: 4_000,
        shortWarningBeforeMs: 2_000,
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

      // Drive into the line_processing decision phase per P18.3‑1.
      await coordinator.waitForPhase('player1', 'line_processing', 15_000);

      // Ensure a player_choice_required has been emitted for the active player.
      const choicePayload = await coordinator.waitFor('player1', {
        type: 'event',
        eventName: 'player_choice_required',
        predicate: (payload) => {
          const choice = payload as any;
          return choice?.gameId === gameId;
        },
        timeout: 30_000,
      });
      expect(choicePayload).toBeDefined();

      // Apply degraded network conditions (packet loss + latency) to player1
      // before terminating their sessions.
      coordinator.network.setCondition('player1', {
        packetLoss: 0.4,
        latencyMs: 150,
      });

      // Trigger account deletion for user1 via HTTP. The backend implementation
      // of DELETE /api/users/me calls WebSocketServer.terminateUserSessions,
      // which should terminate the WebSocket session and cancel any pending
      // decisions without emitting new game_state/player_choice_required events.
      const apiBaseUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';
      const deleteUrl = `${apiBaseUrl.replace(/\/$/, '')}/api/users/me`;

      const terminationTime = Date.now();
      const deleteResponse = await page1.request.delete(deleteUrl, {
        headers: {
          Authorization: `Bearer ${token1}`,
        },
      });
      expect(deleteResponse.ok()).toBeTruthy();

      // The opponent should eventually receive a terminal game_over for this game.
      const p2GameOver = (await coordinator.waitForGameOver('player2', 30_000)) as any;
      expect(p2GameOver?.data?.gameResult).toBeDefined();
      expect(p2GameOver.data.gameId).toBe(gameId);

      // Allow a short grace period for any in-flight messages to be delivered.
      await page2.waitForTimeout(2_000);

      // After terminateUserSessions (via account deletion), player1 must not
      // receive any additional game_state or player_choice_required events.
      const postTerminateMessages = coordinator.getMessagesMatching('player1', (msg) => {
        if (msg.timestamp < terminationTime) return false;
        return msg.eventName === 'game_state' || msg.eventName === 'player_choice_required';
      });

      expect(postTerminateMessages.length).toBe(0);
    } finally {
      coordinator.network.clearCondition('player1');
      await coordinator.cleanup();
      await context1.close();
      await context2.close();
    }
  });
});
