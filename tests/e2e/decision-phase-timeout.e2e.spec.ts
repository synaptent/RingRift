import { test, expect } from '@playwright/test';
import {
  generateTestUser,
  registerUser,
  waitForApiReady,
  type TestUser,
  createFixtureGame,
} from './helpers/test-utils';
import {
  createMultiClientCoordinator,
  type MultiClientCoordinator,
} from '../helpers/MultiClientCoordinator';
import { createNetworkAwareCoordinator } from '../helpers/NetworkSimulator';
import type { DecisionPhaseTimedOutPayload } from '../../src/shared/types/websocket';

/**
 * E2E Test Suite: Decision-Phase Timeout Events
 * ============================================================================
 *
 * This suite drives shortTimeoutMs-backed decision-phase fixtures for:
 * - line_processing
 * - territory_processing
 *
 * and asserts that:
 * - Both players receive a `decision_phase_timed_out` WebSocket event
 * - The payload includes the expected `phase` value
 * - `autoSelectedMoveId` is a non-empty string and matches for both players
 *
 * RUN COMMAND:
 *   npx playwright test decision-phase-timeout.e2e.spec.ts
 *
 * REQUIREMENTS:
 * - PostgreSQL running
 * - Redis running
 * - Backend server running on http://localhost:3000 (or E2E_API_BASE_URL)
 */

test.describe('Decision-phase timeout E2E', () => {
  test.slow();
  test.setTimeout(120_000);

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

  async function setupCoordinatorAndPlayers(browser: import('@playwright/test').Browser): Promise<{
    coordinator: MultiClientCoordinator;
    context1: import('@playwright/test').BrowserContext;
    context2: import('@playwright/test').BrowserContext;
    page1: import('@playwright/test').Page;
    page2: import('@playwright/test').Page;
    user1: TestUser;
    user2: TestUser;
    token1: string;
    token2: string;
  }> {
    const coordinator = createMultiClientCoordinator(serverUrl);

    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();

    await waitForApiReady(page1);

    const user1 = generateTestUser();
    const user2 = generateTestUser();

    const token1 = await createUserAndGetToken(page1, user1);
    const token2 = await createUserAndGetToken(page2, user2);

    return {
      coordinator,
      context1,
      context2,
      page1,
      page2,
      user1,
      user2,
      token1,
      token2,
    };
  }

  async function waitForTimedOutEvents(
    coordinator: MultiClientCoordinator,
    expectedPhase: 'line_processing' | 'territory_processing' | 'chain_capture'
  ): Promise<{
    p1: DecisionPhaseTimedOutPayload;
    p2: DecisionPhaseTimedOutPayload;
  }> {
    const results = await coordinator.waitForAll(['player1', 'player2'], {
      type: 'event',
      eventName: 'decision_phase_timed_out',
      predicate: (payload) => {
        const msg = payload as DecisionPhaseTimedOutPayload;
        return msg?.data?.phase === expectedPhase && !!msg.data.autoSelectedMoveId;
      },
      timeout: 40_000,
    });

    const p1 = results.get('player1') as DecisionPhaseTimedOutPayload | undefined;
    const p2 = results.get('player2') as DecisionPhaseTimedOutPayload | undefined;

    if (!p1 || !p2) {
      throw new Error('Expected decision_phase_timed_out payloads for both players');
    }

    return { p1, p2 };
  }

  test('line_processing decision phase times out with deterministic autoSelectedMoveId', async ({
    browser,
  }) => {
    const { coordinator, context1, context2, page1, page2, user1, user2, token1, token2 } =
      await setupCoordinatorAndPlayers(browser);

    try {
      const { gameId } = await createFixtureGame(page1, {
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

      await coordinator.joinGame('player1', gameId);
      await coordinator.joinGame('player2', gameId);

      const { p1, p2 } = await waitForTimedOutEvents(coordinator, 'line_processing');

      expect(p1.data.gameId).toBe(gameId);
      expect(p2.data.gameId).toBe(gameId);
      expect(p1.data.phase).toBe('line_processing');
      expect(p2.data.phase).toBe('line_processing');
      expect(typeof p1.data.autoSelectedMoveId).toBe('string');
      expect(p1.data.autoSelectedMoveId.length).toBeGreaterThan(0);
      expect(p1.data.autoSelectedMoveId).toBe(p2.data.autoSelectedMoveId);
    } finally {
      await coordinator.cleanup();
      await context1.close();
      await context2.close();
    }
  });

  test('territory_processing decision phase times out with deterministic autoSelectedMoveId', async ({
    browser,
  }) => {
    const { coordinator, context1, context2, page1, page2, user1, user2, token1, token2 } =
      await setupCoordinatorAndPlayers(browser);

    try {
      const { gameId } = await createFixtureGame(page1, {
        scenario: 'territory_processing',
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

      await coordinator.joinGame('player1', gameId);
      await coordinator.joinGame('player2', gameId);

      const { p1, p2 } = await waitForTimedOutEvents(coordinator, 'territory_processing');

      expect(p1.data.gameId).toBe(gameId);
      expect(p2.data.gameId).toBe(gameId);
      expect(p1.data.phase).toBe('territory_processing');
      expect(p2.data.phase).toBe('territory_processing');
      expect(typeof p1.data.autoSelectedMoveId).toBe('string');
      expect(p1.data.autoSelectedMoveId.length).toBeGreaterThan(0);
      expect(p1.data.autoSelectedMoveId).toBe(p2.data.autoSelectedMoveId);
    } finally {
      await coordinator.cleanup();
      await context1.close();
      await context2.close();
    }
  });

  test('chain_capture decision phase times out with deterministic autoSelectedMoveId', async ({
    browser,
  }) => {
    const { coordinator, context1, context2, page1, page2, user1, user2, token1, token2 } =
      await setupCoordinatorAndPlayers(browser);

    try {
      const { gameId } = await createFixtureGame(page1, {
        scenario: 'chain_capture_choice',
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

      await coordinator.joinGame('player1', gameId);
      await coordinator.joinGame('player2', gameId);

      const { p1, p2 } = await waitForTimedOutEvents(coordinator, 'chain_capture');

      expect(p1.data.gameId).toBe(gameId);
      expect(p2.data.gameId).toBe(gameId);
      expect(p1.data.phase).toBe('chain_capture');
      expect(p2.data.phase).toBe('chain_capture');
      expect(typeof p1.data.autoSelectedMoveId).toBe('string');
      expect(p1.data.autoSelectedMoveId.length).toBeGreaterThan(0);
      expect(p1.data.autoSelectedMoveId).toBe(p2.data.autoSelectedMoveId);
    } finally {
      await coordinator.cleanup();
      await context1.close();
      await context2.close();
    }
  });

  test('line_processing timeout still delivers deterministic autoSelectedMoveId under packet loss', async ({
    browser,
  }) => {
    const coordinator = createNetworkAwareCoordinator(serverUrl);

    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();

    await waitForApiReady(page1);

    const user1 = generateTestUser();
    const user2 = generateTestUser();

    try {
      const token1 = await createUserAndGetToken(page1, user1);
      const token2 = await createUserAndGetToken(page2, user2);

      const { gameId } = await createFixtureGame(page1, {
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

      await coordinator.joinGame('player1', gameId);
      await coordinator.joinGame('player2', gameId);

      // Drive into the line_processing decision phase per P18.3‑1.
      await coordinator.waitForPhase('player1', 'line_processing', 15_000);

      // Apply degraded network conditions to player1: moderate packet loss + latency.
      coordinator.network.setCondition('player1', {
        packetLoss: 0.4,
        latencyMs: 150,
      });

      const results = await coordinator.waitForAll(['player1', 'player2'], {
        type: 'event',
        eventName: 'decision_phase_timed_out',
        predicate: (payload) => {
          const msg = payload as DecisionPhaseTimedOutPayload;
          return msg?.data?.phase === 'line_processing' && !!msg.data.autoSelectedMoveId;
        },
        timeout: 40_000,
      });

      const p1 = results.get('player1') as DecisionPhaseTimedOutPayload | undefined;
      const p2 = results.get('player2') as DecisionPhaseTimedOutPayload | undefined;

      if (!p1 || !p2) {
        throw new Error('Expected decision_phase_timed_out payloads for both players');
      }

      expect(p1.data.gameId).toBe(gameId);
      expect(p2.data.gameId).toBe(gameId);
      expect(p1.data.phase).toBe('line_processing');
      expect(p2.data.phase).toBe('line_processing');
      expect(typeof p1.data.autoSelectedMoveId).toBe('string');
      expect(p1.data.autoSelectedMoveId.length).toBeGreaterThan(0);
      expect(p1.data.autoSelectedMoveId).toBe(p2.data.autoSelectedMoveId);
    } finally {
      coordinator.network.clearCondition('player1');
      await coordinator.cleanup();
      await context1.close();
      await context2.close();
    }
  });
});
