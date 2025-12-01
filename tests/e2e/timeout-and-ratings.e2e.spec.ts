import { test, expect } from '@playwright/test';
import { generateTestUser, registerUser, waitForApiReady } from './helpers/test-utils';
import { HomePage, GamePage } from './pages';

/**
 * E2E Test Suite: Timeout & Rating Behaviour
 * ============================================================================
 *
 * Covers multiplayer game completion by move-clock timeout and verifies
 * rating updates for rated vs unrated games.
 *
 * - Creates short-timeout backend games directly via the HTTP API
 * - Starts both human players in browser contexts so GameSession/GameEngine
 *   timers run as in production
 * - Waits for timeout-driven game_over and asserts:
 *   - gameResult.reason === 'timeout'
 *   - Ratings change for rated games
 *   - Ratings do not change for unrated games
 *
 * RUN COMMAND: npx playwright test timeout-and-ratings.e2e.spec.ts
 *
 * REQUIREMENTS:
 * - PostgreSQL running
 * - Redis running
 * - Backend server running on http://localhost:3000 (or E2E_API_BASE_URL)
 */

test.describe('Timeout & Rating E2E', () => {
  test.setTimeout(120_000);

  const apiBaseUrl = (process.env.E2E_API_BASE_URL || 'http://localhost:3000').replace(/\/$/, '');

  async function getAuthToken(page: import('@playwright/test').Page): Promise<string> {
    const token = await page.evaluate(() => {
      return localStorage.getItem('auth_token') ?? localStorage.getItem('token');
    });
    if (!token) {
      throw new Error('Failed to get auth token after registration/login');
    }
    return token;
  }

  async function readRating(page: import('@playwright/test').Page): Promise<number> {
    const homePage = new HomePage(page);
    await homePage.goto();
    await homePage.goToProfile();
    await page.waitForURL('**/profile', { timeout: 10_000 });
    const ratingText = await page.locator('.text-emerald-400').first().textContent();
    return parseInt((ratingText || '').replace(/[^0-9]/g, ''), 10);
  }

  async function createShortTimeoutGame(
    creatorPage: import('@playwright/test').Page,
    token: string,
    { isRated }: { isRated: boolean }
  ): Promise<string> {
    const response = await creatorPage.request.post(`${apiBaseUrl}/api/games`, {
      data: {
        boardType: 'square8',
        maxPlayers: 2,
        isRated,
        isPrivate: true,
        timeControl: {
          type: 'blitz',
          initialTime: 5, // 5 seconds per player for fast timeout
          increment: 0,
        },
      },
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    expect(response.ok()).toBeTruthy();
    const json = await response.json();
    const gameId = json?.data?.game?.id as string | undefined;
    if (!gameId) {
      throw new Error(`Unexpected createGame response: ${JSON.stringify(json)}`);
    }
    return gameId;
  }

  async function joinGameHttp(
    page: import('@playwright/test').Page,
    token: string,
    gameId: string
  ): Promise<void> {
    const response = await page.request.post(`${apiBaseUrl}/api/games/${gameId}/join`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    expect(response.ok()).toBeTruthy();
  }

  async function waitForTimeoutGameOver(
    page: import('@playwright/test').Page,
    token: string,
    gameId: string
  ): Promise<{ reason: string; winner?: number | null }> {
    const deadline = Date.now() + 30_000;
    let lastJson: any;

    while (Date.now() < deadline) {
      const resp = await page.request.get(`${apiBaseUrl}/api/games/${gameId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      expect(resp.ok()).toBeTruthy();
      const body = await resp.json();
      lastJson = body;

      const game = body?.data?.game;
      const status = game?.status as string | undefined;
      const finalState = game?.finalState as { gameResult?: { reason?: string; winner?: number } };
      const result = finalState?.gameResult;

      if (status === 'completed' && result && typeof result.reason === 'string') {
        return {
          reason: result.reason,
          winner: result.winner ?? null,
        };
      }

      await page.waitForTimeout(1_000);
    }

    throw new Error(
      `Timed out waiting for game ${gameId} to complete. Last state: ${JSON.stringify(lastJson)}`
    );
  }

  test('rated vs unrated timeouts update rating appropriately', async ({ browser }) => {
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();

    try {
      await waitForApiReady(page1);

      const user1 = generateTestUser();
      const user2 = generateTestUser();

      // Register users (each context maintains its own session)
      await registerUser(page1, user1.username, user1.email, user1.password);
      await registerUser(page2, user2.username, user2.email, user2.password);

      const token1 = await getAuthToken(page1);
      const token2 = await getAuthToken(page2);

      const initialRating = await readRating(page1);
      expect(initialRating).toBeGreaterThan(0);

      // --- Rated timeout: rating should change ---
      const ratedGameId = await createShortTimeoutGame(page1, token1, { isRated: true });
      await joinGameHttp(page2, token2, ratedGameId);

      // Open the game for both players so GameSession/GameEngine timers are active.
      await page1.goto(`/game/${ratedGameId}`);
      await new GamePage(page1).waitForReady();
      await page2.goto(`/game/${ratedGameId}`);
      await new GamePage(page2).waitForReady();

      const ratedResult = await waitForTimeoutGameOver(page1, token1, ratedGameId);
      expect(ratedResult.reason).toBe('timeout');

      // Allow backend to persist rating updates.
      await page1.waitForTimeout(2_000);
      const afterRatedTimeout = await readRating(page1);
      expect(afterRatedTimeout).not.toBe(initialRating);

      // --- Unrated timeout: rating should NOT change ---
      const unratedGameId = await createShortTimeoutGame(page1, token1, { isRated: false });
      await joinGameHttp(page2, token2, unratedGameId);

      await page1.goto(`/game/${unratedGameId}`);
      await new GamePage(page1).waitForReady();
      await page2.goto(`/game/${unratedGameId}`);
      await new GamePage(page2).waitForReady();

      const unratedResult = await waitForTimeoutGameOver(page1, token1, unratedGameId);
      expect(unratedResult.reason).toBe('timeout');

      await page1.waitForTimeout(2_000);
      const afterUnratedTimeout = await readRating(page1);
      expect(afterUnratedTimeout).toBe(afterRatedTimeout);
    } finally {
      await context1.close();
      await context2.close();
    }
  });
});
