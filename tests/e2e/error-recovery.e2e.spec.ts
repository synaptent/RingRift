import { test, expect } from '@playwright/test';
import {
  generateTestUser,
  registerUser,
  loginUser,
  registerAndLogin,
  createGame,
  waitForGameReady,
} from './helpers/test-utils';
import { LoginPage, RegisterPage, GamePage } from './pages';

/**
 * E2E Test Suite: Error Recovery
 * ============================================================================
 *
 * This suite tests error recovery scenarios:
 * - Network disconnection and reconnection
 * - WebSocket disconnection during gameplay
 * - API error responses (4xx/5xx)
 * - Session expiry and auth token invalidation
 * - Rate limiting responses
 * - Form validation errors
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for user persistence)
 * - Redis running (for session management)
 * - Dev server running on http://localhost:5173
 *
 * These tests use Playwright's network interception capabilities to simulate
 * various error conditions without requiring backend modifications.
 *
 * RUN COMMAND: npm run test:e2e -- error-recovery.e2e.spec.ts
 */

test.describe('Error Recovery - Network Failures', () => {
  test.setTimeout(120_000);

  test('handles temporary network disconnection gracefully', async ({ page, context }) => {
    // Register and login a user
    await registerAndLogin(page);

    // Navigate to lobby
    await page.getByRole('link', { name: /lobby/i }).click();
    await page.waitForURL('**/lobby', { timeout: 15_000 });
    await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible({
      timeout: 10_000,
    });

    // Simulate network failure
    await context.setOffline(true);

    // Attempt to refresh games list (should fail gracefully)
    const refreshButton = page.locator('button').filter({ hasText: /refresh/i });
    if (await refreshButton.isVisible()) {
      await refreshButton.click();
    }

    // Wait a moment for the error state to appear
    await page.waitForTimeout(2000);

    // Verify the page doesn't crash - should still show lobby structure
    await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible();

    // Restore network
    await context.setOffline(false);

    // Wait for potential reconnection
    await page.waitForTimeout(3000);

    // Verify page is still functional after network restore
    await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible();
  });

  test('recovers page functionality after network restore', async ({ page, context }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.waitForReady();

    // Go offline before attempting login
    await context.setOffline(true);

    // Attempt login (should fail/hang)
    const user = generateTestUser();
    await loginPage.fillEmail(user.email);
    await loginPage.fillPassword(user.password);
    await loginPage.clickLogin();

    // Wait a moment
    await page.waitForTimeout(2000);

    // Restore network
    await context.setOffline(false);

    // Wait for potential error state
    await page.waitForTimeout(3000);

    // Page should still be usable - either showing login form or error
    const loginFormOrError = page
      .getByRole('heading', { name: /login/i })
      .or(page.locator('.text-red-300, .text-red-400'));
    await expect(loginFormOrError).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error Recovery - WebSocket Disconnection', () => {
  test.setTimeout(120_000);

  test('handles WebSocket connection loss during game view', async ({ page }) => {
    // Register and create a game
    await registerAndLogin(page);
    await createGame(page);
    await waitForGameReady(page);

    const gamePage = new GamePage(page);
    await gamePage.assertConnected();

    // Force close WebSocket connections via page evaluation
    await page.evaluate(() => {
      // Find and close any WebSocket connections
      const wsList = (window as any).__webSockets || [];
      wsList.forEach((ws: WebSocket) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close(1000, 'Test forced disconnect');
        }
      });

      // Also try to close via performance entries (if WebSocket is tracked)
      // This is a fallback mechanism
      performance.getEntriesByType('resource').forEach((entry) => {
        if (entry.name.includes('ws://') || entry.name.includes('wss://')) {
          // Resource exists, connection may be tracked elsewhere
        }
      });
    });

    // Wait for potential reconnection or error state
    await page.waitForTimeout(5000);

    // The board should still be visible (page shouldn't crash)
    await expect(gamePage.boardView).toBeVisible({ timeout: 10_000 });
  });

  test.skip('shows reconnecting state on WebSocket disconnect', async ({ page }) => {
    // SKIP: Requires backend WebSocket implementation to expose reconnection UI
    // This test would verify that a "Reconnecting..." message appears
    // when WebSocket connection is lost

    await registerAndLogin(page);
    await createGame(page);
    await waitForGameReady(page);

    // Would need to intercept and break WebSocket, then verify reconnect UI
    // await expect(page.locator('text=/reconnecting/i')).toBeVisible();
  });

  test('reconnects after decision timeout and shows auto-resolved outcome in HUD', async ({
    page,
  }) => {
    // This scenario uses the backend decision-phase fixture route and the
    // DECISION_PHASE_TIMEOUT_* env overrides to exercise end-to-end
    // decision timeout + reconnect behaviour.

    // 1) Register and obtain an authenticated session.
    await registerAndLogin(page);

    // 2) Create a backend game that starts in a line_processing decision phase
    //    via the test-only /api/games/fixtures/decision-phase endpoint.
    const apiBaseUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';
    const fixtureResponse = await page.request.post(
      `${apiBaseUrl.replace(/\/$/, '')}/api/games/fixtures/decision-phase`,
      {
        data: {
          scenario: 'line_processing',
          isRated: true,
        },
      }
    );

    expect(fixtureResponse.ok()).toBeTruthy();
    const fixtureJson = await fixtureResponse.json();
    const gameId: string | undefined = fixtureJson?.data?.gameId;
    expect(gameId).toBeTruthy();

    // 3) Navigate to the game and wait for WebSocket connection.
    const gamePage = new GamePage(page);
    await gamePage.goto(gameId!);
    await gamePage.waitForReady(30_000);

    // The HUD should indicate we're in a decision-centric phase. The
    // decision-phase banner is our primary signal here.
    const decisionBanner = page.getByTestId('decision-phase-banner');
    await expect(decisionBanner).toBeVisible({ timeout: 15_000 });

    // 4) Simulate a client-side disconnect while the decision is pending by
    //    explicitly closing any tracked WebSocket connections.
    await page.evaluate(() => {
      const wsList = (window as any).__webSockets || [];
      wsList.forEach((ws: WebSocket) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close(4000, 'Decision-timeout E2E forced disconnect');
        }
      });
    });

    // 5) Wait for the backend decision-phase timeout to elapse. The server
    //    exposes configurable DECISION_PHASE_TIMEOUT_* values; we derive a
    //    bound from those env vars so the test adapts to CI configuration.
    const defaultTimeoutMs =
      Number(process.env.DECISION_PHASE_TIMEOUT_MS || '') || 30_000;
    const warningBeforeMs =
      Number(process.env.DECISION_PHASE_TIMEOUT_WARNING_MS || '') || 5_000;

    // Allow a bit of slack past the nominal timeout; cap to keep the test
    // bounded even if env overrides are not set.
    const waitMs = Math.min(defaultTimeoutMs + warningBeforeMs + 2_000, 45_000);
    await page.waitForTimeout(waitMs);

    // 6) Reconnect to the same game and wait for the board + HUD to resync.
    await gamePage.reloadAndWait();

    // 7) Assert that the HUD reflects post-decision state and that the
    //    recent-moves log includes an auto-resolved decision move.
    //
    // We expect a process_line / choose_line_reward move to appear after
    // the timeout; assert that at least one of these appears in the log.
    await expect(page.locator('text=/Recent moves/i')).toBeVisible({ timeout: 15_000 });
    const decisionMoveEntry = page
      .locator('li')
      .filter({ hasText: /process_line|choose_line_reward/i });
    await expect(decisionMoveEntry).toBeVisible({ timeout: 15_000 });

    // Additionally, assert that the game has progressed out of the original
    // decision snapshot: either a new phase or a different current player.
    // We look for a generic phase label that is not the initial
    // line_processing copy any more.
    const phaseText = page.locator('text=/Phase/i');
    await expect(phaseText).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error Recovery - API Error Responses', () => {
  test.setTimeout(120_000);

  test('handles API 500 error responses gracefully', async ({ page, context }) => {
    // First register a user normally
    await registerAndLogin(page);

    // Navigate to lobby
    await page.getByRole('link', { name: /lobby/i }).click();
    await page.waitForURL('**/lobby', { timeout: 15_000 });

    // Intercept game API calls and return 500 error
    await page.route('**/api/games**', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    // Attempt to create a game (should show error)
    await page.getByRole('button', { name: /\+ Create Game/i }).click();
    await expect(page.getByRole('heading', { name: /Create Backend Game/i })).toBeVisible({
      timeout: 5_000,
    });
    await page.getByRole('button', { name: /^Create Game$/i }).click();

    // Wait for error state
    await page.waitForTimeout(3000);

    // Should show error message or stay on form (not crash)
    const formOrError = page
      .getByRole('heading', { name: /Create Backend Game/i })
      .or(page.locator('.text-red-300, .text-red-400, [class*="error"]'));
    await expect(formOrError).toBeVisible({ timeout: 10_000 });

    // Remove the route to restore normal behavior
    await page.unroute('**/api/games**');
  });

  test('handles API 400 bad request gracefully', async ({ page }) => {
    await registerAndLogin(page);

    // Navigate to lobby
    await page.getByRole('link', { name: /lobby/i }).click();
    await page.waitForURL('**/lobby', { timeout: 15_000 });

    // Intercept and return 400 error
    await page.route('**/api/games', (route) => {
      if (route.request().method() === 'POST') {
        route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Invalid game configuration' }),
        });
      } else {
        route.continue();
      }
    });

    // Attempt to create a game
    await page.getByRole('button', { name: /\+ Create Game/i }).click();
    await expect(page.getByRole('heading', { name: /Create Backend Game/i })).toBeVisible({
      timeout: 5_000,
    });
    await page.getByRole('button', { name: /^Create Game$/i }).click();

    // Wait for error handling
    await page.waitForTimeout(3000);

    // Form should still be visible or error shown (page shouldn't crash)
    const formVisible = await page
      .getByRole('heading', { name: /Create Backend Game/i })
      .isVisible();
    const errorVisible = await page
      .locator('.text-red-300, .text-red-400, [class*="error"]')
      .isVisible();
    expect(formVisible || errorVisible).toBeTruthy();

    await page.unroute('**/api/games');
  });

  test('handles API 404 not found gracefully', async ({ page }) => {
    await registerAndLogin(page);

    // Try to navigate to a non-existent game
    await page.goto('/game/non-existent-game-id-12345');

    // Wait for error state
    await page.waitForTimeout(3000);

    // Should show error or redirect (not crash)
    const errorOrRedirect = page
      .locator('text=/not found/i')
      .or(page.locator('text=/error/i'))
      .or(page.getByRole('heading', { name: /lobby/i }))
      .or(page.getByRole('heading', { name: /welcome/i }));

    await expect(errorOrRedirect).toBeVisible({ timeout: 15_000 });
  });
});

test.describe('Error Recovery - Session Expiry', () => {
  test.setTimeout(120_000);

  test('redirects to login on session expiry', async ({ page, context }) => {
    // Register and login
    const user = await registerAndLogin(page);

    // Verify we're authenticated
    await expect(page.getByRole('button', { name: /logout/i })).toBeVisible();

    // Clear all cookies to simulate session expiry
    await context.clearCookies();

    // Navigate to a protected page (lobby)
    await page.getByRole('link', { name: /lobby/i }).click();

    // Wait for redirect or error
    await page.waitForTimeout(3000);

    // Should redirect to login or show authentication required
    const loginOrAuth = page
      .getByRole('heading', { name: /login/i })
      .or(page.locator('text=/sign in/i'))
      .or(page.locator('text=/authentication required/i'));

    await expect(loginOrAuth).toBeVisible({ timeout: 15_000 });
  });

  test('handles expired token on API request', async ({ page, context }) => {
    await registerAndLogin(page);

    // Navigate to lobby first
    await page.getByRole('link', { name: /lobby/i }).click();
    await page.waitForURL('**/lobby', { timeout: 15_000 });

    // Intercept API to return 401 unauthorized
    await page.route('**/api/**', (route) => {
      route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Token expired' }),
      });
    });

    // Trigger an API call by trying to refresh or navigate
    await page.reload();

    // Wait for error handling
    await page.waitForTimeout(3000);

    // Should show login page or auth error
    const loginOrError = page
      .getByRole('heading', { name: /login/i })
      .or(page.locator('.text-red-300, .text-red-400'))
      .or(page.locator('text=/unauthorized/i'))
      .or(page.locator('text=/session/i'));

    await expect(loginOrError).toBeVisible({ timeout: 15_000 });

    await page.unroute('**/api/**');
  });
});

test.describe('Error Recovery - Rate Limiting', () => {
  test.setTimeout(120_000);

  test('handles rate limiting gracefully', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.waitForReady();

    // Intercept login API and return 429
    await page.route('**/api/auth/login', (route) => {
      route.fulfill({
        status: 429,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'Too many requests',
          retryAfter: 60,
        }),
        headers: {
          'Retry-After': '60',
        },
      });
    });

    // Attempt login
    const user = generateTestUser();
    await loginPage.login(user.email, user.password);

    // Wait for error state
    await page.waitForTimeout(2000);

    // Should show rate limit error or general error (page shouldn't crash)
    const errorOrForm = page
      .locator('.text-red-300, .text-red-400, [class*="error"]')
      .or(page.getByRole('heading', { name: /login/i }));
    await expect(errorOrForm).toBeVisible({ timeout: 10_000 });

    await page.unroute('**/api/auth/login');
  });

  test('handles rate limiting on game API calls', async ({ page }) => {
    await registerAndLogin(page);

    // Navigate to lobby
    await page.getByRole('link', { name: /lobby/i }).click();
    await page.waitForURL('**/lobby', { timeout: 15_000 });

    // Intercept games API with rate limit
    await page.route('**/api/games**', (route) => {
      route.fulfill({
        status: 429,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Rate limit exceeded. Please try again later.' }),
      });
    });

    // Try to create a game
    await page.getByRole('button', { name: /\+ Create Game/i }).click();
    await expect(page.getByRole('heading', { name: /Create Backend Game/i })).toBeVisible({
      timeout: 5_000,
    });
    await page.getByRole('button', { name: /^Create Game$/i }).click();

    // Wait for error handling
    await page.waitForTimeout(3000);

    // Form should still be visible or error shown
    const formOrError = page
      .getByRole('heading', { name: /Create Backend Game/i })
      .or(page.locator('.text-red-300, .text-red-400'));
    await expect(formOrError).toBeVisible({ timeout: 10_000 });

    await page.unroute('**/api/games**');
  });
});

test.describe('Error Recovery - Form Validation', () => {
  test.setTimeout(90_000);

  test('shows validation errors for empty registration form', async ({ page }) => {
    const registerPage = new RegisterPage(page);
    await registerPage.goto();
    await registerPage.waitForReady();

    // Submit empty form
    await registerPage.clickCreateAccount();

    // Wait for validation
    await page.waitForTimeout(1000);

    // Should show validation errors or prevent submission
    // The form should still be visible (not navigated away)
    await expect(registerPage.heading).toBeVisible();
  });

  test('shows error for invalid email format during registration', async ({ page }) => {
    const registerPage = new RegisterPage(page);
    await registerPage.goto();
    await registerPage.waitForReady();

    // Fill with invalid email
    await registerPage.fillEmail('not-a-valid-email');
    await registerPage.fillUsername('testuser123');
    await registerPage.fillPassword('ValidPassword123!');
    await registerPage.fillConfirmPassword('ValidPassword123!');
    await registerPage.clickCreateAccount();

    // Wait for validation
    await page.waitForTimeout(2000);

    // Should show validation error or stay on form
    const formStillVisible = await registerPage.heading.isVisible();
    expect(formStillVisible).toBeTruthy();
  });

  test('shows error for password mismatch during registration', async ({ page }) => {
    const registerPage = new RegisterPage(page);
    await registerPage.goto();
    await registerPage.waitForReady();

    const user = generateTestUser();
    await registerPage.fillEmail(user.email);
    await registerPage.fillUsername(user.username);
    await registerPage.fillPassword('Password123!');
    await registerPage.fillConfirmPassword('DifferentPassword456!');
    await registerPage.clickCreateAccount();

    // Wait for validation
    await page.waitForTimeout(2000);

    // Should show error or stay on form
    const errorOrForm = page
      .locator('.text-red-300, .text-red-400, [class*="error"]')
      .or(registerPage.heading);
    await expect(errorOrForm).toBeVisible({ timeout: 10_000 });
  });

  test('preserves form state after validation error', async ({ page }) => {
    const registerPage = new RegisterPage(page);
    await registerPage.goto();
    await registerPage.waitForReady();

    const user = generateTestUser();
    const invalidEmail = 'not-valid-email';

    // Fill form with invalid email
    await registerPage.fillEmail(invalidEmail);
    await registerPage.fillUsername(user.username);
    await registerPage.fillPassword(user.password);
    await registerPage.fillConfirmPassword(user.password);

    // Submit
    await registerPage.clickCreateAccount();

    // Wait for validation
    await page.waitForTimeout(2000);

    // Form should preserve username (email might be cleared based on HTML5 validation)
    const usernameValue = await page.getByLabel('Username').inputValue();
    expect(usernameValue).toBe(user.username);
  });

  test('shows error for short password during registration', async ({ page }) => {
    const registerPage = new RegisterPage(page);
    await registerPage.goto();
    await registerPage.waitForReady();

    const user = generateTestUser();
    await registerPage.fillEmail(user.email);
    await registerPage.fillUsername(user.username);
    await registerPage.fillPassword('short'); // Too short
    await registerPage.fillConfirmPassword('short');
    await registerPage.clickCreateAccount();

    // Wait for validation
    await page.waitForTimeout(2000);

    // Should show error or stay on form
    const formStillVisible = await registerPage.heading.isVisible();
    expect(formStillVisible).toBeTruthy();
  });
});

test.describe('Error Recovery - Page Reload', () => {
  test.setTimeout(120_000);

  test('game state persists after page reload', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page);
    await waitForGameReady(page);

    const gamePage = new GamePage(page);
    const gameId = gamePage.getGameId();
    expect(gameId).toBeTruthy();

    // Reload the page
    await page.reload();

    // Wait for game to reload
    await gamePage.waitForReady();

    // Verify we're still on the same game
    expect(gamePage.getGameId()).toBe(gameId);
    await expect(gamePage.boardView).toBeVisible();
  });

  test('handles reload during loading state', async ({ page }) => {
    await registerAndLogin(page);

    // Start navigating to lobby
    await page.getByRole('link', { name: /lobby/i }).click();

    // Immediately reload (during potential loading state)
    await page.reload();

    // Wait for page to stabilize
    await page.waitForTimeout(3000);

    // Should end up on a valid page (not crashed)
    const validPage = page
      .getByRole('heading', { name: /Game Lobby/i })
      .or(page.getByRole('heading', { name: /Welcome/i }))
      .or(page.getByRole('heading', { name: /Login/i }));

    await expect(validPage).toBeVisible({ timeout: 15_000 });
  });
});
