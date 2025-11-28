import { test, expect } from '@playwright/test';
import { registerAndLogin, createGame } from './helpers/test-utils';
import { LoginPage, RegisterPage, HomePage, GamePage } from './pages';

/**
 * Visual Regression Test Suite
 * ============================================================================
 *
 * This suite captures screenshots of key UI components and pages to detect
 * unintended visual changes. Screenshots are stored in __snapshots__ and
 * compared against baselines.
 *
 * RUNNING TESTS:
 *   npm run test:e2e:visual          - Run visual regression tests
 *   npm run test:e2e:visual:update   - Update baseline screenshots
 *
 * FIRST RUN:
 *   The first run will generate baseline screenshots. Subsequent runs will
 *   compare against these baselines. Use --update-snapshots to regenerate
 *   baselines when intentional UI changes are made.
 *
 * BEST PRACTICES:
 * - Prefer element screenshots over full page (more stable)
 * - Disable animations before capturing (configured in playwright.config.ts)
 * - Use consistent viewport sizes (from device presets)
 * - Run in CI with --update-snapshots only for intentional changes
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL + Redis running (for authenticated pages)
 * - Dev server running on http://localhost:5173
 */

test.describe('Visual Regression Tests', () => {
  // Use longer timeout for visual tests that may require setup
  test.setTimeout(120_000);

  test.describe('Page Screenshots', () => {
    test('home page visual appearance', async ({ page }) => {
      await page.goto('/');

      // Wait for page to be fully loaded
      await expect(page.getByRole('heading', { name: /Welcome to RingRift/i })).toBeVisible({
        timeout: 10_000,
      });

      // Wait for any animations to settle
      await page.waitForTimeout(1000);

      // Capture the main content area
      await expect(page).toHaveScreenshot('home-page.png', {
        fullPage: true,
        mask: [
          // Mask any dynamic elements (e.g., timestamps, user-specific data)
          page.locator('[data-testid="timestamp"]'),
        ],
      });
    });

    test('login page visual appearance', async ({ page }) => {
      const loginPage = new LoginPage(page);
      await loginPage.goto();
      await loginPage.waitForReady();

      // Wait for page to settle
      await page.waitForTimeout(500);

      // Capture the login form
      await expect(page).toHaveScreenshot('login-page.png', {
        fullPage: true,
      });
    });

    test('register page visual appearance', async ({ page }) => {
      const registerPage = new RegisterPage(page);
      await registerPage.goto();
      await registerPage.waitForReady();

      // Wait for page to settle
      await page.waitForTimeout(500);

      // Capture the registration form
      await expect(page).toHaveScreenshot('register-page.png', {
        fullPage: true,
      });
    });
  });

  test.describe('Game Board Screenshots', () => {
    test('initial game board state', async ({ page }) => {
      // Register and create a game to get to the game board
      await registerAndLogin(page);
      await createGame(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Wait for board to fully render
      await page.waitForTimeout(1000);

      // Capture just the board view element (more stable than full page)
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('initial-game-board.png');
    });

    test('game board with valid placement targets highlighted', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Wait for valid targets to appear (during ring placement phase)
      await gamePage.assertValidTargetsVisible();

      // Wait a bit for highlights to render
      await page.waitForTimeout(500);

      // Capture board with highlighted cells
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('board-with-valid-targets.png');
    });

    test('game board after placing a ring', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Place a ring
      await gamePage.clickFirstValidTarget();

      // Wait for the move to be processed and board to update
      await page.waitForTimeout(1500);

      // Capture board state after placement
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('board-after-placement.png');
    });
  });

  test.describe('Component Screenshots', () => {
    test('game HUD appearance', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // The HUD contains turn indicator, connection status, phase info
      // These should be visible in the game area
      // Find the HUD container (typically at the top of game view)
      const hudArea = page.locator('.flex.items-center.justify-between').first();

      // Wait for HUD to be fully rendered
      await page.waitForTimeout(500);

      // Capture HUD if it exists, otherwise the full game header area
      const hudVisible = await hudArea.isVisible();
      if (hudVisible) {
        await expect(hudArea).toHaveScreenshot('game-hud.png');
      } else {
        // Fallback to capturing the turn indicator area
        const turnArea = page.locator('text=/Turn/i').locator('..');
        await expect(turnArea).toHaveScreenshot('game-hud.png');
      }
    });

    test('game event log appearance', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Make a move to populate the event log
      await gamePage.clickFirstValidTarget();
      await page.waitForTimeout(1500);

      // Wait for the log to update
      await expect(gamePage.recentMovesSection).toBeVisible({ timeout: 15_000 });

      // Capture the game log section
      const gameLogSection = page.locator('text=/Game log/i').locator('..').locator('..');
      await expect(gameLogSection).toHaveScreenshot('game-event-log.png');
    });

    test('lobby page appearance (authenticated)', async ({ page }) => {
      await registerAndLogin(page);

      // Navigate to lobby
      await page.getByRole('link', { name: /lobby/i }).click();
      await page.waitForURL('**/lobby', { timeout: 15_000 });

      // Wait for lobby to load
      await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible({
        timeout: 10_000,
      });

      // Wait for content to settle
      await page.waitForTimeout(1000);

      // Capture the lobby page
      await expect(page).toHaveScreenshot('lobby-page.png', {
        fullPage: true,
        mask: [
          // Mask dynamic content like game IDs, timestamps
          page.locator('[data-testid="game-id"]'),
          page.locator('[data-testid="timestamp"]'),
          // Mask username which is dynamic
          page.locator('text=/e2e-user-/'),
        ],
      });
    });

    test('home page appearance (authenticated)', async ({ page }) => {
      await registerAndLogin(page);

      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.waitForReady();

      // Wait for content to settle
      await page.waitForTimeout(500);

      // Capture authenticated home page
      await expect(page).toHaveScreenshot('home-page-authenticated.png', {
        fullPage: true,
        mask: [
          // Mask dynamic username
          page.locator('text=/e2e-user-/'),
        ],
      });
    });
  });
});

/**
 * NOTE: Victory modal and other game state screenshots are intentionally
 * not included here because:
 * 1. Victory conditions require many moves to reach
 * 2. AI responses can be non-deterministic
 * 3. These states are better tested via unit/integration tests
 *
 * If you need to add victory modal screenshots, consider:
 * - Using a mock game state
 * - Triggering the modal programmatically via page.evaluate()
 * - Using test.skip() until the modal component is stable
 *
 * Example (requires modal to be triggerable):
 *
 * test.skip('victory modal appearance', async ({ page }) => {
 *   // Navigate to a game
 *   await registerAndLogin(page);
 *   await createGame(page);
 *
 *   // Trigger victory modal programmatically (requires implementation)
 *   await page.evaluate(() => {
 *     // This would require a testable hook in the app
 *     // window.__TEST_TRIGGER_VICTORY_MODAL__?.('P1');
 *   });
 *
 *   const modal = page.locator('[data-testid="victory-modal"]');
 *   await expect(modal).toBeVisible({ timeout: 5000 });
 *   await expect(modal).toHaveScreenshot('victory-modal.png');
 * });
 */
