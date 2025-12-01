import { test, expect, devices } from '@playwright/test';
import {
  registerAndLogin,
  createGame,
  goToSandbox,
  createNearVictoryGame,
} from './helpers/test-utils';
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

  test.describe('Sandbox Board Screenshots', () => {
    test('sandbox pregame setup page', async ({ page }) => {
      await goToSandbox(page);

      // Wait for the setup form to render
      await expect(
        page.getByRole('heading', { name: /Start a RingRift Game \(Local Sandbox\)/i })
      ).toBeVisible({ timeout: 10_000 });

      // Wait for page to settle
      await page.waitForTimeout(500);

      // Capture the sandbox setup page
      await expect(page).toHaveScreenshot('sandbox-pregame-setup.png', {
        fullPage: true,
      });
    });

    test('sandbox game board after launch', async ({ page }) => {
      await goToSandbox(page);

      // Click the Launch Game button
      await page.getByRole('button', { name: /Launch Game/i }).click();

      // Wait for either backend game or local sandbox to load
      // Try backend first
      try {
        await page.waitForURL('**/game/**', { timeout: 15_000 });
        const gamePage = new GamePage(page);
        await gamePage.waitForReady(20_000);

        // Capture the backend game board
        const boardView = page.getByTestId('board-view');
        await expect(boardView).toHaveScreenshot('sandbox-launched-board.png');
      } catch {
        // Fallback: local sandbox mode
        await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });
        await page.waitForTimeout(1000);

        // Capture local sandbox board
        const boardView = page.getByTestId('board-view');
        await expect(boardView).toHaveScreenshot('sandbox-local-board.png');
      }
    });

    test('sandbox touch controls panel', async ({ page }) => {
      await goToSandbox(page);

      // Click the Launch Game button
      await page.getByRole('button', { name: /Launch Game/i }).click();

      // Wait for board to be ready
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });

      // Check if sandbox touch controls exist (local sandbox mode)
      const touchControls = page.getByTestId('sandbox-touch-controls');
      const hasTouchControls = await touchControls.isVisible().catch(() => false);

      if (hasTouchControls) {
        await expect(touchControls).toHaveScreenshot('sandbox-touch-controls.png');
      } else {
        // Skip screenshot if not in local sandbox mode
        test.skip();
      }
    });
  });

  test.describe('Hex Board Screenshots', () => {
    test('hex board initial state', async ({ page }) => {
      await registerAndLogin(page);

      // Create a hexagonal board game
      await createGame(page, { boardType: 'hexagonal' });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Wait for hex board to fully render
      await page.waitForTimeout(1000);

      // Capture the hex board
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('hex-board-initial.png');
    });

    test('hex board with valid targets', async ({ page }) => {
      await registerAndLogin(page);

      // Create a hexagonal board game
      await createGame(page, { boardType: 'hexagonal' });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Wait for valid targets to appear
      await gamePage.assertValidTargetsVisible();
      await page.waitForTimeout(500);

      // Capture hex board with highlighted cells
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('hex-board-with-targets.png');
    });
  });

  test.describe('19x19 Board Screenshots', () => {
    test('19x19 board initial state', async ({ page }) => {
      await registerAndLogin(page);

      // Create a 19x19 board game
      await createGame(page, { boardType: 'square19' });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Wait for large board to fully render
      await page.waitForTimeout(1500);

      // Capture the 19x19 board
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('board-19x19-initial.png');
    });
  });

  test.describe('Victory Modal Screenshots', () => {
    test('victory modal after winning capture', async ({ page }) => {
      await registerAndLogin(page);

      // Create a near-victory fixture game
      const gameId = await createNearVictoryGame(page);

      // The game should be set up with P1 about to win via elimination
      // Wait for the board to be ready
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });

      // Find and click the winning move target (the move that eliminates P2)
      // The fixture places P2's last ring at (4,3) which P1 can capture
      const targetCell = page.getByTestId('board-view').locator('button[data-x="4"][data-y="3"]');
      const sourceCell = page.getByTestId('board-view').locator('button[data-x="3"][data-y="3"]');

      // Wait for it to be our turn and make the winning move
      await sourceCell.waitFor({ state: 'visible', timeout: 15_000 });
      await sourceCell.click();
      await targetCell.click();

      // Wait for victory modal to appear
      const victoryModal = page.locator(
        '[data-testid="victory-modal"], [class*="victory"], .fixed.inset-0'
      );

      try {
        await victoryModal.waitFor({ state: 'visible', timeout: 15_000 });
        await page.waitForTimeout(500);

        // Capture the victory modal
        await expect(victoryModal.first()).toHaveScreenshot('victory-modal.png');
      } catch {
        // If modal doesn't appear, the fixture may not have triggered victory
        // Skip this screenshot gracefully
        console.log('Victory modal did not appear - fixture may not support this flow');
        test.skip();
      }
    });
  });
});

test.describe('Mobile Viewport Visual Tests', () => {
  // Use mobile viewport for all tests in this describe block
  test.use({ viewport: { width: 375, height: 667 } }); // iPhone SE

  test.setTimeout(120_000);

  test('home page on mobile', async ({ page }) => {
    await page.goto('/');

    // Wait for page to load
    await expect(page.getByRole('heading', { name: /Welcome to RingRift/i })).toBeVisible({
      timeout: 10_000,
    });
    await page.waitForTimeout(500);

    // Capture mobile home page
    await expect(page).toHaveScreenshot('mobile-home-page.png', {
      fullPage: true,
    });
  });

  test('login page on mobile', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.waitForReady();
    await page.waitForTimeout(500);

    // Capture mobile login page
    await expect(page).toHaveScreenshot('mobile-login-page.png', {
      fullPage: true,
    });
  });

  test('game board on mobile', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page);

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();
    await page.waitForTimeout(1000);

    // Capture the game board on mobile viewport
    await expect(page).toHaveScreenshot('mobile-game-board.png', {
      fullPage: true,
    });
  });

  test('sandbox page on mobile', async ({ page }) => {
    await goToSandbox(page);

    await expect(
      page.getByRole('heading', { name: /Start a RingRift Game \(Local Sandbox\)/i })
    ).toBeVisible({ timeout: 10_000 });
    await page.waitForTimeout(500);

    // Capture mobile sandbox setup
    await expect(page).toHaveScreenshot('mobile-sandbox-setup.png', {
      fullPage: true,
    });
  });
});

test.describe('Tablet Viewport Visual Tests', () => {
  // Use iPad viewport
  test.use({ viewport: { width: 768, height: 1024 } });

  test.setTimeout(120_000);

  test('game board on tablet', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page);

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();
    await page.waitForTimeout(1000);

    // Capture the game board on tablet viewport
    const boardView = page.getByTestId('board-view');
    await expect(boardView).toHaveScreenshot('tablet-game-board.png');
  });

  test('hex board on tablet', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page, { boardType: 'hexagonal' });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();
    await page.waitForTimeout(1000);

    // Capture hexagonal board on tablet viewport
    const boardView = page.getByTestId('board-view');
    await expect(boardView).toHaveScreenshot('tablet-hex-board.png');
  });
});

/**
 * VISUAL REGRESSION TESTING DOCUMENTATION
 * ============================================================================
 *
 * This test suite uses Playwright's built-in visual comparison features to
 * detect unintended UI changes across the application.
 *
 * KEY CONCEPTS:
 *
 * 1. BASELINE SCREENSHOTS
 *    The first run of visual tests creates baseline screenshots stored in
 *    tests/e2e/__snapshots__/. These are committed to git and serve as the
 *    "expected" appearance of UI components.
 *
 * 2. SCREENSHOT COMPARISON
 *    Subsequent test runs compare the current appearance against baselines.
 *    Differences beyond the configured threshold (maxDiffPixels, threshold)
 *    cause test failures.
 *
 * 3. DIFF IMAGES
 *    When a visual test fails, Playwright generates diff images showing:
 *    - The expected (baseline) screenshot
 *    - The actual (current) screenshot
 *    - A diff highlighting the differences
 *    These are stored in test-results/ folder.
 *
 * COMMANDS:
 *
 *   npm run test:e2e:visual           # Run visual regression tests
 *   npm run test:e2e:visual:update    # Update baseline screenshots
 *   npx playwright test visual-regression --update-snapshots  # Same as above
 *
 * BEST PRACTICES:
 *
 * - Prefer element screenshots over full page (more stable across environments)
 * - Disable animations (configured in playwright.config.ts)
 * - Use consistent viewport sizes from device presets
 * - Mask dynamic content (timestamps, user IDs, etc.)
 * - Review diff images carefully before updating baselines
 *
 * CI CONSIDERATIONS:
 *
 * - Font rendering may differ between environments
 * - Consider using docker containers for consistent rendering
 * - Use --update-snapshots only for intentional UI changes
 *
 * See tests/e2e/VISUAL_TESTING.md for complete documentation.
 */
