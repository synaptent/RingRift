import { test, expect } from '@playwright/test';
import { goToSandbox } from './helpers/test-utils';
import { LoginPage, RegisterPage } from './pages';

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
 * - Dev server running on http://localhost:5173 (Playwright webServer starts it)
 */

test.describe('Visual Regression Tests', () => {
  // Use longer timeout for visual tests that may require setup
  test.setTimeout(120_000);

  test.describe('Page Screenshots', () => {
    test('entry route (guest) redirects to login', async ({ page }) => {
      await page.goto('/');

      // Guests are redirected to /login; assert the login shell is visible.
      await expect(page.getByRole('heading', { name: /login/i })).toBeVisible({
        timeout: 10_000,
      });

      // Wait for any animations to settle
      await page.waitForTimeout(1000);

      await expect(page).toHaveScreenshot('entry-guest.png', {
        fullPage: true,
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
      await goToSandbox(page);
      await page.getByRole('button', { name: /Learn the Basics/i }).click();

      // Wait for board to fully render
      await page.waitForTimeout(1000);

      // Capture just the board view element (more stable than full page)
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('initial-game-board.png');
    });

    test('game board with valid placement targets highlighted', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Learn the Basics/i }).click();

      const validTargets = page
        .getByTestId('board-view')
        .locator('button[class*="outline-emerald"]');
      await expect(validTargets.first()).toBeVisible({ timeout: 25_000 });

      // Wait a bit for highlights to render
      await page.waitForTimeout(500);

      // Capture board with highlighted cells
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('board-with-valid-targets.png');
    });

    test('game board after placing a ring', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Learn the Basics/i }).click();

      const validTargets = page
        .getByTestId('board-view')
        .locator('button[class*="outline-emerald"]');
      await validTargets.first().waitFor({ state: 'visible', timeout: 25_000 });
      await validTargets.first().click();

      // Wait for the move to be processed and board to update
      await page.waitForTimeout(1500);

      // Capture board state after placement
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('board-after-placement.png');
    });
  });

  test.describe('Component Screenshots', () => {
    test('game HUD appearance', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Learn the Basics/i }).click();

      const hudArea = page.getByTestId('game-hud');

      // Wait for HUD to be fully rendered
      await page.waitForTimeout(500);

      await expect(hudArea).toHaveScreenshot('game-hud.png');
    });

    test('game event log appearance', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Learn the Basics/i }).click();

      // Make a move to populate the event log
      const validTargets = page
        .getByTestId('board-view')
        .locator('button[class*="outline-emerald"]');
      await validTargets.first().waitFor({ state: 'visible', timeout: 25_000 });
      await validTargets.first().click();
      await page.waitForTimeout(1500);

      // Open advanced panels to reveal the event log in sandbox mode.
      const advancedPanels = page.getByTestId('sandbox-advanced-sidebar-panels');
      await advancedPanels.locator('summary').click();
      await expect(advancedPanels).toHaveAttribute('open', '', { timeout: 10_000 });

      // Capture the game log section
      const gameLogSection = page.locator('text=/Game log/i').locator('..').locator('..');
      await expect(gameLogSection).toHaveScreenshot('game-event-log.png');
    });

    // Note: authenticated lobby/home visuals are covered by dedicated E2E suites and
    // intentionally omitted from visual baselines to keep the screenshot suite
    // backend-independent and stable.
  });

  test.describe('Sandbox Board Screenshots', () => {
    test('sandbox pregame setup page', async ({ page }) => {
      await goToSandbox(page);

      // Wait for the setup form to render
      await expect(page.getByRole('heading', { name: /Start a Game \(Sandbox\)/i })).toBeVisible({
        timeout: 10_000,
      });

      // Wait for page to settle
      await page.waitForTimeout(500);

      // Capture the sandbox setup page
      await expect(page).toHaveScreenshot('sandbox-pregame-setup.png', {
        fullPage: true,
      });
    });

    test('sandbox game board after launch', async ({ page }) => {
      await goToSandbox(page);

      // Click a preset to launch a local sandbox game immediately.
      await page.getByRole('button', { name: /Learn the Basics/i }).click();

      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });
      await page.waitForTimeout(1000);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('sandbox-local-board.png');
    });

    test('sandbox touch controls panel', async ({ page }) => {
      await goToSandbox(page);

      await page.getByRole('button', { name: /Learn the Basics/i }).click();

      // Wait for board to be ready
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });

      // Check if sandbox touch controls exist (local sandbox mode)
      const touchControls = page.getByTestId('sandbox-touch-controls');
      const hasTouchControls = await touchControls.isVisible().catch(() => false);

      if (hasTouchControls) {
        await expect(touchControls).toHaveScreenshot('sandbox-touch-controls.png');
      } else {
        // SKIP-REASON: environment-dependent - requires local sandbox mode with touch controls
        test.skip();
      }
    });
  });

  test.describe('Hex Board Screenshots', () => {
    test('hex board initial state', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Hex Challenge/i }).click();

      // Wait for hex board to fully render
      await page.waitForTimeout(1000);

      // Capture the hex board
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('hex-board-initial.png');
    });

    test('hex board with valid targets', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Hex Challenge/i }).click();

      const validTargets = page
        .getByTestId('board-view')
        .locator('button[class*="outline-emerald"]');
      await expect(validTargets.first()).toBeVisible({ timeout: 25_000 });
      await page.waitForTimeout(500);

      // Capture hex board with highlighted cells
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('hex-board-with-targets.png');
    });
  });

  test.describe('19x19 Board Screenshots', () => {
    test('19x19 board initial state', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Full Board vs AI/i }).click();

      // Wait for large board to fully render
      await page.waitForTimeout(1500);

      // Capture the 19x19 board
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('board-19x19-initial.png');
    });
  });

  test.describe('Victory Modal Screenshots', () => {
    // SKIP-REASON: KEEP-SKIPPED - covered by scenario-driven E2E; keeping baseline suite backend-independent
    test.skip(
      'victory modal screenshots are covered by scenario-driven E2E runs; keeping the baseline suite backend-independent'
    );
  });
});

test.describe('Mobile Viewport Visual Tests', () => {
  // Use mobile viewport for all tests in this describe block
  test.use({ viewport: { width: 375, height: 667 } }); // iPhone SE

  test.setTimeout(120_000);

  test('entry route (guest) on mobile', async ({ page }) => {
    await page.goto('/');

    // Wait for page to load
    await expect(page.getByRole('heading', { name: /login/i })).toBeVisible({
      timeout: 10_000,
    });
    await page.waitForTimeout(500);

    // Capture mobile entry route (guests are redirected to login)
    await expect(page).toHaveScreenshot('mobile-entry-guest.png', {
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
    await goToSandbox(page);
    await page.getByRole('button', { name: /Learn the Basics/i }).click();
    await page.waitForTimeout(1000);

    // Capture the game board on mobile viewport
    await expect(page).toHaveScreenshot('mobile-game-board.png', {
      fullPage: true,
    });
  });

  test('sandbox page on mobile', async ({ page }) => {
    await goToSandbox(page);

    await expect(page.getByRole('heading', { name: /Start a Game \(Sandbox\)/i })).toBeVisible({
      timeout: 10_000,
    });
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
    await goToSandbox(page);
    await page.getByRole('button', { name: /Learn the Basics/i }).click();
    await page.waitForTimeout(1000);

    // Capture the game board on tablet viewport
    const boardView = page.getByTestId('board-view');
    await expect(boardView).toHaveScreenshot('tablet-game-board.png');
  });

  test('hex board on tablet', async ({ page }) => {
    await goToSandbox(page);
    await page.getByRole('button', { name: /Hex Challenge/i }).click();
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
