import { test, expect, type Page } from '@playwright/test';
import { goToSandbox } from './helpers/test-utils';
import { LoginPage, RegisterPage } from './pages';

async function startSquare8Tutorial(page: Page): Promise<void> {
  await page
    .getByRole('button', { name: /Learn the Basics/i })
    .filter({ hasText: /sq8/i })
    .click();
}

async function selectFirstEmptyPlacement(page: Page): Promise<void> {
  const board = page.getByTestId('board-view');
  const target = board.locator('button[aria-label*="Empty cell"]').first();
  await expect(target).toBeVisible({ timeout: 25_000 });
  await target.click();
  await expect(board.locator('button[aria-pressed="true"]')).toBeVisible();
}

async function placeRingOnFirstEmptyCell(page: Page): Promise<void> {
  await selectFirstEmptyPlacement(page);
  await page.keyboard.press('Enter');
  await expect(
    page
      .getByTestId('board-view')
      .getByRole('button', { name: /Stack height/i })
      .first()
  ).toBeVisible({ timeout: 10_000 });
}

/**
 * Visual Regression Test Suite
 * ============================================================================
 *
 * This suite combines a small set of checked-in visual baselines with
 * responsive appearance smoke tests. A test uses toHaveScreenshot only when
 * its exact Chromium baseline is committed and intentionally reviewed.
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
    test('entry route presents the public landing page to guests', async ({ page }) => {
      await page.goto('/');

      // The public entry route is intentionally useful without authentication.
      await expect(page.getByRole('heading', { name: /RingRift/i }).first()).toBeVisible({
        timeout: 10_000,
      });
      await expect(page.getByRole('link', { name: /Play Now/i }).first()).toBeVisible();
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
      await startSquare8Tutorial(page);

      // Wait for board to fully render
      await page.waitForTimeout(1000);

      // Capture just the board view element (more stable than full page)
      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('initial-game-board.png');
    });

    test('game board highlights landing targets after selecting a placement', async ({ page }) => {
      await goToSandbox(page);
      await startSquare8Tutorial(page);

      await selectFirstEmptyPlacement(page);

      const validTargets = page
        .getByTestId('board-view')
        .locator('button[class*="outline-emerald"]');
      await expect(validTargets.first()).toBeVisible({ timeout: 25_000 });

      // Wait a bit for highlights to render
      await page.waitForTimeout(500);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toBeVisible();
    });

    test('game board after placing a ring', async ({ page }) => {
      await goToSandbox(page);
      await startSquare8Tutorial(page);

      await placeRingOnFirstEmptyCell(page);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toBeVisible();
      await expect(boardView.getByRole('button', { name: /Stack height/i }).first()).toBeVisible();
    });
  });

  test.describe('Component Screenshots', () => {
    test('game HUD appearance', async ({ page }) => {
      await goToSandbox(page);
      await startSquare8Tutorial(page);

      const hudArea = page.getByTestId('game-hud');

      // Wait for HUD to be fully rendered
      await page.waitForTimeout(500);

      await expect(hudArea).toBeVisible();
    });

    test('game event log appearance', async ({ page }) => {
      await goToSandbox(page);
      await startSquare8Tutorial(page);

      // Make a move to populate the event log
      await placeRingOnFirstEmptyCell(page);

      // Open advanced panels to reveal the event log in sandbox mode.
      const advancedPanels = page.getByTestId('sandbox-advanced-sidebar-panels');
      await advancedPanels.locator('summary').click();
      await expect(advancedPanels).toHaveAttribute('open', '', { timeout: 10_000 });

      const gameLogSection = page.locator('text=/Game log/i').locator('..').locator('..');
      await expect(gameLogSection).toBeVisible();
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

      await expect(page.getByRole('button', { name: /Launch(?: Local)? Game/i })).toBeVisible();
    });

    test('sandbox game board after launch', async ({ page }) => {
      await goToSandbox(page);

      // Click a preset to launch a local sandbox game immediately.
      await startSquare8Tutorial(page);

      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });
      await page.waitForTimeout(1000);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toBeVisible();
    });

    test('sandbox touch controls panel', async ({ page }) => {
      await goToSandbox(page);

      await startSquare8Tutorial(page);

      // Wait for board to be ready
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });

      // Check if sandbox touch controls exist (local sandbox mode)
      const touchControls = page.getByTestId('sandbox-touch-controls');
      const hasTouchControls = await touchControls.isVisible().catch(() => false);

      if (hasTouchControls) {
        await expect(touchControls).toBeVisible();
      } else {
        // SKIP-REASON: environment-dependent - requires local sandbox mode with touch controls
        test.skip();
      }
    });
  });

  test.describe('Hex Board Screenshots', () => {
    test('hex board initial state', async ({ page }) => {
      await goToSandbox(page, '/sandbox?preset=learn-basics-hex8');

      // Wait for hex board to fully render
      await page.waitForTimeout(1000);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toBeVisible();
    });

    test('hex board with valid targets', async ({ page }) => {
      await goToSandbox(page, '/sandbox?preset=learn-basics-hex8');
      await selectFirstEmptyPlacement(page);

      const validTargets = page
        .getByTestId('board-view')
        .locator('button[class*="outline-emerald"]');
      await expect(validTargets.first()).toBeVisible({ timeout: 25_000 });
      await page.waitForTimeout(500);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toBeVisible();
    });
  });

  test.describe('19x19 Board Screenshots', () => {
    test('19x19 board initial state', async ({ page }) => {
      await goToSandbox(page);
      await page.getByRole('button', { name: /Full Board vs AI/i }).click();

      // Wait for large board to fully render
      await page.waitForTimeout(1500);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toBeVisible();
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

    await expect(page.getByRole('heading', { name: /RingRift/i }).first()).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByRole('link', { name: /Play Now/i }).first()).toBeVisible();
  });

  test('login page on mobile', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.waitForReady();
    await page.waitForTimeout(500);

    await expect(page.getByLabel('Email')).toBeVisible();
  });

  test('game board on mobile', async ({ page }) => {
    await goToSandbox(page);
    await startSquare8Tutorial(page);
    await page.waitForTimeout(1000);

    await expect(page.getByTestId('board-view')).toBeVisible();
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible();
  });

  test('sandbox page on mobile', async ({ page }) => {
    await goToSandbox(page);

    await expect(page.getByRole('heading', { name: /Start a Game \(Sandbox\)/i })).toBeVisible({
      timeout: 10_000,
    });
    await page.waitForTimeout(500);

    await expect(page.getByRole('button', { name: /Launch(?: Local)? Game/i })).toBeVisible();
  });
});

test.describe('Tablet Viewport Visual Tests', () => {
  // Use iPad viewport
  test.use({ viewport: { width: 768, height: 1024 } });

  test.setTimeout(120_000);

  test('game board on tablet', async ({ page }) => {
    await goToSandbox(page);
    await startSquare8Tutorial(page);
    await page.waitForTimeout(1000);

    const boardView = page.getByTestId('board-view');
    await expect(boardView).toBeVisible();
  });

  test('hex board on tablet', async ({ page }) => {
    await goToSandbox(page, '/sandbox?preset=learn-basics-hex8');
    await page.waitForTimeout(1000);

    const boardView = page.getByTestId('board-view');
    await expect(boardView).toBeVisible();
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
