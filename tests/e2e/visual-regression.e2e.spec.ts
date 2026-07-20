import { test, expect, type Page } from '@playwright/test';
import { goToSandbox } from './helpers/test-utils';
import { LoginPage, RegisterPage } from './pages';

async function mockSiteStats(page: Page): Promise<void> {
  await page.route('**/api/stats', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ playersOnline: 8, activeGames: 3, gamesPlayed: 1_248 }),
    });
  });
}

test.beforeEach(async ({ page }) => {
  // React Query devtools are development-only chrome, not part of the product
  // surface. Hide their asynchronously mounted launcher before every document
  // loads so snapshots cannot race its appearance.
  await page.addInitScript(() => {
    document.addEventListener(
      'DOMContentLoaded',
      () => {
        const style = document.createElement('style');
        style.textContent = '.tsqd-parent-container { display: none !important; }';
        document.head.append(style);
      },
      { once: true }
    );
  });
});

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

async function expectBoardToFitScalingWrapper(page: Page): Promise<void> {
  const board = page.getByTestId('board-view');
  const wrapper = board.locator('..');
  const [boardBox, wrapperBox] = await Promise.all([board.boundingBox(), wrapper.boundingBox()]);

  if (!boardBox || !wrapperBox) {
    throw new Error('Board or scaling wrapper did not produce a measurable bounding box');
  }

  expect(boardBox.x + boardBox.width).toBeLessThanOrEqual(wrapperBox.x + wrapperBox.width + 1);
  expect(boardBox.y + boardBox.height).toBeLessThanOrEqual(wrapperBox.y + wrapperBox.height + 1);
}

/**
 * Visual Regression Test Suite
 * ============================================================================
 *
 * This suite captures checked-in visual baselines for the public entry pages,
 * core game surfaces, supported board shapes, and responsive layouts. Every
 * appearance contract below uses an intentionally reviewed Chromium baseline.
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
      await mockSiteStats(page);
      await page.goto('/');

      // The public entry route is intentionally useful without authentication.
      await expect(page.getByRole('heading', { name: /RingRift/i }).first()).toBeVisible({
        timeout: 10_000,
      });
      await expect(page.getByRole('link', { name: /Play Now/i }).first()).toBeVisible();

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
      await expect(boardView).toHaveScreenshot('board-with-valid-targets.png');
    });

    test('game board after placing a ring', async ({ page }) => {
      await goToSandbox(page);
      await startSquare8Tutorial(page);

      await placeRingOnFirstEmptyCell(page);

      const boardView = page.getByTestId('board-view');
      await expect(boardView.getByRole('button', { name: /Stack height/i }).first()).toBeVisible();
      await expect(boardView).toHaveScreenshot('board-after-placement.png');
    });
  });

  test.describe('Component Screenshots', () => {
    test('game HUD appearance', async ({ page }) => {
      await page.clock.setFixedTime(new Date('2026-01-01T12:00:00.000Z'));
      await goToSandbox(page);
      await startSquare8Tutorial(page);

      const hudArea = page.getByTestId('game-hud');

      // Wait for HUD to be fully rendered
      await page.waitForTimeout(500);

      await expect(hudArea).toHaveScreenshot('game-hud.png');
    });

    test('game event log appearance', async ({ page }) => {
      await goToSandbox(page);
      await startSquare8Tutorial(page);

      // Make a move to populate the event log
      await placeRingOnFirstEmptyCell(page);

      // Open advanced panels to reveal the event log in sandbox mode.
      const advancedPanels = page.getByTestId('sandbox-advanced-sidebar-panels');
      const advancedPanelsOpen = await advancedPanels.evaluate(
        (element) => (element as HTMLDetailsElement).open
      );
      if (!advancedPanelsOpen) {
        await advancedPanels.locator('summary').click();
      }
      await expect(advancedPanels).toHaveAttribute('open', '', { timeout: 10_000 });

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

      await expect(page.getByRole('button', { name: /Launch(?: Local)? Game/i })).toBeVisible();
      await expect(page).toHaveScreenshot('sandbox-pregame-setup.png', {
        fullPage: true,
      });
    });

    test('sandbox game board after launch', async ({ page }) => {
      await goToSandbox(page);

      // Click a preset to launch a local sandbox game immediately.
      await startSquare8Tutorial(page);

      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });
      await page.waitForTimeout(1000);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('sandbox-local-board.png');
    });

    test('sandbox touch controls panel', async ({ page }) => {
      await goToSandbox(page);

      await startSquare8Tutorial(page);

      // Wait for board to be ready
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });

      const touchControls = page.getByTestId('sandbox-touch-controls');
      await expect(touchControls).toBeVisible({ timeout: 10_000 });
      await expect(touchControls).toHaveScreenshot('sandbox-touch-controls.png');
    });
  });

  test.describe('Hex Board Screenshots', () => {
    test('hex board initial state', async ({ page }) => {
      await goToSandbox(page, '/sandbox?preset=learn-basics-hex8');

      // Wait for hex board to fully render
      await page.waitForTimeout(1000);

      const boardView = page.getByTestId('board-view');
      await expect(boardView).toHaveScreenshot('hex-board-initial.png');
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
      await expect(boardView).toHaveScreenshot('hex-board-with-targets.png');
    });
  });

  test.describe('19x19 Board Screenshots', () => {
    test('19x19 board initial state', async ({ page }) => {
      await goToSandbox(page, '/sandbox?preset=sq19-1h-1ai');

      // Wait for large board to fully render
      await page.waitForTimeout(1500);

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
    await mockSiteStats(page);
    await page.goto('/');

    await expect(page.getByRole('heading', { name: /RingRift/i }).first()).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByRole('link', { name: /Play Now/i }).first()).toBeVisible();
    await expect(page).toHaveScreenshot('mobile-entry-guest.png', {
      fullPage: true,
    });
  });

  test('login page on mobile', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.waitForReady();
    await page.waitForTimeout(500);

    await expect(page.getByLabel('Email')).toBeVisible();
    await expect(page).toHaveScreenshot('mobile-login-page.png', {
      fullPage: true,
    });
  });

  test('game board on mobile', async ({ page }) => {
    await goToSandbox(page);
    await startSquare8Tutorial(page);
    await page.waitForTimeout(1000);

    await expect(page.getByTestId('board-view')).toBeVisible();
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible();
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

    await expect(page.getByRole('button', { name: /Launch(?: Local)? Game/i })).toBeVisible();
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
    await startSquare8Tutorial(page);
    await page.waitForTimeout(1000);
    await page.mouse.move(767, 1023);
    await page.waitForTimeout(250);

    const boardView = page.getByTestId('board-view');
    await expectBoardToFitScalingWrapper(page);
    await expect(boardView).toHaveScreenshot('tablet-game-board.png');
  });

  test('hex board on tablet', async ({ page }) => {
    await goToSandbox(page, '/sandbox?preset=learn-basics-hex8');
    await page.waitForTimeout(1000);
    await page.mouse.move(767, 1023);
    await page.waitForTimeout(250);

    const boardView = page.getByTestId('board-view');
    await expectBoardToFitScalingWrapper(page);
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
 *   npx playwright test visual-regression --project=chromium --update-snapshots
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
