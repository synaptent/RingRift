import { test, expect } from '@playwright/test';
import { goToSandbox, assertNoErrors } from './helpers/test-utils';

/**
 * E2E Test Suite: Sandbox Host UX (@host-ux)
 * ============================================================================
 *
 * Focuses on sandbox host behaviours that go beyond generic game-flow tests:
 *
 * - Backend creation attempt + local sandbox fallback.
 * - Human vs AI sandbox flow (local engine) with AI response.
 * - Touch controls panel overlay toggles (valid targets + movement grid).
 * - Stall warning and "Copy AI trace" diagnostics banner.
 *
 * These tests intentionally run mostly unauthenticated, since /sandbox is
 * public and designed for local experimentation.
 */

test.describe('sandbox host @host-ux', () => {
  test.setTimeout(120_000);

  test('falls back to local sandbox when backend /api/games creation fails', async ({ page }) => {
    // Force the backend /api/games route to fail so that the sandbox host must
    // initialise a ClientSandboxEngine-backed local game.
    await page.route('**/api/games**', (route) => {
      if (route.request().method() === 'POST') {
        return route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'E2E forced backend failure' }),
        });
      }
      return route.continue();
    });

    await goToSandbox(page);

    await page.getByRole('button', { name: /Launch Game/i }).click();

    // We should remain on /sandbox and see a local sandbox game view.
    await expect(page).toHaveURL(/\/sandbox/);

    await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible({ timeout: 15_000 });

    await assertNoErrors(page);

    await page.unroute('**/api/games**');
  });

  test('runs a short human vs AI sandbox game where AI responds after a human move', async ({
    page,
  }) => {
    // Ensure we always exercise the local sandbox path for determinism.
    await page.route('**/api/games**', (route) => {
      if (route.request().method() === 'POST') {
        return route.fulfill({
          status: 502,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'E2E forcing local sandbox fallback' }),
        });
      }
      return route.continue();
    });

    await goToSandbox(page);

    // Configure seats so that Player 1 is Human and Player 2 is AI.
    const player2Card = page.getByText('Player 2').locator('..').locator('..');
    await player2Card.getByRole('button', { name: /Computer/i }).click();

    await page.getByRole('button', { name: /Launch Game/i }).click();

    // Local sandbox game should start on /sandbox with board + touch controls.
    await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible({ timeout: 15_000 });

    const boardView = page.getByTestId('board-view');

    // Human makes a simple move by tapping any central cell then a highlighted target.
    const sourceCell = boardView.locator('button[data-x="3"][data-y="3"]').first();
    await sourceCell.click();

    const validTargetSelector = '[data-testid="board-view"] button[class*="outline-emerald"]';
    await page.waitForSelector(validTargetSelector, {
      state: 'visible',
      timeout: 25_000,
    });

    const firstTarget = page.locator(validTargetSelector).first();
    await firstTarget.click();

    // After the human move, the sandbox event log should show activity.
    await expect(page.locator('text=/Game log/i')).toBeVisible({ timeout: 30_000 });
    await expect(page.locator('text=/Recent moves/i')).toBeVisible({ timeout: 30_000 });

    // Allow time for the local AI (Player 2) to respond, then look for a P2
    // entry in the log, mirroring the backend AI E2E tests.
    await expect(async () => {
      const aiMoveEntry = page.locator('li').filter({ hasText: /P2/i });
      await expect(aiMoveEntry.first()).toBeVisible({ timeout: 5_000 });
    }).toPass({ timeout: 60_000 });

    await page.unroute('**/api/games**');
  });

  test('touch controls panel toggles valid targets and movement grid overlays', async ({ page }) => {
    // Force local sandbox path for deterministic overlays.
    await page.route('**/api/games**', (route) => {
      if (route.request().method() === 'POST') {
        return route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'E2E forced backend failure for overlays test' }),
        });
      }
      return route.continue();
    });

    await goToSandbox(page);
    await page.getByRole('button', { name: /Launch Game/i }).click();

    const board = page.getByTestId('board-view');

    await expect(board).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible({ timeout: 15_000 });

    // Click a cell to create a selection and valid targets.
    const sourceCell = board.locator('button[data-x="0"][data-y="0"]').first();
    await sourceCell.click();

    const validTargetSelector = '[data-testid="board-view"] button[class*="outline-emerald"]';
    await page.waitForSelector(validTargetSelector, {
      state: 'visible',
      timeout: 25_000,
    });

    // Movement grid overlay should be rendered initially.
    await expect(board.locator('svg')).toHaveCount(1);

    const validTargetsToggle = page.getByLabel('Show valid targets');
    const movementGridToggle = page.getByLabel('Show movement grid');

    await expect(validTargetsToggle).toBeChecked();
    await expect(movementGridToggle).toBeChecked();

    // Hide valid targets: highlight class should disappear but board remains.
    await validTargetsToggle.click();
    await expect(page.locator(validTargetSelector)).toHaveCount(0);

    // Hide movement grid: SVG overlay should be removed.
    await movementGridToggle.click();
    await expect(board.locator('svg')).toHaveCount(0);

    await page.unroute('**/api/games**');
  });

  test('shows stall warning banner and AI trace diagnostics via test-only helper', async ({ page }) => {
    // Ensure we are running against a local sandbox host.
    await page.route('**/api/games**', (route) => {
      if (route.request().method() === 'POST') {
        return route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'E2E forced fallback for stall diagnostics' }),
        });
      }
      return route.continue();
    });

    await goToSandbox(page);
    await page.getByRole('button', { name: /Launch Game/i }).click();

    await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible({ timeout: 15_000 });

    // Use the test-only helper exposed by SandboxContext in non-production
    // builds to seed a stall warning and AI trace payload.
    await page.evaluate(() => {
      const anyWindow = window as any;
      if (typeof anyWindow.__RINGRIFT_E2E_SET_SANDBOX_STALL__ !== 'function') {
        throw new Error('Sandbox stall E2E helper not available on window');
      }
      anyWindow.__RINGRIFT_E2E_SET_SANDBOX_STALL__('E2E sandbox stall banner', [
        { kind: 'stall', timestamp: Date.now(), details: 'e2e-stall' },
      ]);
    });

    const banner = page.getByText('E2E sandbox stall banner');
    await expect(banner).toBeVisible({ timeout: 10_000 });

    // The diagnostics banner should expose a "Copy AI trace" button.
    const copyButton = page.getByRole('button', { name: /Copy AI trace/i });
    await expect(copyButton).toBeVisible();
    await copyButton.click();

    // Dismiss the banner and ensure it disappears.
    const dismissButton = page.getByRole('button', { name: /Dismiss/i });
    await dismissButton.click();
    await expect(page.getByText('E2E sandbox stall banner')).toHaveCount(0);

    await page.unroute('**/api/games**');
  });
});