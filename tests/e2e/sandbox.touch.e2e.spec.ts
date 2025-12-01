import { test, expect } from '@playwright/test';
import { goToSandbox, assertNoErrors } from './helpers/test-utils';

/**
 * E2E Test Suite: Sandbox Touch-First Flows (Mobile)
 * ============================================================================
 *
 * Exercises the /sandbox local sandbox host using tap-first interactions:
 * - Starts from the sandbox setup view.
 * - Launches a local sandbox game (fallback when backend /games creation fails).
 * - Uses tap-only interactions to:
 *   - Place a ring during ring_placement.
 *   - Complete a simple movement/capture step by tapping a highlighted target.
 *
 * The test is intended to run under a mobile Playwright project, e.g.:
 *   npm run test:e2e -- --project="Mobile Chrome" sandbox.touch.e2e.spec.ts
 */

test.describe('Sandbox touch-first flows (mobile)', () => {
  test.setTimeout(120_000);

  test('tap to place and move in local sandbox with touch controls panel', async ({ page }) => {
    // Navigate to sandbox pre-game setup (unauthenticated guest).
    await goToSandbox(page);

    // Launch Game: for unauthenticated runs this typically fails backend
    // /games creation and falls back to local ClientSandboxEngine on /sandbox.
    await page.getByRole('button', { name: /Launch Game/i }).click();

    // We expect to remain on /sandbox with a local board; tolerate environments
    // where routing may briefly change but ensure the sandbox layout is present.
    await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible({ timeout: 10_000 });

    // Tap a central cell to attempt a placement (ring_placement phase).
    const boardView = page.getByTestId('board-view');
    const sourceCell = boardView.locator('button[data-x="3"][data-y="3"]');
    await sourceCell.click();

    // After placement, valid movement targets should be highlighted on the board.
    const validTargetSelector = '[data-testid="board-view"] button[class*="outline-emerald"]';
    await page.waitForSelector(validTargetSelector, {
      state: 'visible',
      timeout: 25_000,
    });

    // Tap one highlighted target to complete a simple move or capture segment.
    const targetCell = page.locator(validTargetSelector).first();
    await targetCell.click();

    // There should be no visible error banners after the interaction.
    await assertNoErrors(page);

    // Verify that at least one stack is present on the board (H* C* label),
    // indicating that a ring was placed and a follow-up move/capture completed.
    await expect(
      boardView.locator('text=/H[1-9][0-9]* C[1-9][0-9]*/')
    ).toBeVisible({ timeout: 15_000 });

    // Touch controls panel should still be visible and show a valid phase label.
    await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible();
    await expect(
      page.getByTestId('sandbox-touch-controls').getByText(/Phase:/i)
    ).toBeVisible();
  });
});