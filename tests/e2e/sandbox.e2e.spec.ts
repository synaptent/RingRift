import { test, expect } from '@playwright/test';
import { goToSandbox } from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * E2E Test Suite: Sandbox Host Flow
 * ============================================================================
 *
 * Verifies that the `/sandbox` route can launch a playable game by:
 * - Rendering the pre-game sandbox setup view.
 * - Navigating to a backend game when "Launch Game" is clicked.
 * - Showing a fully ready game view (board + HUD connection + turn indicator).
 *
 * RUN COMMAND: npm run test:e2e -- sandbox.e2e.spec.ts
 */

test.describe('Sandbox host E2E', () => {
  test.setTimeout(120_000);

  test('Launch Game from sandbox uses backend when available, otherwise falls back to local sandbox', async ({
    page,
  }) => {
    // Navigate to the sandbox pre-game setup.
    await goToSandbox(page);

    // Click the Launch Game button in the sandbox host.
    await page.getByRole('button', { name: /Launch Game/i }).click();

    // Prefer the happy path where the sandbox host creates a real backend game
    // and navigates to /game/:gameId on success, but tolerate environments
    // where the backend /api/games route is unavailable and the host falls
    // back to a purely local sandbox game.
    let navigatedToBackend = false;
    try {
      await page.waitForURL('**/game/**', { timeout: 20_000 });
      navigatedToBackend = true;
    } catch {
      // If we did not reach /game/:gameId within the timeout, treat this as a
      // backend creation failure and expect the local sandbox fallback path.
    }

    if (navigatedToBackend) {
      const gamePage = new GamePage(page);
      await gamePage.waitForReady(30_000);

      // Sanity-check core backend game UI elements.
      await expect(gamePage.boardView).toBeVisible();
      await gamePage.assertConnected();
      await expect(gamePage.turnIndicator).toBeVisible();
    } else {
      // Fallback: remain on /sandbox with a local ClientSandboxEngine-backed
      // game. We should still see a playable board and the sandbox touch
      // controls panel.
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });
      await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible({ timeout: 15_000 });
    }
  });

  test('Launch Game from sandbox creates a real backend game when API is healthy', async ({
    page,
  }) => {
    // This test is stricter than the fallback-tolerant variant above: in CI and
    // healthy dev environments we expect the sandbox host to create a backend
    // game and navigate to /game/:gameId. If the backend /api/games route is
    // unavailable, this test should fail rather than silently exercising the
    // local ClientSandboxEngine fallback.

    // Navigate to the sandbox pre-game setup.
    await goToSandbox(page);

    // Click the Launch Game button in the sandbox host.
    await page.getByRole('button', { name: /Launch Game/i }).click();

    // Require navigation to /game/:gameId.
    await page.waitForURL('**/game/**', { timeout: 30_000 });

    // Once on a backend game, assert that the game page is fully ready under
    // the standard backend host (board + connected HUD + turn indicator).
    const gamePage = new GamePage(page);
    await gamePage.waitForReady(30_000);

    await expect(gamePage.boardView).toBeVisible();
    await gamePage.assertConnected();
    await expect(gamePage.turnIndicator).toBeVisible();
    }
  });
});
