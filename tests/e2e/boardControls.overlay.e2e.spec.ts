import { test, expect } from '@playwright/test';
import { createBackendGameFromLobby, goToSandbox } from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * E2E Test Suite: Board Controls & Shortcuts Overlay
 * ============================================================================
 *
 * Verifies the in-UI help/shortcuts overlay behaviour for:
 * - Backend host games (desktop).
 * - Sandbox host games (mobile-friendly touch UX).
 *
 * SELECTORS:
 * - data-testid="board-controls-button" – help entry point in HUD/header.
 * - data-testid="board-controls-overlay" – root of the overlay.
 * - data-testid="board-controls-basic-section" – basic mouse/touch controls.
 * - data-testid="board-controls-keyboard-section" – keyboard shortcuts (backend).
 * - data-testid="board-controls-sandbox-section" – sandbox touch controls copy.
 * - data-testid="board-controls-close-button" – close affordance in overlay.
 */

test.describe('Board controls & shortcuts overlay', () => {
  test.setTimeout(120_000);

  test.describe('Backend game (desktop)', () => {
    test('shows help overlay via HUD button and documents click + "?" shortcut', async ({ page }) => {
      await createBackendGameFromLobby(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady(30_000);

      // HUD help button should be present. Scope to the HUD container to
      // avoid strict mode conflicts with other test IDs.
      const helpButton = page.getByTestId('game-hud').getByTestId('board-controls-button');
      await expect(helpButton).toBeVisible();

      // Clicking the button opens the overlay.
      await helpButton.click();

      const overlay = page.getByTestId('board-controls-overlay');
      await expect(overlay).toBeVisible();

      // Basic mouse/touch controls section should mention click/tap semantics.
      await expect(
        page.getByTestId('board-controls-basic-section').getByText(/click or tap a cell/i)
      ).toBeVisible();

      // Keyboard shortcuts section should mention the "?" toggle shortcut.
      await expect(
        page
          .getByTestId('board-controls-keyboard-section')
          .getByText(/keyboard shortcuts/i)
      ).toBeVisible();
      await expect(
        page
          .getByTestId('board-controls-keyboard-section')
          .getByText(/\?/i)
      ).toBeVisible();

      // Pressing Escape should close the overlay.
      await page.keyboard.press('Escape');
      await expect(page.getByTestId('board-controls-overlay')).toBeHidden();
    });

    test('toggles overlay with "?" keyboard shortcut', async ({ page }) => {
      await createBackendGameFromLobby(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady(30_000);

      // Ensure overlay is initially closed.
      await expect(
        page.getByTestId('board-controls-overlay')
      ).toHaveCount(0);

      // Press Shift + "/" (rendered in UI as "?") to open overlay.
      await page.keyboard.press('Shift+/');

      await expect(
        page.getByTestId('board-controls-overlay')
      ).toBeVisible();

      // Press Shift + "/" again to toggle closed.
      await page.keyboard.press('Shift+/');

      await expect(
        page.getByTestId('board-controls-overlay')
      ).toHaveCount(0);
    });
  });

  test.describe('Sandbox game (mobile / touch-first)', () => {
    test('shows sandbox touch controls overlay and allows dismissal via close button', async ({
      page,
    }) => {
      // Navigate to sandbox setup.
      await goToSandbox(page);

      // Launch Game: this may fall back to local sandbox engine.
      await page.getByRole('button', { name: /Launch Game/i }).click();

      // Wait for sandbox board + touch controls to appear.
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });
      await expect(page.getByTestId('sandbox-touch-controls')).toBeVisible({ timeout: 15_000 });

      // Sandbox host should render the help button in its header.
      const helpButton = page.getByTestId('board-controls-button');
      await expect(helpButton).toBeVisible();

      // Open the overlay.
      await helpButton.click();

      const overlay = page.getByTestId('board-controls-overlay');
      await expect(overlay).toBeVisible();

      // Overlay should mention sandbox touch controls and key sandbox panel actions.
      await expect(
        page.getByRole('heading', { name: 'Sandbox touch controls', exact: true })
      ).toBeVisible();

      await expect(page.getByText(/Clear selection/i)).toBeVisible();
      await expect(page.getByText(/Finish move/i)).toBeVisible();
      await expect(page.getByText(/Show valid targets/i)).toBeVisible();
      await expect(page.getByText(/Show movement grid/i)).toBeVisible();

      // Dismiss via close button (touch-friendly).
      const closeButton = page.getByTestId('board-controls-close-button');
      await expect(closeButton).toBeVisible();
      await closeButton.click();

      await expect(
        page.getByTestId('board-controls-overlay')
      ).toHaveCount(0);
    });
  });
});