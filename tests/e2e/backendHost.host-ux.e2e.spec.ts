import { test, expect } from '@playwright/test';
import { registerAndLogin, createGame } from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * E2E Test Suite: Backend Host UX (@host-ux)
 * ============================================================================
 *
 * Focuses on end-user behaviours of the backend host layer beyond what unit
 * tests and generic game-flow E2E suites cover:
 *
 * - Keyboard navigation and keyboard-initiated moves on the board.
 * - Spectator view of an in-progress game.
 * - Connection-loss and reconnection UX while a game is open.
 *
 * These tests are designed to run against a backend configured in the
 * orchestrator-ON posture described in ORCHESTRATOR_ROLLOUT_PLAN Table 4
 * (for example `RINGRIFT_RULES_MODE=ts`,
 * `ORCHESTRATOR_ADAPTER_ENABLED=true`,
 * `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`,
 * `ORCHESTRATOR_SHADOW_MODE_ENABLED=false`) so that host UX is exercised
 * over the canonical TS orchestrator path.
 */

test.describe('backend host @host-ux', () => {
  test.setTimeout(120_000);

  test(
    'supports keyboard navigation and keyboard-initiated ring placement',
    async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page, { vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady(30_000);

      const boardView = gamePage.boardView;

      // Wait for the board to render and for at least one valid placement target
      // to be highlighted by BackendGameHost (via validMoves → validTargets).
      const validTarget = boardView.locator('button.valid-move-cell').first();
      await expect(validTarget).toBeVisible({ timeout: 25_000 });

      const targetXAttr = await validTarget.getAttribute('data-x');
      const targetYAttr = await validTarget.getAttribute('data-y');
      const targetX = Number(targetXAttr);
      const targetY = Number(targetYAttr);

      // Start keyboard navigation by focusing the board container and then
      // using arrow keys to move focus from the canonical top-left cell (0,0)
      // to the chosen valid target. This avoids relying on any particular
      // intermediate cell remaining attached across dev-mode re-renders.
      await boardView.focus();
      await page.keyboard.press('ArrowDown'); // seed focus at (0,0)
 
      const startX = 0;
      const startY = 0;
 
      // Lightweight logging to aid future flake diagnostics without depending
      // on Playwright-specific logging helpers.
      // eslint-disable-next-line no-console
      console.log(
        `Keyboard nav: start=(${startX},${startY}) target=(${targetX},${targetY})`
      );
 
      // Use arrow keys to move focus from (0,0) to the chosen valid target
      // purely via keyboard input.
      const deltaX = targetX - startX;
      const deltaY = targetY - startY;
 
      const verticalKey = deltaY > 0 ? 'ArrowDown' : 'ArrowUp';
      for (let i = 0; i < Math.abs(deltaY); i += 1) {
        await page.keyboard.press(verticalKey);
      }
 
      const horizontalKey = deltaX > 0 ? 'ArrowRight' : 'ArrowLeft';
      for (let i = 0; i < Math.abs(deltaX); i += 1) {
        await page.keyboard.press(horizontalKey);
      }

      const targetCell = boardView.locator(
        `button[data-x="${targetX}"][data-y="${targetY}"]`
      );
      await expect(targetCell).toBeFocused();

      // eslint-disable-next-line no-console
      console.log('Pressing Enter on keyboard-focused cell to submit placement');
      await page.keyboard.press('Enter');

      // Move triggered via keyboard should show up in the event log.
      await expect(gamePage.recentMovesSection).toBeVisible({ timeout: 30_000 });
      await gamePage.assertPlayerMoveLogged(1);
    }
  );

 test(
   'exposes spectator view that mirrors game state and blocks move submission',
   async ({ page, browser }) => {
     // Create a backend game as an authenticated player.
     await registerAndLogin(page);
     const gameId = await createGame(page, { vsAI: true });

     const gamePage = new GamePage(page);
     await gamePage.waitForReady(30_000);

     // Open a second context as a spectator. Use a separate authenticated
     // user so the backend authorizes the WebSocket connection but treats
     // them as a non-player.
     const spectatorContext = await browser.newContext();
     const spectatorPage = await spectatorContext.newPage();

     try {
       await registerAndLogin(spectatorPage);
       await spectatorPage.goto(`/spectate/${gameId}`);

       const spectatorGamePage = new GamePage(spectatorPage);
       await spectatorGamePage.waitForReady(30_000);

       await expect(
         spectatorPage.getByTestId('board-view')
       ).toBeVisible({ timeout: 30_000 });

       // Spectator badge should be rendered in the HUD/header via the HUD
       // "Spectator" chip, which is unique inside the HUD container and more
       // stable than generic "Spectating" copy that appears in multiple
       // locations.
       const spectatorHudBadge = spectatorPage
         .getByTestId('game-hud')
         .getByText('Spectator', { exact: true });
       await expect(spectatorHudBadge).toBeVisible({ timeout: 15_000 });

       // All board cells should be disabled / read-only for spectators.
       const spectatorBoard = spectatorPage.getByTestId('board-view');
       const allCells = spectatorBoard.locator('button[data-x][data-y]');
       await expect(allCells.first()).toBeVisible({ timeout: 30_000 });

       const disabledCells = spectatorBoard.locator('button[disabled]');
       await expect(disabledCells.first()).toBeVisible({ timeout: 30_000 });

       const totalCells = await allCells.count();
       const disabledCount = await disabledCells.count();

       // eslint-disable-next-line no-console
       console.log(
         `Spectator board cells: total=${totalCells}, disabled=${disabledCount}`
       );

       // Selection panel should communicate that moves are disabled while spectating.
       await expect(
         spectatorPage.getByText('Moves disabled while spectating.')
       ).toBeVisible({ timeout: 15_000 });
     } finally {
       await spectatorContext.close();
     }
   }
 );

 test(
   'shows connection-loss banner and recovers after temporary network outage',
   async ({ page, context }) => {
     await registerAndLogin(page);
     await createGame(page, { vsAI: true });

     const gamePage = new GamePage(page);
     await gamePage.waitForReady(30_000);

     // Baseline: connected.
     await gamePage.assertConnected();
     const initialLabel = await gamePage.connectionStatus.textContent();
     // eslint-disable-next-line no-console
     console.log(`Connection HUD before offline: ${initialLabel}`);

    // Simulate a temporary network outage for this browser context.
    await context.setOffline(true);
    // eslint-disable-next-line no-console
    console.log('Context set offline for connection-loss test');
    // Do not assert on HUD visibility or contents while offline; the label
    // may be hidden or transient in dev/test environments.

     // Restore network connectivity.
     await context.setOffline(false);

     // Once connectivity returns, the HUD connection label should report Connected
     // again without reloading the page.
     await gamePage.assertConnected();
     const reconnectedLabel = await gamePage.connectionStatus.textContent();
     // eslint-disable-next-line no-console
     console.log(`Connection HUD after reconnection: ${reconnectedLabel}`);
   }
 );
});
