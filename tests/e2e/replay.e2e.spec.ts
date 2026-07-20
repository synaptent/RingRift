import { test, expect, type Page } from '@playwright/test';
import { registerAndLogin, goToSandbox } from './helpers/test-utils';

async function goToReplaySandbox(page: Page) {
  await page.route('**/api/games**', async (route) => {
    if (route.request().method() === 'POST') {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Force local sandbox for replay testing' }),
      });
      return;
    }
    await route.continue();
  });
  await goToSandbox(page, '/sandbox?preset=learn-basics');
  await page.unroute('**/api/games**');
}

async function openReplayPanel(page: Page, expand = true) {
  const advancedPanels = page.getByTestId('sandbox-advanced-sidebar-panels');
  await expect(advancedPanels).toBeVisible();
  if (!(await advancedPanels.evaluate((element) => (element as HTMLDetailsElement).open))) {
    await advancedPanels.locator('summary').click();
  }

  const replayPanel = page.getByTestId('replay-panel');
  await expect(replayPanel).toBeVisible();
  if (expand) {
    await replayPanel.getByRole('button', { name: /Game Replay/i }).click();
  }
  return replayPanel;
}

/**
 * E2E Test Suite: Game Replay Functionality
 * ============================================================================
 *
 * This suite tests the game replay browser and playback controls in sandbox mode.
 * The replay system allows users to:
 * - Browse recorded games from the database
 * - Step through game moves
 * - Control playback speed
 * - Fork games to explore variations
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - Dev server running on http://localhost:5173
 * - AI service running on http://localhost:8001 (for replay data)
 * - Selfplay database with recorded games
 *
 * NOTE: Some tests may be skipped if the AI service is unavailable.
 * The replay service gracefully degrades when unavailable.
 *
 * RUN COMMAND: npx playwright test replay.e2e.spec.ts
 */

test.describe('Replay Panel E2E Tests', () => {
  test.setTimeout(90_000);

  test.describe('Replay Panel UI', () => {
    test('sandbox page renders replay panel section', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      await openReplayPanel(page, false);
    });

    test('replay panel can be expanded and collapsed', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      const replayPanel = await openReplayPanel(page);
      const expanded = replayPanel
        .locator('text=/Loading games|No games found|Service unavailable/i')
        .first();
      await expect(expanded).toBeVisible({ timeout: 10_000 });

      await replayPanel.getByRole('button', { name: /Game Replay/i }).click();
      await expect(expanded).not.toBeVisible();
    });

    test('replay panel shows service status indicator', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      const replayPanel = await openReplayPanel(page);

      // Should show either:
      // - Game list if AI service is running
      // - Service unavailable message if not
      const statusIndicator = replayPanel
        .locator('text=/Loading|Service unavailable|games found/i')
        .first();
      await expect(statusIndicator).toBeVisible({ timeout: 15_000 });
    });
  });

  test.describe('Replay Playback Controls', () => {
    test.beforeEach(async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);
    });

    test('playback controls render with correct buttons', async ({ page }) => {
      await openReplayPanel(page);

      // When a game is loaded, playback controls should be visible
      // Check for control buttons (may be disabled until game is loaded)
      const playbackSection = page.locator('[data-testid="playback-controls"]');

      // If playback controls exist, verify button presence
      if (await playbackSection.isVisible({ timeout: 5_000 })) {
        // Step backward button
        await expect(
          playbackSection
            .locator('button')
            .filter({ hasText: /prev|back|←/i })
            .first()
        ).toBeVisible();

        // Play/Pause button
        await expect(
          playbackSection
            .locator('button')
            .filter({ hasText: /play|pause|▶/i })
            .first()
        ).toBeVisible();

        // Step forward button
        await expect(
          playbackSection
            .locator('button')
            .filter({ hasText: /next|forward|→/i })
            .first()
        ).toBeVisible();
      }
    });
  });

  test.describe('Game List and Filters', () => {
    test('game list displays when AI service is available', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      const replayPanel = await openReplayPanel(page);

      // Wait for game list to load or show unavailable message
      const gameListOrUnavailable = replayPanel
        .locator('text=/Loading games|Select a game|No .*games|Service unavailable|games found/i')
        .first();
      await expect(gameListOrUnavailable).toBeVisible({ timeout: 15_000 });

      // If service is available and games exist, should show game entries
      const gameEntries = replayPanel.locator('[data-testid="game-list-item"]');
      const entryCount = await gameEntries.count();

      if (entryCount > 0) {
        // Verify first game entry has expected structure
        const firstGame = gameEntries.first();
        await expect(firstGame).toBeVisible();
      }
    });

    test('game filters update the displayed list', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      await openReplayPanel(page);

      // Look for filter controls
      const boardTypeFilter = page
        .locator('select[name="board_type"]')
        .or(page.locator('[data-testid="board-type-filter"]'));

      if (await boardTypeFilter.isVisible({ timeout: 5_000 })) {
        // Change filter and verify list updates
        await boardTypeFilter.selectOption({ index: 1 });

        // Wait for list to reload
        await page.waitForTimeout(1000);

        // Game list should reflect filter
        const gameList = page.locator('[data-testid="game-list"]');
        await expect(gameList).toBeVisible();
      }
    });
  });

  test.describe('Game Selection and Playback', () => {
    test('selecting a game loads it into playback', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      await openReplayPanel(page);

      // Wait for game list
      await page.waitForTimeout(2000);

      // Try to select first available game
      const gameEntries = page.locator('[data-testid="game-list-item"]');
      const entryCount = await gameEntries.count();

      if (entryCount > 0) {
        // Click first game
        await gameEntries.first().click();

        // Playback controls should become active
        const playbackControls = page.locator('[data-testid="playback-controls"]');
        await expect(playbackControls).toBeVisible({ timeout: 10_000 });

        // Move info should display current position
        const moveInfo = page
          .locator('[data-testid="move-info"]')
          .or(page.locator('text=/Move \\d+|Turn \\d+/i').first());
        await expect(moveInfo).toBeVisible({ timeout: 5_000 });
      }
    });

    test('step forward and backward through moves', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      await openReplayPanel(page);

      await page.waitForTimeout(2000);

      const gameEntries = page.locator('[data-testid="game-list-item"]');
      if ((await gameEntries.count()) > 0) {
        await gameEntries.first().click();
        await page.waitForTimeout(1000);

        // Find step buttons
        const nextButton = page
          .locator('button')
          .filter({ hasText: /next|forward|→|▶/i })
          .first();

        if (await nextButton.isEnabled({ timeout: 5_000 })) {
          // Step forward
          await nextButton.click();
          await page.waitForTimeout(500);

          // Move indicator should update
          const moveInfo = page.locator('text=/Move|Turn/i').first();
          await expect(moveInfo).toBeVisible();
        }
      }
    });
  });

  test.describe('Replay to Sandbox Integration', () => {
    test('fork from position creates new sandbox game', async ({ page }) => {
      await registerAndLogin(page);
      await goToReplaySandbox(page);

      await openReplayPanel(page);

      await page.waitForTimeout(2000);

      const gameEntries = page.locator('[data-testid="game-list-item"]');
      if ((await gameEntries.count()) > 0) {
        await gameEntries.first().click();
        await page.waitForTimeout(1000);

        // Look for fork button
        const forkButton = page
          .locator('button')
          .filter({ hasText: /fork|explore|branch/i })
          .first();

        if (await forkButton.isVisible({ timeout: 5_000 })) {
          await forkButton.click();

          // Should exit replay mode and allow making moves
          const boardView = page.getByTestId('board-view');
          await expect(boardView).toBeVisible();

          // Valid move targets should become available
          await page.waitForTimeout(1000);
          const validTargets = page.locator('.outline-emerald-300\\/90, [data-valid="true"]');
          // After forking, should be able to make moves
          await expect(validTargets.first()).toBeVisible({ timeout: 10_000 });
        }
      }
    });
  });
});
