import { test, expect } from '@playwright/test';
import {
  generateTestUser,
  registerAndLogin,
  createGame,
  waitForGameReady,
  clickValidPlacementTarget,
} from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * E2E Test Suite: AI Game Flow
 * ============================================================================
 *
 * This suite tests games against AI opponents:
 * - Creating and starting AI games
 * - Verifying AI makes moves automatically
 * - Testing game completion against AI
 * - AI difficulty levels
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for game persistence)
 * - Redis running (for WebSocket sessions and game state)
 * - AI service running (for AI moves)
 * - Dev server running on http://localhost:5173
 *
 * RUN COMMAND: npx playwright test ai-game.e2e.spec.ts
 */

test.describe('AI Game E2E Tests', () => {
  // Increase timeout for AI games which may take longer
  test.setTimeout(180_000);

  test('creates a game against AI opponent', async ({ page }) => {
    await registerAndLogin(page);
    const gameId = await createGame(page, { vsAI: true });

    const gamePage = new GamePage(page);

    // Verify we're in a game
    expect(gameId).toBeTruthy();
    await expect(gamePage.boardView).toBeVisible();

    // Verify turn indicator shows
    await expect(gamePage.turnIndicator).toBeVisible();

    // Verify connection is established
    await gamePage.assertConnected();
  });

  test('AI opponent makes moves automatically after human move', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page, { vsAI: true });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Make a human move during ring placement
    await gamePage.clickFirstValidTarget();

    // Wait for the game log to update - AI should respond
    await expect(gamePage.recentMovesSection).toBeVisible({ timeout: 30_000 });

    // After human move, AI should make a move - look for P2 (AI) move in log
    // Allow time for AI service to respond
    await expect(async () => {
      // Check for AI move (P2) in the recent moves
      const moveLog = page.locator('text=/P2/i');
      await expect(moveLog).toBeVisible({ timeout: 5_000 });
    }).toPass({ timeout: 60_000 });
  });

  test('game board updates after AI moves', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page, { vsAI: true });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Get initial cell count
    const initialCellCount = await gamePage.getCellCount();
    expect(initialCellCount).toBeGreaterThan(0);

    // Make a human move
    await gamePage.clickFirstValidTarget();

    // Wait for AI to respond and game state to update
    await page.waitForTimeout(5_000);

    // Board should still be visible and functional
    await expect(gamePage.boardView).toBeVisible();

    // Cell count should remain the same (same board)
    const postMoveCellCount = await gamePage.getCellCount();
    expect(postMoveCellCount).toBe(initialCellCount);
  });

  test('displays AI player information in game HUD', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page, { vsAI: true });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Look for AI indicator in the HUD or player list
    // AI players are typically labeled with 🤖 or "AI" badge
    const aiIndicator = page.locator('text=/AI|Computer|🤖/i');
    await expect(aiIndicator.first()).toBeVisible({ timeout: 15_000 });
  });

  test('human player can make multiple moves against AI', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page, { vsAI: true });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Make first human move
    await gamePage.clickFirstValidTarget();
    await expect(gamePage.recentMovesSection).toBeVisible({ timeout: 30_000 });

    // Wait for AI to respond
    await page.waitForTimeout(10_000);

    // Check if it's human's turn again
    const validTargetCount = await gamePage.getValidTargetCount();

    // If we have valid targets, make another move
    if (validTargetCount > 0) {
      await gamePage.clickFirstValidTarget();

      // Verify move was logged
      await expect(gamePage.recentMovesSection).toBeVisible();
    }

    // Game should still be functional
    await expect(gamePage.boardView).toBeVisible();
  });

  test('game shows correct phase during AI game', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page, { vsAI: true });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Should start in ring_placement phase
    await expect(page.locator('text=/ring_placement|Ring Placement|placement/i')).toBeVisible({
      timeout: 10_000,
    });
  });

  test('AI game maintains connection after multiple turns', async ({ page }) => {
    await registerAndLogin(page);
    await createGame(page, { vsAI: true });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Make a move
    await gamePage.clickFirstValidTarget();

    // Wait for AI response
    await page.waitForTimeout(10_000);

    // Verify connection is still active
    await gamePage.assertConnected();

    // Turn indicator should still work
    await expect(gamePage.turnIndicator).toBeVisible();
  });

  test.describe('AI Difficulty Tests', () => {
    /**
     * NOTE: These tests verify AI difficulty display when different
     * difficulty levels are configured. The actual AI behavior differences
     * require the AI service to be running with appropriate configurations.
     */

    test.skip('displays AI difficulty level in game info', async ({ page }) => {
      // Skip: Requires specific AI difficulty selection UI in game creation
      // which may not be fully implemented in the current lobby
      await registerAndLogin(page);
      await createGame(page, { vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Look for difficulty indicators like "Beginner", "Intermediate", "Advanced", "Expert"
      const difficultyIndicator = page.locator(
        'text=/Beginner|Intermediate|Advanced|Expert|Lv\\d/i'
      );
      await expect(difficultyIndicator.first()).toBeVisible({ timeout: 10_000 });
    });
  });
});
