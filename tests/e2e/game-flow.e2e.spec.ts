import { test, expect } from '@playwright/test';
import {
  generateTestUser,
  registerAndLogin,
  createGame,
  waitForGameReady,
  clickValidPlacementTarget,
  waitForMoveLog,
  goToLobby,
} from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * E2E Test Suite: Game Flow
 * ============================================================================
 *
 * This suite tests the core game functionality happy path:
 * - Creating a game from the lobby
 * - Game board rendering with correct data-testid selectors
 * - Making moves and seeing them logged
 * - State persistence after page reload
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for game persistence)
 * - Redis running (for WebSocket sessions and game state)
 * - Dev server running on http://localhost:5173
 * - Backend WebSocket server listening
 *
 * SELECTORS:
 * - data-testid="board-view" on the BoardView component
 * - Valid move targets have class "outline-emerald-300/90" (outline highlight)
 * - "Game log" heading in GameEventLog component
 * - "Recent moves" subheading when moves exist
 *
 * RUN COMMAND: npm run test:e2e -- --timeout 60000
 */

/**
 * Creates a backend AI game from the lobby and waits for the game page to load.
 * Returns the game URL for reference.
 */
async function createBackendGameFromLobby(page: import('@playwright/test').Page): Promise<string> {
  await registerAndLogin(page);
  await createGame(page);
  return page.url();
}

test.describe('Backend game flow E2E', () => {
  // Increase timeout for game operations that involve WebSocket and DB
  test.setTimeout(120_000);

  test('creates AI game from lobby and renders board + HUD', async ({ page }) => {
    await createBackendGameFromLobby(page);

    const gamePage = new GamePage(page);

    // Verify core game UI elements are present
    await expect(gamePage.boardView).toBeVisible();

    // GameHUD should show connection status (WebSocket connected indicator)
    await gamePage.assertConnected();

    // Turn indicator should be visible
    await expect(gamePage.turnIndicator).toBeVisible();

    // Game log section header
    await expect(gamePage.gameLogSection).toBeVisible();
  });

  test('game board has interactive cells during ring placement', async ({ page }) => {
    await createBackendGameFromLobby(page);

    const gamePage = new GamePage(page);

    // During ring placement phase, valid targets should be highlighted
    await expect(gamePage.boardView).toBeVisible();

    // Find cells that are clickable (all cells are buttons in BoardView)
    const cellCount = await gamePage.getCellCount();

    // Board should have cells (8x8=64 for default board type)
    expect(cellCount).toBeGreaterThan(0);
  });

  test('submits a ring placement move and logs it', async ({ page }) => {
    await createBackendGameFromLobby(page);

    const gamePage = new GamePage(page);

    // Wait for board to be fully ready and game to initialize
    await gamePage.waitForReady();

    // Click on a valid placement target (highlighted with outline-emerald)
    await gamePage.clickFirstValidTarget();

    // After making a move, the game log should update
    await expect(gamePage.gameLogSection).toBeVisible();

    // Wait for move to be logged - should show "Recent moves" section
    await expect(gamePage.recentMovesSection).toBeVisible({ timeout: 15_000 });

    // The move entry should mention P1 (player 1) for the first human move
    await gamePage.assertPlayerMoveLogged(1);
  });

  test('resyncs game state after full page reload', async ({ page }) => {
    const initialUrl = await createBackendGameFromLobby(page);

    const gamePage = new GamePage(page);

    // Verify initial state
    await expect(gamePage.boardView).toBeVisible();

    // Reload and wait for resync
    await gamePage.reloadAndWait();

    // Should return to the same game URL
    expect(page.url()).toBe(initialUrl);

    // Board should be visible after reload
    await expect(gamePage.boardView).toBeVisible();

    // HUD should show connection restored
    await gamePage.assertConnected();

    // Turn indicator should be restored
    await expect(gamePage.turnIndicator).toBeVisible();
  });

  test('can navigate back to lobby from game page', async ({ page }) => {
    await createBackendGameFromLobby(page);

    const gamePage = new GamePage(page);

    // Verify we're on a game page
    await expect(gamePage.boardView).toBeVisible();

    // Navigate back to lobby via page method
    await gamePage.goToLobby();

    // Lobby should show
    await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible();
  });

  test('displays correct game phase during ring placement', async ({ page }) => {
    await createBackendGameFromLobby(page);

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Should be in ring placement phase at start
    await expect(page.locator('text=/ring_placement|Ring Placement|placement/i')).toBeVisible({
      timeout: 10_000,
    });
  });
});
