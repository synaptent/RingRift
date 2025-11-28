/**
 * E2E Test Utilities
 * ============================================================================
 *
 * Reusable helper functions for Playwright E2E tests.
 * These helpers abstract common operations like authentication, game creation,
 * and board interactions to make tests more readable and maintainable.
 */

import { Page, expect } from '@playwright/test';

// ============================================================================
// Types
// ============================================================================

export interface TestUser {
  username: string;
  email: string;
  password: string;
}

export interface CreateGameOptions {
  boardType?: 'square8' | 'square19' | 'hexagonal';
  vsAI?: boolean;
  maxPlayers?: number;
  isRated?: boolean;
}

// ============================================================================
// Utility Helpers
// ============================================================================

/**
 * Generates unique user credentials for test isolation.
 * Each test run creates distinct users to avoid conflicts between parallel runs.
 */
export function generateTestUser(): TestUser {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1_000_000);
  const slug = `${timestamp}-${random}`;
  return {
    email: `e2e+${slug}@example.com`,
    username: `e2e-user-${slug}`,
    password: 'E2E_test_password_123!',
  };
}

/**
 * Wait for network to be idle (no pending requests for a period).
 * Useful after navigation or form submissions.
 */
export async function waitForNetworkIdle(page: Page, timeout = 5000): Promise<void> {
  await page.waitForLoadState('networkidle', { timeout });
}

// ============================================================================
// Authentication Helpers
// ============================================================================

/**
 * Registers a new user via the registration form.
 * Waits for successful redirect to home page after registration.
 */
export async function registerUser(
  page: Page,
  username: string,
  email: string,
  password: string
): Promise<void> {
  await page.goto('/register');
  await expect(page.getByRole('heading', { name: /create an account/i })).toBeVisible();

  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Username').fill(username);
  await page.getByLabel('Password', { exact: true }).fill(password);
  await page.getByLabel('Confirm password').fill(password);

  await page.getByRole('button', { name: /create account/i }).click();

  // Wait for redirect to authenticated home page
  await page.waitForURL('**/', { timeout: 30_000 });
  await expect(page.getByRole('button', { name: /logout/i })).toBeVisible({ timeout: 10_000 });
}

/**
 * Logs in an existing user via the login form.
 * Waits for successful redirect to home page after login.
 */
export async function loginUser(page: Page, email: string, password: string): Promise<void> {
  await page.goto('/login');
  await expect(page.getByRole('heading', { name: /login/i })).toBeVisible();

  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Password', { exact: true }).fill(password);

  await page.getByRole('button', { name: /login/i }).click();

  // Wait for redirect to home
  await page.waitForURL('**/', { timeout: 30_000 });
  await expect(page.getByRole('button', { name: /logout/i })).toBeVisible({ timeout: 10_000 });
}

/**
 * Logs out the current user.
 * Waits for redirect to login page after logout.
 */
export async function logout(page: Page): Promise<void> {
  const logoutButton = page.getByRole('button', { name: /logout/i });
  await expect(logoutButton).toBeVisible({ timeout: 10_000 });
  await logoutButton.click();

  // After logout, should redirect to /login
  await page.waitForURL('**/login', { timeout: 10_000 });
}

/**
 * Registers a new user and ensures they're logged in.
 * Convenience function that combines generateTestUser and registerUser.
 * Returns the user credentials for reference.
 */
export async function registerAndLogin(page: Page): Promise<TestUser> {
  const user = generateTestUser();
  await registerUser(page, user.username, user.email, user.password);
  await expect(page.getByText(user.username)).toBeVisible({ timeout: 10_000 });
  return user;
}

// ============================================================================
// Game Connection Helpers
// ============================================================================

/**
 * Waits for WebSocket connection to be established.
 * Looks for connection status indicator in the game HUD.
 */
export async function waitForWebSocketConnection(page: Page, timeout = 15_000): Promise<void> {
  // The HUD displays "Connection:" followed by status
  await expect(page.locator('text=/Connection/i')).toBeVisible({ timeout });
}

/**
 * Waits for the game to be fully ready and interactive.
 * This includes board rendering and connection establishment.
 */
export async function waitForGameReady(page: Page, timeout = 20_000): Promise<void> {
  // Wait for board to be visible
  await expect(page.getByTestId('board-view')).toBeVisible({ timeout });

  // Wait for connection status
  await waitForWebSocketConnection(page, timeout);

  // Wait for turn indicator
  await expect(page.locator('text=/Turn/i')).toBeVisible({ timeout: 10_000 });
}

// ============================================================================
// Game Action Helpers
// ============================================================================

/**
 * Creates a new game from the lobby and navigates to it.
 * Returns the game ID extracted from the URL.
 */
export async function createGame(page: Page, options: CreateGameOptions = {}): Promise<string> {
  const { boardType = 'square8', vsAI = true } = options;

  // Navigate to lobby
  await page.getByRole('link', { name: /lobby/i }).click();
  await page.waitForURL('**/lobby', { timeout: 15_000 });

  // Verify lobby page loaded
  await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible({ timeout: 10_000 });

  // Open create game form
  await page.getByRole('button', { name: /\+ Create Game/i }).click();
  await expect(page.getByRole('heading', { name: /Create Backend Game/i })).toBeVisible({
    timeout: 5_000,
  });

  // Configure board type if not default
  if (boardType !== 'square8') {
    const boardSelect = page.locator('select').filter({ hasText: /8x8/ }).first();
    await boardSelect.selectOption(boardType);
  }

  // Submit game creation with default settings (human vs AI)
  await page.getByRole('button', { name: /^Create Game$/i }).click();

  // Wait for redirect to game page
  await page.waitForURL('**/game/**', { timeout: 30_000 });

  // Verify board is rendered
  await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 15_000 });

  // Extract game ID from URL
  const url = page.url();
  const match = url.match(/\/game\/([a-zA-Z0-9-]+)/);
  if (!match) {
    throw new Error(`Could not extract game ID from URL: ${url}`);
  }

  return match[1];
}

/**
 * Creates a backend game from the lobby after registering a user.
 * This is a convenience function that combines registration and game creation.
 * Returns the game URL.
 */
export async function createBackendGameFromLobby(page: Page): Promise<string> {
  await registerAndLogin(page);
  await createGame(page);
  return page.url();
}

/**
 * Joins an existing game by ID.
 */
export async function joinGame(page: Page, gameId: string): Promise<void> {
  await page.goto(`/game/${gameId}`);
  await waitForGameReady(page);
}

/**
 * Makes a move by clicking source and destination cells.
 * Used in movement and capture phases.
 *
 * @param from - Source position in "x,y" format (e.g., "3,4")
 * @param to - Destination position in "x,y" format (e.g., "5,4")
 */
export async function makeMove(page: Page, from: string, to: string): Promise<void> {
  const boardView = page.getByTestId('board-view');

  // Parse positions
  const [fromX, fromY] = from.split(',').map(Number);
  const [toX, toY] = to.split(',').map(Number);

  // Click source cell
  const sourceCell = boardView.locator(`button[data-x="${fromX}"][data-y="${fromY}"]`);
  await sourceCell.click();

  // Click destination cell
  const destCell = boardView.locator(`button[data-x="${toX}"][data-y="${toY}"]`);
  await destCell.click();
}

/**
 * Places a piece at the specified position during ring placement phase.
 * Clicks on a valid placement target highlighted on the board.
 *
 * @param position - Position in "x,y" format (e.g., "0,3")
 */
export async function placePiece(page: Page, position: string): Promise<void> {
  const boardView = page.getByTestId('board-view');
  const [x, y] = position.split(',').map(Number);

  // Find and click the cell at the specified position
  const cell = boardView.locator(`button[data-x="${x}"][data-y="${y}"]`);
  await cell.click();
}

/**
 * Clicks on a valid placement target (highlighted cell) during ring placement.
 * This is useful when you want to click any valid target without specifying position.
 */
export async function clickValidPlacementTarget(page: Page): Promise<void> {
  const validTargetSelector = '[data-testid="board-view"] button[class*="outline-emerald"]';

  // Wait for valid targets to appear
  await page.waitForSelector(validTargetSelector, {
    state: 'visible',
    timeout: 25_000,
  });

  const targetCell = page.locator(validTargetSelector).first();
  await targetCell.click();
}

// ============================================================================
// Board State Assertions
// ============================================================================

/**
 * Asserts the state of specific cells on the board.
 *
 * @param expectedCells - Map of position (e.g., "3,4") to expected state
 *                        States can be: "empty", "P1", "P2", etc.
 */
export async function assertBoardState(
  page: Page,
  expectedCells: Record<string, string>
): Promise<void> {
  const boardView = page.getByTestId('board-view');

  for (const [position, expectedState] of Object.entries(expectedCells)) {
    const [x, y] = position.split(',').map(Number);
    const cell = boardView.locator(`button[data-x="${x}"][data-y="${y}"]`);

    if (expectedState === 'empty') {
      // Check cell has no ring/stack indicators
      await expect(cell.locator('[data-player]')).toHaveCount(0);
    } else {
      // Check cell has expected player's piece
      const playerIndicator = cell.locator(`[data-player="${expectedState}"]`);
      await expect(playerIndicator).toBeVisible();
    }
  }
}

/**
 * Asserts which player's turn it is.
 *
 * @param playerNumber - Expected player number (1, 2, 3, or 4)
 */
export async function assertPlayerTurn(page: Page, playerNumber: number): Promise<void> {
  const turnIndicator = page.locator('text=/Turn/i');
  await expect(turnIndicator).toBeVisible();

  // Check for player number in the turn indicator or HUD
  const playerText = page.locator(`text=/P${playerNumber}/i`);
  await expect(playerText).toBeVisible({ timeout: 5_000 });
}

/**
 * Asserts the current game phase.
 *
 * @param phase - Expected phase name (e.g., "ring_placement", "movement", "capture")
 */
export async function assertGamePhase(page: Page, phase: string): Promise<void> {
  // The phase is displayed in the header or HUD
  const phaseText = page.locator(`text=/${phase}/i`);
  await expect(phaseText).toBeVisible({ timeout: 5_000 });
}

/**
 * Asserts that a move was logged in the game event log.
 *
 * @param movePattern - Regex pattern to match in the move log (e.g., /P1.*placed/)
 */
export async function assertMoveLogged(page: Page, movePattern: RegExp): Promise<void> {
  // Wait for "Recent moves" section to appear
  await expect(page.locator('text=/Recent moves/i')).toBeVisible({ timeout: 15_000 });

  // Check for the move pattern in the log
  const moveEntry = page.locator('li').filter({ hasText: movePattern });
  await expect(moveEntry).toBeVisible({ timeout: 10_000 });
}

/**
 * Waits for the game log to show recent moves.
 */
export async function waitForMoveLog(page: Page, timeout = 15_000): Promise<void> {
  await expect(page.locator('text=/Game log/i')).toBeVisible({ timeout });
  await expect(page.locator('text=/Recent moves/i')).toBeVisible({ timeout });
}

// ============================================================================
// Navigation Helpers
// ============================================================================

/**
 * Navigates to the lobby page.
 */
export async function goToLobby(page: Page): Promise<void> {
  await page.getByRole('link', { name: /lobby/i }).click();
  await page.waitForURL('**/lobby', { timeout: 10_000 });
  await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible();
}

/**
 * Navigates to a specific game by ID.
 */
export async function goToGame(page: Page, gameId: string): Promise<void> {
  await page.goto(`/game/${gameId}`);
  await waitForGameReady(page);
}

/**
 * Navigates to the home page.
 */
export async function goToHome(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /Welcome to RingRift/i })).toBeVisible({
    timeout: 10_000,
  });
}

// ============================================================================
// Error Handling Helpers
// ============================================================================

/**
 * Checks if an error message is displayed on the page.
 */
export async function assertErrorMessage(page: Page, errorPattern: RegExp): Promise<void> {
  const errorElement = page.locator('.text-red-300, .text-red-400, [class*="error"]');
  await expect(errorElement.filter({ hasText: errorPattern })).toBeVisible({ timeout: 10_000 });
}

/**
 * Checks that no error messages are displayed.
 */
export async function assertNoErrors(page: Page): Promise<void> {
  const errorElement = page.locator('.text-red-300, .text-red-400, [class*="error"]');
  await expect(errorElement).toHaveCount(0);
}

// ============================================================================
// Multi-Player Helpers (for multiple browser contexts)
// ============================================================================

import type { Browser, BrowserContext } from '@playwright/test';

/**
 * Context holder for a player in multiplayer tests.
 */
export interface PlayerContext {
  context: BrowserContext;
  page: Page;
  user: TestUser;
}

/**
 * Result of setting up a multiplayer game with two players.
 */
export interface MultiplayerGameSetup {
  player1: PlayerContext;
  player2: PlayerContext;
  gameId: string;
}

/**
 * Creates a new browser context with a fresh session.
 * Useful for testing multi-player scenarios.
 */
export async function createFreshContext(page: Page): Promise<BrowserContext> {
  const browser = page.context().browser();
  if (!browser) {
    throw new Error('Browser not available');
  }
  return browser.newContext();
}

/**
 * Creates a multiplayer game context with two registered players.
 * Player 1 creates the game, Player 2 joins it.
 *
 * @param browser - The Playwright Browser instance
 * @returns Promise containing both player contexts and the game ID
 *
 * @example
 * ```typescript
 * test('multiplayer scenario', async ({ browser }) => {
 *   const { player1, player2, gameId } = await setupMultiplayerGame(browser);
 *   // Both players are now in the same game
 *   await player1.page.close();
 *   await player2.page.close();
 * });
 * ```
 */
export async function setupMultiplayerGame(browser: Browser): Promise<MultiplayerGameSetup> {
  // Create two independent browser contexts
  const context1 = await browser.newContext();
  const context2 = await browser.newContext();
  const page1 = await context1.newPage();
  const page2 = await context2.newPage();

  // Generate unique users
  const user1 = generateTestUser();
  const user2 = generateTestUser();

  // Register and login both players
  await registerUser(page1, user1.username, user1.email, user1.password);
  await registerUser(page2, user2.username, user2.email, user2.password);

  // Player 1 creates a game
  await page1.getByRole('link', { name: /lobby/i }).click();
  await page1.waitForURL('**/lobby', { timeout: 15_000 });
  await expect(page1.getByRole('heading', { name: /Game Lobby/i })).toBeVisible({
    timeout: 10_000,
  });

  await page1.getByRole('button', { name: /\+ Create Game/i }).click();
  await expect(page1.getByRole('heading', { name: /Create Backend Game/i })).toBeVisible({
    timeout: 5_000,
  });

  await page1.getByRole('button', { name: /^Create Game$/i }).click();
  await page1.waitForURL('**/game/**', { timeout: 30_000 });
  await expect(page1.getByTestId('board-view')).toBeVisible({ timeout: 15_000 });

  // Extract game ID
  const url = page1.url();
  const match = url.match(/\/game\/([a-zA-Z0-9-]+)/);
  if (!match) {
    throw new Error(`Could not extract game ID from URL: ${url}`);
  }
  const gameId = match[1];

  // Player 2 joins the game
  await page2.goto(`/game/${gameId}`);
  await expect(page2.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });
  await expect(page2.locator('text=/Connection/i')).toBeVisible({ timeout: 15_000 });

  return {
    player1: { context: context1, page: page1, user: user1 },
    player2: { context: context2, page: page2, user: user2 },
    gameId,
  };
}

/**
 * Coordinates a turn between two players in a multiplayer game.
 * The current player makes a move, then we wait for the waiting player
 * to see the update via WebSocket.
 *
 * @param currentPlayer - The page of the player making the move
 * @param waitingPlayer - The page of the player waiting for the move
 * @param move - The move to make (from/to positions in "x,y" format)
 *
 * @example
 * ```typescript
 * await coordinateTurn(player1Page, player2Page, { from: '3,4', to: '5,4' });
 * ```
 */
export async function coordinateTurn(
  currentPlayer: Page,
  waitingPlayer: Page,
  move: { from: string; to: string }
): Promise<void> {
  const boardView = currentPlayer.getByTestId('board-view');

  // Parse positions
  const [fromX, fromY] = move.from.split(',').map(Number);
  const [toX, toY] = move.to.split(',').map(Number);

  // Make the move on current player's page
  const sourceCell = boardView.locator(`button[data-x="${fromX}"][data-y="${fromY}"]`);
  await sourceCell.click();

  const destCell = boardView.locator(`button[data-x="${toX}"][data-y="${toY}"]`);
  await destCell.click();

  // Wait for WebSocket to propagate the move to the waiting player
  // We look for the move to appear in their game log
  await waitingPlayer.waitForTimeout(2000); // Initial delay for WebSocket sync
  await expect(waitingPlayer.locator('text=/Recent moves/i')).toBeVisible({ timeout: 15_000 });
}

/**
 * Waits for a specific player's turn on a page.
 *
 * @param page - The page to check
 * @param playerNumber - The player number (1, 2, 3, or 4)
 * @param timeout - Maximum time to wait in milliseconds
 */
export async function waitForTurn(
  page: Page,
  playerNumber: number,
  timeout = 30_000
): Promise<void> {
  await expect(page.locator(`text=/P${playerNumber}/i`)).toBeVisible({ timeout });
}

/**
 * Checks if it's currently a specific player's turn.
 *
 * @param page - The page to check
 * @param playerNumber - The player number to check for
 * @returns true if it's that player's turn
 */
export async function isPlayerTurn(page: Page, playerNumber: number): Promise<boolean> {
  try {
    const turnIndicator = page.locator(`text=/P${playerNumber}/i`);
    return await turnIndicator.isVisible();
  } catch {
    return false;
  }
}

/**
 * Makes a ring placement move by clicking first valid target.
 * Useful during ring_placement phase of multiplayer games.
 *
 * @param page - The page to make the placement on
 * @param timeout - Maximum time to wait for valid targets
 */
export async function makeRingPlacement(page: Page, timeout = 25_000): Promise<void> {
  const validTargetSelector = '[data-testid="board-view"] button[class*="outline-emerald"]';

  await page.waitForSelector(validTargetSelector, {
    state: 'visible',
    timeout,
  });

  const targetCell = page.locator(validTargetSelector).first();
  await targetCell.click();
}

/**
 * Cleans up multiplayer game contexts.
 * Call this in afterEach to properly close browser contexts.
 *
 * @param setup - The multiplayer game setup to clean up
 */
export async function cleanupMultiplayerGame(setup: MultiplayerGameSetup): Promise<void> {
  await setup.player1.context.close();
  await setup.player2.context.close();
}
