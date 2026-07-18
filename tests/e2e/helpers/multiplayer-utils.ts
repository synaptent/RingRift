/**
 * Multiplayer E2E Test Utilities
 * ============================================================================
 *
 * Advanced helpers for testing complex multiplayer scenarios including:
 * - WebSocket event coordination between multiple browser contexts
 * - Decision phase timeout testing
 * - Reconnection flow testing
 * - Concurrent player action coordination
 *
 * These utilities build on top of the base test-utils.ts helpers.
 */

import { Page, BrowserContext, expect } from '@playwright/test';
import type { Browser } from '@playwright/test';
import {
  generateTestUser,
  registerUser,
  waitForGameReady,
  createFixtureGame,
  type TestUser,
  type FixtureScenario,
  type CreateFixtureGameOptions,
} from './test-utils';
import { GamePage } from '../pages';

// ============================================================================
// Types
// ============================================================================

export interface PlayerContext {
  context: BrowserContext;
  page: Page;
  user: TestUser;
  gamePage: GamePage;
}

export interface MultiplayerGameSetup {
  player1: PlayerContext;
  player2: PlayerContext;
  gameId: string;
}

export interface WebSocketEventCapture {
  eventName: string;
  payload: unknown;
  timestamp: number;
}

export interface WaitForEventOptions {
  timeout?: number;
  /** Optional predicate to filter events */
  predicate?: (payload: unknown) => boolean;
}

// ============================================================================
// WebSocket Event Capturing
// ============================================================================

/**
 * Injects a WebSocket event listener into the page that captures specified events.
 * This allows E2E tests to verify WebSocket events are received correctly.
 *
 * @param page - Playwright page to inject listener into
 * @param eventNames - Array of WebSocket event names to capture
 * @returns Cleanup function to remove the listener
 *
 * @example
 * ```typescript
 * const cleanup = await injectWebSocketCapture(page, ['player_disconnected', 'player_reconnected']);
 * // ... perform actions ...
 * const events = await getWebSocketEvents(page);
 * await cleanup();
 * ```
 */
export async function injectWebSocketCapture(
  page: Page,
  eventNames: string[]
): Promise<() => Promise<void>> {
  await page.evaluate((events) => {
    // Create a global array to store captured events
    (window as any).__wsEventCapture = (window as any).__wsEventCapture || [];

    // Store the original socket.on if we need to restore it
    const gameContext = (window as any).__gameContext;
    if (gameContext?.socket) {
      events.forEach((eventName) => {
        gameContext.socket.on(eventName, (payload: unknown) => {
          (window as any).__wsEventCapture.push({
            eventName,
            payload,
            timestamp: Date.now(),
          });
        });
      });
    }
  }, eventNames);

  return async () => {
    await page.evaluate(() => {
      (window as any).__wsEventCapture = [];
    });
  };
}

/**
 * Retrieves captured WebSocket events from the page.
 */
export async function getWebSocketEvents(page: Page): Promise<WebSocketEventCapture[]> {
  return page.evaluate(() => {
    return (window as any).__wsEventCapture || [];
  });
}

/**
 * Waits for a specific WebSocket event to be received.
 * Uses polling since we can't directly hook into Socket.IO from Playwright.
 */
export async function waitForWebSocketEvent(
  page: Page,
  eventName: string,
  options: WaitForEventOptions = {}
): Promise<WebSocketEventCapture | null> {
  const { timeout = 30_000, predicate } = options;
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const events = await getWebSocketEvents(page);
    const matchingEvent = events.find((e) => {
      if (e.eventName !== eventName) return false;
      if (predicate && !predicate(e.payload)) return false;
      return true;
    });

    if (matchingEvent) {
      return matchingEvent;
    }

    await page.waitForTimeout(200);
  }

  return null;
}

// ============================================================================
// Alternative: Console-based Event Detection
// ============================================================================

/**
 * Sets up console log monitoring to detect WebSocket events.
 * This is more reliable than injecting into the Socket.IO instance.
 *
 * The approach:
 * 1. Add console.log statements to GameContext event handlers
 * 2. Monitor console output in Playwright
 * 3. Parse the logs to detect events
 *
 * This requires the app to log WebSocket events (which it does for debugging).
 */
export async function setupConsoleEventMonitor(
  page: Page
): Promise<{ getEvents: () => WebSocketEventCapture[]; cleanup: () => void }> {
  const capturedEvents: WebSocketEventCapture[] = [];

  const handler = (msg: any) => {
    const text = msg.text();
    // Look for WebSocket event patterns in console logs
    // Common patterns: "[WS] event_name", "Received event_name", etc.
    const wsEventMatch = text.match(/\[WS\]\s*(\w+)|WebSocket\s+event:\s*(\w+)/i);
    if (wsEventMatch) {
      capturedEvents.push({
        eventName: wsEventMatch[1] || wsEventMatch[2],
        payload: null,
        timestamp: Date.now(),
      });
    }
  };

  page.on('console', handler);

  return {
    getEvents: () => [...capturedEvents],
    cleanup: () => {
      page.off('console', handler);
    },
  };
}

// ============================================================================
// UI-Based Event Detection (Most Reliable)
// ============================================================================

/**
 * Waits for a player disconnection to be reflected in the UI.
 * This is the most reliable approach as it tests actual user-visible behavior.
 */
export async function waitForPlayerDisconnectedUI(
  page: Page,
  options: { timeout?: number } = {}
): Promise<boolean> {
  const { timeout = 15_000 } = options;
  try {
    // Look for disconnection indicators in the UI
    await expect(page.locator('text=/disconnected|offline|left.*game/i').first()).toBeVisible({
      timeout,
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Waits for a player reconnection to be reflected in the UI.
 */
export async function waitForPlayerReconnectedUI(
  page: Page,
  options: { timeout?: number } = {}
): Promise<boolean> {
  const { timeout = 15_000 } = options;
  try {
    // Look for reconnection indicators or connection restoration
    await expect(
      page.locator('text=/reconnected|back online|Connection: Connected/i').first()
    ).toBeVisible({ timeout });
    return true;
  } catch {
    return false;
  }
}

/**
 * Waits for a timeout warning to appear in the UI.
 */
export async function waitForTimeoutWarningUI(
  page: Page,
  options: { timeout?: number } = {}
): Promise<boolean> {
  const { timeout = 35_000 } = options;
  try {
    await expect(
      page.locator('text=/timeout|time.*running.*out|seconds.*remaining|auto-resolved/i').first()
    ).toBeVisible({ timeout });
    return true;
  } catch {
    return false;
  }
}

// ============================================================================
// Multiplayer Game Setup Helpers
// ============================================================================

/**
 * Creates a complete multiplayer game setup with two registered players.
 * Both players are connected to the same game and ready to play.
 *
 * @param browser - Playwright browser instance
 * @returns Setup with both player contexts and game ID
 */
export async function setupMultiplayerGameAdvanced(
  browser: Browser
): Promise<MultiplayerGameSetup> {
  // Create two independent browser contexts
  const context1 = await browser.newContext();
  const context2 = await browser.newContext();
  const page1 = await context1.newPage();
  const page2 = await context2.newPage();

  // Generate unique users
  const user1 = generateTestUser();
  const user2 = generateTestUser();

  // Register both players
  await registerUser(page1, user1.username, user1.email, user1.password);
  await registerUser(page2, user2.username, user2.email, user2.password);

  // Player 1 creates the game
  await page1.getByRole('link', { name: 'Lobby', exact: true }).click();
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

  // Wait for Player 1's game to be ready
  const p1GamePage = new GamePage(page1);
  await p1GamePage.waitForReady(30_000);

  // Extract game ID
  const url = page1.url();
  const match = url.match(/\/game\/([a-zA-Z0-9-]+)/);
  if (!match) {
    throw new Error(`Could not extract game ID from URL: ${url}`);
  }
  const gameId = match[1];

  // Player 2 joins the game
  await page2.goto(`/game/${gameId}`);
  const p2GamePage = new GamePage(page2);
  await p2GamePage.waitForReady(20_000);

  return {
    player1: { context: context1, page: page1, user: user1, gamePage: p1GamePage },
    player2: { context: context2, page: page2, user: user2, gamePage: p2GamePage },
    gameId,
  };
}

export interface SetupMultiplayerFixtureOptions {
  scenario: FixtureScenario;
  shortTimeoutMs?: number;
  shortWarningBeforeMs?: number;
  isRated?: boolean;
}

/**
 * Creates a multiplayer game using a fixture for a specific scenario.
 * This allows testing specific game states without playing through many moves.
 *
 * @param browser - Playwright browser instance
 * @param scenarioOrOptions - The fixture scenario (string) or full options object
 * @returns Setup with both player contexts and game ID
 */
export async function setupMultiplayerFixtureGame(
  browser: Browser,
  scenarioOrOptions: FixtureScenario | SetupMultiplayerFixtureOptions
): Promise<MultiplayerGameSetup> {
  const options: SetupMultiplayerFixtureOptions =
    typeof scenarioOrOptions === 'string' ? { scenario: scenarioOrOptions } : scenarioOrOptions;
  // Create two independent browser contexts
  const context1 = await browser.newContext();
  const context2 = await browser.newContext();
  const page1 = await context1.newPage();
  const page2 = await context2.newPage();

  // Generate unique users
  const user1 = generateTestUser();
  const user2 = generateTestUser();

  // Register both players
  await registerUser(page1, user1.username, user1.email, user1.password);
  await registerUser(page2, user2.username, user2.email, user2.password);

  // Player 1 creates a fixture game
  const { gameId } = await createFixtureGame(page1, {
    scenario: options.scenario,
    isRated: options.isRated ?? false,
    secondPlayerUsername: user2.username,
    ...(options.shortTimeoutMs !== undefined && { shortTimeoutMs: options.shortTimeoutMs }),
    ...(options.shortWarningBeforeMs !== undefined && {
      shortWarningBeforeMs: options.shortWarningBeforeMs,
    }),
  });

  // Both players navigate to the game
  await page1.goto(`/game/${gameId}`);
  await page2.goto(`/game/${gameId}`);

  const p1GamePage = new GamePage(page1);
  const p2GamePage = new GamePage(page2);

  await p1GamePage.waitForReady(20_000);
  await p2GamePage.waitForReady(20_000);

  return {
    player1: { context: context1, page: page1, user: user1, gamePage: p1GamePage },
    player2: { context: context2, page: page2, user: user2, gamePage: p2GamePage },
    gameId,
  };
}

/**
 * Cleans up a multiplayer game setup.
 */
export async function cleanupMultiplayerSetup(setup: MultiplayerGameSetup): Promise<void> {
  await setup.player1.context.close();
  await setup.player2.context.close();
}

// ============================================================================
// Turn Coordination Helpers
// ============================================================================

/**
 * Waits for it to be a specific player's turn, with visual confirmation.
 * More reliable than just checking text because it waits for valid targets.
 */
export async function waitForPlayerTurnWithTargets(
  playerContext: PlayerContext,
  timeout = 30_000
): Promise<boolean> {
  const { gamePage } = playerContext;
  try {
    await gamePage.assertValidTargetsVisible();
    return true;
  } catch {
    return false;
  }
}

/**
 * Coordinates a turn between two players where one makes a move
 * and the other waits for the update.
 */
export async function coordinatePlayerTurn(
  activePlayer: PlayerContext,
  waitingPlayer: PlayerContext,
  options: { waitAfterMove?: number } = {}
): Promise<void> {
  const { waitAfterMove = 2000 } = options;

  // Active player makes a move
  await activePlayer.gamePage.clickFirstValidTarget();

  // Wait for WebSocket propagation
  await waitingPlayer.page.waitForTimeout(waitAfterMove);

  // Verify waiting player sees the update
  await waitingPlayer.gamePage.assertRecentMovesVisible(10_000);
}

/**
 * Executes multiple turns in sequence, alternating between players.
 */
export async function executeAlternatingTurns(
  setup: MultiplayerGameSetup,
  turnCount: number,
  options: { delayBetweenTurns?: number } = {}
): Promise<number> {
  const { delayBetweenTurns = 2000 } = options;
  let successfulTurns = 0;

  for (let i = 0; i < turnCount; i++) {
    const activePlayer = i % 2 === 0 ? setup.player1 : setup.player2;
    const waitingPlayer = i % 2 === 0 ? setup.player2 : setup.player1;

    try {
      // Check if active player has valid targets
      const hasTargets = await waitForPlayerTurnWithTargets(activePlayer, 10_000);
      if (!hasTargets) {
        // Game may have ended or it's not their turn
        break;
      }

      await coordinatePlayerTurn(activePlayer, waitingPlayer);
      successfulTurns++;

      await activePlayer.page.waitForTimeout(delayBetweenTurns);
    } catch {
      // Turn failed, stop
      break;
    }
  }

  return successfulTurns;
}

// ============================================================================
// Disconnection/Reconnection Simulation
// ============================================================================

/**
 * Simulates a player disconnection by closing their page.
 * The context remains open so the player can reconnect.
 */
export async function simulateDisconnection(playerContext: PlayerContext): Promise<void> {
  await playerContext.page.close();
}

/**
 * Simulates a player reconnection by creating a new page in the same context
 * and navigating to the game.
 */
export async function simulateReconnection(
  playerContext: PlayerContext,
  gameId: string
): Promise<void> {
  // Create a new page in the same context (preserves session)
  const newPage = await playerContext.context.newPage();
  playerContext.page = newPage;
  playerContext.gamePage = new GamePage(newPage);

  // Navigate to the game
  await newPage.goto(`/game/${gameId}`);
  await playerContext.gamePage.waitForReady(20_000);
}

/**
 * Simulates a network interruption by blocking WebSocket connections.
 * Note: This requires route interception which may not work for WebSockets
 * in all browsers.
 */
export async function simulateNetworkInterruption(page: Page, durationMs: number): Promise<void> {
  // Block WebSocket connections
  await page.route('**/socket.io/**', (route) => route.abort());

  // Wait for the interruption duration
  await page.waitForTimeout(durationMs);

  // Restore connections
  await page.unroute('**/socket.io/**');
}

// ============================================================================
// Victory Detection Helpers
// ============================================================================

/**
 * Waits for victory modal to appear on a player's page.
 */
export async function waitForVictoryModal(
  playerContext: PlayerContext,
  options: { timeout?: number } = {}
): Promise<boolean> {
  const { timeout = 30_000 } = options;
  try {
    await expect(
      playerContext.page.locator('[data-testid="victory-modal"], .victory-modal')
    ).toBeVisible({ timeout });
    return true;
  } catch {
    return false;
  }
}

/**
 * Checks if a player won or lost by examining the victory modal.
 */
export async function getGameOutcome(
  playerContext: PlayerContext
): Promise<'victory' | 'defeat' | 'draw' | null> {
  const page = playerContext.page;

  const victoryText = await page.locator('text=/victory|you.*win|winner/i').count();
  if (victoryText > 0) return 'victory';

  const defeatText = await page.locator('text=/defeat|you.*lost|loser/i').count();
  if (defeatText > 0) return 'defeat';

  const drawText = await page.locator('text=/draw|tie/i').count();
  if (drawText > 0) return 'draw';

  return null;
}

// ============================================================================
// Decision Phase Testing Helpers
// ============================================================================

/**
 * Creates a game in a decision phase (line_processing, territory_processing, etc.)
 * and sets up both players to observe the decision.
 */
export async function setupDecisionPhaseGame(
  browser: Browser,
  scenario: 'line_processing' | 'territory_processing' | 'chain_capture_choice'
): Promise<MultiplayerGameSetup> {
  return setupMultiplayerFixtureGame(browser, scenario);
}

/**
 * Waits for a decision choice dialog to appear.
 */
export async function waitForChoiceDialog(
  playerContext: PlayerContext,
  options: { timeout?: number } = {}
): Promise<boolean> {
  const { timeout = 15_000 } = options;
  try {
    await expect(
      playerContext.page.locator('[data-testid="choice-dialog"], [role="dialog"]').first()
    ).toBeVisible({ timeout });
    return true;
  } catch {
    return false;
  }
}

/**
 * Selects an option in a choice dialog.
 */
export async function selectChoiceOption(
  playerContext: PlayerContext,
  optionIndex: number
): Promise<void> {
  const buttons = playerContext.page.locator(
    '[data-testid="choice-dialog"] button, [role="dialog"] button'
  );
  const button = buttons.nth(optionIndex);
  await button.click();
}
