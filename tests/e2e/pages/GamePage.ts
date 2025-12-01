/**
 * GamePage Page Object Model
 * ============================================================================
 *
 * Encapsulates all interactions with the game page (/game/:gameId).
 * Provides clean, reusable methods for game-related E2E tests.
 */

import { Page, Locator, expect } from '@playwright/test';

export class GamePage {
  readonly page: Page;

  // Locators for key elements
  readonly boardView: Locator;
  readonly connectionStatus: Locator;
  readonly turnIndicator: Locator;
  readonly gameLogSection: Locator;
  readonly recentMovesSection: Locator;
  readonly phaseIndicator: Locator;
  readonly currentPlayerIndicator: Locator;
  readonly chatInput: Locator;
  readonly chatSendButton: Locator;
  readonly chatMessages: Locator;

  constructor(page: Page) {
    this.page = page;

    // Initialize locators
    this.boardView = page.getByTestId('board-view');
    // Narrow connection status to the HUD label ("Connection: …") so that
    // Playwright strict mode does not match event-log entries like
    // "Connection restored".
    this.connectionStatus = page.getByTestId('game-hud').getByText('Connection:', { exact: false });
    // Scope the "Turn" label to the HUD container and require an exact text
    // match so that strict mode does not conflict with "Current Turn".
    this.turnIndicator = page.getByTestId('game-hud').getByText('Turn', { exact: true });
    this.gameLogSection = page.locator('text=/Game log/i');
    this.recentMovesSection = page.locator('text=/Recent moves/i');
    this.phaseIndicator = page.locator('text=/Phase/i');
    this.currentPlayerIndicator = page.locator('text=/Current player/i');
    this.chatInput = page.locator('input[placeholder*="Type a message"]');
    this.chatSendButton = page.locator('button:has-text("Send")');
    this.chatMessages = page.locator('.flex-1.overflow-y-auto');
  }

  /**
   * Navigate directly to a game by ID.
   */
  async goto(gameId: string): Promise<void> {
    await this.page.goto(`/game/${gameId}`);
  }

  /**
   * Wait for the game to be fully loaded and ready for interaction.
   */
  async waitForReady(timeout = 20_000): Promise<void> {
    await expect(this.boardView).toBeVisible({ timeout });
    // Require the HUD connection label to report "Connected" so that we
    // only proceed once the WebSocket has stabilised at least once.
    await expect(this.connectionStatus).toContainText('Connection: Connected', {
      timeout: 20_000,
    });
    await expect(this.turnIndicator).toBeVisible({ timeout: 10_000 });
  }

  /**
   * Check if the board is displayed.
   */
  async isBoardVisible(): Promise<boolean> {
    return this.boardView.isVisible();
  }

  /**
   * Get all cells on the board.
   */
  getCells(): Locator {
    return this.boardView.locator('button');
  }

  /**
   * Get the cell at a specific position.
   */
  getCell(x: number, y: number): Locator {
    return this.boardView.locator(`button[data-x="${x}"][data-y="${y}"]`);
  }

  /**
   * Get cells that are highlighted as valid targets.
   */
  getValidTargets(): Locator {
    return this.boardView.locator('button[class*="outline-emerald"]');
  }

  /**
   * Click a cell at the specified position.
   *
   * Uses a Locator-based click with an explicit attachment wait to make the
   * interaction resilient to React dev-mode re-renders that briefly detach
   * board cells from the DOM.
   */
  async clickCell(x: number, y: number): Promise<void> {
    const cell = this.getCell(x, y);
    await cell.waitFor({ state: 'attached', timeout: 10_000 });
    await cell.click();
  }

  /**
   * Click the first valid placement target.
   */
  async clickFirstValidTarget(): Promise<void> {
    const validTargets = this.getValidTargets();
    await validTargets.first().waitFor({ state: 'visible', timeout: 25_000 });
    await validTargets.first().click();
  }

  /**
   * Make a move from one position to another.
   */
  async makeMove(fromX: number, fromY: number, toX: number, toY: number): Promise<void> {
    await this.clickCell(fromX, fromY);
    await this.clickCell(toX, toY);
  }

  /**
   * Get the number of cells on the board.
   */
  async getCellCount(): Promise<number> {
    return this.getCells().count();
  }

  /**
   * Get the number of valid target cells.
   */
  async getValidTargetCount(): Promise<number> {
    const targets = this.getValidTargets();
    // Wait briefly for targets to appear
    try {
      await targets.first().waitFor({ state: 'visible', timeout: 5_000 });
    } catch {
      // No targets visible, return 0
      return 0;
    }
    return targets.count();
  }

  /**
   * Assert that valid placement targets are visible.
   */
  async assertValidTargetsVisible(): Promise<void> {
    const validTargets = this.getValidTargets();
    await expect(validTargets.first()).toBeVisible({ timeout: 25_000 });
  }

  /**
   * Assert that a move was logged.
   */
  async assertMoveLogged(pattern: RegExp | string): Promise<void> {
    await expect(this.recentMovesSection).toBeVisible({ timeout: 15_000 });
    const moveEntry = this.page.locator('li').filter({
      hasText: pattern instanceof RegExp ? pattern : new RegExp(pattern, 'i'),
    });
    await expect(moveEntry).toBeVisible({ timeout: 10_000 });
  }

  /**
   * Assert that a player's move is shown in the log.
   */
  async assertPlayerMoveLogged(playerNumber: number): Promise<void> {
    await this.assertMoveLogged(new RegExp(`P${playerNumber}`));
  }

  /**
   * Assert the current game phase.
   */
  async assertPhase(phase: string): Promise<void> {
    await expect(this.page.locator(`text=/${phase}/i`)).toBeVisible({ timeout: 5_000 });
  }

  /**
   * Assert which player's turn it is.
   */
  async assertCurrentPlayer(playerNumber: number): Promise<void> {
    await expect(this.page.locator(`text=/P${playerNumber}/i`)).toBeVisible({ timeout: 5_000 });
  }

  /**
   * Assert that the connection status shows connected.
   */
  async assertConnected(): Promise<void> {
    await expect(this.connectionStatus).toContainText('Connection: Connected', {
      timeout: 10_000,
    });
  }

  /**
   * Send a chat message.
   */
  async sendChatMessage(message: string): Promise<void> {
    await this.chatInput.fill(message);
    await this.chatSendButton.click();
  }

  /**
   * Get the current URL (useful for extracting game ID).
   */
  getUrl(): string {
    return this.page.url();
  }

  /**
   * Extract the game ID from the current URL.
   */
  getGameId(): string | null {
    const url = this.getUrl();
    const match = url.match(/\/game\/([a-zA-Z0-9-]+)/);
    return match ? match[1] : null;
  }

  /**
   * Navigate back to lobby.
   */
  async goToLobby(): Promise<void> {
    await this.page.getByRole('link', { name: /lobby/i }).click();
    await this.page.waitForURL('**/lobby', { timeout: 10_000 });
  }

  /**
   * Reload the page and wait for game to re-sync.
   */
  async reloadAndWait(): Promise<void> {
    const currentUrl = this.getUrl();
    await this.page.reload();
    await this.page.waitForURL(currentUrl, { timeout: 15_000 });
    await this.waitForReady();
  }
}
