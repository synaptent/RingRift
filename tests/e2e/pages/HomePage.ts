/**
 * HomePage Page Object Model
 * ============================================================================
 *
 * Encapsulates all interactions with the home page (/).
 * Provides clean, reusable methods for navigation and verification.
 */

import { Page, Locator, expect } from '@playwright/test';

export class HomePage {
  readonly page: Page;

  // Locators for key elements
  readonly heading: Locator;
  readonly logoutButton: Locator;
  readonly lobbyLink: Locator;
  readonly sandboxLink: Locator;
  readonly leaderboardLink: Locator;
  readonly profileLink: Locator;
  readonly welcomeText: Locator;

  constructor(page: Page) {
    this.page = page;

    // Initialize locators
    this.heading = page.getByRole('heading', { name: /Welcome to RingRift/i });
    this.logoutButton = page.getByRole('button', { name: /logout/i });
    this.lobbyLink = page.getByRole('link', { name: /Enter Lobby|lobby/i });
    this.sandboxLink = page.getByRole('link', { name: /Open Local Sandbox|sandbox/i });
    this.leaderboardLink = page.getByRole('link', { name: /View Leaderboard|leaderboard/i });
    this.profileLink = page.getByRole('link', { name: /Profile|profile/i });
    this.welcomeText = page.locator("text=You're signed in");
  }

  /**
   * Navigate to the home page.
   */
  async goto(): Promise<void> {
    await this.page.goto('/');
  }

  /**
   * Wait for the home page to be fully loaded.
   */
  async waitForReady(): Promise<void> {
    await expect(this.heading).toBeVisible({ timeout: 10_000 });
  }

  /**
   * Check if user is authenticated by looking for logout button.
   */
  async isAuthenticated(): Promise<boolean> {
    return this.logoutButton.isVisible();
  }

  /**
   * Assert that the user is authenticated.
   */
  async assertAuthenticated(): Promise<void> {
    await expect(this.logoutButton).toBeVisible({ timeout: 10_000 });
  }

  /**
   * Assert that the user is not authenticated.
   */
  async assertNotAuthenticated(): Promise<void> {
    await expect(this.logoutButton).not.toBeVisible();
  }

  /**
   * Assert that the username is displayed.
   */
  async assertUsernameDisplayed(username: string): Promise<void> {
    await expect(this.page.getByText(username)).toBeVisible({ timeout: 10_000 });
  }

  /**
   * Perform logout.
   */
  async logout(): Promise<void> {
    await this.logoutButton.click();
    await this.page.waitForURL('**/login', { timeout: 10_000 });
  }

  /**
   * Navigate to lobby.
   */
  async goToLobby(): Promise<void> {
    await this.lobbyLink.click();
    await this.page.waitForURL('**/lobby', { timeout: 10_000 });
  }

  /**
   * Navigate to sandbox.
   */
  async goToSandbox(): Promise<void> {
    await this.sandboxLink.click();
    await this.page.waitForURL('**/sandbox', { timeout: 10_000 });
  }

  /**
   * Navigate to leaderboard.
   */
  async goToLeaderboard(): Promise<void> {
    await this.leaderboardLink.click();
    await this.page.waitForURL('**/leaderboard', { timeout: 10_000 });
  }

  /**
   * Navigate to profile.
   */
  async goToProfile(): Promise<void> {
    await this.profileLink.click();
    await this.page.waitForURL('**/profile', { timeout: 10_000 });
  }

  /**
   * Get the welcome message text.
   */
  async getWelcomeText(): Promise<string | null> {
    if (await this.welcomeText.isVisible()) {
      return this.welcomeText.textContent();
    }
    return null;
  }
}
