/**
 * RegisterPage Page Object Model
 * ============================================================================
 *
 * Encapsulates all interactions with the registration page (/register).
 * Provides clean, reusable methods for registration-related E2E tests.
 */

import { Page, Locator, expect } from '@playwright/test';

export class RegisterPage {
  readonly page: Page;

  // Locators for key elements
  readonly heading: Locator;
  readonly emailInput: Locator;
  readonly usernameInput: Locator;
  readonly passwordInput: Locator;
  readonly confirmPasswordInput: Locator;
  readonly createAccountButton: Locator;
  readonly loginLink: Locator;
  readonly sandboxLink: Locator;
  readonly errorMessage: Locator;

  constructor(page: Page) {
    this.page = page;

    // Initialize locators
    this.heading = page.getByRole('heading', { name: /create an account/i });
    this.emailInput = page.getByLabel('Email');
    this.usernameInput = page.getByLabel('Username');
    this.passwordInput = page.getByLabel('Password', { exact: true });
    this.confirmPasswordInput = page.getByLabel('Confirm password');
    this.createAccountButton = page.getByRole('button', { name: /create account/i });
    this.loginLink = page.getByRole('link', { name: /log in/i });
    this.sandboxLink = page.getByRole('link', { name: /play local sandbox game/i });
    this.errorMessage = page.locator('.text-red-300');
  }

  /**
   * Navigate to the registration page.
   */
  async goto(): Promise<void> {
    await this.page.goto('/register');
  }

  /**
   * Wait for the registration page to be fully loaded.
   */
  async waitForReady(): Promise<void> {
    await expect(this.heading).toBeVisible();
    await expect(this.emailInput).toBeVisible();
    await expect(this.usernameInput).toBeVisible();
    await expect(this.passwordInput).toBeVisible();
    await expect(this.confirmPasswordInput).toBeVisible();
    await expect(this.createAccountButton).toBeVisible();
  }

  /**
   * Fill in the email field.
   */
  async fillEmail(email: string): Promise<void> {
    await this.emailInput.fill(email);
  }

  /**
   * Fill in the username field.
   */
  async fillUsername(username: string): Promise<void> {
    await this.usernameInput.fill(username);
  }

  /**
   * Fill in the password field.
   */
  async fillPassword(password: string): Promise<void> {
    await this.passwordInput.fill(password);
  }

  /**
   * Fill in the confirm password field.
   */
  async fillConfirmPassword(password: string): Promise<void> {
    await this.confirmPasswordInput.fill(password);
  }

  /**
   * Click the create account button.
   */
  async clickCreateAccount(): Promise<void> {
    await this.createAccountButton.click();
  }

  /**
   * Fill all registration fields.
   */
  async fillForm(email: string, username: string, password: string): Promise<void> {
    await this.fillEmail(email);
    await this.fillUsername(username);
    await this.fillPassword(password);
    await this.fillConfirmPassword(password);
  }

  /**
   * Perform a complete registration.
   */
  async register(email: string, username: string, password: string): Promise<void> {
    await this.fillForm(email, username, password);
    await this.clickCreateAccount();
  }

  /**
   * Register and wait for successful redirect to home.
   */
  async registerAndWaitForHome(email: string, username: string, password: string): Promise<void> {
    await this.register(email, username, password);
    await this.page.waitForURL('**/', { timeout: 30_000 });
    await expect(this.page.getByRole('button', { name: /logout/i })).toBeVisible({
      timeout: 10_000,
    });
  }

  /**
   * Navigate to login page.
   */
  async goToLogin(): Promise<void> {
    await this.loginLink.click();
    await this.page.waitForURL('**/login', { timeout: 10_000 });
  }

  /**
   * Navigate to sandbox game.
   */
  async goToSandbox(): Promise<void> {
    await this.sandboxLink.click();
    await this.page.waitForURL('**/sandbox', { timeout: 10_000 });
  }

  /**
   * Check if an error message is displayed.
   */
  async hasError(): Promise<boolean> {
    return this.errorMessage.isVisible();
  }

  /**
   * Get the error message text.
   */
  async getErrorMessage(): Promise<string | null> {
    if (await this.hasError()) {
      return this.errorMessage.textContent();
    }
    return null;
  }

  /**
   * Assert that an error message is displayed.
   */
  async assertError(expectedText?: string | RegExp): Promise<void> {
    await expect(this.errorMessage).toBeVisible({ timeout: 10_000 });
    if (expectedText) {
      await expect(this.errorMessage).toHaveText(expectedText);
    }
  }

  /**
   * Assert that no error message is displayed.
   */
  async assertNoError(): Promise<void> {
    await expect(this.errorMessage).toHaveCount(0);
  }
}
