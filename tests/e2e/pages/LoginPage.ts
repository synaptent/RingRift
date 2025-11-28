/**
 * LoginPage Page Object Model
 * ============================================================================
 *
 * Encapsulates all interactions with the login page (/login).
 * Provides clean, reusable methods for authentication-related E2E tests.
 */

import { Page, Locator, expect } from '@playwright/test';

export class LoginPage {
  readonly page: Page;

  // Locators for key elements
  readonly heading: Locator;
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly loginButton: Locator;
  readonly createAccountLink: Locator;
  readonly sandboxLink: Locator;
  readonly errorMessage: Locator;

  constructor(page: Page) {
    this.page = page;

    // Initialize locators
    this.heading = page.getByRole('heading', { name: /login/i });
    this.emailInput = page.getByLabel('Email');
    this.passwordInput = page.getByLabel('Password', { exact: true });
    this.loginButton = page.getByRole('button', { name: /login/i });
    this.createAccountLink = page.getByRole('link', { name: /create an account/i });
    this.sandboxLink = page.getByRole('link', { name: /play local sandbox game/i });
    this.errorMessage = page.locator('.text-red-300');
  }

  /**
   * Navigate to the login page.
   */
  async goto(): Promise<void> {
    await this.page.goto('/login');
  }

  /**
   * Wait for the login page to be fully loaded.
   */
  async waitForReady(): Promise<void> {
    await expect(this.heading).toBeVisible();
    await expect(this.emailInput).toBeVisible();
    await expect(this.passwordInput).toBeVisible();
    await expect(this.loginButton).toBeVisible();
  }

  /**
   * Fill in the email field.
   */
  async fillEmail(email: string): Promise<void> {
    await this.emailInput.fill(email);
  }

  /**
   * Fill in the password field.
   */
  async fillPassword(password: string): Promise<void> {
    await this.passwordInput.fill(password);
  }

  /**
   * Click the login button.
   */
  async clickLogin(): Promise<void> {
    await this.loginButton.click();
  }

  /**
   * Perform a complete login with email and password.
   */
  async login(email: string, password: string): Promise<void> {
    await this.fillEmail(email);
    await this.fillPassword(password);
    await this.clickLogin();
  }

  /**
   * Login and wait for successful redirect to home.
   */
  async loginAndWaitForHome(email: string, password: string): Promise<void> {
    await this.login(email, password);
    await this.page.waitForURL('**/', { timeout: 30_000 });
    await expect(this.page.getByRole('button', { name: /logout/i })).toBeVisible({
      timeout: 10_000,
    });
  }

  /**
   * Navigate to registration page.
   */
  async goToRegister(): Promise<void> {
    await this.createAccountLink.click();
    await this.page.waitForURL('**/register', { timeout: 10_000 });
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

  /**
   * Assert that the login button is disabled (e.g., during submission).
   */
  async assertLoginButtonDisabled(): Promise<void> {
    await expect(this.loginButton).toBeDisabled();
  }

  /**
   * Assert that the login button is enabled.
   */
  async assertLoginButtonEnabled(): Promise<void> {
    await expect(this.loginButton).toBeEnabled();
  }
}
