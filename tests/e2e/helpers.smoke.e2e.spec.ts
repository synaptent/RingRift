import { test, expect } from '@playwright/test';
import {
  generateTestUser,
  registerUser,
  loginUser,
  logout,
  registerAndLogin,
  createGame,
  waitForGameReady,
  waitForNetworkIdle,
  goToLobby,
  goToHome,
} from './helpers/test-utils';
import { LoginPage, RegisterPage, HomePage, GamePage } from './pages';

/**
 * E2E Smoke Tests for Test Helpers
 * ============================================================================
 *
 * This suite verifies that all E2E test helpers and page object models
 * are working correctly. It serves as a quick sanity check for the
 * testing infrastructure.
 *
 * RUN COMMAND: npm run test:e2e -- helpers.smoke.e2e.spec.ts
 */

test.describe('E2E Helpers Smoke Tests', () => {
  test.setTimeout(120_000);

  test.describe('Utility Helpers', () => {
    test('generateTestUser creates unique users', () => {
      const user1 = generateTestUser();
      const user2 = generateTestUser();

      // Both should have required fields
      expect(user1.username).toBeTruthy();
      expect(user1.email).toBeTruthy();
      expect(user1.password).toBeTruthy();

      // Should be unique
      expect(user1.email).not.toBe(user2.email);
      expect(user1.username).not.toBe(user2.username);

      // Email should have correct format
      expect(user1.email).toMatch(/^e2e\+\d+-\d+@example\.com$/);
    });
  });

  test.describe('Page Object Models', () => {
    test('LoginPage can navigate and verify elements', async ({ page }) => {
      const loginPage = new LoginPage(page);

      await loginPage.goto();
      await loginPage.waitForReady();

      // Verify all expected elements are present
      await expect(loginPage.heading).toBeVisible();
      await expect(loginPage.emailInput).toBeVisible();
      await expect(loginPage.passwordInput).toBeVisible();
      await expect(loginPage.loginButton).toBeVisible();
    });

    test('RegisterPage can navigate and verify elements', async ({ page }) => {
      const registerPage = new RegisterPage(page);

      await registerPage.goto();
      await registerPage.waitForReady();

      // Verify all expected elements are present
      await expect(registerPage.heading).toBeVisible();
      await expect(registerPage.emailInput).toBeVisible();
      await expect(registerPage.usernameInput).toBeVisible();
      await expect(registerPage.passwordInput).toBeVisible();
      await expect(registerPage.confirmPasswordInput).toBeVisible();
      await expect(registerPage.createAccountButton).toBeVisible();
    });

    test('Page objects can navigate between pages', async ({ page }) => {
      const loginPage = new LoginPage(page);
      const registerPage = new RegisterPage(page);

      // Start at login
      await loginPage.goto();
      await loginPage.waitForReady();

      // Navigate to register
      await loginPage.goToRegister();
      await registerPage.waitForReady();

      // Navigate back to login
      await registerPage.goToLogin();
      await loginPage.waitForReady();
    });
  });

  test.describe('Authentication Helpers', () => {
    test('registerAndLogin creates authenticated user', async ({ page }) => {
      const user = await registerAndLogin(page);

      // Should return valid user data
      expect(user.username).toBeTruthy();
      expect(user.email).toBeTruthy();
      expect(user.password).toBeTruthy();

      // Should be on home page and authenticated
      const homePage = new HomePage(page);
      await homePage.assertAuthenticated();
      await homePage.assertUsernameDisplayed(user.username);
    });

    test('logout redirects to login page', async ({ page }) => {
      await registerAndLogin(page);

      // Logout
      await logout(page);

      // Should be on login page
      const loginPage = new LoginPage(page);
      await loginPage.waitForReady();
    });
  });

  test.describe('Game Helpers', () => {
    test('createGame creates game and returns game ID', async ({ page }) => {
      await registerAndLogin(page);

      // Create a game
      const gameId = await createGame(page);

      // Should have a valid game ID
      expect(gameId).toBeTruthy();
      expect(typeof gameId).toBe('string');

      // Should be on game page
      expect(page.url()).toContain(`/game/${gameId}`);

      // GamePage should be ready
      const gamePage = new GamePage(page);
      await gamePage.waitForReady();
    });

    test('GamePage can interact with board', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page);

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Board should be visible
      await expect(gamePage.boardView).toBeVisible();

      // Should have cells
      const cellCount = await gamePage.getCellCount();
      expect(cellCount).toBeGreaterThan(0);

      // Should have valid targets during placement
      const targetCount = await gamePage.getValidTargetCount();
      expect(targetCount).toBeGreaterThanOrEqual(0); // May be 0 if AI goes first
    });

    test('waitForGameReady waits for all game elements', async ({ page }) => {
      await registerAndLogin(page);
      const gameId = await createGame(page);

      // Navigate away and back
      await page.goto('/');
      await page.goto(`/game/${gameId}`);

      // waitForGameReady should wait for all elements
      await waitForGameReady(page);

      const gamePage = new GamePage(page);
      await expect(gamePage.boardView).toBeVisible();
      await expect(gamePage.connectionStatus).toBeVisible();
    });
  });

  test.describe('Navigation Helpers', () => {
    test('goToLobby navigates to lobby after auth', async ({ page }) => {
      await registerAndLogin(page);
      await goToLobby(page);

      await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible();
    });

    test('goToHome navigates to home page', async ({ page }) => {
      // Navigate to login first
      await page.goto('/login');

      // Then to home (guest mode or after setup)
      await goToHome(page);

      // Home should show welcome message
      await expect(page.getByRole('heading', { name: /Welcome to RingRift/i })).toBeVisible();
    });
  });
});
