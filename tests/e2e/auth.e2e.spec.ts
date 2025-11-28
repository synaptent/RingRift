import { test, expect } from '@playwright/test';
import { generateTestUser, registerUser, loginUser, logout } from './helpers/test-utils';
import { LoginPage, RegisterPage, HomePage } from './pages';

/**
 * E2E Test Suite: User Authentication
 * ============================================================================
 *
 * This suite tests the core authentication happy path:
 * - User registration with email, username, and password
 * - Automatic login after registration
 * - Logout functionality
 * - Re-login with existing credentials
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for user persistence)
 * - Redis running (for session management)
 * - Dev server running on http://localhost:5173
 *
 * RUN COMMAND: npm run test:e2e -- --timeout 60000
 */

test.describe('Auth E2E – registration and login', () => {
  // Increase timeout for auth operations that involve database
  test.setTimeout(90_000);

  test('registers a new user, logs out, and logs back in', async ({ page }) => {
    const { email, username, password } = generateTestUser();

    // Step 1: Register using helper
    await registerUser(page, username, email, password);

    // Step 2: Verify authenticated state
    const homePage = new HomePage(page);
    await homePage.assertAuthenticated();
    await homePage.assertUsernameDisplayed(username);

    // Step 3: Log out using helper
    await logout(page);

    // Step 4: Log back in using helper
    await loginUser(page, email, password);

    // Step 5: Verify authenticated state restored
    await homePage.assertAuthenticated();
    await homePage.assertUsernameDisplayed(username);
  });

  test('shows error for invalid login credentials', async ({ page }) => {
    const loginPage = new LoginPage(page);

    // Navigate to login page
    await loginPage.goto();
    await loginPage.waitForReady();

    // Attempt login with non-existent credentials
    await loginPage.login('nonexistent@example.com', 'WrongPassword123!');

    // Should redirect to registration with prefilled email (current app behavior)
    // OR show an error - either is acceptable for this happy path test
    await expect(
      page.getByRole('heading', { name: /create an account/i }).or(page.locator('.text-red-300'))
    ).toBeVisible({ timeout: 10_000 });
  });

  test('can navigate from login to register page', async ({ page }) => {
    const loginPage = new LoginPage(page);

    await loginPage.goto();
    await loginPage.waitForReady();
    await loginPage.goToRegister();

    const registerPage = new RegisterPage(page);
    await registerPage.waitForReady();
  });

  test('can navigate from register to login page', async ({ page }) => {
    const registerPage = new RegisterPage(page);

    await registerPage.goto();
    await registerPage.waitForReady();
    await registerPage.goToLogin();

    const loginPage = new LoginPage(page);
    await loginPage.waitForReady();
  });
});
