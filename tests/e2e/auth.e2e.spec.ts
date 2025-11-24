import { test, expect } from '@playwright/test';

function generateUserCredentials() {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1_000_000);
  const slug = `${timestamp}-${random}`;
  const email = `e2e+${slug}@example.com`;
  const username = `e2e-user-${slug}`;
  const password = 'E2E_test_password_123!';
  return { email, username, password };
}

test.describe('Auth E2E – registration and login', () => {
  test('registers a new user, logs out, and logs back in', async ({ page }) => {
    const { email, username, password } = generateUserCredentials();

    // Go directly to registration page
    await page.goto('/register');

    // Fill registration form
    await page.getByLabel('Email').fill(email);
    await page.getByLabel('Username').fill(username);
    await page.getByLabel('Password').fill(password);
    await page.getByLabel('Confirm password').fill(password);

    await page.getByRole('button', { name: /create account/i }).click();

    // Successful registration redirects into the authenticated shell at "/"
    await page.waitForURL('**/');

    // Authenticated navbar should show a Logout button and the username
    await expect(page.getByRole('button', { name: /logout/i })).toBeVisible();
    await expect(page.getByText(username)).toBeVisible();

    // Log out
    await page.getByRole('button', { name: /logout/i }).click();

    // After logout, unauthenticated routes redirect to /login
    await page.waitForURL('**/login');

    // Log back in with the same credentials
    await page.getByLabel('Email').fill(email);
    await page.getByLabel('Password').fill(password);

    await page.getByRole('button', { name: /login/i }).click();

    await page.waitForURL('**/');

    await expect(page.getByRole('button', { name: /logout/i })).toBeVisible();
    await expect(page.getByText(username)).toBeVisible();
  });
});
