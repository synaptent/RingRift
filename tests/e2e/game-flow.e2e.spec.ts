import { test, expect } from '@playwright/test';

function generateUserCredentials() {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1_000_000);
  const slug = `${timestamp}-${random}`;
  const email = `e2e+${slug}@example.com`;
  const username = `e2e-game-${slug}`;
  const password = 'E2E_test_password_123!';
  return { email, username, password };
}

async function registerAndLogin(page: any) {
  const { email, username, password } = generateUserCredentials();

  await page.goto('/register');

  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Username').fill(username);
  await page.getByLabel('Password').fill(password);
  await page.getByLabel('Confirm password').fill(password);

  await page.getByRole('button', { name: /create account/i }).click();

  await page.waitForURL('**/');

  await page.getByText(username).waitFor();

  return { email, username, password };
}

async function createBackendGameFromLobby(page: any) {
  await registerAndLogin(page);

  await page.getByRole('link', { name: /lobby/i }).click();
  await page.waitForURL('**/lobby');

  await page.getByRole('heading', { name: /Game Lobby/i }).waitFor();

  await page.getByRole('button', { name: /\+ Create Game/i }).click();
  await page.getByRole('heading', { name: /Create Backend Game/i }).waitFor();

  await page.getByRole('button', { name: /^Create Game$/i }).click();

  await page.waitForURL('**/game/**');

  await expect(page.getByTestId('board-view')).toBeVisible();

  return page.url();
}

test.describe('Backend game flow E2E', () => {
  test('creates AI game from lobby and renders board + HUD', async ({ page }) => {
    await createBackendGameFromLobby(page);

    await expect(page.getByTestId('board-view')).toBeVisible();
    await expect(page.getByText(/Connection:/i)).toBeVisible();
    await expect(page.getByText(/Turn/i)).toBeVisible();
    await expect(page.getByText(/Game log/i)).toBeVisible();
  });

  test('submits a move and logs it in the game event log', async ({ page }) => {
    await createBackendGameFromLobby(page);

    const targetCell = page
      .locator('[data-testid="board-view"] button[class*="outline-emerald"]')
      .first();

    await targetCell.waitFor({ state: 'visible', timeout: 20_000 });
    await targetCell.click();

    await expect(page.getByText(/Recent moves/i)).toBeVisible();

    const moveEntry = page.locator('li').filter({ hasText: /P1/ }).first();

    await expect(moveEntry).toBeVisible();
  });

  test('resyncs game state after full page reload', async ({ page }) => {
    const initialUrl = await createBackendGameFromLobby(page);

    await expect(page.getByTestId('board-view')).toBeVisible();

    await page.reload();
    await page.waitForURL(initialUrl);

    await expect(page.getByTestId('board-view')).toBeVisible();
    await expect(page.getByText(/Connection:/i)).toBeVisible();
    await expect(page.getByText(/Turn/i)).toBeVisible();
  });
});
