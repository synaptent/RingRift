import { test, expect } from '@playwright/test';

test.describe('Landing Page', () => {
  test('renders hero section with title and CTAs', async ({ page }) => {
    await page.goto('/');

    // Title visible
    await expect(page.locator('h1')).toContainText('RingRift');

    // Tagline visible
    await expect(page.getByText('multiplayer territory-control strategy game')).toBeVisible();

    // CTA buttons
    await expect(page.getByRole('link', { name: 'Play Now' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Sign In' })).toBeVisible();
  });

  test('Play Now links to sandbox', async ({ page }) => {
    await page.goto('/');

    const playNow = page.getByRole('link', { name: 'Play Now' });
    await expect(playNow).toHaveAttribute('href', /\/sandbox/);
  });

  test('Sign In links to login page', async ({ page }) => {
    await page.goto('/');

    const signIn = page.getByRole('link', { name: 'Sign In' });
    await expect(signIn).toHaveAttribute('href', '/login');
  });

  test('shows feature cards', async ({ page }) => {
    await page.goto('/');

    await expect(
      page.getByRole('heading', { name: 'Board Geometries', exact: true })
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Smart AI Opponents', exact: true })
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: '2-4 Player Multiplayer', exact: true })
    ).toBeVisible();
  });

  test('shows how-to-play section', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('heading', { name: 'Place Rings', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Form Lines', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Claim Territory', exact: true })).toBeVisible();
  });

  test('footer CTA links work', async ({ page }) => {
    await page.goto('/');

    // Scroll to bottom CTA
    const trySandbox = page.getByRole('link', { name: 'Try the Sandbox' });
    if (await trySandbox.isVisible()) {
      await expect(trySandbox).toHaveAttribute('href', /\/sandbox/);
    }
  });
});

test.describe('Navigation', () => {
  test('unauthenticated user sees landing page at /', async ({ page }) => {
    await page.goto('/');

    // Should see landing page, not login redirect
    await expect(page.locator('h1')).toContainText('RingRift');
    await expect(page.getByText('Play Now')).toBeVisible();
  });

  test('sandbox page loads without authentication', async ({ page }) => {
    await page.goto('/sandbox');

    // Sandbox should render (it's fully client-side)
    await expect(page).toHaveURL(/\/sandbox/);
  });

  test('login page loads', async ({ page }) => {
    await page.goto('/login');
    await expect(page).toHaveURL(/\/login/);
  });
});
