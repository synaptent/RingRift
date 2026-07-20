import { test, expect, devices } from '@playwright/test';

test.describe('Responsive Design', () => {
  test('landing page renders on mobile viewport', async ({ browser }) => {
    const context = await browser.newContext({
      ...devices['iPhone 12'],
    });
    const page = await context.newPage();

    await page.goto('/');

    // Hero content visible
    await expect(page.locator('h1')).toContainText('RingRift');
    await expect(page.getByRole('link', { name: 'Play Now' })).toBeVisible();

    // No horizontal overflow
    const body = page.locator('body');
    const bodyWidth = await body.evaluate((el) => el.scrollWidth);
    const viewportWidth = page.viewportSize()?.width ?? 0;
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 1);

    await context.close();
  });

  test('landing page renders on tablet viewport', async ({ browser }) => {
    const context = await browser.newContext({
      ...devices['iPad Mini'],
    });
    const page = await context.newPage();

    await page.goto('/');

    await expect(page.locator('h1')).toContainText('RingRift');
    await expect(
      page.getByRole('heading', { name: 'Board Geometries', exact: true })
    ).toBeVisible();

    await context.close();
  });
});
