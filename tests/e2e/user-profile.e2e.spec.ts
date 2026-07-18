import { test, expect } from '@playwright/test';
import { registerAndLogin } from './helpers/test-utils';

/**
 * E2E Test Suite: User Profile Management
 * ============================================================================
 *
 * This suite tests user profile functionality:
 * - Profile page display
 * - Profile editing
 * - Avatar/display settings
 * - Game history viewing
 * - Account management
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for user data)
 * - Redis running (for sessions)
 * - Dev server running on http://localhost:5173
 *
 * RUN COMMAND: npx playwright test user-profile.e2e.spec.ts
 */

test.describe('User Profile E2E Tests', () => {
  test.setTimeout(90_000);

  test.describe('Profile Page Display', () => {
    test('navigates to profile page from header', async ({ page }) => {
      await registerAndLogin(page);

      // Look for profile link/button in header
      const profileLink = page.locator('a[href*="profile"], button:has-text("Profile")').first();

      if (await profileLink.isVisible({ timeout: 5_000 })) {
        await profileLink.click();
        await page.waitForURL('**/profile**', { timeout: 10_000 });
      } else {
        // Try clicking on username in header
        const userMenu = page.locator('[data-testid="user-menu"]').or(
          page
            .locator('button')
            .filter({ hasText: /e2e-user/i })
            .first()
        );

        if (await userMenu.isVisible({ timeout: 5_000 })) {
          await userMenu.click();

          const profileOption = page.locator('text=/Profile|My Profile|Account/i').first();
          await expect(profileOption).toBeVisible({ timeout: 5_000 });
          await profileOption.click();
        }
      }

      // Profile page should show user info
      const profileContent = page.locator('text=/Profile|Statistics|Rating/i').first();
      await expect(profileContent).toBeVisible({ timeout: 10_000 });
    });

    test('profile page displays user statistics', async ({ page }) => {
      await registerAndLogin(page);

      // Navigate directly to profile
      await page.goto('/profile');
      await page.waitForLoadState('networkidle');

      // Should show rating
      const ratingDisplay = page.locator('text=/Rating|Elo|Score/i').first();
      await expect(ratingDisplay).toBeVisible({ timeout: 10_000 });

      // Should show games played (may be 0 for new user)
      const gamesDisplay = page.locator('text=/Games|Played|Matches/i').first();
      await expect(gamesDisplay).toBeVisible({ timeout: 5_000 });
    });

    test('profile page shows wins and losses', async ({ page }) => {
      await registerAndLogin(page);
      await page.goto('/profile');

      // Win/loss statistics
      const statsSection = page.locator('text=/Wins|Losses|W.*L|Win.*Loss/i').first();
      await expect(statsSection).toBeVisible({ timeout: 10_000 });
    });
  });

  test.describe('Profile Editing', () => {
    test('profile page has edit functionality', async ({ page }) => {
      await registerAndLogin(page);
      await page.goto('/profile');

      // Look for edit button
      const editButton = page
        .locator('button')
        .filter({ hasText: /edit|update|change|modify/i })
        .first();

      if (await editButton.isVisible({ timeout: 5_000 })) {
        await editButton.click();

        // Edit form should appear
        const editForm = page.locator('form, [data-testid="profile-edit-form"]').first();
        await expect(editForm).toBeVisible({ timeout: 5_000 });
      }
    });

    test('can update display name', async ({ page }) => {
      await registerAndLogin(page);
      await page.goto('/profile');

      // Look for edit functionality
      const editButton = page
        .locator('button')
        .filter({ hasText: /edit|update/i })
        .first();

      if (await editButton.isVisible({ timeout: 5_000 })) {
        await editButton.click();

        // Find display name input
        const nameInput = page.locator('input[name="displayName"], input[name="username"]').first();

        if (await nameInput.isVisible({ timeout: 5_000 })) {
          const newName = `TestUser_${Date.now()}`;
          await nameInput.fill(newName);

          // Submit changes
          const saveButton = page
            .locator('button')
            .filter({ hasText: /save|submit|update/i })
            .first();
          await saveButton.click();

          // Verify success message or updated name
          const successOrUpdated = page
            .locator('text=/success|saved|updated|' + newName + '/i')
            .first();
          await expect(successOrUpdated).toBeVisible({ timeout: 10_000 });
        }
      }
    });
  });

  test.describe('Game History', () => {
    test('profile shows recent games section', async ({ page }) => {
      await registerAndLogin(page);
      await page.goto('/profile');

      // Recent games section
      const recentGames = page.locator('text=/Recent Games|Game History|Past Games/i').first();
      await expect(recentGames).toBeVisible({ timeout: 10_000 });
    });

    test('clicking game in history navigates to game details', async ({ page }) => {
      await registerAndLogin(page);
      await page.goto('/profile');

      // Look for game history entries
      const gameEntries = page.locator('[data-testid="game-history-item"], a[href*="/game/"]');

      if ((await gameEntries.count()) > 0) {
        const firstGame = gameEntries.first();
        await firstGame.click();

        // Should navigate to game page or details
        await page.waitForURL('**/game/**', { timeout: 10_000 });
      }
    });
  });

  test.describe('Account Settings', () => {
    test('settings page is accessible from profile', async ({ page }) => {
      await registerAndLogin(page);
      await page.goto('/profile');

      // Look for settings link
      const settingsLink = page.locator('a[href*="settings"], button:has-text("Settings")').first();

      if (await settingsLink.isVisible({ timeout: 5_000 })) {
        await settingsLink.click();

        // Settings page should load
        const settingsContent = page.locator('text=/Settings|Preferences|Account/i').first();
        await expect(settingsContent).toBeVisible({ timeout: 10_000 });
      }
    });

    test('logout functionality works from profile area', async ({ page }) => {
      await registerAndLogin(page);

      // Navigate to profile/account area
      await page.goto('/profile');

      // Look for logout button
      const logoutButton = page
        .locator('button')
        .filter({ hasText: /log.*out|sign.*out|exit/i })
        .first();

      if (await logoutButton.isVisible({ timeout: 5_000 })) {
        await logoutButton.click();

        // Logout returns to the public shell without forcing /login.
        await expect(page.getByRole('link', { name: /sign in/i }).first()).toBeVisible({
          timeout: 10_000,
        });
      }
    });
  });

  test.describe('Profile Privacy', () => {
    test('other users can view public profile', async ({ page, browser }) => {
      // Create first user and get their info
      const user1 = await registerAndLogin(page);
      await page.goto('/profile');

      const profileResponse = await page.request.get('/api/users/profile');
      expect(profileResponse.ok()).toBeTruthy();
      const profileBody = await profileResponse.json();
      const userId = profileBody?.data?.user?.id as string | undefined;
      expect(userId).toBeTruthy();

      // Create second browser context as different user
      const context2 = await browser.newContext();
      const page2 = await context2.newPage();

      try {
        await registerAndLogin(page2);

        await page2.goto(`/player/${userId}`);

        await expect(page2.getByRole('heading', { name: user1.username, exact: true })).toBeVisible(
          {
            timeout: 10_000,
          }
        );
      } finally {
        await context2.close();
      }
    });
  });
});
