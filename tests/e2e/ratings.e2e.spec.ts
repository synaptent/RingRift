import { test, expect } from '@playwright/test';
import { registerAndLogin, createGame, generateTestUser } from './helpers/test-utils';
import { GamePage, HomePage } from './pages';

/**
 * E2E Test Suite: Ratings and Leaderboard
 * ============================================================================
 *
 * This suite tests rating and leaderboard functionality:
 * - Leaderboard page loading and display
 * - Rating display on profile page
 * - Initial rating for new users
 * - Rating updates after games (limited - requires completed games)
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for user and rating persistence)
 * - Redis running (for session management)
 * - Dev server running on http://localhost:5173
 *
 * RUN COMMAND: npx playwright test ratings.e2e.spec.ts
 */

test.describe('Rating and Leaderboard E2E Tests', () => {
  test.setTimeout(120_000);

  test.describe('Leaderboard Page', () => {
    test('leaderboard page loads and displays rankings', async ({ page }) => {
      await registerAndLogin(page);

      // Navigate to leaderboard
      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToLeaderboard();

      // Verify leaderboard heading
      await expect(page.getByRole('heading', { name: /leaderboard/i })).toBeVisible({
        timeout: 10_000,
      });

      // Verify table structure exists
      const table = page.locator('table');
      await expect(table).toBeVisible({ timeout: 10_000 });

      // Verify column headers
      await expect(page.locator('th', { hasText: /rank/i })).toBeVisible();
      await expect(page.locator('th', { hasText: /player/i })).toBeVisible();
      await expect(page.locator('th', { hasText: /rating/i })).toBeVisible();
    });

    test('leaderboard shows player statistics', async ({ page }) => {
      await registerAndLogin(page);

      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToLeaderboard();

      // Wait for table to load
      await expect(page.locator('table')).toBeVisible({ timeout: 10_000 });

      // Check for win rate column
      const winRateHeader = page.locator('th', { hasText: /win rate/i });
      await expect(winRateHeader).toBeVisible();

      // Check for games played column
      const gamesHeader = page.locator('th', { hasText: /games/i });
      await expect(gamesHeader).toBeVisible();
    });

    test('leaderboard displays rating values', async ({ page }) => {
      await registerAndLogin(page);

      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToLeaderboard();

      // Wait for table to load
      await expect(page.locator('table')).toBeVisible({ timeout: 10_000 });

      // If there are users with ratings, verify they're displayed as numbers
      const ratingCells = page.locator('td.font-mono');
      const count = await ratingCells.count();

      if (count > 0) {
        // Get first rating cell text and verify it's a number
        const firstRating = await ratingCells.first().textContent();
        if (firstRating) {
          expect(Number(firstRating.trim())).not.toBeNaN();
        }
      }
    });
  });

  test.describe('Profile Page Ratings', () => {
    test('profile page displays user rating', async ({ page }) => {
      const user = await registerAndLogin(page);

      // Navigate to profile
      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToProfile();

      // Wait for profile to load
      await page.waitForURL('**/profile', { timeout: 10_000 });

      // Verify rating is displayed
      const ratingSection = page.locator('text=/rating/i');
      await expect(ratingSection.first()).toBeVisible({ timeout: 10_000 });

      // Look for the rating value - should be a number (default is usually 1200 or 1500)
      const ratingValue = page.locator('.text-emerald-400, [class*="rating"]').filter({
        hasText: /\d+/,
      });
      await expect(ratingValue.first()).toBeVisible({ timeout: 5_000 });
    });

    test('new user starts with initial rating', async ({ page }) => {
      const user = await registerAndLogin(page);

      // Navigate to profile
      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToProfile();

      await page.waitForURL('**/profile', { timeout: 10_000 });

      // New users typically start with a default rating (commonly 1200 or 1500)
      // Look for rating display
      const ratingDisplay = page.locator('text=/\\d{3,4}/');
      await expect(ratingDisplay.first()).toBeVisible({ timeout: 5_000 });

      // Verify games played is 0 for new user
      const gamesPlayed = page.locator('text=/games played/i');
      await expect(gamesPlayed).toBeVisible();

      // The value should be 0
      const zeroGames = page.locator(':text("0")');
      await expect(zeroGames.first()).toBeVisible();
    });

    test('profile shows wins and losses statistics', async ({ page }) => {
      await registerAndLogin(page);

      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToProfile();

      await page.waitForURL('**/profile', { timeout: 10_000 });

      // Check for wins statistic
      const winsSection = page.locator('text=/wins/i');
      await expect(winsSection.first()).toBeVisible({ timeout: 5_000 });

      // Check for win rate statistic
      const winRateSection = page.locator('text=/win rate/i');
      await expect(winRateSection).toBeVisible();
    });

    test('profile displays recent games section', async ({ page }) => {
      await registerAndLogin(page);

      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToProfile();

      await page.waitForURL('**/profile', { timeout: 10_000 });

      // Look for recent games section
      const recentGamesHeader = page.getByRole('heading', { name: /recent games/i });
      await expect(recentGamesHeader).toBeVisible({ timeout: 5_000 });

      // For a new user, should show "No games played yet" or similar
      const noGamesMessage = page.locator('text=/no games|no matches/i');
      // Either there's a message or game entries - both are valid states
      const gameEntries = page.locator('[class*="game"]').filter({
        hasText: /victory|defeat|draw/i,
      });

      const hasMessage = (await noGamesMessage.count()) > 0;
      const hasGames = (await gameEntries.count()) > 0;

      // One of these should be true
      expect(hasMessage || hasGames).toBeTruthy();
    });
  });

  test.describe('Rating Updates', () => {
    test.skip('rating updates after completing a game', async ({ page }) => {
      // Skip: Requires completing a full game to observe rating changes
      // This is a complex end-to-end scenario

      await registerAndLogin(page);

      // Navigate to profile and record initial rating
      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToProfile();

      await page.waitForURL('**/profile', { timeout: 10_000 });

      // Get initial rating
      const initialRatingElement = page.locator('.text-emerald-400').first();
      const initialRatingText = await initialRatingElement.textContent();
      const initialRating = parseInt(initialRatingText || '1200', 10);

      // Create and complete a game
      await homePage.goto();
      await createGame(page, { vsAI: true });

      // ... game completion logic would go here ...
      // This requires the game to actually end (win/lose/draw)

      // Navigate back to profile
      await homePage.goto();
      await homePage.goToProfile();

      // Check if rating changed (it may not change for unrated games)
      const newRatingElement = page.locator('.text-emerald-400').first();
      await expect(newRatingElement).toBeVisible();
    });

    test.skip('rated games affect rating while unrated games do not', async ({ page }) => {
      // Skip: Requires both rated and unrated game completion to verify
      // Marking as skip until full game completion flow is testable

      await registerAndLogin(page);

      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToProfile();

      const initialRating = await page.locator('.text-emerald-400').first().textContent();

      // Create an unrated game and complete it
      // ... unrated game completion ...

      await homePage.goto();
      await homePage.goToProfile();

      const ratingAfterUnrated = await page.locator('.text-emerald-400').first().textContent();

      // Rating should not change for unrated games
      expect(ratingAfterUnrated).toBe(initialRating);
    });
  });
});
