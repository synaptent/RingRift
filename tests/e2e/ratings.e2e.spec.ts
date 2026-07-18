import { test, expect } from '@playwright/test';
import {
  registerAndLogin,
  createGame,
  generateTestUser,
  createFixtureGame,
  makeMove,
  waitForApiReady,
  waitForGameReady,
} from './helpers/test-utils';
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
    test('rating updates after completing a rated game', async ({ page }) => {
      // Uses near-victory fixture to fast-forward to a state where one
      // capture triggers elimination victory. The fixture creates a rated
      // game where Player 1 can win with a single move.
      await registerAndLogin(page);

      // Navigate to profile and record initial rating
      const homePage = new HomePage(page);
      await homePage.goto();
      await homePage.goToProfile();

      await page.waitForURL('**/profile', { timeout: 10_000 });

      // Get initial rating
      const initialRatingElement = page.getByTestId('profile-rating');
      const initialRatingText = await initialRatingElement.textContent();
      const initialRating = parseInt(initialRatingText || '1200', 10);
      expect(initialRating).toBeGreaterThan(0);

      // A rated result needs two persisted players. Register the fixture
      // opponent through the API without disturbing Player 1's browser auth.
      await waitForApiReady(page);
      const opponent = generateTestUser();
      const apiBaseUrl = (process.env.E2E_API_BASE_URL || 'http://localhost:3000').replace(
        /\/$/,
        ''
      );
      const registerOpponent = await page.request.post(`${apiBaseUrl}/api/auth/register`, {
        data: {
          username: opponent.username,
          email: opponent.email,
          password: opponent.password,
          confirmPassword: opponent.password,
        },
      });
      expect(registerOpponent.ok()).toBe(true);

      // Create a rated near-victory fixture game
      const { gameId } = await createFixtureGame(page, {
        scenario: 'near_victory_elimination',
        isRated: true,
        secondPlayerUsername: opponent.username,
      });

      await page.goto(`/game/${gameId}`);
      await waitForGameReady(page);

      // Make the winning marker landing: (3,3) -> (4,3)
      await makeMove(page, '3,3', '4,3');

      // Wait for victory modal to confirm game completed
      const victoryModal = page.locator('[data-testid="victory-modal"], .victory-modal');
      await expect(victoryModal).toBeVisible({ timeout: 30_000 });

      // Give the backend time to persist rating updates
      await page.waitForTimeout(2_000);

      // Navigate back to profile and verify rating changed
      await homePage.goto();
      await homePage.goToProfile();
      await page.waitForURL('**/profile', { timeout: 10_000 });

      const newRatingElement = page.getByTestId('profile-rating');
      await expect(newRatingElement).toBeVisible();
      const newRatingText = await newRatingElement.textContent();
      const newRating = parseInt(newRatingText || '0', 10);

      // Rating should have changed after winning a rated game
      expect(newRating).not.toBe(initialRating);
    });

    test('rated resignations affect rating while unrated resignations do not', async ({ page }) => {
      await registerAndLogin(page);

      const homePage = new HomePage(page);

      // Helper to read current rating from profile
      const readRating = async (): Promise<number> => {
        await homePage.goto();
        await homePage.goToProfile();
        await page.waitForURL('**/profile', { timeout: 10_000 });
        const ratingText = await page.getByTestId('profile-rating').textContent();
        return parseInt((ratingText || '').replace(/[^0-9]/g, ''), 10);
      };

      const initialRating = await readRating();
      expect(initialRating).toBeGreaterThan(0);

      // Create a rated game (default behaviour isRated=true for backend games)
      await homePage.goto();
      const ratedGame = await createGame(page, { vsAI: true });
      const ratedGameId = ratedGame.id;

      const ratedGamePage = new GamePage(page);
      await ratedGamePage.waitForReady();

      // Make at least one move so the game is clearly active
      await ratedGamePage.clickFirstValidTarget();

      // Resign via HTTP leave endpoint; this routes through GameSession and
      // RatingService.finishGame for rated games.
      await page.request.post(`/api/games/${ratedGameId}/leave`, {
        headers: {
          Authorization: `Bearer ${await page.evaluate(() => localStorage.getItem('token'))}`,
        },
      });

      // Give the backend a short window to persist rating updates
      await page.waitForTimeout(2_000);

      const afterRatedResignRating = await readRating();
      expect(afterRatedResignRating).not.toBeNaN();
      expect(afterRatedResignRating).not.toBe(initialRating);

      // Now create an unrated game and resign; rating should not change.
      await homePage.goto();
      const unratedGame = await createGame(page, { vsAI: true, isRated: false });
      const unratedGameId = unratedGame.id;

      const unratedGamePage = new GamePage(page);
      await unratedGamePage.waitForReady();
      await unratedGamePage.clickFirstValidTarget();

      await page.request.post(`/api/games/${unratedGameId}/leave`, {
        headers: {
          Authorization: `Bearer ${await page.evaluate(() => localStorage.getItem('token'))}`,
        },
      });

      await page.waitForTimeout(2_000);

      const afterUnratedResignRating = await readRating();
      expect(afterUnratedResignRating).toBe(afterRatedResignRating);
    });
  });
});
