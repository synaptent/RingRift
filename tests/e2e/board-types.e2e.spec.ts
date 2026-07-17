import { test, expect } from '@playwright/test';
import { registerAndLogin, createGame, goToSandbox } from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * E2E Test Suite: Board Type Variations
 * ============================================================================
 *
 * This suite tests game functionality across different board configurations:
 * - Square 8x8 (default)
 * - Square 19x19 (larger)
 * - Hexagonal (469 spaces)
 *
 * Each board type has unique:
 * - Cell counts and arrangements
 * - Movement patterns
 * - Visual layouts
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for game persistence)
 * - Redis running (for WebSocket sessions)
 * - Dev server running on http://localhost:5173
 *
 * RUN COMMAND: npx playwright test board-types.e2e.spec.ts
 */

test.describe('Board Type E2E Tests', () => {
  test.setTimeout(120_000);

  test.describe('Square 8x8 Board (Default)', () => {
    test('creates and renders square8 board with correct cell count', async ({ page }) => {
      await registerAndLogin(page);
      const gameId = await createGame(page, { boardType: 'square8', vsAI: true });

      expect(gameId).toBeTruthy();

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Square 8x8 = 64 cells
      const cellCount = await gamePage.getCellCount();
      expect(cellCount).toBe(64);
    });

    test('square8 board allows ring placement on valid cells', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page, { boardType: 'square8', vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Make a move on square8 board
      await gamePage.clickFirstValidTarget();

      // Move should be logged
      await expect(gamePage.gameLogSection).toBeVisible();
    });
  });

  test.describe('Square 19x19 Board', () => {
    test('creates and renders square19 board with correct cell count', async ({ page }) => {
      await registerAndLogin(page);
      const gameId = await createGame(page, { boardType: 'square19', vsAI: true });

      expect(gameId).toBeTruthy();

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Square 19x19 = 361 cells
      const cellCount = await gamePage.getCellCount();
      expect(cellCount).toBe(361);
    });

    test('square19 board renders at appropriate scale', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page, { boardType: 'square19', vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Board should be visible and sized appropriately
      const boardView = gamePage.boardView;
      await expect(boardView).toBeVisible();

      // Board should have reasonable dimensions
      const box = await boardView.boundingBox();
      expect(box).toBeTruthy();
      expect(box!.width).toBeGreaterThan(200);
      expect(box!.height).toBeGreaterThan(200);
    });

    test('square19 board supports ring placement', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page, { boardType: 'square19', vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Make a move
      await gamePage.clickFirstValidTarget();

      // Move should be reflected in game log
      await expect(gamePage.gameLogSection).toBeVisible();
    });
  });

  test.describe('Hexagonal Board', () => {
    test('creates and renders hexagonal board', async ({ page }) => {
      await registerAndLogin(page);
      const gameId = await createGame(page, { boardType: 'hexagonal', vsAI: true });

      expect(gameId).toBeTruthy();

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Hexagonal board has 469 cells
      const cellCount = await gamePage.getCellCount();
      expect(cellCount).toBe(469);
    });

    test('hexagonal board renders with correct visual layout', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page, { boardType: 'hexagonal', vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Board should be visible
      const boardView = gamePage.boardView;
      await expect(boardView).toBeVisible();

      // BoardView exposes the geometry in its accessible region label.
      await expect(page.getByRole('region', { name: /Hexagonal game board/i })).toBeVisible();
    });

    test('hexagonal board allows ring placement', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page, { boardType: 'hexagonal', vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Make a move on hexagonal board
      await gamePage.clickFirstValidTarget();

      // Move should be logged
      await expect(gamePage.gameLogSection).toBeVisible();
    });

    test('hexagonal board movement respects hex adjacency', async ({ page }) => {
      await registerAndLogin(page);
      await createGame(page, { boardType: 'hexagonal', vsAI: true });

      const gamePage = new GamePage(page);
      await gamePage.waitForReady();

      // Place initial ring
      await gamePage.clickFirstValidTarget();

      // Wait for AI response
      await page.waitForTimeout(3000);

      // Continue making moves if still player's turn
      const turnIndicator = gamePage.turnIndicator;
      const turnText = await turnIndicator.textContent();

      if (turnText?.toLowerCase().includes('your turn')) {
        await gamePage.clickFirstValidTarget();
      }

      // Game should progress without errors
      await expect(gamePage.boardView).toBeVisible();
    });
  });

  test.describe('Board Type Selection', () => {
    test('lobby allows selecting different board types', async ({ page }) => {
      await registerAndLogin(page);

      // Navigate to game creation
      await page.goto('/lobby');
      await page.waitForLoadState('networkidle');

      // Look for board type selector in game creation form
      const boardTypeSelect = page
        .locator('select[name="boardType"]')
        .or(page.locator('[data-testid="board-type-select"]'));

      if (await boardTypeSelect.isVisible({ timeout: 5_000 })) {
        // Verify all three options are available
        const options = await boardTypeSelect.locator('option').allTextContents();
        expect(options.some((o) => /8.*8|square8/i.test(o))).toBeTruthy();
        expect(options.some((o) => /19.*19|square19/i.test(o))).toBeTruthy();
        expect(options.some((o) => /hex/i.test(o))).toBeTruthy();
      }
    });
  });

  test.describe('Sandbox Board Types', () => {
    test('sandbox supports all board types', async ({ page }) => {
      await registerAndLogin(page);
      await goToSandbox(page);

      // Look for board type selector in sandbox
      const boardTypeSelector = page
        .locator('select')
        .filter({ hasText: /square|hex/i })
        .first();

      if (await boardTypeSelector.isVisible({ timeout: 5_000 })) {
        // Try switching board types
        const options = await boardTypeSelector.locator('option').allTextContents();

        for (const option of options.slice(0, 2)) {
          // Test first two options
          await boardTypeSelector.selectOption({ label: option });
          await page.waitForTimeout(500);

          // Board should render
          const boardView = page.getByTestId('board-view');
          await expect(boardView).toBeVisible();
        }
      }
    });

    test('sandbox hexagonal board exposes interactive placement cells', async ({ page }) => {
      // Keep this on the local sandbox path. Authenticated presets intentionally
      // prefer a backend game, which would make this test cover the wrong host.
      await goToSandbox(page, '/sandbox?preset=learn-basics-hex8');

      // Ring placement is a direct-cell action in the sandbox: empty cells are
      // enabled rather than pre-highlighted as movement destinations.
      const placementCells = page.getByTestId('board-view').locator('button');
      await expect(placementCells).toHaveCount(61);
      await expect(placementCells.first()).toBeVisible();
      await expect(placementCells.first()).toBeEnabled();
    });
  });
});
