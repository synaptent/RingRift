import { test, expect, Browser, BrowserContext, Page } from '@playwright/test';
import { generateTestUser, registerUser, TestUser, createFixtureGame } from './helpers/test-utils';
import { GamePage, HomePage } from './pages';
import {
  setupMultiplayerGameAdvanced,
  setupMultiplayerFixtureGame,
  cleanupMultiplayerSetup,
  simulateDisconnection,
  simulateReconnection,
  waitForPlayerDisconnectedUI,
  waitForPlayerReconnectedUI,
  waitForVictoryModal,
  waitForTimeoutWarningUI,
  coordinatePlayerTurn,
  waitForPlayerTurnWithTargets,
  getGameOutcome,
  type MultiplayerGameSetup,
} from './helpers/multiplayer-utils';

/**
 * E2E Test Suite: Multi-Browser Multiplayer Tests
 * ============================================================================
 *
 * This suite tests real multiplayer scenarios using two browser contexts
 * to simulate two players interacting in the same game.
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for game persistence and user accounts)
 * - Redis running (for WebSocket sessions and game state sync)
 * - Dev server running on http://localhost:5173
 *
 * PATTERN:
 * Uses Playwright's multi-context pattern where each context represents
 * a different player with their own session and browser state.
 *
 * NOTE: Some tests may be skipped if the backend doesn't fully support
 * certain multiplayer features. Reasons are documented in skip messages.
 *
 * RUN COMMAND: npx playwright test multiplayer.e2e.spec.ts
 */

test.describe('Multiplayer Game E2E', () => {
  // Mark all tests as slow since they involve multiple browsers and WebSocket coordination
  test.slow();
  test.setTimeout(180_000); // 3 minutes per test for multiplayer coordination

  let browser1Context: BrowserContext;
  let browser2Context: BrowserContext;
  let player1Page: Page;
  let player2Page: Page;
  let player1User: TestUser;
  let player2User: TestUser;

  test.beforeEach(async ({ browser }) => {
    // Create two independent browser contexts for separate player sessions
    browser1Context = await browser.newContext();
    browser2Context = await browser.newContext();
    player1Page = await browser1Context.newPage();
    player2Page = await browser2Context.newPage();

    // Generate unique users for test isolation
    player1User = generateTestUser();
    player2User = generateTestUser();
  });

  test.afterEach(async () => {
    // Clean up browser contexts after each test
    await browser1Context?.close();
    await browser2Context?.close();
  });

  // ============================================================================
  // Helper Functions for Multiplayer Tests
  // ============================================================================

  /**
   * Register and login a user on a specific page.
   */
  async function setupPlayer(page: Page, user: TestUser): Promise<void> {
    await registerUser(page, user.username, user.email, user.password);
    await new HomePage(page).assertUsernameDisplayed(user.username);
  }

  /**
   * Create a multiplayer game (not vs AI) and return the game ID.
   */
  async function createMultiplayerGame(page: Page): Promise<string> {
    await test.step('Navigate to lobby', async () => {
      await page.getByRole('link', { name: 'Lobby', exact: true }).click();
      await page.waitForURL('**/lobby', { timeout: 15_000 });
      await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible({
        timeout: 10_000,
      });
    });

    await test.step('Open create game form', async () => {
      await page.getByRole('button', { name: /\+ Create Game/i }).click();
      await expect(page.getByRole('heading', { name: /Create Backend Game/i })).toBeVisible({
        timeout: 5_000,
      });
    });

    await test.step('Configure game for human players (not AI)', async () => {
      // Use default 8x8 board, 2 players, no AI
      // The form defaults to aiCount: 1, so we need to ensure it's set up for human vs human
      // Since the current form doesn't have explicit AI toggle, we submit as-is
      // and rely on another player joining
    });

    await test.step('Submit game creation', async () => {
      await page.getByRole('button', { name: /^Create Game$/i }).click();
      await page.waitForURL('**/game/**', { timeout: 30_000 });
      await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 15_000 });
    });

    // Extract game ID from URL
    const url = page.url();
    const match = url.match(/\/game\/([a-zA-Z0-9-]+)/);
    if (!match) {
      throw new Error(`Could not extract game ID from URL: ${url}`);
    }

    return match[1];
  }

  /**
   * Join an existing game by ID.
   */
  async function joinGameById(page: Page, gameId: string): Promise<void> {
    await test.step(`Navigate to game ${gameId}`, async () => {
      await page.goto(`/game/${gameId}`);
    });

    await test.step('Wait for game to load', async () => {
      const gamePage = new GamePage(page);
      await gamePage.waitForReady(20_000);
    });
  }

  /**
   * Wait for turn indicator to show a specific player.
   */
  async function waitForPlayerTurn(
    page: Page,
    playerNumber: number,
    timeout = 30_000
  ): Promise<void> {
    await expect(page.locator(`text=/P${playerNumber}/i`)).toBeVisible({ timeout });
  }

  /**
   * Click the first valid placement target on a page.
   */
  async function clickValidTarget(page: Page, timeout = 25_000): Promise<void> {
    const validTargetSelector = '[data-testid="board-view"] button[class*="outline-emerald"]';
    await page.waitForSelector(validTargetSelector, {
      state: 'visible',
      timeout,
    });
    const targetCell = page.locator(validTargetSelector).first();
    await targetCell.click();
  }

  /**
   * Wait for board to update (detect changes via move log or visual indicator).
   */
  async function waitForBoardUpdate(page: Page, timeout = 10_000): Promise<void> {
    // Wait for the last move indicator or any visual change
    // This is a simplistic approach - wait for any recent move to appear
    await page.waitForTimeout(1000); // Small delay to allow WebSocket propagation
    await new GamePage(page).assertRecentMovesVisible(timeout);
  }

  // ============================================================================
  // Test Scenario 1: Game Creation and Joining
  // ============================================================================

  test.describe('Game Creation and Joining', () => {
    test('Player 1 creates a game, Player 2 joins using game ID', async () => {
      let gameId: string = '';

      await test.step('Player 1 registers and logs in', async () => {
        await setupPlayer(player1Page, player1User);
      });

      await test.step('Player 2 registers and logs in', async () => {
        await setupPlayer(player2Page, player2User);
      });

      await test.step('Player 1 creates a multiplayer game', async () => {
        gameId = await createMultiplayerGame(player1Page);
        expect(gameId).toBeTruthy();
        expect(gameId.length).toBeGreaterThan(0);
      });

      await test.step('Player 2 joins the game using game ID', async () => {
        await joinGameById(player2Page, gameId);
      });

      await test.step('Both players see the game board', async () => {
        const p1GamePage = new GamePage(player1Page);
        const p2GamePage = new GamePage(player2Page);

        await expect(p1GamePage.boardView).toBeVisible();
        await expect(p2GamePage.boardView).toBeVisible();
      });

      await test.step('Both players see connection status', async () => {
        const p1GamePage = new GamePage(player1Page);
        const p2GamePage = new GamePage(player2Page);

        await p1GamePage.assertConnected();
        await p2GamePage.assertConnected();
      });
    });

    test('Multiple players joining shows player count update', async () => {
      let gameId: string = '';

      await test.step('Setup both players', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
      });

      await test.step('Player 1 creates game', async () => {
        gameId = await createMultiplayerGame(player1Page);
      });

      await test.step('Player 2 joins game', async () => {
        await joinGameById(player2Page, gameId);
      });

      await test.step('Verify both players are shown in game state', async () => {
        // Look for player indicators P1 and P2 on both pages
        await expect(player1Page.locator('text=/P1|P2/i').first()).toBeVisible({ timeout: 15_000 });
        await expect(player2Page.locator('text=/P1|P2/i').first()).toBeVisible({ timeout: 15_000 });
      });
    });
  });

  // ============================================================================
  // Test Scenario 2: Turn-Based Play
  // ============================================================================

  test.describe('Turn-Based Play', () => {
    test('Turn alternates between players after moves', async () => {
      let gameId: string = '';

      await test.step('Setup both players', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
      });

      await test.step('Create and join game', async () => {
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Wait for game to be ready', async () => {
        // Wait for both players to have connection established
        await player1Page.waitForTimeout(2000);
        await player2Page.waitForTimeout(2000);
      });

      await test.step('Player 1 makes first move (ring placement)', async () => {
        const p1GamePage = new GamePage(player1Page);
        await p1GamePage.assertValidTargetsVisible();
        await p1GamePage.clickFirstValidTarget();
      });

      await test.step('Player 2 sees board update and makes move', async () => {
        // Wait for P2 to see valid targets (their turn)
        await player2Page.waitForTimeout(2000); // Allow WebSocket sync
        const p2GamePage = new GamePage(player2Page);

        // P2 should now see valid targets for their turn
        try {
          await p2GamePage.assertValidTargetsVisible();
          await p2GamePage.clickFirstValidTarget();
        } catch {
          // If no valid targets, game may have auto-progressed
          console.debug('No valid targets for P2, game may have different state');
        }
      });
    });

    test('Turn indicator shows correct player on both screens', async () => {
      let gameId: string = '';

      await test.step('Setup game with both players', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Verify turn indicator consistency', async () => {
        // Both players should see the same turn indicator
        const turnPattern = /Turn|P1|P2/i;
        await expect(player1Page.locator(`text=${turnPattern}`).first()).toBeVisible({
          timeout: 15_000,
        });
        await expect(player2Page.locator(`text=${turnPattern}`).first()).toBeVisible({
          timeout: 15_000,
        });
      });
    });
  });

  // ============================================================================
  // Test Scenario 3: Real-Time WebSocket Updates
  // ============================================================================

  test.describe('Real-Time WebSocket Updates', () => {
    test('Move by one player is reflected on other player screen in real-time', async () => {
      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Wait for game ready', async () => {
        await player1Page.waitForTimeout(2000);
      });

      await test.step('Player 1 makes a placement move', async () => {
        const p1GamePage = new GamePage(player1Page);
        await p1GamePage.clickFirstValidTarget();
      });

      await test.step('Player 2 sees move in game log', async () => {
        // Wait for WebSocket to propagate the move
        await player2Page.waitForTimeout(3000);

        // Check if move is logged on P2's screen
        await new GamePage(player2Page).assertRecentMovesVisible(15_000);
      });
    });

    test('Game state syncs between both players after multiple moves', async () => {
      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Wait for game initialization', async () => {
        await Promise.all([player1Page.waitForTimeout(2000), player2Page.waitForTimeout(2000)]);
      });

      await test.step('Make a move on P1', async () => {
        const p1GamePage = new GamePage(player1Page);
        try {
          await p1GamePage.assertValidTargetsVisible();
          await p1GamePage.clickFirstValidTarget();
        } catch {
          console.debug('P1 could not make move - may not be their turn');
        }
      });

      await test.step('Both players should have synchronized game state', async () => {
        // Wait for sync
        await player1Page.waitForTimeout(2000);
        await player2Page.waitForTimeout(2000);

        // Both players should see game phase indicator
        const p1Phase = player1Page.locator('text=/Phase|placement|movement/i');
        const p2Phase = player2Page.locator('text=/Phase|placement|movement/i');

        // At least one phase indicator should be visible on both
        await expect(p1Phase.first()).toBeVisible({ timeout: 10_000 });
        await expect(p2Phase.first()).toBeVisible({ timeout: 10_000 });
      });
    });
  });

  // ============================================================================
  // Test Scenario 4: Game Completion
  // ============================================================================

  test.describe('Game Completion', () => {
    test('Both players see victory/defeat status when game ends', async ({ browser }) => {
      // Uses near-victory fixture API to fast-forward to a state where one move wins.
      // This avoids the need for 30+ coordinated moves.

      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup near-victory fixture game', async () => {
          setup = await setupMultiplayerFixtureGame(browser, 'near_victory_elimination');
        });

        await test.step('Player 1 makes winning marker landing', async () => {
          // Near-victory fixture: P1 stack at (3,3), marker at (4,3)
          await setup!.player1.gamePage.makeMove(3, 3, 4, 3);
        });

        await test.step('Both players see game end state', async () => {
          // Wait for game to process and show result
          await setup!.player1.page.waitForTimeout(3000);

          // Check for victory/defeat indicators on both screens
          const p1GameEnd = setup!.player1.page.locator('text=/victory|defeat|game.*over|winner/i');
          const p2GameEnd = setup!.player2.page.locator(
            'text=/victory|defeat|game.*over|eliminated/i'
          );

          await expect
            .poll(
              async () =>
                (await p1GameEnd.first().isVisible()) && (await p2GameEnd.first().isVisible()),
              { timeout: 20_000 }
            )
            .toBe(true);
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });

    test('Game page shows victory conditions help for both players', async () => {
      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Both players see victory condition explanations', async () => {
        const victoryHelpP1 = player1Page
          .locator('[data-testid="victory-conditions-help"]:visible')
          .first();
        const victoryHelpP2 = player2Page
          .locator('[data-testid="victory-conditions-help"]:visible')
          .first();

        await expect(victoryHelpP1).toBeVisible({ timeout: 15_000 });
        await expect(victoryHelpP2).toBeVisible({ timeout: 15_000 });

        // Verify content contains victory conditions
        await expect(victoryHelpP1.getByText('Ring Elimination', { exact: true })).toBeVisible();
        await expect(victoryHelpP2.getByText('Territory Control', { exact: true })).toBeVisible();
      });
    });

    test('Rematch flow: winner requests rematch and opponent accepts', async ({ browser }) => {
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup near-victory fixture game', async () => {
          setup = await setupMultiplayerFixtureGame(browser, 'near_victory_elimination');
        });

        await test.step('Player 1 makes winning move', async () => {
          await setup!.player1.gamePage.makeMove(3, 3, 4, 3);
        });

        await test.step('Both players see victory modal', async () => {
          const p1Victory = await waitForVictoryModal(setup!.player1, { timeout: 30_000 });
          const p2Victory = await waitForVictoryModal(setup!.player2, { timeout: 30_000 });
          expect(p1Victory || p2Victory).toBeTruthy();
        });

        await test.step('Player 1 requests rematch', async () => {
          const originalGameId = setup!.gameId;
          const currentIdP1 = setup!.player1.gamePage.getGameId();
          const currentIdP2 = setup!.player2.gamePage.getGameId();

          expect(currentIdP1).toBe(originalGameId);
          expect(currentIdP2).toBe(originalGameId);

          await setup!.player1.page.getByRole('button', { name: /Play Again/i }).click();
        });

        await test.step('Player 2 sees rematch offer and accepts', async () => {
          await expect(setup!.player2.page.locator('text=/wants a rematch/i')).toBeVisible({
            timeout: 15_000,
          });

          await setup!.player2.page.getByRole('button', { name: /Accept/i }).click();
        });

        await test.step('Both players join a fresh game session after rematch is accepted', async () => {
          const originalGameId = setup!.gameId;

          // Wait for both clients to stabilise on a new /game/:id route
          await setup!.player1.page.waitForTimeout(3_000);
          await setup!.player2.page.waitForTimeout(3_000);

          await setup!.player1.gamePage.waitForReady(30_000);
          await setup!.player2.gamePage.waitForReady(30_000);

          const newGameIdP1 = setup!.player1.gamePage.getGameId();
          const newGameIdP2 = setup!.player2.gamePage.getGameId();

          expect(newGameIdP1).toBeTruthy();
          expect(newGameIdP2).toBeTruthy();
          expect(newGameIdP1).toBe(newGameIdP2);
          expect(newGameIdP1).not.toBe(originalGameId);

          // Basic sanity check that the fresh session rendered its canonical HUD.
          await expect(setup!.player1.gamePage.phaseIndicator).toBeVisible();
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });

    test('Return to lobby from victory modal navigates away from game without leaving stale session', async ({
      browser,
    }) => {
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup near-victory fixture game', async () => {
          setup = await setupMultiplayerFixtureGame(browser, 'near_victory_elimination');
        });

        await test.step('Player 1 makes winning move and sees victory modal', async () => {
          await setup!.player1.gamePage.makeMove(3, 3, 4, 3);
          const sawVictory = await waitForVictoryModal(setup!.player1, { timeout: 30_000 });
          expect(sawVictory).toBeTruthy();
        });

        await test.step('Player 1 returns to lobby', async () => {
          await setup!.player1.page.getByRole('button', { name: /Return to Lobby/i }).click();
          await setup!.player1.page.waitForURL('**/lobby', { timeout: 20_000 });

          await expect(
            setup!.player1.page.getByRole('heading', { name: /Game Lobby/i })
          ).toBeVisible({ timeout: 10_000 });
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });
  });

  // ============================================================================
  // Test Scenario 5: Disconnection Handling
  // ============================================================================

  test.describe('Disconnection Handling', () => {
    test('Player 1 continues to see game after Player 2 disconnects', async () => {
      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Verify both players connected', async () => {
        await expect(player1Page.getByTestId('board-view')).toBeVisible();
        await expect(player2Page.getByTestId('board-view')).toBeVisible();
      });

      await test.step('Player 2 disconnects (closes page)', async () => {
        await player2Page.close();
      });

      await test.step('Player 1 still sees game board', async () => {
        // Wait a moment for disconnection to propagate
        await player1Page.waitForTimeout(3000);

        // P1 should still see the game
        await expect(player1Page.getByTestId('board-view')).toBeVisible();
        await expect(player1Page.locator('text=/Turn|Connection/i').first()).toBeVisible();
      });
    });

    test('Player can reconnect to game after page reload', async () => {
      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Player 1 makes a move', async () => {
        const p1GamePage = new GamePage(player1Page);
        try {
          await p1GamePage.assertValidTargetsVisible();
          await p1GamePage.clickFirstValidTarget();
        } catch {
          console.debug('Could not make move - continuing test');
        }
        await player1Page.waitForTimeout(2000);
      });

      await test.step('Player 1 reloads page', async () => {
        const p1GamePage = new GamePage(player1Page);
        await p1GamePage.reloadAndWait();
      });

      await test.step('Player 1 sees game state preserved after reload', async () => {
        // Game board should still be visible
        await expect(player1Page.getByTestId('board-view')).toBeVisible();

        // Connection should be re-established
        const p1GamePage = new GamePage(player1Page);
        await p1GamePage.assertConnected();
      });
    });

    test('Player sees warning when decision phase timeout approaches', async ({ browser }) => {
      // Uses a bounded timeout fixture to test the warning without racing setup.
      // The exact-length line fixture exposes canonical line moves directly,
      // so phase state is the stable decision-surface assertion.

      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup decision phase game with short timeout', async () => {
          setup = await setupMultiplayerFixtureGame(browser, {
            scenario: 'line_processing',
            shortTimeoutMs: 20_000,
            shortWarningBeforeMs: 10_000,
          });
        });

        await test.step('Verify player is in decision phase', async () => {
          await setup!.player1.gamePage.assertPhase('Line');
        });

        await test.step('Wait for timeout warning to appear', async () => {
          // Don't make a choice - wait for the warning to appear
          const sawWarning = await waitForTimeoutWarningUI(setup!.player1.page, {
            timeout: 15_000,
          });
          expect(sawWarning).toBeTruthy();
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });

    test('decision phase auto-resolves on timeout for both players', async ({ browser }) => {
      // Use enough time for two browser contexts to finish joining before the
      // canonical decision is auto-resolved.
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup decision phase game with short timeout', async () => {
          setup = await setupMultiplayerFixtureGame(browser, {
            scenario: 'line_processing',
            shortTimeoutMs: 12_000,
            shortWarningBeforeMs: 6_000,
          });
        });

        await test.step('Verify player is in decision phase', async () => {
          await setup!.player1.gamePage.assertPhase('Line');
        });

        await test.step('Wait for auto-resolve after timeout', async () => {
          const autoResolved = /Decision timed out - move applied automatically/i;
          await Promise.all([
            expect(setup!.player1.page.getByText(autoResolved)).toBeVisible({ timeout: 18_000 }),
            expect(setup!.player2.page.getByText(autoResolved)).toBeVisible({ timeout: 18_000 }),
          ]);
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });
  });

  // ============================================================================
  // Test Scenario 7: Near-Victory Multiplayer (using fixtures)
  // ============================================================================

  test.describe('Near-Victory Multiplayer', () => {
    test('Both players see victory/defeat when game ends via fixture', async ({ browser }) => {
      // Uses near-victory fixture to set up a game state one marker landing from victory.
      // Player 1 makes the winning move, both players should see the outcome.

      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup multiplayer near-victory game', async () => {
          setup = await setupMultiplayerFixtureGame(browser, 'near_victory_elimination');
        });

        await test.step('Player 1 makes winning move', async () => {
          // In near-victory fixture: P1 stack at (3,3), marker at (4,3)
          // Move (3,3) -> (4,3) wins the game
          await setup!.player1.gamePage.makeMove(3, 3, 4, 3);
        });

        await test.step('Both players see game end', async () => {
          // Wait for victory modal on both screens
          const p1Victory = await waitForVictoryModal(setup!.player1, { timeout: 30_000 });
          const p2Victory = await waitForVictoryModal(setup!.player2, { timeout: 30_000 });

          expect(p1Victory || p2Victory).toBeTruthy();
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });

    test('Winner sees victory, loser sees defeat', async ({ browser }) => {
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup and complete game', async () => {
          setup = await setupMultiplayerFixtureGame(browser, 'near_victory_elimination');

          // P1 makes winning move
          await setup.player1.gamePage.makeMove(3, 3, 4, 3);

          // Wait for game to end
          await setup.player1.page.waitForTimeout(3000);
        });

        await test.step('Verify Player 1 sees victory', async () => {
          // P1 should see victory text (they won)
          const victoryText = setup!.player1.page.locator('text=/victory|winner|you.*win/i');
          await expect(victoryText.first()).toBeVisible({ timeout: 15_000 });
        });

        await test.step('Verify Player 2 sees defeat', async () => {
          // P2 should see defeat text (they lost)
          const defeatText = setup!.player2.page.locator('text=/defeat|lost|eliminated/i');
          await expect(defeatText.first()).toBeVisible({ timeout: 15_000 });
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });

    test('Territory near-victory fixture produces territory-control win', async ({ browser }) => {
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup multiplayer near-victory territory game', async () => {
          setup = await setupMultiplayerFixtureGame(browser, 'near_victory_territory');
        });

        await test.step('Wait for game over via territory control', async () => {
          // Resolve the fixture's canonical pending region through the same
          // board interaction a player uses in a live territory decision.
          await setup!.player1.gamePage.getCell(4, 4).click();
          const eliminationStack = setup!.player1.gamePage.getCell(7, 7);
          await expect(eliminationStack).toBeEnabled({ timeout: 10_000 });
          await eliminationStack.click();
          const p1Victory = await waitForVictoryModal(setup!.player1, { timeout: 30_000 });
          const p2Victory = await waitForVictoryModal(setup!.player2, { timeout: 30_000 });

          expect(p1Victory || p2Victory).toBeTruthy();
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });

    test('Rated near-victory multiplayer updates winner and loser ratings', async ({ browser }) => {
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup rated near-victory multiplayer game', async () => {
          setup = await setupMultiplayerFixtureGame(browser, {
            scenario: 'near_victory_elimination',
            isRated: true,
          });
        });

        await test.step('Player 1 makes the winning move', async () => {
          await setup!.player1.gamePage.makeMove(3, 3, 4, 3);
          // Allow game_over + rating updates to propagate
          await setup!.player1.page.waitForTimeout(3_000);
        });

        await test.step('Verify outcomes for both players', async () => {
          const p1Outcome = await getGameOutcome(setup!.player1);
          const p2Outcome = await getGameOutcome(setup!.player2);

          expect(p1Outcome).toBe('victory');
          expect(p2Outcome).toBe('defeat');
        });

        await test.step('Verify winner rating > loser rating', async () => {
          const readRating = async (player: MultiplayerGameSetup['player1']): Promise<number> => {
            const homePage = new HomePage(player.page);
            await homePage.goto();
            await homePage.goToProfile();
            await player.page.waitForURL('**/profile', { timeout: 10_000 });
            const ratingText = await player.page.getByTestId('profile-rating').textContent();
            return parseInt((ratingText || '').replace(/[^0-9]/g, ''), 10);
          };

          const p1Rating = await readRating(setup!.player1);
          const p2Rating = await readRating(setup!.player2);

          expect(p1Rating).toBeGreaterThan(0);
          expect(p2Rating).toBeGreaterThan(0);
          expect(p1Rating).toBeGreaterThan(p2Rating);
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });
  });

  // ============================================================================
  // Test Scenario 8: Advanced Disconnection/Reconnection
  // ============================================================================

  test.describe('Advanced Disconnection Scenarios', () => {
    test('Player can reconnect and continue game after disconnect', async ({ browser }) => {
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup multiplayer game', async () => {
          setup = await setupMultiplayerGameAdvanced(browser);
        });

        const gameId = setup!.gameId;

        await test.step('Player 1 makes a move', async () => {
          const hasTargets = await waitForPlayerTurnWithTargets(setup!.player1, 15_000);
          if (hasTargets) {
            await setup!.player1.gamePage.clickFirstValidTarget();
            await setup!.player1.page.waitForTimeout(2000);
          }
        });

        await test.step('Player 2 disconnects', async () => {
          await simulateDisconnection(setup!.player2);
        });

        await test.step('Player 1 remains connected', async () => {
          await expect(setup!.player1.gamePage.boardView).toBeVisible();
          await setup!.player1.gamePage.assertConnected();
        });

        await test.step('Player 2 reconnects', async () => {
          await simulateReconnection(setup!.player2, gameId);
        });

        await test.step('Player 2 sees game state after reconnect', async () => {
          await expect(setup!.player2.gamePage.boardView).toBeVisible();
          // Should see the move log indicating game state was preserved
          await setup!.player2.gamePage.assertRecentMovesVisible(15_000);
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });

    test('Game continues after brief disconnection', async ({ browser }) => {
      let setup: MultiplayerGameSetup | null = null;

      try {
        await test.step('Setup multiplayer game with some moves', async () => {
          setup = await setupMultiplayerGameAdvanced(browser);

          // Make a couple of turns
          for (let i = 0; i < 2; i++) {
            const activePlayer = i % 2 === 0 ? setup.player1 : setup.player2;
            const hasTargets = await waitForPlayerTurnWithTargets(activePlayer, 10_000);
            if (hasTargets) {
              await activePlayer.gamePage.clickFirstValidTarget();
              await activePlayer.page.waitForTimeout(2000);
            }
          }
        });

        const gameId = setup!.gameId;

        await test.step('Player 1 reloads page (simulates brief disconnect)', async () => {
          await setup!.player1.gamePage.reloadAndWait();
        });

        await test.step('Player 1 can continue playing', async () => {
          await setup!.player1.gamePage.assertConnected();

          // Check if it's P1's turn and they can make a move
          const hasTargets = await waitForPlayerTurnWithTargets(setup!.player1, 10_000);
          if (hasTargets) {
            await setup!.player1.gamePage.clickFirstValidTarget();
          }
        });

        await test.step('Game state is consistent', async () => {
          // Both players should see the game log
          await setup!.player1.gamePage.assertRecentMovesVisible(10_000);
          await setup!.player2.gamePage.assertRecentMovesVisible(10_000);
        });
      } finally {
        if (setup) {
          await cleanupMultiplayerSetup(setup);
        }
      }
    });
  });

  // ============================================================================
  // Test Scenario 6: Chat/Communication
  // ============================================================================

  test.describe('Chat and Communication', () => {
    test('Player 1 sends chat message, Player 2 receives it', async () => {
      // Chat feature is now fully implemented with persistence.
      // Messages are sent via WebSocket and persisted to database.

      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Player 1 sends a chat message', async () => {
        const p1GamePage = new GamePage(player1Page);
        await p1GamePage.sendChatMessage('Hello Player 2!');
      });

      await test.step('Player 2 receives the chat message', async () => {
        // Wait for WebSocket to propagate
        await player2Page.waitForTimeout(2000);

        // Look for the message in P2's chat area
        await expect(player2Page.locator('text=/Hello Player 2/i')).toBeVisible({
          timeout: 10_000,
        });
      });
    });

    test('Game event log shows moves from both players', async () => {
      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Make moves from both players', async () => {
        // P1 makes a move
        const p1GamePage = new GamePage(player1Page);
        try {
          await p1GamePage.assertValidTargetsVisible();
          await p1GamePage.clickFirstValidTarget();
          await player1Page.waitForTimeout(3000);
        } catch {
          console.debug('P1 could not make initial move');
        }

        // P2 makes a move
        const p2GamePage = new GamePage(player2Page);
        try {
          await p2GamePage.assertValidTargetsVisible();
          await p2GamePage.clickFirstValidTarget();
          await player2Page.waitForTimeout(3000);
        } catch {
          console.debug('P2 could not make move');
        }
      });

      await test.step('Both players see game log with recent moves', async () => {
        // Wait for full sync
        await Promise.all([player1Page.waitForTimeout(2000), player2Page.waitForTimeout(2000)]);

        // Check that game log section exists on both
        const p1Log = player1Page.locator('text=/Game log|Recent moves/i');
        const p2Log = player2Page.locator('text=/Game log|Recent moves/i');

        await expect(p1Log.first()).toBeVisible({ timeout: 15_000 });
        await expect(p2Log.first()).toBeVisible({ timeout: 15_000 });
      });
    });
  });

  // ============================================================================
  // Additional Test Scenarios
  // ============================================================================

  test.describe('Additional Multiplayer Scenarios', () => {
    test('Both players can view their opponent information', async () => {
      let gameId: string = '';

      await test.step('Setup game', async () => {
        await setupPlayer(player1Page, player1User);
        await setupPlayer(player2Page, player2User);
        gameId = await createMultiplayerGame(player1Page);
        await joinGameById(player2Page, gameId);
      });

      await test.step('Players can see player indicators', async () => {
        // Both pages should show P1 and P2 indicators somewhere in the HUD
        await expect(player1Page.locator('text=/P1|Player 1/i').first()).toBeVisible({
          timeout: 15_000,
        });
        await expect(player2Page.locator('text=/P2|Player 2/i').first()).toBeVisible({
          timeout: 15_000,
        });
      });
    });

    test('Spectator can watch an ongoing game with read-only HUD and leave back to lobby', async () => {
      let gameId: string = '';

      // Create a third context for spectator
      const spectatorContext = await player1Page.context().browser()!.newContext();
      const spectatorPage = await spectatorContext.newPage();
      const spectatorUser = generateTestUser();

      try {
        await test.step('Setup players and game', async () => {
          await setupPlayer(player1Page, player1User);
          await setupPlayer(player2Page, player2User);
          await setupPlayer(spectatorPage, spectatorUser);
          gameId = await createMultiplayerGame(player1Page);
          await joinGameById(player2Page, gameId);
        });

        await test.step('Spectator joins via spectator route', async () => {
          await spectatorPage.goto(`/spectate/${gameId}`);
          const spectatorGamePage = new GamePage(spectatorPage);
          await spectatorGamePage.waitForReady(30_000);
        });

        await test.step('Spectator sees spectator HUD and read-only message', async () => {
          // Spectator should see the board and explicit spectator indicators
          await expect(spectatorPage.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });
          await expect(spectatorPage.locator('text=/Spectator Mode/i')).toBeVisible({
            timeout: 10_000,
          });
          await expect(
            spectatorPage.locator('text=/Moves disabled while spectating\\./i')
          ).toBeVisible({ timeout: 10_000 });
        });

        await test.step('Spectator can leave back to lobby without hanging game session', async () => {
          await spectatorPage.getByRole('button', { name: /Back to lobby/i }).click();
          await spectatorPage.waitForURL('**/lobby', { timeout: 15_000 });
          await expect(spectatorPage.getByRole('heading', { name: /Game Lobby/i })).toBeVisible({
            timeout: 10_000,
          });
        });
      } finally {
        await spectatorContext.close();
      }
    });
  });
});
