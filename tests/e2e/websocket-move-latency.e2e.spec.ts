import { test, expect, type Page } from '@playwright/test';
import { registerAndLogin, createGame } from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * WebSocket move latency E2E spec
 * ---------------------------------------------------------------------------
 *
 * Measures browser-observed round-trip latency for human moves over the
 * real SPA + GameContext WebSocket path. This is intended as a small
 * "perf smoke" guardrail that runs against dev/staging environments.
 *
 * The test:
 * - Logs in a fresh test user
 * - Creates a backend human-vs-AI game (square8) via the normal lobby flow
 * - Waits for the game board and WebSocket connection to be ready
 * - Plays a series of simple human moves via the UI
 * - For each move, measures confirmation-to-GameEventLog update latency
 * - Asserts p95 and p99 RTTs are within the staging-level SLOs from
 *   STRATEGIC_ROADMAP.md §2.2
 */

// Staging-level WebSocket gameplay SLOs (see STRATEGIC_ROADMAP.md §2.2).
// TODO: Allow overriding via env (e.g. E2E_WS_MOVE_P95_MS) if environments
// need looser/tighter thresholds without code changes.
const P95_SLO_MS = 300;
const P99_SLO_MS = 600;

test.describe('WebSocket move latency E2E', () => {
  // AI games involve backend + AI service, so give the test some headroom.
  test.setTimeout(120_000);

  test('websocket move latency stays within SLOs', async ({ page }) => {
    // 1. Login as a fresh test user using existing helpers.
    await registerAndLogin(page);

    // 2. Create a human-vs-AI backend game via the lobby UI.
    //
    // createGame() navigates to /lobby, opens "Create Backend Game",
    // submits the form, and waits for redirect to /game/:gameId with the
    // board rendered. The default configuration is a square8 backend game
    // with an AI opponent when vsAI is true.
    const gameId = await createGame(page, { boardType: 'square8', vsAI: true, isRated: false });
    expect(gameId).toBeTruthy();

    // 3. Wait for the WebSocket game view to be fully ready.
    const gamePage = new GamePage(page);
    await gamePage.waitForReady(30_000);
    // Ensure the GameEventLog is present so we can detect move updates.
    await expect(page.getByTestId('game-event-log')).toBeVisible({ timeout: 30_000 });

    // 4. Play a sequence of human moves and record RTT samples.
    const moveRtts: number[] = [];
    const targetSamples = 12; // aim for 12 samples; enforce a minimum of 10 later

    for (let i = 0; i < targetSamples; i++) {
      const rtt = await measureMoveRtt(page, gamePage);
      moveRtts.push(rtt);
      // Small pause between moves to allow AI responses and UI to settle.
      await page.waitForTimeout(250);
    }

    // Require at least 10 successful RTT samples; otherwise treat as failure
    // rather than silently asserting on a too-small dataset.
    if (moveRtts.length < 10) {
      throw new Error(
        'Expected at least 10 WebSocket move RTT samples but collected ' +
          moveRtts.length +
          '. Samples: ' +
          JSON.stringify(moveRtts)
      );
    }

    // 5. Compute distribution (p50, p95, p99) and assert against SLOs.
    const sorted = [...moveRtts].sort((a, b) => a - b);
    const p = (q: number) => {
      if (sorted.length === 0) return NaN;
      const idx = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
      return sorted[idx];
    };
    const p50 = p(0.5);
    const p95 = p(0.95);
    const p99 = p(0.99);

    // Log samples and aggregates for debugging and perf dashboards.
    // These appear in the Playwright output but do not affect assertions.
    // eslint-disable-next-line no-console
    console.log('WebSocket move RTT samples (ms):', moveRtts);
    // eslint-disable-next-line no-console
    console.log({ p50, p95, p99 });

    expect(
      p95,
      `p95 RTT ${p95}ms exceeded SLO ${P95_SLO_MS}ms. Distribution: ${JSON.stringify(sorted)}`
    ).toBeLessThanOrEqual(P95_SLO_MS);
    expect(
      p99,
      `p99 RTT ${p99}ms exceeded SLO ${P99_SLO_MS}ms. Distribution: ${JSON.stringify(sorted)}`
    ).toBeLessThanOrEqual(P99_SLO_MS);
  });
});

/**
 * Measure a single human move round-trip time (RTT) from the browser's
 * perspective.
 *
 * The RTT is defined as:
 *
 *   performance.now() at placement confirmation → first "Recent moves"
 *   signature change corresponding to that move's game_state update.
 *
 * The implementation uses a DOM-based detector that:
 * - Resolves a visible empty placement target before timing begins
 * - Opens the placement-count control and chooses one ring before timing
 * - Submits through the visible Place action
 * - Waits (in the browser context) until the move signature changes
 *   and returns the elapsed performance.now() delta in milliseconds
 */
async function measureMoveRtt(page: Page, gamePage: GamePage): Promise<number> {
  // Resolve a deterministic legal placement before timing begins. Use the
  // context-menu count control so each sample places one ring and the test can
  // collect twelve samples without exhausting the player's opening supply.
  const placementTarget = gamePage.boardView
    .locator('button[class*="outline-emerald"][aria-label*="Empty cell"]')
    .first();
  await expect(placementTarget).toBeVisible({ timeout: 25_000 });
  await placementTarget.click({ button: 'right' });

  const placementDialog = page.getByTestId('ring-placement-count-overlay');
  await expect(placementDialog).toBeVisible();
  await placementDialog.getByLabel('Number of rings').fill('1');
  const placeButton = placementDialog.getByRole('button', { name: 'Place', exact: true });
  await expect(placeButton).toBeEnabled();

  // Snapshot the visible move list. The event log is intentionally capped, so
  // its text changes after later moves even when its item count stays constant.
  const previousMoveSignature = await page.evaluate(() => {
    const log = document.querySelector('[data-testid="game-event-log"]');
    if (!log) return '';

    // Find the "Recent moves" header within the event log.
    const headings = Array.from(log.querySelectorAll('div')).filter((el) =>
      /Recent moves/i.test(el.textContent || '')
    );
    const container = (headings[0]?.parentElement ?? log) as HTMLElement;
    const list = container.querySelector('ul');
    if (!list) return '';

    return Array.from(list.querySelectorAll('li'))
      .map((item) => item.textContent?.trim() ?? '')
      .join('|');
  });

  // Capture start time as close as possible to confirmation that submits the move.
  const startTime = await page.evaluate(() => performance.now());

  // Start immediately before the user-visible action invokes the submission
  // handler. Dialog setup above is intentionally outside the measured RTT.
  await placeButton.click();

  // Wait in the browser context until the move signature changes, then return
  // the elapsed time since startTime.
  const timeoutMs = 10_000;
  const rttHandle = await page.waitForFunction(
    (state: { previousSignature: string; startedAt: number }) => {
      const log = document.querySelector('[data-testid="game-event-log"]');
      if (!log) return false;

      const headings = Array.from(log.querySelectorAll('div')).filter((el) =>
        /Recent moves/i.test(el.textContent || '')
      );
      const container = (headings[0]?.parentElement ?? log) as HTMLElement;
      const list = container.querySelector('ul');
      if (!list) return false;

      const signature = Array.from(list.querySelectorAll('li'))
        .map((item) => item.textContent?.trim() ?? '')
        .join('|');
      if (!signature || signature === state.previousSignature) {
        return false;
      }

      // First observed change in the "Recent moves" list is treated as the
      // authoritative game_state update for this move.
      return performance.now() - state.startedAt;
    },
    { previousSignature: previousMoveSignature, startedAt: startTime },
    { timeout: timeoutMs }
  );

  const rttMs = (await rttHandle.jsonValue()) as number;
  return rttMs;
}
