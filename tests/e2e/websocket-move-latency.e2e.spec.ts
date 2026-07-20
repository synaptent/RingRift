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
 * - Creates a fresh backend human-vs-AI game for each sample
 * - Waits for each game board and WebSocket connection to be ready
 * - Plays the first one-ring placement via the UI
 * - Measures confirmation-to-authoritative-move-counter latency
 * - Asserts p95 and p99 RTTs are within the staging-level SLOs from
 *   STRATEGIC_ROADMAP.md §2.2
 */

// Staging-level WebSocket gameplay SLOs (see STRATEGIC_ROADMAP.md §2.2).
// TODO: Allow overriding via env (e.g. E2E_WS_MOVE_P95_MS) if environments
// need looser/tighter thresholds without code changes.
const P95_SLO_MS = 300;
const P99_SLO_MS = 600;

test.describe('WebSocket move latency E2E', () => {
  // Each sample creates a fresh game so setup time needs headroom. Setup is
  // intentionally outside the per-move SLO measurement.
  test.setTimeout(180_000);

  test('websocket move latency stays within SLOs', async ({ page }) => {
    // 1. Login as a fresh test user using existing helpers.
    await registerAndLogin(page);

    // 2. Create a fresh game for each first-placement sample. A single game
    // enters movement/capture after placement and would require measuring a
    // different interaction contract on later iterations.
    const moveRtts: number[] = [];
    const targetSamples = 12; // aim for 12 samples; enforce a minimum of 10 later

    for (let i = 0; i < targetSamples; i++) {
      const gameId = await createGame(page, {
        boardType: 'square8',
        vsAI: true,
        isRated: false,
      });
      expect(gameId).toBeTruthy();

      const gamePage = new GamePage(page);
      await gamePage.waitForReady(30_000);
      const rtt = await measureMoveRtt(page, gamePage);
      moveRtts.push(rtt);
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

    // 3. Compute distribution (p50, p95, p99) and assert against SLOs.
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
 *   performance.now() at placement confirmation → first higher Move # value
 *   corresponding to that move's game_state update.
 *
 * The implementation uses a DOM-based detector that:
 * - Resolves a visible empty placement target before timing begins
 * - Pins that target by board coordinates before selection changes its styling
 * - Confirms the pending one-ring placement on that exact cell
 * - Waits (in the browser context) until the authoritative move number advances
 *   and returns the elapsed performance.now() delta in milliseconds
 */
async function measureMoveRtt(page: Page, gamePage: GamePage): Promise<number> {
  // Resolve a deterministic legal placement before timing begins. Pin its
  // coordinates because selecting the cell changes the class-based valid-target
  // styling used by the discovery locator.
  const placementTarget = gamePage.boardView
    .locator('button[class*="outline-emerald"][aria-label*="Empty cell"]')
    .first();
  await expect(placementTarget).toBeVisible({ timeout: 25_000 });
  const targetCoordinates = await placementTarget.evaluate((element) => ({
    x: (element as HTMLElement).dataset.x,
    y: (element as HTMLElement).dataset.y,
  }));
  expect(targetCoordinates.x).toBeDefined();
  expect(targetCoordinates.y).toBeDefined();

  const exactPlacementTarget = gamePage.boardView.locator(
    `button[data-x="${targetCoordinates.x}"][data-y="${targetCoordinates.y}"]`
  );
  await exactPlacementTarget.click();
  // Allow the pending-placement state and its latest double-click handler to
  // commit before beginning the measured interval.
  await page.waitForTimeout(50);

  // Snapshot the canonical move counter. Human-readable event strings can be
  // identical for repeated placements even though the underlying move changed.
  const previousMoveNumber = await page.evaluate(() => {
    const counter = document.querySelector('[data-testid="game-move-number"]');
    const match = counter?.textContent?.match(/Move #(\d+)/);
    return match ? Number.parseInt(match[1] ?? '0', 10) : 0;
  });

  // Capture start time as close as possible to confirmation that submits the move.
  const startTime = await page.evaluate(() => performance.now());

  // Dispatch only the confirmation event to the same selected cell; using the
  // original class-based locator here could re-resolve to a different target.
  await exactPlacementTarget.dispatchEvent('dblclick');

  // Wait in the browser context until the move number advances, then return
  // the elapsed time since startTime.
  const timeoutMs = 10_000;
  const rttHandle = await page.waitForFunction(
    (state: { previousMoveNumber: number; startedAt: number }) => {
      const counter = document.querySelector('[data-testid="game-move-number"]');
      const match = counter?.textContent?.match(/Move #(\d+)/);
      const currentMoveNumber = match ? Number.parseInt(match[1] ?? '0', 10) : 0;
      if (currentMoveNumber <= state.previousMoveNumber) {
        return false;
      }

      // The move counter is derived from the latest authoritative game state.
      return performance.now() - state.startedAt;
    },
    { previousMoveNumber, startedAt: startTime },
    { timeout: timeoutMs }
  );

  const rttMs = (await rttHandle.jsonValue()) as number;
  return rttMs;
}
