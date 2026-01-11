/**
 * Board Diagnostics Script
 *
 * Captures screenshots, console logs, and DOM inspection data
 * for each board type to help diagnose layout issues.
 *
 * Run with: npx playwright test tests/e2e/board-diagnostics.spec.ts --headed
 * Or debug: npx playwright test tests/e2e/board-diagnostics.spec.ts --debug
 */

import { test, expect, type Page, type ConsoleMessage } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Board types and their direct URL presets
const BOARD_CONFIGS = [
  { type: 'square8', preset: 'human-vs-ai', label: '8×8 Compact' },
  { type: 'square19', preset: 'square19-human-vs-ai', label: '19×19 Classic' },
  { type: 'hex8', preset: 'hex8-vs-ai', label: 'Hex 8 Compact' },
  { type: 'hexagonal', preset: 'hex-challenge', label: 'Full Hexagonal' },
] as const;
type BoardType = (typeof BOARD_CONFIGS)[number]['type'];

// Output directory for diagnostics
const OUTPUT_DIR = path.join(__dirname, '../../diagnostics-output');

interface DiagnosticsData {
  boardType: BoardType;
  timestamp: string;
  viewport: { width: number; height: number };
  consoleLogs: Array<{ type: string; text: string }>;
  boardContainer: {
    boundingBox: { x: number; y: number; width: number; height: number } | null;
    computedStyles: Record<string, string>;
    attributes: Record<string, string>;
  } | null;
  scalingWrapper: {
    boundingBox: { x: number; y: number; width: number; height: number } | null;
    computedStyles: Record<string, string>;
    inlineStyles: string;
  } | null;
  gridContainer: {
    boundingBox: { x: number; y: number; width: number; height: number } | null;
    computedStyles: Record<string, string>;
  } | null;
  bottomPanels: Array<{
    boundingBox: { x: number; y: number; width: number; height: number } | null;
    className: string;
  }>;
  rightSidebar: {
    boundingBox: { x: number; y: number; width: number; height: number } | null;
  } | null;
  layoutAnalysis: {
    boardOverflowsContainer: boolean;
    panelOverlapsBoard: boolean;
    sidebarOverlapsBoard: boolean;
    gaps: {
      headerToBoard: number | null;
      boardToInfoPanel: number | null;
    };
  };
}

// Ensure output directory exists
function ensureOutputDir() {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
}

// Capture console logs
function setupConsoleCapture(page: Page): Array<{ type: string; text: string }> {
  const logs: Array<{ type: string; text: string }> = [];
  page.on('console', (msg: ConsoleMessage) => {
    logs.push({ type: msg.type(), text: msg.text() });
  });
  return logs;
}

// Get computed styles for an element
async function getComputedStyles(page: Page, selector: string): Promise<Record<string, string>> {
  return page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return {};
    const styles = window.getComputedStyle(el);
    return {
      width: styles.width,
      height: styles.height,
      padding: styles.padding,
      margin: styles.margin,
      transform: styles.transform,
      transformOrigin: styles.transformOrigin,
      overflow: styles.overflow,
      display: styles.display,
      position: styles.position,
      gridTemplateColumns: styles.gridTemplateColumns,
      gap: styles.gap,
    };
  }, selector);
}

// Get element attributes
async function getAttributes(page: Page, selector: string): Promise<Record<string, string>> {
  return page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return {};
    const attrs: Record<string, string> = {};
    for (const attr of el.attributes) {
      attrs[attr.name] = attr.value;
    }
    return attrs;
  }, selector);
}

// Navigate directly to sandbox with preset URL (bypasses setup config)
async function setupBoardType(page: Page, preset: string): Promise<void> {
  // Go directly to sandbox with preset - auto-starts the game
  await page.goto(`/sandbox?preset=${preset}`);

  // Wait for board to render (preset auto-launches the game)
  await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 30_000 });

  // Wait for any animations/scaling calculations to complete
  await page.waitForTimeout(1500);
}

// Collect diagnostics data for current page state
async function collectDiagnostics(
  page: Page,
  boardType: BoardType,
  consoleLogs: Array<{ type: string; text: string }>
): Promise<DiagnosticsData> {
  const viewport = page.viewportSize() || { width: 0, height: 0 };

  // Board container
  const boardContainer = page.getByTestId('board-view');
  const boardBox = await boardContainer.boundingBox().catch(() => null);
  const boardStyles = await getComputedStyles(page, '[data-testid="board-view"]');
  const boardAttrs = await getAttributes(page, '[data-testid="board-view"]');

  // Scaling wrapper (parent of board-view)
  const scalingWrapper = page.locator('.board-scaling-wrapper').first();
  const wrapperBox = await scalingWrapper.boundingBox().catch(() => null);
  const wrapperStyles = await getComputedStyles(page, '.board-scaling-wrapper');
  const wrapperInlineStyles = (await scalingWrapper.getAttribute('style').catch(() => '')) || '';

  // Grid container (parent section with grid layout)
  const gridContainer = page.locator('.grid').first();
  const gridBox = await gridContainer.boundingBox().catch(() => null);
  const gridStyles = await getComputedStyles(page, '.grid');

  // Bottom panels (info panel, victory conditions)
  const bottomPanels: DiagnosticsData['bottomPanels'] = [];
  const panelLocators = page.locator('section.p-3, [class*="VICTORY"]').all();
  for (const panel of await panelLocators) {
    const box = await panel.boundingBox().catch(() => null);
    const className = (await panel.getAttribute('class')) || '';
    bottomPanels.push({ boundingBox: box, className });
  }

  // Right sidebar
  const sidebar = page.locator('aside').first();
  const sidebarBox = await sidebar.boundingBox().catch(() => null);

  // Layout analysis
  let headerToBoard: number | null = null;
  let boardToInfoPanel: number | null = null;

  // Find header panel
  const headerPanel = page.locator('.grid > div').first();
  const headerBox = await headerPanel.boundingBox().catch(() => null);

  if (headerBox && boardBox) {
    headerToBoard = boardBox.y - (headerBox.y + headerBox.height);
  }

  if (boardBox && bottomPanels[0]?.boundingBox) {
    boardToInfoPanel = bottomPanels[0].boundingBox.y - (boardBox.y + boardBox.height);
  }

  // Check for overlaps
  const boardOverflowsContainer =
    wrapperBox && boardBox
      ? boardBox.width > wrapperBox.width || boardBox.height > wrapperBox.height
      : false;

  const panelOverlapsBoard =
    boardBox && bottomPanels[0]?.boundingBox
      ? bottomPanels[0].boundingBox.y < boardBox.y + boardBox.height
      : false;

  const sidebarOverlapsBoard =
    boardBox && sidebarBox ? sidebarBox.x < boardBox.x + boardBox.width : false;

  return {
    boardType,
    timestamp: new Date().toISOString(),
    viewport,
    consoleLogs,
    boardContainer: boardBox
      ? {
          boundingBox: boardBox,
          computedStyles: boardStyles,
          attributes: boardAttrs,
        }
      : null,
    scalingWrapper: wrapperBox
      ? {
          boundingBox: wrapperBox,
          computedStyles: wrapperStyles,
          inlineStyles: wrapperInlineStyles,
        }
      : null,
    gridContainer: gridBox
      ? {
          boundingBox: gridBox,
          computedStyles: gridStyles,
        }
      : null,
    bottomPanels,
    rightSidebar: sidebarBox ? { boundingBox: sidebarBox } : null,
    layoutAnalysis: {
      boardOverflowsContainer,
      panelOverlapsBoard,
      sidebarOverlapsBoard,
      gaps: {
        headerToBoard,
        boardToInfoPanel,
      },
    },
  };
}

test.describe('Board Diagnostics', () => {
  test.beforeAll(() => {
    ensureOutputDir();
  });

  for (const config of BOARD_CONFIGS) {
    test(`Capture diagnostics for ${config.label} (${config.type})`, async ({ page }) => {
      // Setup console capture
      const consoleLogs = setupConsoleCapture(page);

      // Navigate to board via direct URL preset
      await setupBoardType(page, config.preset);

      // Take full page screenshot
      const screenshotPath = path.join(OUTPUT_DIR, `${config.type}-full.png`);
      await page.screenshot({ path: screenshotPath, fullPage: true });

      // Take viewport screenshot
      const viewportScreenshotPath = path.join(OUTPUT_DIR, `${config.type}-viewport.png`);
      await page.screenshot({ path: viewportScreenshotPath });

      // Collect diagnostics
      const diagnostics = await collectDiagnostics(page, config.type, consoleLogs);

      // Save diagnostics JSON
      const jsonPath = path.join(OUTPUT_DIR, `${config.type}-diagnostics.json`);
      fs.writeFileSync(jsonPath, JSON.stringify(diagnostics, null, 2));

      // Log summary
      console.log(`\n=== ${config.label} (${config.type}) Diagnostics ===`);
      console.log(`Viewport: ${diagnostics.viewport.width}x${diagnostics.viewport.height}`);
      if (diagnostics.boardContainer?.boundingBox) {
        const bb = diagnostics.boardContainer.boundingBox;
        console.log(
          `Board: ${bb.width.toFixed(0)}x${bb.height.toFixed(0)} at (${bb.x.toFixed(0)}, ${bb.y.toFixed(0)})`
        );
      }
      if (diagnostics.scalingWrapper?.boundingBox) {
        const bb = diagnostics.scalingWrapper.boundingBox;
        console.log(
          `Wrapper: ${bb.width.toFixed(0)}x${bb.height.toFixed(0)} at (${bb.x.toFixed(0)}, ${bb.y.toFixed(0)})`
        );
        console.log(`Wrapper inline styles: ${diagnostics.scalingWrapper.inlineStyles}`);
      }
      console.log(
        `Gaps - Header to Board: ${diagnostics.layoutAnalysis.gaps.headerToBoard?.toFixed(0) ?? 'N/A'}px`
      );
      console.log(
        `Gaps - Board to Panel: ${diagnostics.layoutAnalysis.gaps.boardToInfoPanel?.toFixed(0) ?? 'N/A'}px`
      );
      console.log(
        `Board overflows container: ${diagnostics.layoutAnalysis.boardOverflowsContainer}`
      );
      console.log(`Panel overlaps board: ${diagnostics.layoutAnalysis.panelOverlapsBoard}`);
      console.log(`Sidebar overlaps board: ${diagnostics.layoutAnalysis.sidebarOverlapsBoard}`);
      console.log(`Screenshots saved to: ${OUTPUT_DIR}`);

      // Assertions for layout issues (soft - don't fail test, just log)
      if (diagnostics.layoutAnalysis.boardOverflowsContainer) {
        console.warn('⚠️ Board overflows its container');
      }
      if (diagnostics.layoutAnalysis.panelOverlapsBoard) {
        console.warn('⚠️ Bottom panel overlaps board');
      }
      if (diagnostics.layoutAnalysis.sidebarOverlapsBoard) {
        console.warn('⚠️ Sidebar overlaps board');
      }
    });
  }

  test('Generate summary report', async () => {
    // This test runs after all board diagnostics and generates a summary
    const summaryPath = path.join(OUTPUT_DIR, 'summary.md');
    let summary = '# Board Diagnostics Summary\n\n';
    summary += `Generated: ${new Date().toISOString()}\n\n`;

    for (const config of BOARD_CONFIGS) {
      const jsonPath = path.join(OUTPUT_DIR, `${config.type}-diagnostics.json`);
      if (fs.existsSync(jsonPath)) {
        const data: DiagnosticsData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

        summary += `## ${config.label} (${config.type})\n\n`;
        summary += `| Metric | Value |\n|--------|-------|\n`;

        if (data.boardContainer?.boundingBox) {
          const bb = data.boardContainer.boundingBox;
          summary += `| Board Size | ${bb.width.toFixed(0)}x${bb.height.toFixed(0)}px |\n`;
        }
        if (data.scalingWrapper?.boundingBox) {
          const bb = data.scalingWrapper.boundingBox;
          summary += `| Container Size | ${bb.width.toFixed(0)}x${bb.height.toFixed(0)}px |\n`;
          summary += `| Inline Styles | \`${data.scalingWrapper.inlineStyles}\` |\n`;
        }
        summary += `| Header→Board Gap | ${data.layoutAnalysis.gaps.headerToBoard?.toFixed(0) ?? 'N/A'}px |\n`;
        summary += `| Board→Panel Gap | ${data.layoutAnalysis.gaps.boardToInfoPanel?.toFixed(0) ?? 'N/A'}px |\n`;
        summary += `| Overflow Issues | ${data.layoutAnalysis.boardOverflowsContainer ? '⚠️ Yes' : '✅ No'} |\n`;
        summary += `| Overlap Issues | ${data.layoutAnalysis.panelOverlapsBoard || data.layoutAnalysis.sidebarOverlapsBoard ? '⚠️ Yes' : '✅ No'} |\n`;
        summary += '\n';
      }
    }

    fs.writeFileSync(summaryPath, summary);
    console.log(`\nSummary report saved to: ${summaryPath}`);
  });
});
