# Visual Regression Testing

## Overview

Visual regression testing uses Playwright's screenshot comparison features to detect unintended UI changes. Screenshots (baselines) are stored in the repository and compared against on each test run.

## Quick Start

### Running Visual Tests

```bash
# Run all visual regression tests
npm run test:e2e -- visual-regression

# Run with specific browser
npm run test:e2e -- visual-regression --project=chromium

# Run in headed mode (see browsers)
npm run test:e2e -- visual-regression --headed
```

### Updating Baselines

When you intentionally change the UI:

```bash
# Update all screenshots
npx playwright test visual-regression --update-snapshots

# Update specific test screenshots
npx playwright test visual-regression -g "game board" --update-snapshots
```

## Test Coverage

The visual regression suite covers:

### Page Screenshots

| Test                 | Description             | Screenshot                    |
| -------------------- | ----------------------- | ----------------------------- |
| Home page            | Landing page for guests | `home-page.png`               |
| Login page           | Authentication form     | `login-page.png`              |
| Register page        | Registration form       | `register-page.png`           |
| Home (authenticated) | Home with user context  | `home-page-authenticated.png` |
| Lobby page           | Game lobby              | `lobby-page.png`              |

### Game Board Screenshots

| Test            | Description                     | Screenshot                     |
| --------------- | ------------------------------- | ------------------------------ |
| Initial board   | Empty 8x8 game board            | `initial-game-board.png`       |
| Valid targets   | Board with placement highlights | `board-with-valid-targets.png` |
| After placement | Board after ring placed         | `board-after-placement.png`    |
| Hex board       | Hexagonal board variant         | `hex-board-initial.png`        |
| 19x19 board     | Large square board              | `board-19x19-initial.png`      |

### Component Screenshots

| Test          | Description            | Screenshot           |
| ------------- | ---------------------- | -------------------- |
| Game HUD      | Turn/status indicators | `game-hud.png`       |
| Event log     | Game move history      | `game-event-log.png` |
| Victory modal | Win condition display  | `victory-modal.png`  |

### Sandbox Screenshots

| Test           | Description           | Screenshot                   |
| -------------- | --------------------- | ---------------------------- |
| Pregame setup  | Sandbox configuration | `sandbox-pregame-setup.png`  |
| Launched board | Active sandbox game   | `sandbox-launched-board.png` |
| Touch controls | Mobile control panel  | `sandbox-touch-controls.png` |

### Responsive Screenshots

| Viewport          | Tests                      | Prefix     |
| ----------------- | -------------------------- | ---------- |
| Mobile (375x667)  | Home, Login, Game, Sandbox | `mobile-*` |
| Tablet (768x1024) | Game board, Hex board      | `tablet-*` |

## Directory Structure

```
tests/e2e/
├── __snapshots__/
│   └── visual-regression.e2e.spec.ts-snapshots/
│       ├── home-page-chromium.png
│       ├── home-page-Mobile-Chrome.png
│       ├── login-page-chromium.png
│       └── ... (other baselines)
├── visual-regression.e2e.spec.ts
└── VISUAL_TESTING.md
```

## Configuration

Visual testing settings are in [`playwright.config.ts`](../../playwright.config.ts):

```typescript
expect: {
  toHaveScreenshot: {
    maxDiffPixels: 100,     // Allow minor pixel differences
    threshold: 0.2,          // 20% color threshold per pixel
    animations: 'disabled',  // Disable CSS animations
  },
  toMatchSnapshot: {
    maxDiffPixelRatio: 0.01, // Allow 1% pixel difference
  },
},
snapshotDir: './tests/e2e/__snapshots__',
```

## Reviewing Failures

When a visual test fails, Playwright generates comparison images:

1. **Location**: `test-results/` directory
2. **Files generated**:
   - `<test-name>-expected.png` - The baseline screenshot
   - `<test-name>-actual.png` - The current screenshot
   - `<test-name>-diff.png` - Visual diff highlighting changes

### Viewing Diff Images

```bash
# Open the HTML report which includes visual diffs
npx playwright show-report
```

### Analyzing Failures

1. Check if the change is intentional (new feature, design update)
2. Review the diff image to understand what changed
3. If intentional, update baselines with `--update-snapshots`
4. If unintentional, investigate and fix the regression

## Best Practices

### Writing Visual Tests

1. **Prefer element screenshots** over full-page screenshots for stability:

   ```typescript
   // Better - more stable
   const boardView = page.getByTestId('board-view');
   await expect(boardView).toHaveScreenshot('board.png');

   // Less stable - affected by unrelated UI changes
   await expect(page).toHaveScreenshot('full-page.png', { fullPage: true });
   ```

2. **Mask dynamic content** to prevent false failures:

   ```typescript
   await expect(page).toHaveScreenshot('page.png', {
     mask: [page.locator('[data-testid="timestamp"]'), page.locator('text=/e2e-user-/')],
   });
   ```

3. **Wait for stability** before capturing:

   ```typescript
   // Wait for animations to complete
   await page.waitForTimeout(500);

   // Wait for specific element to be visible
   await expect(element).toBeVisible();
   ```

4. **Use meaningful names** that describe the UI state:
   ```typescript
   await expect(board).toHaveScreenshot('board-with-rings-placed.png');
   ```

### Maintaining Baselines

1. **Review changes carefully** before updating baselines
2. **Commit baselines** with the code that changed the UI
3. **Use consistent environments** (font rendering varies by OS)
4. **Consider CI environment** - baselines may need to be generated in CI

## CI Integration

### GitHub Actions

Visual tests run automatically in CI. Key considerations:

1. **Font rendering**: Linux CI may render fonts differently than macOS/Windows
2. **Baseline updates**: Generate baselines in CI for consistency
3. **Artifacts**: Failed tests upload comparison images as artifacts

### Updating Baselines in CI

If your local baselines don't match CI:

```bash
# 1. Run tests in CI and download the actual screenshots
# 2. Or generate baselines using Docker for consistent rendering

# Using Docker to match CI environment:
docker run -it --rm -v $(pwd):/work -w /work mcr.microsoft.com/playwright:v1.40.0 \
  npx playwright test visual-regression --update-snapshots
```

## Troubleshooting

### Common Issues

| Issue                 | Cause                      | Solution                                |
| --------------------- | -------------------------- | --------------------------------------- |
| Tests fail only in CI | Font rendering differences | Generate baselines in CI                |
| Flaky tests           | Animations/timing          | Add `waitForTimeout()` or mask elements |
| Large diffs           | Viewport differences       | Check viewport configuration            |
| Missing baselines     | First run                  | Run with `--update-snapshots`           |

### Debugging

```bash
# Run with debug mode
DEBUG=pw:api npx playwright test visual-regression

# Run in headed mode to watch
npx playwright test visual-regression --headed

# Run specific test
npx playwright test visual-regression -g "game board"
```

## Adding New Visual Tests

1. Add test to `visual-regression.e2e.spec.ts`
2. Run tests to generate initial baselines:
   ```bash
   npx playwright test visual-regression -g "your test name" --update-snapshots
   ```
3. Review generated screenshots in `__snapshots__/`
4. Commit the test and baselines together

### Example Test

```typescript
test('new component appearance', async ({ page }) => {
  // Setup - navigate to page
  await page.goto('/path');

  // Wait for component to be ready
  const component = page.getByTestId('my-component');
  await expect(component).toBeVisible();

  // Optional: wait for animations
  await page.waitForTimeout(500);

  // Capture screenshot
  await expect(component).toHaveScreenshot('my-component.png');
});
```

## Related Documentation

- [Playwright Visual Comparisons](https://playwright.dev/docs/test-snapshots)
- [Local E2E setup](../../QUICKSTART.md)
- [Playwright config](../../playwright.config.ts)
