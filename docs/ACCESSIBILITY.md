# RingRift Accessibility Guide

This guide documents the accessibility features in RingRift, including keyboard navigation, screen reader support, and visual accessibility options.

---

## Table of Contents

1. [Keyboard Navigation](#keyboard-navigation)
2. [Screen Reader Support](#screen-reader-support)
3. [Visual Accessibility](#visual-accessibility)
4. [Quick Reference](#quick-reference)

---

## Keyboard Navigation

RingRift is fully navigable using only a keyboard. Press **?** at any time during a game to open the **Board controls & shortcuts** overlay.

### Board Navigation

| Key                    | Action                                  |
| ---------------------- | --------------------------------------- |
| **↑ ↓ ← →**            | Navigate between board cells            |
| **Enter** or **Space** | Select current cell / Confirm action    |
| **Escape**             | Cancel current action / Clear selection |
| **Home**               | Jump to first cell (top-left)           |
| **End**                | Jump to last cell (bottom-right)        |

### Game Actions

| Key   | Action                                |
| ----- | ------------------------------------- |
| **?** | Show board controls & shortcuts       |
| **R** | Resign from game (backend games only) |
| **M** | Toggle sound/mute                     |
| **F** | Toggle fullscreen                     |

### Dialog Navigation

When a dialog (like choice selection or confirmation) is open:

| Key                    | Action                        |
| ---------------------- | ----------------------------- |
| **↑ ↓**                | Navigate between options      |
| **Enter** or **Space** | Select focused option         |
| **Tab**                | Move focus between elements   |
| **Escape**             | Close dialog (if cancellable) |

### General Navigation

| Key           | Action                               |
| ------------- | ------------------------------------ |
| **Tab**       | Move to next interactive element     |
| **Shift+Tab** | Move to previous interactive element |

### Focus Indicators

All interactive elements have visible focus indicators. When you navigate between board cells with the keyboard, the focused cell is outlined in amber.

---

## Screen Reader Support

RingRift includes comprehensive screen reader support with ARIA labels and live regions.

### Game Board

The game board is marked as a `grid` with comprehensive labels:

- **Board container**: Announces board type, dimensions, and navigation instructions
- **Each cell**: Announces position (e.g., "a1"), stack contents, and whether it's a valid move target
- **Selection**: Live announcements when selecting cells and showing valid moves

### Game HUD

The Heads-Up Display includes:

- **Phase indicator**: Announces current phase (e.g., "Movement phase - Click a stack to move") with `role="status"`
- **Player cards**: Announce player name, stats, turn status, and ring counts
- **Decision banners**: Use `aria-live` regions with appropriate urgency levels (polite/assertive)
- **Spectator info**: Announces spectator mode and count

### Live Announcements

Game state changes are announced automatically:

- **Turn changes**: "It's now Player 2's turn"
- **Phase transitions**: "Entering territory processing phase"
- **Moves and captures**: "Player 1 moved from E4 to E6", "Player 2 captured 3 rings at D5"
- **Victory/defeat**: "Player 1 wins by ring elimination!"
- **Timer warnings**: Assertive announcements when decision time is running low

### Recommended Screen Readers

RingRift has been tested with:

- **NVDA** (Windows) - Recommended
- **VoiceOver** (macOS/iOS)
- **JAWS** (Windows)
- **Orca** (Linux)

---

## Visual Accessibility

### Accessibility Settings

Access the accessibility settings panel from the game menu to configure:

1. **High Contrast Mode**: Stronger borders and increased color differentiation
2. **Color Vision Mode**: Colorblind-friendly palettes for deuteranopia, protanopia, and tritanopia
3. **Reduced Motion**: Disables animations (also respects system preference)
4. **Large Text**: Increases font sizes for better readability

### Color Palettes

RingRift uses distinct color palettes optimized for different color vision deficiencies:

#### Normal Vision

| Player   | Color              |
| -------- | ------------------ |
| Player 1 | Emerald (#10b981)  |
| Player 2 | Sky Blue (#0ea5e9) |
| Player 3 | Amber (#f59e0b)    |
| Player 4 | Fuchsia (#d946ef)  |

#### Deuteranopia / Protanopia (Red-Green)

| Player   | Color            |
| -------- | ---------------- |
| Player 1 | Blue (#2563eb)   |
| Player 2 | Orange (#ea580c) |
| Player 3 | Cyan (#0891b2)   |
| Player 4 | Violet (#7c3aed) |

#### Tritanopia (Blue-Yellow)

| Player   | Color            |
| -------- | ---------------- |
| Player 1 | Pink (#db2777)   |
| Player 2 | Cyan (#06b6d4)   |
| Player 3 | Lime (#84cc16)   |
| Player 4 | Orange (#f97316) |

### Pattern Differentiation

In addition to colors, colorblind modes add subtle patterns to player indicators:

- **Player 1**: Diagonal stripes
- **Player 2**: Dots
- **Player 3**: Horizontal stripes
- **Player 4**: Crosshatch

### High Contrast Mode

When enabled, high contrast mode provides:

- **2px borders** on all elements (up from 1px)
- **Stronger focus outlines** (3px amber outline)
- **Increased text contrast** on dark backgrounds
- **Visible button borders**

### Reduced Motion

Reduced motion mode:

- Disables all CSS animations and transitions
- Respects the system `prefers-reduced-motion` preference automatically
- Can be manually overridden in settings

### Text Scaling

RingRift supports browser zoom up to 200% while maintaining usability. Large text mode additionally:

- Increases base font size to 18px
- Scales all text sizes proportionally
- Increases line height for readability

### Touch Targets

All interactive elements have minimum touch targets of 44×44 pixels, meeting WCAG 2.1 AA requirements.

---

## Quick Reference

### Keyboard Shortcuts Summary

```
Board:      ↑↓←→ = Navigate | Enter/Space = Select | Esc = Cancel | Home/End = Jump
Game:       ? = Help | R = Resign (backend) | M = Mute | F = Fullscreen
Dialogs:    ↑↓ = Options | Enter = Confirm | Tab = Focus | Esc = Close
```

### ARIA Roles Used

| Element          | Role                      | Purpose                       |
| ---------------- | ------------------------- | ----------------------------- |
| Board container  | `grid`                    | Identifies the game board     |
| Board cells      | `gridcell`                | Individual playable positions |
| Phase indicator  | `status`                  | Current game phase            |
| Decision banner  | `status` with `aria-live` | Time-sensitive decisions      |
| Player cards     | `region`                  | Player information            |
| Spectator banner | `status`                  | Spectator mode indication     |

### Accessibility Settings Location

1. Open the game menu (☰ icon or keyboard)
2. Navigate to "Settings"
3. Select "Accessibility"
4. Adjust preferences as needed

Settings are saved to your browser and persist across sessions.

---

## Reporting Accessibility Issues

If you encounter any accessibility barriers while using RingRift, please report them:

1. Open a GitHub issue with the `accessibility` label
2. Describe the barrier you encountered
3. Include your assistive technology (screen reader, etc.) and browser
4. Provide steps to reproduce if possible

We're committed to making RingRift accessible to all players.

---

## Technical Implementation

For developers contributing to RingRift, see:

- [`src/client/contexts/AccessibilityContext.tsx`](../src/client/contexts/AccessibilityContext.tsx) - Accessibility preferences management
- [`src/client/hooks/useKeyboardNavigation.ts`](../src/client/hooks/useKeyboardNavigation.ts) - Keyboard navigation hooks
- [`src/client/components/ScreenReaderAnnouncer.tsx`](../src/client/components/ScreenReaderAnnouncer.tsx) - Live region announcements
- [`src/client/styles/accessibility.css`](../src/client/styles/accessibility.css) - Accessibility CSS utilities
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) - Accessibility guidelines for contributors
