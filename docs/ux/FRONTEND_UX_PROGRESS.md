# Frontend UX Progress Log

This document tracks shipped UX/GUI improvements so work is not duplicated or lost across iterations.

## 2025-12-11 — Controls & Accessibility Polish

**Shipped**

- Fixed the “`?` does nothing when the board is focused” dead-end by wiring `BoardView` help requests into hosts and avoiding unconditional default-prevention when no handler is present.
- Added consistent global keyboard shortcuts in game hosts:
  - `?` toggles **Board controls & shortcuts**
  - `M` toggles mute
  - `F` toggles fullscreen
  - `R` opens the resign confirmation (backend games only)
- Reduced keyboard friction on the board:
  - Roving tabindex for cells (Tab does not traverse every cell)
  - `Home`/`End` jump-to-start/end navigation
  - Screen-reader cell labels use canonical coordinates via `formatPosition()` (e.g. `a1`)
- Hardened `BoardControlsOverlay` as a proper modal (focus trap, Escape closes, focus restore).

**Primary code paths**

- `src/client/pages/BackendGameHost.tsx`
- `src/client/pages/SandboxGameHost.tsx`
- `src/client/components/BoardView.tsx`
- `src/client/components/BoardControlsOverlay.tsx`
- `src/client/hooks/useKeyboardNavigation.ts`
- `src/client/components/ResignButton.tsx`

**Docs updated**

- `docs/ACCESSIBILITY.md`
- `KNOWN_ISSUES.md` (P1.1 capability updates)

## 2025-12-12 — Sidebar Declutter & Help Consolidation

**Shipped**

- Consolidated the in-game keyboard shortcuts/help surface: removed the unused `KeyboardShortcutsHelp` component/tests so `BoardControlsOverlay` is the single `?` help overlay.
- Reduced sidebar density in both hosts via a persisted “Advanced” toggle:
  - Backend: `ringrift_backend_sidebar_show_advanced` (“Advanced diagnostics”) keeps the move log visible and hides history/evaluation by default.
  - Sandbox: `ringrift_sandbox_sidebar_show_advanced` (“Advanced panels”) hides replays/logs/recording by default; touch controls show on mobile or when advanced panels are open.
- Backend diagnostics log now includes a dedicated Last-Player-Standing entry when the game ends by LPS.

**Primary code paths**

- `src/client/pages/BackendGameHost.tsx`
- `src/client/pages/SandboxGameHost.tsx`
- `src/client/components/BoardControlsOverlay.tsx`

**Docs updated**

- `docs/ux/UX_RULES_WEIRD_STATES_SPEC.md`
- `docs/ux/UX_RULES_TELEMETRY_SPEC.md`
- `CONTRIBUTING.md`
- `docs/planning/IMPROVEMENT.md`
- `docs/planning/WAVE_2025_12.md`

## 2025-12-15 — Spectator UI Polish (P1-UX-02)

**Shipped**

- Enhanced [`SpectatorHUD.tsx`](../../src/client/components/SpectatorHUD.tsx:1) with complete game state display and educational features for non-playing observers:
  - Clear "Spectator Mode" banner with proper ARIA labels (`role="banner"`, `aria-label="Spectator Mode - You are watching this game"`)
  - Live indicator dot for connection status
  - Helpful message: "Moves are disabled while spectating. Use the teaching topics below to learn game mechanics."
  - Current phase, turn number, and move number displayed
  - Current player indicator with ring color
  - Player standings with rings in hand, eliminated rings, and territory count
  - `VictoryConditionsPanel` for educational value about win conditions
  - `TeachingTopicButtons` for quick access to learn game mechanics (movement, capture, lines, territory, victory conditions)
  - `TeachingOverlay` integration for comprehensive in-depth learning
  - Evaluation graph and move analysis panel in collapsible "Analysis & Insights" section
  - Recent moves with annotations
- Verified spectators cannot trigger player actions:
  - SpectatorHUD has no action buttons (no resign, no choice dialogs)
  - BoardView correctly disables all cell interactions when `isSpectator` is true
  - BackendGameHost properly hides ChoiceDialog and ResignButton for spectators

**Primary code paths**

- `src/client/components/SpectatorHUD.tsx` — Enhanced with TeachingOverlay, VictoryConditionsPanel, ARIA labels
- `src/client/components/TeachingOverlay.tsx` — Already supports spectator integration via useTeachingOverlay hook
- `src/client/components/BoardView.tsx` — Properly disables interactions for spectators
- `src/client/pages/BackendGameHost.tsx` — Properly hides player-only UI for spectators

**Tests verified**

- `tests/unit/components/SpectatorHUD.test.tsx` — Passes
- `tests/unit/components/GameHUD.spectator.test.tsx` — Passes
- `tests/unit/GameSession.spectatorFlow.test.ts` — Passes
- `tests/unit/GameSession.spectatorLateJoin.test.ts` — Passes
- All 142 GameHUD/TeachingOverlay tests pass

## 2025-12-15 — Mobile/Touch Ergonomics Improvements (P1-UX-04)

**Shipped**

- Verified and enhanced touch ergonomics for mobile/tablet users per WCAG 2.1 AAA guidelines (44×44px minimum touch targets):
  - **BoardView.tsx**: Board cells already meet 44px minimums (`w-11 h-11` = 44px). Added `onCellLongPress` prop for dedicated long-press handler that shows cell info on touch devices.
  - **SandboxTouchControlsPanel.tsx**: All buttons updated with `min-h-[44px]`, `px-4 py-2.5`, `touch-manipulation`, and `active:scale-[0.98]` for proper touch feedback.
  - **ChoiceDialog.tsx**: All choice option buttons updated with `min-h-[44px]`, `touch-manipulation`, `active:scale-[0.98]`, and proper spacing (`space-y-2`).
- Mobile viewport meta tags properly configured in `index.html`:
  - `viewport-fit=cover` for edge-to-edge display
  - `apple-mobile-web-app-capable` for iOS home screen support
  - `apple-mobile-web-app-status-bar-style="black-translucent"` for status bar styling
  - `theme-color="#0f172a"` for mobile browser chrome theming
- Touch interaction CSS enhancements in `globals.css`:
  - Touch-friendly tap feedback (W3-13): Active state scaling, touch-action: manipulation, visual feedback rings
  - Long-press visual feedback (W3-14): Animations for long-press indicators and cell info tooltips
  - Safe area padding for notches/home indicators: `env(safe-area-inset-*)` support
  - Mobile viewport handling (W3-12): Board scroll containers for oversized boards, 44px minimum touch targets
- Long-press on board cells triggers `onCellLongPress` handler (if provided) or falls back to `onCellContextMenu` for multi-ring placement dialogs

**Primary code paths**

- `src/client/components/BoardView.tsx` — Added `onCellLongPress` prop and updated touch handlers
- `src/client/components/SandboxTouchControlsPanel.tsx` — Touch-friendly button sizing
- `src/client/components/ChoiceDialog.tsx` — Touch-friendly choice options
- `src/client/index.html` — Mobile viewport meta tags
- `src/client/styles/globals.css` — Touch interaction CSS (W3-12, W3-13, W3-14 sections)

**Tests verified**

- 517 tests passed (3 pre-existing failures unrelated to touch ergonomics)
- SandboxGameHost.test.tsx passes
- GameHUD.test.tsx passes
- ChoiceDialog and BoardView component tests pass

## 2025-12-15 — HUD Visual Polish (P1-UX-05)

**Shipped**

- Comprehensive visual polish for `GameHUD.tsx` and `MobileGameHUD.tsx` components with improved visual hierarchy, color contrast, and subtle animations:
  - **Phase Indicator**: Enhanced with `rounded-xl`, `ring-1 ring-white/10`, gradient backgrounds, icon container with `shadow-inner`, improved typography with `font-semibold tracking-tight`, and `transition-all duration-300` for smooth state changes
  - **Player Cards**: Gradient backgrounds (`bg-gradient-to-br from-slate-700 via-slate-700 to-slate-800`) for current player, improved shadow/ring effects (`shadow-lg ring-2 ring-offset-1`), larger color dots (20×20px) with pulse animation for current player
  - **Ring Stats Panel**: Card-style container with `bg-slate-700/50` border, larger font sizes for emphasized numbers, better hover states with `hover:bg-slate-600/70`
  - **Victory Conditions Panel**: Color-coded sections (rose for ring elimination, emerald for territory, purple for LPS), icons (🎯, 🏰, 👑), gradient backgrounds, pill-style badges for thresholds
  - **Compact Score Summary**: Gradient background (`bg-gradient-to-br from-slate-700 via-slate-700/95 to-slate-800`), individual player rows with colored badges, emoji icons (⚔ for rings, 🏰 for territory), color-coded stat pills
  - **Weird State Banner**: Gradient backgrounds matching tone (amber for warning, rose for danger, blue for info), icon containers with tone-matched backgrounds, pill-style badges for weird state types
  - **LPS Tracking Indicator**: Gradient styling, larger progress dots (w-3.5 h-3.5), "Victory imminent!" pulsing animation when a player nears LPS victory
  - **Mobile Components**: Matching visual polish applied to all `MobileGameHUD` sub-components for consistency

**Visual Design Principles Applied**

1. **Visual Hierarchy**: Phase indicator is most prominent, followed by current player, then supporting information
2. **Color Contrast**: Improved accessibility with better contrast ratios using white/10 rings and gradient overlays
3. **Smooth Transitions**: `transition-all duration-300` on interactive elements for polished feel
4. **Consistent Spacing**: Standardized padding (px-4 py-3 for indicators, space-y-2 for sections)
5. **Subtle Depth**: Layered shadows and rings create visual depth without being distracting
6. **Animation**: Pulse animation for current player indicator and victory imminent states

**Primary code paths**

- `src/client/components/GameHUD.tsx` — Main HUD with all visual polish updates
- `src/client/components/MobileGameHUD.tsx` — Mobile-optimized HUD with matching polish

**Tests verified**

- All 17 GameHUD/MobileGameHUD test suites pass (120 tests total)
- Snapshot test updated to reflect visual changes
- Test assertions updated to match new visual output (icons instead of text labels in score summary)

## Next Candidates (Not Yet Implemented)

- Migrate more game-end UX copy to rely on `GameEndExplanation` as the single source of truth (HUD/Victory/Teaching consistency).
