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

## Next Candidates (Not Yet Implemented)

- Reduce “developer-centric” sidebar density via a Basic/Advanced toggle (keep diagnostics available without overwhelming new players).
- Migrate more game-end UX copy to rely on `GameEndExplanation` as the single source of truth (HUD/Victory/Teaching consistency).
- Decide whether to delete or wire `src/client/components/KeyboardShortcutsHelp.tsx` to avoid duplicated “help overlay” concepts.
