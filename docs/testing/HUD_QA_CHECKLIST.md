# Game HUD – Manual QA Checklist

This checklist is for **lightweight, high‑signal QA** of the Game HUD after UX changes (e.g. contextual tooltips, visual hierarchy tweaks), focusing on _real gameplay flows_ rather than isolated component stories.

It covers both primary HUD entry points:

- `BackendGameHost` – full backend game with real session state.
- `SandboxGameHost` – adapter/view‑model path powered by the sandbox orchestrator.

Use this after any HUD change that touches:

- Phase indicator behaviour or copy.
- Turn vs Move / Game Progress display.
- Victory conditions panel.
- Spectator vs active‑player HUD behaviour.

---

## 0. Pre‑requisites

1. **Environment up and running**
   - Backend API + WebSocket server running (follow `QUICKSTART.md`).
   - Frontend dev server running (typically `npm run dev`).

2. **Test data / accounts**
   - At least **two browser identities** available (e.g. normal window + incognito, or two browsers) so you can check spectator and opponent views.
   - One account or session with permission to create games.

3. **Starting points**
   - **Backend host path:** Game page reachable via `BackendGameHost` (normal multiplayer game).
   - **Sandbox host path:** Game page reachable via `SandboxGameHost` (local sandbox game creation).

You do **not** need to inspect code for this checklist; it’s intended for repeated manual smoke passes.

---

## 1. BackendGameHost – Active Player HUD Pass

Goal: Verify that the HUD correctly explains **phase**, **turn vs move**, and **victory conditions** while you are the active player in a real backend game.

### 1.1 Create and start a game

- [ ] Start a standard multiplayer game via the normal lobby / game creation UI.
- [ ] Confirm the **Game HUD** renders without console errors.
- [ ] Note the **current phase** (e.g. Ring Placement, Movement, etc.).

### 1.2 Phase indicator + tooltip (active player)

- [ ] Locate the **phase indicator card** (e.g. “Movement Phase”).
- [ ] Hover or focus the **`?` tooltip icon** next to the phase label.
- [ ] Verify the tooltip shows:
  - [ ] First line: **phase description** that matches what the game is actually asking you to do.
  - [ ] A blank line (visual spacing).
  - [ ] A third line beginning with **`On your turn:`** that gives concrete next‑action guidance (e.g. “Select your stack, then click a destination.”).
- [ ] Confirm the inline **action hint pill** (if visible) is consistent with the tooltip guidance.

### 1.3 Victory conditions panel + tooltips

- [ ] Locate the **Victory** panel in the HUD.
- [ ] Confirm the three bullets are present:
  - [ ] **Elimination** – concise one‑line description.
  - [ ] **Territory** – concise one‑line description.
  - [ ] **Last Player Standing** – concise one‑line description.
- [ ] For each bullet, hover the associated `?` icon and verify:
  - [ ] Tooltip copy is **multi‑line** and explains _how that victory is actually triggered_.
  - [ ] Wording matches the mental model from `ringrift_complete_rules.md` / in‑game behaviour (no obvious contradictions).
  - [ ] Tooltips appear and disappear correctly on hover/focus/blur.

### 1.4 Turn vs Move / Game Progress

- [ ] Locate the **Turn / Move** information in the HUD (e.g. `Turn 3 · Move 7`).
- [ ] Confirm that after each completed move:
  - [ ] **Move count** increments as expected.
  - [ ] **Turn count** only increments after a full cycle of players completes.
- [ ] If a Turn vs Move tooltip is present in this path, briefly check that its explanation matches observed behaviour.

### 1.5 Basic time / urgency cues (baseline)

Even before Wave 8.2, ensure there are **no regressions** in:

- [ ] Any visible player clocks (if time controls are enabled).
- [ ] Any decision‑phase countdown banners.
- [ ] No overlapping or truncated time labels when the browser window is narrowed moderately.

Use this as a baseline for later time‑pressure UX improvements.

---

## 2. BackendGameHost – Spectator HUD Pass

Goal: Verify that a **spectator** sees appropriate HUD copy, especially in phase tooltips and banners.

### 2.1 Join as spectator

- [ ] From a second browser/identity, join the same game **after it has started**.
- [ ] Confirm you are treated as a **spectator** (no legal moves available).

### 2.2 Phase indicator + tooltip (spectator)

- [ ] Locate the **phase indicator** on the spectator client.
- [ ] Hover/focus the `?` icon and verify the tooltip shows:
  - [ ] First line: same **phase description** as the active player.
  - [ ] A blank line.
  - [ ] A third line beginning with **`Spectators:`** that describes what is happening from a watcher’s perspective (e.g. “Player is placing rings on the board.”).
- [ ] Confirm there is **no misleading “On your turn” copy** for spectators.

### 2.3 Spectator banners / status

- [ ] Verify any **spectator banner** or label is visible and clearly indicates that this client is not an active participant.
- [ ] Check that basic HUD layout (phase, victory panel, event log) still reads sensibly on the spectator view.

---

## 3. SandboxGameHost – Adapter/View‑Model Path Pass

Goal: Exercise the **modern HUD path** (via `HUDViewModel` and `PhaseViewModel`) to ensure the same semantics hold when the sandbox is driving state.

### 3.1 Start a sandbox game

- [ ] Open the **Sandbox** host entry point and start a local game (e.g. default scenario).
- [ ] Verify the HUD renders and uses the **view‑model‑based layout** (modern HUD path).

### 3.2 Phase indicator + tooltip (active sandbox player)

- [ ] Repeat the **Phase tooltip** checks from section 1.2 while taking a few moves in the sandbox.
- [ ] Confirm that `On your turn:` guidance remains correct as you move between phases (e.g. placement → movement → territory/cleanup).

### 3.3 Victory conditions panel + tooltips

- [ ] Verify the **Victory** panel and its tooltips behave identically in the sandbox HUD path:
  - [ ] Same three bullets, same triggers.
  - [ ] Tooltips show the same copy and semantics.

### 3.4 Turn vs Move correctness in sandbox

- [ ] Perform a sequence of moves involving multiple players (e.g. 2P or 3P sandbox scenario).
- [ ] Confirm **Turn** and **Move** semantics match those observed in the backend game:
  - [ ] Moves increment per action.
  - [ ] Turns increment per full player cycle.

---

## 4. Small‑Window / Layout Sanity Checks

Goal: Catch obvious layout regressions when the viewport is narrower than your primary dev monitor.

- [ ] Gradually **shrink the browser window width** to a typical laptop size.
- [ ] Ensure that:
  - [ ] Phase indicator remains readable (no clipped labels or icons).
  - [ ] Tooltips appear inside the viewport and are not completely off‑screen.
  - [ ] Victory panel text wraps reasonably and `?` icons stay visually associated with their bullets.
  - [ ] No text overlaps or unreadable color combinations appear.

Document any issues as GitHub issues or in `TODO.md` with clear repro steps.

---

## 5. Capturing Findings

After running this checklist for a new HUD change:

- [ ] Record any **copy mismatches** between tooltips and the actual rules in:
  - `TODO.md` or
  - the relevant rules documentation file (e.g. `ringrift_complete_rules.md`) as inline TODO comments.
- [ ] File issues (or add TODOs) for any of the following:
  - [ ] Tooltip content that becomes stale when rules logic changes.
  - [ ] Layout problems (clipping, overlap) at common resolutions.
  - [ ] Spectator HUD states that are confusing or misleading.
- [ ] If all checks pass, note the date and commit hash in `CURRENT_STATE_ASSESSMENT.md` or your release notes so you have a historical point where HUD UX was validated.

This file is intentionally **manual and high‑level**. It should evolve as new HUD features land (e.g. time‑pressure cues, onboarding coachmarks) so that each new UX slice also carries a clear QA ritual.
