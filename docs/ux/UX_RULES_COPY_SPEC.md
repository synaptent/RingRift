# UX Rules Copy Spec (HUD & Sandbox)

> Canonical UX copy for RingRift client surfaces that explain rules in HUD, Sandbox, TeachingOverlay, Onboarding, and curated scenarios. This spec is semantics-first and must remain aligned with RR‑CANON and the current engine.

## 1. Purpose and scope

This document defines **canonical UX copy** for rules-related client surfaces. It is the single reference for:

- HUD victory conditions and ring/territory stats.
- Sandbox phase copy (movement, capture, chain, lines, territory).
- TeachingOverlay topics for movement, capture, chains, lines, territory, and victory.
- Onboarding victory summary.
- Curated sandbox scenarios' `rulesSnippet` text.

Semantics must always match:

- [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1)
- [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1)
- [`ringrift_compact_rules.md`](ringrift_compact_rules.md:1)
- [`docs/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/ACTIVE_NO_MOVES_BEHAVIOUR.md:1)
- [`docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:1)

## 2. Terminology & invariants

**Rings vs markers vs territory**

- _Ring_ – a physical piece on a stack. Still “in play” unless eliminated.
- _Marker_ – a marker token on a space (used to form lines). Lines are **markers only**, never rings.
- _Territory space_ – a collapsed space you permanently own. Territory victory counts territory spaces.

**Elimination vs capture**

- _Captured ring_ – taken by overtaking capture and moved to the **bottom** of your stack. It **stays in play** and does **not** directly count toward elimination victory.
- _Eliminated ring_ – permanently removed from play and credited to the player who caused it (via movement onto markers, line rewards, territory processing, forced elimination, etc).

**Global elimination threshold (R060–R061)**

- Per RR-CANON-R061, the victory threshold is calculated as: `round(ringsPerPlayer × (1/3 + 2/3 × (numPlayers - 1)))`.
- For 2-player games, threshold = `ringsPerPlayer`: 18 (8×8), 48 (19×19), 72 (hexagonal).
- For 3-player games, threshold increases: 30 (8×8), 80 (19×19), 120 (hexagonal).
- For 4-player games, threshold increases further: 42 (8×8), 112 (19×19), 168 (hexagonal).
- A player wins by **Ring Elimination** when their credited eliminated rings reach or exceed this threshold.

**Territory threshold (R062, R140–R145)**

- Let `totalSpaces` be the board’s cell count.
- Territory victory threshold is `floor(totalSpaces / 2) + 1`.
- Territory victory uses **territory spaces**, not raw markers or stacks.

**Real moves vs forced elimination (R070–R072, R170–R173)**

- _Real moves_ are placements, movements, and captures.
- Forced elimination and automatic line/territory processing are **not** real moves for LPS.
- Last‑Player‑Standing is determined only by availability of real moves across a full round.

**Elimination cost distinctions (RR-CANON-R022)**

There are three distinct elimination contexts with different costs and eligible targets:

| Context                  | Cost            | Eligible Stacks                                                                             |
| ------------------------ | --------------- | ------------------------------------------------------------------------------------------- |
| **Line Processing**      | 1 ring from top | Any controlled stack (including standalone rings)                                           |
| **Territory Processing** | Entire cap      | Multicolor stacks OR single-color stacks height > 1 only. **Standalone rings NOT eligible** |
| **Forced Elimination**   | Entire cap      | Any controlled stack (including standalone rings)                                           |

- _Line processing_ is the most permissive: any stack you control can lose one ring from the top.
- _Territory processing_ requires an "entire cap" from eligible stacks only—you cannot process a territory if your only outside stacks are standalone rings.
- _Forced elimination_ also removes entire caps but accepts any controlled stack, including standalone rings.

## 3. Victory conditions copy

This section defines the canonical labels and one‑liners used by:

- [`VictoryConditionsPanel`](src/client/components/GameHUD.tsx:843)
- HUD summary rows in [`GameHUD.GameHUDFromViewModel()`](src/client/components/GameHUD.tsx:984)
- Victory messaging in [`VictoryModal`](src/client/components/VictoryModal.tsx:365) and [`toVictoryViewModel()`](src/client/adapters/gameViewModels.ts:1311)

### 3.1 Ring Elimination

**HUD label**

- `"Ring Elimination"`

**HUD one‑liner**

- `"Win by eliminating a number of rings equal to the starting ring supply per player."`

**Tooltip (multi‑line)**

- Line 1: `"You win Ring Elimination when your eliminated rings reach or exceed the starting ring supply per player (ringsPerPlayer)."`
- Line 2: `"Only *eliminated* rings count – captured rings you carry in stacks still remain in play."`
- Line 3: `"Eliminations can come from movement onto markers, line rewards, territory processing, or forced elimination."`

**TeachingOverlay victory topic – elimination**
Short description (used in [`TeachingOverlay.TEACHING_CONTENT.victory_elimination`](src/client/components/TeachingOverlay.tsx:96)):

- `"Win by eliminating a number of rings equal to the starting ring supply per player (ringsPerPlayer) – not just one opponent’s set. Eliminated rings are permanently removed; captured rings you carry in stacks do not count toward this threshold."`

### 3.2 Territory Control

**HUD label**

- `"Territory Control"`

**HUD one‑liner**

- `"Win by controlling more than half of all board spaces as territory."`

**Tooltip (multi‑line)**

- Line 1: `"Territory spaces are collapsed cells you permanently own."`
- Line 2: `"If your territory spaces exceed 50% of all board spaces during territory processing, you win immediately."`
- Line 3: `"Claiming a region usually requires eliminating rings from a stack you control outside that region (mandatory self‑elimination cost)."`

**TeachingOverlay victory topic – territory**
Description (used in [`TeachingOverlay.TEACHING_CONTENT.victory_territory`](src/client/components/TeachingOverlay.tsx:107)):

- `"Win by owning more than half of all board spaces as Territory. Territory comes from collapsing marker lines and resolving disconnected regions, and once a space becomes Territory it can’t be captured back."`

### 3.3 Last Player Standing (LPS)

**HUD label**

- `"Last Player Standing"`

**HUD one‑liner**

- `"Win when, for two consecutive rounds, you are the only player with any real moves (placements, movements, or captures)."`

**Tooltip (multi‑line)**

- Line 1: `"Real moves are placements, movements, and captures – forced elimination and automatic line/territory processing do not count."`
- Line 2: `"Last Player Standing requires two consecutive full rounds where you have and take at least one real action while all other players have none."`
- Line 3: `"If any other player regains a real move before both rounds complete, the LPS condition resets and victory is not declared."`

**TeachingOverlay victory topic – stalemate / LPS**
Description (used in [`TeachingOverlay.TEACHING_CONTENT.victory_stalemate`](src/client/components/TeachingOverlay.tsx:118)):

- `"Last Player Standing happens when, for TWO consecutive complete rounds, you are the only player who can still make real moves (placements, movements, or captures). In the first round you must have and take at least one real action while all others have none; in the second round the condition must persist. Forced eliminations and automatic territory processing do not count as real actions for LPS."`

## 4. Movement semantics

**Canonical rule (R090–R092)**

- A stack with height `H` moves in a straight line.
- It must move **at least** `H` spaces and may move farther if the path stays legal.
- The path cannot pass through other stacks or collapsed territory spaces; markers do not block movement.
- The landing cell must not contain a stack or collapsed territory; it may contain a marker.
- If you land on a marker, the top ring of your moving stack is eliminated and credited to you.

**TeachingOverlay – Stack Movement description**
Used in [`TeachingOverlay.TEACHING_CONTENT.stack_movement`](src/client/components/TeachingOverlay.tsx:36):

- `"Move a stack you control (your ring on top) in a straight line at least as many spaces as the stack’s height. You can keep going farther as long as the path has no stacks or territory spaces blocking you; markers are allowed and may eliminate your top ring when you land on them."`

**Sandbox phase copy – Movement**
Used in [`SandboxGameHost.PHASE_COPY.movement.summary`](src/client/pages/SandboxGameHost.tsx:238):

- `"Pick a stack and move it in a straight line at least as far as its height, and farther if the path stays clear (stacks and territory block; markers do not)."`

## 5. Capture and chain capture semantics

**Basic overtaking capture (R101–R102)**

- Capture is always a **jump** over exactly one target stack along a straight line to a landing cell.
- Your stack’s cap height must be ≥ the target stack’s cap height.
- The path from origin to landing may not cross other stacks or collapsed territory spaces (other than the single target stack).
- The landing cell must be empty or contain markers (no stack, no collapsed territory).
- You remove the **top ring** from the target stack and place it on the **bottom** of your stack: captured rings stay in play.

**Chain capture (R103)**

- Starting a capture is **optional**.
- Once you take a capture segment, chain continuation is **mandatory**: if another capture exists from your new position, you must keep capturing.
- When multiple capture directions are available, you choose which capture to take next.
- The chain ends only when no legal capture segments remain.

**TeachingOverlay – Capturing description**
Used in [`TeachingOverlay.TEACHING_CONTENT.capturing`](src/client/components/TeachingOverlay.tsx:48):

- `"To capture, jump over an adjacent opponent stack in a straight line and land on the empty space just beyond it. You take the top ring from the jumped stack and add it to the bottom of your own stack. Captured rings stay in play – only later *eliminations* move rings out of the game."`

**TeachingOverlay – Chain Capture description**
Used in [`TeachingOverlay.TEACHING_CONTENT.chain_capture`](src/client/components/TeachingOverlay.tsx:60):

- `"If your capturing stack can jump again after a capture, you are in a chain capture. Starting the first capture is optional, but once the chain begins you must keep capturing as long as any capture is available. When several jumps exist, you choose which target to take next."`

**Sandbox phase copy – Capture / Chain Capture**
Used in [`SandboxGameHost.PHASE_COPY.capture`](src/client/pages/SandboxGameHost.tsx:244) and [`SandboxGameHost.PHASE_COPY.chain_capture`](src/client/pages/SandboxGameHost.tsx:249):

- Capture phase label: `"Capture"`
- Capture summary: `"Start an overtaking capture by jumping over an adjacent stack and landing on an empty or marker space beyond it."`
- Chain Capture label: `"Chain Capture"`
- Chain Capture summary: `"Continue the capture chain: you must keep jumping while any capture exists, but you choose which capture direction to take next."`

## 6. Lines and rewards

**Core semantics (R120–R122)**

- Lines are formed by **markers**, not rings.
- A contiguous run of your markers of length ≥ the board's configured `lineLength` is a completed line.
- **Line elimination cost:** Eliminate **one ring** from the top of any stack you control (including height-1 standalone rings). Any controlled stack is an eligible target for line processing.
- **Exact‑length line:** all markers in the line must collapse into your territory, and you must pay the line elimination cost (one ring).
- **Overlength line:** you choose between:
  - (A) Collapse the whole line into territory **and** pay the line elimination cost (one ring), or
  - (B) Collapse only a contiguous segment of length `lineLength` into territory and **skip elimination**.

**TeachingOverlay – Lines description**
Used in [`TeachingOverlay.TEACHING_CONTENT.line_bonus`](src/client/components/TeachingOverlay.tsx:72):

- `"Lines are built from your markers. When a straight line of your markers reaches the minimum length for this board, it becomes a scoring line: you collapse markers in that line into permanent Territory and, on many boards, must pay a ring elimination cost from a stack you control."`

**TeachingOverlay – Lines tips**
Replace any legacy “ring to hand” text with:

- `"Exact‑length lines always collapse fully into Territory and usually require you to eliminate a ring from one of your stacks."`
- `"Overlength lines can trade safety for value: you may collapse a shorter scoring segment with no elimination, or collapse the full line and pay the ring cost."`

**Curated scenarios – line‑focused rulesSnippet template**
For scenarios like `learn.lines.formation.Rules_11_2_Q7_Q20` in [`curated.json`](src/client/public/scenarios/curated.json:175):

- `"When a contiguous line of your markers reaches the minimum scoring length, it immediately becomes a line to resolve. You choose how to reward each completed line: collapse markers into permanent Territory, and on many boards pay a small ring‑elimination cost from one of your stacks. Rings are never returned to hand as a line reward."`

## 7. Territory and disconnected regions

**Core semantics (R140–R145)**

- Territory processing examines **disconnected regions** of stacks and markers for each player.
- When you process a region you control:
  - All spaces in that region become your Territory spaces (collapsed).
  - All rings in the region are eliminated and credited to you.
  - You must pay the **territory elimination cost** from a stack you control **outside** the region.
- **Territory elimination cost:** Eliminate the **entire cap** from an eligible stack. Eligible targets must be either: (1) a **multicolor stack** you control (with other players' rings buried beneath your cap), or (2) a **single-color stack of height > 1** consisting entirely of your colour. **Height-1 standalone rings are NOT eligible for territory processing.**
- Territory victory is checked whenever territory processing completes and your territory spaces exceed the threshold.

**TeachingOverlay – Territory description**
Used in [`TeachingOverlay.TEACHING_CONTENT.territory`](src/client/components/TeachingOverlay.tsx:84):

- `"Territory spaces are collapsed cells that you permanently own. When a disconnected region of your pieces is processed, all of its spaces become your Territory and its rings are eliminated, at the cost of eliminating the entire cap from one of your other stacks (recovery actions pay with a buried ring instead). If your Territory passes more than half of the board, you win immediately."`

**Sandbox phase copy – Territory Processing**
Used in [`SandboxGameHost.PHASE_COPY.territory_processing.summary`](src/client/pages/SandboxGameHost.tsx:258):

- `"Resolve disconnected regions into permanent Territory, eliminating rings in that region and paying any required self‑elimination cost from your other stacks."`

## 8. Sandbox phase‑copy summary (movement, capture, chains, lines, territory)

The canonical short summaries used in [`SandboxGameHost.PHASE_COPY`](src/client/pages/SandboxGameHost.tsx:227):

- **Ring Placement – label**: `"Ring Placement"`
  - summary: `"Place new rings or add to existing stacks while keeping at least one real move available for your next turn."`
- **Movement – label**: `"Movement"`
  - summary: `"Pick a stack and move it in a straight line at least as far as its height, and farther if the path stays clear (stacks and territory block; markers do not)."`
- **Capture – label**: `"Capture"`
  - summary: `"Start an overtaking capture by jumping over an adjacent stack and landing on an empty or marker space beyond it."`
- **Chain Capture – label**: `"Chain Capture"`
  - summary: `"Continue the capture chain: you must keep jumping while any capture exists, but you choose which capture direction to take next."`
- **Line Processing – label**: `"Line Completion"`
  - summary: `"Resolve completed marker lines into Territory and choose whether to take or skip any ring‑elimination reward."`
- **Territory Processing – label**: `"Territory Claim"`
  - summary: `"Evaluate disconnected regions, collapse them into Territory, and pay any required self‑elimination cost; territory wins are checked here."`

## 9. Surface‑ID to component mapping

This section ties each text block in this spec to its implementation surface.

### 9.1 HUD & victory surfaces

- `hud.victory.ring_elimination.label` → [`VictoryConditionsPanel`](src/client/components/GameHUD.tsx:843) elimination row label.
- `hud.victory.ring_elimination.tooltip` → [`VictoryConditionsPanel`](src/client/components/GameHUD.tsx:843) elimination tooltip.
- `hud.victory.territory.label` and `.tooltip` → same component, territory row.
- `hud.victory.lps.label` and `.tooltip` → same component, Last Player Standing row.
- `hud.stats.rings_eliminated.label` →
  - [`RingStats`](src/client/components/GameHUD.tsx:405) elimination label.
  - [`RingStatsFromVM`](src/client/components/GameHUD.tsx:552) elimination label.
  - [`CompactScoreSummary`](src/client/components/GameHUD.tsx:933) "Rings Eliminated" label.
- `victory_modal.ring_elimination.description` → [`getVictoryMessage()`](src/client/adapters/gameViewModels.ts:1408) branch for `ring_elimination`.
- `victory_modal.territory.description` → [`getVictoryMessage()`](src/client/adapters/gameViewModels.ts:1524) branch for `territory_control`.
- `victory_modal.lps.description` → [`getVictoryMessage()`](src/client/adapters/gameViewModels.ts:1524) branch for `last_player_standing`.
- `victory_modal.structural_stalemate.description` → [`getVictoryMessage()`](src/client/adapters/gameViewModels.ts:1524) branch for `game_completed`.
- `victory_modal.table.rings_eliminated.header` → [`FinalStatsTable`](src/client/components/VictoryModal.tsx:108) elimination column header.
- `hud.game_over.structural_stalemate.banner` → [`getGameOverBannerText()`](src/client/utils/gameCopy.ts:3) branch for `game_completed`.

### 9.2 Teaching & onboarding surfaces

- `teaching.ring_placement` → [`TEACHING_CONTENT.ring_placement`](src/client/components/TeachingOverlay.tsx:23).
- `teaching.stack_movement` → [`TEACHING_CONTENT.stack_movement`](src/client/components/TeachingOverlay.tsx:36).
- `teaching.capturing` → [`TEACHING_CONTENT.capturing`](src/client/components/TeachingOverlay.tsx:48).
- `teaching.chain_capture` → [`TEACHING_CONTENT.chain_capture`](src/client/components/TeachingOverlay.tsx:60).
- `teaching.lines` → [`TEACHING_CONTENT.line_bonus`](src/client/components/TeachingOverlay.tsx:72).
- `teaching.territory` → [`TEACHING_CONTENT.territory`](src/client/components/TeachingOverlay.tsx:84).
- `teaching.active_no_moves` → [`TEACHING_CONTENT.active_no_moves`](src/client/components/TeachingOverlay.tsx:98).
- `teaching.forced_elimination` → [`TEACHING_CONTENT.forced_elimination`](src/client/components/TeachingOverlay.tsx:110).
- `teaching.victory_elimination` → [`TEACHING_CONTENT.victory_elimination`](src/client/components/TeachingOverlay.tsx:96).
- `teaching.victory_territory` → [`TEACHING_CONTENT.victory_territory`](src/client/components/TeachingOverlay.tsx:107).
- `teaching.victory_stalemate` → [`TEACHING_CONTENT.victory_stalemate`](src/client/components/TeachingOverlay.tsx:144).
- `onboarding.victory.elimination` → Ring Elimination card in [`OnboardingModal.VictoryStep`](src/client/components/OnboardingModal.tsx:76).

### 9.3 Sandbox & curated scenarios

- `sandbox.phase.movement.summary` → [`PHASE_COPY.movement.summary`](src/client/pages/SandboxGameHost.tsx:238).
- `sandbox.phase.capture.summary` → [`PHASE_COPY.capture.summary`](src/client/pages/SandboxGameHost.tsx:244).
- `sandbox.phase.chain_capture.summary` → [`PHASE_COPY.chain_capture.summary`](src/client/pages/SandboxGameHost.tsx:249).
- `sandbox.phase.line_processing.summary` → [`PHASE_COPY.line_processing.summary`](src/client/pages/SandboxGameHost.tsx:254).
- `sandbox.phase.territory_processing.summary` → [`PHASE_COPY.territory_processing.summary`](src/client/pages/SandboxGameHost.tsx:258).
- `scenario.learn.movement.basics.rulesSnippet` → `learn.movement.basics` in [`curated.json`](src/client/public/scenarios/curated.json:59).
- `scenario.learn.capture.chain.rulesSnippet` → `learn.capture.chain` in [`curated.json`](src/client/public/scenarios/curated.json:117).
- `scenario.learn.lines.formation.rulesSnippet` → `learn.lines.formation.Rules_11_2_Q7_Q20` in [`curated.json`](src/client/public/scenarios/curated.json:175).

**Curated sandbox rules‑concept mapping (runtime `public/scenarios/curated.json`)**

Each curated sandbox scenario loaded by [`loadCuratedScenarios()`](src/client/sandbox/scenarioLoader.ts:112) declares a `rulesConcept` and optional `uxSpecAnchor`:

- `learning.empty.square8` → `rulesConcept: 'board_intro_square8'`
  - Intro to 8×8 board, basic placement and movement.
  - Anchors: §2 Terminology & invariants; §4 Movement semantics.
- `learning.empty.hexagonal` and `learning.fourplayer.start` → `rulesConcept: 'board_intro_hex'`
  - Hex board geometry and 6‑direction adjacency.
  - Anchors: [`RULES_CANONICAL_SPEC.md` board types](RULES_CANONICAL_SPEC.md:46); §4 Movement semantics.
- `learning.capture.setup` → `rulesConcept: 'capture_basic'`, `uxSpecAnchor: "capture.basic"`
  - Single‑segment overtaking capture; captured rings stay in play.
  - Anchors: §5 Capture and chain capture semantics.
- `learning.chain.capture` → `rulesConcept: 'chain_capture_mandatory'`, `uxSpecAnchor: "capture.chain_capture"`
  - Optional start / mandatory continuation chain captures with choice of direction.
  - Anchors: §5 Capture and chain capture semantics.
- `learning.line.almost` → `rulesConcept: 'lines_basic'`, `uxSpecAnchor: "lines.core"`
  - Exact‑length line completion and basic graduated rewards.
  - Anchors: §6 Lines and rewards.
- `learning.territory.disconnect` → `rulesConcept: 'territory_basic'`, `uxSpecAnchor: "territory.core"`
  - Disconnected‑region processing, ring elimination inside the region, and self‑elimination cost.
  - Anchors: §7 Territory and disconnected regions.
- `learning.advanced.multistack` → `rulesConcept: 'stack_height_mobility'`, `uxSpecAnchor: "movement.semantics"`
  - Stack height vs minimum movement distance and marker interaction.
  - Anchors: §4 Movement semantics.
- `learning.midgame.balanced` → `rulesConcept: 'movement_basic'`, `uxSpecAnchor: "movement.semantics"`
  - Standard non‑capture movement and basic tactics in a balanced midgame.
  - Anchors: §4 Movement semantics.
- `learning.endgame.elimination` → `rulesConcept: 'victory_ring_elimination'`, `uxSpecAnchor: "victory.elimination"`
  - Ring Elimination HUD and threshold behaviour.
  - Anchors: §3.1 Ring Elimination.
- `learning.near_victory.territory` → `rulesConcept: 'territory_near_victory'`, `uxSpecAnchor: "victory.territory"`
  - Territory Control victory threshold and immediate win during Territory processing.
  - Anchors: §3.2 Territory Control and §7 Territory and disconnected regions.
- `advanced.multi_phase_turn` → `rulesConcept: 'turn_multi_phase'`, `uxSpecAnchor: "turn.multi_phase.line_then_territory"`
  - Full movement/capture → chain_capture → line_processing → territory_processing turn sequence.
  - Anchors: §4 Turn / Phase / Step Structure and §4.5 Active‑No‑Moves & Forced Elimination Semantics.

All changes to HUD, TeachingOverlay, OnboardingModal, SandboxGameHost, and curated scenarios **must** remain consistent with this document and the canonical rules references in §0.

## 10. Weird States: Active‑No‑Moves, Forced Elimination & Structural Stalemate Banners

This section defines the canonical UX copy for the “weird state” banners and teaching topics that explain:

- Active‑No‑Moves (ANM) in different phases,
- Forced Elimination (FE) sequences,
- Structural stalemate / plateau endings.

These surfaces are **explanatory only** and must not redefine rules semantics; they mirror behaviour specified in:

- [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/ACTIVE_NO_MOVES_BEHAVIOUR.md:1)
- [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:1)
- [`RULES_CONSISTENCY_EDGE_CASES.md`](RULES_CONSISTENCY_EDGE_CASES.md:1)
- [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1)

### 10.1 ANM banners (movement / line / territory)

#### Movement‑phase ANM (no real moves, stacks remain)

HUD banner text (used by [`HUDWeirdStateViewModel`](src/client/adapters/gameViewModels.ts:115) in [`GameHUD`](src/client/components/GameHUD.tsx:943) and [`MobileGameHUD`](src/client/components/MobileGameHUD.tsx:266)):

- **Title (local player)**
  `"You have no legal moves this turn"`
- **Title (other player)**
  `"<Name> has no legal moves"`
- **Body (both)**
  `"You have no legal placements, movements, or captures this turn. Forced elimination will now resolve automatically according to the rulebook."`

Semantics reference:

- This is an ANM state from [`isANMState()`](src/shared/engine/globalActions.ts:1) in the **movement family** of phases (including capture/chain, when applicable).
- “Real moves” are placements, movements, and captures; forced elimination and automatic processing are _not_ real moves for LPS (see §2 “Real moves vs forced elimination”).

#### Line‑processing ANM

HUD banner:

- **Title (local player)**
  `"No legal line actions available"`
- **Title (other player)**
  `"<Name> has no line actions"`
- **Body**
  `"No valid line actions are available. The game will auto-resolve this phase according to the rulebook."`

Semantics note:

- This reflects ANM in `line_processing` – engine has no interactive line-processing choices left and will auto‑resolve.

#### Territory‑processing ANM

HUD banner:

- **Title (local player)**
  `"No legal territory actions available"`
- **Title (other player)**
  `"<Name> has no territory actions"`
- **Body**
  `"No valid territory or self-elimination actions are available. The game will auto-resolve this phase according to the rulebook."`

Semantics note:

- This matches ANM in `territory_processing` – no remaining region‑order choices or self‑elimination moves; engine advances via its automatic territory logic.

### 10.2 Forced Elimination (FE) banners

HUD banner (movement‑family phases where FE is available but no real moves exist):

- **Badge label**
  `"Forced Elimination"`
- **Title**
  `"Forced Elimination"`
- **Body (local player)**
  `"You control stacks but have no legal placements, movements, or captures. A cap will be removed from one of your stacks until a legal move becomes available, following the forced-elimination rules."`
- **Body (other player)**
  `"<Name> controls stacks but has no legal placements, movements, or captures. A cap will be removed from one of their stacks until a legal move becomes available, following the forced-elimination rules."`

Semantics reference:

- FE availability matches the global summary used by invariants:
  - Player has turn material,
  - No legal placements or movement/capture,
  - At least one legal forced‑elimination action.
- Rings removed by FE are **eliminated rings** and count toward elimination victory (see §3.1).

### 10.3 Structural stalemate / plateau banners

HUD banner (when the game ends by plateau / structural stalemate):

- **Badge label**
  `"Structural stalemate"`
- **Title**
  `"Structural stalemate"`
- **Body**
  `"No legal placements, movements, captures, or forced eliminations remain for any player. The game ends by structural stalemate and the winner is chosen by the tiebreak ladder: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers, 4) Who made the last real action."`

Game‑over banner (host‑level banner after dismissing VictoryModal):

- **Game over banner copy**
  - [`getGameOverBannerText()`](src/client/utils/gameCopy.ts:3) for `reason: 'game_completed'`:
  - `"Game over – structural stalemate. Winner decided by tiebreak ladder: territory spaces, eliminated rings (including rings in hand), markers, then last real action."`

VictoryModal description:

- [`getVictoryMessage()`](src/client/adapters/gameViewModels.ts:1524) when `reason === 'game_completed'`:
  - Title: `"🧱 Structural Stalemate"`
  - Description:
    `"No players had any legal placements, movements, captures, or forced eliminations left. The board reached a structural stalemate and the winner was chosen by the tiebreak ladder: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers, 4) Who made the last real action."`

### 10.4 TeachingOverlay topics for weird states

Two dedicated topics explain weird states:

- `teaching.active_no_moves` → [`TEACHING_CONTENT.active_no_moves`](src/client/components/TeachingOverlay.tsx:98)
  - **Title**: `"When you have no legal moves"`
  - **Icon**: `⛔`
  - **Description** (canonical):
    `"Sometimes it is your turn but there are no legal placements, movements, or captures available. This is an Active–No–Moves state: the rules engine will either trigger forced elimination of your stacks, or, if no eliminations are possible, treat you as structurally stuck for Last Player Standing and plateau detection."`
  - **Key tips (aligned with ANM docs)**:
    - Real moves = placements, movements, captures; FE and auto line/territory do **not** count for LPS.
    - If you still control stacks but have no placements or movements, FE will remove caps until a real move exists or you run out of stacks.
    - When no players have real moves _or_ FE available, the game reaches a structural plateau and ends by structural stalemate.

- `teaching.forced_elimination` → [`TEACHING_CONTENT.forced_elimination`](src/client/components/TeachingOverlay.tsx:110)
  - **Title**: `"Forced Elimination (FE)"`
  - **Icon**: `💥`
  - **Description** (canonical):
    `"Forced Elimination happens when you control stacks but have no legal placements, movements, or captures. Caps are removed from your stacks automatically until either a real move becomes available or your stacks are gone. These eliminations are mandatory and follow the rules, not player choice."`
  - **Key tips**:
    - FE eliminations are permanent and count toward Ring Elimination.
    - FE is not a “real move” for Last Player Standing.
    - FE is mandatory once its conditions are met; stack and cap choices follow the engine’s rules, not ad‑hoc UX.

HUD surfaces route weird-state banner help buttons to these topics:

- Desktop HUD: [`GameHUD.WeirdStateBanner` + help](src/client/components/GameHUD.tsx:320) mapped in [`GameHUDFromViewModel`](src/client/components/GameHUD.tsx:1114).
- Mobile HUD: [`MobileWeirdStateBanner` + help](src/client/components/MobileGameHUD.tsx:45) mapped in [`MobileGameHUD`](src/client/components/MobileGameHUD.tsx:266).

These mappings must remain stable when copy is updated so that context‑sensitive “?” help consistently opens the correct explanation for ANM, FE, or structural stalemate.
