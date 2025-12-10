# UX Rules Weird States Spec (ANM, Forced Elimination, Structural Stalemate, LPS)

> **Doc Status (2025-12-05): Draft – UX mapping for W‑UX‑2**
>
> **Role:** Define canonical internal reason codes, HUD/VictoryModal/TeachingOverlay copy, and rules‑doc anchors for the most confusing rules outcomes:
>
> - Active‑No‑Moves (ANM) shapes across phases.
> - Forced Elimination (FE) sequences.
> - Structural stalemate / plateau endings.
> - Last‑Player‑Standing (LPS) early victory semantics.
>
> **Inputs:**
>
> - Canonical rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:232) (R070–R072, R100, R170–R173, R190–R191).
> - Behavioural catalogue [`docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:1).
> - Edge‑case report [`docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:184).
> - Final audit [`archive/FINAL_RULES_AUDIT_REPORT.md`](archive/FINAL_RULES_AUDIT_REPORT.md:136).
> - Canonical copy baselines in [`docs/UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:331).
> - Telemetry schema in [`docs/UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:1).
> - Concept navigation index in [`docs/UX_RULES_CONCEPTS_INDEX.md`](docs/UX_RULES_CONCEPTS_INDEX.md:1) to keep reason codes aligned with rules anchors, teaching topics, and telemetry labels.

This spec is **semantics‑first**. It does not define UI layout or styling; instead, it pins:

- Stable **reason codes** that identify weird / complex rule outcomes.
- Canonical short **HUD banner titles**.
- Canonical full **explanations** for VictoryModal and TeachingOverlay.
- **Rules doc anchors** for deep links.
- Mapping tables for HUD / VictoryModal / TeachingOverlay and `rules_context` used in [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:172).

Code‑mode tasks are expected to:

- Map existing engine / UI enums (e.g. `WeirdStateReason`, game result reasons) to these reason codes.
- Wire the correct copy keys and TeachingOverlay topics for each reason.
- Emit telemetry with `rules_context` derived from these mappings.

---

## 0. Iteration Log & Improvement History

This spec is part of the rules‑UX improvement loop described in [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:24). Concrete changes to weird‑state reason codes, HUD/VictoryModal copy, and TeachingOverlay routing are recorded in numbered iteration files under `docs/ux/rules_iterations/`:

- [`UX_RULES_IMPROVEMENT_ITERATION_0001.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0001.md:1) – Initial hotspot‑oriented plan for ANM/FE loops, structural stalemate, and mini‑regions.
- [`UX_RULES_IMPROVEMENT_ITERATION_0002.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0002.md:1) – Backfilled record for W1–W5 work that aligned game‑end explanations (HUD + VictoryModal + TeachingOverlay) and telemetry with ANM/FE, structural stalemate, and territory mini‑regions.
- [`UX_RULES_IMPROVEMENT_ITERATION_0003.md`](docs/ux/rules_iterations/UX_RULES_IMPROVEMENT_ITERATION_0003.md:1) – Backfilled record for W1–W5 work on territory mini‑regions (Q23) and capture‑chain teaching flows.

Future rules‑UX iterations SHOULD continue this numbering (0004, 0005, …) and update both this spec and the iteration files together when introducing or refining weird‑state reason codes, copy, or telemetry mapping.

## 1. Inventory of Weird / Complex States

The following states are considered “weird” or high‑confusion from a UX perspective:

1. **ANM‑Movement‑FE** – Active‑No‑Moves in the movement family of phases (movement / capture / chain_capture) where:
   - The player controls stacks,
   - Has no legal placements, moves, or captures,
   - Forced elimination is available and will fire ([`RR‑CANON‑R072`](RULES_CANONICAL_SPEC.md:210), [`RR‑CANON‑R100`](RULES_CANONICAL_SPEC.md:443), ANM‑SCEN‑01 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:39)).

2. **ANM‑Line‑Auto** – ANM in the line‑processing phase:
   - `currentPhase == line_processing`,
   - No `process_line` / `choose_line_reward` decisions remain, but game is still ACTIVE,
   - Engine auto‑advances (ANM‑SCEN‑05).

3. **ANM‑Territory‑Auto** – ANM in the territory‑processing phase:
   - `currentPhase == territory_processing`,
   - No `process_territory_region` / `eliminate_rings_from_stack` decisions remain,
   - Engine auto‑advances or terminates (ANM‑SCEN‑04).

4. **FE‑Sequence** – Visible forced‑elimination sequence where a player repeatedly loses caps due to FE (often in endgame), without understanding why caps are disappearing.

5. **Structural‑Stalemate** – **Global stalemate / plateau**:
   - No legal placements, moves, captures, or forced eliminations for any player,
   - Game ends via tie‑break ladder (territory → eliminated rings → markers → last actor) per [`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:619).

6. **LPS‑Early‑Win** – **Last‑Player‑Standing** early victory:
   - Exactly one player has “real actions” (placements, moves, captures) over a full round,
   - Others are permanently stuck or only have FE,
   - Canonical rules treat this as a distinct victory condition ([`RR‑CANON‑R172`](RULES_CANONICAL_SPEC.md:603); ANM‑SCEN‑07).

This document defines dedicated **reason codes** and UX for these six categories. Additional territory chain‑reaction or line‑+‑territory cascades are visually dramatic but largely covered by standard victory copy; they are **not** treated as separate weird‑state reasons here.

---

## 2. Canonical Reason Codes

### 2.1 Reason Code Catalogue

Each reason code is intended to be stable across hosts (backend, sandbox, Python) and used in:

- HUD weird‑state banners.
- VictoryModal explanations (where applicable).
- TeachingOverlay topic routing.
- Telemetry `payload.reason_code` and `rules_context` fields ([`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:218)).

#### RWS‑001 ANM_MOVEMENT_FE_BLOCKED

- **Description:** Current player has turn‑material (stacks and/or rings in hand) but **no legal placements, movements, or captures** in the movement family phases; forced elimination will remove caps until a legal move exists or stacks are gone. In canonical recordings this always appears as one or more explicit `forced_elimination` moves in the dedicated `forced_elimination` phase, not as a silent side effect.
- **Rules references:**
  - [`RR‑CANON‑R072`](RULES_CANONICAL_SPEC.md:210) (forced‑elimination entry).
  - [`RR‑CANON‑R100`](RULES_CANONICAL_SPEC.md:443) (forced elimination when blocked).
  - ANM‑SCEN‑01 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:39).
- **Canonical `rules_context`:** `anm_forced_elimination` (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:145)).

#### RWS‑002 ANM_LINE_NO_ACTIONS

- **Description:** No remaining interactive line decisions (no `process_line` or `choose_line_reward` moves) in `line_processing`; engine must auto‑advance to territory or victory.
- **Rules references:**
  - [`RR‑CANON‑R120–R122`](RULES_CANONICAL_SPEC.md:495) (lines and rewards).
  - [`RR‑CANON‑R204`](RULES_CANONICAL_SPEC.md:271) (line‑processing exit must not leave ANM states).
  - ANM‑SCEN‑05 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:121).
- **Canonical `rules_context`:** `line_reward_exact` or `line_reward_overlength` (both map to this reason code).
- **UX surface guidance:** Telemetry/debug only for the normal explicit `no_line_action` bookkeeping move. Do **not** show HUD/Victory banners for routine no‑line phases; reserve UI surfacing for anomalies (e.g., host surfaced a line banner but then offered zero options unexpectedly).

#### RWS‑003 ANM_TERRITORY_NO_ACTIONS

- **Description:** No remaining interactive territory decisions (no `process_territory_region`, `choose_territory_option`, or `eliminate_rings_from_stack` moves) in `territory_processing`; engine must either apply forced elimination or end the turn / game.
- **Rules references:**
  - [`RR‑CANON‑R140–R145`](RULES_CANONICAL_SPEC.md:535) (territory processing).
  - [`RR‑CANON‑R204`](RULES_CANONICAL_SPEC.md:272) (territory‑processing exit rules).
  - ANM‑SCEN‑04 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:100).
- **Canonical `rules_context`:** `territory_mini_region` or `territory_multi_region` (depending on geometry).
- **UX surface guidance:** Telemetry/debug only for the normal explicit `no_territory_action` bookkeeping move. Do **not** show HUD/Victory banners for routine no‑territory phases; reserve UI surfacing for anomalies (e.g., host surfaced a territory banner but then offered zero options unexpectedly).

#### RWS‑004 FE_SEQUENCE_CURRENT_PLAYER

- **Description:** A forced‑elimination **sequence** is removing caps from the **local player’s** stacks because they are repeatedly blocked on their turns. In logs and parity tooling this corresponds to a run of `forced_elimination` moves in the `forced_elimination` phase between normal interactive turns.
- **Rules references:**
  - [`RR‑CANON‑R100`](RULES_CANONICAL_SPEC.md:443) (forced elimination semantics).
  - [`RR‑CANON‑R205–R207`](RULES_CANONICAL_SPEC.md:290) (forced‑elimination taxonomy and progress).
  - CCE‑007 in [`RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:459) (selection heuristics).
- **Canonical `rules_context`:** `anm_forced_elimination`.

#### RWS‑005 STRUCTURAL_STALEMATE_TIEBREAK

- **Description:** **Structural stalemate / plateau**: no legal placements, moves, captures, or forced eliminations for any player; game ends and is resolved by territory and eliminated‑ring tiebreakers.
- **Rules references:**
  - [`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:619) (global stalemate and tiebreak ladder).
  - ANM‑SCEN‑06 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:140).
- **Canonical `rules_context`:** `structural_stalemate`.

#### RWS‑006 LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS

- **Description:** **Last‑Player‑Standing** victory: requires **two consecutive complete rounds** where exactly one player has real actions (placements, moves, captures) and all others have none (only FE or nothing). In the first round, the exclusive player must have and take at least one real action. After the second round completes with the same condition, that player wins by LPS.
- **Rules references:**
  - [`RR‑CANON‑R172`](RULES_CANONICAL_SPEC.md:603) (Last‑Player‑Standing).
  - ANM‑SCEN‑07 in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:160).
  - CCE‑006 in [`RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:443) (implementation compromise).
- **Canonical `rules_context`:** `last_player_standing`.

---

## 3. Canonical Copy by Reason Code

This section defines UX copy blocks per reason code. Where possible, it builds directly on the canonical baselines in [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:331) §10.

### 3.1 HUD Banners

#### RWS‑001 ANM_MOVEMENT_FE_BLOCKED

- **HUD badge label:** `"Forced Elimination"`
- **HUD title (local player):**  
  `"You have no legal moves this turn"`
- **HUD title (other player):**  
  `"<Name> has no legal moves"`
- **HUD body (local player):**  
  `"You control stacks but have no legal placements, movements, or captures. A cap will be removed from one of your stacks until a real move becomes available, following the forced-elimination rules."`
- **HUD body (other player):**  
  `"<Name> controls stacks but has no legal placements, movements, or captures. A cap will be removed from one of their stacks until a real move becomes available, following the forced-elimination rules."`

These refine and specialise the general ANM and FE banners in [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:347).

#### RWS‑002 ANM_LINE_NO_ACTIONS

- **HUD title (local player):**  
  `"No legal line actions available"`
- **HUD title (other player):**  
  `"<Name> has no line actions"`
- **HUD body:**  
  `"There are no valid line actions to take. The game will auto-resolve this phase and move on according to the line-processing rules."`

#### RWS‑003 ANM_TERRITORY_NO_ACTIONS

- **HUD title (local player):**  
  `"No legal territory actions available"`
- **HUD title (other player):**  
  `"<Name> has no territory actions"`
- **HUD body:**  
  `"There are no valid territory or self-elimination actions to take. The game will auto-resolve this phase and move on according to the territory rules."`

#### RWS‑004 FE_SEQUENCE_CURRENT_PLAYER

Used when the local player has seen multiple successive FE events in a short period, even across multiple turns.

- **HUD badge label:** `"Forced Elimination"`
- **HUD title:**  
  `"Forced elimination is shrinking your stacks"`
- **HUD body:**  
  `"Because you control stacks but have no legal placements or moves on some turns, forced elimination repeatedly removes caps from your stacks. Each removal permanently eliminates rings and counts toward Ring Elimination."`

#### RWS‑005 STRUCTURAL_STALEMATE_TIEBREAK

- **HUD badge label:** `"Structural stalemate"`
- **HUD title:**  
  `"Structural stalemate"`
- **HUD body:**  
  `"No players have any legal placements, movements, captures, or forced eliminations left. The game ends here and the final score is computed from territory and eliminated rings."`

#### RWS‑006 LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS

Displayed when the game ends by LPS (once engines fully implement [`RR‑CANON‑R172`](RULES_CANONICAL_SPEC.md:603)).

- **HUD badge label:** `"Last Player Standing"`
- **HUD title (winner local):**
  `"You are the last player with real moves"`
- **HUD title (winner other):**
  `"<Name> is the last player with real moves"`
- **HUD body:**
  `"For TWO consecutive complete rounds, only one player had any real moves available (placements, movements, or captures). In the first round they took at least one real action while all others had none; in the second round the condition persisted. The game ends by Last Player Standing."`

---

### 3.2 VictoryModal Explanations

These strings are intended as the **canonical explanation paragraphs** for VictoryModal when a weird‑state reason is involved.

#### RWS‑001 ANM_MOVEMENT_FE_BLOCKED (Non‑terminal explanatory text)

Typically used as a **sub‑section** within a longer VictoryModal explanation when FE played a key role **before** the final victory condition.

- **Title:** `"Forced Elimination on your turns"`
- **Body:**  
  `"Several of your turns reached a state where you controlled stacks but had no legal placements, movements, or captures. In those moments the forced-elimination rule removed caps from your stacks, permanently eliminating rings until a real move became available or your stacks ran out. These eliminations counted toward Ring Elimination and may have contributed to the final result."`
- **Doc anchor:** [`ringrift_complete_rules.md`](ringrift_complete_rules.md:499) §4.4 _Forced Elimination When Blocked_; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:443) R100.

#### RWS‑005 STRUCTURAL_STALEMATE_TIEBREAK (Terminal reason)

- **Title:** `"Structural stalemate – final score from territory and rings"`
- **Body:**  
  `"The game reached a structural stalemate: no player had any legal placements, movements, captures, or forced eliminations left. At that point the rules convert any rings in hand to eliminated rings and compute the final score in four steps: first by total Territory spaces, then by eliminated rings (including rings in hand), then by markers, and finally by who took the last real action. The winner is the player highest on this ladder."`
- **Doc anchor:** [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1381) §13.4 _End of Game "Stalemate" Resolution_; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:619) R173.

#### RWS‑006 LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS (Terminal reason)

- **Title:** `"Last Player Standing – only one player had real moves for two rounds"`
- **Body:**
  `"The game ended by Last Player Standing. For TWO consecutive complete rounds, only one player had any real actions available at the start of their turns – real actions meaning placements, non-capture movements, or overtaking captures. In the first round, this player had and took at least one real action while all others had none. In the second round, the same condition persisted. After the second round completed, the remaining active player won by Last Player Standing."`
- **Doc anchor:** [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1360) §13.3 _Last Player Standing_; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:603) R172.

---

### 3.3 TeachingOverlay Topics

For each weird state, TeachingOverlay SHOULD offer a deep‑dive topic, either existing (per [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:438)) or newly added.

#### RWS‑001 / RWS‑004 (ANM + Forced Elimination)

Map both to existing topics:

- `teaching.active_no_moves` ([`TeachingOverlay.TEACHING_CONTENT.active_no_moves`](src/client/components/TeachingOverlay.tsx:98)):
  - **Title:** `"When you have no legal moves"`.
  - Focus: definition of ANM, real moves vs FE, structural stalemate.
- `teaching.forced_elimination` ([`TeachingOverlay.TEACHING_CONTENT.forced_elimination`](src/client/components/TeachingOverlay.tsx:110)):
  - **Title:** `"Forced Elimination (FE)"`.
  - Focus: FE conditions, how caps are removed, how eliminations are credited.

**Recommended routing:**

- If a HUD FE banner fires (`RWS‑001` or `RWS‑004`) and the user clicks help:
  - Open `teaching.active_no_moves` first, then allow navigation to `teaching.forced_elimination`.

#### RWS‑002 ANM_LINE_NO_ACTIONS

Recommended new focus area inside the existing lines topic:

- Reuse `teaching.line_bonus` ([`TeachingOverlay.TEACHING_CONTENT.line_bonus`](src/client/components/TeachingOverlay.tsx:72)) and extend copy (in a future docs pass) with a sub‑section:
  - `"When there are no line decisions left, the game automatically moves on to Territory and victory checks – you cannot make further line choices this turn."`

#### RWS‑003 ANM_TERRITORY_NO_ACTIONS

Route to `teaching.territory` ([`TeachingOverlay.TEACHING_CONTENT.territory`](src/client/components/TeachingOverlay.tsx:84)).

Add clarifying bullet in that topic:

- `"If there are no more disconnected regions you can afford to process and no self-elimination choices left, the Territory phase ends automatically and the game moves on to the next player or to victory checks."`

#### RWS‑005 STRUCTURAL_STALEMATE_TIEBREAK

Route to `teaching.victory_stalemate` ([`TeachingOverlay.TEACHING_CONTENT.victory_stalemate`](src/client/components/TeachingOverlay.tsx:144)).

Ensure that topic clearly enumerates the four‑step tiebreak ladder used in [`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:619).

#### RWS‑006 LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS

Route to `teaching.victory_stalemate` as well, but with a distinct sub‑section titled `"Last Player Standing"` that:

- Emphasises the difference between:
  - LPS (one player retains real actions over a full round), and
  - Structural stalemate (no moves or FE for anyone).
- Clarifies that FE and automatic processing do **not** count as real actions.

---

## 4. Mapping Tables

This section defines the canonical mapping between reason codes and:

- HUD banner keys.
- VictoryModal explanation keys.
- TeachingOverlay topics.
- Telemetry `rules_context`.

Code‑mode tasks should implement these mappings in a configuration module (e.g. `WeirdStateUxMapping`) rather than duplicating strings across components.

### 4.1 HUD Mapping: `reason_code → banner_text_key`

Suggested key names (in a small central dictionary):

| Reason code                                   | HUD title key                              | HUD body key                              |
| :-------------------------------------------- | :----------------------------------------- | :---------------------------------------- |
| `ANM_MOVEMENT_FE_BLOCKED`                     | `hud.weird.anm_movement_fe.title`          | `hud.weird.anm_movement_fe.body`          |
| `ANM_LINE_NO_ACTIONS`                         | `hud.weird.anm_line_no_actions.title`      | `hud.weird.anm_line_no_actions.body`      |
| `ANM_TERRITORY_NO_ACTIONS`                    | `hud.weird.anm_territory_no_actions.title` | `hud.weird.anm_territory_no_actions.body` |
| `FE_SEQUENCE_CURRENT_PLAYER`                  | `hud.weird.fe_sequence.title`              | `hud.weird.fe_sequence.body`              |
| `STRUCTURAL_STALEMATE_TIEBREAK`               | `hud.weird.structural_stalemate.title`     | `hud.weird.structural_stalemate.body`     |
| `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS` | `hud.weird.lps_early_win.title`            | `hud.weird.lps_early_win.body`            |

All keys above MUST use copy drawn from §3.1, possibly via [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:331) §10 consolidation.

### 4.2 VictoryModal Mapping: `reason_code → explanation_text_key, rules_doc_anchor`

| Reason code                                   | VictoryModal explanation key           | Rules doc anchor                                                                                                                     |
| :-------------------------------------------- | :------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `ANM_MOVEMENT_FE_BLOCKED`                     | `victory.explain.fe_sequence`          | [`ringrift_complete_rules.md`](ringrift_complete_rules.md:499) §4.4; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:443) R100   |
| `STRUCTURAL_STALEMATE_TIEBREAK`               | `victory.explain.structural_stalemate` | [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1381) §13.4; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:619) R173 |
| `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS` | `victory.explain.last_player_standing` | [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1360) §13.3; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:603) R172 |

For `ANM_LINE_NO_ACTIONS` and `ANM_TERRITORY_NO_ACTIONS`, VictoryModal typically uses standard line / territory text; no extra reason‑specific key is required beyond possible debug / dev tooling.

### 4.3 TeachingOverlay Mapping: `reason_code → teaching_topic`

| Reason code                                   | Primary TeachingOverlay topic id           | Secondary / related topics                                                  |
| :-------------------------------------------- | :----------------------------------------- | :-------------------------------------------------------------------------- |
| `ANM_MOVEMENT_FE_BLOCKED`                     | `teaching.active_no_moves`                 | `teaching.forced_elimination`                                               |
| `ANM_LINE_NO_ACTIONS`                         | `teaching.line_bonus`                      | —                                                                           |
| `ANM_TERRITORY_NO_ACTIONS`                    | `teaching.territory`                       | —                                                                           |
| `FE_SEQUENCE_CURRENT_PLAYER`                  | `teaching.forced_elimination`              | `teaching.active_no_moves`                                                  |
| `STRUCTURAL_STALEMATE_TIEBREAK`               | `teaching.victory_stalemate`               | `teaching.victory_elimination`, `teaching.victory_territory` (for contrast) |
| `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS` | `teaching.victory_stalemate` (LPS section) | `teaching.active_no_moves`                                                  |

### 4.4 Telemetry Mapping: `reason_code → rules_context`

This mapping MUST be used by the telemetry layer (see [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:218)):

| Reason code                                   | `rules_context`                                                                 |
| :-------------------------------------------- | :------------------------------------------------------------------------------ |
| `ANM_MOVEMENT_FE_BLOCKED`                     | `anm_forced_elimination`                                                        |
| `ANM_LINE_NO_ACTIONS`                         | `line_reward_exact` or `line_reward_overlength` (depending on current position) |
| `ANM_TERRITORY_NO_ACTIONS`                    | `territory_mini_region` or `territory_multi_region`                             |
| `FE_SEQUENCE_CURRENT_PLAYER`                  | `anm_forced_elimination`                                                        |
| `STRUCTURAL_STALEMATE_TIEBREAK`               | `structural_stalemate`                                                          |
| `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS` | `last_player_standing`                                                          |

---

## 5. Test‑Oriented Examples

This section defines **scenario‑level expectations** for Code‑mode tests. Scenario ids may correspond to existing scenarios or be newly added.

### 5.1 Example 1 – Movement‑Phase ANM with Forced Elimination (RWS‑001)

- **Scenario id (planned):** `WeirdState.ANM_FE.square8_two_player_blocked`
- **Shape:**
  - Square‑8, 2‑player game.
  - Player A controls exactly one stack; A has `ringsInHand == 0`.
  - No legal placements or movements/captures exist for A.
  - Forced elimination is required under [`RR‑CANON‑R100`](RULES_CANONICAL_SPEC.md:443).
- **Expected HUD banner (local A):**
  - Title: `"You have no legal moves this turn"`.
  - Badge: `"Forced Elimination"`.
  - Body: as in §3.1 RWS‑001.
- **Expected VictoryModal text:**
  - None yet (game continues), but history / replay UI SHOULD tag the relevant turn with reason code `ANM_MOVEMENT_FE_BLOCKED` for logs and teaching suggestions.
- **Expected `rules_context`:** `anm_forced_elimination`.

### 5.2 Example 2 – Structural Stalemate Tiebreak (RWS‑005)

- **Scenario id (existing / planned):** `ForcedEliminationAndStalemate.structural_stalemate_three_player`
- **Shape:**
  - 3‑player Square‑19 or Hex game.
  - No stacks remain on the board.
  - Each player has some markers, territory spaces, and rings in hand.
  - No placements are legal for any player (no‑dead‑placement, ring cap, or geometry constraints), so no moves or FE are possible.
- **Expected HUD / game‑over banner:**
  - `"Structural stalemate"` as title, body as in §3.1 RWS‑005.
- **Expected VictoryModal:**
  - Title: `"🧱 Structural Stalemate"`.
  - Body: explanation of plateau and tiebreak ladder as in §3.2 RWS‑005.
- **Expected `rules_context`:** `structural_stalemate`.
- **Expected engine reason mapping:**
  - Game result reason / termination reason code should be mapped to `STRUCTURAL_STALEMATE_TIEBREAK`.

### 5.3 Example 3 – Last‑Player‑Standing Early Victory (RWS‑006)

- **Scenario id (planned):** `Victory.LastPlayerStanding.single_real_actor_with_buried_rings`
- **Shape:**
  - 3‑player Square‑8 game.
  - P1 controls several stacks with legal moves.
  - P2 owns buried rings inside P1’s mixed stacks but has no legal placements, movements, or captures (only FE, or none).
  - P3 has no stacks and no rings in hand.
  - Over one full round of turns:
    - At the start of P1’s turns, `getValidRealMoves(P1)` is non‑empty.
    - At the start of P2’s and P3’s turns, `getValidRealMoves(P2/3)` is empty (no placements, no moves/captures).
- **Expected behaviour:**
  - At the start of P1’s next turn, the game ends with:
    - Winner: P1.
    - Result reason: `LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS`.
- **Expected HUD / VictoryModal:**
  - HUD uses the LPS banner in §3.1 RWS‑006.
  - VictoryModal uses the explanation from §3.2 RWS‑006.
- **Expected `rules_context`:** `last_player_standing`.

---

## 6. Relationship to Engine / Code Enums

Current TypeScript and Python engines already expose termination / weird‑state reasons (e.g. game result reasons, special HUD states). Code‑mode tasks should:

1. **Introduce a central mapping module** (e.g. `WeirdStateUxMapping`) that:
   - Defines the reason codes listed in §2 as string constants.
   - Maps from engine‑level enums (e.g. `GameResultReason`, `WeirdStateReason`) to these canonical codes.

2. **Ensure 1:1 coverage:**
   - Every weird‑state path that leads to banners or special explanations must map to exactly one reason code.
   - Non‑weird, standard victories (simple elimination / territory with no ANM / FE / plateau complexity) MAY continue to use only the standard victory reason and copy without a weird‑state reason code.

3. **Keep this document authoritative:**
   - Any new reason code added in code must be added to §2–§4 here.
   - Any removal / consolidation of reason codes must be reflected here before code changes are merged.

---

## 7. Relationship to Other UX Specs

- [`UX_RULES_COPY_SPEC.md`](docs/UX_RULES_COPY_SPEC.md:331) remains the **canonical copy source** for generic movement, capture, lines, territory, and victory text. This document only defines **weird‑state‑specific overlays** on top.
- [`UX_RULES_TELEMETRY_SPEC.md`](docs/UX_RULES_TELEMETRY_SPEC.md:172) relies on the `reason_code` and `rules_context` mappings in §4.4 for:
  - `weird_state_banner_impression`,
  - `weird_state_details_open`,
  - `resign_after_weird_state`.
- For the structured payload consumed by end-of-game UX surfaces, see [`UX_RULES_EXPLANATION_MODEL_SPEC.md`](docs/UX_RULES_EXPLANATION_MODEL_SPEC.md:1).
- [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1) should reference the reason codes in §2 when defining flows that specifically demystify ANM/FE, structural stalemate, or LPS.

Together, these documents give a complete blueprint for:

- Explaining weird rules states consistently in HUD, VictoryModal, TeachingOverlay, and Sandbox.
- Instrumenting them with low‑cardinality telemetry for W‑UX‑1 / W‑UX‑4.
- Adding future curated scenarios and teaching flows that target the most confusing endings and forced‑move situations.

# RingRift Weird / Complex Rules States UX Spec

> **Doc Status (2025‑12‑05): Partially implemented – reason codes and HUD banners are wired for core ANM / forced‑elimination / stalemate cases; this doc is the contract for extending copy, surfaces, and telemetry.**
>
> **Role:** Define a small, stable set of **weird / complex rules “reason codes”** and how they map to:
>
> - HUD / VictoryModal banners and TeachingOverlay topics.
> - Rules‑UX telemetry (`weird_state_type`, `reason_code`, `rules_context`).
> - Curated scenarios and teaching flows.

This spec complements:

- `RULES_CANONICAL_SPEC.md` – semantics for ANM, forced elimination, territory edge cases.
- `docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md` – ANM behaviour and forced elimination loops.
- `UX_RULES_TELEMETRY_SPEC.md` – telemetry envelope and metrics.
- `UX_RULES_TEACHING_SCENARIOS.md` – curated scenarios and flows.

---

## 1. Taxonomy and Reason Codes

We distinguish between:

- A **coarse weird state type** used for metrics (`weird_state_type`).
- A more **specific reason code** used for copy, TeachingOverlay topics, and sandbox scenarios.

### 1.1 Coarse weird state types (metrics)

These values are already wired in `RulesUxWeirdStateType` and exposed as the
`weird_state_type` label on `ringrift_rules_ux_events_total`:

- `active-no-moves-movement` – Active‑no‑moves (ANM) reached from movement phase.
- `active-no-moves-line` – ANM reached while line processing decisions are pending.
- `active-no-moves-territory` – ANM reached while territory / region decisions are pending.
- `forced-elimination` – Forced elimination loop (ANM + elimination requirements).
- `structural-stalemate` – No legal moves for any player and no remaining line/territory decisions.

Any additional values MUST be rare and explicitly documented here before being
added to `RulesUxWeirdStateType`.

### 1.2 Reason codes (copy, flows, telemetry payload)

Reason codes are **string constants** used in:

- HUD / VictoryModal banner selection.
- TeachingOverlay topics and flows.
- `payload.reason_code` on rules‑UX telemetry events
  (`weird_state_banner_impression`, `weird_state_details_open`,
  `weird_state_overlay_*`, `resign_after_weird_state`).

They are more specific than `weird_state_type` but still low‑cardinality.

Initial set (to be kept small and stable):

- `ANM_FE_FORCED_ELIMINATION` – Active‑no‑moves reached and at least one stack
  must have rings eliminated; see `ACTIVE_NO_MOVES_BEHAVIOUR.md`.
- `ANM_LAST_PLAYER_STANDING` – ANM resolution leads to a last‑player‑standing
  outcome (other players cannot move or are eliminated).
- `ANM_LINE_PROCESSING_DEADLOCK` – ANM reached while line rewards or penalties
  are pending and cannot be fully resolved.
- `ANM_TERRITORY_DISCONNECTION` – ANM related to territory disconnection /
  mini‑regions during `territory_processing`.
- `STRUCTURAL_STALEMATE_NO_MOVES` – No legal moves for any player, no pending
  line or territory decisions, game ends by structural stalemate.

Mapping from reason code → `weird_state_type`:

| reason_code                     | weird_state_type            |
| ------------------------------- | --------------------------- |
| `ANM_FE_FORCED_ELIMINATION`     | `forced-elimination`        |
| `ANM_LAST_PLAYER_STANDING`      | `active-no-moves-movement`  |
| `ANM_LINE_PROCESSING_DEADLOCK`  | `active-no-moves-line`      |
| `ANM_TERRITORY_DISCONNECTION`   | `active-no-moves-territory` |
| `STRUCTURAL_STALEMATE_NO_MOVES` | `structural-stalemate`      |

Any new reason codes MUST specify:

- A mapping to exactly one `weird_state_type`.
- A primary `rules_context` (see telemetry spec) for hotspot analysis.

---

## 2. Surfaces and Copy Contract

This section defines where weird‑state reason codes appear and what copy
surfaces MUST exist. Concrete wording can evolve, but the **intent and
information hierarchy** should remain.

### 2.1 HUD weird‑state banners

When `getWeirdStateBanner` (client) returns a non‑null banner, the HUD MUST:

- Show a short, prominent banner with:
  - A title (e.g. “Forced elimination required”).
  - A one‑sentence summary.
  - A “Details” affordance that opens TeachingOverlay or VictoryModal copy.
- Emit telemetry:
  - `weird_state_banner_impression` with `payload.reason_code` and
    `rules_context` derived from the reason code.
  - `rules_weird_state_help` when the user opens detailed help from this banner.

Suggested title patterns:

- `ANM_FE_FORCED_ELIMINATION` → “Forced elimination required”
- `ANM_LAST_PLAYER_STANDING` → “No moves left – last player standing”
- `ANM_LINE_PROCESSING_DEADLOCK` → “Line resolution locked up”
- `ANM_TERRITORY_DISCONNECTION` → “Territory disconnected – special handling”
- `STRUCTURAL_STALEMATE_NO_MOVES` → “No moves for any player (stalemate)”

### 2.2 VictoryModal

When a game ends due to a weird / complex rule:

- The VictoryModal SHOULD:
  - Show a compact banner or subheading explaining the reason code.
  - Provide a “What happened?” / “See details” link to TeachingOverlay.
- Telemetry:
  - `weird_state_banner_impression` with `source = 'victory_modal'`.
  - `weird_state_details_open` when the details link is clicked.

### 2.3 TeachingOverlay

When opened from a weird‑state banner or VictoryModal with `reason_code`:

- TeachingOverlay MUST:
  - Use a topic and intro text that explicitly names the weird state.
  - Provide at least one diagram or annotated example matching the condition.
  - Optionally offer a link to a curated sandbox scenario.
- Telemetry:
  - `weird_state_overlay_shown` / `weird_state_overlay_dismiss`.
  - `teaching_step_started` / `teaching_step_completed` for multi‑step flows.

The `rules_context` used here should be derived from reason code:

- `ANM_FE_FORCED_ELIMINATION` → `anm_forced_elimination`
- `ANM_LAST_PLAYER_STANDING` → `anm_last_player_standing`
- `ANM_LINE_PROCESSING_DEADLOCK` → `line_reward_overlength` (or similar)
- `ANM_TERRITORY_DISCONNECTION` → `territory_multi_region`
- `STRUCTURAL_STALEMATE_NO_MOVES` → `structural_stalemate`

---

## 3. Scenario and Teaching Flow Integration

Curated scenarios that are specifically designed to illustrate weird states
MUST:

- Declare a `rulesConcept` aligned with the mappings above.
- Use `reason_code` in their metadata where applicable.

Examples:

- `scenario_id = "anm_forced_elimination_loop_intro"`
  - `rulesConcept = "anm_forced_elimination"`
  - `reason_code = "ANM_FE_FORCED_ELIMINATION"`
- `scenario_id = "territory_disconnection_intro"`
  - `rulesConcept = "territory_multi_region"`
  - `reason_code = "ANM_TERRITORY_DISCONNECTION"`

When such a scenario is loaded in sandbox:

- Sandbox MUST emit `sandbox_scenario_loaded` with:
  - `rules_context` = `rulesConcept`.
  - `payload.scenario_id` and, when applicable, `payload.flow_id`.
- On completion, Sandbox MUST emit `sandbox_scenario_completed` with
  `payload.success`, `moves_taken`, and `ms_to_complete`.

---

## 4. Telemetry Integration Summary

This section links the reason codes and weird‑state types back to the
telemetry spec.

- `weird_state_banner_impression`
  - `payload.reason_code` – one of the reason codes above.
  - `rules_context` – derived mapping.
  - `weird_state_type` – taken from the reason‑code mapping table.
- `weird_state_details_open`, `weird_state_overlay_*`
  - MUST re‑use the same `reason_code`, `rules_context`, and
    `weird_state_type` for the given `overlay_session_id`.
- `resign_after_weird_state`
  - MUST include `payload.reason_code`, `rules_context`,
    and timing fields as per telemetry spec.

Metrics:

- `ringrift_rules_ux_events_total{type="rules_weird_state_resign", weird_state_type=...}`
  gives a coarse view of resigns by weird state type.
- Higher‑fidelity analysis of **which exact reason codes** are most
  problematic should be performed via logs or offline processing of
  `reason_code` from the event payloads.

---

## 5. Change Management

- Any new `weird_state_type` values MUST be:
  - Added to `RulesUxWeirdStateType` in `src/shared/telemetry/rulesUxEvents.ts`.
  - Documented in §1.1 with intended semantics.
- Any new `reason_code` values MUST be:
  - Documented in §1.2 with a mapping to `weird_state_type` and `rules_context`.
  - Plumbed through HUD / TeachingOverlay / VictoryModal copy and sandbox
    scenarios as appropriate.

This keeps weird‑state UX, telemetry, and teaching flows aligned while
allowing incremental expansion as the rules UI matures.
