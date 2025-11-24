# RingRift Ruleset Clarifications (Second-Pass Audit)

This document records remaining or newly-identified rules clarifications (`CLAR-XXX`) from the second-pass Lead Integrator audit.

It is intended to complement the canonical rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1), the prior audit in [`FINAL_RULES_AUDIT_REPORT.md`](FINAL_RULES_AUDIT_REPORT.md:1), and the analysis artefacts:

- [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md:1)
- [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md:1)
- [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:1)
- [`RULES_CONSISTENCY_EDGE_CASES.md`](RULES_CONSISTENCY_EDGE_CASES.md:1)

Only items that are **not fully resolved** in Sections 11–12 of [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:489) are included here. Each CLAR entry is structured for use by an orchestrator agent to ask the human rules author targeted questions.

## 1. CLAR Items Overview

| ID       | Type                                           | Status   | Priority | Short description                                                                                                                                           |
| :------- | :--------------------------------------------- | :------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CLAR-001 | Ambiguity / underspecification                 | Resolved | Low      | RR-CANON-R090 references non-existent RR-CANON-R093 in movement availability text.                                                                          |
| CLAR-002 | Cross-document contradiction / design decision | Resolved | High     | Last-player-standing victory (RR-CANON-R172) is specified in rules texts but only partially implemented; priority and intended behaviour need confirmation. |
| CLAR-003 | Ambiguity / underspecification                 | Resolved | Medium   | Per-player ring cap semantics with captured rings (RR-CANON-R020, R080–R082) are not fully specified; implementation uses a conservative approximation.     |

## 2. Detailed Clarifications

### 2.1 CLAR-001 – Reference to RR-CANON-R093 in Movement Availability

- **ID:** CLAR-001
- **Type:** Ambiguity / underspecification
- **Status:** Resolved (canonical editorial fix applied)
- **Priority:** Low
- **Resolution:** Resolved as a typo; RR-CANON-R090 now refers to R091–R092 only.

**Sources**

- [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:232) – RR-CANON-R090 text: "any stack in S that satisfies RR-CANON-R091–R093 for at least one direction has at least one legal move."
- [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:239) – RR-CANON-R091 Path and distance for non-capture movement.
- [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:255) – RR-CANON-R092 Marker interaction during non-capture movement.
- No RR-CANON-R093 definition exists in the canonical spec.

**Problem description**

RR-CANON-R090 defines movement availability in terms of stacks that "satisfy RR-CANON-R091–R093", but only RR-CANON-R091 and RR-CANON-R092 are actually defined. There is no RR-CANON-R093 rule block in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1). This creates a dangling rule reference and minor ambiguity for downstream tooling that expects every referenced rule ID to exist.

**Candidate interpretations**

- **Interpretation A:** "R091–R093" is a typo and was intended to read "R091–R092" (path/distance plus marker interaction). Under this reading, there is **no missing rule**; RR-CANON-R090 simply depends on the two defined rules.
- **Interpretation B:** There was originally an intended third rule (RR-CANON-R093) covering an additional movement constraint (for example, a boundary condition or version-specific exception) that was removed or folded into R091/R092, but the reference was not updated.

**Current implementation behaviour**

- All known movement logic in the TypeScript and Python engines implements movement availability using exactly the semantics described in RR-CANON-R091 and RR-CANON-R092, as mapped in [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md:254) §3.4 and analysed in [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md:811) §2.4 and [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:227) §2.4.
- No code, tests, or documentation refer to a distinct RR-CANON-R093 behaviour. The system behaves as if Interpretation A is correct.

**Recommended clarification / questions for rules author**

- **Proposed canonical wording change (if Interpretation A is intended):** In RR-CANON-R090, replace "RR-CANON-R091–R093" with "RR-CANON-R091–R092".
- **Question to author:**
  - "Was RR-CANON-R090 intended to refer only to RR-CANON-R091 and RR-CANON-R092, or is there a missing third movement rule (RR-CANON-R093) that should still be documented? If the former, can we update the canonical spec to change 'R091–R093' to 'R091–R092' to remove the dangling reference?"

---

### 2.2 CLAR-002 – Last-Player-Standing Victory (RR-CANON-R172) vs Implementation

- **ID:** CLAR-002
- **Type:** Cross-document contradiction / design decision
- **Status:** Resolved (binding semantics chosen)
- **Priority:** High
- **Resolution:** Resolved as a binding Last-Player-Standing victory with a one-full-round exclusive real-action condition, as encoded in RR-CANON-R172 and mirrored in the Complete and Compact rules.

**Sources**

- [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:425) – RR-CANON-R172 Last-player-standing victory: defines an explicit early victory condition when exactly one player has any legal action on their next turn and all others have none.
- [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1228) §13.3 Last Player Standing – narrative description and examples of last-player-standing as a primary victory path alongside elimination and territory.
- [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1897) §16.6 and [`ringrift_complete_rules.md`](ringrift_complete_rules.md:2101) §16.9.4.5 – summaries that restate last-player-standing as a distinct victory path.
- [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md:381) §3.8 – notes that victory logic encodes elimination, territory, last-player-standing, and stalemate in `victoryLogic.evaluateVictory()`.
- [`RULES_CONSISTENCY_EDGE_CASES.md`](RULES_CONSISTENCY_EDGE_CASES.md:316) CCE-006 – explicitly calls out that RR-CANON-R172 is **not** fully implemented; engines effectively play to completion and rely on elimination, territory, or bare-board stalemate.

**Problem description**

All three rule documents (Complete, Compact, Canonical) agree that **Last Player Standing** is a distinct, third path to victory: a player should win immediately when, after a full turn, they are the only player who has any legal action on their next turn while all others have none, even if ring-elimination and territory thresholds have not yet been reached.

However, the current engines (TypeScript backend, sandbox, and Python) effectively implement the following behaviour:

- Games terminate only via:
  - Ring-elimination victory (RR-CANON-R170), or
  - Territory-control victory (RR-CANON-R171), or
  - Global stalemate and tiebreakers on a **bare board** with no stacks (RR-CANON-R173).
- Non-bare-board positions where a single player is the only one with legal actions are **not** treated as immediate wins; play continues until one of the other conditions is met.

This is consistent and terminating, but diverges from the literal reading of RR-CANON-R172 and the Complete Rules text, and creates a latent risk for:

- Rules lawyers and competitive players who may expect strict R172 early termination.
- Future engine ports (e.g., alternative languages or mobile clients) that might choose to implement R172 literally and thus diverge from the current TS/Python behaviour.

**Candidate interpretations**

- **Interpretation A (rules-text literal):** RR-CANON-R172 defines a **binding early termination rule**. As soon as the R172 condition holds after a full turn (exactly one player has any legal action on their next turn, all others have none), the game **must end immediately** with a Last Player Standing victory, even if elimination or territory thresholds have not yet been reached.
- **Interpretation B (current engine behaviour):** Last-player-standing is **conceptual only**, describing one way a player tends to win via elimination/territory in practice, but there is **no separate early-termination rule**. The only official victory triggers are elimination, territory, and bare-board stalemate; last-player-standing is effectively subsumed by these conditions.
- **Interpretation C (compromise variant):**
  - Keep R172 as a formal rule, but **constrain its scope**, for example:
    - Only apply last-player-standing when:
      - No further ring placements are possible for any player, and
      - Any additional forced-elimination steps would not change which players have legal actions, or
      - The R172 condition holds for **k** consecutive full rounds of turns.
  - This would make last-player-standing harder to trigger and closer to a pre-stalemate condition while still being a distinct rule.

**Current implementation behaviour**

- [`TypeScript.victoryLogic.evaluateVictory`](src/shared/engine/victoryLogic.ts:45) checks for:
  - Ring-elimination and territory thresholds, and
  - Global stalemate only once **no stacks remain**, at which point rings in hand are converted to eliminated rings and the tie-break ladder runs (territory → eliminated rings → markers → last actor).
- There is **no explicit check** implementing RR-CANON-R172 as written; instead, the system relies on continued play, forced elimination (RR-CANON-R100), and eventual stalemate or threshold victories.
- Python `GameEngine` mirrors this behaviour; strict invariants focus on "no active player without a move or forced elimination" rather than early last-player-standing victories, as described in [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md:873) §3.3 and [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:665) §4.1.

**Recommended clarification / questions for rules author**

- **Design decision (engine vs rules text):**
  - "Should Last Player Standing (RR-CANON-R172) remain a **distinct, early termination condition** in the official rules, or should the canonical rules be updated to match the existing engines (elimination / territory / bare-board stalemate only)?"
- **If Interpretation A is chosen (keep R172 as binding):**
  - "Do you want engines to:
    - (i) Implement an explicit RR-CANON-R172 check in `victoryLogic.evaluateVictory()` and Python `GameEngine`, and
    - (ii) Define a clear **priority order** among victory conditions (Elimination → Territory → Last Player Standing → Stalemate) to remove any residual ambiguity?"
- **If Interpretation B is chosen (deprecate R172 as a separate rule):**
  - "Can we update RR-CANON-R172 and the corresponding sections in `ringrift_complete_rules.md` and `ringrift_compact_rules.md` to describe last-player-standing **only as an emergent pattern** of elimination/territory play, and to state explicitly that the only formal victory triggers are RR-CANON-R170, R171, and R173?"

---

### 2.3 CLAR-003 – Per-Player Ring Cap Semantics with Captured Rings

- **ID:** CLAR-003
- **Type:** Ambiguity / underspecification
- **Status:** Resolved (canonical semantics chosen)
- **Priority:** Medium
- **Resolution:** Resolved: `ringsPerPlayer` is an own-colour ring supply cap only; captured opponent rings in controlled stacks do not count against this cap.

**Sources**

- [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:61) – RR-CANON-R020 rings per player; [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:218) RR-CANON-R081 and [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:225) RR-CANON-R082 reference a per-player `ringsPerPlayer` maximum when placing.
- [`ringrift_compact_rules.md`](ringrift_compact_rules.md:18) §1.1 version table – defines `ringsPerPlayer` = 18 (square8) or 36 (square19/hexagonal).
- [`ringrift_complete_rules.md`](ringrift_complete_rules.md:342) §3.2.1 – states "Each player has 36 rings (for 19x19 and Hexagonal versions; 18 for 8x8)".
- [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md:755) §2.3.3 – describes the current implementation approximation: per-player ring cap counts **all rings in stacks controlled by a player**, including captured rings of other colours, when deciding whether further placements are allowed (CCE-002).
- [`RULES_CONSISTENCY_EDGE_CASES.md`](RULES_CONSISTENCY_EDGE_CASES.md:365) CCE-002 – classifies this as an "implementation compromise" and recommends either canonising or tightening it.

**Problem description**

The rules texts clearly define a per-player ring budget `ringsPerPlayer` (18 or 36 depending on board type), and R081/R082 use this to limit **placements** ("must not exceed ... the per-player `ringsPerPlayer` maximum for the board"). However, it is not fully specified how this interacts with **captured rings** and mixed-colour stacks:

- Does `ringsPerPlayer` constrain only rings of a player’s own colour they may **introduce from hand to the board** (i.e., initial supply), or
- Does it also limit how many rings they may **control or be associated with** in stacks, including captured rings of other colours?

The current implementation adopts a conservative approximation: it treats the sum of heights of all stacks where the player is the controlling colour (top ring) as part of that player’s cap when checking further placements. This has two consequences:

- In rare long games with many captured rings, a player can hit their effective cap **earlier** than a strict "own-colour rings only" interpretation would allow, because captured opponent rings in their stacks count toward their cap.
- The approximation is safe with respect to ring conservation (it only forbids extra placements, it never allows too many), but it is not obviously what the rules texts intend.

**Candidate interpretations**

- **Interpretation A (own-colour placement cap only):**
  - `ringsPerPlayer` is a cap on the number of rings of **that player’s colour** that can ever exist (initial supply).
  - Captured rings of other colours in a player’s stacks do **not** count toward that player’s cap; they remain part of the original owner’s colour budget for conservation and victory-threshold purposes.
  - Placement legality should therefore consider only own-colour rings in hand + own-colour rings already on the board when enforcing the cap; foreign-colour rings in your stacks do not reduce your remaining placement capacity.
- **Interpretation B (control-based cap, current approximation):**
  - `ringsPerPlayer` is a cap on the total number of rings a player may **control on the board at once**, regardless of ring colour.
  - The current implementation (counting total heights of stacks where `controllingPlayer == P`) is an approximate realisation of this: once a player controls stacks whose total heights reach `ringsPerPlayer`, they effectively cannot place additional rings, even if many of those rings belong to other players by colour.
  - This gives a stronger resource constraint and may have been intentionally chosen as a design simplification or balance lever.
- **Interpretation C (hybrid):**
  - Treat `ringsPerPlayer` as an own-colour placement cap (Interpretation A) **and** add an explicit secondary rule limiting the total height of stacks a player may control (Interpretation B), but with separate parameters or a soft cap. This would require explicit text in the canonical spec and rulebooks.

**Current implementation behaviour**

- Backend and sandbox placement validation (`validatePlacementOnBoard` and `RuleEngine.getValidRingPlacements`) approximate a player’s "rings on board" as the **sum of heights** of stacks where that player is the controlling colour; captured rings of other colours buried in those stacks contribute to this total.
- When that total reaches `ringsPerPlayer` for the board type, additional placements may be rejected even if the player technically still has own-colour rings in hand and relatively few own-colour rings on the board.
- Python rules engine mirrors this behaviour for parity, as documented in [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md:755) and [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:666).

**Recommended clarification / questions for rules author**

- **Primary question:**
  - "Should the per-player `ringsPerPlayer` limit apply only to a player’s **own-colour rings** (Interpretation A), or to **all rings in stacks they control** regardless of colour (Interpretation B)?"
- **If Interpretation A is intended:**
  - "Can we update the canonical spec and implementation to:
    - (i) Define `ringsPerPlayer` explicitly as an own-colour budget, and
    - (ii) Adjust placement-cap checks in both TypeScript and Python engines to count only own-colour rings on the board plus rings in hand when deciding whether further placements are legal?"
- **If Interpretation B is intended (current behaviour is desired):**
  - "Can we update [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:61) and the Complete / Compact rules to state clearly that:
    - (i) `ringsPerPlayer` is effectively a cap on total **stack height** under a player’s control (not just own-colour rings), and
    - (ii) captured rings of other colours in a player’s stacks count toward that player’s control cap for placement purposes?"
- **If a hybrid or alternative approach (Interpretation C) is preferred:**
  - "Please specify the exact intended relationship between own-colour ring budgets, captured rings, and any control-based caps so we can encode this precisely in RR-CANON and in the engines."

---

## 3. Status Summary

- CLAR-001, CLAR-002, and CLAR-003 are now **Resolved**, and their chosen interpretations have been integrated into [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1), [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1), and [`ringrift_compact_rules.md`](ringrift_compact_rules.md:1).
- There are currently **no** open CLAR items that block implementation or documentation work on Last-Player-Standing victory or per-player ring caps.
- Future clarification items, if any, should be added as new `CLAR-00X` entries below, with their own status and resolution notes.
