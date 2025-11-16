# Debugging Plan: RingRift Territory Anomaly at `a8`

## 1. Anomaly Overview

During Red's (Player 0) Turn 7 (logged as 'Turn 8 - Player 0 (Red)', 'Turn Count: 7'):
1.  Red performs a capture: "Action: capture segment `a4` over `a5` to `a8`".
2.  Red then claims a *first* territory (e.g., around `a1`).
3.  As part of processing this first territory, Red self-eliminates rings from its stack at `a8`. This results in `a8` now containing a Green stack (e.g., `G ◎ G (H1,C1)`).
4.  The game log indicates that at this point, the squares adjacent to `a8` are `a7` (Red marker/stack) and `b8` (Red marker/stack).
5.  **The Anomaly:** According to game rules ([Section 12.2](ringrift_complete_rules.md:1110), [12.3](ringrift_complete_rules.md:1202)), the game should re-check for new disconnected regions. `a8` (now Green) appears to meet the criteria to be a disconnected region claimable by Red:
    *   Physically disconnected by Red markers/stacks at `a7` and `b8`.
    *   Lacks Red/Blue representation (assuming Red/Blue have stacks elsewhere).
    *   Red can likely meet the self-elimination prerequisite.
    However, Red does not make this *second* territory claim on `a8` before its turn ends. Green subsequently claims `a8` on its turn.

The primary goal is to understand why the AI, after its first territory collapse (which resulted in `a8` becoming Green), did not then identify and process `a8` as a newly formed disconnected territory claimable by Red.

## 2. Debugging Phases

### Phase 1: Detailed State Reconstruction and Rule Application

1.  **Precise State Verification (Post-First-Collapse):**
    *   Utilize the game's replay/history feature to advance to Red's Turn 7.
    *   Step through the ring placement (`d4`), movement (`d4` to `a4`), and capture (`a4` over `a5` to `a8`).
    *   Pause execution *immediately after* the first "Board State After Territory Collapse (Player Red)" is fully resolved (i.e., after the `a1`-area is claimed and `a8` becomes Green due to Red's self-elimination from `a8`).
    *   At this exact micro-step, meticulously record or verify the state of:
        *   Square `a8` (expected: `G ◎ G`).
        *   Square `a7` (expected: `R ◉ R` or similar Red marker/stack).
        *   Square `b8` (expected: `R ◉ R` or similar Red marker/stack).
        *   All other orthogonally adjacent squares to `a8` to confirm its isolation by Red pieces.
        *   The presence of active Red and Blue stacks elsewhere on the board (relevant for the "Color Representation" rule).
        *   Red's remaining pieces/stacks on the board (to confirm the "Self-Elimination Prerequisite" can be met).

### Phase 2: AI Logic Interrogation for the "Re-check Disconnected Regions" Step

This phase focuses on the game's logic that should iterate on finding and processing disconnected regions as per [Section 4.5 Post-Movement Processing](ringrift_complete_rules.md:427) and [Section 12.3 Chain Reactions](ringrift_complete_rules.md:1202).

1.  **Trace Territory Re-evaluation Logic:**
    *   Examine the game code responsible for the "Post-Movement Processing" loop, specifically the mechanism that "check[s] again for disconnected regions" after a territory collapse occurs.
    *   **Key Questions for Code Analysis/Debugging:**
        *   Is this re-check loop or iterative step correctly entered after Red's first territory collapse is processed?
        *   When the re-check occurs, does the territory identification algorithm correctly identify the single square `a8` (now containing the Green stack) as a potential disconnected region?
        *   How does the algorithm define and evaluate the *boundary* of this `a8` region? Does it correctly interpret `a7` (Red) and `b8` (Red) as forming a complete Red-only border according to Von Neumann adjacency rules?
        *   Does the algorithm correctly apply the "Color Representation" rule (i.e., `a8` is Green, and thus lacks Red and Blue representation, assuming active Red and Blue players)?
        *   Does the algorithm correctly perform the "Self-Elimination Prerequisite Check" for Red (as detailed in [FAQ Q15](ringrift_complete_rules.md:1630) and [FAQ Q23](ringrift_complete_rules.md:1726))?
            *   What is the outcome of this prerequisite check? If it fails, what is the specific reason (e.g., AI incorrectly determines Red has no valid pieces to sacrifice)?
        *   If all conditions (Physical Disconnection, Color Representation, Self-Elimination Prerequisite) are met, what prevents the AI from initiating the claim of `a8`?
            *   Is there a flaw in the AI's action execution sequence for this potential second claim?
            *   Is there an incorrect heuristic, priority setting, or state flag in the AI that causes it to skip or ignore this mandatory claim? (The rules imply processing valid disconnections is not optional).

### Phase 3: Hypothesis Formulation

Based on the findings from Phase 2, potential hypotheses for the anomaly include:

*   **H1: Re-check Not Triggered:** The game logic fails to re-initiate the disconnected region check after the first territory collapse completes.
*   **H2: Boundary Misinterpretation:** The algorithm incorrectly evaluates the boundary around `a8`, perhaps not recognizing `a7` (Red) and `b8` (Red) as forming a valid Red-only border for disconnection purposes.
*   **H3: Color Representation Error:** The check for missing player colors within the `a8` region is flawed.
*   **H4: Self-Elimination Prerequisite Error:** The check for Red's ability to self-eliminate a piece is incorrect (either a bug in the check itself or an incorrect assessment of Red's available assets).
*   **H5: AI Decision/Action Flaw:** The AI correctly identifies `a8` as claimable, but a bug in its decision-making process or action execution prevents it from acting on this mandatory claim.
*   **H6: State Update Lag/Inconsistency:** The game state used by the re-check logic is stale and does not accurately reflect `a8` being Green or `a7`/`b8` being Red at that precise micro-step in the turn.

### Phase 4: Plan for Implementation (Hand-off to `code` mode)

1.  **Propose Code Investigation Focus:**
    *   Target source files for initial investigation:
        *   [`ringrift/src/rules/territory.rs`](ringrift/src/rules/territory.rs) (for territory identification and processing logic).
        *   [`ringrift/src/game/phase.rs`](ringrift/src/game/phase.rs) or equivalent (for turn structure, post-movement processing sequence, and iterative checks).
        *   The relevant AI module (e.g., [`ringrift/src/ai/minimax.rs`](ringrift/src/ai/minimax.rs) or other if a different AI is in use) for decision-making logic.
    *   **Debugging Techniques:**
        *   Suggest adding detailed logging statements at each critical step of the territory re-evaluation specifically for the `a8` region: boundary component identification, pieces within the region, color representation check results, self-elimination prerequisite check outcome, and the AI's final decision regarding the claim.
        *   Recommend using the game's replay feature in conjunction with a debugger to set breakpoints at these critical junctures to inspect live variable values and execution flow.

## 3. Mermaid Diagram: Expected Flow for Second Claim on `a8`

```mermaid
flowchart TD
    A[Red's Turn 7: First Territory Claim Processed (e.g., around a1)] --> B{a8 becomes G ◎ G due to Red's self-elimination from a8};
    B --> C{Rules: Re-check for new disconnected regions};
    C --> D{Algorithm: Identify potential region at a8 (Green)};
    D --> E{Algorithm: Evaluate boundary of a8};
    E --> F{Boundary pieces: a7 (Red), b8 (Red)};
    F --> G{Is a8 physically disconnected by Red markers only? (Von Neumann)};
    G -- Yes --> H{Does a8 lack Red & Blue representation? (a8 is Green)};
    H -- Yes --> I{Can Red meet Self-Elimination Prerequisite?};
    I -- Yes --> J[Expected: Red SHOULD claim a8 (Green)];
    J --> K{Actual: Did Red AI claim a8?};
    K -- No (Anomaly Observed) --> L[Investigate: Why AI did not claim? (Focus of this plan)];
    K -- Yes (If it did claim) --> M[Outcome: Problem lies elsewhere or was a misunderstanding of logs];

    I -- No --> N[Red CANNOT claim a8. Investigate prerequisite check failure.];

    style L fill:#ffeb3b,stroke:#333,stroke-width:2px
    style N fill:#ffcc00,stroke:#333,stroke-width:2px
    style M fill:#ccffcc,stroke:#333,stroke-width:2px
```

This plan aims to systematically diagnose why the expected second territory claim on `a8` by Red did not occur.