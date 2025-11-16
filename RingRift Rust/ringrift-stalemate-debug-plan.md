# RingRift Stalemate Debugging Plan (Turn 10 Anomaly)

## 1. Core Issue Identified

The primary bug appears to be a **game flow/turn management error**. The check for "no standard moves available" (Rule 4.4: Forced Elimination When Blocked), which can lead to forced cap elimination, is occurring at the wrong point in Player P0's Turn 10 in the provided game log. It happens *after* P0's placement and mandatory capture, whereas the (now clarified) `ringrift_complete_rules.md` (Section 4.4) and game logic flowchart (Section 15.2.1) indicate this check should occur at the *beginning* of a player's turn, before any other actions are taken. This incorrect timing is the likely direct cause of the subsequent erroneous stalemate.

## 2. Phase 1: Verification & Rule Interpretation (Focus on Turn Start)

### 2.1. Confirm Rule 4.4 Timing
- **Objective:** Ensure the team understands that the clarified Rule 4.4 in `ringrift_complete_rules.md` (Section 4.4) mandates the "Forced Elimination When Blocked" check at the *start* of a player's turn.
- **Reference:** The recent update to Section 4.4 explicitly states: "At the *beginning* of a player's turn, before any placement... or movement selection... is made, if that player has no valid placement, standard move, or capture option available..."

### 2.2. Analyze P0's State at the *Start* of Turn 10
- **Objective:** Determine if P0 *actually* had no valid placements, standard moves, or capture options available *before* any action was taken in Turn 10.
- **Board State Reference:** Use the board state from the *end of P2's Turn 9 / start of P0's Turn 10* (as provided in the initial game log).
- **P0's Resources at Start of Turn 10:**
    - Rings in Hand: 1
    - Controlled Stacks: `a5 (H4,C2)`, `h2 (H3,C3)`, `e1 (R ◉ R)`, `c2 (R ◉ R)`, `f2 (R ◉ R)`, `g2 (R ◉ R)`, and other `▓▓ R ◉ R ▓▓` stacks.
- **Actions to Verify for P0 at Turn Start:**
    1.  **Valid Placements:** Could P0 place its 1 ring in hand on an empty space or an existing stack such that the resulting stack would have a legal move? (Rules 4.1, 6.1-6.3). (The log shows `a2` was chosen, suggesting this was possible).
    2.  **Valid Standard Moves:** For *each* of P0's controlled stacks at the start of Turn 10, were there any valid standard moves? (Rules 4.2.1, 8.1/16.4.1).
    3.  **Valid Captures:** For *each* of P0's controlled stacks at the start of Turn 10, were there any valid overtaking captures? (Rules 4.3, 10.1).
- **Expected Outcome:** If any of these options were available, the "no standard moves available" condition (Rule 4.4) should *not* have been met at the start of Turn 10.

### 2.3. Note Secondary Log vs. Rule Discrepancies
- **Objective:** Keep track of other deviations between the log and rules, as these indicate additional bugs.
- **Key Examples:**
    - Capture path/mechanics for `a2` over `e2` to `g2` (potential pass-through, state of `e2` changing to Green).
    - Unexplained changes to markers/pieces at `b2`, `c2`.
    - Unlogged line formation implied by `f2` collapse and score change.
    - `a5` stack changing to Green control after Red's cap elimination.
    - P0's "Eliminated" score update timing.

## 3. Phase 2: Debugging Strategy (Code Investigation)

### 3.1. Hypotheses
- **Primary Hypothesis (H_Flow):** Incorrect Turn/Phase Management. The main game loop or phase transition logic incorrectly calls/allows the Rule 4.4 check *after* a placement and mandatory move, instead of at the beginning of the turn.
- **Secondary Hypotheses:**
    - **H_MoveGen:** Flawed move generation (less likely for the primary issue if P0 could act, but could affect the *initial* turn start check).
    - **H_StateUpdate:** Incorrect game state updates after actions (contributing to the secondary discrepancies).
    - **H_Stalemate:** Premature or incorrect stalemate check (a consequence of H_Flow).

### 3.2. Key Code Modules for Inspection
- **Highest Priority (Game Flow):**
    - `ringrift/src/game/turn.rs`
    - `ringrift/src/game/phase.rs`
    - `ringrift/src/main.rs` or `ringrift/src/lib.rs` (main game loop)
- **Secondary Priority (State Updates, Move Logic, Victory Conditions):**
    - `ringrift/src/rules/movement.rs`
    - `ringrift/src/rules/capture.rs`
    - `ringrift/src/game/state.rs`
    - `ringrift/src/game/action.rs`
    - `ringrift/src/rules/post_movement_processor.rs`
    - `ringrift/src/rules/victory.rs`

### 3.3. Debugging Steps & Data Logging
1.  **Trace Turn Initiation for P0 Turn 10:**
    - **Log Points (at the very beginning of P0's Turn 10, before any P0 action):**
        - All available placement options for P0.
        - All available standard moves for all of P0's stacks.
        - All available capture moves for all of P0's stacks.
        - Explicitly log whether the Rule 4.4 check (Forced Elimination When Blocked) is performed *at this stage* and its outcome.
2.  **Trace P0's Actual Turn 10 Actions (as observed in log):**
    - Log board state *after* placement at `a2`.
    - Log board state *immediately after* the capture `a2` over `e2` to `g2` (focus on `a2`-`h2` row and scores).
    - Pinpoint exactly *when and why* the "Player 0 (Red) has no standard moves available" message is generated in the code. Is it a separate check run *after* the capture, or an incorrect status returned by the capture/move function?
3.  **Investigate Forced Cap Elimination at `a5` (if the erroneous check triggers it):**
    - Log the state of the `a5` stack *before* and *after* elimination.
    - Log P0's "Eliminated" score *before and after* this specific action.
4.  **Examine Stalemate Trigger:**
    - Confirm it's triggered due to the mis-timed "no moves" check for P0.
    - Verify that P1 (Blue) and P2 (Green) are not evaluated for moves before stalemate is declared.
    - Log the criteria values used in the stalemate tiebreaker.

## 4. Phase 3: Proposed Corrections

### 4.1. Correct Game Flow / Turn Management (Primary Fix)
- **Target Modules:** `ringrift/src/game/turn.rs`, `ringrift/src/game/phase.rs`, main game loop.
- **Action:**
    - Ensure the check for "no valid placement, move, or capture option available" (Rule 4.4) is performed *exclusively* at the beginning of a player's turn.
    - If options are available, the player proceeds with their turn (placement/movement).
    - If no options are available AND the player controls stacks, then forced cap elimination occurs, followed by passing the turn (unless victory).
    - Remove or relocate any logic that performs this check mid-turn after an action has been taken.

### 4.2. Address Secondary State Update Issues
- **Target Modules:** `ringrift/src/rules/capture.rs`, `ringrift/src/game/action.rs`, `ringrift/src/game/state.rs`, `ringrift/src/rules/post_movement_processor.rs`.
- **Action:** Based on findings from debugging secondary discrepancies:
    - Correct logic for updating the state of captured pieces (e.g., `e2` color).
    - Ensure only intended pieces/markers are modified during moves/captures.
    - Correctly process and log line formations if they occur as a side effect of a capture.
    - Ensure stack state and player scores are updated promptly and correctly after actions like cap elimination (e.g., `a5` control, P0's elimination score).

## 5. Conceptual Debugging Flow Diagram

```mermaid
graph TD
    A[Start P0 Turn 10 - BEGINNING] --> B{Check ALL Valid Options for P0 (Placements, Standard Moves, Captures)};
    B --> C{Any Valid Option Exists?};
    C -- Yes --> D[P0 Proceeds with Turn (e.g., Places at a2)];
    D --> E[P0 Moves Placed Piece (e.g., Captures a2 over e2 to g2)];
    E --> F[Log Board State Post-Action & Score];
    F --> G{Line Formation Triggered?};
    G -- Yes --> H[Process Line(s) & Log];
    G -- No --> I[End of P0 Action Phase];
    H --> I;
    I --> J[Pass Turn to P1 (Blue) - CORRECTED PATH];

    C -- No (No Options at Turn Start) --> K{P0 Controls Stacks?};
    K -- Yes --> L[Rule 4.4: Forced Cap Elimination];
    L --> M[Log Elimination & Score];
    M --> J;
    K -- No (No stacks to elim) --> J;

    subgraph "Logged Anomaly Path (To Be Corrected)"
        direction LR
        E --> X[BUG: Erroneous Rule 4.4 Check AFTER P0's action];
        X --> Y[Forced Cap Elim at a5 (due to bug)];
        Y --> Z[Incorrect Stalemate Declared (due to bug)];
    end

    subgraph "Key Code Modules for Game Flow"
        direction LR
        CM_Turn[game/turn.rs]
        CM_Phase[game/phase.rs]
        CM_Main[main.rs / lib.rs Game Loop]
    end
    subgraph "Key Code Modules for State/Rules Logic"
        direction LR
        CM_Movement[rules/movement.rs]
        CM_Capture[rules/capture.rs]
        CM_State[game/state.rs]
        CM_Action[game/action.rs]
        CM_Victory[rules/victory.rs]
    end

    B -.-> CM_Movement;
    B -.-> CM_Capture;
    B -.-> CM_Turn;
    C -.-> CM_Turn;
    C -.-> CM_Phase;
    E -.-> CM_Capture;
    E -.-> CM_Action;
    E -.-> CM_State;
    X -.-> CM_Phase; %% Investigate where this erroneous check is called
    Z -.-> CM_Victory;
```

This plan should provide a clear roadmap for debugging and resolving the incorrect stalemate issue.