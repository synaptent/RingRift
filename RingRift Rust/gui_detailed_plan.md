# RingRift GUI Detailed Implementation Plan

This document outlines the prioritized, multi-phase plan to enhance the RingRift GUI, aiming for feature parity with the CLI and adherence to the complete game rules.

## 1. Information Gathered Summary:

*   **GUI Roadmap ([`gui_implementation_roadmap.md`](gui_implementation_roadmap.md)):** Provides a 4-step plan for core gameplay interaction:
    1.  Implement `GameState` helper methods for valid action queries ([`ringrift/src/game/state.rs`](ringrift/src/game/state.rs)).
    2.  Enhance `BoardCanvas::draw` with dynamic highlighting.
    3.  Refine action handling in `RingRiftGui::update`.
    4.  UI polish (message area, player info, aesthetics).
*   **Rules Document ([`ringrift_complete_rules.md`](ringrift_complete_rules.md)):** Details mechanics for 3 game versions, complex turn structures, stack/cap height, movement/capture/landing rules, line formation (4+ or 5+) with graduated rewards, territory disconnection, victory conditions, and stalemate resolution.
*   **CLI ([`ringrift/src/main.rs`](ringrift/src/main.rs)):** Implements game setup (board type, player count, AI), core gameplay commands (`place`, `move`, `capture`, `skip`), save/load, replay, forced cap elimination, coordinate parsing, and integration with game logic modules.

## 2. Analysis & Gap Identification Summary:

The existing GUI roadmap covers essential interaction. Key remaining gaps for full CLI feature parity and rule adherence include:

*   **Save/Load UI & Logic:** Currently, "Load Game" is a placeholder. Save functionality is missing.
*   **Complex Rule Handling & Feedback (for Human Players):**
    *   **Forced Cap Elimination UI:** Human players need a way to select which cap to eliminate.
    *   **Graduated Line Reward Choice UI:** Human players need to be prompted to choose between line collapse options.
    *   **Player Choice for Self-Elimination (Lines/Territory):** Human players need to select which ring/cap to eliminate when required by these events.
*   **Game Over/Stalemate Display:** Needs a dedicated, detailed screen instead of just a game message.
*   **Visual Feedback for Complex Events:** While basic board updates occur, more explicit visual cues for line collapses and territory disconnections could be beneficial.

## 3. Prioritized Plan for GUI Implementation:

### Phase 1: Foundation & Core Gameplay
*   **STATUS: LARGELY COMPLETE**

1.  **Implement Game Setup UI:**
    *   **STATUS: COMPLETED.** UI for board type, player count, Human/AI, and seed input is functional.
    *   **Goal:** Allow users to configure and start a new game.
    *   **UI Components:** Initial screen/modal with selections for Board Type, Player Count, Human/AI assignment, "Start Game" button, (Optional) RNG Seed input.
    *   **Interaction:** User selects options, clicks "Start Game".
    *   **Validation:** Ensure player count is valid (2-4).
    *   **Feedback:** Display summary of choices. Instantiate `GameState` based on selections.

2.  **Implement GUI Roadmap Steps 1-3 (Core Interaction & Feedback):**
    *   **STATUS: LARGELY COMPLETE.** `GameState` helpers are done. `BoardCanvas::draw` has dynamic highlighting. `RingRiftGui::update` handles core actions, state transitions, button enabling, history logging, and `num_rings` input.
    *   **Goal:** Enable basic turn-based gameplay (placement, movement, capture) with visual feedback.
    *   **Step 1 (Roadmap):** Implement `GameState` helper methods (`get_valid_placement_targets`, `get_valid_move_targets_from`, etc.) in [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs). **(DONE)**
    *   **Step 2 (Roadmap):** Enhance `BoardCanvas::draw` in [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs) to use helper methods for highlighting valid targets. **(DONE)**
    *   **Step 3 (Roadmap):** Refine `RingRiftGui::update` in [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs): **(LARGELY DONE - needs ongoing testing/refinement)**
        *   Connect UI interactions to `GameState` methods.
        *   Manage `ActionState` transitions.
        *   Implement basic button enabling/disabling.
        *   Integrate `GameHistoryLogger`.
        *   **Add:** UI element (e.g., number input) for `num_rings` selection during placement. **(DONE)**

### Phase 2: Handling Complex Rules & Core Features
*   **STATUS: PARTIALLY IMPLEMENTED**

3.  **Implement Chain Capture UI & Logic:**
    *   **STATUS: LARGELY IMPLEMENTED.** GUI handles multi-step captures and transitions to `SelectingChainCaptureLanding`. AI can end chains.
    *   **Goal:** Allow users to correctly execute mandatory chain captures with choices.
    *   **UI Components:** Modal/overlay listing numbered choices for multiple valid chain capture segments. "End Chain Capture" button. (Highlighting for next step exists, explicit "End Chain" button for humans could be added if rules allow premature end).
    *   **Interaction:** User clicks a choice or "End Chain Capture".
    *   **Validation:** Ensure selection is valid. Call `continue_chain_capture` or `end_chain_capture`.
    *   **Feedback:** Update board state. Message area confirms action.

4.  **Implement Forced Cap Elimination Rule:**
    *   **STATUS: BACKEND COMPLETE, GUI FOR HUMAN PENDING.** `GameState` handles logic. AI uses it.
    *   **Goal:** Handle scenario where a player has no valid moves but controls stacks.
    *   **Trigger:** Check `has_legal_moves` and `player_has_stacks`.
    *   **UI Components:** **(PENDING for Human)** Modal/overlay listing controllable stacks for cap elimination.
    *   **Interaction:** User selects a stack.
    *   **Validation:** Ensure selection is valid. Call `execute_force_eliminate_cap`.
    *   **Feedback:** Message area confirms. Update board. Auto-advance turn.

5.  **Implement Save/Load Functionality:**
    *   **STATUS: PENDING.** Load button is a placeholder. Save is missing.
    *   **Goal:** Allow users to save progress and resume.
    *   **UI Components:** Menu items/buttons ("Save Game", "Load Game"). Native file dialogs.
    *   **Interaction:** User selects option, chooses file path.
    *   **Validation:** Handle file I/O errors. Deserialize `SavedGame` and update application state.
    *   **Feedback:** Message area confirms success/failure.

### Phase 3: Polish & Advanced Features
*   **STATUS: PARTIALLY IMPLEMENTED**

6.  **Implement GUI Roadmap Step 4 (UI Polish & Feedback):**
    *   **STATUS: PARTIALLY IMPLEMENTED.** Message area and player info exist.
    *   **Goal:** Improve usability and information display.
    *   **UI Components:** Dedicated `Text` area for messages. Refined player info panels. Review/improve aesthetics.
    *   **Feedback:** Clear status updates and error messages.

7.  **Enhance Visual Feedback for Complex Events:**
    *   **STATUS: BASIC IMPLEMENTATION.** Board updates, messages appear.
    *   **Goal:** Clearly communicate results of line collapses and territory disconnections.
    *   **UI Components:** Use message area. Consider temporary canvas animations/overlays (flashing collapsing markers, highlighting disconnected regions, fading eliminated rings).
    *   **Feedback:** Combine text messages with visual cues.

8.  **Implement Graduated Line Reward Choice:**
    *   **STATUS: BACKEND COMPLETE, GUI FOR HUMAN PENDING.** `GameState` can handle this.
    *   **Goal:** Allow player to choose outcome for lines longer than required.
    *   **UI Components:** **(PENDING for Human)** Modal: "Line of X detected. Choose: [Collapse All & Eliminate Ring] or [Collapse Y Markers & Keep Ring]".
    *   **Interaction:** User clicks a choice.
    *   **Validation:** Pass choice to `PostMovementProcessor`.
    *   **Feedback:** Message area confirms choice and outcome.

9.  **Implement Detailed Game Over/Stalemate Display:**
    *   **STATUS: PENDING.** Basic message exists.
    *   **Goal:** Clearly show game result and reasons.
    *   **UI Components:** Final screen/modal displaying winner/condition, stalemate tiebreaker results, final scores.
    *   **Feedback:** Clear presentation of game outcome.

10. **AI Integration:**
    *   **STATUS: COMPLETED.** AI turns are triggered, commands processed.
    *   **Goal:** Allow playing against AI opponents.
    *   **Logic:** When AI's turn, call AI command function, feed command into input processing.
    *   **UI:** Indicate when AI is "thinking".

### Phase 4: Future Enhancements

11. **Replay Functionality:** UI for selecting replay file and stepping through actions.
12. **Advanced Configuration:** More detailed options (e.g., specific AI types).
13. **Network Play:** (Significant effort, outside current scope).

## High-Level UI Flow Diagram

```mermaid
graph TD
    A[Start Application] --> B{Load/Replay Args?};
    B -- Yes --> C[Load/Replay Logic];
    B -- No --> D[Show Setup Screen];
    D -- Start Game --> E[Initialize GameState];
    C --> E;
    E --> F{Game Over?};
    F -- No --> G[Display Board & Info];
    G --> H{Current Player Turn};
    H --> I{AI or Human?};
    I -- AI --> J[Get AI Command];
    I -- Human --> K[Await Human Input/Interaction];
    J --> L[Process Command/Action];
    K --> L;
    L -- Valid Action --> M[Update GameState];
    L -- Invalid Action/Input --> G; %% Re-display/prompt
    M --> N[Run Post-Movement Processing];
    N --> F; %% Check Game Over again
    F -- Yes --> O[Show Game Over Screen];
    O --> P{Play Again?};
    P -- Yes --> D;
    P -- No --> Q([Exit Application]);

    subgraph Setup Screen [D]
        direction LR
        D1[Select Board Type] --> D2[Select Player Count] --> D3[Assign Human/AI] --> D4[Start Button]
    end

    subgraph Gameplay Loop [G, H, I, J, K, L, M, N]
        direction TB
        G --> H --> I
        I --> J & K
        J & K --> L
        L --> M --> N
    end

    subgraph Game Over Screen [O]
        O1[Display Winner/Reason] --> O2[Display Scores/Tiebreakers] --> O3[Play Again/Quit Buttons]
    end