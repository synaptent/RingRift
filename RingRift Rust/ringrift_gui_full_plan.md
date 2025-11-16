# RingRift GUI Full Implementation Plan

This document outlines the comprehensive, phased plan to develop the RingRift GUI. The goal is to achieve full feature parity with the CLI, ensure strict adherence to all rules documented in [`ringrift_complete_rules.md`](ringrift_complete_rules.md), and provide a user-friendly, visually informative experience. This plan integrates insights from [`gui_implementation_roadmap.md`](gui_implementation_roadmap.md), [`gui_detailed_plan.md`](gui_detailed_plan.md), and analysis of [`ringrift/src/main.rs`](ringrift/src/main.rs).

## 1. Current Status & Goal

*   **Current GUI:** Possesses basic board rendering capabilities, hex selection, rudimentary action state management, and placeholder UI elements. The foundational next steps are outlined in [`gui_implementation_roadmap.md`](gui_implementation_roadmap.md).
*   **Existing Detailed Plan ([`gui_detailed_plan.md`](gui_detailed_plan.md)):** Provides a solid phased approach which this document expands upon.
*   **Goal:** A fully functional GUI enabling users to play RingRift according to all established rules, incorporating all setup and utility features currently available in the CLI, presented in an intuitive and visually clear manner.

## 2. Next Undone Implementation Steps (Immediate Priorities)

The immediate priorities are to establish the foundational elements for gameplay and game setup:

1.  **Implement Game Setup UI:** Allow users to configure board type, player count, and AI assignments.
2.  **Implement Core GUI Roadmap Steps 1-3:**
    *   **`GameState` Helper Methods:** Develop methods within [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs) to enable the GUI to query valid actions (placements, moves, captures).
    *   **Dynamic Visual Feedback in `BoardCanvas::draw`:** Utilize the helper methods to highlight valid action targets on the game board, implemented in [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs).
    *   **Refine Action Handling in `RingRiftGui::update`:** Connect UI interactions to `GameState` methods, manage `ActionState` transitions, and implement UI for selecting the number of rings during placement.

## 3. Detailed Implementation Plan

This plan integrates requirements from all analyzed documents into a phased structure.

### Phase 1: Foundation & Core Gameplay

#### 3.1. Implement Game Setup UI

*   **Objective:** Allow users to configure and start a new game, mirroring CLI options.
*   **Relevant CLI Features:**
    *   `--board-type` ([`ringrift/src/main.rs:179`](ringrift/src/main.rs:179))
    *   `--players` ([`ringrift/src/main.rs:183`](ringrift/src/main.rs:183))
    *   `--ai` ([`ringrift/src/main.rs:187`](ringrift/src/main.rs:187))
    *   `--seed` ([`ringrift/src/main.rs:211`](ringrift/src/main.rs:211))
    *   Interactive setup prompts in CLI ([`ringrift/src/main.rs:240-335`](ringrift/src/main.rs:240-335)).
*   **Relevant Rules:** Game versions (Rule 1.2), Player counts (Rule 1.2.1, Q19).
*   **UI Components:**
    *   Initial screen or modal dialog.
    *   Dropdown/Radio buttons for **Board Type**: "8x8 Square", "19x19 Square", "Hexagonal".
    *   Dropdown/Radio buttons/Number input for **Player Count**: 2, 3, 4.
    *   For each player slot (up to selected count):
        *   Dropdown/Radio buttons for **Player Type**: "Human", "AI".
    *   Text input field for **RNG Seed** (optional, uses random if blank).
    *   "Start Game" button.
    *   "Load Game" button (leads to functionality in Phase 2).
*   **User Interaction Flow:**
    1.  User launches application. Setup screen appears.
    2.  User selects board type, player count. Player type selectors update dynamically.
    3.  User assigns Human/AI to each player slot.
    4.  User (optionally) enters an RNG seed.
    5.  User clicks "Start Game".
*   **Data Validation:**
    *   Player count must be 2-4.
    *   Seed input should be parseable as `u64`.
    *   Ensure `GameState` is instantiated with the correct `BoardType`, player colors, AI indices, and seed.
*   **Visual Feedback:**
    *   Selections are clearly reflected in the UI.
    *   "Start Game" button enabled only when valid selections are made.
    *   Transition to the game board view upon starting.
    *   Display a summary of chosen settings briefly or in a persistent info panel.

#### 3.2. Implement GUI Roadmap Steps 1-3 (Core Interaction & Feedback)

*   **Objective:** Enable basic turn-based gameplay (placement, movement, initial capture) with clear visual cues for valid actions.
*   **Relevant CLI Features:** `place`, `move`, `capture` commands and their coordinate parsing logic ([`ringrift/src/main.rs:1117-1633`](ringrift/src/main.rs:1117-1633)).
*   **Relevant Rules:** Ring Placement (Rule 6), Non-Capture Movement (Rule 8), Capture (Overtaking) Movement (Rule 10), Turn Sequence (Rule 4).

    **3.2.1. `GameState` Helper Methods (Roadmap Step 1)**
    *   **Location:** [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs)
    *   **Methods:**
        *   `pub fn get_valid_placement_targets(&self) -> Vec<Position>`
        *   `pub fn get_valid_move_origins(&self) -> Vec<Position>` (pieces current player can move)
        *   `pub fn get_valid_move_targets_from(&self, origin: Position) -> Vec<Position>`
        *   `pub fn get_valid_capture_origins(&self) -> Vec<Position>` (pieces current player can use to capture)
        *   `pub fn get_valid_capture_over_targets_from(&self, origin: Position) -> Vec<Position>`
        *   `pub fn get_valid_capture_landing_targets(&self, origin: Position, over: Position) -> Vec<Position>`
    *   **Logic:** These methods will primarily use `GameState::get_valid_actions()` and filter/map results.

    **3.2.2. Enhance `BoardCanvas::draw` (Roadmap Step 2)**
    *   **Location:** [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs)
    *   **UI Components:** Visual highlights on the canvas.
    *   **Interaction Flow & Visual Feedback:**
        *   **`ActionState::Idle`:** Highlight pieces the current player can select for movement or capture (using `get_valid_move_origins`, `get_valid_capture_origins`).
        *   **`ActionState::PendingPlacement`:** Highlight valid empty spaces for ring placement (using `get_valid_placement_targets`).
        *   **`ActionState::PieceSelectedToMove(origin)`:** Highlight `origin` distinctively. Highlight valid destination spots (using `get_valid_move_targets_from(origin)`).
        *   **`ActionState::PieceSelectedToCapture(origin)`:** Highlight `origin` distinctively. Highlight opponent pieces that can be jumped over (using `get_valid_capture_over_targets_from(origin)`).
        *   **`ActionState::SelectingCaptureLanding(origin, over)`:** Highlight `origin` and `over` pieces. Highlight valid landing spots (using `get_valid_capture_landing_targets(origin, over)`).
    *   **Highlight Styles:** Use distinct colors/styles (e.g., translucent overlays, border colors) for different types of targets (placement, move, capture-over, capture-land).

    **3.2.3. Refine `RingRiftGui::update` (Roadmap Step 3)**
    *   **Location:** [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs)
    *   **UI Components:** Existing action buttons ("Place Ring", "Move", "Capture", "Skip Turn"), board click interaction. Add UI for `num_rings` placement.
        *   **Number of Rings for Placement:** A small number input widget (e.g., `iced::widget::NumberInput`) or +/- buttons, visible during `PendingPlacement` or when "Place Ring" is clicked. Default to 1. Max should be player's `rings_in_hand`.
    *   **User Interaction Flow & Logic:**
        *   **"Place Ring" Button Click:**
            *   If in `RingPlacement` phase: Set `action_state = ActionState::PendingPlacement`.
        *   **Board Click (Hex Selection):**
            *   **If `action_state == ActionState::PendingPlacement`:**
                *   `target_pos = clicked_hex`.
                *   `num_to_place = value_from_num_rings_widget`.
                *   Call `game_state.place_new_stack_on_board(target_pos, num_to_place, current_player_color)`.
                *   Handle `Ok`/`Err`. On `Ok`, log action, update GUI, reset `action_state`.
            *   **If `action_state == ActionState::Idle`:**
                *   If `clicked_hex` contains current player's piece: `selected_hex = Some(clicked_hex)`. Update highlights.
                *   Else: `selected_hex = None`.
            *   **If `action_state == ActionState::PieceSelectedToMove(origin)`:**
                *   `target_pos = clicked_hex`.
                *   Call `game_state.move_stack(&origin, &target_pos)`.
                *   Handle `Ok`/`Err`. On `Ok`, log action, update GUI, transition to `Capture` phase, reset `action_state`.
            *   **If `action_state == ActionState::PieceSelectedToCapture(origin)`:**
                *   `over_pos = clicked_hex`.
                *   If `over_pos` is a valid jump target: `action_state = ActionState::SelectingCaptureLanding(origin, over_pos)`. Update highlights for landing spots.
                *   Else: Invalid selection, provide feedback.
            *   **If `action_state == ActionState::SelectingCaptureLanding(origin, over)`:**
                *   `landing_pos = clicked_hex`.
                *   Call `game_state.execute_capture(&origin, &over, &landing_pos)`.
                *   Handle `Ok`/`Err`. On `Ok`, log action, update GUI. If chain captures possible, transition to `ActionState::PieceSelectedToCapture(landing_pos)` (for chain). Else, transition to `PostMovementProcessing`, reset `action_state`.
        *   **"Move" Button Click:**
            *   If `selected_hex` contains a piece owned by current player and in `Movement` phase: `action_state = ActionState::PieceSelectedToMove(selected_hex_pos)`.
        *   **"Capture" Button Click:**
            *   If `selected_hex` contains a piece owned by current player and in `Movement` or `Capture` phase: `action_state = ActionState::PieceSelectedToCapture(selected_hex_pos)`.
        *   **"Skip Turn/Phase" Button Click:**
            *   Generate appropriate `Action::Forfeit` or phase-specific skip.
            *   Call `game_state.apply_player_choice(...)` or `game_state.advance_phase(false)`.
            *   Log action. Update GUI.
    *   **Button Enabling/Disabling:**
        *   In `RingRiftGui::view`, enable/disable buttons based on `game_state.current_phase` and `game_state.get_valid_actions()`.
        *   E.g., "Place Ring" only enabled in `RingPlacement` if placements are possible and player has rings. "Move" only if valid moves exist. "Capture" only if valid captures exist.
    *   **Game History Logging:** Integrate `game_history_logger.record_action(&action, &game_state)` after every successful game state mutation.

### Phase 2: Handling Complex Rules & Core Features

#### 3.3. Implement Chain Capture UI & Logic

*   **Objective:** Allow users to correctly execute mandatory chain captures, including choosing between multiple valid segments.
*   **Relevant CLI Features:** CLI prompts for chain capture choices if multiple exist ([`ringrift/src/main.rs:1422-1504`](ringrift/src/main.rs:1422-1504)).
*   **Relevant Rules:** Rule 10.3 (Chain Overtaking is mandatory, player chooses path).
*   **UI Components:**
    *   When `game_state.is_chain_capture_active()` and multiple `PlayerChoice::ContinueChainCapture` options exist:
        *   A modal dialog or an overlay panel listing the available capture segments. Each choice should clearly indicate the piece to be jumped *over* and the *landing* position.
        *   Choices could be numbered or directly clickable representations on the board.
    *   "End Chain Capture" button (becomes active if rules allow ending, e.g., no more *mandatory* jumps). The `end` command in CLI for chain capture ([`ringrift/src/main.rs:1460`](ringrift/src/main.rs:1460)).
*   **User Interaction Flow:**
    1.  After a capture, if `game_state.start_chain_capture()` indicates further captures are possible from the landing spot:
    2.  GUI queries `game_state.get_valid_chain_capture_steps()`.
    3.  If multiple options: Display modal/overlay with choices. User clicks a choice.
    4.  If one option: Auto-select or briefly highlight and proceed.
    5.  GUI calls `game_state.continue_chain_capture(target, landing)`.
    6.  Board updates. Repeat from step 2 from new landing spot.
    7.  If no options, or user clicks "End Chain Capture": Call `game_state.end_chain_capture()`.
*   **Data Validation:**
    *   Ensure selected choice is from the valid list.
    *   `GameState` handles core validation of the capture itself.
*   **Visual Feedback:**
    *   Clear highlighting of the current piece that must make the chain capture.
    *   Highlighting of potential jump-over targets and landing spots for the selected/hovered chain segment.
    *   Message area confirms each segment of the chain capture.

#### 3.4. Implement Forced Cap Elimination Rule

*   **Objective:** Handle the scenario where a player has no valid standard moves but controls stacks, requiring them to eliminate a cap.
*   **Relevant CLI Features:** Logic for detecting this state and prompting for elimination ([`ringrift/src/main.rs:710-777`](ringrift/src/main.rs:710-777)).
*   **Relevant Rules:** Rule 4.4 (Forced Elimination When Blocked).
*   **UI Components:**
    *   Modal dialog or overlay.
    *   Lists the player's controllable stacks by position.
    *   "Eliminate Cap" button (or similar, might be implicit on selection).
*   **User Interaction Flow:**
    1.  At the start of a player's turn, GUI checks `game_state.has_legal_moves()` and `game_state.player_has_stacks()`.
    2.  If no legal moves but player has stacks: Display the modal.
    3.  User clicks on one of their stacks listed in the modal.
    4.  GUI calls `game_state.execute_force_eliminate_cap(chosen_position)`.
*   **Data Validation:**
    *   Ensure selected stack belongs to the current player and is a valid target.
    *   `GameState` handles core validation.
*   **Visual Feedback:**
    *   Clear message indicating why this action is required.
    *   Selected stack for cap elimination is highlighted.
    *   Board updates to show the eliminated cap.
    *   Message area confirms the action. Turn auto-advances.

#### 3.5. Implement Save/Load Functionality

*   **Objective:** Allow users to save their game progress and resume later.
*   **Relevant CLI Features:** `--load-file` argument ([`ringrift/src/main.rs:191`](ringrift/src/main.rs:191)), `save <filename>` command ([`ringrift/src/main.rs:1091`](ringrift/src/main.rs:1091)). `SavedGame` struct and its methods.
*   **UI Components:**
    *   Menu items: "File > Save Game", "File > Load Game".
    *   Native file dialogs for selecting save path/load file.
*   **User Interaction Flow:**
    *   **Save:** User clicks "Save Game". File dialog appears. User names file and saves.
    *   **Load:** User clicks "Load Game" (from setup screen or in-game menu). File dialog appears. User selects a `.json` game file.
*   **Data Validation:**
    *   Handle file I/O errors (e.g., permissions, file not found).
    *   Validate that the loaded file is a compatible `SavedGame` format.
*   **Visual Feedback:**
    *   Message area confirms "Game Saved to X" or "Game Loaded from Y".
    *   On load, the entire game view (board, player info, etc.) updates to the loaded state.
    *   Error messages for failed save/load operations.

### Phase 3: Polish & Advanced Features

#### 3.6. Implement GUI Roadmap Step 4 (UI Polish & Feedback)

*   **Objective:** Improve overall usability, information clarity, and aesthetics.
*   **UI Components:**
    *   **Dedicated Game Message Area:** A scrollable `Text` widget (e.g., `iced::widget::TextInput` in read-only mode or `iced::widget::Scrollable` containing `Text` widgets) to display game status, prompts, action results, and error messages.
    *   **Refined Player Info Panels:**
        *   Clearly display for each player: Name/ID, Color, AI/Human status, Rings in Hand, Eliminated Rings count, Collapsed Territory count.
        *   Highlight the current player.
    *   **Aesthetics:** Review and refine colors, fonts, spacing, button styles for better visual appeal and readability.
*   **Visual Feedback:**
    *   GUI provides clear, concise, and timely feedback for all actions and game state changes.

#### 3.7. Enhance Visual Feedback for Complex Events (Line/Territory Collapse)

*   **Objective:** Clearly communicate the process and results of line formations/collapses and territory disconnections.
*   **Relevant Rules:** Rule 11 (Line Formation & Collapse), Rule 12 (Area Disconnection & Collapse).
*   **UI Components & Visual Feedback:**
    *   **Line Formation/Collapse:**
        *   When a line is formed: Briefly highlight the markers forming the line.
        *   Animate or visually transition markers to collapsed space representation.
        *   If a ring/cap is eliminated: Briefly highlight the stack, show rings disappearing.
        *   Update player scores for eliminated rings and collapsed territory.
        *   Message area: "Player X formed a line of Y! Z markers collapsed. 1 ring eliminated."
    *   **Territory Disconnection/Collapse:**
        *   When a region is disconnected: Briefly highlight the disconnecting border and the enclosed region.
        *   Animate or visually transition the region's spaces and border markers to collapsed state of the claiming player's color.
        *   Show rings within the region being eliminated (e.g., fade out).
        *   If player self-eliminates a ring/cap: Show that.
        *   Update scores.
        *   Message area: "Player X disconnected a region! Y spaces collapsed. Z rings eliminated (including self-elimination)."
    *   Consider temporary canvas animations or overlays (e.g., flashing markers, color fills for regions, fading effects).

#### 3.8. Implement Graduated Line Reward Choice

*   **Objective:** Allow the player to choose the outcome when forming lines longer than the minimum required length.
*   **Relevant Rules:** Rule 11.2 (Option 1: Collapse All & Eliminate Ring vs. Option 2: Collapse Required & Keep Ring).
*   **UI Components:**
    *   Modal dialog when a line longer than required is formed (e.g., 6+ for 19x19/Hex, 5+ for 8x8).
    *   Dialog text: "Line of X markers detected (Y required). Choose an outcome:"
    *   Button 1: "Collapse all X markers & Eliminate 1 Ring/Cap"
    *   Button 2: "Collapse Y markers (your choice) & Keep Ring"
*   **User Interaction Flow:**
    1.  `PostMovementProcessor` detects a long line.
    2.  GUI displays the choice modal.
    3.  If user chooses Option 2:
        *   The modal might need to allow selection of *which* segment of Y markers to collapse.
        *   **Refined Interaction for Option 2:** After choosing Option 2, the GUI could highlight the long line and prompt "Click the first marker of the {Y} consecutive markers you wish to collapse." Then, after the click, it confirms the segment and processes.
    4.  User clicks a choice.
    5.  GUI informs `PostMovementProcessor` / `GameState` of the choice.
*   **Data Validation:** Ensure the choice is valid.
*   **Visual Feedback:** Message area confirms the choice and the outcome. Board updates accordingly.

#### 3.9. Implement Detailed Game Over/Stalemate Display

*   **Objective:** Clearly present the game's result, including winner, reason, and tiebreaker details if applicable.
*   **Relevant CLI Features:** Printing winner, reason, and stalemate tiebreaker scores ([`ringrift/src/main.rs:631-705`](ringrift/src/main.rs:631-705)).
*   **Relevant Rules:** Rule 13 (Victory Conditions), Rule 13.4 (End of Game "Stalemate" Resolution).
*   **UI Components:**
    *   A final screen or modal dialog displayed when `game_state.game_over` is true.
    *   **Content:**
        *   "Game Over!" title.
        *   Winner: "Player X (Color) Wins!" or "Stalemate!"
        *   Reason: e.g., "By Ring Elimination", "By Territory Control", "Last Player Standing", "Stalemate Resolution".
        *   **If Stalemate:** Detailed breakdown of tiebreaker scores for all players (Collapsed Spaces, Eliminated Rings (incl. hand), Markers).
        *   Final scores for all players (eliminated rings, territory).
    *   Buttons: "Play Again" (returns to Game Setup Screen), "Quit".
*   **Visual Feedback:** Clear, unambiguous presentation of the game outcome and all relevant scoring details.

#### 3.10. AI Integration

*   **Objective:** Allow users to play against AI opponents as configured in the Game Setup.
*   **Relevant CLI Features:** `--ai` flag, `RandomAI::get_ai_command` ([`ringrift/src/main.rs:1049`](ringrift/src/main.rs:1049)).
*   **Logic:**
    *   In `RingRiftGui::update`, if `game_state.current_player().is_ai`:
        *   Disable human input for board interaction.
        *   Call the appropriate AI function (e.g., `RandomAI::get_ai_command(&game_state, &mut seeded_rng)`).
        *   The AI function will return a command string.
        *   Feed this command string into the existing CLI command parsing and action execution logic within the GUI's `update` method.
*   **UI Components & Visual Feedback:**
    *   Player info panel should clearly indicate which players are AI.
    *   When it's AI's turn:
        *   Display a message like "Player X (AI) is thinking..."
        *   Optionally, add a short artificial delay.
        *   After AI makes a move, the board updates, and the message area shows what action the AI took.

### Phase 4: Future Enhancements

#### 3.11. Replay Functionality

*   **Objective:** Allow users to load a saved game history and step through actions.
*   **Relevant CLI Features:** `--replay-file`, `--turn`, `--player_idx`, `--validate` ([`ringrift/src/main.rs:195-208`](ringrift/src/main.rs:195-208)). Logic for replaying actions ([`ringrift/src/main.rs:436-580`](ringrift/src/main.rs:436-580)).
*   **UI Components:**
    *   "Replay Game" option on the initial screen or File menu.
    *   File dialog to select a saved game `.json` file.
    *   Replay controls panel: "Play/Pause", "Next Action", "Previous Action", "Go to Start", "Go to End", Slider/input for turn/action number.
    *   Display area for current action being replayed.
*   **User Interaction Flow:**
    1.  User selects "Replay Game", chooses file.
    2.  Game board loads in initial state.
    3.  User uses replay controls.
*   **Logic:** Load `SavedGame`. Instantiate `GameState`. Apply actions one by one. "Previous Action" is complex.
*   **Visual Feedback:** Board updates with each action. Current action details displayed.

#### 3.12. Advanced Configuration & Other Features

*   **Specific AI Types:** Selection in Game Setup.
*   **Help/Rules Summary:** In-GUI screen or link.
*   **Coordinate Display:** Option on board.
*   **Sound Effects:** Optional.

## 4. General Considerations for GUI Implementation

*   **Error Handling:** Robustly handle errors from `GameState` operations; display user-friendly messages.
*   **Responsiveness:** Ensure GUI remains responsive. Use asynchronous operations (`iced::Command`) if needed.
*   **State Management:** Carefully manage `RingRiftGui` state.
*   **Modularity:** Separate rendering (`view`) from update logic (`update`).
*   **Testing:** Unit tests for `GameState` helpers. Thorough manual UI testing.

## 5. High-Level UI Flow Diagram

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