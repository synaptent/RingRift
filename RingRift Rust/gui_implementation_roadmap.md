# RingRift GUI: Next Steps Implementation Roadmap

This document outlines the detailed execution roadmap for the next critical steps in implementing the RingRift GUI, building upon the existing foundation. The primary goal is to enhance interactivity and visual feedback, making the GUI more functional for gameplay.

## Current GUI Status Summary:
*   **Framework**: `iced`
*   **Core Structure**: Basic `Application` loop, `GameState` integration.
*   **Board Rendering**:
    *   Displays a grid of intersection points and triangular lines.
    *   Renders rings, markers, and collapsed spaces with distinct visual styles.
    *   Hexagon math and rendering adjusted for flat-top hexes, improving tiling.
    *   Board elements scaled down by ~50%.
    *   Intersection dots are brighter than grid lines.
*   **Interactivity**:
    *   Hex selection via mouse click is implemented.
    *   Selected hexes are highlighted.
    *   `ActionState` management (`Idle`, `PendingPlacement`, `PieceSelectedToMove`, `PieceSelectedToCapture`, `SelectingInitialCaptureLanding`, `SelectingChainCaptureLanding`) is implemented.
    *   Buttons for "Place Ring", "Move", "Capture", "Skip Turn".
    *   Logic in `update` method handles these actions and transitions.
*   **Player Info**: Basic display of player stats and current turn/phase.
*   **Game Setup**: UI for selecting board type, player count, player types (Human/AI), and RNG seed is functional.
*   **AI Integration**: Basic AI integration is functional, with the GUI triggering AI turns and processing AI-generated command strings.

## Next Critical Steps Roadmap:

### Step 1: Implement `GameState` Helper Methods for Valid Action Queries
*   **STATUS: COMPLETED**
*   **Objective**: Enable the GUI to query `GameState` for valid actions and target locations, which is crucial for providing accurate visual feedback and enabling/disabling controls.
*   **Key Tasks**:
    1.  **Define and Implement in [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs)**:
        *   `pub fn get_valid_placement_targets(&self) -> Vec<Position>`: Returns a list of all empty positions where the current player can legally place a ring.
        *   `pub fn get_valid_move_targets_from(&self, origin: Position) -> Vec<Position>`: Given an `origin` position with the current player's piece, returns all valid target positions for a move.
        *   `pub fn get_valid_capture_over_targets_from(&self, origin: Position) -> Vec<Position>`: Given an `origin` position with the current player's piece, returns all positions of opponent pieces that can be jumped over.
        *   `pub fn get_valid_capture_landing_targets(&self, origin: Position, over: Position) -> Vec<Position>`: Given an `origin` and an `over` position, returns all valid landing positions for the capture.
    2.  **Logic for Helper Methods**: These methods will likely iterate through the results of the existing `get_valid_actions()` method (which returns `PlayerChoice` enum variants) and filter/map them to extract the relevant `Position` data.
*   **Resources**:
    *   Existing `GameState::get_valid_actions()` method.
    *   Game rules for placement, movement, and capture.
    *   [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs)
    *   [`ringrift/src/game/action.rs`](ringrift/src/game/action.rs) (for `PlayerChoice` structure)
*   **Timeline**: 2-3 hours (requires careful implementation and testing of game logic).
*   **Success Indicators**:
    *   All four helper methods are implemented in `GameState`.
    *   Unit tests for these methods pass, covering various game scenarios (e.g., board edges, blocked paths, different phases).
    *   Methods correctly return empty vectors when no valid targets exist.

### Step 2: Enhance `BoardCanvas::draw` with Dynamic Visual Feedback
*   **STATUS: LARGELY IMPLEMENTED**
*   **Objective**: Use the new `GameState` helper methods to provide dynamic visual feedback on the canvas, highlighting valid action targets based on the current `ActionState`.
*   **Key Tasks** (in [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs) - `BoardCanvas::draw`):
    1.  **PendingPlacement Highlighting**:
        *   If `self.action_state == ActionState::PendingPlacement`, call `self.game_state.get_valid_placement_targets()`.
        *   For each position returned, draw a distinct highlight (e.g., light blue translucent overlay or border) on the corresponding intersection point.
    2.  **PieceSelectedToMove Highlighting**:
        *   If `self.action_state == ActionState::PieceSelectedToMove(origin)`, call `self.game_state.get_valid_move_targets_from(origin)`.
        *   Highlight the `origin` piece distinctively (already partially done).
        *   For each target position returned, draw a "valid move target" highlight (e.g., light green overlay/border).
    3.  **PieceSelectedToCapture Highlighting**:
        *   If `self.action_state == ActionState::PieceSelectedToCapture(origin)`, call `self.game_state.get_valid_capture_over_targets_from(origin)`.
        *   Highlight the `origin` piece.
        *   For each "over" target position returned, draw a "valid capture jump" highlight (e.g., light red overlay/border).
        *   *(Future sub-task: If a jump target is then clicked, this would transition to a new state like `SelectingCaptureLanding`, and then `get_valid_capture_landing_targets` would be used to highlight landing spots).* - This sub-task is also implemented.
*   **Resources**:
    *   Newly implemented `GameState` helper methods.
    *   [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs) (`BoardCanvas::draw` method).
    *   `iced::widget::canvas::Frame` drawing primitives.
*   **Timeline**: 2-4 hours (involves careful conditional drawing logic).
*   **Success Indicators**:
    *   Valid placement spots are clearly highlighted when `ActionState` is `PendingPlacement`.
    *   When a piece is selected for a move, valid destination points are highlighted.
    *   When a piece is selected for a capture, valid pieces to jump over are highlighted.
    *   Highlighting updates correctly as `ActionState` and `selected_hex` change.

### Step 3: Refine Action Handling in `RingRiftGui::update`
*   **STATUS: PARTIALLY IMPLEMENTED - Core logic for placement, move, capture (including multi-step and chain), button enabling, history logging is present. Needs thorough testing and refinement for complex rule interactions and state transitions.**
*   **Objective**: Make the GUI action buttons and hex clicks fully drive the game logic by correctly calling `GameState` methods and managing `ActionState` transitions.
*   **Key Tasks** (in [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs) - `RingRiftGui::update`):
    1.  **Placement Logic**:
        *   `Message::AttemptPlacement`: If in `RingPlacement` phase, set `action_state = ActionState::PendingPlacement`.
        *   `Message::ClickHex` when `action_state == ActionState::PendingPlacement`:
            *   Call `self.game_state.place_new_stack_on_board(clicked_pos, num_rings, current_player_color)` or `add_rings_to_stack`. (UI for `num_rings` selection exists).
            *   Handle `Ok` (update `game_state.placed_piece_pos`, advance phase, log action) and `Err` (display message).
            *   Reset `action_state = ActionState::Idle`, `selected_hex = None`.
    2.  **Move Logic**:
        *   `Message::AttemptMove`: If in `Movement` phase and `selected_hex` contains a valid piece of the current player, set `action_state = ActionState::PieceSelectedToMove(selected_hex_pos)`. Otherwise, reset to `Idle`.
        *   `Message::ClickHex` when `action_state == ActionState::PieceSelectedToMove(origin_pos)`:
            *   `target_pos = clicked_pos`.
            *   Call `self.game_state.move_stack(&origin_pos, &target_pos)`.
            *   Handle `Ok` (update `game_state` fields like `last_move_landing_pos`, `has_moved`, transition to `Capture` phase, log action) and `Err`.
            *   Reset `action_state = ActionState::Idle`, `selected_hex = None`.
    3.  **Capture Logic (Initial & Chain)**:
        *   `Message::AttemptCapture`: If in `Movement` or `Capture` phase and `selected_hex` contains a valid piece, set `action_state = ActionState::PieceSelectedToCapture(selected_hex_pos)`.
        *   `Message::ClickHex` when `action_state == ActionState::PieceSelectedToCapture(origin_pos)`:
            *   `target_jump_over_pos = clicked_pos`.
            *   Find valid landing spots using `get_valid_capture_landing_targets`.
            *   If one landing spot, attempt `self.game_state.execute_capture()` or `continue_chain_capture`.
            *   If multiple, transition to `SelectingInitialCaptureLanding` or `SelectingChainCaptureLanding`.
            *   Handle `Ok`:
                *   Update `game_state.has_captured`, log action.
                *   If chain continues, update `action_state` and `selected_hex`.
                *   Else, transition phase, reset `action_state`.
            *   Handle `Err`.
    4.  **Button Enabling/Disabling**:
        *   In `RingRiftGui::view`, conditionally enable/disable action buttons based on `self.game_state.current_phase` and the output of `self.game_state.get_valid_actions()`. For example, "Place Ring" button only enabled during `RingPlacement` phase if placements are possible.
    5.  **Game History Logging**: Integrate calls to a `GameHistoryLogger` instance after successful actions.
*   **Resources**:
    *   [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs) (`RingRiftGui::update`, `Message`, `ActionState`).
    *   `GameState` methods for performing actions and validation.
    *   `iced::widget::Button` API for enabling/disabling.
*   **Timeline**: 4-6 hours (complex state transitions and game rule integration).
*   **Success Indicators**:
    *   Players can complete full turns (placement, move/capture) using GUI interactions.
    *   `ActionState` transitions correctly manage multi-step actions.
    *   Game logic from `GameState` is correctly invoked and respected.
    *   Action buttons are appropriately enabled/disabled.
    *   Basic error messages or feedback are provided for invalid actions (e.g., via `println!`).

### Step 4: UI Enhancements and Polish
*   **STATUS: PARTIALLY IMPLEMENTED - Game message area and player info display exist. Game Setup UI is functional. Missing: Load Game, Save Game, dedicated Game Over screen.**
*   **Objective**: Improve the user experience with a more polished UI.
*   **Key Tasks**:
    1.  **Dedicated Game Message Area**: Add a `Text` widget in `RingRiftGui::view` to display game status, prompts, and error messages from action attempts. `RingRiftGui` will need a field to store the current message string. (Implemented via `self.game_message`)
    2.  **Refined Player Info Panels**: Improve formatting and information density of player stats. (Basic implementation exists)
    3.  **Additional Controls**: Consider buttons for "End Chain Capture" (if not handled by Skip/auto-detection) and "New Game" / "Load Game" / "Save Game" (more advanced, may require `iced::Application::Flags`). (New Game exists via Setup. Load/Save missing.)
    4.  **Aesthetic Polish**: Review colors, fonts, spacing for better visual appeal.
*   **Resources**: `iced` widget documentation.
*   **Timeline**: 2-4 hours.
*   **Success Indicators**:
    *   GUI provides clear feedback to the user about game state and action results.
    *   UI is intuitive and visually appealing.

This roadmap prioritizes core gameplay functionality in the GUI. Further enhancements like AI integration for GUI, advanced settings, and more sophisticated visual effects can be planned subsequently.