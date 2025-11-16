# RingRift Hexagonal Board: Bug Fix & GUI Implementation Plan

This document outlines the technical plan to diagnose and resolve bugs related to the hexagonal board option in the RingRift game, implement CLI rendering for the hexagonal board, and design and implement a Graphical User Interface (GUI) for playing on a hexagonal board.

## Phase 1: Diagnose and Resolve Hexagonal Board Bugs (CLI)

This phase focused on correcting input validation and coordinate system management for hexagonal grids according to the game rules specified in `ringrift_complete_rules.md`.

**Status: Completed**

### Task 1.1: Refactor Coordinate Parsing in `main.rs`
*   **Objective**: Modify input parsing functions (`parse_coordinates`, `parse_move_coordinates`, `parse_capture_initiation_coordinates`) in [`ringrift/src/main.rs`](ringrift/src/main.rs) to correctly handle hexagonal board coordinates (axial `q, r`) alongside existing square board formats.
*   **Implementation**:
    *   `parse_coordinates` was updated to accept `BoardType` and return `Result<Position, String>`.
    *   Logic added to parse two numeric tokens as `q` and `r` for `BoardType::Hexagonal`, validating them against board boundaries (derived from `board_type.default_size()`, which is 11 for hex, meaning `n=10`, so `q,r,s` are in `[-10, 10]`).
    *   `Position::hex_axial(q, r)` is used to create the `Position`.
    *   Square board parsing logic was maintained, using `Position::square(row, col)`.
    *   `parse_move_coordinates` and `parse_capture_initiation_coordinates` were updated to use the new `parse_coordinates` and handle 4 numeric tokens for hex moves/captures (`q1 r1 q2 r2`).
*   **Files Modified**: [`ringrift/src/main.rs`](ringrift/src/main.rs)

### Task 1.2: Standardize Hex Coordinate Display
*   **Objective**: Ensure consistent display of hexagonal coordinates, aligning with the `q r` input format.
*   **Implementation**:
    *   The `Display` implementation for `Position` in [`ringrift/src/models/topology.rs`](ringrift/src/models/topology.rs) was updated. For `Position::Hex`, it now formats as `H(q,r)` instead of `H(q,r,s)`.
*   **Files Modified**: [`ringrift/src/models/topology.rs`](ringrift/src/models/topology.rs)

### Task 1.3: Review and Update AI Move Generation/Validation
*   **Objective**: Ensure AI-generated moves for hexagonal boards are compatible with the new parsing logic.
*   **Implementation**:
    *   Reviewed `RandomAI` in [`ringrift/src/ai/random.rs`](ringrift/src/ai/random.rs).
    *   The `format_action_as_command` function uses `super::format_position` (located in [`ringrift/src/ai/mod.rs`](ringrift/src/ai/mod.rs)).
    *   `ai::format_position` was verified to correctly format `Position::Hex` as two space-separated numbers (`q r`), which aligns with the updated parsing in `main.rs`.
*   **Status**: No code changes required.

### Task 1.4: Verify Adjacency Mode for Hexagonal Territory
*   **Objective**: Confirm that 6-way adjacency is used for territory calculations on hexagonal boards, as per game rules.
*   **Implementation**:
    *   [`BoardType::default_adjacency_mode()`](ringrift/src/models/topology.rs) correctly returns `AdjacencyMode::VonNeumann` for `BoardType::Hexagonal`.
    *   Neighbor finding logic in `BoardTopology` and `HexTopology` (in [`ringrift/src/models/topology.rs`](ringrift/src/models/topology.rs)) correctly defers to `HexTopology::get_neighbors()`, which implements 6-way adjacency irrespective of the `AdjacencyMode` enum value (as hex grids inherently have only one type of direct adjacency).
    *   Territory processing in [`ringrift/src/rules/territory.rs`](ringrift/src/rules/territory.rs) (e.g., `find_physically_disconnected_regions`) uses `topology.get_neighbors_with_mode(&current, AdjacencyMode::VonNeumann)`. For hex boards, this correctly resolves to 6-way adjacency.
*   **Status**: No code changes required.

## Phase 2: CLI Rendering for Hexagonal Game Board

This phase focused on creating a clear, text-based CLI rendering of the hexagonal game board.

**Status: Completed**

### Task 2.1 & 2.2: Design and Implement Hexagonal Board Display Logic
*   **Objective**: Modify the `Display` implementation for the `Board` struct in [`ringrift/src/models/board.rs`](ringrift/src/models/board.rs) to render hexagonal boards.
*   **Implementation**:
    *   The `fmt` method for `Board` was updated.
    *   For `BoardType::Hexagonal`, it now:
        *   Converts axial coordinates `(q, r)` to "odd-r" display coordinates `(display_col, display_row)`.
        *   Determines display grid bounds.
        *   Uses a `HashMap` to map display coordinates to formatted cell content (stack ID, marker, or collapsed state).
        *   Iterates through display rows and columns, printing cell content with appropriate indentation for odd rows to create a staggered hex layout.
*   **Files Modified**: [`ringrift/src/models/board.rs`](ringrift/src/models/board.rs)

### Task 2.3: Update `print_game_state`
*   **Objective**: Ensure the main game state printing function utilizes the new hexagonal board display.
*   **Implementation**:
    *   The `print_game_state` function in [`ringrift/src/debug_utils.rs`](ringrift/src/debug_utils.rs) was modified.
    *   When `game_state.board.board_type` is `Hexagonal`, it now directly prints `game_state.board` using its updated `Display` implementation.
    *   The `format_board_state_for_log` function was similarly updated.
*   **Files Modified**: [`ringrift/src/debug_utils.rs`](ringrift/src/debug_utils.rs)

## Phase 3: GUI for Hexagonal Board

This phase focuses on designing and implementing an intuitive GUI for playing RingRift on a hexagonal board.

**Status: In Progress**

### Task 3.1: Choose GUI Framework
*   **Decision**: `iced` framework.
*   **Rationale**: Chosen by the user. `iced` is a data-driven, cross-platform GUI library for Rust, inspired by Elm.

### Task 3.2: Design GUI Layout and Interaction
*   **Layout**: Standard layout:
    *   Central area for rendering the hexagonal game board.
    *   Side/bottom panels for player information (color, rings, scores, etc.).
    *   Status bar/panel for game messages, current turn/phase.
*   **Interaction**: Primarily mouse-click based for selecting hexes and triggering actions. Buttons for global actions.

### Task 3.3: Add `iced` and Graphics Dependencies
*   **Implementation**: Added `iced = { version = "0.12", features = ["canvas", "tokio"] }` to [`ringrift/Cargo.toml`](ringrift/Cargo.toml).
*   **Status**: Completed.

### Task 3.4: Create Basic `iced` Application Structure
*   **Implementation**:
    *   Created [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs).
    *   Defined `RingRiftGui` struct implementing `iced::Application`.
    *   Implemented basic `new`, `title`, `update`, `view`, and `theme` methods.
    *   Added `pub mod gui;` to [`ringrift/src/lib.rs`](ringrift/src/lib.rs).
    *   Added a `--gui` flag to `Args` in [`ringrift/src/main.rs`](ringrift/src/main.rs) to launch `gui::run_gui()`.
*   **Status**: Completed.

### Task 3.5: Integrate `GameState` into `RingRiftGui`
*   **Implementation**:
    *   `RingRiftGui` struct now holds `game_state: GameState`.
    *   `RingRiftGui::new` initializes a default hexagonal `GameState`.
    *   `RingRiftGui::view` updated to display basic info from `game_state`.
*   **Status**: Completed.

### Task 3.6: Implement `Canvas` for Hex Board Rendering
*   **Implementation**:
    *   Created `BoardCanvas<'a>` struct implementing `iced::widget::canvas::Program`.
    *   `BoardCanvas::draw` method:
        *   Iterates through `game_state.board.spaces`.
        *   Converts hex axial coordinates to pixel coordinates using `BoardCanvas::axial_to_pixel` (pointy-top hexes, centered on canvas).
        *   Draws hexagon paths.
        *   Fills hexagons based on `MarkerState` or stack presence/color.
        *   Displays stack height (`S<H>`) or marker initial (`R`, `XG`) inside hexes.
    *   `BoardCanvas::update` method:
        *   Handles mouse clicks.
        *   Converts pixel click coordinates to axial hex coordinates using `BoardCanvas::pixel_to_axial`.
        *   Sends `Message::ClickHex(q, r)` if the click is on a valid board hex.
    *   `RingRiftGui::view` integrates the `Canvas` widget, passing `game_state` and `selected_hex`.
*   **Status**: Basic rendering and click detection implemented. Coordinate conversion refinement and detailed visual feedback for game actions are pending.

### Task 3.7: Implement GUI Elements for Player Info and Game State
*   **Implementation**:
    *   Added `view_player_info` method to `RingRiftGui`.
    *   Displays player ID, color, AI status, rings in hand, eliminated count, and territory count.
    *   Highlights the current player.
    *   Integrated into the main `view` layout.
*   **Status**: Basic implementation completed.

### Task 3.8: Implement GUI Controls for Game Actions (Detailed Plan)
*   **Objective**: Allow users to perform game actions through GUI interactions.
*   **Message Enum Expansion**: Add variants for specific game actions:
    ```rust
    pub enum Message {
        // ... existing messages
        PlaceRingRequest, // Button press
        MoveStackRequest, // Button press
        CaptureRequest,   // Button press
        // EndChainCapture, // If a specific button is desired
        // ForfeitTurn, // If different from SkipTurn in some contexts
        // Potentially: SelectPiece(Position), SelectTarget(Position) if managing selection state explicitly
    }
    ```
*   **`RingRiftGui` State Expansion**:
    *   `selected_hex: Option<Position>` (already added).
    *   `action_state: ActionState` enum to manage multi-step actions:
        ```rust
        enum ActionState {
            Idle,
            PendingPlacement(Position), // Hex selected for placement
            PieceSelected(Position),    // Piece selected for move/capture
            TargetSelected(Position, Position), // Origin and target selected for move/capture
        }
        ```
    *   Initialize `action_state: ActionState::Idle`.
*   **`RingRiftGui::update` Logic**:
    *   **`Message::ClickHex(q, r)`**:
        *   Convert `(q,r)` to `clicked_pos: Position`.
        *   Update `self.selected_hex = Some(clicked_pos)`.
        *   Based on `self.game_state.current_phase` and `self.action_state`:
            *   `TurnPhase::RingPlacement`:
                *   If `action_state` is `Idle` or `PendingPlacement`:
                    *   Attempt `self.game_state.place_new_stack_on_board(clicked_pos, num_rings, player_color)`.
                        *   `num_rings` could be defaulted to 1 or taken from a UI input.
                    *   If successful, reset `action_state = ActionState::Idle`, clear `selected_hex`.
                    *   Handle errors by displaying a message.
            *   `TurnPhase::Movement` / `TurnPhase::Capture`:
                *   If `action_state` is `Idle`:
                    *   Check if `clicked_pos` contains a piece owned by the current player.
                    *   If yes, `action_state = ActionState::PieceSelected(clicked_pos)`.
                *   If `action_state` is `PieceSelected(origin_pos)`:
                    *   `action_state = ActionState::TargetSelected(origin_pos, clicked_pos)`.
                    *   Now, determine if it's a move or capture.
                        *   Check valid moves: `self.game_state.validate_move(&origin_pos, &clicked_pos)`. If valid, attempt `self.game_state.move_stack()`.
                        *   Check valid captures: `CaptureProcessor::new(&self.game_state).find_valid_landings_for_segment(origin_pos, clicked_pos)`.
                            *   If landings exist (prompt if multiple, or auto-select if one), attempt `self.game_state.execute_capture()`.
                    *   If action successful, reset `action_state = ActionState::Idle`, clear `selected_hex`.
                    *   Handle errors.
            *   Invalidate canvas cache (`board_cache.clear()`) to reflect selection changes.
    *   **Button Messages (`PlaceRingRequest`, `MoveStackRequest`, etc.)**:
        *   These might set `action_state` to prepare for hex clicks (e.g., `MoveStackRequest` sets `action_state = ActionState::Idle` but indicates the next click is for selecting a piece to move).
        *   `Message::SkipTurn`: Call `self.game_state.advance_phase(false)` (or a more robust skip/forfeit method in `GameState`).
*   **`BoardCanvas::draw` Enhancements**:
    *   Use `self.game_state` to color hexes based on actual content (stacks, markers, collapsed).
    *   Display stack height/marker type text.
    *   Highlight `self.selected_hex`.
    *   If `gui.action_state` is `PieceSelected(pos)`, highlight `pos` differently.
    *   If `gui.action_state` indicates move/capture targeting, query `GameState` for valid target hexes from the selected piece and highlight them (e.g., light green for move, light red for capture).
*   **GUI Elements (View)**:
    *   Add `iced::widget::Button` for "Place Ring", "Move", "Capture", "End Chain".
        *   Enable/disable buttons based on `game_state.current_phase` and `game_state.get_valid_actions()`.
    *   Player info panel: more detailed scores, status.
    *   Game message area: display errors from `GameState` or informational messages.
*   **Files to Modify**: [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs)

### Task 3.9: Testing and Refinement
*   **Objective**: Thoroughly test GUI interactions, hex rendering, and game logic integration.
*   **Steps**:
    *   Manual testing of all game phases and actions via the GUI.
    *   Verify visual feedback is correct.
    *   Test edge cases and error handling.
    *   Refine coordinate conversions and click detection for accuracy.

This detailed plan for Phase 3 provides a roadmap for completing the GUI implementation.