# Technical Plan: Iterative Territory Processing & Intermediate State Display

**1. Goal:**

*   Modify the Ringrift game logic to correctly handle sequential territory collapses within a single turn, ensuring all valid disconnections are processed iteratively.
*   Implement logging or display of intermediate board states after each individual collapse during such sequences.
*   Ensure compliance with Rules 4.4 and 12.3 regarding post-movement processing order and chain reactions.

**2. Target Files/Modules:**

Based on the project structure and the nature of the required changes, the primary files to modify are likely:

*   `ringrift/src/rules/post_movement_processor.rs`: Contains the main logic orchestrating checks after movement (lines, territories).
*   `ringrift/src/rules/territory.rs`: Contains functions for finding and processing disconnected regions.
*   `ringrift/src/game/state.rs`: May need adjustments for state representation or logging hooks.
*   `ringrift/src/game/history.rs`: Likely place to add the intermediate state logging method.
*   `ringrift/src/debug_utils.rs` or similar: If a dedicated logging/display function exists for board state.

**3. Algorithm & Control Flow Modifications:**

The core change involves transforming the territory processing part of the `post_movement_processor` from a single check into an iterative loop.

**3.1. Modify `process_post_movement` (in `post_movement_processor.rs`):**

```rust
// Placeholder for the function signature, adjust as needed
pub fn process_post_movement(game_state: &mut GameState, player_id: PlayerId, history: &mut GameHistory) -> Result<(), GameError> {

    // --- Step 1: Process Lines (Existing Logic) ---
    // Ensure this part correctly handles multiple lines if formed simultaneously,
    // processing them one by one as per Rule 11.3.
    // Let's assume a function `process_all_lines` exists and handles this.
    process_all_lines(game_state, player_id, history)?;
    // Log state after line processing if desired (optional, but good for debugging)
    // history.log_intermediate_state("After Line Processing");


    // --- Step 2: Iteratively Process Disconnected Regions ---
    let mut iteration_count = 0; // Safety counter / for logging
    loop {
        iteration_count += 1;
        log::debug!("Territory Check Iteration {}", iteration_count);

        let mut collapse_occurred_in_iteration = false;

        // Find *all* currently disconnected regions eligible for processing by player_id.
        // This function MUST incorporate the self-elimination prerequisite check (Rule 12.2).
        let eligible_regions = territory::find_eligible_disconnected_regions(game_state, player_id);

        if eligible_regions.is_empty() {
            log::debug!("No more eligible disconnected regions found. Exiting loop.");
            break; // Exit the loop if no processable regions are found
        }

        // --- Player Choice (Rule 12.3) ---
        // If multiple regions are found, the player must choose which one to process first.
        // This needs integration with the AI/Player input logic.
        // For now, let's assume a function `select_region_to_process` handles this choice.
        // This function might live in GameState or be called via a trait implemented by Player/AI.
        let chosen_region_index = game_state.select_region_to_process(player_id, &eligible_regions)?;
        let chosen_region = eligible_regions.into_iter().nth(chosen_region_index).ok_or_else(|| GameError::InvalidInput("Chosen region index out of bounds".to_string()))?;
        // ---

        // Process the *single* chosen region
        match territory::process_single_disconnected_region(game_state, player_id, chosen_region, history) {
            Ok(processed) => {
                if processed {
                    collapse_occurred_in_iteration = true;
                    log::info!("Successfully processed disconnected region. Iteration {}.", iteration_count);

                    // --- Intermediate State Output ---
                    // Log/display the board state AFTER this single collapse.
                    history.log_intermediate_state(game_state, &format!("After Territory Collapse Iteration {}", iteration_count));
                    // Optionally, add more details about the collapsed region to the log message.
                    // ---

                } else {
                    // This might happen if the prerequisite check failed between finding and processing,
                    // though `find_eligible_disconnected_regions` should prevent this. Log and break.
                    log::warn!("Region processing returned false unexpectedly (prerequisite likely failed). Breaking loop.");
                    break;
                }
            }
            Err(e) => {
                log::error!("Error processing disconnected region: {}. Aborting turn processing.", e);
                return Err(e); // Propagate the error
            }
        }

        // Safety break / sanity check - should be handled by finding no regions
        if !collapse_occurred_in_iteration {
             log::warn!("Territory loop iteration {} completed without processing a collapse. Breaking.", iteration_count);
             break;
        }

        // Add a safety break for excessive iterations
        if iteration_count > game_state.board.size() { // Or some other reasonable limit
            log::error!("Exceeded maximum territory processing iterations ({}). Aborting turn.", iteration_count);
            return Err(GameError::InternalError("Exceeded maximum territory processing iterations".to_string()));
        }

        // Loop continues: The state has changed, so we re-evaluate `find_eligible_disconnected_regions`
    } // End of territory processing loop


    // --- Step 3: Victory Check (Existing Logic) ---
    // This runs only *after* all lines and *all* iterative territory collapses are complete.
    game_state.check_victory_conditions()?;

    Ok(())
}
```

**3.2. Modify `territory.rs`:**

*   **`find_eligible_disconnected_regions` function:**
    *   This function needs to perform the full check as described in Rule 12.2 and Q15:
        1.  Identify all physically disconnected regions (using Von Neumann adjacency). This likely involves a graph traversal (like BFS or DFS) starting from unvisited empty/marker spaces, stopping at rings/stacks, collapsed spaces, or the board edge. Keep track of the border composition.
        2.  For each physically distinct region, check if its border is valid (collapsed spaces, board edge, or markers of *only one* player color).
        3.  For valid bordered regions, check the Color Representation rule (does it lack representation from at least one active player?).
        4.  For regions satisfying (1-3), perform the *hypothetical* Self-Elimination Prerequisite Check: Simulate removing internal rings and see if the `player_id` would still have a ring/cap elsewhere.
    *   Return a `Vec<RegionInfo>` (or similar struct containing region squares, border info) for regions that satisfy all conditions.
*   **`process_single_disconnected_region` function:**
    *   Takes `game_state: &mut GameState`, `player_id: PlayerId`, the specific `chosen_region: RegionInfo`, and `history: &mut GameHistory` as arguments.
    *   Performs the actual state changes:
        *   Collapse region spaces and the single-color border markers to `player_id`'s color in `game_state.board`.
        *   Identify and remove internal rings from `game_state.board` and update player scores/eliminated counts in `game_state.players`. Add details to `history`.
        *   Perform the mandatory self-elimination for `player_id` (removing a ring/cap from `game_state.board` and updating `game_state.players`). Add details to `history`.
    *   Returns `Ok(true)` on successful processing, `Ok(false)` if it couldn't process (e.g., prerequisite failed on final check - though ideally caught earlier), or `Err` on error.

**4. Intermediate State Output Implementation:**

*   **Mechanism:** Add a method to `GameHistory` (e.g., in `ringrift/src/game/history.rs`).
    ```rust
    // In history.rs
    impl GameHistory {
        pub fn log_intermediate_state(&mut self, game_state: &GameState, context: &str) {
            // Create a snapshot of the relevant game state (board, player scores)
            let state_snapshot = game_state.create_snapshot(); // Assuming such a method exists or can be created
            // Format the snapshot for logging/display
            let formatted_state = format_board_state_for_log(&state_snapshot); // Use or adapt existing logging utils

            // Log using the standard logging facade (e.g., `log::info!`) or add to a history vector
            log::info!("--- Intermediate State: {} ---\n{}", context, formatted_state);

            // Optionally, store this intermediate state in the history struct if needed for replay/analysis
            // self.intermediate_states.push((context.to_string(), state_snapshot));
        }
    }
    ```
*   **Trigger Point:** Inside the `loop` in `process_post_movement`, immediately after `process_single_disconnected_region` returns `Ok(true)`.
*   **Output:** The `log::info!` macro will output to the configured logger. The format should mimic the end-of-turn state display for consistency, clearly marked with the context string (e.g., "After Territory Collapse Iteration 1").

**5. Testing Procedures:**

*   **Unit Tests:**
    *   Test `find_eligible_disconnected_regions` extensively:
        *   Simple disconnections.
        *   Regions bordered by mixed collapsed/marker types.
        *   Regions failing color representation.
        *   Regions failing the self-elimination prerequisite.
        *   Multiple regions found simultaneously.
    *   Test `process_single_disconnected_region`:
        *   Correct space collapsing (region + border).
        *   Correct internal ring elimination counts.
        *   Correct self-elimination (single ring vs. cap).
        *   State updates (board, player scores).
    *   Test the loop logic in `process_post_movement` using mock `territory` functions.
*   **Integration Tests (`ringrift/tests/`):**
    *   **Turn 6 Scenario:** Replicate the board state before Red's Turn 6 capture. Run the turn and assert:
        *   First region ({g1, h1}) collapses.
        *   Intermediate state is logged.
        *   Second region ({f2}) collapses.
        *   Final state is correct.
        *   Eliminated counts are correct.
    *   **Multiple Simultaneous Regions:** Test player choice (if implemented) and iterative processing.
    *   **Cascading Collapses:** Verify the loop detects and processes secondary disconnections caused by primary ones.
    *   **Self-Elimination Prerequisite Failure:** Test that a region isn't processed if the prerequisite fails.
    *   **Interaction with Lines:** Ensure lines are processed fully *before* the territory loop starts.
*   **Log Verification:** Manually review logs for all integration tests to confirm:
    *   Intermediate states appear correctly and only when needed.
    *   States accurately reflect the board post-collapse.
    *   Log messages are clear.

**6. Mermaid Diagram (Updated Flow):**

```mermaid
flowchart TD
    Start([Start Post-Movement]) --> ProcessLines{1. Process All Lines}
    ProcessLines --> LoopStart{Start Territory Loop}
    LoopStart --> FindRegions{2a. Find Eligible Disconnected Regions (incl. Prereq Check)}
    FindRegions --> AnyRegions{Any Found?}
    AnyRegions -->|No| VictoryCheck{3. Check Victory Conditions}
    AnyRegions -->|Yes| ChooseRegion{2b. Player Chooses Region}
    ChooseRegion --> ProcessRegion{2c. Process Single Region (Collapse, Elim Internal, Self-Elim)}
    ProcessRegion --> Success{Processed OK?}
    Success -->|Yes| LogState{2d. Log Intermediate State}
    LogState --> LoopStart
    Success -->|No / Error| HandleError{Handle Error / Break Loop}
    HandleError --> VictoryCheck
    VictoryCheck --> EndTurn([End Turn Processing])

    style LoopStart fill:#f9f,stroke:#333,stroke-width:2px
    style LogState fill:#ccf,stroke:#333,stroke-width:2px