# Plan for Implementing `apply_historical_action` for Capture-Related Actions

**Overarching Goal:** Refactor capture logic in `GameState` to separate core state-mutating operations from history recording, and use these core operations within `apply_historical_action` for deterministic replay.

## I. Single Capture Segment (`Action::Capture`)

1.  **Create `core_apply_capture_segment` function:**
    *   **Signature:** `fn core_apply_capture_segment(&mut self, from: &Position, target: &Position, landing: &Position, attacker_color: PlayerColor, captured_color: PlayerColor) -> Result<CaptureRecord, CaptureError>`
        *   `attacker_color` will be derived from `self.current_player().color` during live play and taken from the `ActionStateRecord`'s `player_color` field (associated with the `Action::Capture`) during replay.
        *   `captured_color` will be derived during live play and taken directly from the `Action::Capture`'s `captured_color` field during replay.
    *   **Logic:** This function will encapsulate the state modification logic currently within `execute_capture`. This includes:
        *   Retrieving and updating attacker and defender stacks.
        *   Handling defender stack removal if it becomes empty.
        *   Moving the attacker stack (removing from `from`, adding captured ring, placing at `landing`).
        *   Processing path markers using `process_path_markers(self, ...)`.
        *   Setting `stack.has_captured = true` on the attacker stack.
        *   Updating `self.action_taken_this_turn = true;` and `self.last_move_landing_pos = None;`.
    *   **Return:** A `CaptureRecord` detailing the capture.
    *   **Constraint:** This function must *not* call `record_action_and_state`.

2.  **Modify `pub fn execute_capture`:**
    *   Derive `attacker_color` (e.g., `self.current_color()`) and `captured_color` (from the top ring of the target stack).
    *   Call `let capture_details = self.core_apply_capture_segment(from, target, landing, attacker_color, captured_color)?;`.
    *   Construct an `Action::Capture` variant:
        ```rust
        let history_action = Action::Capture {
            from: *from,
            target: *target,
            to: *landing, // 'to' in Action::Capture corresponds to 'landing'
            captured_color: capture_details.captured_color,
        };
        ```
    *   Call `self.record_action_and_state(history_action)?;`.

3.  **Implement `Action::Capture` arm in `apply_historical_action`:**
    *   `Action::Capture { from, target, to, captured_color } => { ... }`
    *   Retrieve `attacker_color` from the implicit `ActionStateRecord.player_color` associated with this historical action.
    *   Call `self.core_apply_capture_segment(*from, *target, *to, attacker_color, *captured_color)?;`. The returned `CaptureRecord` is ignored for direct state change but its data might be needed for chain state.
    *   **Chain Capture Context:**
        *   If `self.chain_capture_state.is_some()`, this `Action::Capture` is a step in an ongoing chain.
        *   The logic must update `self.chain_capture_state` similarly to how `continue_chain_capture` does after its internal call to `execute_capture_on_clone` (which will be replaced by `core_apply_capture_segment` principles). This involves:
            *   Adding the `CaptureRecord` (re-create it from the action's details or have `core_apply_capture_segment` return it even for replay if simpler) to `state.captures`.
            *   Updating `state.current_position = *to;`.
            *   Updating `state.visited_positions`.
            *   Recalculating `state.available_options` based on the new `state.current_position` and `state.visited_positions`.
            *   Setting `state.is_complete = state.available_options.is_empty();`.

## II. Chain Capture Start (`Action::StartChainCapture`)

1.  **Create `core_start_chain_capture` function:**
    *   **Signature:** `fn core_start_chain_capture(&mut self, from_pos: &Position) -> Result<Vec<Position>, CaptureError>`
    *   **Logic:** Contains the logic from the current `start_chain_capture` responsible for:
        *   Backing up `self.board_before_chain_capture = Some(self.board.clone());`
        *   Backing up `self.stacks_before_chain_capture = Some(self.stacks.clone());`
        *   Determining initial capture targets using `CaptureProcessor`.
        *   Initializing and setting `self.chain_capture_state`.
        *   Setting `self.last_move_landing_pos = None;`.
    *   **Constraint:** Does *not* call `record_action_and_state`.
    *   **Return:** `Vec<Position>` of initial available targets.

2.  **Modify `pub fn start_chain_capture`:**
    *   Call `let targets = self.core_start_chain_capture(from)?;`.
    *   Construct `let history_action = Action::StartChainCapture { position: *from };`.
    *   Call `self.record_action_and_state(history_action)?;`.
    *   Return `Ok(targets)`.

3.  **Implement `Action::StartChainCapture` arm in `apply_historical_action`:**
    *   `Action::StartChainCapture { position } => { ... }`
    *   Call `self.core_start_chain_capture(position)?;`. The returned `Vec<Position>` is ignored.

## III. Chain Capture Step (Handled by `Action::Capture` arm)

*   As noted, `Action::ChainCaptureStep` is a `PlayerChoice`. The recorded action for each step of a chain is `Action::Capture`. The replay logic for these steps is handled within the `Action::Capture` arm of `apply_historical_action` as detailed in section I.3.

## IV. Chain Capture Completion (`Action::CompleteChainCapture`)

1.  **Create `core_complete_chain_capture` function:**
    *   **Signature:** `fn core_complete_chain_capture(&mut self) -> Result<ChainCaptureResult, CaptureError>`
    *   **Logic:** Contains the logic from `end_chain_capture` responsible for:
        *   Finalizing and taking `self.chain_capture_state`.
        *   Clearing `self.board_before_chain_capture` and `self.stacks_before_chain_capture`.
        *   Setting `self.action_taken_this_turn = true;` and `self.last_move_landing_pos = None;`.
    *   **Constraint:** Does *not* call `record_action_and_state`.
    *   **Return:** `ChainCaptureResult`.

2.  **Modify `pub fn end_chain_capture`:**
    *   Call `let result = self.core_complete_chain_capture()?;`.
    *   Construct `let history_action = Action::CompleteChainCapture { rings_captured: result.total_rings_captured, final_position: result.final_position };`.
    *   Call `self.record_action_and_state(history_action)?;`.
    *   Return `Ok(result)`.

3.  **Implement `Action::CompleteChainCapture` arm in `apply_historical_action`:**
    *   `Action::CompleteChainCapture { rings_captured: _, final_position: _ } => { ... }`
    *   Call `self.core_complete_chain_capture()?;`. The returned `ChainCaptureResult` is ignored.

## Diagram (Simplified Flow for `Action::Capture`):

```mermaid
graph TD
    subgraph LiveGameplay
        A[Player chooses Capture] --> B(execute_capture)
        B --> C{core_apply_capture_segment}
        C --> D[CaptureRecord]
        B -- Constructs --> E[Action::Capture from Record]
        B --> F(record_action_and_state with E)
    end

    subgraph HistoryReplay
        G[Load Action::Capture from history_records] --> H(apply_historical_action)
        H -- Action::Capture arm --> I{core_apply_capture_segment}
        I --> J[CaptureRecord (potentially used for chain state update)]
        H -- If chain capture context --> K[Update self.chain_capture_state using J's data]
    end

    C --> L[Mutate GameState: move stacks, update board, etc.]
    I --> L