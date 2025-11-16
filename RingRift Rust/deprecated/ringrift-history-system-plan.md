# RingRift High-Fidelity Game History System Plan

## 1. Overarching Goal

Create a system for storing, saving, replaying, and reproducing RingRift game states with maximum fidelity for debugging and analysis. This system prioritizes capturing the complete game state after **every atomic action** over resource efficiency (memory, disk, CPU).

## 2. Data Structures & Storage Strategy

### 2.1. Primary History Storage
-   A `Vec<ActionStateRecord>` will be stored, likely as a field within the main `GameState` struct (e.g., `history_records: Vec<ActionStateRecord>`). This vector holds the chronological sequence of states resulting from atomic actions.

### 2.2. `ActionStateRecord` Struct
-   This struct captures the state *after* an action and the action that *led* to it.
    ```rust
    // Likely in ringrift/src/game/history.rs or a new history_record.rs
    use serde::{Serialize, Deserialize};
    use crate::game::state::GameState; // Assuming GameState is serializable
    use crate::game::action::Action;
    use crate::models::marker::PlayerColor;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ActionStateRecord {
        /// Turn number when this action occurred.
        pub turn_number: usize,
        /// Player ID who performed the action.
        pub player_id: usize,
        /// Player color.
        pub player_color: PlayerColor,
        /// The specific atomic action that was just completed.
        pub action_taken: Action,
        /// A full snapshot of the GameState *after* action_taken was applied.
        /// GameState must derive Clone, Serialize, Deserialize.
        pub resulting_state: GameState,
        // Optional: Timestamp, RNG state snapshot if needed mid-action.
    }
    ```

### 2.3. Action Log
-   The `action_taken` field within each `ActionStateRecord` serves as the detailed, chronological action log.

### 2.4. Keyframes
-   Explicit keyframe storage is **redundant** for state restoration with this per-action snapshot approach. Navigation in a UI might benefit from marking the start-of-turn records, but this can be derived or added later if needed.

### 2.5. Initial State & Seed
-   The `GameState` at Turn 0 (before any actions) must be reconstructible or stored separately (e.g., the state before the first record in `history_records`).
-   The initial RNG seed used for the game must be stored, ideally as a field within `GameState` itself (e.g., `initial_rng_seed: u64`).

## 3. Core Logic & Algorithms

### 3.1. Defining Atomic Actions
-   An "atomic action" is the smallest unit of game progression after which the state must be captured. This includes the completion of:
    -   `PlaceRing`
    -   `MoveStack` (after marker processing)
    -   *Each segment* of a `Capture` / `ChainCaptureStep` (after landing)
    -   `StartChainCapture` (only if it immediately modifies state)
    -   `CompleteChainCapture`
    -   Processing *each individual* `FormLine` (after collapse & elimination)
    -   Processing *each individual* `DisconnectRegion` (after collapse & eliminations)
    -   `EliminateRing` (if a distinct action)
    -   `ForceEliminateCap`
    -   `Forfeit`

### 3.2. Snapshot Trigger Mechanism
-   Core game logic functions within `GameState` that apply these atomic actions must be modified.
-   **Pattern:** After successfully applying the state changes for an atomic action, call a helper function `record_action_and_state(action)`.
    ```rust
    // Inside impl GameState
    fn record_action_and_state(&mut self, action: Action) -> Result<(), String> {
        // Ensure GameState derives Clone
        let state_snapshot = self.clone(); // High cost operation
        let record = ActionStateRecord {
            turn_number: self.turn_count,
            player_id: self.current_player_index,
            player_color: self.current_color(),
            action_taken: action,
            resulting_state: state_snapshot,
        };
        self.history_records.push(record); // Assuming field exists
        Ok(())
    }
    ```
-   This requires careful refactoring to insert the `record_action_and_state` call at the correct points *after* each atomic state change.

### 3.3. Ensuring Determinism
-   **RNG:**
    -   Store `initial_rng_seed` in `GameState`.
    -   Initialize game RNG from this seed.
    -   **AI:** Store the AI's *chosen `PlayerChoice`* in the action log. Replay uses the stored choice.
    -   **Game Mechanics:** Avoid RNG in core rules. If unavoidable, store outcomes or ensure predictable re-derivation from the seed.
-   **Hash Map Iteration:** Fix any logic sensitive to `HashMap` iteration order (e.g., by sorting keys before iterating if order matters).
-   **Floating Point/Concurrency/External State:** Avoid reliance on these for core logic.

## 4. Proposed API Design (Conceptual)

```rust
// In ringrift/src/game/state.rs (or managed externally)

use crate::game::history::ActionStateRecord; // Assuming definition exists

// Add to GameState struct:
// pub history_records: Vec<ActionStateRecord> = Vec::new();
// pub initial_rng_seed: u64 = 0; // Or initialize properly

impl GameState {
    // --- Core Logic (Internal) ---

    /// Internal helper called after every atomic action application.
    fn record_action_and_state(&mut self, action: Action) -> Result<(), String>; // Implemented as above

    /// Internal function to apply actions during replay/restore. MUST BE DETERMINISTIC.
    fn apply_historical_action(&mut self, action: &Action) -> Result<(), String>; // Placeholder for complex logic

    // --- Public API ---

    /// Saves the initial seed and the entire history (vector of records) to a file.
    pub fn save_history(&self, file_path: &str) -> Result<(), String>; // Implemented using serialization

    /// Loads a history file, returning the initial seed and the records.
    pub fn load_history(file_path: &str) -> Result<(u64, Vec<ActionStateRecord>), String>; // Implemented using deserialization

    /// Retrieves the exact game state snapshot from *after* a specific action index. Instant reproduction.
    pub fn get_state_at_action_index(history: &[ActionStateRecord], index: usize) -> Option<GameState> {
         history.get(index).map(|record| record.resulting_state.clone())
    }

    /// Gets the state *before* a specific action index (state from previous record).
    pub fn get_state_before_action_index(history: &[ActionStateRecord], index: usize) -> Option<GameState> {
        // Needs handling for index 0 (requires initial game state)
        if index == 0 { None } else { history.get(index - 1).map(|record| record.resulting_state.clone()) }
    }

    /// Visually replays the game by iterating through stored states.
    /// Calls a callback with the full record (action + resulting state).
    pub fn replay_history_visual<F>(history: &[ActionStateRecord], mut on_step: F)
    where
        F: FnMut(&ActionStateRecord),
    {
        for record in history.iter() {
            on_step(record);
            // Optional: Add delays/prompts for visualization.
        }
    }
}
```

## 5. Integration Steps

1.  Add `history_records: Vec<ActionStateRecord>` and `initial_rng_seed: u64` fields to `GameState`.
2.  Implement the `record_action_and_state` helper function in `GameState`.
3.  Refactor all functions in `GameState` (and potentially related modules like `rules::capture`, `rules::post_movement_processor`) that execute atomic actions to call `record_action_and_state` upon successful completion.
4.  Implement the public API functions (`save_history`, `load_history`, `get_state_at_action_index`, `replay_history_visual`).
5.  Ensure the `apply_historical_action` function (used internally by potential future replay-from-log features, though less critical with per-action snapshots) is deterministic.
6.  Integrate calls to the public API from the UI or main application logic.

## 6. Acknowledged Trade-offs

-   **High Memory Usage:** Storing full `GameState` clones frequently will consume significant RAM, especially for long games.
-   **Large Save Files:** Serialized history files will be large.
-   **Runtime Performance:** Frequent cloning of `GameState` can impact performance during gameplay.
-   **Benefit:** Provides the highest possible fidelity for debugging and instant state restoration at any point in the game's action history.