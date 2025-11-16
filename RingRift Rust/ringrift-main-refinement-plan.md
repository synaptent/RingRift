# Plan for Refining `ringrift/src/main.rs`

This document outlines the plan for refining the `ringrift/src/main.rs` file, with a primary focus on ensuring robust save, load, and replay functionalities, addressing potential bugs, and improving the clarity of game state management.

**Overall Goal:**
To refine and ensure the correctness of `ringrift/src/main.rs`, focusing on robust save, load, and replay functionalities, addressing potential bugs, and improving the clarity of game state management throughout these processes.

**I. High-Level Control Flow in `main()`**

The `main()` function should primarily decide whether to start a new game, load an existing game, or replay a game based on command-line arguments.

```mermaid
graph TD
    A[Start main()] --> CheckArgs[Parse Command-Line Arguments];
    CheckArgs --> Replay{Args: --replay-file?};
    Replay -- Yes --> ReplayLogic[Execute Game Replay from File];
    ReplayLogic --> ValidateReplay{Args: --validate?};
    ValidateReplay -- Yes --> CompareStates[Compare Replayed State vs. Saved Final State];
    CompareStates --> PrintValidation[Print Validation Result];
    PrintValidation --> End[Exit Program];
    ValidateReplay -- No --> End;
    Replay -- No --> Load{Args: --load-file?};
    Load -- Yes --> LoadLogic[Load Game State from File];
    LoadLogic --> InitLoadedGame[Initialize Game from Loaded State];
    InitLoadedGame --> MainLoop[Start Main Game Loop];
    Load -- No --> NewGameLogic[Initialize New Game (Interactive or CLI Args)];
    NewGameLogic --> MainLoop;
    MainLoop --> GameOver{Game Over?};
    GameOver -- Yes --> PrintResults[Print Final Scores/Winner];
    PrintResults --> End;
    GameOver -- No --> ProcessTurn[Process Player Turn / Handle Commands];
    ProcessTurn --> CheckSave{Command: save?};
    CheckSave -- Yes --> SaveGame[Save Current Game State to File];
    SaveGame --> MainLoop;
    CheckSave -- No --> MainLoop;
```

**II. Key Areas for Review and Refinement:**

1.  **Board Topology Initialization (`board.init_topology()`):**
    *   **Issue:** The non-serializable `topology` field within `Board` needs to be re-initialized after a `GameState` (which contains a `Board`) is deserialized or created.
    *   **Plan:**
        *   Ensure `game_state.board.init_topology();` is called immediately after `game_state` is populated from a loaded file (`SavedGame::load_from_file`).
        *   Ensure `game_state.board.init_topology();` is called after a new `GameState` is created for the purpose of replaying a game.
        *   When validating a replayed game against the `final_game_state` from a `SavedGame`, ensure `loaded_game.final_game_state.board.init_topology();` is called before the comparison.

2.  **Score Updates During Replay's Post-Movement Processing (PMP):**
    *   **Issue:** For an accurate replay, all state changes, including score updates from PMP, must be replicated.
    *   **Plan:**
        *   In the replay loop, when the PMP `processor.process_step(&mut game_state)` returns `ProcessStepResult::Finished(result)`, use `result.calculate_elimination_points()` to update `game_state.eliminated_rings` (and any other relevant score fields in `game_state.players` if they are directly used for win conditions). This should mirror the score update logic present in the main game loop's PMP handling section.

3.  **Phase Advancement Logic in Replay:**
    *   **Issue:** Accurately advancing the game phase, current player, and turn count during replay is critical and complex.
    *   **Plan:**
        *   Meticulously review the phase advancement logic within the replay loop. Ensure it correctly mirrors the phase transitions that occur in the main game loop for each type of replayed `Action`.
        *   Pay close attention to how `game_state.advance_phase()`, manual phase assignments (e.g., `game_state.current_phase = TurnPhase::Capture;`), and player/turn increments are handled after an action and its associated PMP.
        *   Confirm that `Action::CompleteChainCapture { .. }` is used correctly in `matches!` macros.

4.  **`save` Command Implementation:**
    *   **Issue:** Ensure the `save` command correctly captures all necessary information.
    *   **Plan:**
        *   Verify that `ai_indices` for `SavedGame::new(...)` are derived from the *current* `game_state.players.iter().filter(|p| p.is_ai).map(|p| p.id).collect()`.
        *   Confirm that the `initial_seed` passed to `SavedGame::new(...)` is the one the game was originally started or loaded with.

5.  **Stalemate Resolution and Score Adjustments:**
    *   **Issue:** The main loop's game over section adjusts scores for stalemates (`score.eliminated += player.rings_in_hand;`).
    *   **Plan:**
        *   Briefly review this logic to ensure it's consistent with the game rules and how `GameState::get_winner()` uses these adjusted scores for tie-breaking.

6.  **Error Handling and Logging:**
    *   **Issue:** Robust error handling and informative logging improve debuggability.
    *   **Plan (Recommendation):**
        *   Enhance logging (`info!`, `debug!`) around file operations (load, save), key decisions in the replay logic (e.g., action being replayed, PMP results in replay), and any unexpected conditions.
        *   Ensure user-facing error messages for invalid commands or file issues are clear.