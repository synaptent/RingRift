# Project Plan: RingRift GUI AI & Rules Completion

**Overall Objective:**
1.  Resolve the critical bug in the RingRift GUI where AI players in AI-only matches fail to execute actions.
2.  Refactor the AI system integration within the GUI to ensure AI turn processing and action generation precisely mirrors the established functionality of the CLI version (by having AI generate command strings, which are then parsed by `GameState` or GUI).
3.  Implement all remaining game rules as defined in [`ringrift_complete_rules.md`](ringrift_complete_rules.md), applying these rules consistently across both GUI and CLI versions.

---

### Part 1: AI System Refactoring and GUI Bug Fix

**Goal:** Ensure GUI AI players utilize the same underlying AI logic (e.g., `RandomAI::get_ai_command`) as CLI AI players, and that their actions are correctly processed, fixing the AI-only match bug.

**Proposed AI Flow (Consistent with CLI):**

```mermaid
graph TD
    subgraph GUI [RingRift GUI]
        direction LR
        GUI_Trigger[trigger_ai_turn()] --> GUI_GS_Call{Call gs.get_parsed_ai_player_choice(rng)}
        GUI_GS_Call --> GUI_PlayerChoice[Receives PlayerChoice]
        GUI_PlayerChoice --> GUI_Process[Processes PlayerChoice directly]
    end

    subgraph GameState [GameState]
        direction LR
        GS_GetParsedAI[get_parsed_ai_player_choice(rng)] --> GS_Call_AI_Module{Call AI_Module.get_ai_command(self, rng)}
        GS_Call_AI_Module --> GS_ReceiveString[Receives Command String]
        GS_ReceiveString --> GS_ParseCommand{Parse Command String to PlayerChoice}
        GS_ParseCommand --> GS_ReturnChoice[Returns PlayerChoice]
    end

    subgraph AI_Module [AI Module (e.g., RandomAI)]
        AI_GetCommand[get_ai_command(gs, rng)] --> AI_String[Returns Command String]
    end

    GUI_GS_Call --> GS_GetParsedAI
    GS_Call_AI_Module --> AI_GetCommand
```

**Step 1.1: Modify `GameState` to Integrate AI Command String Generation and Parsing**
*   **File:** [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs)
    *   **Create `GameState::get_parsed_ai_player_choice(&self, rng: &mut impl rand::RngCore) -> PlayerChoice`**:
        *   Replaces current `get_ai_action()` logic.
        *   Calls `crate::ai::random::RandomAI::get_ai_command(self, rng)`.
        *   Calls a new internal parsing function (see below) to convert the command string to `PlayerChoice`.
        *   Returns `PlayerChoice` or `PlayerChoice::Skip` on error.
    *   **Implement `fn parse_command_string_to_player_choice(&self, command_str: &str) -> Result<PlayerChoice, String>`**:
        *   Replicates CLI command parsing logic from [`ringrift/src/main.rs`](ringrift/src/main.rs) (for `place`, `move`, `capture`, `skip`, `force_eliminate`, `end`).
        *   Uses `self.get_valid_actions()` for context if needed (e.g., chain capture indexing).

**Step 1.2: Update `RingRiftGui` to Use the Refactored `GameState` AI Action Retrieval**
*   **File:** [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs)
    *   **Modify `RingRiftGui::trigger_ai_turn()`**:
        *   Change call from `gs.get_ai_action()` to `gs.get_parsed_ai_player_choice(&mut self.seeded_rng)`.
    *   **Ensure RNG Initialization**:
        *   Initialize `self.seeded_rng` in `Message::StartGame` handler.

**Step 1.3: Testing and Validation**
*   Test AI-only matches and Human vs. AI matches in GUI.
*   Compare AI behavior with CLI using fixed seeds.
*   Check logs for errors.

---

### Part 2: Implementation of All Remaining Game Rules

**Goal:** Achieve full rules parity as defined in [`ringrift_complete_rules.md`](ringrift_complete_rules.md).

**Step 2.1: Comprehensive Rules Audit**
*   Create a checklist mapping rules in [`ringrift_complete_rules.md`](ringrift_complete_rules.md) to:
    *   Relevant `GameState` methods or rule processors (in [`ringrift/src/rules/`](ringrift/src/rules/)).
    *   Coverage in [`ringrift_gui_full_plan.md`](ringrift_gui_full_plan.md).
    *   Current implementation status.
*   Focus on: Turn phases, placement (incl. move prerequisite), movement (distance, markers, landing), captures (types, cap height, chain, landing), line formation (length, graduated rewards, multiple lines), territory disconnection (connectivity, color representation, self-elimination prerequisite, chain reactions), forced elimination, victory conditions, version differences.

**Step 2.2: Prioritize and Plan Implementation for Gaps**
*   Based on the audit, prioritize implementing missing/incomplete rules.
*   For each gap, define changes to `GameState`/rule processors, new `PlayerChoice` options, and GUI modifications.

**Step 2.3: Implement Rule Logic in `GameState` and Rule Processors**
*   **Files:** [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs), [`ringrift/src/rules/`](ringrift/src/rules/), [`ringrift/src/game/action.rs`](ringrift/src/game/action.rs).
*   Modify/create functions/structs for rule gaps. Examples:
    *   Placement: "subsequent move prerequisite" check (Rule 4.1).
    *   Line Formation: "Graduated Line Rewards" choice (Rule 11.2).
    *   Territory Disconnection: "self-elimination prerequisite check" (Rule 12.2).

**Step 2.4: Implement GUI Support for New/Modified Rules**
*   **File:** [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs)
*   Update `RingRiftGui::update` for new `PlayerChoice` options/game states.
*   Modify `RingRiftGui::view` and `BoardCanvas::draw` for new UI elements/feedback.

**Step 2.5: Testing and Validation**
*   Create specific test scenarios for each rule.
*   Test in CLI and GUI, using fixed seeds.
*   Verify game state, rule outcomes, and UI feedback.

---

### Part 3: Overall Project Considerations
*   **Code Style and Documentation:** Maintain consistency, add comments, update relevant docs.
*   **Testing Strategy:** Utilize/add unit tests; focus manual testing on GUI and CLI consistency.
*   **Incremental Commits:** Clear, focused commits.