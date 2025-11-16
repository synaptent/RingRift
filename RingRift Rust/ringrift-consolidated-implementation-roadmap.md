# RingRift Consolidated Implementation Roadmap

This document consolidates all actionable insights from the project's markdown documentation into a prioritized implementation roadmap with specific, concrete next steps for immediate codebase improvements.

## Executive Summary

The RingRift project has extensive documentation covering GUI implementation, bug fixes, code refactoring, and feature enhancements. This roadmap organizes all tasks into priority tiers based on impact and dependencies.

## Priority 1: Critical Fixes (Immediate Action Required)

### 1.1 Compiler Errors and Warnings
**Source**: `ringrift-compiler-fix-plan-v2.md`
**Impact**: Code won't compile without these fixes

1. **Fix import errors in [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs)**:
   - Remove line 3: `graphics::FillRule`
   - Ensure `FillRule` is in the `canvas` import block (line 9)
   - Change line 17 from `Viewport` to `widget::scrollable::Viewport`

2. **Remove unused imports in [`ringrift/src/gui/mod.rs`](ringrift/src/gui/mod.rs)**:
   - Remove: `Container`, `Size`, `button`, `event`, `pick_list`, `radio`, `text_input`
   - Remove: `self` from canvas imports
   - Remove: `HashMap`, `Duration`, `Instant`, `SavedGame`, `Player`, `Ring`, `BoardTopology`
   - Keep: `ChainCaptureResult` and implement usage in chain capture feedback

3. **Fix unused assignments in [`ringrift/src/game/state.rs`](ringrift/src/game/state.rs)**:
   - Refactor `collapse_all_for_core` and `elimination_details_for_core` initialization (lines 2165-2166)
   - Use match expression to return tuple directly

### 1.2 GUI Rendering Defect
**Source**: `ringrift-gui-diagnostic-plan.md`
**Impact**: GUI shows blank board after clicking "Start Game"

1. **Add diagnostic logging**:
   - Add error-level logs in `RingRiftGui::new()`, `update()`, `view()`
   - Add logs in `BoardCanvas::draw()` to verify execution
   - Log `GameState` initialization in `Message::StartGamePressed`

2. **Verify data flow**:
   - Check if `GameState` properly initializes board spaces
   - Verify `BoardCanvas` receives valid game state
   - Ensure drawing parameters (scale, offset) are calculated correctly

### 1.3 AI System Integration Bug
**Source**: `ringrift-gui-ai-rules-plan.md`
**Impact**: AI players fail to execute actions in GUI

1. **Refactor AI integration**:
   - Create `GameState::get_parsed_ai_player_choice()` method
   - Implement `parse_command_string_to_player_choice()` in `GameState`
   - Update `RingRiftGui::trigger_ai_turn()` to use new method
   - Ensure RNG initialization in GUI

## Priority 2: Core Functionality Gaps

### 2.1 History System Implementation
**Source**: `ringrift-history-refactor-plan.md`, `ringrift-apply-historical-action-capture-plan.md`
**Impact**: Cannot replay games or implement undo/redo

1. **Refactor capture logic**:
   - Create `core_apply_capture_segment()` function
   - Create `core_start_chain_capture()` function
   - Create `core_complete_chain_capture()` function
   - Separate state mutation from history recording

2. **Implement `apply_historical_action()`**:
   - Add support for all `Action` variants
   - Handle chain capture context properly
   - Ensure deterministic replay

### 2.2 Territory Disconnection Bug
**Source**: `ringrift-a8-territory-debug-plan.md`, `territory-disconnection-algorithm-analysis.md`
**Impact**: Territory disconnection not processing correctly after first claim

1. **Fix re-check logic**:
   - Ensure territory re-evaluation occurs after each collapse
   - Fix boundary interpretation for single-square regions
   - Correct self-elimination prerequisite check
   - Handle chain reactions properly

2. **Update algorithm**:
   - Implement proper Von Neumann adjacency for territory checks
   - Fix color representation evaluation
   - Ensure all disconnected regions are processed sequentially

### 2.3 Missing GUI Features
**Source**: `gui_detailed_plan.md`, `ringrift_gui_full_plan.md`
**Impact**: GUI lacks feature parity with CLI

1. **Implement forced cap elimination UI**:
   - Create modal for stack selection
   - Handle `execute_force_eliminate_cap()` calls
   - Add visual feedback

2. **Implement graduated line reward choice**:
   - Create modal for Option 1 vs Option 2 selection
   - Allow segment selection for partial collapse
   - Update board visualization

3. **Implement save/load functionality**:
   - Add file dialogs for save/load
   - Integrate with `SavedGame` struct
   - Handle file I/O errors gracefully

## Priority 3: Code Quality and Architecture

### 3.1 Main Function Refinement
**Source**: `ringrift-main-refinement-plan.md`
**Impact**: Improves maintainability and testability

1. **Refactor `main.rs`**:
   - Extract command parsing logic
   - Create dedicated game loop handler
   - Separate concerns into modules
   - Improve error handling

### 3.2 Post-Movement Processor Design
**Source**: `post-movement-processor-design.md`
**Impact**: Cleaner separation of concerns

1. **Implement `PostMovementProcessor`**:
   - Create dedicated struct for post-movement logic
   - Handle line detection and territory disconnection
   - Manage player choices for graduated rewards
   - Process chain reactions systematically

## Priority 4: Visual Enhancements

### 4.1 GUI Polish
**Source**: `ringrift-gui-diagnostic-plan.md`, GUI roadmap documents
**Impact**: Better user experience

1. **Implement diagonal line styling** (square boards):
   - Make diagonal lines 50% closer to background color
   - Use color interpolation for visual distinction

2. **Enhance visual feedback**:
   - Add animations for line collapse
   - Highlight territory disconnections
   - Show chain capture sequences clearly
   - Improve game over screen with detailed stats

3. **Dynamic board resizing**:
   - Verify current scaling logic works properly
   - Add min/max scale limits
   - Ensure consistent padding

## Priority 5: Future Enhancements

### 5.1 Advanced Features
**Source**: Various GUI planning documents

1. **Replay functionality**:
   - UI for loading replay files
   - Step-through controls
   - Action visualization

2. **Network play**:
   - Multiplayer support
   - Server architecture
   - Synchronization logic

3. **AI improvements**:
   - Multiple AI difficulty levels
   - Minimax implementation
   - MCTS integration

## Implementation Order

### Week 1: Critical Fixes
1. Fix all compiler errors (1.1)
2. Diagnose and fix GUI rendering (1.2)
3. Fix AI integration bug (1.3)

### Week 2: Core Functionality
1. Implement history system (2.1)
2. Fix territory disconnection (2.2)
3. Add forced cap elimination UI (2.3.1)

### Week 3: Feature Completion
1. Add graduated line reward UI (2.3.2)
2. Implement save/load (2.3.3)
3. Refactor main function (3.1)

### Week 4: Polish and Testing
1. Implement PostMovementProcessor (3.2)
2. Add visual enhancements (4.1)
3. Comprehensive testing and bug fixes

## Technical Debt to Address

1. **Obsolete Documentation**: Move outdated docs to `deprecated/` folder
2. **Code Organization**: Consolidate related functionality into appropriate modules
3. **Test Coverage**: Add unit tests for critical game logic
4. **Error Handling**: Improve error messages and recovery strategies
5. **Performance**: Profile and optimize hot paths

## Success Metrics

- All code compiles without warnings
- GUI renders correctly for all board types
- AI players function properly in GUI
- All game rules implemented correctly
- Feature parity between CLI and GUI
- Comprehensive test coverage
- Clean, maintainable codebase

## Next Immediate Steps

1. Switch to code mode
2. Fix compiler errors in `gui/mod.rs`
3. Add diagnostic logging for GUI rendering issue
4. Test and verify fixes
5. Proceed with Priority 2 items

This roadmap provides a clear path forward for improving the RingRift codebase while minimizing technical debt and ensuring a permanent, high-quality solution.