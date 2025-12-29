# Legacy Rules Differences Documentation

**Created:** December 2025
**Purpose:** Document differences between legacy game recordings and canonical RR-CANON spec
**Status:** ACTIVE - Reference for replay compatibility

---

## Overview

This document catalogs rule changes and format differences between schema versions
to support accurate replay of historical game recordings. Games recorded under
legacy versions may require special handling to replay correctly.

---

## Schema Version History

| Version | Date     | Major Changes                                   |
| ------- | -------- | ----------------------------------------------- |
| 1-7     | Pre-2024 | Legacy format, various bugs and inconsistencies |
| 8       | Dec 2024 | Hex geometry change (radius 10 → 12)            |
| 9       | Dec 2025 | Current canonical version (RR-CANON compliant)  |

---

## 1. Move Type Renames

### December 2024 Consolidation

The following move types were renamed for consistency with the 7-phase state machine:

| Legacy Name                | Canonical Name            | Reason              |
| -------------------------- | ------------------------- | ------------------- |
| `CHOOSE_LINE_REWARD`       | `choose_line_option`      | Standardized naming |
| `LINE_REWARD`              | `choose_line_option`      | Alias consolidation |
| `LINE_CHOICE`              | `choose_line_option`      | Alias consolidation |
| `PROCESS_TERRITORY_REGION` | `choose_territory_option` | Standardized naming |
| `TERRITORY_REGION`         | `choose_territory_option` | Alias consolidation |
| `TERRITORY_CHOICE`         | `choose_territory_option` | Alias consolidation |

### Phase-Specific Action Consolidation

| Legacy Name             | Canonical Name        | Reason                        |
| ----------------------- | --------------------- | ----------------------------- |
| `NO_LINE_REWARD`        | `no_line_action`      | Standardized no-action naming |
| `SKIP_LINE_REWARD`      | `no_line_action`      | Standardized no-action naming |
| `NO_TERRITORY_REWARD`   | `no_territory_action` | Standardized no-action naming |
| `SKIP_TERRITORY_REWARD` | `no_territory_action` | Standardized no-action naming |

### Capture Phase Aliases

| Legacy Name       | Canonical Name       | RR-CANON Rule |
| ----------------- | -------------------- | ------------- |
| `CAPTURE_RING`    | `overtaking_capture` | RR-CANON-R100 |
| `PERFORM_CAPTURE` | `overtaking_capture` | RR-CANON-R100 |
| `CAPTURE`         | `overtaking_capture` | RR-CANON-R100 |

### Recovery Phase Aliases

| Legacy Name      | Canonical Name   | Notes                      |
| ---------------- | ---------------- | -------------------------- |
| `RECOVER_RING`   | `recovery_slide` | Phase 5 of 7-phase machine |
| `STACK_RECOVERY` | `recovery_slide` | Alternative legacy name    |

### Miscellaneous Aliases

| Legacy Name           | Canonical Name       | Notes                   |
| --------------------- | -------------------- | ----------------------- |
| `ELIMINATE_PLAYER`    | `forced_elimination` | Phase 7 trigger         |
| `FORCED_ELIMINATE`    | `forced_elimination` | Variant spelling        |
| `PIE_RULE`            | `swap_sides`         | Classic Go terminology  |
| `SWAP_SIDES_ACCEPTED` | `swap_sides`         | Verbose form            |
| `SWAP_COLORS`         | `swap_sides`         | Alternative terminology |

**Implementation:** `app/rules/legacy/move_type_aliases.py`

---

## 2. Game Status Changes

### Status Value Normalization

| Legacy Status | Canonical Status | Context                      |
| ------------- | ---------------- | ---------------------------- |
| `finished`    | `completed`      | Game ended normally          |
| `FINISHED`    | `completed`      | Uppercase variant            |
| `Finished`    | `completed`      | Title case variant           |
| `ended`       | `completed`      | Alternative terminology      |
| `done`        | `completed`      | Informal variant             |
| `in_progress` | `active`         | Game ongoing                 |
| `IN_PROGRESS` | `active`         | Uppercase variant            |
| `started`     | `active`         | Initial state after creation |
| `pending`     | `active`         | Waiting for moves            |
| `waiting`     | `active`         | Alternative terminology      |

**Implementation:** `app/rules/legacy/state_normalization.py`

---

## 3. Phase Name Changes

### Uppercase to Lowercase Normalization

All phase names normalized to lowercase snake_case:

| Legacy Phase     | Canonical Phase        |
| ---------------- | ---------------------- |
| `RING_PLACEMENT` | `ring_placement`       |
| `MOVEMENT`       | `movement`             |
| `CAPTURE`        | `capture`              |
| `LINE_FORMATION` | `line_processing`      |
| `LINE_REWARD`    | `line_processing`      |
| `TERRITORY`      | `territory_processing` |
| `RECOVERY`       | `movement`             |
| `GAME_OVER`      | `game_over`            |

### Semantic Phase Consolidation

| Legacy Phase         | Canonical Phase   | Notes                         |
| -------------------- | ----------------- | ----------------------------- |
| `PLACEMENT`          | `ring_placement`  | Abbreviated form              |
| `PLACE_RINGS`        | `ring_placement`  | Verbose form                  |
| `CAPTURING`          | `capture`         | Gerund form                   |
| `LINE` / `LINES`     | `line_processing` | Abbreviated forms             |
| `RECOVER`            | `movement`        | Recovery is movement subphase |
| `ENDED` / `FINISHED` | `game_over`       | Termination aliases           |

### CamelCase Variants (Old Serialization)

| Legacy Phase    | Canonical Phase   |
| --------------- | ----------------- |
| `RingPlacement` | `ring_placement`  |
| `LineFormation` | `line_processing` |
| `LineReward`    | `line_processing` |
| `GameOver`      | `game_over`       |

**Implementation:** `app/rules/legacy/state_normalization.py`

---

## 4. Board Geometry Changes

### Hexagonal Board (Schema Version 8)

| Property        | Legacy (v1-7) | Canonical (v8+) |
| --------------- | ------------- | --------------- |
| Board radius    | 10            | 12              |
| Total cells     | 331           | 469             |
| Ring count (2P) | 44 per player | 96 per player   |

**Detection logic:** Games with 331 hex cells are identified as legacy geometry
and require legacy replay handling.

**Implementation:** `app/rules/legacy/replay_compatibility.py:89`

---

## 5. Seven-Phase State Machine (RR-CANON)

The canonical game uses a 7-phase state machine per RR-CANON-R001:

| Phase | Name                   | Description                         |
| ----- | ---------------------- | ----------------------------------- |
| 1     | `ring_placement`       | Initial ring placement, swap rule   |
| 2     | `movement`             | Stack movement, includes recovery   |
| 3     | `capture`              | Overtaking capture, chain captures  |
| 4     | `line_processing`      | Line detection and reward selection |
| 5     | `territory_processing` | Territory claiming                  |
| 6     | `forced_elimination`   | Elimination when no actions taken   |
| 7     | `game_over`            | Terminal state                      |

### Legacy Deviations

1. **Recovery as Separate Phase:** Some legacy games treat recovery as phase 5,
   but canonically it's part of phase 2 (movement).

2. **Line Formation vs Line Processing:** Legacy code used "line_formation"
   for detection and "line_reward" for selection. Canonical uses single
   "line_processing" phase.

3. **Territory Variants:** Legacy used "territory" phase, canonical uses
   "territory_processing" for clarity.

---

## 6. Replay Compatibility Detection

A game requires legacy replay handling if ANY of these conditions are met:

1. **Schema version 1-7:** Predates canonical spec
2. **Hex geometry 331 cells:** Old board radius
3. **Legacy move types present:** Any move type in alias table
4. **Legacy phase names:** Uppercase or non-standard phases
5. **Legacy status values:** Non-canonical game status

**Detection function:** `app/rules/legacy/replay_compatibility.py:requires_legacy_replay()`

---

## 7. Migration Status

### Games Requiring Legacy Handling

| Category          | Estimated Count | Status             |
| ----------------- | --------------- | ------------------ |
| Schema v1-7 games | ~1,879          | Tracked in DB      |
| Old hex geometry  | Unknown         | Detection in place |
| Legacy move types | Unknown         | Runtime conversion |

### Tracking (TODO)

Legacy replay usage is not yet tracked. See:
`app/rules/legacy/replay_compatibility.py:get_legacy_replay_stats()` (line 301)

**Recommendation:** Implement metrics collection to track:

- Legacy replays per day
- Canonical vs legacy ratio
- Specific legacy features triggered

---

## 8. Code Locations

| Purpose              | File                                       | Key Functions                   |
| -------------------- | ------------------------------------------ | ------------------------------- |
| Replay fallback      | `app/rules/legacy/replay_compatibility.py` | `replay_with_legacy_fallback()` |
| Move type conversion | `app/rules/legacy/move_type_aliases.py`    | `convert_legacy_move_type()`    |
| State normalization  | `app/rules/legacy/state_normalization.py`  | `normalize_legacy_state()`      |
| Phase inference      | `app/rules/legacy/state_normalization.py`  | `infer_phase_from_moves()`      |
| Legacy engine        | `app/_game_engine_legacy.py`               | `GameEngine` class              |

---

## 9. Deprecation Timeline

| Milestone              | Target Date | Action                              |
| ---------------------- | ----------- | ----------------------------------- |
| Metrics implementation | Q1 2025     | Track legacy replay usage           |
| Migration analysis     | Q2 2025     | Identify games that can be migrated |
| Soft deprecation       | Q3 2025     | Warnings on legacy replay           |
| Hard deprecation       | Q4 2025     | Remove legacy code paths            |

**Note:** Timeline depends on legacy game migration progress. Games that cannot
be migrated will require permanent legacy support.

---

## 10. Adding New Legacy Mappings

When discovering new legacy formats, add mappings to:

1. **Move types:** `app/rules/legacy/move_type_aliases.py:LEGACY_TO_CANONICAL_MOVE_TYPE`
2. **Status values:** `app/rules/legacy/state_normalization.py:LEGACY_STATUS_MAPPING`
3. **Phase names:** `app/rules/legacy/state_normalization.py:LEGACY_PHASE_MAPPING`
4. **Detection logic:** `app/rules/legacy/replay_compatibility.py:requires_legacy_replay()`

Always log legacy usage at DEBUG level for tracking.

---

## References

- RR-CANON Specification: `../../../RULES_CANONICAL_SPEC.md`
- 7-Phase State Machine: `GAME_NOTATION_SPEC.md`
- FSM Implementation: `app/rules/fsm.py`
- Phase Machine: `app/rules/phase_machine.py`
