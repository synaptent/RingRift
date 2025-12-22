# Archived AI Implementations

This directory contains experimental AI implementations that were archived on 2025-12-21.
These implementations are not actively used in production but contain valuable patterns
that may be harvested for future development.

## Restored Files

### gmo_mcts_hybrid.py - UNARCHIVED 2025-12-21

**New location**: `app/ai/gmo_mcts_hybrid.py`
**Reason**: Restored for evaluation as potential approach to improve AI strength to 2000 Elo

The hybrid combines GMO's gradient-based move scoring with MCTS tree search:

- GMO provides prior probabilities to guide MCTS
- Uncertainty-based exploration bonus
- Optional GMO-based rollout policy

---

## Archived Files

### cage_ai.py (233 lines) + cage_network.py

**Status**: Research prototype - PATTERNS HARVESTED ✅
**Reason for archival**: Never deployed to production ladder

**Harvested patterns (December 2025):**

- ✅ Graph neural network for board representation → `app/ai/neural_net/graph_encoding.py`
  - `board_to_graph()` for square boards
  - `board_to_graph_hex()` for hexagonal boards
  - Node/edge feature encoding for GNN architectures

**Remaining patterns (not yet harvested):**

- Primal-dual optimization for constrained move selection
- Energy-based move evaluation (lower energy = better move)

**Potential future use:**

- Constraint-aware optimization applicable to legal move filtering
- Energy-based formulation for value network alternatives

---

### ebmo_online.py (621 lines)

**Status**: Experimental, limited testing
**Reason for archival**: Online learning not stable enough for production

**Valuable patterns to harvest:**

- TD-Energy updates during gameplay
- Rolling buffer of recent games (`GameRecord`, `Transition` dataclasses)
- Outcome-weighted contrastive loss (winner moves -> low energy)
- Per-game gradient accumulation with eligibility traces

**Potential future use:**

- Online learning infrastructure could enable real-time adaptation
- TD-Energy concept could improve value network training
- Game record structure reusable for any replay-based learning

---

## Deprecation Notice

These files are preserved for reference but should NOT be imported in new code.
If you need functionality from these modules:

1. Review the "valuable patterns" section above
2. Extract the relevant pattern into an active module
3. Update imports to use the active module
4. Do NOT add new dependencies on archived code

## Restoration

If an archived implementation needs to be restored:

1. Move file from `archive/` to parent `ai/` directory
2. Update `app/ai/__init__.py` exports if needed
3. Add to AI factory if production use is intended
4. Update this README to remove the restored file
