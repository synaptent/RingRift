# Security Considerations

## Model Loading Security (December 2025)

### Issue: torch.load with weights_only=False

The codebase uses `torch.load` to load model checkpoints. When `weights_only=False`
(the default in older PyTorch versions), arbitrary Python code in the checkpoint
file can be executed during unpickling.

### Current Status

- **2 usages** with `weights_only=True` (safe)
- **22 usages** with explicit `weights_only=False`
- **30 usages** without `weights_only` parameter (defaults to unsafe)

### Mitigation

1. **Safe Loading Utility**: `app/utils/torch_utils.py` provides `safe_load_checkpoint()`
   which attempts safe loading first and falls back to unsafe mode only when necessary.

2. **Internal Checkpoints**: Most checkpoints are generated internally during training
   and are trusted. The unsafe loading is required because checkpoints contain:
   - Model configuration metadata
   - Training state (optimizer, scheduler)
   - Versioning information

3. **External Models**: For any externally-provided models, use:
   ```python
   from app.utils.torch_utils import safe_load_checkpoint
   checkpoint = safe_load_checkpoint(path, allow_unsafe=False)
   ```

### Migration Plan

1. **Phase 1** (Complete): Created `safe_load_checkpoint` utility
2. **Phase 2** (In Progress): Migrating critical paths to use safe loading by default
   - ✅ `app/training/checkpoint_utils.py` - migrated
   - ✅ `app/training/checkpointing.py` - migrated
   - ✅ `app/training/elo_service.py` - migrated (Dec 2025)
   - ✅ `app/ai/nnue_policy.py` - migrated
   - ✅ `app/ai/minimax_ai.py` - migrated
   - ✅ `app/ai/mcts_ai.py` - migrated
   - ✅ `app/ai/gpu_parallel_games.py` - migrated
   - ✅ `app/ai/cage_ai.py` - migrated
   - ✅ `app/ai/ebmo_network.py` - migrated
   - ✅ `app/ai/ebmo_online.py` - migrated
   - ✅ `app/ai/ig_gmo.py` - migrated
   - ✅ `scripts/fix_hex_checkpoint_metadata.py` - migrated (Dec 2025)
   - ✅ `scripts/cleanup_phantom_elo_entries.py` - migrated (Dec 2025)
3. **Phase 3** (TODO): Update checkpoint format to be fully weights_only compatible

### Files Still Requiring Migration

The following files still use direct `torch.load` with `weights_only=False`:

- `app/ai/_neural_net_legacy.py` (5 usages) - legacy, being deprecated
- `app/training/train.py` (1 usage)
- `app/training/training_enhancements.py` (4 usages)
- `app/training/model_versioning.py` (3 usages) - includes validation loads
- `app/tournament/runner.py` (1 usage)
- `app/models/discovery.py` (1 usage)

### Bandit Findings

Bandit security scanner identifies these as B614 (pytorch_load) issues with
Medium severity. This is documented and tracked for future migration.
