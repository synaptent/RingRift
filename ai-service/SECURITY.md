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
2. **Phase 2** (TODO): Migrate critical paths to use safe loading by default
3. **Phase 3** (TODO): Update checkpoint format to be fully weights_only compatible

### Files Requiring Attention

The following files use `torch.load` with unsafe settings:

- `app/ai/_neural_net_legacy.py` (5 usages)
- `app/training/train.py` (1 usage)
- `app/training/training_enhancements.py` (4 usages)
- `app/training/model_versioning.py` (4 usages)
- `app/training/checkpointing.py` (2 usages)
- `app/training/checkpoint_utils.py` (1 usage)
- `app/training/checkpoint_unified.py` (1 usage)
- Various other training and AI modules

### Bandit Findings

Bandit security scanner identifies these as B614 (pytorch_load) issues with
Medium severity. This is documented and tracked for future migration.
