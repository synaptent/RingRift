# RingRift Next Steps - January 13, 2026

## Priority 0: Critical AI Weakness (BLOCKING)

**Problem**: All neural network models are weaker than the heuristic baseline.

| Model                  | vs Heuristic | Estimated Elo | Target |
| ---------------------- | ------------ | ------------- | ------ |
| square8_2p (prod)      | 38%          | 800           | 1500+  |
| square8_2p (retrained) | 31%          | 870           | 1500+  |
| hex8_2p                | 72%          | 1002          | 1500+  |
| Heuristic baseline     | 50%          | 1517          | -      |

### Immediate Actions

1. **Generate high-quality training data on cluster**

   ```bash
   # On cluster leader node
   ssh ubuntu@<cluster-leader>
   cd ~/ringrift/ai-service

   # Run quality selfplay with b800 budget
   python scripts/selfplay.py --board square8 --num-players 2 \
     --num-games 5000 --engine gumbel --budget 800 \
     --output-dir data/games/quality_sq8_2p
   ```

2. **Export training data from quality games**

   ```bash
   python scripts/export_replay_dataset.py \
     --use-discovery --board-type square8 --num-players 2 \
     --min-budget 600 --output data/training/sq8_2p_quality.npz
   ```

3. **Train with overfitting fixes**

   ```bash
   python -m app.training.train --board-type square8 --num-players 2 \
     --data-path data/training/sq8_2p_quality.npz \
     --epochs 20 --patience 5 --lr 0.0001
   ```

4. **Evaluate and iterate**
   ```bash
   python -m app.gauntlet.runner --board-type square8 --num-players 2 \
     --model-path models/latest.pth --games 50
   ```

### Success Criteria

- Model achieves >55% win rate vs heuristic
- Model achieves >85% win rate vs random
- Estimated Elo > 1450

---

## Priority 1: Production Stability

### Fix Health Endpoint

The `/api/health` endpoint returns 404. Need to investigate server routing.

### Commit Frontend Changes

```bash
git status  # Shows modified:
# - src/client/components/SandboxGameConfig.tsx
# - src/client/hooks/useSandboxDecisionHandlers.ts
```

---

## Priority 2: Model Deployment

Once models meet success criteria:

1. Sync improved models to production server
2. Update MODEL_ZOO.md with new Elo ratings
3. Monitor production AI performance

---

## Technical Debt

- Model checksum/architecture verification needs hardening
- Training pipeline needs better quality gates
- Need automated CI/CD for model deployment

---

## References

- AI_IMPROVEMENT_PLAN.md - Detailed technical analysis
- MODEL_ZOO.md - Model catalog and Elo ratings
- EVALUATION_METHODOLOGY.md - How models are evaluated
