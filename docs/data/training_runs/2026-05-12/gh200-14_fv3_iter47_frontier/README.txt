gh200-14 fv3 iter 47 frontier evidence

Captured: 2026-05-12T22:25:00Z
Node: gh200-14
Workdir: data/minimal_loop_hex8_2p_v5_heavy_fv3

Summary:
- iteration 47 promoted at stage 1
- eval score 33-17 over 50 games, win_rate=0.66
- estimated_elo=2583.9, total_promotions=19
- quality_gate.summary="quality gate passed"
- best.pth and candidate_047.pth have identical SHA256

Elo interpretation:
- This is an estimated promotion-ladder Elo, not a globally calibrated rating.
- The promotion and checkpoint provenance are solid; the absolute Elo scale can
  drift until periodically anchored against fixed checkpoints or a rating pool.

Files:
- metrics.jsonl: copied from the live workdir
- iter047_final.json: exact iteration 47 metrics row extracted from metrics.jsonl
- progress.json: live progress after promotion, showing iteration 48 selfplay started
- model_sha256.txt: SHA256 for best.pth and candidate_047.pth
- processes.txt: live minimal_alphazero_loop.py command line at capture time
- git_head.txt: remote checkout HEAD at capture time
