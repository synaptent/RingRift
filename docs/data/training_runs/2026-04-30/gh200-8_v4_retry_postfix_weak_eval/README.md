# gh200-8 v4 retry post-fix weak-eval evidence

Captured: 2026-04-30T13:25Z

This bundle preserves the first post-fix `gh200-8` v4 multiplayer retry after
the fixed-seat head and padded-target fixes were deployed.

The lane no longer failed mechanically:

- `HexNeuralNet_v4` loaded with `num_players=3`.
- `candidate_001.pth` was produced.
- Evaluation started against `best.pth`.

The candidate was not competitive:

- Partial eval at capture: 165 games, candidate 54, best 111, draws 0.
- Candidate win rate: 32.7%.
- At 165/200 games, the candidate could no longer reach the 55% promote
  threshold by the next stage boundary, so the diagnostic loop was stopped and
  the GH200 was repurposed to a fresh fv3 replication seed.

Evidence archive:

- `gh200-8_v4_postfix_weak_eval_20260430T1325Z_evidence.tgz`

The archive includes progress/eval JSON, process command line, git head/status,
model checksums, and the retry log.
