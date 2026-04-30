# gh200-8 v4 retry num_players mismatch

Preserved from gh200-8 on 2026-04-30 after the v4 multiplayer retry was
stopped.

Summary:

- Launch intended a 3-player hex8 v4 retry with `--num-players 3`,
  `--multi-player`, feature version 3, and rank-distribution loss enabled.
- Iter 1 and iter 2 both completed selfplay, then training failed with:
  `Model value head mismatch (after model creation): model.num_players=4 but
training expects 3 players`.
- The old loop treated that training failure as skippable and started iter 3
  selfplay against unchanged `best.pth`.

Archive:

- `gh200-8_v4_retry_num_players_mismatch_20260430T0230Z_evidence.tgz`
  contains the remote `progress.json`, launch command, git heads, process
  snapshot, log tail, and `error_index.txt`.

Fix coverage:

- Hex model construction now keeps hex multiplayer value/rank heads aligned to
  the actual requested player count.
- `minimal_alphazero_loop.py` now halts on training failure before starting
  the next selfplay iteration.
