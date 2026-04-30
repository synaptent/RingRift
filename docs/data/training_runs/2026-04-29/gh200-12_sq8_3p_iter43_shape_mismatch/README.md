# gh200-12 square8_3p iter 43 shape mismatch

Preserved from gh200-12 on 2026-04-29 after the node reboot cleared the prior
disk I/O hang and exposed a training contract mismatch.

Summary:

- Lane: `square8_3p`, iter 43.
- Model output: fixed-seat 3-player value head, shape `(batch, 3)`.
- NPZ targets: max-slot `values_mp` layout, shape `(batch, 4)`.
- Failure:
  `multi_player_value_loss expects pred_values and target_values to share the
same shape; got pred_values=(512, 3) target_values=(512, 4)`.

Archive:

- `gh200-12_sq8_3p_iter43_shape_mismatch_20260429T1555Z_evidence.tgz`
  contains the remote `progress.json`, metrics, data-quality history,
  process snapshot, workdir listing, git head, and training log tail.

Fix coverage:

- Multiplayer value loss now masks only active player slots and tolerates
  inactive max-player target padding when the requested active count fits both
  tensors.
- Rank-target generation can align targets to a fixed-seat model rank head.
