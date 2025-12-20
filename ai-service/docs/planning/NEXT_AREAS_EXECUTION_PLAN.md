# Next Areas Execution Plan (Remote)

**Created:** 2025-12-20
**Status:** Lane 1 in_progress; others pending

---

## Lane 1: Canonical Data Pipeline (cluster)

- [ ] Confirm selfplay jobs are canonical pipeline or raw
- [ ] For each board (square19, hex), run canonical gate:
  - [ ] Parity check (TS vs Python)
  - [ ] Canonical phase-history validation
- [ ] Write `db_health.canonical_<board>.json` for each DB
- [ ] Update `TRAINING_DATA_REGISTRY.md` with `canonical_ok` status and provenance

**Status:** In Progress

### Lane 1 Progress Log

| Date | Board | Parity | Phase-History | DB Health JSON | Notes |
| ---- | ----- | ------ | ------------- | -------------- | ----- |
|      |       |        |               |                |       |

---

## Lane 2: AI Factory - IG-GMO (experimental tier)

- [ ] Wire IG-GMO into AI factory mapping (server and ai-service)
- [ ] Gate it behind experimental flag or difficulty tier
- [ ] Add doc entry under AI difficulty ladder and service endpoints

**Status:** Pending (blocked on Lane 1)

---

## Lane 3: Parity Hardening

- [ ] Add unit tests for territory detection (empty region semantics)
- [ ] Add replay contract tests for `forced_elimination` and `no_territory_action` sequencing

**Status:** Pending

---

## Lane 4: Documentation Audit

- [ ] Search docs for outdated parity/canonical DB notes
- [ ] Update docs to reflect current gating flow and new IG-GMO tier

**Status:** Pending

---

## Cluster Nodes

| Node  | SSH Command                      | Purpose                     |
| ----- | -------------------------------- | --------------------------- |
| A40   | `ssh -p 38742 root@ssh8.vast.ai` | Primary training/validation |
| 5070  | `ssh -p 10042 root@ssh2.vast.ai` | Secondary training          |
| 4080S | `ssh -p 19940 root@ssh3.vast.ai` | Benchmarking                |

---

## Scripts Reference

```bash
# Parity gate
PYTHONPATH=. python ai-service/scripts/check_ts_python_replay_parity.py \
  --db <path_to_db> --compact

# Canonical history gate
PYTHONPATH=. python ai-service/scripts/check_canonical_phase_history.py \
  --db <path_to_db>
```

---

## Background Tasks

| Task       | PID   | Command                                                                               | Log                                | Started          |
| ---------- | ----- | ------------------------------------------------------------------------------------- | ---------------------------------- | ---------------- |
| Model Sync | 56000 | `sync_models.py --sync --use-sync-coordinator --config config/distributed_hosts.yaml` | `logs/sync_models_coordinator.log` | 2025-12-20 00:24 |

Monitor: `tail -f logs/sync_models_coordinator.log`
