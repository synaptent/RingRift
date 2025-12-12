# Recovery Analysis Runbook (RR-CANON-R110–R115)

This doc captures the current, canonical tooling for analyzing Recovery Slide
eligibility and availability across self-play logs, so work is not duplicated.

## Scripts

- `ai-service/scripts/analyze_game_statistics.py`
  - High-level self-play summaries (victory types, recovery usage, game length).
- `ai-service/scripts/analyze_recovery_across_games.py`
  - Replays games with `GameEngine.apply_move(..., trace_mode=True)` and checks
    recovery eligibility (RR-CANON-R110) for **all players at every state**.
- `ai-service/scripts/analyze_recovery_opportunities.py`
  - Replays games with the canonical GameEngine and tracks **eligibility
    windows**; during a window it checks whether `MoveType.RECOVERY_SLIDE`
    is actually offered by `GameEngine.get_valid_moves` when it’s that player’s
    turn.

## Recommended commands (from repo root)

Generate a quick markdown stats report over current self-play logs:

`PYTHONPATH=ai-service python ai-service/scripts/analyze_game_statistics.py --jsonl-dir ai-service/logs/selfplay --format markdown --output /tmp/selfplay_stats.md`

Check recovery eligibility frequency and sample states (writes JSON with error samples):

`PYTHONPATH=ai-service python ai-service/scripts/analyze_recovery_across_games.py --input-dir ai-service/logs/selfplay --pattern '*.jsonl' --max-games 50 --output /tmp/recovery_across.json`

Track eligibility windows vs actual recovery availability (limit is per JSONL file):

`PYTHONPATH=ai-service python ai-service/scripts/analyze_recovery_opportunities.py --dir ai-service/logs/selfplay --limit 10 --verbose`

## Interpreting results

- Many eligibility states but few “recovery offered” turns usually means:
  - eligibility (RR-CANON-R110) is being met, but `recovery_slide` legal-move
    generation is rare/blocked in practice; or
  - the player is eligible but rarely reaches a movement turn while eligibility
    persists (should be unlikely in canonical history, but the window analysis
    will show it).

If you see replay errors:

- Prefer fixing parsing/replay tooling (these scripts should replay move payloads
  directly) rather than weakening canonical rules or skipping phases.
