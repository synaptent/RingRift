# RingRift Puzzle Format (schema_version 1)

Puzzles are static JSON assets mined from recorded games (self-play replay
databases) by `ai-service/scripts/mine_chain_capture_puzzles.py`. Epic
tracking: [#104](https://github.com/synaptent/RingRift/issues/104).

## File shape

```json
{
  "schema_version": 1,
  "theme": "chain_capture",
  "count": 60,
  "puzzles": [ <puzzle>, ... ]
}
```

## Puzzle object

| Field                        | Type   | Meaning                                                                                                           |
| ---------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------- |
| `id`                         | string | `{board}_{N}p_{gameIdPrefix}_{ply}` — stable, unique within a file                                                |
| `schema_version`             | int    | currently `1`                                                                                                     |
| `theme`                      | string | currently only `"chain_capture"`                                                                                  |
| `board_type`                 | string | `square8` \| `square19` \| `hex8` \| `hexagonal`                                                                  |
| `num_players`                | int    | 2–4                                                                                                               |
| `player_to_move`             | int    | player number whose turn it is (the solver's seat)                                                                |
| `state`                      | object | full canonical `GameState` (camelCase, same serialization the TS engine consumes); always in the `movement` phase |
| `solution.moves`             | Move[] | principal variation: the initiating `overtaking_capture` followed by the forced `continue_capture_segment` moves  |
| `solution.score`             | int    | rings captured by the solution chain                                                                              |
| `solution.second_best_score` | int    | best chain available to any _other_ first move                                                                    |
| `solution.margin`            | int    | `score - second_best_score` (≥ the miner's `--min-margin`; ≥1 guarantees a unique best first move)                |
| `source.db`                  | string | replay DB file the position came from                                                                             |
| `source.game_id`             | string | source game                                                                                                       |
| `source.ply`                 | int    | move index within the source game                                                                                 |

## Semantics and validation contract

- **Objective (chain_capture theme)**: from `state`, the solver must find
  the first move of the maximum forced capture chain. Only
  `solution.moves[0]` is graded; the remaining moves document the forced
  line for reveal/teaching UI.
- **Uniqueness**: `margin >= 1` by construction, so exactly one first move
  achieves `score`. The miner re-validates every emitted puzzle by
  recomputing all chain scores from the stored state
  (`validate_puzzle`), and `--validate <file>` re-runs that check on any
  puzzle file.
- **Scoring metric** is structural (rings captured in the mandatory chain,
  RR-CANON-R103) — independent of any neural network, so puzzles remain
  valid across model generations.
- **State fidelity**: `state` deserializes with
  `GameState.model_validate` (Python) and matches the shared TS
  `GameState` type; a client can load it directly into the sandbox board.

## Mining

```bash
cd ai-service
PYTHONPATH=. python scripts/mine_chain_capture_puzzles.py \
  --db data/games/canonical_hex8_2p.db --copy-to-temp \
  --min-chain 3 --min-margin 2 --max-puzzles 60 \
  --output ../src/client/public/puzzles/hex8_2p_chain_capture.json

# Re-validate any puzzle file
PYTHONPATH=. python scripts/mine_chain_capture_puzzles.py --validate <file>
```

`--copy-to-temp` mines from a temporary copy so live training databases are
never opened read-write.

## Future themes (out of scope for v1)

- `territory_seal` — one move completes a physical disconnection worth ≥N cells
- `line_collapse` — one move completes an overlength line with an Option-1/2 decision
- Value-swing puzzles mined with the trained network (requires torch + checkpoint)
