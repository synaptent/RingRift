# ringrift-env

RingRift as a multi-agent reinforcement-learning environment.

[RingRift](https://github.com/synaptent/RingRift) is a deterministic,
perfect-information territory game for **2–4 players** with stacking,
mandatory chain captures, marker lines, and territory enclosure — a
combination that is hard to find in standard multi-agent benchmarks. This
package exposes the canonical Python rules engine (parity-tested against the
TypeScript source of truth) behind a
[PettingZoo](https://pettingzoo.farama.org/) AEC-style API, without
requiring torch or pettingzoo for the core environment.

## Install

From a RingRift repository checkout:

```bash
pip install -e packages/ringrift-env
```

Outside the repository, point the package at an ai-service checkout:

```bash
export RINGRIFT_AI_SERVICE_PATH=/path/to/RingRift/ai-service
```

Dependencies: `numpy`, `pydantic`, `psutil`. **No torch required** for
rules-only use (playouts, move enumeration, action masks).

## Quickstart: random self-play

```python
from ringrift_env import RingRiftAECEnv

env = RingRiftAECEnv(board_type="hex8", num_players=2)
env.reset(seed=42)

while env.agents and not (any(env.terminations.values()) or any(env.truncations.values())):
    obs = env.observe()                    # {"state": GameState, "action_mask": np.ndarray}
    action = env.sample_random_action()    # or: your policy over obs["action_mask"]
    env.step(action)

print(env.rewards)   # e.g. {"player_1": -1.0, "player_2": 1.0}
print(env.infos)     # winner, victory_reason, move_count
```

## API

| Member                                                    | Description                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `RingRiftAECEnv(board_type, num_players, max_moves=None)` | boards: `square8`, `square19`, `hex8`, `hexagonal`; players: 2–4                     |
| `reset(seed=None)`                                        | new episode; seeds `env.rng` (the engine itself is deterministic)                    |
| `agent_selection`                                         | `"player_N"` whose turn it is                                                        |
| `observe(agent=None)`                                     | `{"state": GameState, "action_mask": bool ndarray}`                                  |
| `step(action: int)`                                       | apply a legal action index for the acting agent                                      |
| `last()`                                                  | `(obs, reward, terminated, truncated, info)` for the acting agent                    |
| `legal_action_indices()` / `legal_moves()`                | current legal actions                                                                |
| `decode_action(idx)` / `step_move(move)`                  | index ↔ `Move` escape hatches                                                        |
| `sample_random_action()`                                  | uniform legal action from `env.rng`                                                  |
| `canonical_action_space_size`                             | canonical policy size (square8: 7000, square19: 67000, hex8: 4500, hexagonal: 91876) |
| `action_space_size`                                       | canonical size + overflow block (default 256 slots)                                  |

### Semantics

- **Turn-based (AEC)**: exactly one agent acts per `step`; RingRift turns
  span several engine moves (placement, movement, captures, line/territory
  processing), each of which is one `step`.
- **Perfect information**: all agents observe the same `GameState`.
- **Rewards** (terminal only): winner `+1.0`, all others `-1.0`;
  draws/stalemates/truncations give everyone `0.0`.
- **Termination vs truncation**: rules-engine game end (ring elimination,
  territory control, last-player-standing, structural stalemate) sets
  `terminations`; hitting `max_moves` sets `truncations`.
- **Determinism**: identical seeds + identical policies ⇒ identical
  trajectories. The engine has no internal randomness.
- **Action indices**: indices below `canonical_action_space_size` are
  RingRift's canonical policy indices — the same indexing used by the
  project's trained networks, so masks and policies transfer directly
  between this env and RingRift checkpoints. RingRift's canonical
  encoding deliberately collapses _choice_ moves (line/territory option
  selection, elimination targets, and all hex special moves) onto shared
  indices; legal moves that cannot be distinguished canonically appear in
  a per-state _overflow block_ starting at `canonical_action_space_size`,
  in deterministic engine enumeration order. Use `decode_action(idx)` to
  inspect what any index means in the current state.

## Baseline strength anchors

For calibrating learned agents (promotion-ladder Elo from the RingRift
training pipeline): random ≈ 400, scripted heuristic ≈ 1200, non-trained
MCTS-medium ≈ 1700; the strongest published RingRift network (hex8 2-player)
is ≈ 2584. See
[docs/RESULTS.md](https://github.com/synaptent/RingRift/blob/main/docs/RESULTS.md).

## Scope and roadmap

Current (v0.1): rules-complete env, integer action space with exact masks,
all 4 boards × 2–4 players, deterministic replay, torch-free install.

Planned (tracked in the RingRift repo, issue #100): tensor observation
planes via the canonical NN encoders, rank-aware multiplayer rewards,
PyPI publication, and an OpenSpiel/Pgx contribution.

## Tests

```bash
cd packages/ringrift-env
pip install -e ".[test]"
pytest -m "not slow"     # fast suite (~1 min)
pytest                    # includes full-length large-board games
```
