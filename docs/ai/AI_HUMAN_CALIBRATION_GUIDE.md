# RingRift AI Human Calibration Guide

This guide defines a repeatable, human-facing procedure for checking whether
the RingRift AI difficulty ladder matches real player experience.

It is intentionally **manual** and aimed at internal and invited external
playtesters. It does not run in CI or production.

This document complements, but does not replace:

- `docs/ai/AI_DIFFICULTY_SPEC.md` — logical difficulty tiers (D1–D10) and
  intended strength ordering.
- `ai-service/app/config/ladder_config.py` — concrete engine settings per
  `(difficulty, board_type, num_players)` via `LadderTierConfig`.
- Tier evaluation and gating tools in `ai-service/app/training/` —
  automated AI-vs-AI checks that keep neighbouring tiers ordered.

Human calibration answers a different question:

> “For a person of skill level X, does playing at tier Dn feel about right?”

Results are used to adjust **which tiers are surfaced to players**, default
recommendations, and future gating thresholds, not to change rules or move
semantics.

## 1. Target configurations (initial focus)

All initial human calibration runs use the canonical Square‑8 two‑player
environment.

- **Board:** `square8` (8×8, compact ruleset).
- **Players:** 2 (head‑to‑head).
- **Game mode:** Standard rules, swap rule enabled for 2‑player games.
- **AI engine:** As configured in the ladder via `LadderTierConfig`.

For square8 2‑player, the ladder defines these anchor tiers:

| Tier | Ladder difficulty | Engine family | Notes                                         |
| ---- | ----------------- | ------------- | --------------------------------------------- |
| D2   | 2                 | Heuristic     | Easy baseline; first non‑random tier.         |
| D4   | 4                 | Minimax       | Intermediate search depth and think time.     |
| D6   | 6                 | Minimax       | High search budget; punishes clear mistakes.  |
| D8   | 8                 | MCTS          | Strong search; intended "near‑expert" anchor. |

### 1.1 Mapping to client difficulty labels and ladder configs

To keep human calibration, UX copy, and ladder configs aligned, use the
following mapping as the single mental model for Square‑8 2‑player:

| Ladder tier | Ladder config (ai‑service)                                                                              | Client difficulty label (`difficultyUx.ts`)    | Intended strength band                         |
| ----------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| D2          | `difficulty=2`, `BoardType.SQUARE8`, `num_players=2`, `ai_type=HEURISTIC`, `model_id="heuristic_v1_2p"` | **Learner (D2 – Casual / Learning)** (`id: 2`) | Casual / learning; brand‑new RingRift players. |
| D4          | `difficulty=4`, `BoardType.SQUARE8`, `num_players=2`, `ai_type=MINIMAX`, `model_id="v1-minimax-4"`      | **Challenging (D4 – Intermediate)** (`id: 4`)  | Intermediate players / strong casuals.         |
| D6          | `difficulty=6`, `BoardType.SQUARE8`, `num_players=2`, `ai_type=MINIMAX`, `model_id="v1-minimax-6"`      | **Advanced (D6 – Strong club)** (`id: 6`)      | Strong club‑level / advanced RingRift players. |
| D8          | `difficulty=8`, `BoardType.SQUARE8`, `num_players=2`, `ai_type=MCTS`, `model_id="v1-mcts-8"`            | **Strong Expert (D8 – Near‑expert)** (`id: 8`) | Very strong / near‑expert players.             |

Interpolated client tiers (D3, D5, D7) are UX‑level “bridges” between these
anchors and do not introduce new ladder configs; they should be interpreted
as sitting strictly between the neighbouring anchor tiers above.

Unless otherwise stated, calibration sessions should:

- Use **Square‑8, 2 players**.
- Use a **rapid** or **classical** time control (e.g. 10+0, 15+10, or
  “play at a comfortable pace” if no clock UI is available).
- Randomise colours/starting seat when possible, or alternate between
  games in a set.

## 2. Qualitative difficulty anchors by tier

The tables below describe **who** each tier is for and what outcome ranges
are considered healthy over small sets of games. They are intentionally
coarse; use them as guidance, not exact targets.

### D2 – Casual / learning tier

- **Intended player profile:**
  - First‑time strategy players.
  - People comfortable with rules explanations but new to deep tactics.
- **Expected behaviour from the AI:**
  - Makes obviously sub‑optimal trades and misses some short tactics.
  - Plays legal, coherent moves; rarely hangs entire stacks for free.
  - Occasionally sets up simple lines or territory threats.
- **Expected results (3–5 game mini‑set):**
  - Brand‑new players should win **at least one** game in their first
    3–5 games.
  - D2 should not win **> ~80%** of games against true beginners.
  - Experienced board‑gamers will usually beat D2 consistently and
    report it as “too easy” or “slightly easy”.

### D4 – Intermediate tier

- **Intended player profile:**
  - Players who comfortably beat D2.
  - People with experience in other abstract games (e.g. club‑night
    Go/Chess players, strong tactics‑heavy board‑gamers).
- **Expected behaviour from the AI:**
  - Rarely blunders whole stacks without compensation.
  - Sees simple 1–2 turn tactics and basic traps.
  - Converts clear material or territory advantages reliably.
- **Expected results (5–10 games):**
  - Intermediate testers should achieve roughly **30–70%** win rate.
  - Strong club‑level testers often win most games and may rate D4 as
    “easy” or “borderline”.

### D6 – Advanced tier

- **Intended player profile:**
  - Strong club‑level abstract players.
  - Advanced RingRift players who consistently beat D4.
- **Expected behaviour from the AI:**
  - Avoids most obvious tactical shots and short‑term greed.
  - Punishes over‑extensions, slow play, and badly defended stacks.
  - Converts small advantages across many moves; endgames feel tight.
- **Expected results (10+ games):**
  - Strong testers should cluster around **40–60%** win rate.
  - Intermediates will usually be pushed below **30%** and report that
    games feel "hard but sometimes winnable".

### D8 – Strong / near‑expert tier

- **Intended player profile:**
  - Very strong RingRift players.
  - Players with deep experience in Go/Chess or similar games at
    serious club or online‑ladder strength.
- **Expected behaviour from the AI:**
  - Rare outright blunders; most losses come from long‑term strategic
    errors by the human.
  - Consistently converts small edges; punishes greedy trades and
    unsafe territory fights.
- **Expected results (10–20 games):**
  - Even very strong testers should struggle to maintain **> 60%**
    win rate.
  - Intermediate players should almost always report D8 as “hard” or
    “too hard” and win only rarely.

## 3. Calibration experiment templates

This section defines small, repeatable experiment blocks. Each block is a
self‑contained session that a tester can run and log in 30–90 minutes.

### Template A – New‑player quick check (D2/D4, 3‑game sets)

- **Audience:** Players who have just learned the rules or have played
  fewer than ~10 games of RingRift.
- **Setup:**
  - Board: square8, 2 players.
  - Time: no clock or 10+0 (rapid); allow thinking time as needed.
  - Tiers: D2 first; optionally D4 if D2 feels trivial.
  - Colours: random or alternating between games.
- **Protocol:**
  1. Explain win conditions and basic tactics using the normal teaching
     flow; let the player play 1 warm‑up game vs D2 if desired.
  2. Play a **3‑game set vs D2**.
  3. Optionally, if the player wins 2–3 games easily, repeat a 3‑game
     set vs D4.
- **Logging (per game):**
  - Tier (D2/D4).
  - Human side (first/second).
  - Result: Win / Loss / Draw / Abandoned.
  - Perceived difficulty (1–5):
    - 1 = Far too easy.
    - 2 = Slightly easy.
    - 3 = About right.
    - 4 = Slightly hard.
    - 5 = Far too hard.

A simple row format is sufficient, e.g.:

| Game | Tier | Human side | Result | Perceived difficulty (1–5) |
| ---- | ---- | ---------- | ------ | -------------------------- |
| 1    | D2   | First      | Win    | 2                          |
| 2    | D2   | Second     | Loss   | 3                          |
| 3    | D2   | First      | Loss   | 4                          |

### Template B – Intermediate validation (D4/D6, 5‑game blocks)

- **Audience:** Players who:
  - Comfortably beat D2 and usually beat D4.
  - Have non‑trivial tactics/strategy experience in other games.
- **Setup:**
  - Board: square8, 2 players.
  - Time: 10+0 or 15+10 (rapid/classical).
  - Tiers: 5 games vs D4, then 5 games vs D6 (can be split across
    sessions).
- **Logging:**
  - Same per‑game fields as Template A.
  - Optionally note any games where the tester felt completely lost or
    completely in control from move 1.

### Template C – Strong‑player session (D6/D8, 10+ games)

- **Audience:**
  - Strong internal RingRift players; designers and AI authors.
  - External testers with serious club or online‑ladder strength.
- **Setup:**
  - Board: square8, 2 players.
  - Time: 15+10 or similar comfortable classical control.
  - Tiers: mixed set of D6 and D8 games; at least 5 games per tier.
  - Colours: alternate strictly between games when possible.
- **Logging:**
  - Per‑game fields as above.
  - Optional short comment (one phrase) per game, e.g.
    “early blunder”, “close endgame”, “AI crushed me in midgame”.

## 4. Interpreting results and feeding back into the ladder

The goal is to detect **gross mismatches** first (tiers that are clearly
too easy or too hard for their audience) and then refine thresholds.

When aggregating results for a given player profile and tier, look for:

- **Win‑rate bands:**
  - For the intended audience, a long‑run win rate in the
    **30–70%** band is usually healthy.
  - Systematic **> 80%** win rate with many “1–2” difficulty scores
    suggests the tier is too easy.
  - Systematic **< 20%** win rate with many “4–5” difficulty scores
    suggests the tier is too hard.
- **Self‑report alignment:**
  - Treat perceived difficulty ≈3 as “about right”; sustained 1–2 or
    4–5 across many games is a red flag.
- **Qualitative notes:**
  - If many testers remark that the AI "throws" games from winning
    positions, the underlying engine profile may need review even if
    win‑rates look acceptable.

Map these observations back to the difficulty spec and ladder:

- If D2 is consistently too strong for new players, consider:
  - Promoting a weaker tier (e.g. current D1) into the surfaced range.
  - Or increasing D2 randomness / reducing think time in the ladder
    config, then re‑running AI‑vs‑AI gating before exposing the change.
- If D6 feels like the true “expert” tier and D8 is unreachable for
  almost everyone, treat D8 as experimental or hidden until the
  ladder is re‑tuned.

Calibration results should be recorded in a shared, access‑controlled
document or dashboard. Avoid storing user identifiers; it is enough to
tag sessions by **player profile** (e.g. "new", "intermediate",
"strong club") and date.

## 5. Relation to in‑game calibration UX and telemetry

Client‑side difficulty UX surfaces and calibration telemetry (see the
difficulty descriptors and `difficulty_calibration_*` events in the code
base) are intended to mirror the setups in this guide:

- When a player opts into **calibration mode** in the UI, the game
  should be created using the same board, player count, and tier
  presets described above (Square‑8, 2‑player, D2/D4/D6/D8).
- Telemetry should record only coarse configuration and outcomes:
  board type, number of players, difficulty, result, moves played, and
  optional perceived difficulty ratings.

As these telemetry pipelines mature, their aggregate data should be:

- Interpreted using the qualitative anchors and experiment templates in
  this guide (Templates A/B/C), and
- Analysed using the workflow and decision rules defined in
  [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1), which joins
  human calibration data with automated tier evaluation, perf budgets,
  and the Square‑8 2‑player tier candidate registry.

Operationally:

- **This guide** defines _how to run and log human calibration sessions_ and what outcomes (“too easy”, “about right”, “too hard”) mean at a qualitative level.
- **`AI_DIFFICULTY_CALIBRATION_ANALYSIS`** defines _how aggregated calibration telemetry and playtest results feed back into concrete ladder tuning decisions_ for the Square‑8 2‑player D2/D4/D6/D8 tiers.
