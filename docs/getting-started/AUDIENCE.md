# Who Is RingRift For?

> **Last Updated:** 2025-12-17

This document helps potential players, contributors, and users understand whether RingRift is a good fit for their interests.

---

## Perfect Fit

### Abstract Strategy Enthusiasts

If you enjoy games like **Tak**, **GIPF**, **Hive**, **Shobu**, or **Onitama**, RingRift will feel familiar yet fresh:

- **Zero randomness** — every outcome is determined by player decisions
- **Perfect information** — no hidden cards, dice, or luck
- **Deep decision trees** — simple rules create emergent complexity
- **Multiple victory paths** — adapt your strategy based on board state

### AI/Game-Playing Researchers

RingRift offers a novel environment for AI research:

- **Non-trivial state space** — up to 469 cells on hex boards with complex stack interactions
- **Cross-language parity** — identical rules in TypeScript and Python (90 contract vectors)
- **Explicit decision points** — no hidden auto-execution; all choices are surfaced
- **AlphaZero-style training pipeline** — distributed self-play with neural network evaluation
- **Multiple board sizes** — 8×8 (64), 19×19 (361), hexagonal (469) for varying complexity

### Software Engineers

RingRift serves as a reference implementation for:

- **Spec-driven game engine architecture** — RR-CANON-RXXX formal specification
- **Domain-driven design** — 8 canonical aggregates for game state
- **Real-time multiplayer** — WebSocket synchronization with spectator support
- **Comprehensive testing** — 10,000+ tests with 100% cross-language parity

---

## Good Fit (With Caveats)

### 3-4 Player Games

RingRift supports 2-4 players, but **3-4 player games introduce political dynamics**:

| Players       | Experience                                       |
| ------------- | ------------------------------------------------ |
| **2 players** | Pure strategy, no politics — closest to chess/Go |
| **3 players** | Moderate politics, dynamic short-term alliances  |
| **4 players** | High politics, kingmaking scenarios possible     |

**What this means:**

- **Alliances are temporary** — today's ally is tomorrow's target
- **"Hit the leader"** — weaker players may gang up on the frontrunner
- **Kingmaking** — a losing player may decide who wins by their final moves

If you enjoy games like **Diplomacy**, **Catan**, or **Risk** for their political elements, you'll appreciate 3-4 player RingRift. If you prefer pure strategy without negotiation, stick to **2-player games**.

### High Cognitive Load

RingRift rewards deep thinking:

- **Chain captures** are mandatory once started — you must plan the entire sequence
- **Line vs. territory processing** — understanding when each triggers and their costs
- **Forced elimination loops** — recognizing when you're blocked and must sacrifice rings
- **Three victory conditions** — tracking multiple win paths simultaneously

This makes RingRift better suited for players who enjoy **learning deep systems** rather than casual pick-up-and-play games.

### No Randomness

Some players prefer dice or cards for tension and variance. RingRift offers none:

- **Every game is determined by skill** — no lucky draws to save you
- **Mistakes are punishing** — you can't blame the dice
- **Analysis is deep** — strong players will consistently beat weaker ones

If you prefer games with "comeback mechanics" via randomness, RingRift may feel unforgiving.

---

## Not Ideal For

### Players Who Dislike Politics

If multiplayer negotiation and temporary alliances frustrate you:

- **Stick to 2-player games** — these are pure strategy with no political dynamics
- **Play against AI** — the 10-level AI ladder offers a politics-free experience

### Casual Players Seeking Quick Games

RingRift games can be:

- **Mentally demanding** — each turn requires planning several moves ahead
- **Variable in length** — games can extend when players trade territory back and forth
- **Complex to learn** — the tutorial helps, but mastery takes time

### Players Who Prefer Hidden Information

RingRift is a **perfect information** game:

- **No hidden cards** or secret objectives
- **Board state is fully visible** to all players
- **No bluffing** or deception mechanics

If you enjoy games where reading your opponent's hidden intentions matters, RingRift may feel too "open."

---

## Recommended Starting Point

| Player Type                 | Recommendation                                     |
| --------------------------- | -------------------------------------------------- |
| **New to RingRift**         | 2-player game on 8×8 board against Level 2-3 AI    |
| **Learning the rules**      | Use the tutorial mode and teaching scenarios       |
| **Want pure strategy**      | 2-player games on any board size                   |
| **Want social dynamics**    | 3-4 player games with friends                      |
| **AI researcher**           | Start with Python AI service, 8×8 board            |
| **Engineer exploring code** | Read `RULES_CANONICAL_SPEC.md` then explore engine |

---

## Player Count Recommendations

When creating a game, consider:

```
2-player: Pure strategy, no politics
           Best for: Competitive play, learning, tournaments

3-player: Moderate politics, dynamic alliances
           Best for: Friends who enjoy shifting allegiances

4-player: High politics, kingmaking possible
           Best for: Social groups who enjoy negotiation games
```

**For your first games, we recommend 2-player on the 8×8 board** to focus on learning mechanics without political complexity.

---

## See Also

- [Complete Rules](../rules/COMPLETE_RULES.md) — Full rulebook with examples
- [Quick Reference Diagrams](ux/RULES_QUICK_REFERENCE_DIAGRAMS.md) — Visual guides for common rules
- [Canonical Specification](RULES_CANONICAL_SPEC.md) — Formal rules spec for implementers
