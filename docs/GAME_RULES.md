# RingRift Rules In 5 Minutes

RingRift is a deterministic strategy game for `2-4` players. You build stacks
of rings, leave markers behind as you move, turn marker lines into permanent
territory, and sometimes sacrifice your own rings to keep the board moving.

You can win in three different ways:

- eliminate enough rings
- dominate the board with territory
- become the last player who still has real actions

## 1. Setup

Choose a board and give each player the matching ring supply:

| Board       | Spaces | Rings Per Player | Line Length                        |
| ----------- | -----: | ---------------: | ---------------------------------- |
| `square8`   |   `64` |             `18` | `4` in 2-player, `3` in 3-4 player |
| `hex8`      |   `61` |             `18` | `4` in 2-player, `3` in 3-4 player |
| `square19`  |  `361` |             `72` | `4`                                |
| `hexagonal` |  `469` |             `96` | `4`                                |

Everyone starts with an empty board and their own rings in hand.

Important terms:

- A **stack** is one or more rings in one space.
- You control a stack if your color is on top.
- A **marker** is what you leave behind when a stack moves away.
- **Territory** is a collapsed area that permanently belongs to one player.

In `2`-player games only, Player 2 may use a one-time **swap sides** option
after Player 1's first full turn.

## 2. What You Do On A Turn

Every turn follows the same broad rhythm:

1. optional ring placement
2. movement
3. overtaking captures and any forced chain captures
4. line processing
5. territory processing
6. victory check

### Ring placement

If you still have rings in hand, you may place:

- `1-3` rings on an empty space
- `1` ring onto an existing stack

You may not place a stack that would be immediately dead with no legal move or
capture.

### Movement

Choose a stack you control and move it in a straight line.

- On square boards you can move in `8` directions.
- On hex boards you can move in `6` directions.
- The move distance must be at least the total stack height.
- You cannot move through other stacks or collapsed territory.
- You cannot land on collapsed territory.

When the stack leaves its starting space, it leaves one of your markers behind.

If you land on any marker, that marker is removed and one ring from the top of
your own cap is eliminated immediately.

## 3. How Captures Work

RingRift's main tactical capture is an **overtaking capture**.

If your moving stack jumps over another stack and lands legally beyond it:

- you take one ring from the top of the overtaken stack
- that captured ring goes to the **bottom** of your moving stack
- the ring stays in play, but now it is buried inside your stack

If another overtaking capture is available from the new landing point, you must
continue the chain until no further legal capture segment exists.

This means capture turns can change direction, swing back over earlier stacks,
and completely rewrite who controls which towers.

## 4. Lines And Territory

After movement and capture chains, the board resolves the larger structural
effects.

### Lines

A line is a straight run of your markers of the required length.

When you process a line, you usually:

- collapse those marker spaces into your permanent territory
- pay a ring cost by eliminating one ring from the top of a stack you control

If the line is longer than required, you may choose a smaller partial collapse
to avoid paying that cost.

### Territory

After line collapse, the board may contain disconnected regions surrounded by
one player's markers plus existing collapsed spaces.

If a region is eligible, the acting player may:

- eliminate every stack inside that region
- claim every space in that region as permanent territory
- pay a self-elimination cost outside the region

This is one of the game's signature ideas: the biggest territorial gain often
requires sacrificing some of your own material too.

## 5. Special Rules That Matter

### Forced elimination

If your turn begins and you still control stacks but have no legal placement,
movement, or capture anywhere, you are blocked.

Instead of passing, you must eliminate the entire cap of one stack you control.

That keeps the game progressing and also means "having no move" is not always
safe.

### Recovery

If you control no stacks but still have markers on the board and buried rings
inside opponents' stacks, you may be able to recover.

Recovery actions let a player who looks eliminated fight back by moving a
marker, forming a line, or paying buried-ring costs to re-enter the game.

## 6. How You Win

You win immediately if you satisfy any one of these conditions first.

### Ring elimination victory

Eliminate enough rings to hit the board's threshold.

Common thresholds:

| Board       | 2-Player | 3-Player | 4-Player |
| ----------- | -------: | -------: | -------: |
| `square8`   |     `18` |     `24` |     `30` |
| `hex8`      |     `18` |     `24` |     `30` |
| `square19`  |     `72` |     `96` |    `120` |
| `hexagonal` |     `96` |    `128` |    `160` |

Your total includes rings you eliminated through:

- line costs
- territory costs
- forced elimination
- clearing disconnected regions

### Territory victory

To win on territory, you must satisfy both:

- your territory is at least `floor(total spaces / players) + 1`
- your territory is greater than all opponents' territory combined

So you need both a big footprint and true board dominance.

### Last player standing

This victory is about **real actions**, not merely surviving.

A real action means you can still place, move, or overtake-capture. Forced
elimination alone does not count.

If you are the only player still taking real actions for three consecutive
rounds, you win by Last Player Standing.

### Global stalemate

If nobody can do anything at all anymore, the game ends in a structured
stalemate. Territory breaks the tie first, then eliminated rings, then markers,
then the last player to complete a valid action.

## 7. Why The Game Feels Different

RingRift is strategically interesting because the board never stays neutral.

- Every move leaves a marker, so movement also builds future structure.
- Captures steal rings into stacks instead of simply deleting them.
- Territory usually requires self-sacrifice, so "winning locally" can cost you.
- Buried rings and recovery actions make player elimination less final than it
  first appears.
- In multiplayer, temporary coalitions and board order matter a lot.

## 8. Why It Is Hard For AI

RingRift is difficult for AI for the same reasons it is interesting for people:

- branching factor is high because turns can include placement, movement,
  capture chains, line choices, and territory choices
- many rewards are delayed because markers only become valuable later
- material advantage is not everything; territory shape and buried-ring
  structure matter too
- `3-4` player games introduce coalition pressure and kingmaking-style dynamics
  that are harder than clean `1v1` search

If you want the formal rulebook after this overview, read
[rules/HUMAN_RULES.md](rules/HUMAN_RULES.md) and
[rules/COMPLETE_RULES.md](rules/COMPLETE_RULES.md).
