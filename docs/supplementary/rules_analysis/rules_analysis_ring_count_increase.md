# Rules Analysis: Increasing Ring Counts for 19x19 and Hexagonal Boards

**Status:** Historical proposal. Canonical ring counts are now 18 / 48 / 72 (square8 / square19 / hex); the proposed hex value of 72 was adopted.

## Proposed Change

| Board Type | Current Rings/Player | Proposed Rings/Player | Change |
| ---------- | -------------------- | --------------------- | ------ |
| square8    | 18                   | 18 (no change)        | -      |
| square19   | 36                   | 48                    | +33%   |
| hexagonal  | 48                   | 72                    | +50%   |

**Rationale:** In self-play games, LPS victories appear to dominate over Territory and Ring Elimination victories. The goal is to achieve better balance among the three victory paths.

---

## Current Rule References

### Ring Counts (RR-CANON-R020)

> "For each board type, each player P has a fixed personal supply of rings of P's own colour:
>
> - square8: 18 rings
> - square19: 36 rings
> - hexagonal: 48 rings"

### Victory Thresholds

- **Ring Elimination (RR-CANON-R061):** `victoryThreshold = ringsPerPlayer` (18 for square8, 48 for square19, 72 for hexagonal)
- **Territory (RR-CANON-R062):** `territoryVictoryThreshold = floor(totalSpaces / 2) + 1`
- **LPS (RR-CANON-R172):** Player is only one with "real actions" for 2 consecutive full rounds

---

## Impact Analysis

### 1. Victory Threshold Mathematics

#### 3-Player Games (Most Common for Self-Play)

| Board     | Current Rings | Proposed Rings | Current Elim. Threshold | Proposed Elim. Threshold | Territory Threshold |
| --------- | ------------- | -------------- | ----------------------- | ------------------------ | ------------------- |
| square19  | 108 (3×36)    | 144 (3×48)     | 55                      | 73                       | 181                 |
| hexagonal | 144 (3×48)    | 216 (3×72)     | 73                      | 109                      | 235                 |

#### 2-Player Games

| Board     | Current Rings | Proposed Rings | Current Elim. Threshold | Proposed Elim. Threshold | Territory Threshold |
| --------- | ------------- | -------------- | ----------------------- | ------------------------ | ------------------- |
| square19  | 72 (2×36)     | 96 (2×48)      | 37                      | 49                       | 181                 |
| hexagonal | 96 (2×48)     | 144 (2×72)     | 49                      | 73                       | 235                 |

### 2. Mechanism of Effect

The proposed change would affect victory path balance through:

1. **More material on board → Harder to immobilize opponents → Fewer LPS victories**
   - With more rings, players have more options for placement and stack-building
   - Harder to trap all opponents in FE-only positions for 2 consecutive rounds

2. **More material → More rings available to eliminate → More Elimination victories**
   - Higher threshold, but also more opportunities for line formations and captures
   - Games may have more dramatic ring elimination via territory collapses

3. **Territory threshold unchanged → Relative balance shifts**
   - Same number of spaces to control
   - More rings competing for those spaces could intensify territory battles

---

## Arguments FOR the Change

### A. Addresses Observed Imbalance

If self-play data shows LPS dominating, the proposed change directly addresses the mechanism:

- More rings = more mobility options = harder to achieve LPS
- The change is a targeted intervention at the root cause

### B. Longer, Richer Games

More material generally means:

- More moves before resources deplete
- More complex board states
- Greater opportunity for dramatic reversals via lines and territory cascades

### C. Better Fit for Larger Boards

The rationale for different ring counts per board type is **board density**:

| Board     | Spaces | Current Rings (3P) | Current Density | Proposed Density |
| --------- | ------ | ------------------ | --------------- | ---------------- |
| square8   | 64     | 54                 | 84%             | 84% (unchanged)  |
| square19  | 361    | 108                | 30%             | 40%              |
| hexagonal | 469    | 144                | 31%             | 46%              |

The proposed change increases board density on larger boards, which may lead to:

- More frequent stack interactions
- More line formation opportunities
- More territory pressure

### D. Maintains Relative Scaling

The proposed ratios maintain approximate scaling relationships:

- Current: hex has 1.33× square19 rings (48/36)
- Proposed: hex has 1.5× square19 rings (72/48)
- Both maintain hex > square19 > square8 ordering

---

## Arguments AGAINST the Change

### A. Changes Multiple Variables Simultaneously

The proposed change affects:

1. Total rings in play
2. Ring elimination threshold
3. Game length
4. Board density
5. Placement/movement option counts

This makes it hard to predict net effect and harder to tune if results aren't as expected.

### B. May Overshoot

A 33-50% increase is substantial. If LPS is currently winning 40% of games and the goal is 33% (balanced with elimination and territory), the large increase might push LPS below 20%, creating a different imbalance.

**Alternative:** Consider smaller increments (e.g., 42 and 60) to find the sweet spot.

### C. Game Length Concerns

More rings = longer games. From design goals:

> "Games should remain live and contested for a long time"

But there's a limit. If games become too long:

- AI training becomes more expensive
- Human games may feel like a slog
- Late-game positions become harder to evaluate

### D. LPS May Be Working as Designed

The design explicitly states:

> "Multiple victory paths (ring elimination, territory control, last-player-standing)"

But it doesn't state they should be equally likely. LPS might be _intended_ as the "positional dominance" victory, achievable through superior play. If LPS is winning because good play leads to immobilization, that may be correct.

**Question:** Is LPS dominating because:

1. The rules make it too easy? (Fix: more rings)
2. The AI is finding optimal play? (Not a problem - just means positional play is strong)
3. Other victory conditions are too hard? (Fix: different interventions)

### E. Territory Threshold Becomes Relatively Harder

With more rings competing for the same spaces:

- Territory control requires displacing more opponent material
- The relative difficulty of territory victory increases
- This might make elimination the new dominant path, not a balanced three-way

---

## Alternative Approaches to Consider

### 1. Graduated Increase

Instead of jumping to 48/72, consider intermediate values:

| Board     | Option A | Option B | Option C (Proposed) |
| --------- | -------- | -------- | ------------------- |
| square19  | 40       | 44       | 48                  |
| hexagonal | 56       | 64       | 72                  |

This allows finding the balance point more precisely.

### 2. LPS Rule Adjustment

If LPS is too easy, consider tightening the LPS condition:

- Require 2 consecutive rounds instead of 2
- Require all opponents to have no material (not just no real actions)
- Add a minimum turn count before LPS can trigger

### 3. Line/Territory Reward Tuning

If elimination and territory are underrepresented:

- Increase rings eliminated per line (currently the full line)
- Reduce line length requirement for larger boards
- Increase territory value of collapsed spaces

### 4. Placement Rule Changes

If early immobilization is the issue:

- Allow more flexible placement (e.g., place 1-4 rings instead of 1-3)
- Reduce dead-placement restrictions
- Add "rescue" placement onto trapped stacks

---

## Empirical Study Results

An empirical study was conducted using 124 self-play games across multiple board types and player counts. The AI engine used "descent-only" mode (gradient-based move selection).

### Victory Type Distribution

| Board Type | Players | Games   | LPS       | Elimination | Territory |
| ---------- | ------- | ------- | --------- | ----------- | --------- |
| square8    | 2P      | 62      | 98.4%     | 1.6%        | 0%        |
| square8    | 3P      | 2       | 100%      | 0%          | 0%        |
| square8    | 4P      | 14      | 100%      | 0%          | 0%        |
| square19   | 2P      | 32      | 100%      | 0%          | 0%        |
| square19   | 3P      | 13      | 100%      | 0%          | 0%        |
| square19   | 4P      | 1       | 100%      | 0%          | 0%        |
| **TOTAL**  | -       | **124** | **99.2%** | **0.8%**    | **0%**    |

**Finding:** LPS dominates overwhelmingly (99.2% of victories).

### Gap Analysis: How Close to Non-LPS Victories?

#### 2-Player Games (Closest to Elimination)

| Board    | Elim. Threshold | Avg Achieved | % of Threshold | Gap          |
| -------- | --------------- | ------------ | -------------- | ------------ |
| square8  | 19 rings        | 18 rings     | 95%            | 1 ring short |
| square19 | 37 rings        | 36 rings     | 97%            | 1 ring short |

**Critical Finding:** 2-player games are tantalizingly close to elimination victories—just 1 ring short on average. 100% of 2P games ended within 5 rings of the elimination threshold.

#### 3+ Player Games (Further from Elimination)

| Board    | Players | Elim. Threshold | Avg Achieved | % of Threshold |
| -------- | ------- | --------------- | ------------ | -------------- |
| square8  | 3P      | 28 rings        | 18 rings     | 64%            |
| square8  | 4P      | 37 rings        | 18 rings     | 49%            |
| square19 | 3P      | 55 rings        | 36 rings     | 65%            |
| square19 | 4P      | 73 rings        | 36 rings     | 49%            |

**Finding:** Multi-player games are much further from elimination thresholds.

#### Territory (Uniformly Distant—But With Important Caveat)

| Board    | Players | Terr. Threshold | Avg Achieved | % of Threshold |
| -------- | ------- | --------------- | ------------ | -------------- |
| square8  | 2P      | 33 spaces       | 6.4 spaces   | 19%            |
| square8  | 4P      | 33 spaces       | 9.3 spaces   | 28%            |
| square19 | 2P      | 181 spaces      | 8.4 spaces   | 5%             |
| square19 | 3P      | 181 spaces      | 8.2 spaces   | 5%             |

**Finding:** Territory victories are nowhere close across all configurations. Games end via LPS long before territory can be established.

**Important Caveat:** Territory accumulation is non-linear with game length. As more moves are completed:

1. More markers occupy the board
2. More stacks collapse, creating permanent space claims
3. The board becomes partitioned into enclosed regions
4. Territory "snowballs" as partitions create defensible zones

This means territory percentage grows _acceleratingly_ with game length. The current low territory percentages reflect premature game termination, not a fundamental ceiling on territory accumulation.

### Game Length Analysis

| Board    | Players | Avg Turns |
| -------- | ------- | --------- |
| square8  | 2P      | 17.8      |
| square8  | 3P      | 24.0      |
| square8  | 4P      | 42.1      |
| square19 | 2P      | 26.0      |
| square19 | 3P      | 42.8      |
| square19 | 4P      | 64.0      |

Games are relatively short, with LPS typically triggering before mid-game.

### Key Insights

1. **The Problem is Real:** 99.2% LPS dominance confirms the need for intervention.

2. **2P Games Are Edge Cases:** They're just 1 ring from elimination. A minor rule tweak could tip them over.

3. **Multi-Player Games Are Different:** The gap to elimination widens significantly (49-65% of threshold), suggesting multi-player dynamics favor LPS more strongly.

4. **Territory Is Suppressed, Not Broken:** At 5-28% of threshold currently, territory appears non-viable. However, territory accumulation is _non-linear_—it accelerates as the board fills with markers and collapsed spaces that partition regions. Given enough moves, territory would snowball. The current low percentages reflect games ending prematurely, not a fundamental ceiling.

5. **Root Cause Hypothesis:** LPS triggers too quickly relative to the pace of both material elimination AND territory accumulation. Extending game length would benefit both alternative victory paths—elimination linearly, territory exponentially. _(Note: The canonical LPS rule now requires 2 consecutive full rounds, per RR-CANON-R172.)_

---

## Revised Recommendation

Based on the empirical study, the situation is more severe than anticipated (99.2% LPS vs. the hypothesized 40-50%). This changes the analysis significantly.

### Primary Recommendation: Targeted LPS Adjustment

The empirical data suggests the root cause is **LPS triggering too quickly**, not insufficient material:

1. **2P games are 1 ring away from elimination** - Adding 12 more rings (36→48) per player won't change this by much; the threshold also increases.

2. **Territory is structurally broken** - At 5-28% of threshold, no ring count increase will make territory viable. This requires separate intervention.

3. **The LPS round window was the bottleneck** - Games ended before material dynamics could play out.

**Implemented Intervention:** The canonical rules now require **2 consecutive full rounds** for LPS (RR-CANON-R172), which was adopted based on this analysis.

This change:

- Gives games ~50% more time to reach elimination thresholds
- Allows territory accumulation to progress further
- Maintains the strategic value of immobilization while reducing its dominance
- Is a simpler, more targeted change than altering ring counts

### Secondary Recommendation: Ring Count Increase

If ring count increase is preferred over LPS adjustment:

1. **For 2P games:** Even a modest increase (e.g., square19: 36→40) combined with the higher threshold might tip games toward elimination.

2. **For 3P+ games:** The gap is too large (35-51% of threshold). Ring count alone won't solve this; LPS adjustment is more impactful.

3. **Start smaller than proposed:**
   - square19: 40-42 rings (vs. proposed 48)
   - hexagonal: 56-60 rings (vs. proposed 72)

### Territory Victory: May Self-Correct With Longer Games

The territory system appears undertuned in current data, but this may be an artifact of short game lengths rather than a fundamental design flaw:

- Territory accumulation is **non-linear** with game length
- As moves increase, markers and collapsed spaces partition the board
- Enclosed regions become territory, and this effect compounds
- Longer games (via the LPS 2-round requirement) may naturally enable territory victories

**Wait-and-see approach:** Measure territory percentages with the current LPS 2-round requirement. Only consider threshold adjustments if territory remains suppressed:

- Reducing territory threshold to `floor(totalSpaces / 3) + 1`
- Awarding territory bonuses for controlling connected regions
- Making collapsed stacks count as territory immediately

### Implementation Priority

| Intervention                  | Difficulty | Impact  | Recommendation                          |
| ----------------------------- | ---------- | ------- | --------------------------------------- |
| LPS 2-round requirement       | Low        | High    | **Already implemented (RR-CANON-R172)** |
| Ring count increase           | Medium     | Medium  | Secondary, if LPS change insufficient   |
| Territory threshold reduction | Low        | Unknown | Wait—may self-correct with longer games |

---

## Conclusion

The empirical study confirms severe LPS dominance (99.2%). However, the data suggests **the proposed ring count increase targets the wrong variable**.

The real issue is the LPS 2-round trigger window, which ends games before material-based victories can occur. Critically:

- **Elimination grows linearly** with game length (more turns = more captures)
- **Territory grows non-linearly** with game length (more turns = more board partitioning = exponentially more enclosed regions)

Both alternative victory paths are suppressed by early LPS termination. The proposed ring count increase would:

- Raise elimination thresholds proportionally to added rings (net effect unclear)
- Not directly help territory (but longer games from more material might)
- Add complexity without addressing root cause

**Recommended path forward:**

1. LPS now requires 2 consecutive rounds (RR-CANON-R172)
2. Monitor victory type distribution—territory may self-correct as games lengthen
3. Consider smaller ring count adjustments only if needed after measuring impact
4. Defer territory threshold changes pending measurement

The most elegant solution is the simplest: if games end too fast via LPS, make LPS harder to achieve. Longer games will naturally allow both elimination and territory to compete.
