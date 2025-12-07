# Human Difficulty Calibration Study Design (H-AI-17)

> **Status (2025-12-06): New – H-AI-17 study design.**  
> **Role:** Define a structured study protocol to systematically gather and analyze human calibration feedback for the AI difficulty ladder.

---

## Overview

### Purpose

Systematically validate that AI tier difficulty progression (D2 → D4 → D6 → D8) matches human perception. While automated metrics (win rates, Elo, latency) can indicate "proper" difficulty progression, only human participants can confirm that the **perceived** difficulty curve delivers the intended player experience.

This study design provides a **repeatable, structured protocol** for gathering and analyzing human feedback that can inform ladder tuning decisions.

### Research Questions

#### Primary Questions

1. **RQ1 – Perceptibility of Tier Differences**: Do players perceive a meaningful difficulty increase between adjacent tiers (D2→D4, D4→D6, D6→D8)?

2. **RQ2 – Perception-Metric Alignment**: At which tiers does perceived difficulty diverge most from automated metrics (win rates, Elo estimates)?

3. **RQ3 – Qualitative Drivers**: What qualitative factors (AI play style, decision speed, mistake frequency, positional understanding) drive difficulty perception?

4. **RQ4 – Skill-Tier Alignment**: Are there specific player skill levels where certain tiers feel significantly misaligned (too easy or too hard)?

#### Secondary Questions

5. **RQ5 – Transition Smoothness**: Does the difficulty curve feel smooth and progressive, or are there abrupt jumps that break immersion?

6. **RQ6 – Label Accuracy**: Do the difficulty labels (Easy, Medium, Hard, Expert) accurately reflect player experience?

7. **RQ7 – Frustration vs Challenge**: At what point does challenge become frustration? Is D8 achievable for its intended audience?

### Scope

- **Board type**: Square-8 (8×8 compact ruleset)
- **Player configuration**: 2-player (head-to-head)
- **Tiers under study**: D2, D4, D6, D8 (primary), with D0 as optional anchor
- **Game mode**: Standard rules, swap rule enabled

---

## Study Design

### Design Type

**Within-subjects, counterbalanced design** (each participant plays multiple tiers).

**Rationale:**

- Within-subjects eliminates individual difference confounds when comparing tier perceptions
- Counterbalancing (varied tier order across participants) prevents order effects
- More statistically efficient than between-subjects for the same sample size
- Allows direct tier-to-tier comparison ratings from each participant

**Counterbalancing scheme:**

- Participants assigned to one of six orderings (Latin square):
  - Order A: D2 → D4 → D6 (→ D8 optional)
  - Order B: D4 → D6 → D2 (→ D8 optional)
  - Order C: D6 → D2 → D4 (→ D8 optional)
  - Order D: D2 → D6 → D4 (→ D8 optional)
  - Order E: D4 → D2 → D6 (→ D8 optional)
  - Order F: D6 → D4 → D2 (→ D8 optional)

D8 is optional for intermediate participants but required for strong participants.

### Tiers Under Study

| Tier | Role in Study                             | Games per Participant    |
| ---- | ----------------------------------------- | ------------------------ |
| D0   | Optional anchor (random baseline)         | 0-1 (for reference only) |
| D2   | Primary – Easy baseline                   | 2-3 games                |
| D4   | Primary – Intermediate checkpoint         | 2-3 games                |
| D6   | Primary – Advanced checkpoint             | 2-3 games                |
| D8   | Primary – Expert anchor (skill-dependent) | 2-3 games                |

**Focus tiers**: D2, D4, D6 are mandatory for all participants. D8 is included for strong participants or as opt-in for others who want the challenge.

### Sample Size

**Target:** 30-45 participants total across skill segments

| Skill Segment        | Target N | Rationale                                              |
| -------------------- | -------- | ------------------------------------------------------ |
| New players          | 10-15    | Primary audience for D2; ensure baseline is accessible |
| Intermediate players | 10-15    | Primary audience for D4; key calibration checkpoint    |
| Strong players       | 10-15    | Primary audience for D6/D8; validate expert anchor     |

**Power analysis considerations:**

- For within-subjects paired comparisons (e.g., perceived difficulty D4 vs D6), with expected effect size d=0.6 (medium-large for adjacent tier differences), power=0.80, α=0.05: minimum N≈25
- 30-45 participants provides buffer for dropouts and stratified analysis

**Practical constraints:**

- Minimum viable study: 20 participants (skewed toward intermediate)
- Ideal: 45 participants evenly distributed across skill segments
- Maximum practical scope: 60 participants (resource-limited for a small team)

---

## Participant Criteria

### Inclusion Criteria

1. **Familiarity with RingRift rules:**
   - Completed in-app tutorial, OR
   - Played at least 3 complete games against any opponent

2. **Skill level targeting** (self-reported or derived):
   - **New**: Fewer than 10 total games played, or self-rated "beginner"
   - **Intermediate**: 10-50 games played, or comfortable with basic tactics
   - **Strong**: 50+ games played, or experience with abstract strategy games at club level

3. **Technical requirements:**
   - Device capable of running RingRift smoothly
   - Stable internet connection (for server-based AI)
   - Availability for 45-90 minute session

### Exclusion Criteria

1. **Internal bias:**
   - Players who participated in AI development or model training
   - Team members with deep knowledge of tier configurations

2. **Extreme outliers:**
   - Players with >500 games (potential ceiling effects even at D8)
   - Players who have never completed a full game after multiple attempts

3. **Technical issues:**
   - Reported device performance issues that affect gameplay
   - Unreliable connection causing frequent game abandonments

### Recruitment

#### Sources (in priority order)

1. **Internal team and friends/family** (warm outreach)
   - Low effort, high completion rate
   - Risk: potential positive bias → mitigate with blinding

2. **Existing player base** (if available)
   - In-app notification or email to opted-in playtesters
   - Filter by game count to target skill segments

3. **External board game communities**
   - Abstract strategy game forums (BoardGameGeek, Reddit r/abstractgames)
   - Discord servers for Go, Chess, or similar games
   - Advantage: built-in strong player segment

#### Recruitment Messaging

**Template:**

> **Help calibrate RingRift's AI difficulty!**
>
> We're running a short study (45-90 minutes) to ensure our AI opponents feel appropriately challenging at each difficulty level.
>
> **What you'll do:**
>
> - Play 6-9 games against AI opponents at various difficulty levels
> - Answer brief surveys about how each game felt
>
> **Who we're looking for:**
>
> - Players who know RingRift basics (completed tutorial or 3+ games)
> - All skill levels welcome! We especially need beginners and advanced players.
>
> **Compensation:** [TBD – e.g., early access, in-game cosmetics, or monetary]
>
> **Interested?** [Contact method / sign-up form link]

#### Incentives

| Option                           | Pros               | Cons                      |
| -------------------------------- | ------------------ | ------------------------- |
| Recognition (thank-you, credits) | Zero cost          | Limited appeal            |
| Early access to new features     | Low cost, relevant | Requires feature pipeline |
| In-game cosmetics                | Low cost, scalable | Requires cosmetic system  |
| Gift card ($10-20)               | Strong motivation  | Budget required           |

**Recommendation:** Offer a small incentive (gift card or in-game reward) to ensure recruitment targets are met, especially for the strong player segment which is typically hardest to recruit.

### Skill Level Stratification

**Classification method:**

1. **Primary**: Self-reported skill questionnaire (administered during recruitment)
   - "How many RingRift games have you played?"
   - "How would you rate your RingRift skill?" (1-5 scale)
   - "Do you have competitive experience in abstract strategy games?"

2. **Secondary**: Derived from game history (if available)
   - Total games played
   - Win rate vs AI at various difficulties
   - Average game length

**Stratification targets:**

- Actively recruit until each skill segment has at least 10 participants
- Over-recruit intermediate segment slightly (most common profile)
- For strong players, accept players from other abstract game communities who complete RingRift tutorial

---

## Protocol

### Pre-Study

#### 1. Consent and Briefing (5-10 minutes)

**Digital consent form covering:**

- Purpose of the study (calibrating AI difficulty)
- What participation involves (playing games, answering surveys)
- Data collected and how it will be used
- Right to withdraw at any time without penalty
- Anonymization of results

**Briefing script includes:**

- Overview of session structure
- Explanation that games will be at different difficulty levels
- Instruction to play naturally (not to try to "test" the AI)
- Reminder that they can take breaks as needed

#### 2. Skill Assessment (5-10 minutes)

**Pre-study questionnaire:**

1. How many RingRift games have you played? (0-2 / 3-10 / 11-50 / 51-200 / 200+)
2. How would you rate your RingRift skill? (1=Complete beginner, 5=Very strong)
3. Have you played other abstract strategy games competitively? (None / Casual / Club level / Tournament)
4. Which games? (Checklist: Chess, Go, Checkers, Othello, Hive, Other)
5. How long have you been playing RingRift? (Less than a week / 1-4 weeks / 1-6 months / 6+ months)

**Optional warm-up game** (for new participants):

- One game against D2 (not counted in data)
- Purpose: Ensure familiarity with game interface

#### 3. Device/Environment Check

- Confirm game loads and responds normally
- Verify no connectivity issues
- Ensure participant has a quiet environment for focused play

### Study Session

#### Session Structure

**Standard session (6 games across 3 tiers):**

| Phase   | Activity                                        | Duration  |
| ------- | ----------------------------------------------- | --------- |
| Setup   | Assignment to tier order; interface orientation | 5 min     |
| Block 1 | 2 games at Tier A + post-game surveys           | 15-25 min |
| Break   | Optional short break                            | 2 min     |
| Block 2 | 2 games at Tier B + post-game surveys           | 15-25 min |
| Break   | Optional short break                            | 2 min     |
| Block 3 | 2 games at Tier C + post-game surveys           | 15-25 min |
| Summary | Post-study summary survey                       | 10 min    |
| Debrief | Reveal tier assignments; thank participant      | 5 min     |

**Total time:** 60-90 minutes

**Extended session (for strong players including D8):**

- Add Block 4: 2-3 games at D8 + surveys
- Total time: 80-120 minutes

#### Per-Game Procedure

1. **Game setup** (automated)
   - Board type: Square-8, 2-player
   - Tier assigned according to counterbalanced order
   - Starting player: randomized (participant or AI first)
   - Swap rule: enabled (standard 2-player rules)

2. **Play the game**
   - No time pressure beyond natural pace
   - Participant may use undo if UI supports (note: consider disabling for cleaner data)
   - Game continues until terminal state (win/loss/draw)

3. **Immediate post-game survey** (see Data Collection)
   - Presented immediately upon game completion
   - 2-3 minute completion time
   - Must complete before next game starts

4. **Optional: Replay review with facilitator** (for moderated sessions)
   - Identify 1-2 moments that felt particularly easy/hard
   - Brief verbal feedback captured by facilitator

#### Blinding

**Participants are NOT told the specific tier difficulty level during the study.**

**Rationale:**

- Knowing "this is D6 (Hard)" may anchor perceived difficulty expectations
- Blinding allows unbiased perception ratings
- Tier assignments revealed only during debrief

**Blinding implementation:**

- Games labeled neutrally: "Game 1", "Game 2", etc.
- No difficulty labels shown in game UI during study session
- Post-game survey refers to "this game" not "this difficulty level"

**Exceptions:**

- After all games completed, participants rank games by perceived difficulty (blind ordering)
- Reveal tier assignments during debrief for participant interest

#### Order of Tiers

**Counterbalancing:**

- Each participant assigned to one of six orderings (see Design Type section)
- Assignment based on participant ID (mod 6)
- Track order assignments to ensure balance

**Within-tier game order:**

- First game at each tier: participant starts as Player 1
- Second game at each tier: participant starts as Player 2

### Post-Study

#### 1. Overall Summary Questionnaire (10 minutes)

Completed after all games, before tier reveal. See full survey in Data Collection section.

#### 2. Debrief and Thank-You (5 minutes)

- Reveal tier assignments for each game block
- Ask: "Does this match your expectations based on how the games felt?"
- Brief opportunity for additional verbal feedback (recorded for qualitative analysis)
- Confirm compensation/incentive delivery
- Invitation to participate in future studies

#### 3. Data Export and Anonymization

- Assign unique participant ID (e.g., P001, P002)
- Strip any identifying information from survey responses
- Link game records to participant ID only
- Store raw data in secure location with access controls

---

## Data Collection

### Quantitative Measures

| Measure                  | Source                  | Description             | Capture Timing |
| ------------------------ | ----------------------- | ----------------------- | -------------- |
| **Game outcome**         | Game engine             | Win / Loss / Draw       | Auto-recorded  |
| **Game length**          | Game engine             | Moves to completion     | Auto-recorded  |
| **Game duration**        | Game engine             | Clock time (seconds)    | Auto-recorded  |
| **Time per move**        | Game engine             | Average decision time   | Auto-recorded  |
| **Perceived difficulty** | Post-game survey        | Likert scale 1-5        | Immediate      |
| **Perceived fairness**   | Post-game survey        | Likert scale 1-5        | Immediate      |
| **Confidence**           | Post-game survey        | Before/after confidence | Immediate      |
| **Participant skill**    | Pre-study questionnaire | Self-reported segment   | Pre-study      |
| **Tier assignment**      | Study design            | D2/D4/D6/D8             | By design      |
| **Tier order**           | Study design            | Order A/B/C/D/E/F       | By design      |

### Qualitative Measures

| Measure                     | Source             | Description                                  |
| --------------------------- | ------------------ | -------------------------------------------- |
| **Open-ended feedback**     | Post-game survey   | "What made this game easy/hard?"             |
| **Difficulty drivers**      | Post-game survey   | Checklist of factors (speed, mistakes, etc.) |
| **Comparative ranking**     | Post-study survey  | Rank games by difficulty (blind)             |
| **Tier alignment**          | Post-study survey  | Which tier felt most appropriate?            |
| **Improvement suggestions** | Post-study survey  | Open text feedback                           |
| **Facilitator notes**       | Moderated sessions | Observed frustration, verbal comments        |
| **Key moments**             | Replay review      | Specific moves that surprised participant    |

### Survey Instruments

#### Post-Game Survey (2-3 minutes after each game)

**Administered immediately after game completion:**

---

**Q1. How difficult was this game?**

_(1-5 scale with labels)_

- 1 = Far too easy – I won without thinking
- 2 = Slightly easy – I felt in control throughout
- 3 = About right – Challenging but fair
- 4 = Slightly hard – I had to work hard
- 5 = Far too hard – I felt overwhelmed

---

**Q2. How fair did the AI feel?**

_(1-5 scale with labels)_

- 1 = Unfair in my favor – AI made obvious mistakes
- 2 = Slightly in my favor – AI seemed weaker than expected
- 3 = Fair and balanced – Felt like an even match
- 4 = Slightly against me – AI played very well
- 5 = Unfair against me – AI seemed impossibly strong

---

**Q3. Did you feel you had a chance to win this game?**

_(Single select)_

- Yes – I felt competitive throughout
- Somewhat – Only at certain points
- No – I felt outclassed from early on
- Unsure – Hard to tell

---

**Q4. What made this game easy or hard?**

_(Checkboxes – select all that apply)_

- [ ] AI decision speed (too fast / too slow)
- [ ] AI mistakes (too many / too few)
- [ ] AI positional understanding
- [ ] Surprising or tricky moves
- [ ] Predictable AI patterns
- [ ] My own lack of experience
- [ ] Nothing specific – just felt [easy/hard]

---

**Q5. Any specific observations about this game?**

_(Open text, optional)_

[Text field]

---

#### Post-Study Summary Survey (10 minutes)

**Administered after all games completed, before tier reveal:**

---

**Q1. Please rank the games you played from EASIEST to HARDEST.**

_(Drag-and-drop or numbered ranking)_

Games presented by their neutral labels: "Game 1", "Game 2", etc.

---

**Q2. Which game felt the most appropriately challenging for YOUR skill level?**

_(Single select from game list)_

- Game 1
- Game 2
- ...
- None – all felt misaligned with my skill

---

**Q3. Did any game feel TOO EASY for your skill level?**

_(Multi-select from game list)_

- Game 1
- Game 2
- ...
- None

---

**Q4. Did any game feel TOO HARD for your skill level?**

_(Multi-select from game list)_

- Game 1
- Game 2
- ...
- None

---

**Q5. Thinking across all games, did the difficulty curve feel smooth?**

_(Single select)_

- Yes – difficulty increased gradually
- Somewhat – some jumps felt too large
- No – difficulty felt inconsistent or random
- Not applicable – couldn't perceive differences

---

**Q6. What would improve the difficulty progression?**

_(Checkboxes – select all that apply)_

- [ ] Easier starting point
- [ ] Harder top difficulty
- [ ] More intermediate options between levels
- [ ] Clearer indication of what level I'm playing
- [ ] Better matching to my skill level
- [ ] Other (specify): [text field]

---

**Q7. How does the AI compare to human opponents you've played against?**

_(1-5 scale)_

- 1 = Much weaker than most humans I've played
- 3 = About the same as average human opponent
- 5 = Much stronger than most humans I've played
- N/A = Haven't played human opponents

---

**Q8. Would you play against AI at these difficulty levels again?**

_(Single select)_

- Definitely yes
- Probably yes
- Not sure
- Probably no
- Definitely no

---

**Q9. Any other suggestions or feedback?**

_(Open text)_

[Text field]

---

---

## Analysis Plan

### Quantitative Analysis

#### 1. Per-Tier Difficulty Rating Distribution

**Method:**

- Calculate mean, median, and SD of perceived difficulty (Q1) per tier
- Generate box plots showing distribution by tier

**Expected outcome:**
| Tier | Expected Mean (if calibrated) | Concern Threshold |
|------|------------------------------|-------------------|
| D2 | 2.0-2.5 | Mean >3.0 or <1.5 |
| D4 | 2.8-3.2 | Mean >3.8 or <2.2 |
| D6 | 3.2-3.8 | Mean >4.2 or <2.5 |
| D8 | 3.8-4.5 | Mean >4.8 or <3.0 |

#### 2. Tier Discrimination Testing

**Method:**

- Paired t-tests (or Wilcoxon signed-rank for non-normal distributions)
- Compare perceived difficulty between adjacent tier pairs:
  - D2 vs D4
  - D4 vs D6
  - D6 vs D8

**Hypothesis:**

- H₀: No significant difference in perceived difficulty between D(n) and D(n+2)
- H₁: D(n+2) perceived as significantly harder than D(n)
- α = 0.05, one-tailed (expecting higher difficulty in higher tier)

**Effect size calculation:**

- Cohen's d for each comparison
- Expected: d ≥ 0.5 (medium effect) for adjacent tiers
- Concern: d < 0.3 (small effect) suggests tiers not distinguishable

#### 3. Correlation Analysis

**Metrics to correlate:**

| Comparison                                | Expected Direction                 | Method         |
| ----------------------------------------- | ---------------------------------- | -------------- |
| Perceived difficulty vs actual win rate   | Negative (harder → lower win rate) | Pearson r      |
| Perceived difficulty vs game length       | Positive (harder → longer games)   | Pearson r      |
| Perceived fairness vs actual outcome      | Positive (wins → feels fair)       | Point-biserial |
| Perceived difficulty vs participant skill | Negative (skilled → feels easier)  | Pearson r      |

**Interpretation thresholds:**
| r value | Interpretation |
|---------|---------------|
| r > 0.5 | Strong alignment between perception and metric |
| 0.3 < r ≤ 0.5 | Moderate alignment |
| r ≤ 0.3 | Weak alignment (investigate discrepancy) |

#### 4. Skill-Tier Interaction Analysis

**Method:**

- Two-way mixed ANOVA:
  - Within-subjects factor: Tier (D2, D4, D6, D8)
  - Between-subjects factor: Skill segment (New, Intermediate, Strong)
  - DV: Perceived difficulty rating

**Key questions:**

- Is there a significant Tier × Skill interaction?
- Do skill segments perceive tier differences equally, or does perception differ by skill?

**Expected findings:**

- Main effect of Tier (higher tiers rated harder)
- Main effect of Skill (strong players rate all tiers as easier)
- Possible interaction if D8 is "too hard" only for non-strong players

**Stratified descriptives:**

- Mean perceived difficulty by tier, broken down by skill segment
- Identify specific tier-segment combinations with calibration issues

#### 5. Win Rate vs Target Band

**Method:**

- Calculate actual human win rate per tier
- Compare against target bands from [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:49)

**Target bands (reference):**
| Tier | Target Human Win Rate |
|------|----------------------|
| D2 | 30-50% |
| D4 | 30-70% |
| D6 | 40-60% |
| D8 | 25-45% |

**Concern flags:**

- Win rate >20% outside target band for intended segment
- 90%+ confidence interval excludes target range

### Qualitative Analysis

#### 1. Thematic Coding

**Process:**

- Two coders independently code open-text responses (Q5, post-study Q9)
- Develop initial codebook after first 10 participants
- Refine codes iteratively; calculate inter-rater reliability (Cohen's κ ≥ 0.7)

**Initial code categories:**

- **AI Behavior**: Speed, mistakes, patterns, strategic depth
- **Difficulty Perception**: Too easy, too hard, just right, inconsistent
- **Emotional Response**: Frustration, satisfaction, engagement, boredom
- **Improvement Suggestions**: Specific feature requests or concerns

#### 2. Tier-Specific Pattern Identification

**Method:**

- Group qualitative responses by tier
- Identify themes unique to or overrepresented in specific tiers

**Example findings format:**
| Tier | Common Themes | Concern Level |
|------|---------------|---------------|
| D2 | "Too passive", "Makes obvious mistakes" | Low – expected |
| D4 | "Feels fair", "Good practice opponent" | None |
| D6 | "Punishes mistakes harshly", "Very tactical" | Low – intended |
| D8 | "Frustrating", "Feels unbeatable" | Medium – investigate |

#### 3. Difficulty Driver Analysis

**Method:**

- Analyze Q4 checkbox data (difficulty drivers) by tier
- Identify which factors are most frequently cited at each tier

**Format:**
| Factor | D2 | D4 | D6 | D8 |
|--------|-----|-----|-----|-----|
| AI decision speed | 15% | 22% | 35% | 48% |
| AI mistakes | 65% | 32% | 12% | 5% |
| Positional understanding | 10% | 38% | 55% | 72% |
| Surprising moves | 8% | 25% | 42% | 58% |

#### 4. Actionable Insights Extraction

**For each tier, generate:**

1. **Summary**: One-paragraph assessment of calibration status
2. **Evidence**: Key quantitative and qualitative findings
3. **Recommendation**: Specific action (if any) for ladder tuning
4. **Confidence**: High/Medium/Low based on sample size and consistency

**Example output:**

> **D4 Summary**: Perceived difficulty mean of 2.9 (target: 2.8-3.2) indicates good calibration. Intermediate players achieve 58% win rate (target: 30-70%). Qualitative feedback is predominantly positive ("good challenge", "fair opponent"). No tuning recommended.
>
> **Confidence**: High (N=28 games, consistent findings)

### Analysis Output

#### 1. Summary Report

**Format:** Markdown document suitable for inclusion in `docs/ai/calibration_runs/`

**Contents:**

- Executive summary (1 page)
- Per-tier findings with visualizations
- Cross-tier comparison charts
- Calibration recommendations table
- Appendix: Raw data summaries

#### 2. Calibration Recommendations Table

| Tier | Status        | Key Finding                                      | Recommended Action                     | Priority |
| ---- | ------------- | ------------------------------------------------ | -------------------------------------- | -------- |
| D2   | On target     | Mean difficulty 2.3, win rate 45%                | No action                              | -        |
| D4   | Slightly easy | Mean difficulty 2.4, win rate 68%                | Consider strengthening candidate       | Medium   |
| D6   | On target     | Mean difficulty 3.5, win rate 52%                | No action                              | -        |
| D8   | Too hard      | Mean difficulty 4.7, many "frustrating" comments | Reduce difficulty or improve messaging | High     |

#### 3. Integration with Existing Processes

**Handoff to [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1):**

- Study report becomes input to calibration cycle Step 5 (Decide next actions)
- Recommendations incorporated into `notes.md` for the calibration run

**Update to [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1):**

- If study reveals new difficulty anchors or target bands, update guide accordingly
- If player segment definitions need refinement, update criteria

**Feed into [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:1):**

- Study establishes or validates perceived difficulty targets
- Informs alert thresholds for perceived difficulty metrics

---

## Logistics

### Timeline

| Phase               | Duration  | Description                                  |
| ------------------- | --------- | -------------------------------------------- |
| **Preparation**     | 1-2 weeks | Materials, recruitment setup, pilot          |
| **Pilot testing**   | 3-5 days  | Test protocol with 2-3 internal participants |
| **Recruitment**     | 1-2 weeks | Recruit and screen participants              |
| **Data collection** | 2-4 weeks | Run participant sessions                     |
| **Analysis**        | 1-2 weeks | Analyze data, code qualitative responses     |
| **Reporting**       | 1 week    | Produce summary report and recommendations   |

**Total estimated duration:** 6-10 weeks

**Milestone schedule:**

| Week | Milestone                                        |
| ---- | ------------------------------------------------ |
| 1    | Materials finalized, recruitment begins          |
| 2    | Pilot complete, adjustments made                 |
| 3-4  | Active recruitment, first sessions               |
| 5-6  | Primary data collection period                   |
| 7-8  | Data collection complete, analysis begins        |
| 9    | Analysis complete, draft report                  |
| 10   | Final report delivered, recommendations actioned |

### Resources Required

#### Personnel

| Role                                     | Responsibilities                                       | Time Commitment                  |
| ---------------------------------------- | ------------------------------------------------------ | -------------------------------- |
| **Study Coordinator**                    | Scheduling, participant communication, data management | 10-15 hrs/week during collection |
| **Facilitator** (for moderated sessions) | Guide participants, capture observations               | 2-3 hrs/session                  |
| **Data Analyst**                         | Quantitative analysis, statistics                      | 20-30 hrs total                  |
| **Qualitative Coder** (x2)               | Code open-text responses                               | 10-15 hrs each                   |
| **Report Author**                        | Synthesize findings, write report                      | 15-20 hrs                        |

_Note: Roles may overlap (e.g., Study Coordinator also does analysis)_

#### Technology

| Tool                   | Purpose                                     | Cost              |
| ---------------------- | ------------------------------------------- | ----------------- |
| **Survey platform**    | Host surveys (Google Forms, Typeform, etc.) | Free-$50/mo       |
| **Scheduling tool**    | Session scheduling (Calendly, etc.)         | Free-$15/mo       |
| **Video conferencing** | Moderated sessions (if remote)              | Free (Zoom, Meet) |
| **Data storage**       | Secure storage for responses                | Existing infra    |
| **Analysis tools**     | SPSS, R, Python, or Excel                   | Existing tools    |

#### Space/Equipment

- Quiet location for moderated sessions (if in-person)
- Participants use their own devices (or provide test devices if needed)

### Ethical Considerations

#### Informed Consent

- Clear explanation of study purpose and data use
- Explicit statement that participation is voluntary
- Right to withdraw at any time without explanation
- Contact information for questions or concerns

**Consent form template included in Appendix.**

#### Data Privacy and Anonymization

- Collect no personally identifying information beyond consent
- Assign participant IDs before data collection begins
- Store raw data with access controls
- Aggregate reporting only (no individual profiles)
- Data retention: Archive raw data for 2 years, then delete

#### Right to Withdraw

- Participants may stop at any time
- Partial data may be included (with consent) or excluded entirely
- No penalty for withdrawal

#### Minimizing Frustration

- Offer breaks between game blocks
- Remind participants they can skip D8 if feeling overwhelmed
- Debrief emphasizes study contribution, not personal performance

### Budget

| Item                      | Estimated Cost | Notes                       |
| ------------------------- | -------------- | --------------------------- |
| Participant incentives    | $300-900       | $10-20 × 30-45 participants |
| Survey platform (premium) | $0-50          | May use free tier           |
| Scheduling tool           | $0-15          | May use free tier           |
| Facilitator time          | Internal       | Existing team member        |
| Analysis time             | Internal       | Existing team member        |
| **Total**                 | **$300-965**   | Conservative estimate       |

_Note: Budget assumes small team; larger organizations may factor in additional overhead._

---

## Appendix

### A. Survey Templates

_(Full survey instruments provided in Data Collection section above)_

### B. Consent Form Template

---

**INFORMED CONSENT FORM**

**Study Title:** RingRift AI Difficulty Calibration Study

**Principal Investigator:** [Name, Contact]

**Purpose:**
This study aims to evaluate whether the difficulty levels of RingRift's AI opponents match player expectations. Your participation will help improve the game experience for all players.

**What You Will Do:**

- Complete a brief questionnaire about your gaming experience
- Play 6-9 games of RingRift against AI opponents at various difficulty levels
- Answer short surveys after each game
- Complete a summary survey at the end

**Time Required:** Approximately 60-90 minutes

**Risks:**
There are no significant risks anticipated. Some games may feel challenging, which may cause mild frustration. You may take breaks or stop at any time.

**Benefits:**
Your feedback will directly improve RingRift's AI difficulty calibration. You may also receive [compensation if applicable].

**Confidentiality:**
Your responses will be anonymized. No personally identifying information will be collected beyond this consent form. Data will be stored securely and accessed only by the research team.

**Voluntary Participation:**
Participation is voluntary. You may withdraw at any time without penalty.

**Contact:**
For questions or concerns, contact [email/contact info].

**Consent:**
By signing below (or clicking "I agree"), you confirm that you have read and understood this form and agree to participate.

---

Signature Date

---

### C. Session Facilitator Guide

_(For moderated sessions)_

**Before Session:**

1. Confirm participant has completed consent form
2. Verify technical setup (game loads, surveys accessible)
3. Review participant's pre-study questionnaire for skill segment
4. Assign tier order based on participant ID

**Session Script:**

> "Welcome, and thank you for participating in our study. Today you'll play several games of RingRift against AI opponents. After each game, I'll ask you to complete a short survey about how the game felt.
>
> The games will be at different difficulty levels, but I won't tell you which level each game is at – we'll reveal that at the end.
>
> Please play naturally, as you normally would. There are no right or wrong answers – we're interested in your honest perceptions.
>
> Do you have any questions before we start?"

**During Games:**

- Observe participant reactions (frustration, satisfaction, confusion)
- Note any technical issues
- Minimal interaction – let participant play naturally
- Offer breaks between blocks if participant seems fatigued

**After Each Game:**

- Prompt participant to complete post-game survey
- Note any verbal comments made during or after game

**End of Session:**

- Ensure all surveys completed
- Reveal tier assignments
  > "Now I can tell you which difficulty level each game was at. [Reveal order]"
- Ask: "Does this match what you expected based on how the games felt?"
- Record debrief comments
- Thank participant and confirm incentive delivery

### D. Related Documents

| Document                                                                           | Relationship                                          |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1)                         | Study results feed into calibration cycle decisions   |
| [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1)                 | Existing qualitative anchors and templates            |
| [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:1)     | Metrics and thresholds informed by study findings     |
| [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1) | Quantitative analysis design this study complements   |
| [`AI_LADDER_CHANGE_GUARDRAILS.md`](AI_LADDER_CHANGE_GUARDRAILS.md:1)               | Change governance for acting on study recommendations |

---

_This specification completes H-AI-17 of the AI difficulty calibration remediation track._
