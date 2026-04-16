# DRAFT: Blog Post Outline — Building a Distributed ML Training System in 6 Weeks with Claude Code

> **Status: DRAFT** — This is exploratory narrative material, not the main public-facing project summary.
>
> For the current supported external-facing path, start with:
>
> - [README.md](/README.md)
> - [docs/PROJECT_BRIEF.md](/docs/PROJECT_BRIEF.md)
> - [docs/RESEARCH_SNAPSHOT.md](/docs/RESEARCH_SNAPSHOT.md)
> - [docs/RESULTS.md](/docs/RESULTS.md)

## The Hook

**Title options (ranked):**

1. "I Built 132 Daemons for a Game With Zero Users: A Cautionary Tale About AI-Assisted Development"
2. "Recreating AlphaZero in 6 Weeks: One Non-Programmer, 41 GPUs, and $585 in Accidental AWS Bills"
3. "My Neural Networks Started Getting Dumber: A Story About Distributed Training"
4. "The AI That Taught Itself to Lose: What 6 Weeks of 'Vibe Coding' Really Looks Like"

**Subtitle:** How a non-programmer used AI-assisted development to recreate Google's AlphaZero training pipeline, design a new abstract strategy game, and build a 41-node GPU cluster—then watched the AI train itself to play worse than the backup strategy.

**Opening paragraph (draft):**

> In 2016, DeepMind's AlphaZero team had hundreds of engineers, thousands of TPUs, and millions in funding. They created something that changed how we think about game AI.
>
> Eight years later, I wanted to know: could one person with a laptop and Claude Code do the same thing?
>
> I'm not a programmer. I couldn't write a for loop six weeks ago. But I had two goals: (1) recreate the AlphaZero training pipeline from scratch, and (2) invent a new game worth training it on—a perfect information zero-sum abstract strategy game designed to remain challenging for both humans and AI for decades to come.
>
> Today I have a 41-node GPU cluster running 132 daemons, 11 of which exist solely to recover from the failures of the other 121. The neural networks have successfully trained themselves to play _worse_ than the simple heuristic I wrote as a fallback. The production site was down for a week because I forgot to set an environment variable. And somewhere in AWS, there's an EC2 instance I've been paying $140/month for that I can't even SSH into.
>
> This is the story of democratizing what was once only possible for well-funded AI labs—and what I learned about building the wrong thing really, really well.

---

## Outline

### 1. The Setup (500 words)

**Who I am:**

- Not a programmer, not from a technical background
- [INSERT: What you actually do/did before this]
- Fascinated by AlphaZero since the 2017 paper
- Wondered: what would it take to replicate this today, starting from nothing?

**The five research hypotheses:**

This project tests five interconnected questions. Every outcome — success or failure — produces an interesting finding.

_Hypothesis 1: Can a non-developer design a genuinely ML-resistant pure strategy game?_

- Design a game with properties that should resist neural network optimization: multi-path victory, coalition dynamics, extreme branching, forced elimination cascades
- If AlphaZero-style training fails to produce superhuman play despite adequate data, the game design thesis is validated — interesting for game designers and AI safety researchers
- If training succeeds, the game still works as a product — interesting for different reasons

_Hypothesis 2: Can a non-developer build AlphaZero from scratch using frontier AI models?_

- Not just use an existing implementation — build the entire pipeline from scratch with Claude/GPT
- Self-play, neural network training, evaluation, model promotion, distributed across commodity GPUs
- The system's existence (and its bugs) is data for this hypothesis regardless of AI strength

_Hypothesis 3: Does the self-play loop actually work, and what does that tell us?_

- If models improve iteratively → Hypothesis 1 falsified, Hypothesis 2 validated
- If models stall despite correct pipeline and sufficient data → Hypothesis 1 validated
- As of March 2026: 1.9M games generated, pipeline recently verified end-to-end after 2 months of silent regression. The experiment is ongoing.

_Hypothesis 4: Can these results be presented in a way anyone cares about?_

- The story writes itself regardless of outcome: "solo non-dev builds AlphaZero with Claude" OR "novel game resists ML" OR "$20K of GPU time and 132 daemons for a game with zero users"
- Target: Hacker News, LessWrong, arxiv, game dev conferences

_Hypothesis 5: Can the experiment become a product people actually play?_

- The research narrative IS the marketing strategy
- The game, web client, multiplayer backend, and AI opponents all work today at ringrift.ai
- Users don't need superhuman AI — they need a fun game and an interesting story

---

### 1.5. Why Design a New Game? (800 words) — NEW SECTION

**The Design Philosophy**

Before I could train AlphaZero, I needed something worth training it on. Chess and Go are "solved" from a research perspective—we know the AlphaZero approach works on them. I wanted to create something new, something that might remain challenging for AI into the future.

This is the vision statement I wrote for RingRift:

> _"A cosmos to cleft. Structure birthing emergency."_
>
> _"A game about the tension between freedom and boundaries — between containment and creativity."_
>
> _"Tradeoffs and sacrifices intrinsic — strength and weakness interlaced, implying one another all the way through, from shallow to deep, top to bottom, beginning to end."_

Pretentious? Maybe. But it captures what I was trying to create: a game where complexity emerges from simple rules, where seeming victory can collapse into defeat, where short-term sacrifice is sometimes correct, and where the geometry of the board constantly reshapes incentives.

**The Human-Competitive Design Goal**

Here's what makes RingRift different from Chess or Go as an AI research target:

Unlike most perfect-information strategy games where engines quickly outclass humans, RingRift is explicitly designed so that **strong human players can compete with and sometimes outplay strong AIs**:

- **Multi-player dynamics**: The default 3-player configuration creates social and political dynamics (temporary alliances, leader-punishing behavior) that are genuinely difficult for purely algorithmic agents to model. These dynamics reward reading opponents and adapting on the fly—something humans do naturally.

- **Extreme branching factor**: The game has up to millions of legal choices per turn (compared to Chess's ~35 and Go's ~250). Combined with long tactical chains (captures can cascade through territory disconnections), this makes exhaustive search prohibitively expensive even for well-optimized engines.

- **Multiple victory paths that interact**: Three distinct ways to win (ring elimination, territory control, last-player-standing) create subtle tradeoffs that aren't easily reduced to a single value function.

The goal isn't to create a game AI can't solve—eventually it probably will. The goal is to create a game where the journey to solving it is interesting, and where humans can remain competitive for longer than with Chess or Go.

**The Uniqueness Analysis: ~65-70% Novel**

I did a systematic comparison against every abstract strategy game I could find. RingRift is best described as a "chimeric design" synthesizing mechanics from multiple families:

| Influence                              | What RingRift Borrowed                     |
| -------------------------------------- | ------------------------------------------ |
| **GIPF Project** (YINSH, DVONN, TZAAR) | Stacking, line formation, marker mechanics |
| **Go**                                 | Territory enclosure, pie rule              |
| **Checkers/Fanorona**                  | Mandatory chain capture continuation       |
| **Connect games**                      | Line detection triggers                    |
| **Amazons**                            | Movement leaves permanent markers          |

But the **specific combination** creates genuinely novel decision spaces:

| Novel Mechanic                             | Why It's New                                                                                                                           |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Cap height vs stack height for capture** | No other game distinguishes "consecutive top rings of your color" from "total rings" for capture resolution                            |
| **Marker trails that flip OR collapse**    | YINSH only flips markers; Amazons only burns squares. RingRift: opponent markers flip to your color, own markers collapse to territory |
| **Line → territory + forced elimination**  | YINSH removes line but no territory cost. RingRift creates permanent collapsed territory AND forces self-elimination                   |
| **Three integrated victory paths**         | Most abstracts have single or binary win conditions. Ring elimination, territory control, and last-standing interact                   |
| **Native 3-4 player support**              | Rare in serious abstract strategy games. Creates coalition dynamics                                                                    |

**No existing game combines all of:**

1. Stacking with cap-height-based capture (not total height)
2. Movement trails that transform based on owner (flip vs collapse)
3. Line formation triggering both territory creation AND forced elimination
4. Territory disconnection via marker enclosure
5. Three distinct victory paths
6. Native multi-player support

**The AI Research Testbed Thesis**

Beyond the game itself, RingRift is designed as a testbed for future AI algorithms:

- **Graph Neural Networks**: The board topology (especially hexagonal) maps naturally to GNNs. Stacks are nodes, adjacency defines edges. No CNN padding tricks needed.

- **Transformer architectures**: Multi-player dynamics with attention mechanisms could model coalition formation and threat assessment. Who's threatening whom? Who's likely to ally?

- **Coalition modeling**: 3-4 player variants require modeling temporary alliances—something current MCTS struggles with because it assumes adversarial play.

- **Long-horizon planning**: Territory cascades can ripple across the board. Predicting chain reactions requires genuine lookahead, not just local evaluation.

- **Transfer learning**: Four board geometries test whether learned strategies generalize. Does hex8 knowledge transfer to hexagonal? Does 2-player strategy transfer to 4-player?

This is why I didn't just train on Chess. I wanted to create something that might produce interesting research questions for years.

(Whether that research actually happens now depends on whether the AI ever learns to play better than my simple heuristic fallback. Currently, it's losing.)

---

**The naive starting point:**

- "I'll just train a neural network" (lol)
- Googled "how to make game AI", re-read the AlphaZero paper
- Thought: "The concepts seem straightforward"
- First reality check: you need training data from self-play
- Second reality check: self-play is slow without GPUs
- Third reality check: GPUs are expensive and distributed systems are hard
- Fourth reality check: designing a good game is its own massive challenge
- Fifth reality check: the neural network can absolutely learn the wrong thing

**The before-Claude attempts (what failed):**

- Tried ChatGPT: good for snippets, couldn't maintain context across a project
- Tried Copilot: autocomplete isn't architecture
- Tried learning Python properly: lost patience after week 2
- The breakthrough: Claude Code's ability to read entire codebases

**The 6-week timeline:**

- Week 1-2: Game engine parity (TypeScript + Python)
- Week 3-4: Neural network training pipeline
- Week 5-6: Distributed training cluster
- Week 6+: Refinement, 48h autonomous operation, documenting what went wrong

---

### 2. What I Actually Built (800 words)

**The numbers (a study in misplaced priorities):**

| Metric           | Value     | Comment                                                  |
| ---------------- | --------- | -------------------------------------------------------- |
| Python modules   | 341       | For a game that had zero active players                  |
| Test files       | 984       | Each test a monument to premature optimization           |
| Test coverage    | 99.5%     | Because untested code feels scary, unlike $585 AWS bills |
| Daemon types     | 132       | 116 active, 16 deprecated but never deleted              |
| Recovery daemons | 11        | To recover from the other 121 daemons' failures          |
| Event types      | 292       | I don't know what half of them do                        |
| Lines of code    | ~250,000+ | ~250,000 ways for something to go wrong                  |
| Active users     | ~0        | The only number that actually matters                    |

**The distributed training cluster:**

- 41 nodes across 7 providers (Lambda, Vast.ai, RunPod, Nebius, Vultr, Hetzner, AWS)
- ~1.5TB of GPU memory
- P2P mesh network with leader election
- Self-healing with 11 recovery daemons
- 48-hour autonomous operation
- Burning through compute budget faster than models improve

**The feedback loops that may or may not work:**

- Quality → Training adjustments (quality was low; training continued anyway)
- Elo velocity → Selfplay allocation (velocity was negative)
- Regression detection → Curriculum changes (detected regressions, didn't prevent them)
- Loss anomalies → Exploration triggers (triggered on normal training, halted on actual problems)
- Model promotion → Distribution (promoted models that lost to the heuristic)

**The game itself (RingRift):**

- Territory control game with placement and movement mechanics
- 4 board geometries: square8, square19, hex8, hexagonal (61-469 cells)
- 2-4 player support (multiplayer adds complexity beyond Chess/Go)
- 12 canonical trained models (one per board/player combination)
- GPU-accelerated selfplay (57x speedup)
- 100% parity between TypeScript (web) and Python (training) engines
- Actually quite fun to play (allegedly—no users to confirm)

**Why design a new game instead of using Chess/Go?**

- Chess and Go are "solved" from a research perspective—diminishing returns
- Multiplayer (3-4 players) creates coalition dynamics that don't exist in 2-player games
- Multiple board geometries test whether learned strategies generalize
- Designed with AI research in mind: clean state representation, well-defined action space
- Long-term goal: a testbed for graph neural networks, attention mechanisms, new RL algorithms
- Real reason: I thought it would be fun (it was, until the AI started losing)

---

### 3. How Claude Code Actually Works in Practice (1000 words)

**A real example (include actual transcript excerpt):**

```
Me: "The cluster keeps losing nodes after about 6 hours.
     They're marked as dead but they're actually still running."

Claude: *reads p2p_orchestrator.py, finds the heartbeat timeout logic*

Claude: "The issue is that the heartbeat timeout (90s) is shorter than
        the sync operation time during large transfers. When a node is
        syncing a big NPZ file, it can't respond to heartbeats, so it
        gets marked dead. I'll add a 'busy' state that extends the timeout
        during sync operations."

Me: "That makes sense. But won't that cause problems if a node is
     actually dead during a sync?"

Claude: "Good point. I'll add a separate 'last_activity' timestamp that
        tracks any activity, not just heartbeats. A node is only marked
        dead if both heartbeat AND activity are stale."

*writes 200 lines of code, adds 15 tests, all pass*

*node still gets marked dead 6 hours later due to a different bug*
```

This kind of back-and-forth happened hundreds of times. Each time, I learned a little more about distributed systems—not by reading textbooks, but by solving real problems that we often introduced ourselves.

**A less flattering real example:**

```
Me: "The training completed but Elo dropped from 1188 to 1141.
     Why are the models getting worse?"

Claude: *reads training logs, finds training completed successfully*

Claude: "Training looks fine. Loss decreased as expected."

Me: "But the models are weaker now."

Claude: *reads Elo logs, confirms regression*

Claude: "Let me check the training data..."

*2 hours later*

Claude: "The NPZ sync daemon was refreshing its local list of files
        but not actually pushing them to GPU nodes. The training has
        been using 2-week-old data. The models learned to master
        outdated strategies that no longer work."

Me: "So the self-improvement loop has been making the AI worse for
     two weeks and we just noticed?"

Claude: "Correct."
```

**The workflow that emerged:**

1. **Describe the problem in plain English**
   - "The training keeps stalling after 6 hours"
   - "Nodes are getting stuck and not recovering"
   - "I need the selfplay to prioritize configs that are falling behind"
   - "Wait, why are the models getting dumber?"

2. **Claude investigates the codebase**
   - Reads 20-50 files to understand context
   - Identifies existing patterns and conventions
   - Proposes solutions that fit the architecture
   - Often finds related issues I didn't know about
   - Sometimes misses issues that were there all along

3. **Iterative refinement**
   - "That's close but it should also consider X"
   - "Can you add tests for this?"
   - "This broke something else, can you fix it?"
   - "Actually, let's step back—is this the right approach?"
   - "Why didn't the tests catch this two weeks ago?"

4. **The compound effect**
   - Each session builds on previous work
   - Claude remembers the architecture via CLAUDE.md files
   - Patterns established in week 1 get reused in week 6
   - The codebase develops a consistent "voice"
   - Bugs established in week 1 also persist until week 6

**What surprised me:**

- The code quality is higher than I expected (99.5% test coverage happened naturally)
- It writes tests without being asked
- It catches its own mistakes and fixes them
- It suggests improvements I wouldn't have thought of
- It maintains consistent code style across 341 modules
- It can build systems I don't understand
- I don't understand most of what I built

**What's still hard:**

- Very complex debugging when the issue spans multiple distributed nodes
- When Claude's suggestions don't work, I often can't debug them myself
- Knowing when to push back vs. trust the suggestion
- Understanding _why_ something works (I often don't)
- The temptation to keep building instead of shipping
- Noticing when the system is actively getting worse

**The CLAUDE.md files (the secret sauce):**
I maintain context files that Claude reads at the start of each session. They contain:

- What the system does
- Key architectural decisions
- Current state of each component
- Known issues and gotchas
- Recent changes

This is the difference between "Claude helps me write code" and "Claude is my technical co-founder who remembers everything." It's also the difference between "we fixed that bug" and "we know exactly which 292 event types might be involved."

---

### 4. The Technical Highlights (800 words)

**P2P Mesh Network (no Kubernetes):**

- Custom leader election with SWIM protocol
- Gossip-based state synchronization
- Circuit breakers with 4-tier escalation
- Multi-transport failover (Tailscale → SSH → Base64 → HTTP)
- I don't fully understand any of this but it runs for 48 hours

**48-Hour Autonomous Operation:**

- Progress watchdog detects Elo stalls (after two weeks of not detecting anything)
- Memory pressure controller (60/70/80/90% tiers)
- Automatic model distribution after training
- Graceful degradation when nodes fail
- Ungraceful regression when everything succeeds

**GPU Selfplay Optimization:**

- Started with 80 `.item()` calls (CPU round-trips)
- Ended with 1 (statistics only)
- 57x speedup on RTX 5090
- 100% parity with TypeScript verified across 10K seeds
- Can now generate worse training data 57x faster

**The Daemon Architecture:**

- 132 daemon types managed by DaemonManager
- HandlerBase class for consistent lifecycle
- Event-driven coordination via EventRouter
- Health checks integrated with P2P status
- 11 recovery daemons for when the other 121 fail
- Still couldn't prevent models from regressing

**The Anomaly Detector (a case study in irony):**
Training has an anomaly detector that watches for unusual loss values. It's designed to halt training when something goes wrong. Here's what happened:

- Anomaly detector threshold: 3 standard deviations from mean
- Normal training loss: ~60.9
- "Anomalous" values: 65-67 (well within normal range for early epochs)
- What the detector did: Halted training after detecting "10 consecutive anomalies"
- What was actually anomalous: The training data was 2 weeks old
- Did the detector catch that: No

We disabled the anomaly detector to let training continue. The lesson: it's easier to detect unusual numbers than unusual situations.

---

### 5. Lessons Learned (600 words)

**About AI-assisted development:**

- It's not "AI writes code for you" - it's a collaboration
- The human still needs to understand the problem deeply (or at least notice when models get worse)
- Context management (CLAUDE.md files) is crucial
- Quality of prompts directly affects quality of output
- You can build things you don't understand (this is not always good)

**About building complex systems:**

- Start simple, let complexity emerge from real problems
- Tests aren't optional - they're how Claude knows if it broke something
- Documentation pays dividends (Claude reads it too)
- Don't over-engineer upfront, but do establish patterns early
- Monitoring what matters > monitoring everything
- "Works" and "works correctly" are different things

**About the gap between "working" and "production":**

- The last 20% takes 80% of the time
- Distributed systems fail in ways you can't predict
- Monitoring and observability aren't optional
- Cost management is a feature
- The system can degrade silently for weeks

---

### 6. The Failures and Mistakes (500 words)

**The things that went wrong (a comprehensive list would exceed the word limit):**

_The $585 AWS bill moment:_
I left an r5.4xlarge running "temporarily" for staging. It's been running for a month. I also spun up a "coordinator" instance that I couldn't even SSH into—wrong key, never configured. Just found it today, terminated it. That's $140/month doing literally nothing.

_The infrastructure-before-users trap:_
I built a 41-node GPU cluster for a game with approximately zero users. Classic over-engineering, except I'm not even an engineer. The AI made it so easy to keep building that I never stopped to ask if I should.

_The "it works on my machine" distributed edition:_
Spent 3 days debugging why training failed on cluster nodes. Turned out they didn't have Node.js installed, so the TypeScript parity tests couldn't run. The system helpfully marked all databases as "pending_gate" with no explanation.

_The production site was down for hours (it was actually a week):_
ringrift.ai was returning 500 errors on all assets. Root cause: `ALLOWED_ORIGINS` wasn't set in the production .env. The CORS middleware rejected requests from... ringrift.ai. Found and fixed it today, but it was broken for a week.

_The self-improvement loop that made things worse:_
The entire point of the AlphaZero approach is that the AI improves itself. Mine did the opposite. For two weeks, the NPZ sync daemon was "syncing" by refreshing its local file list without actually transferring files to GPU nodes. Training used 2-week-old data. The models dutifully learned to master outdated strategies. Current best model: 1141 Elo. The simple heuristic fallback: 1200 Elo. The AI successfully trained itself to be worse than the backup plan.

_The sandbox AI that couldn't beat a human:_
The public-facing game has an AI difficulty slider from 1-10. Difficulty 10 should be optimal play. The default was set to 5. At difficulty 5, the AI makes random moves 50% of the time. Users reported the AI was "too easy." They were right—it was literally playing at half strength by design.

**What I'd do differently:**

1. **Start with cost awareness** - Set up billing alerts from day 1. I have no idea what I've spent on Lambda, Vast.ai, and RunPod combined.

2. **Build user-facing features before infrastructure** - The game is playable, but basic. The training infrastructure is enterprise-grade. This is backwards.

3. **Monitor outcomes, not activities** - I knew every daemon was running. I didn't know the models were getting worse.

4. **Question scale earlier** - "Do I actually need 41 nodes?" The answer was always "more nodes = faster training" but faster training for what? A game nobody plays?

5. **Ship something in week 2** - I could have had a working game with a simple heuristic AI in two weeks. Instead, I spent six weeks building a training cluster.

---

### 7. The Honest Assessment (400 words)

**Is this the future of software development?**

- For certain types of projects: absolutely yes
- Non-programmers can build real systems now
- The barrier has fundamentally changed

**But also:**

- You still need taste and judgment
- You still need to understand the problem deeply
- AI-assisted dev makes it easy to build the wrong thing faster
- "Vibe coding" works until it doesn't—then you're stuck
- You can build systems that actively make things worse and not notice

**The uncomfortable truth:**
I built impressive infrastructure for a game that has zero users. The AI component that was supposed to be the whole point—the self-improving neural network—managed to train itself to play worse than a simple heuristic. This is a cautionary tale as much as an inspiring one. The tools are powerful, but they don't give you product sense, and they don't guarantee the system does what you think it does.

**The real metrics:**

| What I Thought Mattered  | What Actually Mattered |
| ------------------------ | ---------------------- |
| 99.5% test coverage      | 0 users                |
| 132 daemon types         | 0 users                |
| 41 GPU nodes             | 0 users                |
| 292 event types          | 0 users                |
| 48h autonomous operation | Still 0 users          |

**What's next:**

- The training pipeline was broken for 2 months (encoding mismatches, disabled validation, model corruption). As of March 2026, root causes are found and fixed. The experiment restarts now.
- If models improve over the next 1-2 weeks of clean iterations → the AlphaZero loop works, the game is solvable, Hypothesis 2 is validated
- If models stall despite correct pipeline and 1.9M games → the game genuinely resists ML optimization, Hypothesis 1 is validated, interesting for AI safety
- Either way, the game is playable today at ringrift.ai with heuristic + current NN opponents
- The research story is the marketing: "come play the game that [defeated/resisted] AlphaZero"

---

### 8. Conclusion (200 words)

**The takeaway:**
A non-programmer built production-grade distributed ML infrastructure in 6 weeks. This was impossible 2 years ago. The barrier to building software has fundamentally changed.

But—and this is important—building _working_ software is different from building _correct_ software. I built a sophisticated system that successfully trained AI to play worse. That's a new kind of failure that AI-assisted development makes possible.

**The real lesson:**
The tools are ready. The question is: ready for what? If you have a clear problem and good judgment about what to build, Claude Code (and tools like it) can 10x your output. If you don't, you can 10x your waste. And if you're not watching closely, you can build systems that confidently do the wrong thing.

**Call to action:**

- Try it yourself—start with a real problem you care about
- Ship something small before building infrastructure
- Don't aim for production on day one
- Let complexity emerge from real needs
- Monitor outcomes, not just activities
- Share what you build (and what you learn from the mistakes)

The barrier is gone. What matters now is knowing which barriers were actually load-bearing—and whether your AI is getting smarter or dumber.

---

## Appendix: Key Files to Reference

For technical readers who want to dig deeper:

- `ai-service/CLAUDE.md` - The context file that helps Claude understand the system
- `scripts/p2p_orchestrator.py` - The 26K LOC heart of the distributed system
- `app/coordination/daemon_manager.py` - How 132 daemons are managed
- `app/coordination/event_router.py` - The event-driven architecture
- `scripts/p2p/loops/` - The 22 background loops that keep things running

---

## Suggested Visuals

**Screenshots to capture:**

1. Terminal showing `pm2 status` with daemons running
2. P2P cluster status: `curl localhost:8770/status | jq`
3. The game board itself (with pieces, mid-game)
4. A CLAUDE.md file showing context management
5. Test output showing 99.5% coverage
6. Architecture diagram (simplify the ASCII ones for clarity)
7. AWS Cost Explorer showing the $585 month
8. Elo chart showing the regression (1188 → 1141)
9. The anomaly detector log showing "10 consecutive anomalies" for normal data

**Code snippets to highlight:**

1. A single CLAUDE.md excerpt showing how context is structured
2. The daemon architecture (how 132 daemons are defined)
3. A circuit breaker in action
4. The event flow for training completion → model promotion
5. The NPZ sync daemon that wasn't syncing

---

## Target Publications

| Platform              | Angle                                 | Format                          |
| --------------------- | ------------------------------------- | ------------------------------- |
| **Hacker News**       | "Built 132 daemons for 0 users"       | Self-post, link to blog         |
| **r/MachineLearning** | "My AlphaZero trained itself to lose" | Cross-post with technical focus |
| **r/ClaudeAI**        | The Claude Code workflow              | Cross-post with tool focus      |
| **Twitter/X**         | Thread with screenshots               | See TWITTER_THREAD.md           |
| **Substack/Blog**     | Full story                            | Primary publication             |
| **Anthropic Discord** | Case study                            | Shorter version with link       |
| **LinkedIn**          | Professional angle                    | "What I learned building..."    |

**Timing:**

- Publish blog first (control the narrative)
- HN submission 2-3 hours later
- Twitter thread same day
- Cross-posts next day for extended reach
