# Reddit Post Draft

## Draft

I built a novel board game, then used AI coding agents to help build an AlphaZero-style training stack for it. The result is a mix of real progress and hard lessons.

The game is RingRift: a territory-control board game played on square or hex boards. Players place rings, stack and move them, form lines, capture, and win by ring elimination, territory control, or last-player-standing conditions. The rules are unusual enough that a hand-written heuristic is not obviously dominant.

The training result so far: iterative self-play improved square8_2p from 1500 to 1782 Elo over five promotions, including two back-to-back 62% win-rate promotions. hex8_2p improved from 1500 to about 1980 Elo before plateauing. A v4 architecture experiment is now running to test whether the hex plateau is model capacity/architecture rather than game saturation.

The messy part: I initially gave agents broad prompts like "make the codebase better." That produced useful work, but also massive sprawl: at one point the Python AI service had roughly 17K Python files, much of it duplicate or archived agent output. We also lost time to infrastructure failures: watchdogs that killed the jobs they were supposed to protect, orphan-process detectors fighting systemd, and self-play data that looked valid until we discovered it lacked the MCTS policy targets needed for training.

What worked was narrowing the system down to a minimal reproducible training loop, adding ratchet tests around codebase boundaries, and treating results as evidence-backed artifacts instead of marketing claims. The repo now has reproducibility docs, result snapshots, and a live sandbox where you can play against the current AI or watch AI-vs-AI games.

The project is not "solved." Multiplayer training is still weak, the hex model is plateaued, and the codebase still carries scars from agent-driven iteration. But the core claim is now real enough to inspect: self-play on this game produces measurable improvement, and the game is playable in a browser.

Play: https://ringrift.ai

Code: https://github.com/synaptent/RingRift

I would be interested in feedback on two things: whether the game itself looks strategically interesting, and whether the engineering lessons match what others are seeing when using coding agents on long-running projects.
