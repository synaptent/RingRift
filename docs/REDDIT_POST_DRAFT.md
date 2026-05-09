# Reddit Post Draft

Target: `r/MachineLearning` or `r/gamedev`

## Draft

I spent 4 months having AI agents build an AlphaZero training system for a novel board game. Here's what happened.

The game is RingRift: a deterministic abstract strategy game on square and hex boards, loosely inspired by TZAAR, DVONN, and Go. Players place rings, move stacks, leave markers, form lines, collapse territory, and can win by ring elimination, territory control, or last player standing. You can try it at https://ringrift.ai.

The training stack is a minimal AlphaZero-style loop backed by a larger Python AI service and TypeScript rules engine. The current cluster uses 7 GH200 GPUs. The cleanest 2-player results are real but narrow: `hex8_2p` improved from `1500 -> 2241.7` Elo on the v5-heavy + fv3 reference lane with 13 promotions, and `square8_2p` improved from `1500 -> 1782.0` Elo with 5 promotions, including two consecutive candidates promoting at 62% win rate.

The failures were the interesting part. Vague prompts like "improve the codebase" created massive sprawl: at one point the Python service had about 17K Python files. A watchdog death spiral killed training jobs while trying to protect them. A P2P orphan detector fought systemd and killed healthy processes. The 3-player evaluator turned out to be unfair because candidate models were effectively tested 1-vs-2. A wrong learning-rate schedule killed learning for roughly 8 iterations before fixed LR became the breakthrough.

What worked was making the system smaller and more evidence-driven. The minimal loop is about 1000 lines and is now the reproducible proof harness. Result claims are tied to snapshots and archived metrics instead of hand-written status notes. The fixed-LR change was dramatic on `square8_2p`, and the v5-heavy + fv3 run is the first `hex8_2p` line to break above the older `1979.8` plateau.

My takeaway: AI agents can build useful systems from vague direction, but they also generate enormous surface area unless you add ratchets, tests, and hard source-of-truth boundaries. For long-running training projects, operational reliability mattered more than code quality because a beautiful refactor does not matter if GPUs are silently idle or killing each other.

I do not think this proves the game is "solved." Multiplayer training is still weak, large boards are immature, and the codebase still has scars. But the core experiment is now inspectable: self-play made the agents measurably stronger, and the game may be interesting enough that multiplayer AI remains nontrivial.

Play: https://ringrift.ai

Code: https://github.com/synaptent/RingRift

Results/evidence: https://github.com/synaptent/RingRift/blob/main/docs/RESULTS.md
