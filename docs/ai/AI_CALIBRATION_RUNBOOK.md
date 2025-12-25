# AI Difficulty Calibration Runbook – Square‑8 2‑Player (H‑AI‑14)

&gt; **Status (2025‑12‑05): New – H‑AI‑14 runbook.**  
&gt; **Role:** Step‑by‑step operational guide for running AI difficulty calibration cycles for the Square‑8 2‑player ladder tiers D2 / D4 / D6 / D8, using the analysis design in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1) and the CLI in [`analyze_difficulty_calibration.py`](../../ai-service/scripts/analyze_difficulty_calibration.py:1).

---

## 1. Purpose and scope

- This runbook is part of the remediation track for the **hardest problem** identified in [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](../archive/assessments/WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:145): _advanced AI strength and a stable high‑tier Square‑8 2‑player ladder_.
- It defines the **operational procedure** for running an AI difficulty calibration cycle:
  - choosing a calibration window;
  - assembling required inputs (telemetry aggregates, registry, eval/perf artefacts);
  - invoking [`analyze_difficulty_calibration.py`](../../ai-service/scripts/analyze_difficulty_calibration.py:1);
  - storing outputs under `docs/ai/calibration_runs/`;
  - capturing human notes and decisions.
- It **does not** redefine calibration theory, metrics, or thresholds. Those live in:
  - [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1) – analysis design and decision rules;
  - [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1) – human‑study templates;
  - the script implementation [`analyze_difficulty_calibration.py`](../../ai-service/scripts/analyze_difficulty_calibration.py:1).

Primary users:

- AI/ML engineers responsible for the Square‑8 2‑player ladder.
- Operators or data engineers asked to run calibration on demand.

---

## 2. Prerequisites

### 2.1 Environment assumptions

- You have a working checkout of the RingRift repo and `ai-service` subproject.
- Python is installed with a version compatible with `ai-service` (see [`requirements.txt`](../../ai-service/requirements.txt:1)).
- Dependencies are installed as described in [`ai-service/README.md`](../../ai-service/README.md:1) (for example via `pip install -r ai-service/requirements.txt` or the project’s bootstrap scripts).
- You can run Python modules from the project root, e.g.:

  ```bash
  python -m ai-service.scripts.analyze_difficulty_calibration --help
  ```

### 2.2 Access requirements

You must be able to:

- Read the **tier candidate registry**:
  - [`tier_candidate_registry.square8_2p.json`](../../ai-service/config/tier_candidate_registry.square8_2p.json:1)
- Read **tier evaluation and perf artefacts** for recently promoted Square‑8 2‑player candidates:
  - directories referenced by the registry’s `source_run_dir` fields, under some shared `--eval-root` (for example `ai-service/logs/`), containing:
    - `tier_eval_result.json`
    - `tier_perf_report.json` (for D4 / D6 / D8)
    - `gate_report.json`
- Access the **calibration aggregates JSON** for the chosen window (see next subsection).

No special infra rights are required beyond read access to these files and the ability to run offline Python scripts.

### 2.3 Required inputs for a calibration window

For a **single calibration window** on Square‑8 2‑player you need three things.

#### 2.3.1 Calibration aggregates JSON

A JSON file exported from metrics or a data warehouse, containing **pre‑aggregated calibration metrics** for the window, constrained to:

- `board = "square8"`
- `num_players = 2`
- `difficulty ∈ {2,4,6,8}`
- Calibration cohort only (e.g. `isCalibrationOptIn = true`)

The file must match the schema validated by [`load_calibration_aggregates()`](../../ai-service/scripts/analyze_difficulty_calibration.py:130) in [`analyze_difficulty_calibration.py`](../../ai-service/scripts/analyze_difficulty_calibration.py:1). Conceptually:

```json
{
  "board": "square8",
  "num_players": 2,
  "window": {
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-28T23:59:59Z"
  },
  "tiers": [
    {
      "tier": "D4",
      "difficulty": 4,
      "segments": [
        {
          "segment": "intermediate",
          "n_games": 90,
          "human_win_rate": 0.72,
          "difficulty_mean": 2.4,
          "difficulty_p10": 1.8,
          "difficulty_p90": 3.2
        }
      ]
    }
  ]
}
```

Notes:

- `tier` must be one of `"D2"`, `"D4"`, `"D6"`, `"D8"`.
- `difficulty` must numerically match the tier (e.g. `"D4"` → `4`).
- Fields are per **segment** (e.g. `new`, `intermediate`, `strong`) as described in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:244).

How this JSON is produced (PromQL, SQL, batch job) is outside this runbook; treat its shape as a **contract**.

#### 2.3.2 Tier candidate registry

The **Square‑8 2‑player tier candidate registry**:

- Default path: [`tier_candidate_registry.square8_2p.json`](../../ai-service/config/tier_candidate_registry.square8_2p.json:1)
- Canonical loader: [`load_square8_two_player_registry()`](../../ai-service/app/training/tier_promotion_registry.py:29)

This registry tells the calibration script:

- Which **model id** is currently live for each tier.
- Which `gated_promote` candidates and `source_run_dir` paths to inspect for evaluation/perf artefacts.

#### 2.3.3 Evaluation/perf artefacts under an eval root

An **eval root directory** containing gating runs, referred to by `source_run_dir` in the registry. Typical layout (example):

```text
ai-service/logs/
  tier_gate/
    sq8_2p/
      D2/2025-11-10T09-00-00Z/
        tier_eval_result.json
        gate_report.json
      D4/2025-11-15T14-15-00Z/
        tier_eval_result.json
        tier_perf_report.json
        gate_report.json
      D6/2025-11-20T10-05-00Z/
        tier_eval_result.json
        tier_perf_report.json
        gate_report.json
      D8/2025-11-25T18-30-00Z/
        tier_eval_result.json
        tier_perf_report.json
        gate_report.json
```

- At minimum, each **current ladder model** for D2 / D4 / D6 / D8 should have:
  - `tier_eval_result.json` under its `source_run_dir`.
- For perf‑budgeted tiers (D4 / D6 / D8), `tier_perf_report.json` is expected.
- `gate_report.json` is optional but recommended for operator context.

The calibration CLI is pointed at the `ai-service/logs/` root via `--eval-root`; it resolves per‑tier directories via the registry.

---

## 3. End‑to‑end procedure for a calibration cycle (Square‑8 2‑player)

This section describes a **single calibration cycle** for Square‑8 2‑player tiers D2 / D4 / D6 / D8.

### 3.1 Step 1 – Select calibration window and tiers; create run directory

1. **Choose the time window.**
   - Default recommendation: the last **28 days** of calibration activity, or “since the previous calibration run”.
   - Ensure:
     - Sufficient sample sizes per tier/segment (see thresholds in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:289)).
     - No major mid‑window ladder changes if possible.

2. **Confirm tiers in scope.**
   - Board: `square8` (8×8 compact ruleset).
   - Players: `2`.
   - Difficulty tiers: `D2`, `D4`, `D6`, `D8`, matching the ladder scope in [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:19).

3. **Create a calibration run directory.**
   - Under `docs/ai/calibration_runs/`, create a directory using the convention:

     ```text
     docs/ai/calibration_runs/YYYY_MM_square8_2p_windowNN/
     ```

     Example:

     ```text
     docs/ai/calibration_runs/2025_12_square8_2p_window01/
     ```

4. **Seed the run with the notes template.**
   - Copy the per‑run template [`TEMPLATE.md`](calibration_runs/TEMPLATE.md:1) into the new directory as `notes.md`:

     ```bash
     cp docs/ai/calibration_runs/TEMPLATE.md \
        docs/ai/calibration_runs/2025_12_square8_2p_window01/notes.md
     ```

   - You will fill `notes.md` during and after the run (see §4).

### 3.2 Step 2 – Collect and normalise inputs

1.  **Export calibration aggregates JSON.**
    - Use your metrics/query tooling to export a JSON file like §2.3.1, filtered to the chosen time window and to:
      - `board="square8"`, `num_players=2`, `difficulty ∈ {2,4,6,8}`
      - calibration cohort only (`isCalibrationOptIn=true` or equivalent)
    - Save it under the run directory, for example:

           ```text
           docs/ai/calibration_runs/2025_12_square8_2p_window01/\

      aggregates.square8_2p.window01.json

      ```

      ```

2.  **Snapshot the tier candidate registry (optional but recommended).**
    - The script can read directly from the live registry:
      - [`tier_candidate_registry.square8_2p.json`](../../ai-service/config/tier_candidate_registry.square8_2p.json:1)

    - For auditability, take a snapshot into the run directory:

           ```bash
           cp ai-service/config/tier_candidate_registry.square8_2p.json \
              docs/ai/calibration_runs/2025_12_square8_2p_window01/\

      tier_candidate_registry.square8_2p.snapshot.json

      ```

      ```

3.  **Choose `--eval-root` and verify tier directories.**
    - Decide which directory root will be used for evaluation/perf artefacts, e.g.:

      ```text
      --eval-root ai-service/logs
      ```

    - Under this root, each `source_run_dir` from the registry should resolve to a directory that contains, at minimum:

      ```text
      &lt;eval-root&gt;/&lt;source_run_dir&gt;/tier_eval_result.json
      &lt;eval-root&gt;/&lt;source_run_dir&gt;/tier_perf_report.json   # D3–D8
      &lt;eval-root&gt;/&lt;source_run_dir&gt;/gate_report.json
      ```

    - These artefacts are typically produced by:
      - [`run_tier_gate.py`](../../ai-service/scripts/run_tier_gate.py:1) and/or
      - the combined wrapper [`run_full_tier_gating.py`](../../ai-service/scripts/run_full_tier_gating.py:1),

      as described in [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:285).

4.  **Record inputs in `notes.md`.**
    - In the run’s `notes.md` (from [`TEMPLATE.md`](calibration_runs/TEMPLATE.md:1)), fill the **Inputs** section with:
      - path to the calibration aggregates JSON;
      - path to the registry (and snapshot, if created);
      - chosen `--eval-root`.

### 3.3 Step 3 – Run the calibration analysis CLI

From the **project root** (`/Users/armand/Development/RingRift`), run a command along the lines of:

```bash
python -m ai-service.scripts.analyze_difficulty_calibration \
  --calibration-aggregates \
    docs/ai/calibration_runs/2025_12_square8_2p_window01/\
aggregates.square8_2p.window01.json \
  --registry-path ai-service/config/tier_candidate_registry.square8_2p.json \
  --eval-root ai-service/logs \
  --output-json \
    docs/ai/calibration_runs/2025_12_square8_2p_window01/\
calibration_summary.json \
  --output-md \
    docs/ai/calibration_runs/2025_12_square8_2p_window01/\
calibration_summary.md \
  --window-label 2025-12-square8-2p-window01
```

Key flags (see [`parse_args()`](../../ai-service/scripts/analyze_difficulty_calibration.py:726)):

- `--calibration-aggregates` – path to the aggregates JSON exported in §3.2.1.
- `--registry-path` – path to the Square‑8 2p registry JSON (defaults to [`DEFAULT_SQUARE8_2P_REGISTRY_PATH`](../../ai-service/app/training/tier_promotion_registry.py:37)).
- `--eval-root` – root directory that prefixes all `source_run_dir` entries from the registry.
- `--output-json` – where to write the machine‑readable calibration summary JSON (recommended: inside the run directory).
- `--output-md` – where to write the human‑readable Markdown summary (also inside the run directory).
- `--window-label` – short label for the window (used in the report header); match the run directory name when possible.
- `--min-sample-size` – minimum `n_games` for a segment to be “well sampled”; default is `30` as in the analysis spec.

On success, the script prints paths where the JSON and Markdown summaries were written. On error, see the message for whether the issue is input validation, missing files under `--eval-root`, or registry problems.

### 3.4 Step 4 – Interpret the outputs

After running the CLI, the run directory should contain:

- `calibration_summary.json`
- `calibration_summary.md`

#### 3.4.1 JSON summary

Shape is defined by [`build_calibration_summary()`](../../ai-service/scripts/analyze_difficulty_calibration.py:576). At a high level it contains:

- `board`, `num_players`
- `window` – including `start`, `end`, and `label`
- `tiers[]` – one entry per tier, each with:
  - `tier`, `difficulty`
  - `ladder` – current ladder model id, AI type, heuristic profile (from [`get_ladder_tier_config`](../../ai-service/app/config/ladder_config.py:279)).
  - `registry` – `current` registry block and `latest_candidate` metadata.
  - `evaluation` – `overall_pass` and win‑rates vs baseline / previous tier, derived from `tier_eval_result.json` when present.
  - `perf` – `overall_pass`, `avg_ms`, `p95_ms`, derived from `tier_perf_report.json` when present.
  - `calibration` – per‑segment entries and an `overall_status` with notes.

This JSON is intended for:

- automated consumers (dashboards, CI, future monitoring in H‑AI‑15);
- comparing multiple calibration windows programmatically.

#### 3.4.2 Markdown summary

The Markdown is produced by [`build_markdown_report()`](../../ai-service/scripts/analyze_difficulty_calibration.py:623) and is the primary **human‑facing report** for this run. For each tier it includes:

- **Ladder model**: model id, AI type, heuristic profile.
- **Evaluation summary**: gate status (`PASS`/`FAIL`) and win‑rates vs baseline / previous tier, if available.
- **Perf summary**: perf budget pass/fail and observed `avg_ms`, `p95_ms`, if the tier has a perf budget (D3–D8).
- **Segment table**: rows per segment with:
  - `n_games`
  - `human_win_rate`
  - `difficulty_mean`, `difficulty_p10`, `difficulty_p90`
  - `sample_ok` flag
  - `status` in `{too_easy, too_hard, in_band, inconclusive}`
- **Overall calibration status** for the tier and short notes summarising the segment‑level picture.

How to interpret `too_easy` / `too_hard` / `in_band` is defined precisely in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:443). Use that document as the **normative reference** when turning this report into actions.

### 3.5 Step 5 – Decide next actions

For each tier, use `calibration_summary.md` together with the automation context to decide what to do next:

Inputs to consider:

- Calibration status and notes per tier/segment (from `calibration_summary.*`).
- Automated evaluation and perf status (from `evaluation` and `perf` blocks).
- Ladder and training context from:
  - [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:1)
  - [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1)

Typical next actions (to record in `notes.md`):

- **Tier appears too easy** for its intended segment and passes gates/perf:
  - Prioritise a stronger candidate for that tier in the next training cycle.
  - Optionally tighten future gate expectations as described in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:500).
- **Tier appears too hard** for its intended segment and passes gates/perf:
  - Consider weakening the tier (e.g. reduced search depth, increased randomness) and then re‑running gating/perf.
  - In extreme cases, consider remapping this model to a higher logical difficulty and re‑labeling UX.
- **Data inconclusive** (low `n_games` or mixed signals):
  - Schedule additional calibration telemetry collection or structured human sessions using templates A/B/C from [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:137).
- **Automated gates or perf fail**:
  - Treat this primarily as an H‑AI‑9/H‑AI‑8 issue; calibration may still be informative but should not drive promotions that violate the invariants documented in [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:287) and [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1).

Record:

- A per‑tier summary (e.g. “D4 too easy for intermediates; queue stronger candidate”, “D2 acceptable; monitor only”).
- Concrete follow‑ups (tickets, planned training runs, future calibration windows).

These decisions become the input to later guardrail and monitoring specs (H‑AI‑15/H‑AI‑16).

---

## 4. Artefact layout and naming conventions

All **human‑readable calibration runs** for Square‑8 2‑player should live under:

```text
docs/ai/calibration_runs/
```

### 4.1 Directory naming

Use:

```text
YYYY_MM_square8_2p_windowNN
```

Where:

- `YYYY_MM` is the year and month containing the **end** of the calibration window.
- `windowNN` is a two‑digit counter for that month (`01`, `02`, …).

Examples:

- `2025_11_square8_2p_window01`
- `2025_12_square8_2p_window02`

### 4.2 Expected files per run

Inside each run directory, keep at least:

```text
docs/ai/calibration_runs/YYYY_MM_square8_2p_windowNN/
  notes.md                                  # from TEMPLATE.md, filled by operator
  aggregates.square8_2p.windowNN.json       # calibration aggregates input
  tier_candidate_registry.square8_2p.snapshot.json   # optional snapshot
  calibration_summary.json                  # JSON output from the CLI
  calibration_summary.md                    # Markdown output from the CLI
```

Optional additions:

- `attachments/` – links or pointers to key gating/perf run dirs.
- `extra_notes.md` – extended analysis or experiment design if needed.

### 4.3 Template usage

The per‑run template [`TEMPLATE.md`](calibration_runs/TEMPLATE.md:1):

- Provides a **consistent skeleton** for `notes.md`:
  - Run metadata and window label.
  - Input paths.
  - Exact commands run.
  - Summary of results per tier.
  - Decisions and follow‑ups.
- Should be treated as **canonical** for operator notes:
  - Copy it into each new run directory before starting.
  - Keep it updated during the run (not just at the end).

---

## 5. Roles and sign‑offs (lightweight)

This runbook assumes lightweight role responsibilities; H‑AI‑16 will define more formal guardrails.

- **AI / Ladder Owner (AI/ML engineer)**
  - Runs or supervises the calibration cycle.
  - Ensures the right inputs (aggregates, registry, eval/perf artefacts) are present.
  - Interprets calibration outputs in the context of training and gating.

- **Data / Telemetry Owner**
  - Produces and validates the calibration aggregates JSON.
  - Confirms that schema changes in telemetry or exports do not break [`analyze_difficulty_calibration.py`](../../ai-service/scripts/analyze_difficulty_calibration.py:1).

- **Product / Game Design Owner**
  - Interprets calibration outcomes relative to target player experience.
  - Decides when to adjust UX difficulty descriptors or surfaced tiers.

- **Release / Operations Owner**
  - Ensures ladder changes driven by calibration follow deployment practices and have rollback plans.

**Sign‑off expectation for a calibration run:**

- AI / Ladder Owner:
  - Confirms that inputs are correct.
  - Runs the CLI.
  - Fills in `notes.md` and proposes next actions.
- Data / Telemetry Owner:
  - Signs off that aggregates are valid for the stated window.
- Product / Game Design Owner:
  - Reviews `calibration_summary.md` and notes.
  - Agrees on any player‑facing difficulty or ladder changes to be queued.

---

## 6. Quick operator checklist

Use this as a compact checklist when running calibration:

1. **Plan**
   - [ ] Pick calibration window (e.g. last 28 days) and confirm anchor tiers D2/D4/D6/D8 in scope (expand if calibrating other tiers).
   - [ ] Create `docs/ai/calibration_runs/YYYY_MM_square8_2p_windowNN/`.
   - [ ] Copy [`TEMPLATE.md`](calibration_runs/TEMPLATE.md:1) → `notes.md`.

2. **Gather inputs**
   - [ ] Export calibration aggregates JSON into the run directory.
   - [ ] (Optional) Snapshot [`tier_candidate_registry.square8_2p.json`](../../ai-service/config/tier_candidate_registry.square8_2p.json:1) into the run directory.
   - [ ] Choose `--eval-root` and verify required `tier_eval_result.json` / `tier_perf_report.json` / `gate_report.json` exist for current ladder models.

3. **Run analysis**
   - [ ] Run [`analyze_difficulty_calibration.py`](../../ai-service/scripts/analyze_difficulty_calibration.py:1) with `--calibration-aggregates`, `--registry-path`, `--eval-root`, `--output-json`, `--output-md`, and a `--window-label`.
   - [ ] Confirm `calibration_summary.json` and `calibration_summary.md` were created in the run directory.

4. **Review and decide**
   - [ ] Read `calibration_summary.md` tier by tier.
   - [ ] Fill **Summary of results** and **Decisions &amp; follow‑ups** in `notes.md`.
   - [ ] File or update tickets / plans for any required training, gating, or UX changes.

5. **Archive**
   - [ ] Treat the run directory as the permanent record for that calibration window.
   - [ ] Ensure future guardrail/monitoring specs (H‑AI‑15/16/17) reference these artefacts as needed.
