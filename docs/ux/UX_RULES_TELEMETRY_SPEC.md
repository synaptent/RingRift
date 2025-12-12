# RingRift Rules UX Telemetry Specification

> **Doc Status (2025-12-05): Partially implemented – core counters and a first event subset are wired; the remaining envelope and event types are a design contract for future UX work.**
>
> **Role:** Defines a minimal but expressive telemetry schema for observing rules‑related confusion, help usage, and weird‑state UX outcomes across the RingRift client surfaces (HUD, VictoryModal, TeachingOverlay, Sandbox, FAQ / docs).
>
> **Inputs:** Canonical rules semantics and edge‑case analysis from [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:181), [`ringrift_complete_rules.md`](ringrift_complete_rules.md:591), [`docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:1), and [`docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:184) plus UX findings in [`docs/supplementary/RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:23).
>
> **Downstream:** Implementation tasks in Code mode should treat this file and [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1), [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1), and [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:1) as the contract for wiring telemetry and metrics.

---

## 1. Objectives and Scope

This spec defines:

- The canonical **Rules UX telemetry envelope** (`RulesUxEvent`) and common fields.
- A small, opinionated set of **event types** focused on rules understanding, weird states (ANM / forced elimination / structural stalemate), and teaching flows.
- **Surface → event mappings** that state which UI components emit which events and how they choose `rules_context`.
- **Metric names and labels** for Prometheus / backend aggregation.
- A handful of **“hotspot” queries** for identifying confusing rules contexts.
- **Privacy and retention expectations** for this class of telemetry.

Non‑goals:

- General gameplay telemetry (move timing, MMR, AI strength, etc.).
- Server‑side engine / AI metrics (covered by [`ai-service/app/metrics.py`](ai-service/app/metrics.py:1) and allied docs).
- Detailed UX event sequencing beyond what is needed to infer rules confusion.

All identifiers and shapes here are **language‑agnostic**. Concrete type definitions in TypeScript / Python / DB schemas must mirror this structure but may add internal fields if they do not affect metrics cardinality.

---

## 2. Event Envelope: `RulesUxEvent`

Every rules‑UX event MUST conform to a shared envelope. Pseudotype:

```ts
type RulesUxEvent = {
  event_type: RulesUxEventType;
  rules_context?: RulesContext;
  source: RulesUxSource;

  // Game / rules state context (when applicable)
  game_id?: string; // short id; may be omitted for pure sandbox / docs views
  board_type?: 'square8' | 'square19' | 'hexagonal';
  num_players?: 2 | 3 | 4;
  difficulty?: 'tutorial' | 'casual' | 'ranked_low' | 'ranked_mid' | 'ranked_high';
  ai_profile?: string; // e.g. 'descent_v3', 'heuristic_easy'; low-cardinality config key
  is_ranked?: boolean;
  is_calibration_game?: boolean; // e.g. Square‑8 2p ladder calibration
  is_sandbox?: boolean;

  // Player / seat context (no PII)
  seat_index?: 1 | 2 | 3 | 4; // seat of the actor or viewer, if applicable
  perspective_player_count?: 1 | 2 | 3 | 4; // number of humans in this game from this client

  // Time & client context
  ts: string; // ISO‑8601 UTC timestamp
  client_build: string; // short build / git sha
  client_platform: 'web' | 'mobile_web' | 'desktop';
  locale?: string; // e.g. 'en-US'
  session_id?: string; // random, per‑device/session UUID; NOT user id

  // Correlation IDs for multi‑step interactions
  help_session_id?: string; // reused across open → topic_view → reopen
  overlay_session_id?: string; // reused across weird‑state overlay interactions
  teaching_flow_id?: string; // reused across a teaching flow, see §4

  // Event‑specific payload (see §3)
  payload?: Record<string, unknown>;
};
```

### 2.1 Enumerations

`RulesUxEventType` (non‑exhaustive, but all Code‑mode additions must be documented here):

- `help_open` – user opens any rules help surface (global rules, FAQ, context help).
- `help_topic_view` – user views a specific topic / FAQ / rules anchor.
- `help_reopen` – user reopens help within a short window after closing.
- `weird_state_banner_impression` – HUD banner for a weird / complex rules state is shown.
- `weird_state_details_open` – user opens an expanded explanation (e.g. from VictoryModal “What happened?” link).
- `weird_state_overlay_shown` – TeachingOverlay auto‑launched for a weird state.
- `weird_state_overlay_dismiss` – TeachingOverlay dismissed (with or without completing flow).
- `resign_after_weird_state` – player resigns / abandons shortly after a weird‑state banner or overlay.
- `sandbox_scenario_loaded` – curated rules scenario loaded into sandbox.
- `sandbox_scenario_completed` – curated scenario successfully completed.
- `teaching_step_started` – scenario‑driven teaching step begins.
- `teaching_step_completed` – teaching step objective met.
- `doc_link_clicked` – click on a rules‑doc link from in‑client surfaces (e.g. “View formal rules”).

`RulesContext` is a **low‑cardinality concept tag** tied to rules semantics, not UI. Examples (to be extended as W‑UX‑2/3 evolve):

- `anm_forced_elimination` – Active‑No‑Moves + forced elimination semantics ([`RR-CANON-R072`](RULES_CANONICAL_SPEC.md:210), [`RR-CANON-R100`](RULES_CANONICAL_SPEC.md:443)).
- `structural_stalemate` – global stalemate / tiebreak per [`RR-CANON-R173`](RULES_CANONICAL_SPEC.md:619).
- `last_player_standing` – R172 semantics and compromises.
- `territory_mini_region` – Q23 mini‑region archetype, 2×2 region processing ([`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19)).
- `territory_multi_region` – multiple disconnected regions & self‑elimination budget.
- `line_reward_exact` – exact‑length line with mandatory elimination.
- `line_reward_overlength` – graduated line reward Options 1 vs 2.
- `capture_chain_mandatory` – chain capture continuation constraints ([`RR-CANON-R103`](RULES_CANONICAL_SPEC.md:480)).
- `landing_on_own_marker` – move / capture landing on own marker and forced self‑elimination ([`RR-CANON-R092`](RULES_CANONICAL_SPEC.md:418)).
- `pie_rule_swap` – 2‑player first‑move balancing.
- `placement_cap` – per‑player ring cap and placement/no‑dead‑placement issues.

`RulesUxSource` identifies the surface that emitted the event:

- `hud` – in‑game HUD (help button, banners, inline prompts).
- `victory_modal` – end‑of‑game modal.
- `teaching_overlay` – dedicated rules TeachingOverlay surface.
- `sandbox` – curated sandbox / scenario browser.
- `faq_panel` – in‑client FAQ / rules browser.
- `system_toast` – transient notification / toast.
- `external_docs` – browser navigations directly into [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1) or allied docs opened from the client.

Implementations MAY introduce additional `RulesContext` values over time, but SHOULD keep `RulesUxSource` and `RulesUxEventType` restricted to the sets above unless this document is updated.

---

## 3. Event Types and Required Fields

For each `event_type`, this section defines:

- When it is emitted.
- Required and optional `payload` fields.
- How to set `rules_context`.

### 3.1 `help_open`

Emitted whenever a user opens a rules help surface from within a game, sandbox, or lobby.

**Required:**

- `source` ∈ { `hud`, `victory_modal`, `teaching_overlay`, `sandbox`, `faq_panel` }.
- `payload.entrypoint` ∈:
  - `hud_help_icon`
  - `hud_phase_hint`
  - `hud_weird_state_banner`
  - `victory_modal_help_link`
  - `sandbox_toolbar_help`
  - `faq_button`
- `help_session_id` – new UUID for this help session.

**Optional:**

- `rules_context` – SHOULD be set when help is opened from a **contextual** entrypoint:
  - `anm_forced_elimination` if the current phase / banner corresponds to ANM + FE (see [`docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:39)).
  - `capture_chain_mandatory` if help is opened while `currentPhase == 'chain_capture'`.
  - `line_reward_exact` / `line_reward_overlength` when help is opened from a line‑processing banner.
  - `territory_mini_region` / `territory_multi_region` when opened from a territory decision banner.

### 3.2 `help_topic_view`

Emitted whenever a specific topic / FAQ / rules section is shown inside the in‑client help UI.

**Required:**

- `help_session_id` – must match the surrounding `help_open`.
- `payload.topic_id` – stable identifier, e.g.:
  - `faq_q15_capture_chains`
  - `faq_q23_disconnected_regions`
  - `rules_section_7_3_last_player_standing`
  - `compact_spec_section_6_territory`
- `payload.doc_anchor` – concrete anchor used for deep‑linking, e.g. `'#q15-how-are-disconnected-regions-evaluated'`.

**Optional:**

- `rules_context` – SHOULD be populated via a static mapping from `topic_id` to `RulesContext`. Example:
  - `faq_q23_disconnected_regions` → `territory_mini_region`.
  - `faq_q11_stalemate` → `structural_stalemate`.
- `payload.ms_since_help_open` – milliseconds between the parent `help_open` and this topic view.

### 3.3 `help_reopen`

Emitted when a user reopens help within a short window (e.g. 30 seconds) of closing it in the **same game**.

**Required:**

- `help_session_id` – reusing the previous session id OR linking via `payload.previous_help_session_id`.
- `payload.ms_since_last_close` – integer.

**Notes:**

- `rules_context` SHOULD match the previous help session’s last `rules_context` when the reopen occurs from the same phase / banner.
- Implementations MAY emit this as a synthetic event if the raw client only logs open/close; the backend can infer “reopen within 30s” as an offline transformation.

### 3.4 `weird_state_banner_impression`

Emitted whenever a weird/complex rules state banner is shown in the HUD (e.g. forced elimination, structural stalemate, ANM‑related forced rotation).

**Required:**

- `source = 'hud'`.
- `payload.reason_code` – **must** reference a stable reason code defined in [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1) (e.g. `ANM_FE_FORCED_ELIMINATION`).
- `rules_context` – derived from reason code (e.g. `anm_forced_elimination`, `structural_stalemate`).
- `overlay_session_id` – new UUID used for subsequent `weird_state_*` events related to the same occurrence.

### 3.5 `weird_state_details_open`

Emitted when a user clicks “Details”, “What happened?”, or equivalent from a weird‑state banner or from the VictoryModal.

**Required:**

- `payload.reason_code` – as in §3.4.
- `rules_context` – as in §3.4.
- `source` ∈ { `hud`, `victory_modal` }.

**Optional:**

- `payload.ms_since_impression` – from earliest `weird_state_banner_impression` / `weird_state_overlay_shown` with same `overlay_session_id`.

### 3.6 `weird_state_overlay_shown` / `weird_state_overlay_dismiss`

Emitted by the TeachingOverlay when auto‑launched or dismissed in response to a weird state.

**Required (both):**

- `source = 'teaching_overlay'`.
- `overlay_session_id`.
- `payload.reason_code`.
- `rules_context`.

**Additional for `weird_state_overlay_dismiss`:**

- `payload.completed` – boolean, `true` if the player reached the recommended end of the teaching flow.
- `payload.steps_completed` – integer count of teaching steps actually finished.

### 3.7 `resign_after_weird_state`

Emitted when a player resigns / abandons a game within a **short window** (e.g. 60 seconds or N moves) after a weird‑state banner / overlay related to the same `reason_code`.

**Required:**

- `payload.reason_code`.
- `rules_context`.
- `payload.ms_since_impression` OR `payload.moves_since_impression`.
- `payload.resign_type` ∈ { `menu_resign`, `timeout`, `disconnect`, `rage_quit` } where distinguishable.

**Notes:**

- This event is the primary basis for “resigns after weird states by context” hotspot queries (see §6.3).

### 3.8 `sandbox_scenario_loaded` / `sandbox_scenario_completed`

Emitted by the sandbox when a curated scenario is loaded or completed.

**Required:**

- `source = 'sandbox'`.
- `payload.scenario_id` – must match scenario ids defined in [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1).
- `payload.flow_id` (if part of a teaching flow).
- `rules_context` – derived from scenario metadata `rulesConcept`.

**Additional for `sandbox_scenario_completed`:**

- `payload.success` – boolean.
- `payload.moves_taken` – integer.
- `payload.ms_to_complete` – integer.

### 3.9 `teaching_step_started` / `teaching_step_completed`

Emitted whenever the user enters or completes a step in a multi‑step teaching flow (see [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1)).

**Required:**

- `teaching_flow_id` – stable id for the flow instance.
- `payload.flow_id` – logical flow key, e.g. `fe_loop_intro`, `mini_region_intro`.
- `payload.step_index` – integer, zero‑ or one‑based but consistent within a flow.
- `payload.scenario_id` – concrete scenario shown in this step.
- `rules_context` – flow’s primary concept (e.g. `anm_forced_elimination`).

---

## 4. Surface → Event Mapping

This section defines where instrumentation MUST occur.

### 4.1 Game HUD

- Help button / hotkey:
  - Emit `help_open` with `source = 'hud'`, `payload.entrypoint = 'hud_help_icon'`.
  - Set `rules_context` when the help is opened while:
    - `currentPhase == 'chain_capture'` → `capture_chain_mandatory`.
    - A forced‑elimination decision is pending → `anm_forced_elimination`.
    - `currentPhase == 'line_processing'` or `territory_processing` with eligible decisions → `line_reward_*` or `territory_*`.
- Phase / rules hints shown inline:
  - When a persistent hint banner is clicked to open help, treat as `help_open` with `payload.entrypoint = 'hud_phase_hint'`.
- Weird‑state banners:
  - On banner render → `weird_state_banner_impression`.
  - On “Details” click → `weird_state_details_open`.

### 4.2 VictoryModal

- When the VictoryModal first opens:
  - If the winner reason_code is a “weird / complex termination” (e.g. ANM‑driven Last‑Player‑Standing, structural stalemate, exotic territory chain), emit:
    - `weird_state_banner_impression` (logical impression in end‑screen form), with `source = 'victory_modal'`.
    - `help_open` if the user clicks into “See why you won / lost`.
- When a “What happened?” or rules explanation link is clicked:
  - `weird_state_details_open` with appropriate `reason_code` and `rules_context`.

### 4.3 TeachingOverlay

- When auto‑launched due to a weird‑state reason code:
  - `weird_state_overlay_shown`.
  - Start a teaching flow: `teaching_step_started` for the first step.
- On each step transition:
  - Emit `teaching_step_started` / `teaching_step_completed` accordingly.
- On close:
  - Emit `weird_state_overlay_dismiss` with `completed` flag and `steps_completed`.

### 4.4 Sandbox

- When a curated rules scenario is loaded from:
  - TeachingOverlay, or
  - Sandbox presets / rules browser,
- Emit `sandbox_scenario_loaded` with:
  - `rules_context` from scenario metadata.
  - `payload.origin` ∈ { `teaching_overlay`, `sandbox_browser`, `direct_link` }.
- On success / failure:
  - Emit `sandbox_scenario_completed`.

### 4.5 FAQ / Docs Panel

- When the embedded FAQ panel opens:
  - `help_open` with `source = 'faq_panel'`, `payload.entrypoint = 'faq_button'`.
- When individual topics are selected:
  - `help_topic_view` with `topic_id` and `doc_anchor`.

---

## 5. Metrics and Labels

Raw events SHOULD be ingested into a log / warehouse for richer analysis. Prometheus‑style metrics SHOULD remain low‑cardinality.

### 5.1 Core Counter

**Current implementation (MetricsService, 2025‑12‑06)**

The backend exposes a single low‑cardinality counter:

```text
ringrift_rules_ux_events_total{
  event_type,    // RulesUxEventType: help_open / help_topic_view / help_reopen / weird_state_* / sandbox_scenario_* / teaching_step_* / doc_link_clicked / legacy rules_* where still emitted
  rules_context, // semantic rules context (anm_forced_elimination, structural_stalemate, etc.) or "none"
  source,        // emitting surface: hud / victory_modal / teaching_overlay / sandbox / faq_panel / system_toast / external_docs / unknown
  board_type,    // square8 / square19 / hexagonal / "unknown"
  num_players,   // 1 / 2 / 3 / 4 / "unknown"
  difficulty,    // primary difficulty bucket or AI level: tutorial / casual / ranked_low / 1–10 / "unknown"
  is_ranked,     // "true" / "false" / "unknown"
  is_sandbox,    // "true" / "false" / "unknown"
}
```

Implementation reference:

- Counter is defined in [`MetricsService.ts`](src/server/services/MetricsService.ts:518) as `rulesUxEventsTotal` with `name: 'ringrift_rules_ux_events_total'`.
- Client payloads are defined in [`rulesUxEvents.ts`](src/shared/telemetry/rulesUxEvents.ts:101) and emitted via [`rulesUxTelemetry.ts`](src/client/utils/rulesUxTelemetry.ts:147).

This is a **strict subset** of the envelope in §2–§4: we intentionally do **not** expose high‑cardinality identifiers such as `game_id`, `session_id`, `scenario_id`, or `teaching_flow_id` as metric labels to keep Prometheus cardinality bounded. Those fields (where present) remain in logs, warehouses, or higher‑volume telemetry streams.

**Design intent (future extension)**

If future W‑UX work requires more granular hotspot queries, we MAY:

- Add a sibling counter keyed by `source` and a compacted `rules_context`, or
- Extend the existing counter with carefully reviewed additional labels.

Any such change MUST:

- Keep label sets drawn from bounded enums / buckets.
- Avoid promoting high‑cardinality identifiers (`game_id`, `session_id`, `scenario_id`, `flow_id`) to labels.
- Be reflected here and in `UX_RULES_IMPROVEMENT_LOOP.md`.

### 5.2 Supporting Game Counters

For normalisation, reuse or introduce:

```text
ringrift_games_started_total{
  board_type,
  num_players,
  difficulty,
  is_ranked,
  is_calibration_game
}

ringrift_games_completed_total{
  board_type,
  num_players,
  difficulty,
  is_ranked,
  terminal_reason // elimination / territory / stalemate / last_player_standing / resign / timeout
}
```

These are already partially covered by existing backend metrics; this spec only states that W‑UX‑1 hotspot queries may assume their existence.

---

## 6. Hotspot Queries (PromQL / Pseudocode)

This section sketches the minimal queries required by W‑UX‑1 and W‑UX‑4. Exact PromQL may be adjusted to local naming.

### 6.1 Top Rules Contexts by Help Opens per 100 Games

```promql
topk(10,
  sum by (rules_context) (
    rate(ringrift_rules_ux_events_total{
      event_type="help_open",
      source!="external_docs"
    }[7d])
  )
  /
  sum by (rules_context) (
    // approximate by attributing each help_open to its rules_context’ board_type / num_players bucket
    rate(ringrift_games_started_total[7d])
  )
)
* 100
```

Interpretation:

- For each `rules_context`, estimate **help opens per 100 games** over the last 7 days.
- Filter to contexts with stable volume (e.g. minimum 50 help opens) before acting.

### 6.2 Contexts with High “Help Reopen Within 30 Seconds”

Because windowed sequences are hard to express in PromQL alone, this can be computed via derived metrics:

1. Ingest raw events into a stream / batch job.
2. For each `(session_id, rules_context)` pair, create a boolean `reopened_within_30s` flag when a `help_reopen` follows `help_open` within 30 seconds.
3. Export:

```text
ringrift_rules_ux_help_sessions_total{
  rules_context,
  reopened_within_30s = "true" | "false"
}
```

Then:

```promql
sum by (rules_context) (ringrift_rules_ux_help_sessions_total{reopened_within_30s="true"})
/
clamp_min(
  sum by (rules_context) (ringrift_rules_ux_help_sessions_total{}),
  1
)
```

yields the **fraction of help sessions per rules_context that saw a rapid reopen**, a strong signal of unresolved confusion.

### 6.3 Resigns After Weird States by Context

With `resign_after_weird_state` events:

```promql
sum by (rules_context) (
  rate(ringrift_rules_ux_events_total{
    event_type="resign_after_weird_state"
  }[30d])
)
/
sum by (rules_context) (
  rate(ringrift_rules_ux_events_total{
    event_type="weird_state_banner_impression"
  }[30d])
)
```

This provides a **resigns‑per‑weird‑state‑impression** ratio per `rules_context`, highlightable in W‑UX‑4 iteration reports as an impact metric for UX changes.

---

## 7. Privacy and Retention

Principles:

- **No PII in telemetry.**
  - Do not log email, username, IP, or raw user ids.
  - `session_id` MUST be a random, per‑device/session identifier with no stable mapping to account ids.
- **Minimal linking.**
  - `game_id` may be included for debugging but SHOULD NOT be exposed as a metric label.
  - Long‑term analysis SHOULD rely on aggregate metrics rather than raw per‑event logs.
- **Retention.**
  - Raw Rules UX events SHOULD be retained for **90 days** or less.
  - Aggregated metrics (Prometheus time‑series, weekly summaries) MAY be retained indefinitely.
- **Sampling.**
  - If needed for scale, low‑value surfaces (e.g. very frequent generic `help_topic_view` events) MAY be sampled, but:
    - Events tied to weird states (`weird_state_*`, `resign_after_weird_state`) SHOULD NOT be sampled.
    - Any sampling MUST be documented and, if probabilistic, the sampling rate MUST be logged in `payload`.

---

## 8. Relationship to W‑UX‑2/3/4

- [`UX_RULES_WEIRD_STATES_SPEC.md`](docs/UX_RULES_WEIRD_STATES_SPEC.md:1) defines **reason codes** and copy for weird / confusing states. `reason_code` in this file MUST match those codes exactly.
- [`UX_RULES_TEACHING_SCENARIOS.md`](docs/UX_RULES_TEACHING_SCENARIOS.md:1) defines **scenario ids** and teaching flows; `scenario_id`, `flow_id`, and `rules_context` values in events are derived from that metadata.
- [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:1) assumes the metrics and hotspot queries described here are available and stable; any schema changes that affect:
  - `ringrift_rules_ux_events_total`,
  - `ringrift_games_started_total`,
  - `ringrift_rules_ux_help_sessions_total`

  MUST be reflected in that process document.

Together, these specs allow future Code‑mode tasks to:

- Instrument a **small, well‑labeled** set of telemetry events around confusing rules contexts.
- Aggregate them safely without exploding metric cardinality.
- Run repeatable hotspot analyses that feed directly into UX copy changes and new / revised teaching scenarios.

---

## 9. Operational Runbook: Using Rules‑UX Telemetry for Iteration

This section describes a lightweight loop for turning recorded telemetry into concrete W‑UX iteration items. It assumes:

- Client and server are emitting `RulesUxEvent` telemetry per this spec.
- Metrics and/or event logs are exported regularly to a warehouse or analytics job.
- Pre‑aggregated snapshots are produced in the [`RulesUxAggregatesRoot`](src/shared/telemetry/rulesUxHotspotTypes.ts:1) shape.

### 9.1 Generate a Rules‑UX Aggregates Snapshot

From logs or warehouse tables containing raw `RulesUxEvent` rows, or from Prometheus counters, produce an aggregate JSON file matching `RulesUxAggregatesRoot`:

- Restrict to a single board type and player count (today: `square8`, `num_players = 2`).
- For each `rules_context`, build `sources: RulesUxSourceAggregate[]` with:
  - `source` (hud / victory_modal / teaching_overlay / sandbox / faq_panel / …),
  - `events` map containing integer counts for:
    - `help_open`, `help_reopen` (or legacy `rules_help_repeat`),
    - `weird_state_banner_impression`,
    - `weird_state_details_open`,
    - `resign_after_weird_state` (or legacy `rules_weird_state_resign`),
    - plus any additional event types you wish to retain (ignored by the analyzer if unused).

The resulting JSON should look like:

```jsonc
{
  "board": "square8",
  "num_players": 2,
  "window": {
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-30T23:59:59Z",
  },
  "games": {
    "started": 1200,
    "completed": 800,
  },
  "contexts": [
    {
      "rulesContext": "anm_forced_elimination",
      "sources": [
        {
          "source": "hud",
          "events": {
            "help_open": 80,
            "help_reopen": 40,
            "weird_state_banner_impression": 200,
            "weird_state_details_open": 120,
            "resign_after_weird_state": 60,
          },
        },
      ],
    },
  ],
}
```

Export this to a file such as:

```bash
results/rules_ux_aggregates.square8_2p.2025-11.json
```

### 9.2 Run the Hotspot Analyzer

Use the Node CLI in [`analyze_rules_ux_telemetry.ts`](scripts/analyze_rules_ux_telemetry.ts:1) to turn aggregates into a hotspot report:

```bash
node scripts/analyze_rules_ux_telemetry.js \
  --input results/rules_ux_aggregates.square8_2p.2025-11.json \
  --output-json docs/ux/rules_ux_hotspots/rules_ux_hotspots.square8_2p.2025-11.json \
  --output-md docs/ux/rules_ux_hotspots/rules_ux_hotspots.square8_2p.2025-11.md \
  --min-events 20 \
  --top-k 5
```

The script:

- Validates the snapshot as a `RulesUxAggregatesRoot`.
- Computes, per `rules_context` and `source`:
  - Help opens per 100 completed games.
  - Help reopen fraction (using both `help_reopen` and legacy `rules_help_repeat`).
  - Resigns‑after‑weird‑state fraction (using both `resign_after_weird_state` and legacy `rules_weird_state_resign`).
- Assigns a `hotspotSeverity` (`LOW` / `MEDIUM` / `HIGH`) per context based on:
  - Rank by help opens per 100 games.
  - Repeat‑help rates.
  - Resign‑after‑weird rates.
- Emits:
  - A machine‑readable JSON summary.
  - A concise Markdown report suitable for pasting into W‑UX iteration docs.

**Smoke / dry-run:** To validate the analyzer without live aggregates, run it against the fixture
`tests/fixtures/rules_ux_hotspots/rules_ux_aggregates.square8_2p.sample.json`:

```bash
node scripts/analyze_rules_ux_telemetry.js \
  --input tests/fixtures/rules_ux_hotspots/rules_ux_aggregates.square8_2p.sample.json \
  --output-json /tmp/rules_ux_hotspots.sample.json \
  --output-md /tmp/rules_ux_hotspots.sample.md \
  --min-events 15 \
  --top-k 2
```

The sample snapshot yields `anm_forced_elimination` and `structural_stalemate` as the top two contexts,
marks the low-volume `territory_mini_region` as `sampleOk: false`, and sets `window.label` to `2025-12`.
The Jest smoke test `tests/unit/RulesUxHotspotAnalysis.test.ts` exercises the same fixture and guards these expectations.

### 9.3 Interpreting the Output

In the JSON and Markdown summary, focus on:

- **Top contexts by help opens per 100 games**:
  - High values mean players repeatedly need help in that rules area.
- **Max help reopen rate per context**:
  - Values ≥ 0.4 indicate a large fraction of help sessions reopen quickly, a strong signal of unresolved confusion.
- **Max resign‑after‑weird‑state rate per context**:
  - Values ≥ 0.2 indicate that a noticeable share of weird‑state impressions are followed by resigns or exits.

Contexts marked `HIGH` severity typically combine:

- High help‑opens per 100 games,
- High repeat‑help rates (`help_reopen` / `help_open`),
- And/or high resign‑after‑weird rates.

These should be treated as primary W‑UX iteration targets.

### 9.4 Turning Hotspots into UX Iteration Items

For each high‑severity `rules_context`:

1. **Inspect surface‑level breakdowns**:
   - Look at per‑source metrics in the report:
     - HUD vs `victory_modal` vs `teaching_overlay` vs `sandbox`.
   - Example: ANM/forced‑elimination confusion primarily from HUD banners vs post‑game VictoryModal.

2. **Map to teaching content and scenarios**:
   - For weird states:
     - Check the relevant topics and steps in [`TeachingOverlay.tsx`](src/client/components/TeachingOverlay.tsx:262) and [`TEACHING_SCENARIOS`](src/shared/teaching/teachingScenarios.ts:1).
   - For sandbox presets:
     - Review curated scenarios in [`curated.json`](src/client/public/scenarios/curated.json:1) and their `rulesConcept`.

3. **Create concrete W‑UX tickets**, for example:
   - Strengthen or shorten copy in the relevant TeachingOverlay topic.
   - Add or adjust curated sandbox scenarios that demonstrate the confusing pattern earlier and more clearly.
   - Add phase‑specific HUD hints or tweak weird‑state banners (without changing rules semantics).

4. **Re‑run the analyzer** after deploying UX changes:
   - Compare:
     - Help‑opens/100 games,
     - Help‑reopen rates,
     - Resign‑after‑weird rates
   - across the same time window before/after the change.
   - Treat reductions in reopen and resign‑after‑weird rates as evidence that the UX change improved understanding.

This runbook should be used alongside the broader improvement loop in [`UX_RULES_IMPROVEMENT_LOOP.md`](docs/UX_RULES_IMPROVEMENT_LOOP.md:1), which covers scheduling, ownership, and how rules‑UX telemetry feeds back into the canonical specs and teaching surfaces.
