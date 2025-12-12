import type { BoardType } from '../types/game';

/**
 * Discriminant for lightweight rules‑UX telemetry events.
 *
 * These events are emitted by the client when players interact with
 * rules-heavy UX surfaces (teaching overlays, weird‑state banners, undo,
 * contextual help, sandbox teaching scenarios, etc.) so that the backend
 * can aggregate where rules understanding breaks down.
 *
 * NOTE: This schema is intentionally low‑cardinality and must not be
 * extended with user identifiers, raw board positions, or free‑text.
 *
 * The string values here are intentionally stable and are mirrored in
 * docs/UX_RULES_TELEMETRY_SPEC.md. We keep the original "rules_*" event
 * names for backwards-compatible metrics while also introducing the
 * newer semantic event types described in the spec.
 */
export type RulesUxEventType =
  // Legacy, metrics‑oriented event identifiers (kept for compatibility)
  | 'rules_help_open'
  | 'rules_help_repeat'
  | 'rules_undo_churn'
  | 'rules_weird_state_resign'
  | 'rules_weird_state_help'
  // Spec‑aligned event identifiers (non‑exhaustive)
  | 'help_open'
  | 'help_topic_view'
  | 'help_reopen'
  | 'weird_state_banner_impression'
  | 'weird_state_details_open'
  | 'weird_state_overlay_shown'
  | 'weird_state_overlay_dismiss'
  | 'resign_after_weird_state'
  | 'sandbox_scenario_loaded'
  | 'sandbox_scenario_completed'
  | 'teaching_step_started'
  | 'teaching_step_completed'
  | 'doc_link_clicked'
  // FSM decision surface events (Phase 5: UI/Telemetry Integration)
  | 'fsm_decision_surface_shown'
  | 'fsm_decision_made'
  | 'fsm_phase_transition';

/**
 * Coarse classification of weird rules states as surfaced in the HUD.
 * Mirrors the values produced by getWeirdStateBanner on the client and
 * the RulesUxWeirdStateType label in metrics.
 */
export type RulesUxWeirdStateType =
  | 'active-no-moves-movement'
  | 'active-no-moves-line'
  | 'active-no-moves-territory'
  | 'last-player-standing'
  | 'forced-elimination'
  | 'structural-stalemate';

/**
 * Low‑cardinality semantic rules context used for hotspot analysis.
 *
 * These correspond to the RulesContext values described in
 * docs/UX_RULES_TELEMETRY_SPEC.md and docs/UX_RULES_WEIRD_STATES_SPEC.md.
 * They deliberately describe *rules concepts*, not UI surfaces.
 */
export type RulesUxContext =
  | 'anm_forced_elimination'
  | 'structural_stalemate'
  | 'last_player_standing'
  | 'territory_mini_region'
  | 'territory_multi_region'
  | 'line_reward_exact'
  | 'line_reward_overlength'
  | 'line_vs_territory_multi_phase'
  | 'capture_chain_mandatory'
  | 'landing_on_own_marker'
  | 'pie_rule_swap'
  | 'placement_cap';

/**
 * Surface that emitted the rules‑UX event.
 *
 * Matches the RulesUxSource values from the telemetry spec.
 */
export type RulesUxSource =
  | 'hud'
  | 'victory_modal'
  | 'teaching_overlay'
  | 'sandbox'
  | 'faq_panel'
  | 'system_toast'
  | 'external_docs';

/**
 * Payload for a single rules‑UX telemetry event.
 *
 * All fields other than {@link type}, {@link boardType}, and
 * {@link numPlayers} are optional and may be omitted when they do not
 * apply to a particular event.
 *
 * This is a concrete, TypeScript‑level projection of the language‑agnostic
 * RulesUxEvent envelope described in docs/UX_RULES_TELEMETRY_SPEC.md. Some
 * fields (e.g. {@link aiDifficulty}) exist primarily for legacy metrics and
 * will eventually be superseded by the higher‑level difficulty / context
 * fields as the telemetry surface matures.
 */
export interface RulesUxEventPayload {
  /**
   * Event discriminant.
   *
   * Historically this used only the "rules_*" values; newer code SHOULD
   * prefer the spec‑aligned identifiers (e.g. "help_open") while the
   * server continues to accept both.
   */
  type: RulesUxEventType;

  /** Coarse board topology; never includes full positions. */
  boardType: BoardType;

  /** Number of seats in the game (2, 3, or 4). */
  numPlayers: number;

  /**
   * AI difficulty when applicable (for example, sandbox vs AI or online
   * games with AI opponents). 1–10 scale, aligned with AI ladder docs.
   *
   * Kept for backwards‑compatible metrics; newer flows may also set the
   * coarse {@link difficulty} bucket.
   */
  aiDifficulty?: number;

  /**
   * Coarse bucket for overall game difficulty / context from the player's
   * perspective. Mirrors the "difficulty" field in the telemetry spec.
   *
   * Examples: "tutorial", "casual", "ranked_low", "ranked_mid", "ranked_high".
   */
  difficulty?: string;

  /** Optional semantic rules context for this event (low-cardinality tag). */
  rulesContext?: RulesUxContext | string;

  /** Surface that emitted the event (HUD, Victory modal, sandbox, etc.). */
  source?: RulesUxSource | string;

  // ────────────────────────────────────────────────────────────────────────
  // Game / rules state context (when applicable)
  // ────────────────────────────────────────────────────────────────────────

  /** Short game id; may be omitted for pure sandbox / docs views. */
  gameId?: string;

  /** Whether this game is rated / ranked on the ladder. */
  isRanked?: boolean;

  /** Whether this game is part of an explicit calibration run. */
  isCalibrationGame?: boolean;

  /** Whether this event originated from a sandbox-only session. */
  isSandbox?: boolean;

  /**
   * Optional stable identifier for the AI profile used in this game
   * (e.g. "descent_v3", "heuristic_easy"). This is a low-cardinality
   * config key and must not contain per-user data.
   */
  aiProfile?: string;

  /** Seat index (1–4) of the acting or viewing player, if applicable. */
  seatIndex?: number;

  /**
   * Number of human players participating from this client’s perspective.
   * For example, 1 in a solo vs AI game, 2 in a hotseat game, etc.
   */
  perspectivePlayerCount?: number;

  // ────────────────────────────────────────────────────────────────────────
  // Time & client context
  // ────────────────────────────────────────────────────────────────────────

  /** ISO‑8601 UTC timestamp, if supplied by the client helper. */
  ts?: string;

  /** Short build / git SHA for the client bundle. */
  clientBuild?: string;

  /** Platform bucket: web / mobile_web / desktop. */
  clientPlatform?: 'web' | 'mobile_web' | 'desktop' | string;

  /** Optional locale string, e.g. "en-US". */
  locale?: string;

  /**
   * Random, per‑device/session identifier used for correlating help /
   * overlay sessions. MUST NOT be a user id or anything linkable to PII.
   */
  sessionId?: string;

  // ────────────────────────────────────────────────────────────────────────
  // Correlation identifiers for multi‑step interactions
  // ────────────────────────────────────────────────────────────────────────

  /** Stable id reused across a single help session (open → topic_view → reopen). */
  helpSessionId?: string;

  /** Stable id reused across a weird‑state overlay / banner interaction. */
  overlaySessionId?: string;

  /** Stable id reused across the lifetime of a teaching flow. */
  teachingFlowId?: string;

  // ────────────────────────────────────────────────────────────────────────
  // Event‑specific payload and legacy fields
  // ────────────────────────────────────────────────────────────────────────

  /**
   * TeachingOverlay topic identifier (e.g. "active_no_moves").
   * Used by existing HUD telemetry and retained for metrics.
   */
  topic?: string;

  /**
   * Curated scenario rulesConcept when applicable. This is separate from
   * {@link rulesContext}, which describes the semantic rules context for
   * the current confusion/help interaction.
   */
  rulesConcept?: string;

  /** Curated scenario identifier when applicable. */
  scenarioId?: string;

  /**
   * Coarse weird‑state classification when event is weird‑state related.
   * Mirrors the RulesUxWeirdStateType used as a metrics label.
   */
  weirdStateType?: RulesUxWeirdStateType | string;

  /**
   * Optional, stable weird‑state reason code (e.g. "ANM_MOVEMENT_FE_BLOCKED",
   * "STRUCTURAL_STALEMATE_TIEBREAK"). When present this should match the
   * reason_code catalogue in docs/UX_RULES_WEIRD_STATES_SPEC.md.
   */
  reasonCode?: string;

  /** Number of undos in the recent streak for undo‑churn events. */
  undoStreak?: number;

  /** Number of times a help topic has been opened during the current game. */
  repeatCount?: number;

  /**
   * Seconds between the first observation of the weird state and a
   * subsequent resign/abandonment event.
   */
  secondsSinceWeirdState?: number;

  /**
   * Generic event‑specific payload. This should only contain small enums,
   * integers, and short strings – never raw board state or free‑text.
   */
  payload?: Record<string, unknown>;

  // ────────────────────────────────────────────────────────────────────────
  // FSM Decision Surface telemetry (Phase 5: UI/Telemetry Integration)
  // ────────────────────────────────────────────────────────────────────────

  /**
   * FSM phase that triggered this event. Low-cardinality phase name from the
   * FSM state machine (e.g., 'line_processing', 'territory_processing').
   */
  fsmPhase?: string;

  /**
   * Type of pending decision from FSM orchestration result.
   * Matches FSMOrchestrationResult.pendingDecisionType.
   */
  fsmDecisionType?:
    | 'chain_capture'
    | 'line_order_required'
    | 'no_line_action_required'
    | 'region_order_required'
    | 'no_territory_action_required'
    | 'forced_elimination';

  /**
   * Count of pending lines in FSM decision surface (line_processing phase).
   * Used to measure complexity of line processing decisions.
   */
  fsmPendingLineCount?: number;

  /**
   * Count of pending regions in FSM decision surface (territory_processing phase).
   * Used to measure complexity of territory processing decisions.
   */
  fsmPendingRegionCount?: number;

  /**
   * Count of chain capture continuations available (chain_capture phase).
   * Used to measure complexity of chain capture decisions.
   */
  fsmChainContinuationCount?: number;

  /**
   * Count of forced eliminations required (forced_elimination phase).
   */
  fsmForcedEliminationCount?: number;
}

/**
 * Exhaustive list of supported rules‑UX telemetry event types, exposed
 * for runtime validation on the server.
 */
export const RULES_UX_EVENT_TYPES: readonly RulesUxEventType[] = [
  'rules_help_open',
  'rules_help_repeat',
  'rules_undo_churn',
  'rules_weird_state_resign',
  'rules_weird_state_help',
  'help_open',
  'help_topic_view',
  'help_reopen',
  'weird_state_banner_impression',
  'weird_state_details_open',
  'weird_state_overlay_shown',
  'weird_state_overlay_dismiss',
  'resign_after_weird_state',
  'sandbox_scenario_loaded',
  'sandbox_scenario_completed',
  'teaching_step_started',
  'teaching_step_completed',
  'doc_link_clicked',
  // FSM decision surface events
  'fsm_decision_surface_shown',
  'fsm_decision_made',
  'fsm_phase_transition',
];

/**
 * Runtime guard for validating arbitrary input as a {@link RulesUxEventType}.
 */
export function isRulesUxEventType(value: unknown): value is RulesUxEventType {
  return typeof value === 'string' && (RULES_UX_EVENT_TYPES as readonly string[]).includes(value);
}
