import type { RulesUxEventPayload, RulesUxEventType } from '../../shared/telemetry/rulesUxEvents';
import api from '../services/api';

/**
 * Lightweight, privacy-aware client helper for emitting rules-UX telemetry.
 *
 * - Uses the shared RulesUxEventPayload schema for client/server compatibility.
 * - Sends events via the existing axios API client to /api/telemetry/rules-ux.
 * - Swallows errors; telemetry must never affect UX flow.
 * - Applies optional sampling for high-frequency help-open events.
 */

function getEnv(): Record<string, string | undefined> {
  // Vite injects client env vars via a synthetic __VITE_ENV__ object on globalThis
  // (see errorReporting.ts for the same pattern).
  return ((globalThis as any).__VITE_ENV__ as Record<string, string | undefined> | undefined) ?? {};
}

function isTelemetryEnabled(): boolean {
  const env = getEnv();
  // Default to enabled; allow explicit opt-out via VITE_RULES_UX_TELEMETRY_ENABLED=false.
  const raw = env.VITE_RULES_UX_TELEMETRY_ENABLED;
  if (raw === 'false' || raw === '0') return false;
  return true;
}

/**
 * Parse the sampling rate for rules_help_open events from env.
 * Expected range: 0.0–1.0. Defaults to 1.0 (no sampling) when unset/invalid.
 */
function getHelpOpenSampleRate(): number {
  const env = getEnv();
  const raw = env.VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE;
  if (!raw) return 1;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return 1;
  if (parsed <= 0) return 0;
  if (parsed >= 1) return 1;
  return parsed;
}

/**
 * Simple deterministic string hash (32-bit, unsigned) for sampling.
 */
function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  // Convert to unsigned 32-bit
  return hash >>> 0;
}

/**
 * Decide whether to emit a given event based on type-specific sampling rules.
 * Currently only rules_help_open is sampled; all other events are always sent
 * (subject to the global enable flag).
 */
function shouldEmitSampled(event: RulesUxEventPayload): boolean {
  const { type } = event;

  if (type !== 'rules_help_open' && type !== 'help_open') {
    // Non-help-open events are expected to be relatively low volume.
    return true;
  }

  const rate = getHelpOpenSampleRate();
  if (rate >= 1) return true;
  if (rate <= 0) return false;

  const topic = event.topic ?? 'unknown';
  const key = [topic, event.boardType, String(event.numPlayers)].join('|');
  const hash = hashString(key);
  const normalized = hash / 0xffffffff; // 0.0–1.0
  return normalized < rate;
}

/**
 * Optionally log a warning in development when telemetry fails.
 */
function logDevWarning(message: string, error: unknown, extra?: Record<string, unknown>): void {
  const env = getEnv();
  if (env.MODE !== 'development') return;

  console.warn('[RulesUxTelemetry]', message, {
    error: error instanceof Error ? error.message : String(error),
    ...extra,
  });
}

let cachedSessionId: string | null = null;

/**
 * Best-effort classification of the current client platform.
 * Used to populate RulesUxEventPayload.clientPlatform.
 */
function getClientPlatform(): 'web' | 'mobile_web' | 'desktop' | string {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return 'desktop';
  }

  const ua = navigator.userAgent || '';
  const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(ua);
  return isMobile ? 'mobile_web' : 'web';
}

/**
 * Best-effort locale detection for telemetry enrichment.
 */
function getLocale(): string | undefined {
  if (typeof navigator !== 'undefined' && typeof navigator.language === 'string') {
    return navigator.language;
  }
  return undefined;
}

/**
 * Lazily generate a per-session identifier for correlating rules-UX events.
 * This is intentionally not tied to user identity and is not persisted
 * beyond the current runtime environment.
 */
function getSessionId(): string {
  if (cachedSessionId) return cachedSessionId;

  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      cachedSessionId = anyCrypto.randomUUID();
      return cachedSessionId;
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }

  cachedSessionId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  return cachedSessionId;
}

/**
 * Send a single rules-UX telemetry event to the backend.
 *
 * - POSTs to /api/telemetry/rules-ux via the shared axios client, so that
 *   authentication and CSRF behaviour mirror other API calls.
 * - Returns a resolved Promise even when the underlying request fails.
 * - Applies type-specific sampling for rules_help_open events.
 */
export async function sendRulesUxEvent(event: RulesUxEventPayload): Promise<void> {
  if (!isTelemetryEnabled()) return;
  if (!shouldEmitSampled(event)) return;

  const enriched: RulesUxEventPayload = {
    ...event,
    ts: event.ts ?? new Date().toISOString(),
    clientBuild:
      event.clientBuild ??
      getEnv().VITE_CLIENT_BUILD ??
      getEnv().VITE_GIT_SHA ??
      getEnv().VITE_APP_VERSION ??
      getEnv().MODE,
    clientPlatform: event.clientPlatform ?? getClientPlatform(),
    locale: event.locale ?? getLocale(),
    sessionId: event.sessionId ?? getSessionId(),
  };

  try {
    // Fire-and-forget; callers do not depend on telemetry success.
    await api.post('/telemetry/rules-ux', enriched);
  } catch (error) {
    logDevWarning('Failed to send rules UX telemetry event', error, {
      type: enriched.type as RulesUxEventType,
      boardType: enriched.boardType,
    });
  }
}

/**
 * High-level helper that enriches a RulesUxEventPayload with common
 * client metadata (timestamp, build, platform, locale, and a
 * per-session identifier) before sending it via {@link sendRulesUxEvent}.
 */
export async function logRulesUxEvent(event: RulesUxEventPayload): Promise<void> {
  await sendRulesUxEvent(event);
}

/**
 * Generate a fresh correlation id for a help session.
 *
 * This is separate from the per-session {@link getSessionId} and is intended
 * to be reused across a single help_open → help_topic_view → help_reopen
 * interaction as described in UX_RULES_TELEMETRY_SPEC.md.
 */
export function newHelpSessionId(): string {
  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      return anyCrypto.randomUUID();
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Generate a fresh correlation id for a weird-state overlay / banner session.
 *
 * Used to tie together weird_state_banner_impression, weird_state_overlay_shown,
 * weird_state_overlay_dismiss, and resign_after_weird_state events.
 */
export function newOverlaySessionId(): string {
  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      return anyCrypto.randomUUID();
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Generate a fresh correlation id for a multi-step teaching flow.
 *
 * Intended for teaching_step_started / teaching_step_completed and
 * sandbox_scenario_* events that are part of a single coherent flow.
 */
export function newTeachingFlowId(): string {
  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      return anyCrypto.randomUUID();
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Options for emitting a spec-aligned help_open event.
 *
 * This is a thin, typed wrapper over {@link sendRulesUxEvent} that fills
 * the envelope fields recommended in UX_RULES_TELEMETRY_SPEC.md §3.1.
 */
export interface HelpOpenEventOptions {
  boardType: RulesUxEventPayload['boardType'];
  numPlayers: RulesUxEventPayload['numPlayers'];
  aiDifficulty?: number;
  difficulty?: string;
  rulesContext?: RulesUxEventPayload['rulesContext'];
  rulesConcept?: RulesUxEventPayload['rulesConcept'];
  topic?: RulesUxEventPayload['topic'];
  scenarioId?: RulesUxEventPayload['scenarioId'];
  source: RulesUxEventPayload['source'];
  /**
   * Low-cardinality identifier for where help was opened from, e.g.:
   * - 'hud_help_chip'
   * - 'mobile_hud_help_chip'
   * - 'victory_modal_help_link'
   * - 'sandbox_toolbar_help'
   * - 'faq_button'
   */
  entrypoint: string;
  gameId?: string;
  isRanked?: boolean;
  isCalibrationGame?: boolean;
  isSandbox?: boolean;
  seatIndex?: number;
  perspectivePlayerCount?: number;
  /**
   * Optional pre-generated help_session_id. When omitted, a new one is
   * created via {@link newHelpSessionId}.
   */
  helpSessionId?: string;
}

/**
 * Emit a spec-aligned help_open event with the given options.
 *
 * This does not replace the legacy rules_help_open event; callers that
 * still need the legacy metrics can emit both. The payload here follows
 * the language-agnostic contract in UX_RULES_TELEMETRY_SPEC.md.
 */
export async function logHelpOpenEvent(options: HelpOpenEventOptions): Promise<void> {
  const {
    boardType,
    numPlayers,
    aiDifficulty,
    difficulty,
    rulesContext,
    rulesConcept,
    topic,
    scenarioId,
    source,
    entrypoint,
    gameId,
    isRanked,
    isCalibrationGame,
    isSandbox,
    seatIndex,
    perspectivePlayerCount,
    helpSessionId,
  } = options;

  const event: RulesUxEventPayload = {
    type: 'help_open',
    boardType,
    numPlayers,
    aiDifficulty,
    difficulty,
    rulesContext,
    rulesConcept,
    topic,
    scenarioId,
    source,
    gameId,
    isRanked,
    isCalibrationGame,
    isSandbox,
    seatIndex,
    perspectivePlayerCount,
    helpSessionId: helpSessionId ?? newHelpSessionId(),
    payload: {
      entrypoint,
    },
  };

  await sendRulesUxEvent(event);
}

/**
 * Canonical UX rules copy blocks used by onboarding and teaching surfaces.
 * The wording for these blocks is copied from docs/UX_RULES_COPY_SPEC.md and
 * must remain aligned with that spec (see inline section references).
 *
 * These structures are intentionally lightweight and UI-agnostic so multiple
 * components (OnboardingModal, TeachingOverlay, Sandbox overlays, etc.) can
 * consume the same canonical strings without duplicating literals.
 */
export interface RulesCopyBlock {
  id: string;
  title: string;
  body: string;
}

/**
 * Canonical onboarding copy for first-time players.
 */
export interface OnboardingCopy {
  /** High-level introduction to RingRift. */
  intro: RulesCopyBlock;
  /** Short phase overview cards shown in onboarding. */
  phases: RulesCopyBlock[];
  /** Victory condition summary cards. */
  victoryConcepts: RulesCopyBlock[];
}

/**
 * Canonical TeachingOverlay topic copy keyed by topic id.
 *
 * topicId is designed to line up with TeachingOverlay's TeachingTopic union
 * (e.g. "stack_movement", "capturing", "victory_elimination").
 */
export interface TeachingTopicCopy {
  topicId: string;
  heading: string;
  body: string;
}

/**
 * Canonical onboarding copy; see UX_RULES_COPY_SPEC.md §3 (victory) and §8
 * (sandbox phase‑copy summary). Non-rulesy intro text is kept short and
 * UI-specific but centralised here to avoid ad-hoc duplication.
 */
export const ONBOARDING_COPY: OnboardingCopy = {
  intro: {
    id: 'onboarding.intro',
    title: 'Welcome to RingRift!',
    body: 'A strategic board game where you place rings, build stacks, and compete for territory.',
  },
  phases: [
    {
      id: 'sandbox.phase.ring_placement',
      title: 'Ring Placement',
      // UX_RULES_COPY_SPEC.md §8 – Ring Placement summary
      body: 'Place new rings or add to existing stacks while keeping at least one real move available for your next turn.',
    },
    {
      id: 'sandbox.phase.movement',
      title: 'Movement',
      // UX_RULES_COPY_SPEC.md §8 – Movement summary
      body: 'Pick a stack and move it in a straight line at least as far as its height, and farther if the path stays clear (stacks and territory block; markers do not).',
    },
    {
      id: 'sandbox.phase.capture',
      title: 'Capture',
      // UX_RULES_COPY_SPEC.md §8 – Capture summary
      body: 'Start an overtaking capture by jumping over an adjacent stack and landing on an empty or marker space beyond it.',
    },
  ],
  victoryConcepts: [
    {
      id: 'onboarding.victory.elimination',
      title: 'Ring Elimination',
      // UX_RULES_COPY_SPEC.md §3.1 – TeachingOverlay victory topic – elimination
      body: 'Win by eliminating more than half of all rings in the game – not just one opponent’s set. Eliminated rings are permanently removed; captured rings you carry in stacks do not count toward this threshold.',
    },
    {
      id: 'onboarding.victory.territory',
      title: 'Territory Control',
      // UX_RULES_COPY_SPEC.md §3.2 – TeachingOverlay victory topic – territory
      body: 'Win by owning more than half of all board spaces as Territory. Territory comes from collapsing marker lines and resolving disconnected regions, and once a space becomes Territory it can’t be captured back.',
    },
    {
      id: 'onboarding.victory.lps',
      title: 'Last Player Standing',
      // UX_RULES_COPY_SPEC.md §3.3 – TeachingOverlay victory topic – stalemate / LPS
      body: 'Last Player Standing happens when, for TWO consecutive complete rounds, you are the only player who can still make real moves (placements, movements, or captures). In the first round you must have and take at least one real action while all others have none; in the second round the condition must persist. Forced eliminations and automatic territory processing do not count as real actions for LPS.',
    },
  ],
};

/**
 * Canonical body copy for TeachingOverlay topics. Titles/headings and tips
 * may still be tailored per surface, but the long-form explanation for each
 * topic lives here and is shared across HUD, Victory surfaces, and Sandbox
 * teaching flows.
 */
export const TEACHING_TOPICS_COPY: Record<string, TeachingTopicCopy> = {
  ring_placement: {
    topicId: 'ring_placement',
    heading: 'Ring Placement',
    // Derived from ring placement rules in ringrift_complete_rules.md; keep
    // this aligned with the canonical rules docs.
    body: 'Players take turns placing rings from their hand onto empty board spaces or on top of existing stacks. Placement sets up future movement, captures, and territory – but you must keep at least one real move available for your next turn.',
  },
  stack_movement: {
    topicId: 'stack_movement',
    heading: 'Stack Movement',
    // UX_RULES_COPY_SPEC.md §4 – TeachingOverlay – Stack Movement description
    body: 'Move a stack you control (your ring on top) in a straight line at least as many spaces as the stack’s height. You can keep going farther as long as the path has no stacks or territory spaces blocking you; markers are allowed and may eliminate your top ring when you land on them.',
  },
  capturing: {
    topicId: 'capturing',
    heading: 'Capturing',
    // UX_RULES_COPY_SPEC.md §5 – TeachingOverlay – Capturing description
    body: 'To capture, jump over an adjacent opponent stack in a straight line and land on the empty space just beyond it. You take the top ring from the jumped stack and add it to the bottom of your own stack. Captured rings stay in play – only later eliminations move rings out of the game.',
  },
  chain_capture: {
    topicId: 'chain_capture',
    heading: 'Chain Capture',
    // UX_RULES_COPY_SPEC.md §5 – TeachingOverlay – Chain Capture description
    body: 'If your capturing stack can jump again after a capture, you are in a chain capture. Starting the first capture is optional, but once the chain begins you must keep capturing as long as any capture is available. When several jumps exist, you choose which target to take next.',
  },
  line_bonus: {
    topicId: 'line_bonus',
    heading: 'Lines and Rewards',
    // UX_RULES_COPY_SPEC.md §6 – TeachingOverlay – Lines description
    body: 'Lines are built from your markers. When a straight line of your markers reaches the minimum length for this board, it becomes a scoring line: you collapse markers in that line into permanent Territory and, on many boards, must pay a ring elimination cost from a stack you control.',
  },
  territory: {
    topicId: 'territory',
    heading: 'Territory',
    // UX_RULES_COPY_SPEC.md §7 – TeachingOverlay – Territory description
    body: 'Territory spaces are collapsed cells that you permanently own. When a disconnected region of your pieces is processed, all of its spaces become your Territory and its rings are eliminated, often at the cost of eliminating a ring from one of your other stacks. If your Territory passes more than half of the board, you win immediately.',
  },
  active_no_moves: {
    topicId: 'active_no_moves',
    heading: 'When you have no legal moves',
    // UX_RULES_COPY_SPEC.md §10.4 – teaching.active_no_moves description
    body: 'Sometimes it is your turn but there are no legal placements, movements, or captures available. This is an Active–No–Moves state: the rules engine will either trigger forced elimination of your stacks, or, if no eliminations are possible, treat you as structurally stuck for Last Player Standing and plateau detection.',
  },
  forced_elimination: {
    topicId: 'forced_elimination',
    heading: 'Forced Elimination (FE)',
    // UX_RULES_COPY_SPEC.md §10.4 – teaching.forced_elimination description
    body: 'Forced Elimination happens when you control stacks but have no legal placements, movements, or captures. Caps are removed from your stacks automatically until either a real move becomes available or your stacks are gone. These eliminations are mandatory and follow the rules, not player choice.',
  },
  victory_elimination: {
    topicId: 'victory_elimination',
    heading: 'Victory: Elimination',
    // UX_RULES_COPY_SPEC.md §3.1 – TeachingOverlay victory topic – elimination
    body: 'Win by eliminating more than half of all rings in the game – not just one opponent’s set. Eliminated rings are permanently removed; captured rings you carry in stacks do not count toward this threshold.',
  },
  victory_territory: {
    topicId: 'victory_territory',
    heading: 'Victory: Territory',
    // UX_RULES_COPY_SPEC.md §3.2 – TeachingOverlay victory topic – territory
    body: 'Win by owning more than half of all board spaces as Territory. Territory comes from collapsing marker lines and resolving disconnected regions, and once a space becomes Territory it can’t be captured back.',
  },
  victory_stalemate: {
    topicId: 'victory_stalemate',
    heading: 'Victory: Last Player Standing',
    // UX_RULES_COPY_SPEC.md §3.3 – TeachingOverlay victory topic – stalemate / LPS
    body: 'Last Player Standing happens when, for TWO consecutive complete rounds, you are the only player who can still make real moves (placements, movements, or captures). In the first round you must have and take at least one real action while all others have none; in the second round the condition must persist. Forced eliminations and automatic territory processing do not count as real actions for LPS.',
  },
};
