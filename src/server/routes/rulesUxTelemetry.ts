import { Router } from 'express';
import type { Request, Response } from 'express';
import { asyncHandler, createError } from '../middleware/errorHandler';
import { telemetryRateLimiter } from '../middleware/rateLimiter';
import { getMetricsService } from '../services/MetricsService';
import { isRulesUxEventType, type RulesUxEventPayload } from '../../shared/telemetry/rulesUxEvents';

const router = Router();

// Payload size limits to prevent memory exhaustion and metric pollution
const MAX_STRING_LENGTH = 256;
const MAX_PAYLOAD_DEPTH = 3;
const MAX_PAYLOAD_KEYS = 20;
const MAX_PAYLOAD_SIZE_BYTES = 4096;

/**
 * Truncate a string to the maximum allowed length.
 */
function truncateString(value: string, maxLength = MAX_STRING_LENGTH): string {
  return value.length > maxLength ? value.slice(0, maxLength) : value;
}

/**
 * Validate and sanitize the optional nested payload object.
 * Rejects payloads that are too deep, have too many keys, or exceed size limits.
 */
function sanitizeNestedPayload(raw: unknown, depth = 0): Record<string, unknown> | null {
  if (raw === null || raw === undefined || typeof raw !== 'object') {
    return null;
  }

  if (depth > MAX_PAYLOAD_DEPTH) {
    return null; // Too deep, reject
  }

  // Check serialized size as a rough bound
  try {
    const serialized = JSON.stringify(raw);
    if (serialized.length > MAX_PAYLOAD_SIZE_BYTES) {
      return null; // Too large, reject
    }
  } catch {
    return null; // Not serializable, reject
  }

  const obj = raw as Record<string, unknown>;
  const keys = Object.keys(obj);

  if (keys.length > MAX_PAYLOAD_KEYS) {
    return null; // Too many keys, reject
  }

  const sanitized: Record<string, unknown> = {};
  for (const key of keys.slice(0, MAX_PAYLOAD_KEYS)) {
    const value = obj[key];
    if (typeof value === 'string') {
      sanitized[key] = truncateString(value);
    } else if (typeof value === 'number' || typeof value === 'boolean') {
      sanitized[key] = value;
    } else if (typeof value === 'object' && value !== null) {
      const nested = sanitizeNestedPayload(value, depth + 1);
      if (nested !== null) {
        sanitized[key] = nested;
      }
    }
    // Skip functions, symbols, undefined
  }

  return sanitized;
}

/**
 * Coerce and validate an arbitrary payload into a RulesUxEventPayload.
 *
 * This performs minimal runtime validation to keep the telemetry surface
 * low-cardinality and free of obviously malformed events. Additional
 * normalisation is applied inside MetricsService.recordRulesUxEvent.
 */
function coerceRulesUxEventPayload(raw: unknown): RulesUxEventPayload {
  const body = (raw ?? {}) as Record<string, unknown>;

  const {
    type,
    boardType,
    numPlayers,
    aiDifficulty,
    difficulty,
    rulesContext,
    source,
    gameId,
    isRanked,
    isCalibrationGame,
    isSandbox,
    aiProfile,
    seatIndex,
    perspectivePlayerCount,
    ts,
    clientBuild,
    clientPlatform,
    locale,
    sessionId,
    helpSessionId,
    overlaySessionId,
    teachingFlowId,
    topic,
    rulesConcept,
    scenarioId,
    weirdStateType,
    reasonCode,
    undoStreak,
    repeatCount,
    secondsSinceWeirdState,
    payload: rawPayload,
  } = body;

  if (!isRulesUxEventType(type)) {
    throw createError('Invalid rules UX telemetry type', 400, 'INVALID_RULES_UX_EVENT_TYPE');
  }

  if (typeof boardType !== 'string' || boardType.trim().length === 0) {
    throw createError(
      'Invalid board type for rules UX telemetry',
      400,
      'INVALID_RULES_UX_BOARD_TYPE'
    );
  }

  if (
    typeof numPlayers !== 'number' ||
    !Number.isFinite(numPlayers) ||
    numPlayers < 1 ||
    numPlayers > 4
  ) {
    throw createError(
      'Invalid numPlayers for rules UX telemetry (expected 1-4)',
      400,
      'INVALID_RULES_UX_NUM_PLAYERS'
    );
  }

  const payload: RulesUxEventPayload = {
    type,
    // RulesUxEventPayload expects BoardType but we accept any string at runtime
    // and rely on MetricsService to normalise label values.
    boardType: boardType as unknown as RulesUxEventPayload['boardType'],
    numPlayers,
  };

  if (typeof aiDifficulty === 'number' && Number.isFinite(aiDifficulty)) {
    payload.aiDifficulty = aiDifficulty;
  }

  if (typeof difficulty === 'string' && difficulty.length > 0) {
    payload.difficulty = truncateString(difficulty);
  }

  if (typeof rulesContext === 'string' && rulesContext.length > 0) {
    payload.rulesContext = truncateString(rulesContext);
  }

  if (typeof source === 'string' && source.length > 0) {
    payload.source = truncateString(source);
  }

  if (typeof gameId === 'string' && gameId.length > 0) {
    payload.gameId = truncateString(gameId);
  }

  if (typeof isRanked === 'boolean') {
    payload.isRanked = isRanked;
  }

  if (typeof isCalibrationGame === 'boolean') {
    payload.isCalibrationGame = isCalibrationGame;
  }

  if (typeof isSandbox === 'boolean') {
    payload.isSandbox = isSandbox;
  }

  if (typeof aiProfile === 'string' && aiProfile.length > 0) {
    payload.aiProfile = truncateString(aiProfile);
  }

  if (
    typeof seatIndex === 'number' &&
    Number.isInteger(seatIndex) &&
    seatIndex >= 1 &&
    seatIndex <= 4
  ) {
    payload.seatIndex = seatIndex;
  }

  if (
    typeof perspectivePlayerCount === 'number' &&
    Number.isInteger(perspectivePlayerCount) &&
    perspectivePlayerCount >= 1 &&
    perspectivePlayerCount <= 4
  ) {
    payload.perspectivePlayerCount = perspectivePlayerCount;
  }

  if (typeof ts === 'string' && ts.length > 0) {
    payload.ts = truncateString(ts);
  }

  if (typeof clientBuild === 'string' && clientBuild.length > 0) {
    payload.clientBuild = truncateString(clientBuild);
  }

  if (typeof clientPlatform === 'string' && clientPlatform.length > 0) {
    payload.clientPlatform = truncateString(clientPlatform);
  }

  if (typeof locale === 'string' && locale.length > 0) {
    payload.locale = truncateString(locale);
  }

  if (typeof sessionId === 'string' && sessionId.length > 0) {
    payload.sessionId = truncateString(sessionId);
  }

  if (typeof helpSessionId === 'string' && helpSessionId.length > 0) {
    payload.helpSessionId = truncateString(helpSessionId);
  }

  if (typeof overlaySessionId === 'string' && overlaySessionId.length > 0) {
    payload.overlaySessionId = truncateString(overlaySessionId);
  }

  if (typeof teachingFlowId === 'string' && teachingFlowId.length > 0) {
    payload.teachingFlowId = truncateString(teachingFlowId);
  }

  if (typeof topic === 'string' && topic.length > 0) {
    payload.topic = truncateString(topic);
  }

  if (typeof rulesConcept === 'string' && rulesConcept.length > 0) {
    payload.rulesConcept = truncateString(rulesConcept);
  }

  if (typeof scenarioId === 'string' && scenarioId.length > 0) {
    payload.scenarioId = truncateString(scenarioId);
  }

  if (typeof weirdStateType === 'string' && weirdStateType.length > 0) {
    payload.weirdStateType = truncateString(weirdStateType);
  }

  if (typeof reasonCode === 'string' && reasonCode.length > 0) {
    payload.reasonCode = truncateString(reasonCode);
  }

  if (typeof undoStreak === 'number' && Number.isFinite(undoStreak) && undoStreak > 0) {
    payload.undoStreak = undoStreak;
  }

  if (typeof repeatCount === 'number' && Number.isFinite(repeatCount) && repeatCount > 0) {
    payload.repeatCount = repeatCount;
  }

  if (
    typeof secondsSinceWeirdState === 'number' &&
    Number.isFinite(secondsSinceWeirdState) &&
    secondsSinceWeirdState >= 0
  ) {
    payload.secondsSinceWeirdState = secondsSinceWeirdState;
  }

  // Sanitize nested payload with depth/size limits to prevent abuse
  if (rawPayload && typeof rawPayload === 'object') {
    const sanitized = sanitizeNestedPayload(rawPayload);
    if (sanitized !== null) {
      payload.payload = sanitized;
    }
  }

  return payload;
}

/**
 * Core handler for POST /api/telemetry/rules-ux.
 *
 * This endpoint is intentionally lightweight and privacy-aware:
 * - No user identifiers or raw board positions are recorded.
 * - Only coarse board / player / AI context and small enums are accepted.
 */
export function handleRulesUxTelemetry(req: Request, res: Response): void {
  const payload = coerceRulesUxEventPayload(req.body);

  const metrics = getMetricsService();
  metrics.recordRulesUxEvent(payload);

  // No body required; telemetry is fire-and-forget.
  res.status(204).send();
}

router.post(
  '/rules-ux',
  telemetryRateLimiter,
  asyncHandler(async (req, res) => {
    handleRulesUxTelemetry(req, res);
  })
);

export default router;
