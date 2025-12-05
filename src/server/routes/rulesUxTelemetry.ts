import { Router } from 'express';
import type { Request, Response } from 'express';
import { asyncHandler, createError } from '../middleware/errorHandler';
import { getMetricsService } from '../services/MetricsService';
import { isRulesUxEventType, type RulesUxEventPayload } from '../../shared/telemetry/rulesUxEvents';

const router = Router();

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
    payload.difficulty = difficulty;
  }

  if (typeof rulesContext === 'string' && rulesContext.length > 0) {
    payload.rulesContext = rulesContext;
  }

  if (typeof source === 'string' && source.length > 0) {
    payload.source = source;
  }

  if (typeof gameId === 'string' && gameId.length > 0) {
    payload.gameId = gameId;
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
    payload.aiProfile = aiProfile;
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
    payload.ts = ts;
  }

  if (typeof clientBuild === 'string' && clientBuild.length > 0) {
    payload.clientBuild = clientBuild;
  }

  if (typeof clientPlatform === 'string' && clientPlatform.length > 0) {
    payload.clientPlatform = clientPlatform;
  }

  if (typeof locale === 'string' && locale.length > 0) {
    payload.locale = locale;
  }

  if (typeof sessionId === 'string' && sessionId.length > 0) {
    payload.sessionId = sessionId;
  }

  if (typeof helpSessionId === 'string' && helpSessionId.length > 0) {
    payload.helpSessionId = helpSessionId;
  }

  if (typeof overlaySessionId === 'string' && overlaySessionId.length > 0) {
    payload.overlaySessionId = overlaySessionId;
  }

  if (typeof teachingFlowId === 'string' && teachingFlowId.length > 0) {
    payload.teachingFlowId = teachingFlowId;
  }

  if (typeof topic === 'string' && topic.length > 0) {
    payload.topic = topic;
  }

  if (typeof rulesConcept === 'string' && rulesConcept.length > 0) {
    payload.rulesConcept = rulesConcept;
  }

  if (typeof scenarioId === 'string' && scenarioId.length > 0) {
    payload.scenarioId = scenarioId;
  }

  if (typeof weirdStateType === 'string' && weirdStateType.length > 0) {
    payload.weirdStateType = weirdStateType;
  }

  if (typeof reasonCode === 'string' && reasonCode.length > 0) {
    payload.reasonCode = reasonCode;
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

  if (rawPayload && typeof rawPayload === 'object') {
    payload.payload = rawPayload as Record<string, unknown>;
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
  asyncHandler(async (req, res) => {
    handleRulesUxTelemetry(req, res);
  })
);

export default router;
