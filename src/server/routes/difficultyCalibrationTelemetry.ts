import { Router } from 'express';
import type { Request, Response } from 'express';
import { asyncHandler, createError } from '../middleware/errorHandler';
import { telemetryRateLimiter } from '../middleware/rateLimiter';
import { getMetricsService } from '../services/MetricsService';
import {
  isDifficultyCalibrationEventType,
  type DifficultyCalibrationEventPayload,
} from '../../shared/telemetry/difficultyCalibrationEvents';

const router = Router();

/**
 * Coerce and validate an arbitrary payload into a DifficultyCalibrationEventPayload.
 *
 * This performs minimal runtime validation to keep the telemetry surface
 * low-cardinality and free of obviously malformed events. Additional
 * normalisation is applied inside MetricsService.recordDifficultyCalibrationEvent.
 */
function coerceDifficultyCalibrationEventPayload(raw: unknown): DifficultyCalibrationEventPayload {
  const body = (raw ?? {}) as Record<string, unknown>;

  const {
    type,
    boardType,
    numPlayers,
    difficulty,
    isCalibrationOptIn,
    result,
    movesPlayed,
    perceivedDifficulty,
  } = body;

  if (!isDifficultyCalibrationEventType(type)) {
    throw createError(
      'Invalid difficulty calibration telemetry type',
      400,
      'INVALID_DIFFICULTY_CALIBRATION_EVENT_TYPE'
    );
  }

  if (typeof boardType !== 'string' || boardType.trim().length === 0) {
    throw createError(
      'Invalid board type for difficulty calibration telemetry',
      400,
      'INVALID_DIFFICULTY_CALIBRATION_BOARD_TYPE'
    );
  }

  if (
    typeof numPlayers !== 'number' ||
    !Number.isFinite(numPlayers) ||
    numPlayers < 1 ||
    numPlayers > 4
  ) {
    throw createError(
      'Invalid numPlayers for difficulty calibration telemetry (expected 1-4)',
      400,
      'INVALID_DIFFICULTY_CALIBRATION_NUM_PLAYERS'
    );
  }

  if (
    typeof difficulty !== 'number' ||
    !Number.isFinite(difficulty) ||
    difficulty < 1 ||
    difficulty > 10
  ) {
    throw createError(
      'Invalid difficulty for difficulty calibration telemetry (expected 1-10)',
      400,
      'INVALID_DIFFICULTY_CALIBRATION_DIFFICULTY'
    );
  }

  if (typeof isCalibrationOptIn !== 'boolean') {
    throw createError(
      'Missing isCalibrationOptIn flag for difficulty calibration telemetry',
      400,
      'INVALID_DIFFICULTY_CALIBRATION_OPT_IN'
    );
  }

  const payload: DifficultyCalibrationEventPayload = {
    type,
    // DifficultyCalibrationEventPayload expects BoardType but we accept any string at runtime
    // and rely on MetricsService to normalise label values.
    boardType: boardType as unknown as DifficultyCalibrationEventPayload['boardType'],
    numPlayers,
    difficulty,
    isCalibrationOptIn,
  };

  if (result === 'win' || result === 'loss' || result === 'draw' || result === 'abandoned') {
    payload.result = result;
  }

  if (typeof movesPlayed === 'number' && Number.isFinite(movesPlayed) && movesPlayed >= 0) {
    payload.movesPlayed = Math.floor(movesPlayed);
  }

  if (typeof perceivedDifficulty === 'number' && Number.isFinite(perceivedDifficulty)) {
    const clamped = Math.min(5, Math.max(1, Math.round(perceivedDifficulty)));
    payload.perceivedDifficulty = clamped;
  }

  return payload;
}

/**
 * Core handler for POST /api/telemetry/difficulty-calibration.
 *
 * This endpoint is intentionally lightweight and privacy-aware:
 * - No user identifiers or raw board positions are recorded.
 * - Only coarse board / player / AI context and small enums are accepted.
 */
export function handleDifficultyCalibrationTelemetry(req: Request, res: Response): void {
  const payload = coerceDifficultyCalibrationEventPayload(req.body);

  const metrics = getMetricsService();
  metrics.recordDifficultyCalibrationEvent(payload);

  // No body required; telemetry is fire-and-forget.
  res.status(204).send();
}

router.post(
  '/difficulty-calibration',
  telemetryRateLimiter,
  asyncHandler(async (req, res) => {
    handleDifficultyCalibrationTelemetry(req, res);
  })
);

export default router;
