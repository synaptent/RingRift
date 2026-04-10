/**
 * Types for the GameReplayDB API.
 *
 * These types match the REST API responses from the Python AI service
 * replay endpoints at `/api/replay/*`.
 *
 * See: docs/archive/plans/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md
 * See: ai-service/app/routes/replay.py
 */

import type { BoardType, GameState } from '../../shared/types/game';

// =============================================================================
// Player Metadata
// =============================================================================

export interface ReplayPlayerMetadata {
  playerNumber: number;
  playerType: 'ai' | 'human';
  aiType?: string;
  aiDifficulty?: number;
  finalEliminatedRings?: number;
  finalTerritorySpaces?: number;
  finalRingsInHand?: number;
}

// =============================================================================
// Game Metadata
// =============================================================================

export interface ReplayGameMetadata {
  gameId: string;
  boardType: BoardType | string;
  numPlayers: number;
  winner: number | null;
  terminationReason: string | null;
  totalMoves: number;
  totalTurns: number;
  createdAt: string;
  completedAt: string | null;
  durationMs: number | null;
  source: string | null;
  /** v2 fields */
  timeControlType?: string;
  initialTimeMs?: number;
  timeIncrementMs?: number;
  /** Player details (included when fetching single game) */
  players?: ReplayPlayerMetadata[];
}

// =============================================================================
// Game List Response
// =============================================================================

export interface ReplayGameListResponse {
  games: ReplayGameMetadata[];
  total: number;
  hasMore: boolean;
  /** Current page number (1-indexed) */
  page?: number;
  /** Number of items per page */
  pageSize?: number;
}

// =============================================================================
// Move Record
// =============================================================================

export interface ReplayMoveRecord {
  moveNumber: number;
  turnNumber: number;
  player: number;
  phase: string;
  moveType: string;
  move: Record<string, unknown>;
  timestamp: string | null;
  thinkTimeMs: number | null;
  /** v2 fields */
  timeRemainingMs?: number;
  engineEval?: number;
  engineEvalType?: string;
  engineDepth?: number;
  engineNodes?: number;
  enginePV?: string[];
  engineTimeMs?: number;
}

// =============================================================================
// Moves Response
// =============================================================================

export interface ReplayMovesResponse {
  moves: ReplayMoveRecord[];
  hasMore: boolean;
}

// =============================================================================
// State Response
// =============================================================================

export interface ReplayStateResponse {
  gameState: GameState;
  moveNumber: number;
  totalMoves: number;
  engineEval?: number;
  enginePV?: string[];
}

// =============================================================================
// Choice Record
// =============================================================================

export interface ReplayChoiceRecord {
  choiceType: string;
  player: number;
  options: Record<string, unknown>[];
  selected: Record<string, unknown>;
  reasoning?: string;
}

export interface ReplayChoicesResponse {
  choices: ReplayChoiceRecord[];
}

// =============================================================================
// Stats Response
// =============================================================================

export interface ReplayStatsResponse {
  totalGames: number;
  totalMoves: number;
  /** Games grouped by board type */
  gamesByBoardType?: Record<string, number>;
  /** Alias for gamesByBoardType (backward compatibility) */
  boardTypes?: Record<string, number>;
  gamesByStatus?: Record<string, number>;
  gamesByTermination?: Record<string, number>;
  /** Games grouped by player count */
  playerCounts?: Record<string, number>;
  schemaVersion?: number;
}

// =============================================================================
// Store Game Request/Response
// =============================================================================

export interface StoreGameRequest {
  gameId?: string;
  initialState: GameState;
  finalState: GameState;
  moves: Record<string, unknown>[];
  choices?: Record<string, unknown>[];
  metadata?: Record<string, unknown>;
}

export interface StoreGameResponse {
  /** Game ID (present on success) */
  gameId?: string;
  /** Total moves stored (present on success) */
  totalMoves?: number;
  success: boolean;
  /** Error message when success is false */
  message?: string;
  /** Whether the server accepted this game for direct training export. */
  acceptedForTraining?: boolean;
  /** Replay/parity status assigned by the server. */
  parityStatus?: string | null;
  /** True when the server ignored a duplicate idempotent submission. */
  deduplicated?: boolean;
}

// =============================================================================
// Training Submission Types (January 2026 - Human Game Training)
// =============================================================================

/**
 * Request format for submitting a sandbox game for AI training.
 * Used when a human wins against an AI to update the shadow model.
 */
export interface TrainingSubmissionRequest {
  board_type: string;
  num_players: number;
  moves: TrainingMoveRecord[];
  winner: number;
  human_player: number;
  human_won: boolean;
  ai_difficulty?: number;
}

/**
 * Move record format for training submission.
 */
export interface TrainingMoveRecord {
  type: string;
  player: number;
  from?: { x: number; y: number };
  to?: { x: number; y: number };
  captureTarget?: { x: number; y: number };
}

/**
 * Response from the training submission endpoint.
 */
export interface TrainingSubmissionResponse {
  success: boolean;
  message?: string;
  metrics?: {
    total_loss?: number;
    td_loss?: number;
    outcome_loss?: number;
    num_transitions?: number;
    shadow_model_path?: string;
  };
}

// =============================================================================
// Query Parameters
// =============================================================================

export interface ReplayGameQueryParams {
  board_type?: string;
  num_players?: number;
  winner?: number;
  termination_reason?: string;
  source?: string;
  min_moves?: number;
  max_moves?: number;
  limit?: number;
  offset?: number;
}

// =============================================================================
// Playback State (for UI)
// =============================================================================

export type PlaybackSpeed = 0.5 | 1 | 2 | 4;

export interface ReplayPlaybackState {
  /** Currently loaded game ID */
  gameId: string | null;
  /** Game metadata */
  metadata: ReplayGameMetadata | null;
  /** Current move number (0 = initial state) */
  currentMoveNumber: number;
  /** Total moves in the game */
  totalMoves: number;
  /** Current game state at currentMoveNumber */
  currentState: GameState | null;
  /** Is auto-playing */
  isPlaying: boolean;
  /** Playback speed multiplier */
  playbackSpeed: PlaybackSpeed;
  /** Loading state */
  isLoading: boolean;
  /** Error message if any */
  error: string | null;
  /** Cached move records for current game */
  moves: ReplayMoveRecord[];
}
