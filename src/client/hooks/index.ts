/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Client Hooks Index
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Centralized exports for all client-side hooks. These hooks provide a
 * clean separation of concerns over the underlying GameContext.
 *
 * Categories:
 * - State Hooks: Read-only access to game state with optional view models
 * - Connection Hooks: WebSocket connection management
 * - Action Hooks: Move submission, choice handling, chat
 */

// ═══════════════════════════════════════════════════════════════════════════
// State Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  // Raw game state
  useGameState,
  // View model hooks
  useHUDViewModel,
  useBoardViewModel,
  useEventLogViewModel,
  useVictoryViewModel,
  // Convenience hooks
  useGamePhase,
  useGameStatus,
  // Types
  type RawGameState,
  type UseHUDViewModelOptions,
  type UseVictoryViewModelOptions,
  type UseEventLogViewModelOptions,
} from './useGameState';

// ═══════════════════════════════════════════════════════════════════════════
// Connection Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  // Full connection management
  useGameConnection,
  // Lightweight status-only hooks
  useConnectionStatus,
  useIsConnected,
  // Types
  type ConnectionHealth,
  type ConnectionActions,
  type GameConnectionState,
} from './useGameConnection';

// ═══════════════════════════════════════════════════════════════════════════
// Action Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  // Core actions
  useGameActions,
  // Focused hooks
  usePendingChoice,
  useChatMessages,
  useValidMoves,
  // Types
  type PartialMove,
  type PlacementAction,
  type MovementAction,
  type PendingChoiceState,
  type ActionCapabilities,
} from './useGameActions';
