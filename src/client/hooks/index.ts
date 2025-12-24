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

// Decision countdown helper
export {
  useDecisionCountdown,
  type UseDecisionCountdownArgs,
  type DecisionCountdownState,
} from './useDecisionCountdown';

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

// ═══════════════════════════════════════════════════════════════════════════
// Animation Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  // Manual animation control
  useMoveAnimation,
  // Auto-detect animations from game state
  useAutoMoveAnimation,
} from './useMoveAnimation';

// ═══════════════════════════════════════════════════════════════════════════
// Onboarding Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  useFirstTimePlayer,
  type OnboardingState,
  type UseFirstTimePlayerResult,
} from './useFirstTimePlayer';

// ═══════════════════════════════════════════════════════════════════════════
// Mobile Detection Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  // Individual detection hooks
  useIsMobile,
  useIsTouchDevice,
  // Combined state hook
  useMobileState,
  // Types
  type MobileState,
} from './useIsMobile';

// ═══════════════════════════════════════════════════════════════════════════
// Touch Gesture Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  // Touch gesture detection
  useTouchGestures,
  useCellTouchGestures,
  // Types
  type TouchGestureOptions,
  type TouchGestureCallbacks,
  type TouchGestureHandlers,
} from './useTouchGestures';

// ═══════════════════════════════════════════════════════════════════════════
// Keyboard Navigation Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  // Board-level keyboard navigation
  useKeyboardNavigation,
  // Global game shortcuts (R, M, ?, etc.)
  useGlobalGameShortcuts,
  // Player-color focus ring helpers
  getPlayerFocusRingClass,
  PLAYER_FOCUS_RING_CLASSES,
  // Types
  type KeyboardNavigationOptions,
  type KeyboardNavigationState,
  type GlobalGameShortcuts,
} from './useKeyboardNavigation';

// ═══════════════════════════════════════════════════════════════════════════
// Sandbox Hooks (for SandboxGameHost decomposition)
// ═══════════════════════════════════════════════════════════════════════════

export {
  useSandboxPersistence,
  type GameSaveStatus,
  type SyncState,
  type LocalPlayerType,
  type SandboxPersistenceOptions,
  type SandboxPersistenceState,
} from './useSandboxPersistence';

export {
  useSandboxScenarios,
  extractChainCapturePath,
  type LoadedScenario,
  type ScenarioData,
  type SandboxScenariosOptions,
  type SandboxScenariosState,
} from './useSandboxScenarios';

export {
  useSandboxEvaluation,
  formatEvaluationScore,
  getEvaluationTrend,
  getKeyFeatures,
  type EvaluationData,
  type SandboxEvaluationOptions,
  type SandboxEvaluationState,
} from './useSandboxEvaluation';

export {
  useBoardViewProps,
  useBoardOverlays,
  type BoardOverlayConfig,
  type UseBoardViewPropsOptions,
  type BoardViewPropsResult,
} from './useBoardViewProps';

export { useSandboxInteractions } from './useSandboxInteractions';

export {
  useSandboxAILoop,
  type UseSandboxAILoopOptions,
  type UseSandboxAILoopReturn,
} from './useSandboxAILoop';

export {
  useSandboxDecisionHandlers,
  type UseSandboxDecisionHandlersOptions,
  type UseSandboxDecisionHandlersReturn,
  type TerritoryRegionPromptState,
} from './useSandboxDecisionHandlers';

export {
  useSandboxRingPlacement,
  type RingPlacementCountPromptState,
  type UseSandboxRingPlacementOptions,
  type UseSandboxRingPlacementReturn,
} from './useSandboxRingPlacement';

export {
  useSandboxMoveHandlers,
  type UseSandboxMoveHandlersOptions,
  type UseSandboxMoveHandlersReturn,
} from './useSandboxMoveHandlers';

// ═══════════════════════════════════════════════════════════════════════════
// Timer/Countdown Hooks
// ═══════════════════════════════════════════════════════════════════════════

export {
  useCountdown,
  useDecisionTimer,
  formatTime,
  formatTimeWithMs,
  formatTimeAdaptive,
  type UseCountdownOptions,
  type CountdownState,
  type UseDecisionTimerOptions,
  type DecisionTimerState,
} from './useCountdown';
