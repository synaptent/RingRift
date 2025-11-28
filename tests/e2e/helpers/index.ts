/**
 * E2E Test Helpers Index
 * ============================================================================
 *
 * Re-exports all E2E test utilities for convenient importing.
 *
 * Usage:
 * ```typescript
 * import { generateTestUser, registerAndLogin, GamePage } from './helpers';
 * ```
 */

// Utility helpers
export {
  generateTestUser,
  waitForNetworkIdle,
  type TestUser,
  type CreateGameOptions,
} from './test-utils';

// Authentication helpers
export { registerUser, loginUser, logout, registerAndLogin } from './test-utils';

// Game connection helpers
export { waitForWebSocketConnection, waitForGameReady } from './test-utils';

// Game action helpers
export {
  createGame,
  createBackendGameFromLobby,
  joinGame,
  makeMove,
  placePiece,
  clickValidPlacementTarget,
} from './test-utils';

// Board state assertions
export {
  assertBoardState,
  assertPlayerTurn,
  assertGamePhase,
  assertMoveLogged,
  waitForMoveLog,
} from './test-utils';

// Navigation helpers
export { goToLobby, goToGame, goToHome } from './test-utils';

// Error handling helpers
export { assertErrorMessage, assertNoErrors } from './test-utils';

// Multi-player helpers
export {
  createFreshContext,
  setupMultiplayerGame,
  coordinateTurn,
  waitForTurn,
  isPlayerTurn,
  makeRingPlacement,
  cleanupMultiplayerGame,
  type PlayerContext,
  type MultiplayerGameSetup,
} from './test-utils';

// Page Object Models
export { LoginPage } from '../pages/LoginPage';
export { RegisterPage } from '../pages/RegisterPage';
export { HomePage } from '../pages/HomePage';
export { GamePage } from '../pages/GamePage';
