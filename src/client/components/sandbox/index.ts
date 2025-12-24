/**
 * Sandbox sub-components for SandboxGameHost.
 *
 * These components extract view-layer complexity from the main SandboxGameHost
 * component into focused, reusable pieces following the decomposition plan in
 * docs/architecture/SANDBOX_GAME_HOST_DECOMPOSITION_PLAN.md
 */

export { SandboxBoardSection, type SandboxBoardSectionProps } from './SandboxBoardSection';
export {
  SandboxGameSidebar,
  type SandboxGameSidebarProps,
  type SaveStatus,
  type SyncState,
} from './SandboxGameSidebar';
