/**
 * Backend sub-components for BackendGameHost.
 *
 * These components extract view-layer complexity from the main BackendGameHost
 * component into focused, reusable pieces following the decomposition plan in
 * docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 */

export { BackendBoardSection, type BackendBoardSectionProps } from './BackendBoardSection';
export { BackendGameSidebar, type BackendGameSidebarProps } from './BackendGameSidebar';
