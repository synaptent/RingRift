/**
 * @fileoverview useSandboxDecisionHandlers Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** over the sandbox engine.
 * It handles UI for player choice decisions, not rules logic.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Decision types: `src/shared/types/game.ts` (PlayerChoice)
 *
 * This adapter:
 * - Ring elimination choices
 * - Territory region order choices
 * - Capture direction choices
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useRef, useCallback } from 'react';
import type { Position, PlayerChoice, PlayerChoiceResponseFor } from '../../shared/types/game';
import { positionsEqual } from '../../shared/types/game';
import { useSandbox } from '../contexts/SandboxContext';
import type { TerritoryRegionOption } from '../components/TerritoryRegionChoiceDialog';

export interface UseSandboxDecisionHandlersOptions {
  choiceResolverRef: React.MutableRefObject<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >;
  maybeRunSandboxAiIfNeeded: () => void;
  bumpSandboxTurn: () => void;
}

export interface TerritoryRegionPromptState {
  options: Array<{
    regionId: string;
    size: number;
    representativePosition: Position;
    moveId: string;
  }>;
  clickedPosition: Position;
}

export interface UseSandboxDecisionHandlersReturn {
  /** Territory region disambiguation prompt state */
  territoryRegionPrompt: TerritoryRegionPromptState | null;
  /** Close the territory region prompt */
  closeTerritoryRegionPrompt: () => void;
  /** Confirm a territory region selection */
  confirmTerritoryRegionPrompt: (selectedOption: {
    regionId: string;
    size: number;
    representativePosition: Position;
    moveId: string;
  }) => void;
  /** Recovery choice prompt open state */
  recoveryChoicePromptOpen: boolean;
  /** Resolve a recovery choice (option1, option2, or null to cancel) */
  resolveRecoveryChoice: (choice: 'option1' | 'option2' | null) => void;
  /** Request a recovery choice from the user */
  requestRecoveryChoice: () => Promise<'option1' | 'option2' | null>;
  /** Handle a cell click that might be a decision selection, returns true if handled */
  handleDecisionClick: (pos: Position) => boolean;
}

/**
 * Hook for handling player choice decisions in sandbox mode.
 */
export function useSandboxDecisionHandlers({
  choiceResolverRef,
  maybeRunSandboxAiIfNeeded,
  bumpSandboxTurn,
}: UseSandboxDecisionHandlersOptions): UseSandboxDecisionHandlersReturn {
  const {
    sandboxEngine,
    sandboxPendingChoice,
    setSandboxPendingChoice,
    sandboxCaptureChoice,
    setSandboxCaptureChoice,
    setSandboxCaptureTargets,
    setSandboxStateVersion,
  } = useSandbox();

  const [recoveryChoicePromptOpen, setRecoveryChoicePromptOpen] = useState(false);
  const recoveryChoiceResolverRef = useRef<((choice: 'option1' | 'option2' | null) => void) | null>(
    null
  );

  const [territoryRegionPrompt, setTerritoryRegionPrompt] =
    useState<TerritoryRegionPromptState | null>(null);

  const requestRecoveryChoice = useCallback(
    () =>
      new Promise<'option1' | 'option2' | null>((resolve) => {
        recoveryChoiceResolverRef.current = resolve;
        setRecoveryChoicePromptOpen(true);
      }),
    []
  );

  const resolveRecoveryChoice = useCallback((choice: 'option1' | 'option2' | null) => {
    recoveryChoiceResolverRef.current?.(choice);
    recoveryChoiceResolverRef.current = null;
    setRecoveryChoicePromptOpen(false);
  }, []);

  const closeTerritoryRegionPrompt = useCallback(() => {
    setTerritoryRegionPrompt(null);
  }, []);

  const confirmTerritoryRegionPrompt = useCallback(
    (selectedOption: {
      regionId: string;
      size: number;
      representativePosition: Position;
      moveId: string;
    }) => {
      if (!sandboxPendingChoice || sandboxPendingChoice.type !== 'region_order') {
        setTerritoryRegionPrompt(null);
        return;
      }

      const currentChoice = sandboxPendingChoice;
      const resolver = choiceResolverRef.current;
      if (resolver) {
        resolver({
          choiceId: currentChoice.id,
          playerNumber: currentChoice.playerNumber,
          choiceType: currentChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<PlayerChoice>);
      }
      choiceResolverRef.current = null;
      setTerritoryRegionPrompt(null);
      window.setTimeout(() => {
        setSandboxPendingChoice(null);
        setSandboxStateVersion((v) => v + 1);
        maybeRunSandboxAiIfNeeded();
      }, 0);
    },
    [
      sandboxPendingChoice,
      choiceResolverRef,
      setSandboxPendingChoice,
      setSandboxStateVersion,
      maybeRunSandboxAiIfNeeded,
    ]
  );

  /**
   * Handle a cell click that might be selecting a decision option.
   * Returns true if the click was handled as a decision, false otherwise.
   */
  const handleDecisionClick = useCallback(
    (pos: Position): boolean => {
      // Handle region_order choice
      if (sandboxPendingChoice && sandboxPendingChoice.type === 'region_order') {
        const currentChoice = sandboxPendingChoice;
        const options = (currentChoice.options ?? []) as Array<{
          regionId: string;
          size: number;
          representativePosition: Position;
          moveId: string;
          spaces?: Position[]; // RR-FIX-2026-01-12: Added for direct position matching
        }>;

        if (options.length === 0) {
          return true; // Handled but nothing to do
        }

        let selectedOption: (typeof options)[number] | undefined;

        // RR-FIX-2026-01-12: First try to match using option.spaces directly.
        // This is more reliable than looking up from territories map which may be stale.
        const optionsWithSpaces = options.filter(
          (opt): opt is TerritoryRegionOption & { spaces: Position[] } =>
            opt.spaces != null && opt.spaces.length > 0
        );
        if (optionsWithSpaces.length > 0) {
          const matchingBySpaces = optionsWithSpaces.filter((opt) =>
            opt.spaces.some((space) => positionsEqual(space, pos))
          );

          if (matchingBySpaces.length === 1) {
            selectedOption = matchingBySpaces[0];
          } else if (matchingBySpaces.length > 1) {
            // Multiple regions overlap at this cell - show disambiguation prompt
            setTerritoryRegionPrompt({
              options: matchingBySpaces,
              clickedPosition: pos,
            });
            return true;
          }
          // If no match via spaces, selectedOption remains undefined - fall through to territories lookup
        }

        // Fallback: Look up from territories map (may have stale regionIds)
        if (!selectedOption) {
          const engine = sandboxEngine;
          const state = engine?.getGameState();
          const territories = state?.board.territories;

          if (engine && state && territories && territories.size > 0) {
            // Identify which territory region(s) contain the clicked cell
            const clickedRegionIds: string[] = [];
            territories.forEach((territory, regionId) => {
              const spaces = territory.spaces ?? [];
              if (spaces.some((space) => positionsEqual(space, pos))) {
                clickedRegionIds.push(regionId);
              }
            });

            // Find all options whose regions contain the clicked cell.
            const matchingOptions = options.filter((opt) =>
              clickedRegionIds.includes(opt.regionId)
            );

            // If multiple regions overlap at this cell, show disambiguation prompt
            if (matchingOptions.length > 1) {
              setTerritoryRegionPrompt({
                options: matchingOptions,
                clickedPosition: pos,
              });
              return true;
            }

            if (matchingOptions.length === 1) {
              selectedOption = matchingOptions[0];
            }

            // Final fallback: representative-position heuristic
            if (!selectedOption) {
              territories.forEach((territory, regionId) => {
                if (selectedOption) return;
                const spaces = territory.spaces ?? [];
                const containsClick = spaces.some((space) => positionsEqual(space, pos));
                if (!containsClick) return;

                const hasRepresentative = spaces.some((space) =>
                  options.some((opt) => positionsEqual(opt.representativePosition, space))
                );
                if (hasRepresentative) {
                  selectedOption = options.find((opt) =>
                    spaces.some((space) => positionsEqual(space, opt.representativePosition))
                  );
                } else {
                  selectedOption = options.find((opt) => opt.regionId === regionId);
                }
              });
            }
          }
        }

        if (selectedOption) {
          const resolver = choiceResolverRef.current;
          if (resolver) {
            resolver({
              choiceId: currentChoice.id,
              playerNumber: currentChoice.playerNumber,
              choiceType: currentChoice.type,
              selectedOption,
            } as PlayerChoiceResponseFor<PlayerChoice>);
          }
          choiceResolverRef.current = null;
          window.setTimeout(() => {
            setSandboxPendingChoice(null);
            setSandboxStateVersion((v) => v + 1);
            maybeRunSandboxAiIfNeeded();
          }, 0);
        }
        return true;
      }

      // Handle ring_elimination choice
      if (sandboxPendingChoice && sandboxPendingChoice.type === 'ring_elimination') {
        const currentChoice = sandboxPendingChoice;
        const options = (currentChoice.options ?? []) as Array<{ stackPosition: Position }>;
        const matching = options.find((opt) => positionsEqual(opt.stackPosition, pos));

        if (matching) {
          const resolver = choiceResolverRef.current;
          if (resolver) {
            resolver({
              choiceId: currentChoice.id,
              playerNumber: currentChoice.playerNumber,
              choiceType: currentChoice.type,
              selectedOption: matching,
            } as PlayerChoiceResponseFor<PlayerChoice>);
          }
          choiceResolverRef.current = null;
          window.setTimeout(() => {
            setSandboxPendingChoice(null);
            setSandboxStateVersion((v) => v + 1);
            maybeRunSandboxAiIfNeeded();
          }, 0);
          return true;
        }

        // RR-FIX-2026-01-12: If there are valid options with a resolver waiting, block
        // fall-through to prevent normal movement handling from interfering. Only return
        // false when options are empty (stale choice) to allow phase handler fallback.
        if (options.length > 0 && choiceResolverRef.current) {
          // User clicked elsewhere while elimination choice is active.
          // Block the click - they must click on a highlighted elimination target.
          // The UI should be showing valid targets via decisionHighlights.
          return true;
        }

        // Options empty or no resolver - allow phase handler fallback via getValidMoves()
        return false;
      }

      // Handle capture_direction choice
      if (sandboxCaptureChoice && sandboxCaptureChoice.type === 'capture_direction') {
        const currentChoice = sandboxCaptureChoice;
        const options = (currentChoice.options ?? []) as Array<{ landingPosition: Position }>;
        const matching = options.find((opt) => positionsEqual(opt.landingPosition, pos));

        if (matching) {
          const resolver = choiceResolverRef.current;
          if (resolver) {
            resolver({
              choiceId: currentChoice.id,
              playerNumber: currentChoice.playerNumber,
              choiceType: currentChoice.type,
              selectedOption: matching,
            } as PlayerChoiceResponseFor<PlayerChoice>);
          }
          choiceResolverRef.current = null;
          setSandboxCaptureChoice(null);
          setSandboxCaptureTargets([]);

          window.setTimeout(() => {
            bumpSandboxTurn();
            maybeRunSandboxAiIfNeeded();
          }, 0);
        }
        return true;
      }

      // RR-FIX-2026-01-12: Handle line_reward_option choice with segments (graphical selection)
      if (
        sandboxPendingChoice &&
        sandboxPendingChoice.type === 'line_reward_option' &&
        sandboxPendingChoice.segments &&
        sandboxPendingChoice.segments.length > 0
      ) {
        const currentChoice = sandboxPendingChoice;
        const segments = currentChoice.segments;

        // Find the segment containing the clicked position
        const clickedSegment = segments.find((segment) =>
          segment.positions.some((p) => positionsEqual(p, pos))
        );

        if (clickedSegment) {
          const resolver = choiceResolverRef.current;
          if (resolver) {
            resolver({
              choiceId: currentChoice.id,
              playerNumber: currentChoice.playerNumber,
              choiceType: currentChoice.type,
              selectedOption: { optionId: clickedSegment.optionId },
            } as PlayerChoiceResponseFor<PlayerChoice>);
          }
          choiceResolverRef.current = null;
          window.setTimeout(() => {
            setSandboxPendingChoice(null);
            setSandboxStateVersion((v) => v + 1);
            maybeRunSandboxAiIfNeeded();
          }, 0);
          return true;
        }
        // Click was not on a highlighted segment, let other handlers try
        return false;
      }

      return false;
    },
    [
      sandboxEngine,
      sandboxPendingChoice,
      sandboxCaptureChoice,
      choiceResolverRef,
      setSandboxPendingChoice,
      setSandboxCaptureChoice,
      setSandboxCaptureTargets,
      setSandboxStateVersion,
      maybeRunSandboxAiIfNeeded,
      bumpSandboxTurn,
    ]
  );

  return {
    territoryRegionPrompt,
    closeTerritoryRegionPrompt,
    confirmTerritoryRegionPrompt,
    recoveryChoicePromptOpen,
    resolveRecoveryChoice,
    requestRecoveryChoice,
    handleDecisionClick,
  };
}

export default useSandboxDecisionHandlers;
