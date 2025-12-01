import type { PlayerChoiceType } from '../../shared/types/game';

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Decision / PlayerChoice → UX Mapping
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This module is the single mapping layer from low-level PlayerChoiceType
 * values to user-facing copy and high-level semantics for decision phases.
 *
 * It is intentionally UI--framework agnostic: the view models here are
 * simple data objects that can be consumed by React components (ChoiceDialog,
 * GameHUD, spectator overlays, etc.) without importing engine or transport
 * details.
 */

export type ChoiceKind =
  | 'line_order'
  | 'line_reward'
  | 'ring_elimination'
  | 'territory_region_order'
  | 'capture_direction'
  | 'other';

export interface ChoiceCopy {
  /** Main title for the acting player's dialog/HUD. */
  title: string;
  /** Optional, more detailed explanation shown below the title. */
  description?: string;
  /** Compact label suitable for chips/badges (e.g. in HUD headers). */
  shortLabel: string;
  /**
   * Spectator-oriented status text. The acting player name may be injected
   * by the caller.
   */
  spectatorLabel: (ctx: { actingPlayerName: string }) => string;
}

export interface ChoiceTimeoutBehavior {
  /** Whether the countdown UI should be shown for this choice type. */
  showCountdown: boolean;
  /** Optional soft warning threshold (e.g. < 5s) for styling. */
  warningThresholdMs?: number;
}

export interface ChoiceViewModel {
  /** Underlying low-level discriminant. */
  type: PlayerChoiceType;
  /** High-level semantic grouping of the choice. */
  kind: ChoiceKind;
  /** Titles, descriptions, and spectator copy for this decision. */
  copy: ChoiceCopy;
  /** Timeout UI semantics. Actual deadline comes from PlayerChoice.timeoutMs. */
  timeout: ChoiceTimeoutBehavior;
}

interface ChoiceViewModelConfig extends Omit<ChoiceViewModel, 'type'> {}

const DEFAULT_TIMEOUT_BEHAVIOR: ChoiceTimeoutBehavior = {
  showCountdown: true,
  warningThresholdMs: 5000,
};

/**
 * Canonical mapping from PlayerChoiceType → high-level UX semantics.
 *
 * NOTE: This table is intentionally the SSOT for decision labels. All
 * components (ChoiceDialog, HUD, spectator views, logs) should ultimately
 * derive human-readable labels from here rather than hard-coding per-type
 * copy in multiple places.
 */
const CHOICE_VIEW_MODEL_MAP: Record<PlayerChoiceType, ChoiceViewModelConfig> = {
  line_order: {
    kind: 'line_order',
    copy: {
      title: 'Choose Line Order',
      description: 'Select which completed line you want to process first.',
      shortLabel: 'Line order',
      spectatorLabel: ({ actingPlayerName }) =>
        `Waiting for ${actingPlayerName} to choose which line to process`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  line_reward_option: {
    kind: 'line_reward',
    copy: {
      title: 'Choose Line Reward',
      description:
        'Your markers formed a line! Choose to collapse them into territory, or take an elimination bonus.',
      shortLabel: 'Line reward',
      spectatorLabel: ({ actingPlayerName }) =>
        `Waiting for ${actingPlayerName} to choose a line reward option`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  ring_elimination: {
    kind: 'ring_elimination',
    copy: {
      title: 'Choose Elimination Stack',
      description:
        'You must eliminate rings to satisfy the territory claim. Select which of your stacks to sacrifice from.',
      shortLabel: 'Ring elimination',
      spectatorLabel: ({ actingPlayerName }) =>
        `Waiting for ${actingPlayerName} to choose a stack for ring elimination`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  region_order: {
    kind: 'territory_region_order',
    copy: {
      title: 'Choose Territory Region',
      description: 'Pick which disconnected region to resolve next.',
      shortLabel: 'Territory region',
      spectatorLabel: ({ actingPlayerName }) =>
        `Waiting for ${actingPlayerName} to choose a territory region to process`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  capture_direction: {
    kind: 'capture_direction',
    copy: {
      title: 'Choose Capture Direction',
      description:
        'You must continue the capture chain. Select which direction to jump next.',
      shortLabel: 'Capture direction',
      spectatorLabel: ({ actingPlayerName }) =>
        `Waiting for ${actingPlayerName} to choose a capture direction`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
};

/**
 * Return the ChoiceViewModel for a given PlayerChoiceType.
 */
export function getChoiceViewModelForType(type: PlayerChoiceType): ChoiceViewModel {
  const config = CHOICE_VIEW_MODEL_MAP[type];

  if (!config) {
    // Fallback that remains safe for unknown/experimental types while still
    // surfacing a reasonable label in the UI.
    const fallback: ChoiceViewModel = {
      type,
      kind: 'other',
      copy: {
        title: 'Decision Required',
        description: 'A decision is required to continue this phase.',
        shortLabel: 'Decision',
        spectatorLabel: ({ actingPlayerName }) =>
          `Waiting for ${actingPlayerName} to make a decision`,
      },
      timeout: DEFAULT_TIMEOUT_BEHAVIOR,
    };
    return fallback;
  }

  return {
    type,
    ...config,
  };
}

/**
 * Convenience helper for callers that already have a PlayerChoice instance
 * and just need the corresponding view model.
 */
export function getChoiceViewModel(choice: { type: PlayerChoiceType }): ChoiceViewModel {
  return getChoiceViewModelForType(choice.type);
}
