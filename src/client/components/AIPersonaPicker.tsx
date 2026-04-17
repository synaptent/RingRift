/**
 * AI persona picker (C2 / #80 phase 3).
 *
 * Flag-gated behind VITE_RINGRIFT_PERSONAS_ENABLED. When the flag is
 * off this component renders nothing; when on it offers a 5-option
 * selector (no-override + the four registered personas).
 *
 * Per-opponent selection (one persona per AI seat in multiplayer
 * games) is deliberately deferred: the lobby-level quick-play flow
 * uses a single persona across every AI opponent, mirroring how it
 * currently spreads one difficulty across the opponent array. If we
 * later expose per-seat config we'll do it alongside
 * plan/AI_QUALITY_STRENGTH_DIVERSITY_PLAN_2026-04-16.md item C3.
 */

import { useId } from 'react';

import { ALL_PERSONAS, PERSONA_COPY, type PersonaId } from '../config/aiQuickPlay';

export interface AIPersonaPickerProps {
  /** Currently-selected persona, or undefined for "use ladder default". */
  value: PersonaId | undefined;
  onChange: (persona: PersonaId | undefined) => void;
  /**
   * Whether to render the "No override (use ladder default)" choice.
   * Defaults to true so players can opt out of personas entirely even
   * once the feature flag is on.
   */
  includeNoOverride?: boolean;
  /** Disable the control (e.g. while a game is being created). */
  disabled?: boolean;
  /** Override the visibility check (used only by tests and storybook). */
  forceVisible?: boolean;
  /** Override the feature-flag env var read (primarily for tests). */
  featureEnabled?: boolean;
}

/**
 * Returns whether the persona feature is enabled in the current build.
 * Mirrors the server-side RINGRIFT_PERSONAS_ENABLED flag: both must be
 * on for a selected persona to actually affect gameplay.
 */
export function personasFeatureEnabled(): boolean {
  const raw = import.meta.env?.VITE_RINGRIFT_PERSONAS_ENABLED;
  if (typeof raw !== 'string') return false;
  return ['1', 'true', 'yes', 'on'].includes(raw.trim().toLowerCase());
}

export function AIPersonaPicker({
  value,
  onChange,
  includeNoOverride = true,
  disabled = false,
  forceVisible = false,
  featureEnabled,
}: AIPersonaPickerProps): JSX.Element | null {
  const enabled = featureEnabled ?? personasFeatureEnabled();
  if (!enabled && !forceVisible) {
    return null;
  }

  const labelId = useId();

  return (
    <fieldset
      className="flex flex-col gap-2"
      data-testid="ai-persona-picker"
      disabled={disabled}
      aria-labelledby={labelId}
    >
      <legend id={labelId} className="text-sm font-semibold text-gray-200">
        AI personality
      </legend>
      <p className="text-xs text-gray-400">
        Same difficulty, different play style. Changes the AI&apos;s heuristic weights; the
        underlying engine and think time are unchanged.
      </p>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
        {includeNoOverride && (
          <PersonaOption
            id={undefined}
            label="Default"
            tagline="Use the ladder's built-in profile"
            detail="No persona override. The AI uses whichever profile the ladder normally assigns for this board and difficulty."
            selected={value === undefined}
            onSelect={onChange}
          />
        )}
        {ALL_PERSONAS.map((personaId) => {
          const copy = PERSONA_COPY[personaId];
          return (
            <PersonaOption
              key={personaId}
              id={personaId}
              label={copy.label}
              tagline={copy.tagline}
              detail={copy.detail}
              selected={value === personaId}
              onSelect={onChange}
            />
          );
        })}
      </div>
    </fieldset>
  );
}

interface PersonaOptionProps {
  id: PersonaId | undefined;
  label: string;
  tagline: string;
  detail: string;
  selected: boolean;
  onSelect: (persona: PersonaId | undefined) => void;
}

function PersonaOption({
  id,
  label,
  tagline,
  detail,
  selected,
  onSelect,
}: PersonaOptionProps): JSX.Element {
  return (
    <button
      type="button"
      onClick={() => onSelect(id)}
      title={detail}
      aria-pressed={selected}
      data-testid={`persona-option-${id ?? 'default'}`}
      className={[
        'flex flex-col items-start rounded-md border p-2 text-left text-sm transition',
        selected
          ? 'border-blue-400 bg-blue-500/20 text-blue-100'
          : 'border-gray-600 bg-gray-800/40 text-gray-200 hover:border-gray-400 hover:bg-gray-800/60',
      ].join(' ')}
    >
      <span className="font-semibold">{label}</span>
      <span className="text-[11px] text-gray-400">{tagline}</span>
    </button>
  );
}
