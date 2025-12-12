import React, { useEffect, useRef } from 'react';
import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { Badge } from './ui/Badge';

export type BoardControlsOverlayMode = 'backend' | 'sandbox' | 'spectator';

export interface BoardControlsOverlayProps {
  mode: BoardControlsOverlayMode;
  /**
   * When true, indicates that a dedicated touch-controls panel is present
   * (e.g. SandboxTouchControlsPanel). Used to tailor copy for sandbox mode.
   */
  hasTouchControlsPanel?: boolean | undefined;
  /**
   * Optional flags for future sections (AI debug, history, etc.). These are
   * included for forward-compatibility but are currently informational only.
   */
  hasAIDebug?: boolean;
  onClose: () => void;
}

const sectionTitleClass = 'text-sm font-semibold text-slate-100 mb-1 flex items-center gap-2';
const bodyTextClass = 'text-xs text-slate-300';
const listClass = 'list-disc list-inside space-y-1 text-xs text-slate-300';
const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

function ModeBadge({ mode }: { mode: BoardControlsOverlayMode }) {
  if (mode === 'sandbox') {
    return (
      <Badge variant="primary" className="uppercase tracking-wide">
        Sandbox
      </Badge>
    );
  }

  if (mode === 'spectator') {
    return (
      <Badge variant="outline" className="uppercase tracking-wide">
        Spectator
      </Badge>
    );
  }

  return (
    <Badge variant="success" className="uppercase tracking-wide">
      Backend game
    </Badge>
  );
}

function BasicControlsSection({ mode }: { mode: BoardControlsOverlayMode }) {
  const isSpectator = mode === 'spectator';

  return (
    <section
      aria-label="Basic board controls"
      data-testid="board-controls-basic-section"
      className="space-y-2"
    >
      <h3 className={sectionTitleClass}>Basic mouse / touch controls</h3>
      <ul className={listClass}>
        <li>
          Click or tap a cell to select a stack or empty space. The Selection panel shows details
          for the currently selected cell.
        </li>
        <li>
          Click or tap a <span className="text-emerald-300 font-semibold">highlighted</span>{' '}
          destination to apply a move (placement, movement, capture, or territory action).
        </li>
        <li>Click or tap the selected cell again to cancel the selection and clear highlights.</li>
        {!isSpectator && (
          <li>
            During <span className="font-semibold">ring placement</span>, highlighted cells show
            legal ring locations. Use single-click/tap for the canonical placement; right-click or
            long-press can open an advanced ring-count prompt on desktop.
          </li>
        )}
        {isSpectator && (
          <li>
            As a spectator the board is read-only: you can click cells to inspect stacks, but cannot
            submit moves.
          </li>
        )}
      </ul>
    </section>
  );
}

function KeyboardShortcutsSection({ mode }: { mode: BoardControlsOverlayMode }) {
  const showResign = mode === 'backend';

  return (
    <section
      aria-label="Keyboard shortcuts"
      data-testid="board-controls-keyboard-section"
      className="space-y-2"
    >
      <h3 className={sectionTitleClass}>Keyboard shortcuts (desktop)</h3>
      <ul className={listClass}>
        <li>
          <span className="font-mono text-slate-100">Arrow keys</span> – move focus around the
          board. The focused cell is outlined in amber.
        </li>
        <li>
          <span className="font-mono text-slate-100">Enter</span> /{' '}
          <span className="font-mono text-slate-100">Space</span> – activate the focused cell (same
          as clicking it).
        </li>
        <li>
          <span className="font-mono text-slate-100">Esc</span> – clear the current selection when
          the overlay is closed, or close this help overlay when it is open.
        </li>
        <li>
          <span className="font-mono text-slate-100">?</span>{' '}
          <span className="text-slate-400">(Shift&nbsp;+&nbsp;/)</span> – toggle this Board Controls
          & Shortcuts overlay.
        </li>
        {showResign && (
          <li>
            <span className="font-mono text-slate-100">R</span> – resign (backend games only).
          </li>
        )}
        <li>
          <span className="font-mono text-slate-100">M</span> – toggle sound/mute.
        </li>
        <li>
          <span className="font-mono text-slate-100">F</span> – toggle fullscreen.
        </li>
        <li>
          <span className="font-mono text-slate-100">Tab</span> – in decision dialogs, move between
          options. Use <span className="font-mono text-slate-100">Enter</span> or{' '}
          <span className="font-mono text-slate-100">Space</span> to select.
        </li>
        <li>
          <span className="font-mono text-slate-100">↑ / ↓</span> – in decision dialogs, navigate
          between options with arrow keys.
        </li>
      </ul>
      {mode !== 'spectator' && (
        <p className={bodyTextClass}>
          Keyboard navigation allows you to play RingRift entirely without a mouse. Focus indicators
          are high contrast and respect system preferences for reduced motion.
        </p>
      )}
    </section>
  );
}

function SandboxTouchSection({
  hasTouchControlsPanel,
}: {
  hasTouchControlsPanel?: boolean | undefined;
}) {
  return (
    <section
      aria-label="Sandbox touch controls"
      data-testid="board-controls-sandbox-section"
      className="space-y-2"
    >
      <h3 className={sectionTitleClass}>Sandbox touch controls</h3>
      <ul className={listClass}>
        <li>
          <span className="font-semibold">Single tap</span> an empty cell during ring placement to
          place one ring there (when legal) and highlight that stack's legal movement targets.
        </li>
        <li>
          <span className="font-semibold">Double tap</span> an empty cell during ring placement to
          request a two-ring placement. If the engine rejects the multi-ring placement, it falls
          back to a single ring when possible.
        </li>
        <li>
          Tapping an existing stack during placement adds a single ring to that stack (when legal)
          and then highlights its legal movement options.
        </li>
        <li>
          During movement/capture, tap a stack once to select it, then tap any{' '}
          <span className="text-emerald-300 font-semibold">highlighted</span> destination to move or
          continue a capture chain. Tapping the selected stack again clears selection.
        </li>
      </ul>

      {hasTouchControlsPanel && (
        <div className="mt-3 space-y-1">
          <h4 className="text-xs font-semibold text-slate-100">Sandbox touch controls panel</h4>
          <ul className={listClass}>
            <li>
              <span className="font-semibold">Clear selection</span> – reset the current selection
              and remove all highlighted targets. This does not undo committed moves.
            </li>
            <li>
              <span className="font-semibold">Finish move</span> – explicitly finish the current
              move segment and clear highlights once you are satisfied with the move.
            </li>
            <li>
              <span className="font-semibold">Undo last segment</span> – reserved for future support
              for undoing individual capture segments once exposed by the sandbox engine.
            </li>
            <li>
              <span className="font-semibold">Show valid targets</span> – toggle bright green
              overlays on all legal destinations for the selected stack.
            </li>
            <li>
              <span className="font-semibold">Show movement grid</span> – toggle a faint movement
              graph overlay that shows board geometry and adjacency.
            </li>
          </ul>
        </div>
      )}
    </section>
  );
}

export const BoardControlsOverlay: React.FC<BoardControlsOverlayProps> = ({
  mode,
  hasTouchControlsPanel,
  hasAIDebug: _hasAIDebug, // currently unused but kept for forward-compatibility
  onClose,
}) => {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  // Focus management and keyboard handling
  useEffect(() => {
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;

      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }

      // Trap focus within the overlay
      if (event.key === 'Tab' && dialogRef.current) {
        const focusable = dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS);
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (!first || !last) return;

        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      previouslyFocusedRef.current?.focus();
    };
  }, [onClose]);

  const title =
    mode === 'sandbox' ? 'Sandbox board controls & shortcuts' : 'Board controls & shortcuts';

  const description =
    mode === 'sandbox'
      ? 'How to tap, drag, and use the sandbox touch controls panel while experimenting locally.'
      : 'How to select stacks, apply moves, and use keyboard shortcuts during online games.';

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/70 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="board-controls-title"
      aria-describedby="board-controls-description"
      data-testid="board-controls-overlay"
      onClick={onClose}
    >
      <Card
        ref={dialogRef}
        padded
        className="max-w-3xl w-full mx-4 space-y-5 pointer-events-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <h2
              id="board-controls-title"
              className="text-lg font-semibold text-slate-50 tracking-tight"
            >
              {title}
            </h2>
            <p id="board-controls-description" className="text-xs text-slate-300">
              {description}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <ModeBadge mode={mode} />
            <Button
              variant="ghost"
              size="sm"
              aria-label="Close board controls"
              onClick={onClose}
              data-testid="board-controls-close-button"
              ref={closeButtonRef}
              className="h-7 w-7 rounded-full border border-slate-600 text-sm leading-none text-slate-200 hover:bg-slate-800/80"
            >
              ✕
            </Button>
          </div>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <div className="space-y-4">
            <BasicControlsSection mode={mode} />
            <KeyboardShortcutsSection mode={mode} />
          </div>

          <div className="space-y-4">
            {mode === 'sandbox' && (
              <SandboxTouchSection hasTouchControlsPanel={hasTouchControlsPanel} />
            )}
          </div>
        </div>

        <p className={`${bodyTextClass} mt-1`}>
          This overlay is informational only: it does not change game rules or submit moves. All
          rules semantics remain in the backend GameEngine and ClientSandboxEngine orchestrator
          paths documented elsewhere in the app.
        </p>
      </Card>
    </div>
  );
};
