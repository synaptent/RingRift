import React, { useEffect, useRef, useCallback } from 'react';

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

export interface KeyboardShortcutsHelpProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
}

const BOARD_SHORTCUTS: ShortcutItem[] = [
  { keys: ['↑', '↓', '←', '→'], description: 'Navigate between board cells' },
  { keys: ['Enter', 'Space'], description: 'Select current cell' },
  { keys: ['Escape'], description: 'Cancel current action / Clear selection' },
  { keys: ['?'], description: 'Show this help dialog' },
];

const DIALOG_SHORTCUTS: ShortcutItem[] = [
  { keys: ['↑', '↓'], description: 'Navigate between options' },
  { keys: ['Enter', 'Space'], description: 'Select focused option' },
  { keys: ['Tab'], description: 'Move focus between elements' },
  { keys: ['Escape'], description: 'Close dialog (if cancellable)' },
];

const GENERAL_SHORTCUTS: ShortcutItem[] = [
  { keys: ['Tab'], description: 'Navigate to next interactive element' },
  { keys: ['Shift', '+', 'Tab'], description: 'Navigate to previous interactive element' },
];

function ShortcutKey({ keyName }: { keyName: string }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[28px] h-7 px-2 text-xs font-mono font-semibold bg-slate-700 text-slate-100 border border-slate-500 rounded shadow-sm">
      {keyName}
    </kbd>
  );
}

function ShortcutRow({ shortcut }: { shortcut: ShortcutItem }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-700 last:border-b-0">
      <span className="text-slate-300 text-sm">{shortcut.description}</span>
      <div className="flex items-center gap-1">
        {shortcut.keys.map((key, index) => (
          <ShortcutKey key={index} keyName={key} />
        ))}
      </div>
    </div>
  );
}

function ShortcutSection({ title, shortcuts }: { title: string; shortcuts: ShortcutItem[] }) {
  return (
    <div className="mb-4 last:mb-0">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
        {title}
      </h3>
      <div className="space-y-0">
        {shortcuts.map((shortcut, index) => (
          <ShortcutRow key={index} shortcut={shortcut} />
        ))}
      </div>
    </div>
  );
}

export const KeyboardShortcutsHelp: React.FC<KeyboardShortcutsHelpProps> = ({
  isOpen,
  onClose,
}) => {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  // Focus trap and keyboard handling
  useEffect(() => {
    if (!isOpen) return;

    const dialogEl = dialogRef.current;
    if (!dialogEl) return;

    // Focus close button on mount
    closeButtonRef.current?.focus();

    const focusable = Array.from(dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    const handleKeyDown = (event: KeyboardEvent) => {
      // Escape or ? closes the dialog
      if (event.key === 'Escape' || event.key === '?') {
        event.preventDefault();
        onClose();
        return;
      }

      // Focus trapping with Tab key
      if (event.key !== 'Tab' || focusable.length === 0) return;

      const active = document.activeElement as HTMLElement | null;
      if (!active) return;

      const isShift = event.shiftKey;

      if (isShift && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!isShift && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Handle click outside to close
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onClick={handleBackdropClick}
      role="presentation"
    >
      <div
        ref={dialogRef}
        className="w-full max-w-lg mx-4 p-6 rounded-lg bg-slate-900 border border-slate-600 shadow-xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="keyboard-shortcuts-title"
      >
        <div className="flex items-center justify-between mb-5">
          <h2 id="keyboard-shortcuts-title" className="text-lg font-semibold text-slate-100">
            Keyboard Shortcuts
          </h2>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-2 focus:ring-offset-slate-900 transition-colors"
            aria-label="Close keyboard shortcuts help"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-5 h-5"
              aria-hidden="true"
            >
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>
        </div>

        <div className="space-y-5 max-h-[60vh] overflow-y-auto">
          <ShortcutSection title="Board Navigation" shortcuts={BOARD_SHORTCUTS} />
          <ShortcutSection title="Dialog Navigation" shortcuts={DIALOG_SHORTCUTS} />
          <ShortcutSection title="General" shortcuts={GENERAL_SHORTCUTS} />
        </div>

        <div className="mt-5 pt-4 border-t border-slate-700">
          <p className="text-xs text-slate-500">
            Press <ShortcutKey keyName="?" /> or <ShortcutKey keyName="Esc" /> to close this dialog
          </p>
        </div>
      </div>
    </div>
  );
};
