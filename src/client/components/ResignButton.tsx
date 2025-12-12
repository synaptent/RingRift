import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/Button';

export interface ResignButtonProps {
  /** Callback when resignation is confirmed */
  onResign: () => void;
  /** Whether the button is disabled (e.g., during API call) */
  disabled?: boolean;
  /** Whether a resignation is currently in progress */
  isResigning?: boolean;
  /**
   * Optional controlled flag for opening the confirmation dialog.
   * When omitted, the component manages its own open state.
   */
  isConfirmOpen?: boolean;
  /** Optional controlled state setter for the confirmation dialog. */
  onConfirmOpenChange?: (isOpen: boolean) => void;
}

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

/**
 * Resign button with confirmation dialog.
 *
 * Accessibility features:
 * - Dialog has role="alertdialog" for urgent confirmation
 * - Focus is trapped within the dialog when open
 * - Escape key cancels resignation
 * - Focus is restored to the trigger button on close
 */
export function ResignButton({
  onResign,
  disabled,
  isResigning,
  isConfirmOpen: controlledIsConfirmOpen,
  onConfirmOpenChange,
}: ResignButtonProps) {
  const [uncontrolledIsConfirmOpen, setUncontrolledIsConfirmOpen] = useState(false);
  const isConfirmOpen = controlledIsConfirmOpen ?? uncontrolledIsConfirmOpen;
  const setIsConfirmOpen = (next: boolean) => {
    if (onConfirmOpenChange) {
      onConfirmOpenChange(next);
      return;
    }
    setUncontrolledIsConfirmOpen(next);
  };
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  // Handle escape key and focus trap
  useEffect(() => {
    if (!isConfirmOpen) return;

    const dialogEl = dialogRef.current;
    if (!dialogEl) return;

    // Focus first button in dialog
    const focusable = Array.from(dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (first) {
      first.focus();
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setIsConfirmOpen(false);
        return;
      }

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

    dialogEl.addEventListener('keydown', handleKeyDown);

    return () => {
      dialogEl.removeEventListener('keydown', handleKeyDown);
      // Restore focus to trigger button
      if (triggerRef.current) {
        triggerRef.current.focus();
      }
    };
  }, [isConfirmOpen]);

  const handleConfirm = () => {
    setIsConfirmOpen(false);
    onResign();
  };

  const handleCancel = () => {
    setIsConfirmOpen(false);
  };

  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      setIsConfirmOpen(false);
    }
  };

  return (
    <>
      <Button
        ref={triggerRef}
        variant="danger"
        size="sm"
        onClick={() => setIsConfirmOpen(true)}
        disabled={disabled || isResigning}
        aria-haspopup="dialog"
        data-testid="resign-button"
      >
        {isResigning ? 'Resigning...' : 'Resign'}
      </Button>

      {isConfirmOpen && (
        <div
          ref={dialogRef}
          className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 backdrop-blur-sm"
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="resign-dialog-title"
          aria-describedby="resign-dialog-description"
          onClick={handleBackdropClick}
        >
          <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-xl max-w-sm w-full mx-4 p-6 space-y-4">
            <h2 id="resign-dialog-title" className="text-xl font-bold text-slate-100">
              Resign Game?
            </h2>
            <p id="resign-dialog-description" className="text-sm text-slate-300">
              Are you sure you want to resign? This will end the game and count as a loss. Your
              opponent will be declared the winner.
            </p>
            <div className="flex gap-3 justify-end">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCancel}
                data-testid="resign-cancel-button"
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                size="sm"
                onClick={handleConfirm}
                data-testid="resign-confirm-button"
              >
                Yes, Resign
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
