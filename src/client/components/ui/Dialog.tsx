import React, { useEffect, useRef } from 'react';

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

let nextDialogId = 1;
let openDialogStack: number[] = [];

function isTopmostDialog(dialogId: number) {
  return openDialogStack.length > 0 && openDialogStack[openDialogStack.length - 1] === dialogId;
}

export interface DialogProps {
  isOpen: boolean;
  onClose: () => void;
  /**
   * Optional ID of the element that labels the dialog. When omitted, callers
   * should provide an accessible name via aria-label.
   */
  labelledBy?: string | undefined;
  /** Optional ID of the element that describes the dialog. */
  describedBy?: string | undefined;
  /** Optional accessible label when labelledBy is not provided. */
  ariaLabel?: string | undefined;
  /** When true (default), Escape closes the dialog. */
  closeOnEscape?: boolean | undefined;
  /** When true (default), clicking the backdrop closes the dialog. */
  closeOnBackdropClick?: boolean | undefined;
  /**
   * When provided, focus will be moved to this element when the dialog opens.
   * Otherwise the first focusable element inside the dialog is focused.
   */
  initialFocusRef?: React.RefObject<HTMLElement | null> | undefined;
  /** Optional test id for the fullscreen overlay container. */
  overlayTestId?: string | undefined;
  /** Optional test id for the dialog container. */
  testId?: string | undefined;
  /** Optional extra classes for the fullscreen overlay container. */
  overlayClassName?: string | undefined;
  /** Optional extra classes for the backdrop layer. */
  backdropClassName?: string | undefined;
  /** Optional extra classes for the dialog container. */
  className?: string | undefined;
  children: React.ReactNode;
}

export function Dialog({
  isOpen,
  onClose,
  labelledBy,
  describedBy,
  ariaLabel,
  closeOnEscape = true,
  closeOnBackdropClick = true,
  initialFocusRef,
  overlayTestId,
  testId,
  overlayClassName,
  backdropClassName,
  className,
  children,
}: DialogProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const dialogIdRef = useRef<number>(0);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);
  const previousBodyOverflowRef = useRef<string | null>(null);

  if (dialogIdRef.current === 0) {
    dialogIdRef.current = nextDialogId++;
  }

  useEffect(() => {
    if (!isOpen) return;

    const dialogId = dialogIdRef.current;
    openDialogStack = [...openDialogStack, dialogId];

    return () => {
      openDialogStack = openDialogStack.filter((id) => id !== dialogId);
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;

    const dialogId = dialogIdRef.current;

    if (typeof document !== 'undefined') {
      previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
      previousBodyOverflowRef.current = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
    }

    const focusDialog = () => {
      const initial = initialFocusRef?.current;
      if (initial) {
        initial.focus();
        return;
      }

      const el = dialogRef.current;
      if (!el) return;
      const focusable = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
      focusable[0]?.focus();
    };

    // Allow the dialog content to mount before focusing.
    const focusTimeout = typeof window !== 'undefined' ? window.setTimeout(focusDialog, 0) : null;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;

      if (!isTopmostDialog(dialogId)) return;

      if (event.key === 'Escape' && closeOnEscape) {
        event.preventDefault();
        onClose();
        return;
      }

      if (event.key !== 'Tab') return;

      const el = dialogRef.current;
      if (!el) return;
      const focusable = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS)).filter(
        (node) => !node.hasAttribute('disabled')
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (!active) return;

      if (!el.contains(active)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
        return;
      }

      if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      if (typeof window !== 'undefined' && focusTimeout !== null) {
        window.clearTimeout(focusTimeout);
      }
      document.removeEventListener('keydown', handleKeyDown);

      if (typeof document !== 'undefined') {
        document.body.style.overflow = previousBodyOverflowRef.current ?? '';
      }

      previouslyFocusedRef.current?.focus();
    };
  }, [isOpen, closeOnEscape, initialFocusRef, onClose]);

  if (!isOpen) return null;

  const overlayClasses = [
    'fixed inset-0 flex items-center justify-center',
    overlayClassName ?? 'z-50',
  ].join(' ');

  const backdropClasses = ['absolute inset-0 bg-black/60 backdrop-blur-sm', backdropClassName]
    .filter(Boolean)
    .join(' ');

  return (
    <div
      className={overlayClasses}
      role="presentation"
      {...(overlayTestId ? { 'data-testid': overlayTestId } : {})}
    >
      <div
        className={backdropClasses}
        onClick={closeOnBackdropClick ? onClose : undefined}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        className={className}
        role="dialog"
        aria-modal="true"
        {...(labelledBy ? { 'aria-labelledby': labelledBy } : {})}
        {...(describedBy ? { 'aria-describedby': describedBy } : {})}
        {...(!labelledBy && ariaLabel ? { 'aria-label': ariaLabel } : {})}
        {...(testId ? { 'data-testid': testId } : {})}
      >
        {children}
      </div>
    </div>
  );
}
