import React, { useId, useState, cloneElement, ReactElement } from 'react';

export interface TooltipProps {
  /**
   * Content to render inside the tooltip. Keep this concise and
   * mechanics-focused (what, why), not implementation details.
   */
  content: React.ReactNode;
  /**
   * The trigger element. This should be a focusable control (e.g. button or
   * link). The tooltip will attach hover/focus handlers and aria-describedby
   * to this element.
   */
  children: ReactElement<Record<string, unknown>>;
}

/**
 * Lightweight, accessible tooltip component used for contextual HUD help.
 *
 * Behaviour:
 * - Appears on hover or keyboard focus of the child element
 * - Uses aria-describedby and role="tooltip" for screen readers
 * - Designed to be visually minimal and non-blocking
 */
export function Tooltip({ content, children }: TooltipProps) {
  const [open, setOpen] = useState(false);
  const id = useId();

  const show = () => setOpen(true);
  const hide = () => setOpen(false);

  const trigger = cloneElement(children as ReactElement<Record<string, unknown>>, {
    onMouseEnter: (event: React.MouseEvent) => {
      children.props.onMouseEnter?.(event);
      show();
    },
    onMouseLeave: (event: React.MouseEvent) => {
      children.props.onMouseLeave?.(event);
      hide();
    },
    onFocus: (event: React.FocusEvent) => {
      children.props.onFocus?.(event);
      show();
    },
    onBlur: (event: React.FocusEvent) => {
      children.props.onBlur?.(event);
      hide();
    },
    'aria-describedby': open ? id : undefined,
  });

  return (
    <span className="relative inline-flex items-center">
      {trigger}
      {open && (
        <div
          id={id}
          role="tooltip"
          className="absolute z-40 bottom-full mb-2 max-w-xs rounded-md border border-slate-600 bg-slate-900 px-2 py-1 text-[11px] leading-snug text-slate-100 shadow-lg whitespace-pre-line"
          data-testid="tooltip-content"
        >
          {content}
        </div>
      )}
    </span>
  );
}
