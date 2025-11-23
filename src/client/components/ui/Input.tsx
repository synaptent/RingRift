import React from 'react';
import clsx from 'clsx';

export type InputSize = 'sm' | 'md' | 'lg';

export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  size?: InputSize;
  invalid?: boolean;
}

const baseClasses =
  'w-full rounded-md border bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500';

const sizeClasses: Record<InputSize, string> = {
  sm: 'text-xs py-1.5',
  md: 'text-sm py-2',
  lg: 'text-base py-2.5',
};

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ size = 'md', invalid, className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={clsx(
          baseClasses,
          sizeClasses[size],
          invalid ? 'border-red-600 focus:ring-red-500' : 'border-slate-600',
          className
        )}
        {...props}
      />
    );
  }
);

Input.displayName = 'Input';
