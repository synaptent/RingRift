import React from 'react';
import clsx from 'clsx';

export type SelectSize = 'sm' | 'md' | 'lg';

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  size?: SelectSize;
  invalid?: boolean;
}

const baseClasses =
  'w-full rounded-lg border bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500';

const sizeClasses: Record<SelectSize, string> = {
  sm: 'text-xs py-1.5',
  md: 'text-sm py-2',
  lg: 'text-base py-2.5',
};

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ size = 'md', invalid, className, children, ...props }, ref) => {
    return (
      <select
        ref={ref}
        className={clsx(
          baseClasses,
          sizeClasses[size],
          invalid ? 'border-red-600 focus:ring-red-500' : 'border-slate-600',
          className
        )}
        {...props}
      >
        {children}
      </select>
    );
  }
);

Select.displayName = 'Select';
