import React from 'react';
import clsx from 'clsx';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  padded?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ padded = true, className, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={clsx(
          'rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg',
          padded && 'p-5',
          className
        )}
        {...props}
      />
    );
  }
);

Card.displayName = 'Card';
