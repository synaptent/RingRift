import React from 'react';
import clsx from 'clsx';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  padded?: boolean;
}

export function Card({ padded = true, className, ...props }: CardProps) {
  return (
    <div
      className={clsx(
        'rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg',
        padded && 'p-5',
        className
      )}
      {...props}
    />
  );
}
