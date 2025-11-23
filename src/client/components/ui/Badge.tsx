import React from 'react';
import clsx from 'clsx';

export type BadgeVariant = 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'outline';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

const baseClasses = 'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold';

const variantClasses: Record<BadgeVariant, string> = {
  default: 'bg-slate-700 text-slate-100',
  primary: 'bg-blue-600 text-white',
  success: 'bg-emerald-600 text-white',
  warning: 'bg-amber-500 text-slate-900',
  danger: 'bg-red-600 text-white',
  outline: 'border border-slate-500 text-slate-200 bg-transparent',
};

export function Badge({ variant = 'default', className, ...props }: BadgeProps) {
  return <span className={clsx(baseClasses, variantClasses[variant], className)} {...props} />;
}
