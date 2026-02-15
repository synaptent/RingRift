import clsx from 'clsx';

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return <div className={clsx('animate-pulse rounded bg-slate-700/50', className)} />;
}

export function SkeletonText({ className }: SkeletonProps) {
  return <Skeleton className={clsx('h-4 rounded', className)} />;
}

/** Table row skeleton matching the leaderboard layout */
export function LeaderboardSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-6">
        <Skeleton className="h-8 w-48 mb-2" />
        <Skeleton className="h-4 w-64" />
      </div>
      <Skeleton className="h-11 w-full mb-4 rounded-lg" />
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="p-4 bg-slate-900/50">
          <div className="flex gap-6">
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-4 w-16 ml-auto" />
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-12" />
          </div>
        </div>
        <div className="divide-y divide-slate-700">
          {Array.from({ length: rows }).map((_, i) => (
            <div key={i} className="flex items-center gap-6 px-6 py-4">
              <Skeleton className="h-7 w-7 rounded-full" />
              <Skeleton className="h-4 w-28" />
              <Skeleton className="h-4 w-14 ml-auto" />
              <Skeleton className="h-4 w-10" />
              <Skeleton className="h-4 w-8" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Game history list skeleton */
export function GameHistorySkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="flex items-center justify-between mb-6">
        <Skeleton className="h-7 w-40" />
        <Skeleton className="h-4 w-28" />
      </div>
      <div className="flex gap-3 mb-6">
        <Skeleton className="h-10 w-32 rounded-lg" />
        <Skeleton className="h-10 w-32 rounded-lg" />
        <Skeleton className="h-4 w-20 ml-auto self-center" />
      </div>
      <div className="space-y-2">
        {Array.from({ length: rows }).map((_, i) => (
          <div
            key={i}
            className="p-4 bg-slate-800 rounded-lg border border-slate-700 flex items-center justify-between"
          >
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-8" />
              </div>
              <div className="flex gap-3">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-14" />
              </div>
            </div>
            <Skeleton className="h-8 w-14 rounded ml-3" />
          </div>
        ))}
      </div>
    </div>
  );
}

/** Profile page skeleton */
export function ProfileSkeleton() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      {/* Header card */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
        <div className="p-6 md:p-8">
          <div className="flex justify-between items-start mb-6">
            <div>
              <Skeleton className="h-8 w-40 mb-2" />
              <Skeleton className="h-4 w-48" />
            </div>
            <Skeleton className="h-9 w-28 rounded-md" />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                <Skeleton className="h-3 w-16 mb-2" />
                <Skeleton className="h-7 w-14" />
              </div>
            ))}
          </div>
          {/* Chart area */}
          <div className="mt-6">
            <Skeleton className="h-48 w-full rounded-lg" />
          </div>
        </div>
      </div>

      {/* Achievements */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-4 w-10" />
          </div>
          <Skeleton className="h-2 w-full rounded-full mb-5" />
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="p-3 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-1">
                  <Skeleton className="h-6 w-6 rounded" />
                  <Skeleton className="h-4 w-24" />
                </div>
                <Skeleton className="h-3 w-full mt-1" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent games */}
      <Skeleton className="h-6 w-36 mb-4" />
      <div className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="p-4 bg-slate-800 rounded-lg border border-slate-700 flex items-center justify-between"
          >
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-3 w-20" />
              </div>
              <Skeleton className="h-3 w-24" />
            </div>
            <Skeleton className="h-8 w-20 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}
