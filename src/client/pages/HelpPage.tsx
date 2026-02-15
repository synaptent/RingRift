import { useState, useMemo } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  TeachingOverlay,
  parseTeachingTopic,
  TEACHING_TOPICS,
} from '../components/TeachingOverlay';
import { StatusBanner } from '../components/ui/StatusBanner';
import { Button } from '../components/ui/Button';
import { ButtonLink } from '../components/ui/ButtonLink';
import { useDocumentTitle } from '../hooks/useDocumentTitle';

function formatTopicLabel(topic: string): string {
  return topic
    .split('_')
    .map((word) => (word.length > 0 ? word[0].toUpperCase() + word.slice(1) : word))
    .join(' ');
}

export default function HelpPage() {
  useDocumentTitle(
    'Help',
    'Learn how to play RingRift. Rules, strategies, and tips for all board types.'
  );
  const navigate = useNavigate();
  const { topic } = useParams<{ topic?: string }>();
  const [search, setSearch] = useState('');

  const parsedTopic = parseTeachingTopic(topic);
  const hasInvalidTopicParam = !!topic && !parsedTopic;

  const filteredTopics = useMemo(() => {
    if (!search.trim()) return TEACHING_TOPICS;
    const q = search.toLowerCase();
    return TEACHING_TOPICS.filter(
      (t) => formatTopicLabel(t).toLowerCase().includes(q) || t.includes(q)
    );
  }, [search]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-4 py-10 space-y-6">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <h1 className="text-2xl sm:text-3xl font-bold flex items-center gap-2">
              <img
                src="/ringrift-icon.png"
                alt="RingRift"
                className="w-7 h-7 sm:w-8 sm:h-8 flex-shrink-0"
              />
              <span>Help & Rules</span>
            </h1>
            <p className="text-sm text-slate-400 max-w-2xl">
              Open a topic for a short, practical explanation of the mechanic and what to do next.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <ButtonLink to="/sandbox?preset=learn-basics" size="sm">
              Learn the Basics
            </ButtonLink>
            <ButtonLink to="/sandbox" variant="secondary" size="sm">
              Open Sandbox
            </ButtonLink>
          </div>
        </header>

        {/* Search */}
        <div className="relative max-w-md">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
            />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search topics..."
            className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-colors"
          />
        </div>

        {hasInvalidTopicParam ? (
          <StatusBanner
            variant="error"
            title="Unknown help topic"
            actions={
              <Button type="button" variant="secondary" size="sm" onClick={() => navigate('/help')}>
                View topics
              </Button>
            }
          >
            <span className="font-mono text-xs">{topic}</span> is not a supported help topic.
          </StatusBanner>
        ) : null}

        {filteredTopics.length === 0 ? (
          <div className="text-center py-8 text-slate-500">No topics match "{search}"</div>
        ) : (
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {filteredTopics.map((t) => (
              <Link
                key={t}
                to={`/help/${t}`}
                className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3 hover:border-emerald-500/60 hover:bg-slate-900 transition-colors"
              >
                <div className="text-sm font-semibold text-slate-100">{formatTopicLabel(t)}</div>
                <div className="text-xs text-slate-400 mt-0.5 font-mono">/help/{t}</div>
              </Link>
            ))}
          </section>
        )}

        <TeachingOverlay
          topic={parsedTopic ?? 'ring_placement'}
          isOpen={parsedTopic !== null}
          onClose={() => navigate('/help', { replace: true })}
          position="center"
        />
      </div>
    </div>
  );
}
