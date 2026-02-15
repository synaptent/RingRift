import { useLocation } from 'react-router-dom';
import { ButtonLink } from '../components/ui/ButtonLink';

export default function NotFoundPage() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950 px-4">
      <div className="text-center max-w-md">
        <p className="text-6xl font-bold text-emerald-500 mb-4">404</p>
        <h1 className="text-2xl font-bold text-white mb-2">Page Not Found</h1>
        <p className="text-slate-400 mb-6">
          The page{' '}
          <code className="text-slate-300 bg-slate-800 px-1.5 py-0.5 rounded text-sm">
            {location.pathname}
          </code>{' '}
          doesn't exist.
        </p>
        <div className="flex gap-3 justify-center">
          <ButtonLink to="/" variant="primary">
            Go Home
          </ButtonLink>
          <ButtonLink to="/sandbox" variant="secondary">
            Try Sandbox
          </ButtonLink>
        </div>
      </div>
    </div>
  );
}
