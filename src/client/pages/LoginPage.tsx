import React, { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { InlineAlert } from '../components/ui/InlineAlert';
import { extractErrorMessage, extractErrorCode } from '../utils/errorReporting';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await login(email, password);
      // On successful login, redirect to the main app shell (home/lobby).
      navigate('/');
    } catch (error: unknown) {
      const errorCode = extractErrorCode(error);

      // If the backend reports invalid credentials for this email, assume this may
      // be a new user and send them directly to the registration flow, carrying
      // the email along so it can be pre-filled.
      if (errorCode === 'INVALID_CREDENTIALS') {
        navigate('/register', { state: { email } });
        return;
      }

      const message = extractErrorMessage(
        error,
        'Login failed. Please check your credentials and try again.'
      );
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-2 sm:px-4 py-4 sm:py-8 space-y-4 sm:space-y-6">
        <header className="flex flex-col gap-2 sm:gap-3">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold mb-1">Login</h1>
          <p className="text-sm text-slate-400">
            Sign in to play backend games, use the lobby, and track your stats. You can also jump
            straight into a local sandbox game without an account.
          </p>
        </header>

        <div className="grid gap-4 sm:gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <section className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 shadow-lg">
            <div className="mb-4">
              <p className="text-xs uppercase tracking-wide text-slate-400">Account</p>
              <h2 className="text-lg font-semibold text-white">Sign in to your account</h2>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-1">
                <label htmlFor="email" className="block text-sm font-medium text-slate-300">
                  Email
                </label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                />
              </div>

              <div className="space-y-1">
                <label htmlFor="password" className="block text-sm font-medium text-slate-300">
                  Password
                </label>
                <Input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                />
              </div>

              {error && <InlineAlert variant="error">{error}</InlineAlert>}

              <Button type="submit" fullWidth disabled={isSubmitting}>
                {isSubmitting ? 'Logging in…' : 'Login'}
              </Button>

              <p className="text-xs text-slate-400">
                Don't have an account yet?{' '}
                <Link to="/register" className="text-emerald-400 hover:text-emerald-300 underline">
                  Create an account
                </Link>
                .
              </p>
            </form>
          </section>

          <section className="p-5 rounded-2xl bg-slate-900/70 border border-slate-700 shadow-lg space-y-4">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-400">Quick Start</p>
              <h2 className="text-lg font-semibold text-white">Play without logging in</h2>
            </div>

            <p className="text-sm text-slate-300">
              The local sandbox runs the full rules engine entirely in your browser. It's ideal for
              experimenting with movement, captures, lines, and territory without creating an
              account.
            </p>

            <div className="space-y-3">
              <Link
                to="/sandbox"
                className="block w-full px-4 py-2.5 rounded-xl border border-slate-600 bg-slate-900/60 text-center text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm font-medium"
              >
                Play Local Sandbox Game
              </Link>

              <div className="flex flex-wrap gap-2">
                <Link
                  to="/sandbox?preset=learn-basics"
                  className="flex-1 px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-center text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm font-medium"
                >
                  Learn the Basics
                </Link>
                <Link
                  to="/help"
                  className="flex-1 px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-center text-slate-200 hover:border-sky-400 hover:text-sky-200 transition text-sm font-medium"
                >
                  Help Topics
                </Link>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
